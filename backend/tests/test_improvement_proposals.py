import json
from pathlib import Path

import pytest

from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.application.improvements.capabilities import ImprovementCapabilityCatalog
from app.application.improvements.models import SupervisedImprovementRecommendation
from app.application.improvements.prompts import ImprovementPromptBuilder
from app.application.improvements.service import ImprovementProposalService
from app.application.improvements.validation import ImprovementProposalValidator
from app.domain.dataset.entities import Dataset
from app.domain.exceptions import ExperimentSpecValidationError
from app.domain.experiment.entities import ExperimentMode
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    PreprocessingStepSpec,
    TaskType,
    ValidationSpec,
)
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyExperimentRepository,
    SqlAlchemyExperimentStrategyRepository,
    SqlAlchemyModelRunRepository,
)
from app.llm.base import LLMCapabilities, LLMProvider, LLMRequest, StructuredResponse
from app.ml.registries.catalog import build_default_registry_catalog


class ProposalProvider(LLMProvider):
    id = "proposal-test"
    capabilities = LLMCapabilities(supports_json_schema=True, is_local=True)

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []
        self.response_models: list[type[StructuredResponse]] = []

    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        self.requests.append(request)
        self.response_models.append(response_model)
        return response_model.model_validate(self.response)


def _recommendation(**model_parameters: object) -> dict[str, object]:
    return {
        "summary": "Compare a regularized baseline under cross-validation.",
        "rationale": ["The proposed C value is within the registered search space."],
        "models": [
            {
                "id": "logistic_regression",
                "enabled": True,
                "parameters": model_parameters or {"C": 0.5},
            }
        ],
        "preprocessing": [{"id": "numeric_scaling", "strategy": "standard", "parameters": {}}],
        "validation": {
            "strategy": "stratified_kfold",
            "test_size": 0.2,
            "folds": 5,
            "shuffle": True,
            "random_seed": 41,
        },
        "metrics": {
            "primary": "f1",
            "secondary": ["accuracy"],
            "averaging": "macro",
            "roc_auc_strategy": None,
            "positive_label": "yes",
        },
        "tuning": {
            "enabled": False,
            "engine": "optuna",
            "trials": 50,
            "metric": None,
        },
        "cautions": ["Compare on the same validation protocol."],
        "unsupported_suggestions": ["Isotonic calibration is not available."],
    }


def _service(
    database: Database,
    provider: LLMProvider,
) -> tuple[ImprovementProposalService, ExperimentService, str]:
    datasets = SqlAlchemyDatasetRepository(database)
    dataset = datasets.add(
        Dataset(
            id="improvement-dataset",
            name="Improvement dataset",
            original_filename="improvement.csv",
            content_hash="sha256:improvement",
            row_count=100,
            column_count=2,
            storage_path="dataset/original.csv",
        )
    )
    registry = build_default_registry_catalog()
    spec_validator = ExperimentSpecValidator(registry)
    experiments = ExperimentService(
        experiments=SqlAlchemyExperimentRepository(database),
        datasets=datasets,
        validator=spec_validator,
    )
    source = experiments.create(
        ExperimentSpec(
            dataset_id=dataset.id,
            dataset_hash=dataset.content_hash,
            profile_version="1.0",
            name="Source experiment",
            objective="Predict target",
            task=TaskType.BINARY_CLASSIFICATION,
            target="target",
            features=["amount"],
            column_roles={"amount": ColumnRole.FEATURE, "target": ColumnRole.TARGET},
            models=[ModelSpec(id="logistic_regression", parameters={"C": 1.0})],
            preprocessing=[PreprocessingStepSpec(id="numeric_scaling", strategy="standard")],
            validation=ValidationSpec(strategy="train_test_split"),
            metrics=MetricsSpec(primary="f1", positive_label="yes"),
        ),
        mode=ExperimentMode.EXPERT,
    )
    capabilities = ImprovementCapabilityCatalog(registry)
    service = ImprovementProposalService(
        provider=provider,
        prompts=ImprovementPromptBuilder(Path(__file__).parents[1] / "prompts"),
        experiments=experiments,
        model_runs=SqlAlchemyModelRunRepository(database),
        capabilities=capabilities,
        validator=ImprovementProposalValidator(
            capabilities=capabilities,
            spec_validator=spec_validator,
        ),
        persistence=SqlAlchemyExperimentStrategyRepository(database),
        random_seed=41,
    )
    return service, experiments, source.id


def test_proposal_uses_real_capabilities_and_preserves_protected_fields(
    migrated_database: Database,
) -> None:
    provider = ProposalProvider(_recommendation())
    service, experiments, source_id = _service(migrated_database, provider)

    proposal = service.propose(
        language="pt-BR",
        source_experiment_id=source_id,
        source_model_run_id=None,
        context={"metrics": {"f1": 0.81}},
    )

    source_spec = ExperimentSpec.model_validate(experiments.get(source_id).spec)
    assert proposal.proposed_spec.dataset_id == source_spec.dataset_id
    assert proposal.proposed_spec.features == source_spec.features
    assert proposal.recommendation.schema_version == "1.0"
    assert proposal.recommendation.scope == "supervised_experiment"
    assert proposal.proposed_spec.models[0].parameters == {"C": 0.5}
    assert {change.path for change in proposal.changes} >= {"models", "validation"}
    payload = json.loads(provider.requests[0].user_prompt)
    logistic = next(
        item
        for item in payload["applicable_capabilities"]["models"]
        if item["id"] == "logistic_regression"
    )
    assert logistic["parameters"]["C"] == {
        "type": "number",
        "minimum": 0.0001,
        "maximum": 100.0,
    }
    assert "dataset_id" in payload["applicable_capabilities"]["protected_fields"]
    generated_fields = provider.response_models[0].model_fields
    assert "schema_version" not in generated_fields
    assert "scope" not in generated_fields


def test_unknown_or_out_of_range_ai_parameter_is_rejected(
    migrated_database: Database,
) -> None:
    provider = ProposalProvider(_recommendation(unsupported=2))
    service, _, source_id = _service(migrated_database, provider)

    with pytest.raises(ExperimentSpecValidationError, match="ai_unknown_parameter"):
        service.propose(
            language="en-US",
            source_experiment_id=source_id,
            source_model_run_id=None,
            context={},
        )


def test_draft_is_atomic_audited_and_never_starts_training(
    migrated_database: Database,
) -> None:
    provider = ProposalProvider(_recommendation())
    service, experiments, source_id = _service(migrated_database, provider)
    proposal = service.propose(
        language="en-US",
        source_experiment_id=source_id,
        source_model_run_id=None,
        context={},
    )

    result = service.create_draft(
        source_experiment_id=source_id,
        source_model_run_id=None,
        source_spec_hash=proposal.source_spec_hash,
        provider_id=proposal.provider_id,
        recommendation=proposal.recommendation,
    )

    draft = experiments.get(result.experiment_id)
    assert draft.status.value == "draft"
    assert draft.parent_experiment_id == source_id
    assert SqlAlchemyModelRunRepository(migrated_database).list_for_experiment(draft.id) == []
    audit = SqlAlchemyExperimentStrategyRepository(migrated_database).get_for_experiment(draft.id)
    assert audit is not None
    assert audit.source == "assisted"
    assert audit.provider_id == "proposal-test"
    assert audit.strategy["kind"] == "ai_improvement"
    assert audit.strategy["recommendation"]["unsupported_suggestions"]


def test_draft_revalidates_source_hash(migrated_database: Database) -> None:
    provider = ProposalProvider(_recommendation())
    service, _, source_id = _service(migrated_database, provider)

    with pytest.raises(ExperimentSpecValidationError, match="source_spec_changed"):
        service.create_draft(
            source_experiment_id=source_id,
            source_model_run_id=None,
            source_spec_hash="stale",
            provider_id="proposal-test",
            recommendation=SupervisedImprovementRecommendation.model_validate(_recommendation()),
        )
