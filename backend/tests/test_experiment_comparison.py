import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_artifact_storage,
    get_experiment_service,
    get_explanation_service,
    get_model_run_repository,
    get_registry_catalog,
)
from app.application.experiments.comparison import (
    build_experiment_comparison,
    build_experiment_comparison_context,
)
from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.application.explanations.models import (
    ResultExplanation,
)
from app.application.explanations.prompts import ExplanationPromptBuilder
from app.application.explanations.service import ExplanationService
from app.domain.dataset.entities import Dataset
from app.domain.experiment.entities import (
    Experiment,
    ExperimentMode,
    ModelRun,
    ModelRunStatus,
)
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    TaskType,
    ValidationSpec,
)
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyExperimentRepository,
    SqlAlchemyModelRunRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.llm.base import LLMCapabilities, LLMProvider, LLMRequest
from app.llm.errors import LLMConfigurationError, LLMProviderError
from app.main import app
from app.ml.registries.catalog import build_default_registry_catalog


class FakeLLMProvider(LLMProvider):
    def __init__(self, explanation: ResultExplanation | None = None) -> None:
        self.recorded_requests: list[LLMRequest] = []
        self.error_to_raise: Exception | None = None
        self.explanation = explanation or ResultExplanation(
            summary="LightGBM superou a regressão logística no ROC AUC.",
            findings=["LightGBM obteve maior AUC.", "Regressão logística teve menor perda."],
            cautions=["Pequenas diferenças de AUC não são estatisticamente conclusivas."],
            next_steps=["Validar em dados futuros."],
            chart_explanations=[],
        )

    @property
    def id(self) -> str:
        return "fake_provider"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(supports_json_schema=True, is_local=True)

    def generate_structured(self, request: LLMRequest, schema: type[Any]) -> Any:
        self.recorded_requests.append(request)
        if self.error_to_raise:
            raise self.error_to_raise
        return self.explanation


def _create_test_experiment(
    database: Database,
    *,
    primary_metric: str = "roc_auc",
    secondary_metrics: list[str] | None = None,
    task: TaskType = TaskType.BINARY_CLASSIFICATION,
) -> Experiment:
    now = datetime.now(UTC)
    dataset = Dataset(
        id="test-dataset-id",
        name="Titanic",
        original_filename="titanic.csv",
        storage_path="datasets/titanic.csv",
        row_count=891,
        column_count=5,
        content_hash="hash-123456",
        created_at=now,
    )
    SqlAlchemyDatasetRepository(database).add(dataset)

    spec = ExperimentSpec(
        dataset_id=dataset.id,
        dataset_hash=dataset.content_hash,
        profile_version="1.0",
        name="Titanic Comparison Test",
        objective="Predict survival",
        task=task,
        target="survived",
        features=["pclass", "sex", "age", "fare"],
        column_roles={
            "survived": ColumnRole.TARGET,
            "pclass": ColumnRole.FEATURE,
            "sex": ColumnRole.FEATURE,
            "age": ColumnRole.FEATURE,
            "fare": ColumnRole.FEATURE,
        },
        models=[
            ModelSpec(id="lightgbm_classifier"),
            ModelSpec(id="logistic_regression"),
            ModelSpec(id="random_forest_classifier"),
        ],
        validation=ValidationSpec(
            strategy="k_fold",
            folds=5,
            random_seed=42,
        ),
        metrics=MetricsSpec(
            primary=primary_metric,
            secondary=secondary_metrics or ["accuracy"],
            positive_label=1,
        ),
    )
    experiment = Experiment.create(
        name="Titanic Comparison Test",
        description="Testing comparison",
        mode=ExperimentMode.EXPERT,
        dataset_id=dataset.id,
        objective="Predict survival",
        spec_version="1.0",
        spec=spec.model_dump(),
    )
    return SqlAlchemyExperimentRepository(database).add(experiment)


def test_comparison_maximize_metric_ordering_and_deltas() -> None:
    catalog = build_default_registry_catalog()
    datetime.now(UTC)
    spec = {
        "metrics": {"primary": "roc_auc"},
        "task": "binary_classification",
        "target": "target",
        "features": ["a", "b"],
        "validation": {"strategy": "k_fold", "folds": 5, "random_seed": 42},
    }
    experiment = Experiment.create(
        name="Titanic AUC",
        mode=ExperimentMode.EXPERT,
        dataset_id="ds-1",
        objective="AUC ranking",
        spec_version="1.0",
        spec=spec,
    )
    runs = [
        ModelRun(
            id="run-1",
            experiment_id="exp-1",
            model_id="logistic_regression",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=1.5,
            metrics_json=json.dumps(
                {
                    "values": {"roc_auc": 0.7820, "accuracy": 0.7500},
                    "standard_deviations": {"roc_auc": 0.02},
                }
            ),
            execution_id="exec-round-1",
        ),
        ModelRun(
            id="run-2",
            experiment_id="exp-1",
            model_id="lightgbm_classifier",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=2.0,
            metrics_json=json.dumps(
                {
                    "values": {"roc_auc": 0.8123, "accuracy": 0.7600},
                    "standard_deviations": {"roc_auc": 0.015},
                }
            ),
            execution_id="exec-round-1",
        ),
        ModelRun(
            id="run-3",
            experiment_id="exp-1",
            model_id="random_forest_classifier",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=3.0,
            metrics_json=json.dumps(
                {
                    "values": {"roc_auc": 0.7950, "accuracy": 0.7670},
                    "standard_deviations": {"roc_auc": 0.018},
                }
            ),
            execution_id="exec-round-1",
        ),
    ]

    comparison = build_experiment_comparison(
        experiment, runs, catalog, execution_id="exec-round-1"
    )
    assert comparison.metric_direction == "maximize"
    assert comparison.can_explain is True
    assert len(comparison.ranked_models) == 3
    assert comparison.best_model is not None
    assert comparison.best_model.model_id == "lightgbm_classifier"
    assert comparison.best_model.rank == 1
    assert math.isclose(comparison.best_model.primary_metric_value, 0.8123)
    assert math.isclose(comparison.best_model.delta_to_best, 0.0)

    # Check order: LightGBM (0.8123) > Random Forest (0.7950) > Logistic Regression (0.7820)
    assert [m.model_id for m in comparison.ranked_models] == [
        "lightgbm_classifier",
        "random_forest_classifier",
        "logistic_regression",
    ]
    assert [m.rank for m in comparison.ranked_models] == [1, 2, 3]

    assert comparison.worst_model is not None
    assert comparison.worst_model.model_id == "logistic_regression"
    assert math.isclose(comparison.worst_model.delta_to_best, 0.8123 - 0.7820, abs_tol=1e-5)
    assert comparison.ranked_models[0].standard_deviation == 0.015
    assert comparison.ranked_models[1].secondary_metrics["accuracy"] == 0.7670


def test_comparison_minimize_metric_ordering() -> None:
    catalog = build_default_registry_catalog()
    datetime.now(UTC)
    spec = {
        "metrics": {"primary": "mae"},
        "task": "regression",
        "target": "fare",
        "features": ["age"],
        "validation": {"strategy": "k_fold", "folds": 3, "random_seed": 42},
    }
    experiment = Experiment.create(
        name="Regression MAE",
        mode=ExperimentMode.EXPERT,
        dataset_id="ds-1",
        objective="MAE comparison",
        spec_version="1.0",
        spec=spec,
    )
    runs = [
        ModelRun(
            id="run-1",
            experiment_id="exp-reg",
            model_id="linear_regression",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=1.0,
            metrics_json=json.dumps({"values": {"mae": 1.5, "rmse": 2.1}}),
            execution_id="exec-round-reg",
        ),
        ModelRun(
            id="run-2",
            experiment_id="exp-reg",
            model_id="catboost_regressor",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=2.5,
            metrics_json=json.dumps({"values": {"mae": 0.8, "rmse": 1.1}}),
            execution_id="exec-round-reg",
        ),
        ModelRun(
            id="run-3",
            experiment_id="exp-reg",
            model_id="svr",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=1.8,
            metrics_json=json.dumps({"values": {"mae": 1.2, "rmse": 1.7}}),
            execution_id="exec-round-reg",
        ),
    ]

    comparison = build_experiment_comparison(
        experiment, runs, catalog, execution_id="exec-round-reg"
    )
    assert comparison.metric_direction == "minimize"
    # CatBoost (0.8) is best, SVR (1.2) is 2nd, Linear (1.5) is 3rd
    assert [m.model_id for m in comparison.ranked_models] == [
        "catboost_regressor",
        "svr",
        "linear_regression",
    ]
    assert comparison.best_model.model_id == "catboost_regressor"
    assert math.isclose(comparison.best_model.delta_to_best, 0.0)
    assert comparison.worst_model.model_id == "linear_regression"
    assert math.isclose(comparison.worst_model.delta_to_best, 0.7, abs_tol=1e-5)


def test_comparison_ties_and_excluded_models() -> None:
    catalog = build_default_registry_catalog()
    datetime.now(UTC)
    spec = {
        "metrics": {"primary": "accuracy"},
        "task": "binary_classification",
        "target": "target",
        "features": ["a"],
        "validation": {"strategy": "train_test_split", "random_seed": 42},
    }
    experiment = Experiment.create(
        name="Ties & Exclusions",
        mode=ExperimentMode.EXPERT,
        dataset_id="ds-1",
        objective="Ties",
        spec_version="1.0",
        spec=spec,
    )
    runs = [
        ModelRun(
            id="run-tie-1",
            experiment_id="exp-ties",
            model_id="model_a",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=1.0,
            metrics_json=json.dumps({"values": {"accuracy": 0.85}}),
            execution_id="round-ties",
        ),
        ModelRun(
            id="run-tie-2",
            experiment_id="exp-ties",
            model_id="model_b",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=2.0,
            metrics_json=json.dumps({"values": {"accuracy": 0.85}}),
            execution_id="round-ties",
        ),
        ModelRun(
            id="run-failed",
            experiment_id="exp-ties",
            model_id="model_failed",
            run_index=1,
            status=ModelRunStatus.FAILED,
            random_seed=42,
            parameters_json="{}",
            error_message="CUDA OOM",
            execution_id="round-ties",
        ),
        ModelRun(
            id="run-cancelled",
            experiment_id="exp-ties",
            model_id="model_cancelled",
            run_index=1,
            status=ModelRunStatus.CANCELLED,
            random_seed=42,
            parameters_json="{}",
            execution_id="round-ties",
        ),
        ModelRun(
            id="run-nan",
            experiment_id="exp-ties",
            model_id="model_nan",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            metrics_json=json.dumps({"values": {"accuracy": "NaN"}}),
            execution_id="round-ties",
        ),
        ModelRun(
            id="run-no-metric",
            experiment_id="exp-ties",
            model_id="model_missing",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            metrics_json=json.dumps({"values": {}}),
            execution_id="round-ties",
        ),
    ]

    comparison = build_experiment_comparison(experiment, runs, catalog, execution_id="round-ties")
    # Both model_a and model_b have 0.85, so both have rank 1
    assert len(comparison.ranked_models) == 2
    assert comparison.ranked_models[0].rank == 1
    assert comparison.ranked_models[1].rank == 1

    # Excluded models: failed, cancelled, nan, missing
    excluded_reasons = {e.model_id: e.reason for e in comparison.excluded_models}
    assert excluded_reasons["model_failed"] == "status_failed"
    assert excluded_reasons["model_cancelled"] == "status_cancelled"
    assert excluded_reasons["model_nan"] == "non_finite_metric"
    assert excluded_reasons["model_missing"] == "missing_metric"


def test_comparison_insufficient_models_and_legacy_unlinked() -> None:
    catalog = build_default_registry_catalog()
    datetime.now(UTC)
    spec = {"metrics": {"primary": "accuracy"}, "task": "binary_classification"}
    experiment = Experiment.create(
        name="Edge Cases",
        mode=ExperimentMode.EXPERT,
        dataset_id="ds-1",
        objective="Edge",
        spec_version="1.0",
        spec=spec,
    )

    # 1 model in round
    single_run = [
        ModelRun(
            id="run-only-1",
            experiment_id="exp-edge",
            model_id="solo_model",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            metrics_json=json.dumps({"values": {"accuracy": 0.80}}),
            execution_id="round-single",
        )
    ]
    comp_single = build_experiment_comparison(
        experiment, single_run, catalog, execution_id="round-single"
    )
    assert comp_single.can_explain is False
    assert comp_single.explanation_block_reason == "insufficient_eligible_models"

    # Legacy runs with execution_id is None
    legacy_runs = [
        ModelRun(
            id="run-legacy-1",
            experiment_id="exp-edge",
            model_id="legacy_1",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            metrics_json=json.dumps({"values": {"accuracy": 0.80}}),
            execution_id=None,
        ),
        ModelRun(
            id="run-legacy-2",
            experiment_id="exp-edge",
            model_id="legacy_2",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            metrics_json=json.dumps({"values": {"accuracy": 0.82}}),
            execution_id=None,
        ),
    ]
    comp_legacy = build_experiment_comparison(experiment, legacy_runs, catalog)
    assert comp_legacy.available_rounds == []
    assert comp_legacy.selected_round_id is None
    assert comp_legacy.can_explain is False
    assert comp_legacy.explanation_block_reason == "no_rounds"


def test_comparison_context_safety_and_no_leakage() -> None:
    catalog = build_default_registry_catalog()
    datetime.now(UTC)
    spec = {
        "metrics": {"primary": "roc_auc"},
        "task": "binary_classification",
        "target": "survived",
        "features": ["age", "fare"],
        "validation": {"strategy": "k_fold", "folds": 5, "random_seed": 42},
    }
    experiment = Experiment.create(
        name="Safety Test",
        mode=ExperimentMode.EXPERT,
        dataset_id="ds-1",
        objective="Verify privacy",
        spec_version="1.0",
        spec=spec,
    )
    runs = [
        ModelRun(
            id="run-1",
            experiment_id="exp-safe",
            model_id="model_a",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=1.0,
            artifact_path="/sensitive/server/path/model_a.joblib",
            metrics_json=json.dumps(
                {
                    "values": {"roc_auc": 0.85},
                    "predictions_path": "/sensitive/predictions.parquet",
                    "folds": [{"test_rows": 100, "train_indices": [0, 1, 2]}],
                }
            ),
            execution_id="round-safe",
        ),
        ModelRun(
            id="run-2",
            experiment_id="exp-safe",
            model_id="model_b",
            run_index=1,
            status=ModelRunStatus.COMPLETED,
            random_seed=42,
            parameters_json="{}",
            duration_seconds=2.0,
            artifact_path="/sensitive/server/path/model_b.joblib",
            metrics_json=json.dumps({"values": {"roc_auc": 0.80}}),
            execution_id="round-safe",
        ),
    ]

    comparison = build_experiment_comparison(experiment, runs, catalog, execution_id="round-safe")
    context = build_experiment_comparison_context(comparison, experiment)
    serialized = json.dumps(context)

    # Check that forbidden sensitive info is nowhere in context
    assert "/sensitive" not in serialized
    assert "joblib" not in serialized
    assert "parquet" not in serialized
    assert "train_indices" not in serialized
    assert "test_rows" not in serialized

    # Check that aggregated safe info is present
    assert context["primary_metric"] == "roc_auc"
    assert context["metric_direction"] == "maximize"
    assert len(context["ranked_models"]) == 2
    assert "limitations" in context
    assert any("statistical significance" in lim.lower() for lim in context["limitations"])

    # Test prompt generation for pt-BR and en-US
    prompt_builder = ExplanationPromptBuilder(Path(__file__).parents[1] / "prompts")
    prompt_pt = prompt_builder.build(
        scope="experiment_comparison",
        language="pt-BR",
        context=context,
        seed=42,
    )
    assert "experiment_comparison" in prompt_pt.user_prompt
    assert "maior é melhor para maximização" in prompt_pt.system_prompt

    prompt_en = prompt_builder.build(
        scope="experiment_comparison",
        language="en-US",
        context=context,
        seed=42,
    )
    assert "higher is better for maximization" in prompt_en.system_prompt


def test_api_comparison_and_explanation_endpoints(
    migrated_database: Database,
    tmp_path: Path,
) -> None:
    experiment = _create_test_experiment(migrated_database)
    runs_repo = SqlAlchemyModelRunRepository(migrated_database)

    # Create 3 runs in execution round 1
    exec_id_1 = "exec-round-1"
    run_1 = runs_repo.create(
        experiment_id=experiment.id,
        model_id="lightgbm_classifier",
        random_seed=42,
        parameters={"n_estimators": 100},
        execution_id=exec_id_1,
    )
    runs_repo.transition(
        run_1.id,
        ModelRunStatus.COMPLETED,
        metrics={
            "values": {"roc_auc": 0.8123, "accuracy": 0.7600},
            "standard_deviations": {"roc_auc": 0.015},
        },
    )

    run_2 = runs_repo.create(
        experiment_id=experiment.id,
        model_id="logistic_regression",
        random_seed=42,
        parameters={"C": 1.0},
        execution_id=exec_id_1,
    )
    runs_repo.transition(
        run_2.id,
        ModelRunStatus.COMPLETED,
        metrics={"values": {"roc_auc": 0.7820, "accuracy": 0.7500}},
    )

    run_3 = runs_repo.create(
        experiment_id=experiment.id,
        model_id="random_forest_classifier",
        random_seed=42,
        parameters={"n_estimators": 50},
        execution_id=exec_id_1,
    )
    runs_repo.transition(
        run_3.id,
        ModelRunStatus.COMPLETED,
        metrics={"values": {"roc_auc": 0.7950, "accuracy": 0.7670}},
    )

    # Create an older execution round 2 with a failed run
    exec_id_2 = "exec-round-2"
    run_4 = runs_repo.create(
        experiment_id=experiment.id,
        model_id="lightgbm_classifier",
        random_seed=42,
        parameters={},
        execution_id=exec_id_2,
    )
    runs_repo.transition(
        run_4.id,
        ModelRunStatus.FAILED,
        error_message="Memory limit",
    )

    fake_provider = FakeLLMProvider()
    prompts = ExplanationPromptBuilder(Path(__file__).parents[1] / "prompts")
    explanation_service = ExplanationService(
        provider=fake_provider,
        prompts=prompts,
        random_seed=42,
    )
    catalog = build_default_registry_catalog()
    storage = LocalFilesystemStorage(tmp_path / "artifacts")
    experiments_repo = SqlAlchemyExperimentRepository(migrated_database)
    datasets_repo = SqlAlchemyDatasetRepository(migrated_database)
    experiment_service = ExperimentService(
        experiments=experiments_repo,
        datasets=datasets_repo,
        validator=ExperimentSpecValidator(catalog),
    )

    app.dependency_overrides[get_experiment_service] = lambda: experiment_service
    app.dependency_overrides[get_model_run_repository] = lambda: runs_repo
    app.dependency_overrides[get_registry_catalog] = lambda: catalog
    app.dependency_overrides[get_explanation_service] = lambda: explanation_service
    app.dependency_overrides[get_artifact_storage] = lambda: storage

    client = TestClient(app)
    try:
        # 1. GET comparison defaults to latest round with completions (exec_id_1)
        response = client.get(f"/api/v1/experiments/{experiment.id}/comparison")
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == experiment.id
        assert data["primary_metric"] == "roc_auc"
        assert data["metric_direction"] == "maximize"
        assert data["selected_round_id"] == exec_id_1
        assert len(data["available_rounds"]) == 2
        assert len(data["ranked_models"]) == 3
        assert data["ranked_models"][0]["model_id"] == "lightgbm_classifier"
        assert data["ranked_models"][0]["primary_metric_value"] == 0.8123
        assert data["ranked_models"][0]["rank"] == 1
        assert data["ranked_models"][2]["model_id"] == "logistic_regression"
        assert data["ranked_models"][2]["rank"] == 3
        assert data["can_explain"] is True

        # Verify LLM was NOT called during GET
        assert len(fake_provider.recorded_requests) == 0

        # 2. GET comparison for exec_id_2 (where the only model failed)
        response_round_2 = client.get(
            f"/api/v1/experiments/{experiment.id}/comparison?execution_id={exec_id_2}"
        )
        assert response_round_2.status_code == 200
        data_2 = response_round_2.json()
        assert data_2["selected_round_id"] == exec_id_2
        assert len(data_2["ranked_models"]) == 0
        assert len(data_2["excluded_models"]) == 1
        assert data_2["excluded_models"][0]["reason"] == "status_failed"
        assert data_2["can_explain"] is False

        # 3. POST explanation on exec_id_2 -> returns 409 Conflict
        post_bad = client.post(
            f"/api/v1/experiments/{experiment.id}/explanation",
            json={"execution_id": exec_id_2, "language": "pt-BR"},
        )
        assert post_bad.status_code == 409
        assert post_bad.json()["detail"]["code"] == "insufficient_eligible_models"
        assert len(fake_provider.recorded_requests) == 0

        # 4. POST explanation on exec_id_1 with pt-BR -> returns 200 and calls LLM
        post_good = client.post(
            f"/api/v1/experiments/{experiment.id}/explanation",
            json={"execution_id": exec_id_1, "language": "pt-BR"},
        )
        assert post_good.status_code == 200
        result = post_good.json()
        assert "explanation" in result
        assert "LightGBM superou a regressão logística" in result["explanation"]["summary"]
        assert len(fake_provider.recorded_requests) == 1
        req = fake_provider.recorded_requests[0]
        assert req.language == "pt-BR"
        user_json = json.loads(req.user_prompt)
        assert user_json["scope"] == "experiment_comparison"
        assert user_json["context"]["best_model"]["model_id"] == "lightgbm_classifier"

        # 5. POST explanation on exec_id_1 with en-US
        post_en = client.post(
            f"/api/v1/experiments/{experiment.id}/explanation",
            json={"execution_id": exec_id_1, "language": "en-US"},
        )
        assert post_en.status_code == 200
        assert len(fake_provider.recorded_requests) == 2
        assert fake_provider.recorded_requests[1].language == "en-US"

        # 6. LLM Configuration error mapping -> 503
        fake_provider.error_to_raise = LLMConfigurationError("LLM not configured")
        post_503 = client.post(
            f"/api/v1/experiments/{experiment.id}/explanation",
            json={"execution_id": exec_id_1, "language": "pt-BR"},
        )
        assert post_503.status_code == 503
        fake_provider.error_to_raise = None

        # 7. LLM Provider error mapping -> 502
        fake_provider.error_to_raise = LLMProviderError("Upstream timeout")
        post_502 = client.post(
            f"/api/v1/experiments/{experiment.id}/explanation",
            json={"execution_id": exec_id_1, "language": "pt-BR"},
        )
        assert post_502.status_code == 502
        fake_provider.error_to_raise = None

        # 8. Non-existent experiment -> 404
        post_404 = client.post(
            "/api/v1/experiments/non-existent-id/explanation",
            json={"execution_id": exec_id_1, "language": "pt-BR"},
        )
        assert post_404.status_code == 404

    finally:
        app.dependency_overrides.clear()
