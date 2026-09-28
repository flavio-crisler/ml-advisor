import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_advisor_service
from app.application.advisor.errors import AdvisorRecommendationError
from app.application.advisor.models import RecommendationSpec
from app.application.advisor.prompts import AdvisorPromptBuilder
from app.application.advisor.service import AdvisorService
from app.application.advisor.validation import AdvisorRecommendationValidator
from app.application.datasets.profiling import DatasetProfile, DeterministicDatasetProfiler
from app.application.experiments.validation import ExperimentSpecValidator
from app.domain.dataset.entities import Dataset
from app.llm.base import LLMCapabilities, LLMProvider, LLMRequest, StructuredResponse
from app.llm.errors import LLMResponseError
from app.main import app
from app.ml.registries.catalog import build_default_registry_catalog


def _recommendation(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "task": "binary_classification",
        "target": "target",
        "features": ["active", "segment", "amount"],
        "models": [
            {"id": "logistic_regression"},
            {"id": "random_forest_classifier"},
        ],
        "preprocessing": [
            {"id": "missing_values", "strategy": "median"},
            {"id": "categorical_encoding", "strategy": "one_hot"},
            {"id": "boolean_encoding", "strategy": "binary"},
            {"id": "numeric_scaling", "strategy": "standard"},
        ],
        "validation": {"strategy": "stratified_kfold", "folds": 5},
        "metrics": {
            "primary": "f1",
            "secondary": ["precision", "recall", "roc_auc"],
            "positive_label": "yes",
        },
        "tuning": {"enabled": False},
        "explanation": "Compare a linear baseline with a non-linear ensemble.",
        "assumptions": ["The target is known at training time."],
        "warnings": ["Class balance should be reviewed."],
    }
    values.update(overrides)
    return values


class FakeProvider(LLMProvider):
    id = "fake"
    capabilities = LLMCapabilities(supports_json_schema=True)

    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None):
        self.payload = payload or _recommendation()
        self.error = error
        self.requests: list[LLMRequest] = []

    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.payload)


class SequenceProvider(FakeProvider):
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        super().__init__()
        self.payloads = payloads

    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.payloads) - 1)
        return response_model.model_validate(self.payloads[index])


class FakeDatasetService:
    def __init__(self, dataset: Dataset, profile: DatasetProfile) -> None:
        self.dataset = dataset
        self.profile = profile

    def get_dataset(self, dataset_id: str) -> Dataset:
        assert dataset_id == self.dataset.id
        return self.dataset

    def get_profile(self, dataset_id: str) -> DatasetProfile:
        assert dataset_id == self.dataset.id
        return self.profile


@pytest.fixture
def advisor_dataset() -> tuple[Dataset, DatasetProfile]:
    csv_path = Path(__file__).parent / "fixtures" / "attendance.csv"
    frame = pd.read_csv(csv_path, sep=";")
    profile = DeterministicDatasetProfiler(preview_rows=5).profile(
        frame,
        size_bytes=csv_path.stat().st_size,
        encoding="utf-8",
        delimiter=";",
    )
    dataset = Dataset(
        id="attendance-dataset",
        name="Attendance",
        original_filename="attendance.csv",
        content_hash="sha256:attendance",
        row_count=len(frame),
        column_count=len(frame.columns),
        storage_path="attendance/original.csv",
        profile_path="attendance/profile.json",
    )
    return dataset, profile


def _service(
    advisor_dataset: tuple[Dataset, DatasetProfile],
    provider: LLMProvider,
    *,
    send_sample: bool = False,
    sample_size: int = 2,
) -> AdvisorService:
    dataset, profile = advisor_dataset
    catalog = build_default_registry_catalog()
    return AdvisorService(
        datasets=FakeDatasetService(dataset, profile),  # type: ignore[arg-type]
        provider=provider,
        catalog=catalog,
        validator=AdvisorRecommendationValidator(ExperimentSpecValidator(catalog)),
        prompts=AdvisorPromptBuilder(Path(__file__).parents[1] / "prompts"),
        send_sample=send_sample,
        sample_size=sample_size,
        random_seed=41,
    )


def test_advisor_builds_valid_proposal_without_sending_rows_by_default(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    provider = FakeProvider()
    result = _service(advisor_dataset, provider).recommend(
        dataset_id="attendance-dataset",
        objective="Predict missed appointments",
        language="en-US",
    )

    assert result.provider_id == "fake"
    assert result.sample_sent is False
    assert result.sample_rows == 0
    assert result.proposal.dataset_hash == "sha256:attendance"
    assert result.proposal.target == "target"
    assert result.proposal.column_roles["customer_id"] == "id"
    assert result.proposal.column_roles["constant"] == "ignored"
    assert result.recommendation.explanation.startswith("Compare")
    prompt_payload = json.loads(provider.requests[0].user_prompt)
    assert "sample" not in prompt_payload
    assert "preview" not in prompt_payload["dataset_profile"]
    assert len(prompt_payload["available_components"]["models"]) == 14
    assert provider.requests[0].temperature == 0
    assert "never claim" in provider.requests[0].system_prompt


def test_sample_is_sent_only_when_explicitly_enabled(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    provider = FakeProvider()
    result = _service(advisor_dataset, provider, send_sample=True, sample_size=2).recommend(
        dataset_id="attendance-dataset",
        objective="Prever faltas",
        language="pt-BR",
    )

    payload = json.loads(provider.requests[0].user_prompt)
    assert len(payload["sample"]) == 2
    assert result.sample_sent is True
    assert result.sample_rows == 2
    assert "nunca afirme" in provider.requests[0].system_prompt


@pytest.mark.parametrize(
    ("overrides", "expected_issue"),
    [
        ({"target": "does_not_exist"}, "unknown_target:does_not_exist"),
        ({"features": ["amount", "ghost"]}, "unknown_feature:ghost"),
        ({"models": [{"id": "invented_model"}]}, "unknown_model:invented_model"),
        (
            {"validation": {"strategy": "kfold"}},
            "validator_task_incompatible:kfold:binary_classification",
        ),
    ],
)
def test_advisor_rejects_unknown_or_incompatible_recommendations(
    advisor_dataset: tuple[Dataset, DatasetProfile],
    overrides: dict[str, Any],
    expected_issue: str,
) -> None:
    provider = FakeProvider(_recommendation(**overrides))

    with pytest.raises(AdvisorRecommendationError) as error:
        _service(advisor_dataset, provider).recommend(
            dataset_id="attendance-dataset",
            objective="Predict missed appointments",
            language="en-US",
        )

    assert expected_issue in error.value.issues


def test_advisor_repairs_column_names_used_as_preprocessors_for_support_dataset() -> None:
    frame = pd.DataFrame(
        {
            "message": [
                f"Customer asks to cancel after a recurring charge {index}"
                if index % 2
                else f"Customer confirms renewal after a useful service {index}"
                for index in range(30)
            ],
            "segment": ["A" if index % 2 else "B" for index in range(30)],
            "target": ["yes" if index % 2 else "no" for index in range(30)],
        }
    )
    profile = DeterministicDatasetProfiler(preview_rows=5).profile(
        frame,
        size_bytes=len(frame.to_csv(index=False).encode()),
        encoding="utf-8",
        delimiter=",",
    )
    dataset = Dataset(
        id="support-dataset",
        name="support_semantic",
        original_filename="support_semantic.csv",
        content_hash="sha256:support",
        row_count=len(frame),
        column_count=len(frame.columns),
        storage_path="support/original.csv",
        profile_path="support/profile.json",
    )
    valid = _recommendation(
        target="target",
        features=["message", "segment"],
        models=[{"id": "logistic_regression"}],
        preprocessing=[
            {"id": "text_vectorization", "strategy": "tfidf"},
            {"id": "categorical_encoding", "strategy": "one_hot"},
        ],
        validation={"strategy": "stratified_kfold", "folds": 3},
        metrics={"primary": "accuracy"},
    )
    invalid = {
        **valid,
        "preprocessing": [
            {"id": column, "strategy": "tfidf"} for column in ("message", "segment", "target")
        ],
    }
    provider = SequenceProvider([invalid, valid])
    catalog = build_default_registry_catalog()
    service = AdvisorService(
        datasets=FakeDatasetService(dataset, profile),  # type: ignore[arg-type]
        provider=provider,
        catalog=catalog,
        validator=AdvisorRecommendationValidator(ExperimentSpecValidator(catalog)),
        prompts=AdvisorPromptBuilder(Path(__file__).parents[1] / "prompts"),
        send_sample=False,
        sample_size=5,
        random_seed=41,
    )

    result = service.recommend(
        dataset_id=dataset.id,
        objective="Predict target using the other variables",
        language="en-US",
    )

    assert result.proposal.target == "target"
    assert result.proposal.features == ["message", "segment"]
    assert [step.id for step in result.proposal.preprocessing] == [
        "text_vectorization",
        "categorical_encoding",
    ]
    assert len(provider.requests) == 2
    feedback = json.loads(provider.requests[1].user_prompt)["validation_feedback"]
    assert feedback["validation_errors"] == ["unknown_preprocessor"]
    assert "available_components.preprocessors" in feedback["instruction"]
    assert "unknown_preprocessor:message" not in provider.requests[1].user_prompt


def test_advisor_stops_after_one_failed_repair(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    invalid = _recommendation(preprocessing=[{"id": "amount", "strategy": "median"}])
    provider = SequenceProvider([invalid, invalid])

    with pytest.raises(AdvisorRecommendationError) as error:
        _service(advisor_dataset, provider).recommend(
            dataset_id="attendance-dataset",
            objective="Predict missed appointments",
            language="en-US",
        )

    assert error.value.issues == ("unknown_preprocessor:amount",)
    assert len(provider.requests) == 2


def test_advisor_contract_feedback_identifies_missing_binary_positive_label(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    dataset, profile = advisor_dataset
    catalog = build_default_registry_catalog()
    validator = AdvisorRecommendationValidator(ExperimentSpecValidator(catalog))
    recommendation = RecommendationSpec.model_validate(_recommendation(metrics={"primary": "f1"}))

    with pytest.raises(AdvisorRecommendationError) as error:
        validator.to_proposal(
            recommendation,
            dataset=dataset,
            profile=profile,
            objective="Predict missed appointments",
        )

    assert error.value.issues == ("invalid_contract:binary_positive_label_required",)
    prompts = AdvisorPromptBuilder(Path(__file__).parents[1] / "prompts")
    request = prompts.build(
        objective="Predict missed appointments",
        profile=profile,
        catalog=catalog,
        language="en-US",
        send_sample=False,
        sample_size=5,
        random_seed=41,
    )
    feedback = prompts.with_validation_feedback(request, error.value.issues)
    assert json.loads(feedback.user_prompt)["validation_feedback"]["validation_errors"] == [
        "invalid_contract:binary_positive_label_required"
    ]


def test_advisor_api_returns_proposal_for_review_without_starting_training(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    service = _service(advisor_dataset, FakeProvider())
    app.dependency_overrides[get_advisor_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/api/v1/advisor/recommend",
            json={
                "dataset_id": "attendance-dataset",
                "objective": "Predict missed appointments",
                "name": "Assisted attendance",
                "language": "en-US",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["proposal"]["name"] == "Assisted attendance"
    assert response.json()["proposal"]["models"][0]["id"] == "logistic_regression"
    assert "job_id" not in response.json()


def test_advisor_api_maps_invalid_provider_response_to_safe_error(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    provider = FakeProvider(error=LLMResponseError("unsafe provider detail"))
    app.dependency_overrides[get_advisor_service] = lambda: _service(advisor_dataset, provider)
    try:
        response = TestClient(app).post(
            "/api/v1/advisor/recommend",
            json={
                "dataset_id": "attendance-dataset",
                "objective": "Predict missed appointments",
                "language": "en-US",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "code": "llm_response_error",
        "message_key": "advisor.errors.llm_response_error",
    }
    assert "unsafe provider detail" not in response.text


def test_advisor_deduplicates_repeated_preprocessing_steps(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    dataset, profile = advisor_dataset
    catalog = build_default_registry_catalog()
    validator = AdvisorRecommendationValidator(ExperimentSpecValidator(catalog))
    recommendation = RecommendationSpec.model_validate(
        _recommendation(
            preprocessing=[
                {"id": "missing_values", "strategy": "median"},
                {"id": "missing_values", "strategy": "mode"},
                {"id": "categorical_encoding", "strategy": "one_hot"},
                {"id": "numeric_scaling", "strategy": "standard"},
            ]
        )
    )

    proposal = validator.to_proposal(
        recommendation,
        dataset=dataset,
        profile=profile,
        objective="Predict missed appointments",
    )

    assert [step.id for step in proposal.preprocessing] == [
        "missing_values",
        "categorical_encoding",
        "numeric_scaling",
    ]
    assert proposal.preprocessing[0].strategy == "median"


def test_advisor_defensively_removes_target_from_features(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    dataset, profile = advisor_dataset
    catalog = build_default_registry_catalog()
    validator = AdvisorRecommendationValidator(ExperimentSpecValidator(catalog))
    recommendation = RecommendationSpec.model_validate(
        _recommendation(features=["target", "active", "segment", "amount", "active"])
    )

    proposal = validator.to_proposal(
        recommendation,
        dataset=dataset,
        profile=profile,
        objective="Predict missed appointments",
    )

    assert proposal.target == "target"
    assert proposal.features == ["active", "segment", "amount"]
    assert "target" not in proposal.features


def test_advisor_deduplicates_models_and_secondary_metrics(
    advisor_dataset: tuple[Dataset, DatasetProfile],
) -> None:
    dataset, profile = advisor_dataset
    catalog = build_default_registry_catalog()
    validator = AdvisorRecommendationValidator(ExperimentSpecValidator(catalog))
    recommendation = RecommendationSpec.model_validate(
        _recommendation(
            models=[
                {"id": "logistic_regression"},
                {"id": "logistic_regression"},
                {"id": "random_forest_classifier"},
            ],
            metrics={
                "primary": "f1",
                "secondary": ["precision", "f1", "recall", "precision"],
                "positive_label": "yes",
            },
        )
    )

    proposal = validator.to_proposal(
        recommendation,
        dataset=dataset,
        profile=profile,
        objective="Predict missed appointments",
    )

    assert [model.id for model in proposal.models] == [
        "logistic_regression",
        "random_forest_classifier",
    ]
    assert proposal.metrics.secondary == ["precision", "recall"]
