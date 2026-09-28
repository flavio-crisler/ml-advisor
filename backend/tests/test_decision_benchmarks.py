import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.application.datasets.profiling import DeterministicDatasetProfiler
from app.application.decision_benchmarks.execution import DecisionExecutionService
from app.application.decision_benchmarks.metrics import evaluate_binary, evaluate_multiclass
from app.application.decisions.registry import (
    DecisionProviderRegistration,
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
)
from app.domain.dataset.entities import Dataset
from app.domain.decision.provider import DecisionProviderCapabilities, ProviderHealth
from app.domain.decision.questions import NoulQuestionSpec
from app.domain.decision.responses import (
    DecisionResponse,
    DecisionUsage,
    NoulDecisionAnswer,
)
from app.domain.decision.runs import DecisionRun, DecisionRunStatus, ZeroShotDecisionSpec
from app.domain.decision.specifications import DecisionSpec, DecisionState
from app.domain.experiment.spec import AveragingStrategy, TaskType
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyCalibrationRunRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyDecisionCacheRepository,
    SqlAlchemyDecisionRunRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage


class FakeDecisionProvider:
    id = "typesafe_jev"
    capabilities = DecisionProviderCapabilities(
        supports_noul=True,
        supports_choice=True,
        supports_score=True,
        supports_model_version_resolution=True,
        is_external=True,
    )

    def __init__(self) -> None:
        self.states: list[DecisionState] = []

    def get_capabilities(self) -> DecisionProviderCapabilities:
        return self.capabilities

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status="available")

    async def evaluate(self, state: DecisionState, questions: dict[str, Any]) -> DecisionResponse:
        self.states.append(state)
        text = str(state.values["message"])
        probability = 0.9 if "cancel" in text else 0.1
        return DecisionResponse(
            provider_id=self.id,
            requested_model="jev-latest",
            resolved_model="jev-1.13.0",
            answers={
                "prediction": NoulDecisionAnswer(
                    probability_yes=probability,
                    confidence=0.8,
                )
            },
            usage=DecisionUsage(input_tokens=5, estimated_cost=0.01),
            latency_ms=12,
        )


def test_binary_metrics_include_calibration_and_thresholds() -> None:
    result = evaluate_binary(
        targets=["yes", "no", "yes", "no"],
        positive_class="yes",
        probabilities=[0.9, 0.2, 0.8, 0.1],
        confidences=[0.8, 0.6, 0.6, 0.8],
        latencies_ms=[10, 20, 30, 40],
        total_cost=0.04,
    )

    assert result.metrics["accuracy"] == 1
    assert result.metrics["brier_score"] == pytest.approx(0.025)
    assert result.calibration.expected_calibration_error == pytest.approx(0.3)
    assert result.calibration.thresholds[0].coverage == 1
    assert {point.estimated_cost for point in result.calibration.thresholds} == {0.04}


def test_multiclass_metrics_use_explicit_averaging() -> None:
    result = evaluate_multiclass(
        targets=["billing", "technical", "cancel"],
        classes=["billing", "technical", "cancel"],
        probabilities=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        predictions=["billing", "technical", "cancel"],
        confidences=[0.7, 0.7, 0.6],
        latencies_ms=[10, 11, 12],
        total_cost=None,
        averaging=AveragingStrategy.MACRO,
    )

    assert result.metrics["accuracy"] == 1
    assert result.metrics["averaging"] == "macro"
    assert set(result.metrics["calibration_by_class"]) == {
        "billing",
        "technical",
        "cancel",
    }


def test_zero_shot_spec_forbids_target_input_and_wrong_question_type() -> None:
    with pytest.raises(ValueError, match="target cannot"):
        ZeroShotDecisionSpec(
            dataset_id="dataset",
            target="label",
            task=TaskType.BINARY_CLASSIFICATION,
            positive_class="yes",
            input_columns=["label"],
            question=NoulQuestionSpec(instructions="Will this cancel?"),
            provider="typesafe_jev",
            model="jev-latest",
        )
    with pytest.raises(ValueError, match="Choice"):
        ZeroShotDecisionSpec(
            dataset_id="dataset",
            target="label",
            task=TaskType.MULTICLASS_CLASSIFICATION,
            input_columns=["message"],
            question=NoulQuestionSpec(instructions="What is the topic?"),
            provider="typesafe_jev",
            model="jev-latest",
        )


def test_execution_excludes_target_persists_artifacts_and_reuses_cache(
    migrated_database: Any, tmp_path: Path
) -> None:
    dataset_storage = LocalFilesystemStorage(tmp_path / "datasets")
    artifact_storage = LocalFilesystemStorage(tmp_path / "artifacts")
    csv = b"message;label\nplease cancel;yes\nall good;no\ncancel now;yes\n"
    path = dataset_storage.save("dataset/original.csv", csv)
    profile = DeterministicDatasetProfiler().profile(
        pd.read_csv(io.BytesIO(csv), sep=";"),
        size_bytes=len(csv),
        encoding="utf-8",
        delimiter=";",
    )
    profile_path = dataset_storage.save("dataset/profile.json", profile.model_dump_json().encode())
    dataset = SqlAlchemyDatasetRepository(migrated_database).add(
        Dataset(
            id="dataset",
            name="Churn",
            original_filename="churn.csv",
            content_hash=hashlib.sha256(csv).hexdigest(),
            row_count=3,
            column_count=2,
            storage_path=path,
            profile_path=profile_path,
        )
    )
    specification = ZeroShotDecisionSpec(
        dataset_id=dataset.id,
        target="label",
        task=TaskType.BINARY_CLASSIFICATION,
        positive_class="yes",
        input_columns=["message"],
        question=NoulQuestionSpec(instructions="Will this customer cancel?"),
        provider="typesafe_jev",
        model="jev-latest",
    )
    question_hash = DecisionSpec(
        provider=specification.provider,
        model=specification.model,
        questions={"prediction": specification.question},
    ).question_spec_hash()
    repositories = {
        "runs": SqlAlchemyDecisionRunRepository(migrated_database),
        "calibrations": SqlAlchemyCalibrationRunRepository(migrated_database),
        "cache": SqlAlchemyDecisionCacheRepository(migrated_database),
    }
    provider = FakeDecisionProvider()
    registry = DecisionProviderRegistry()
    registry.register(
        DecisionProviderRegistration(
            provider_id="typesafe_jev",
            display_name="Fake Jev",
            capabilities=provider.capabilities,
            factory=lambda _config: provider,
        )
    )
    config = DecisionProviderRuntimeConfig(
        enabled=True,
        provider_id="typesafe_jev",
        model="jev-latest",
        api_key="secret",
        concurrency=2,
    )

    def execute(run: DecisionRun) -> dict[str, Any]:
        repositories["runs"].add(run)
        return DecisionExecutionService(
            datasets=SqlAlchemyDatasetRepository(migrated_database),
            runs=repositories["runs"],
            calibrations=repositories["calibrations"],
            cache=repositories["cache"],
            dataset_storage=dataset_storage,
            artifact_storage=artifact_storage,
            providers=registry,
            provider_config=config,
            chunk_size=2,
        ).execute(run.id)

    first = DecisionRun.create(specification, row_count=3, question_spec_hash=question_hash)
    first_result = execute(first)
    persisted = repositories["runs"].get(first.id)
    assert persisted is not None
    assert persisted.status == DecisionRunStatus.COMPLETED
    assert persisted.processed_rows == 3
    assert persisted.request_count == 3
    assert persisted.cache_hits == 0
    assert all(set(state.values) == {"message"} for state in provider.states)
    responses = pd.read_parquet(io.BytesIO(artifact_storage.load(first_result["responses_path"])))
    assert {"cache_hit", "request_count", "input_tokens", "estimated_cost"} <= set(
        responses.columns
    )
    assert not responses["cache_hit"].any()
    assert repositories["calibrations"].get_for_decision_run(first.id) is not None

    provider.states.clear()
    second = DecisionRun.create(specification, row_count=3, question_spec_hash=question_hash)
    second_result = execute(second)
    cached = repositories["runs"].get(second.id)
    assert cached is not None
    assert cached.request_count == 0
    assert cached.cache_hits == 3
    assert provider.states == []
    cached_summary = json.loads(artifact_storage.load(second_result["summary_path"]))
    assert cached_summary["latency_p50_ms"] is None
    assert cached_summary["latency_p95_ms"] is None
    cached_calibration = repositories["calibrations"].get_for_decision_run(second.id)
    assert cached_calibration is not None
    assert cached_calibration.metrics["latency_ms"] == []
