import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.application.decision_benchmarks.comparison import DecisionComparisonService
from app.domain.common import canonical_json
from app.domain.decision.questions import NoulQuestionSpec
from app.domain.decision.runs import DecisionRun, DecisionRunStatus, ZeroShotDecisionSpec
from app.domain.exceptions import DecisionBenchmarkError
from app.domain.experiment.entities import (
    Experiment,
    ExperimentMode,
    ModelRun,
    ModelRunStatus,
)
from app.domain.experiment.spec import TaskType
from app.infrastructure.storage.local import LocalFilesystemStorage


class ItemRepository:
    def __init__(self, item: Any) -> None:
        self.item = item

    def get(self, item_id: str) -> Any | None:
        return self.item if self.item.id == item_id else None


class ModelRunRepository:
    def __init__(self, runs: list[ModelRun]) -> None:
        self.runs = runs

    def list_for_experiment(self, experiment_id: str) -> list[ModelRun]:
        return [run for run in self.runs if run.experiment_id == experiment_id]


def save_parquet(storage: LocalFilesystemStorage, path: str, rows: list[dict[str, Any]]) -> str:
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buffer, index=False)
    return storage.save(path, buffer.getvalue())


def build_service(
    tmp_path: Path, *, experiment_dataset_id: str = "dataset"
) -> tuple[DecisionComparisonService, DecisionRun, Experiment]:
    storage = LocalFilesystemStorage(tmp_path)
    specification = ZeroShotDecisionSpec(
        dataset_id="dataset",
        target="label",
        task=TaskType.BINARY_CLASSIFICATION,
        positive_class="yes",
        input_columns=["message"],
        question=NoulQuestionSpec(instructions="Will this cancel?"),
        provider="typesafe_jev",
        model="jev-latest",
    )
    decision_run = replace(
        DecisionRun.create(specification, row_count=4, question_spec_hash="hash"),
        status=DecisionRunStatus.COMPLETED,
        estimated_cost=0.04,
    )
    save_parquet(
        storage,
        f"decision/runs/{decision_run.id}/responses.parquet",
        [
            {
                "row_id": str(index),
                "target": target,
                "prediction": prediction,
                "probability_yes": probability,
                "confidence": confidence,
                "correct": target == ("yes" if prediction == "yes" else "no"),
                "latency_ms": latency,
                "request_count": request_count,
                "cache_hit": cache_hit,
                "input_tokens": 5 if request_count else 0,
                "estimated_cost": cost,
            }
            for index, (
                target,
                prediction,
                probability,
                confidence,
                latency,
                request_count,
                cache_hit,
                cost,
            ) in enumerate(
                [
                    ("yes", "yes", 0.9, 0.8, 10.0, 1, False, 0.01),
                    ("no", "not:yes", 0.1, 0.8, 20.0, 1, False, 0.01),
                    ("yes", "yes", 0.8, 0.6, 30.0, 1, False, 0.01),
                    ("no", "yes", 0.7, 0.4, 0.0, 0, True, 0.0),
                ]
            )
        ],
    )
    experiment = Experiment.create(
        name="Supervised",
        mode=ExperimentMode.ASSISTED,
        dataset_id=experiment_dataset_id,
        objective="Predict label",
        spec_version="1.0",
        spec={"target": "label", "task": "binary_classification"},
    )
    predictions_path = save_parquet(
        storage,
        "supervised/predictions.parquet",
        [
            {
                "row_id": str(index),
                "fold_index": 0,
                "target": target,
                "prediction": prediction,
                "correct": target == prediction,
                "confidence": max(probabilities.values()),
                "probabilities": probabilities,
            }
            for index, (target, prediction, probabilities) in enumerate(
                [
                    ("yes", "yes", {"no": 0.1, "yes": 0.9}),
                    ("no", "no", {"no": 0.8, "yes": 0.2}),
                    ("yes", "no", {"no": 0.6, "yes": 0.4}),
                    ("no", "no", {"no": 0.7, "yes": 0.3}),
                ]
            )
        ],
    )
    model_run = ModelRun(
        experiment_id=experiment.id,
        model_id="logistic_regression",
        run_index=1,
        status=ModelRunStatus.COMPLETED,
        random_seed=41,
        parameters_json="{}",
        metrics_json=canonical_json({"predictions_path": predictions_path}),
        duration_seconds=1.25,
    )
    return (
        DecisionComparisonService(
            decision_runs=ItemRepository(decision_run),
            experiments=ItemRepository(experiment),
            model_runs=ModelRunRepository([model_run]),
            storage=storage,
        ),
        decision_run,
        experiment,
    )


def test_comparison_aligns_rows_and_reports_disagreement(tmp_path: Path) -> None:
    service, decision_run, experiment = build_service(tmp_path)

    result = service.compare(decision_run.id, experiment.id)

    assert result["contract"]["row_alignment"] == "intersection_by_row_id"
    assert result["contract"]["like_for_like_training"] is False
    model = result["models"][0]
    assert model["sample_size"] == 4
    assert model["run_index"] == 1
    assert model["supervised"]["metrics"]["accuracy"] == 0.75
    assert model["zero_shot"]["metrics"]["accuracy"] == 0.75
    assert model["metric_deltas"]["accuracy"] == 0
    assert model["zero_shot"]["request_count"] == 3
    assert model["zero_shot"]["cache_hits"] == 1
    assert model["zero_shot"]["estimated_cost"] == pytest.approx(0.03)
    assert model["disagreement"] == {
        "both_correct": 2,
        "supervised_only_correct": 1,
        "zero_shot_only_correct": 1,
        "both_wrong": 0,
        "prediction_disagreements": 2,
    }


def test_comparison_reads_legacy_catboost_singleton_labels(tmp_path: Path) -> None:
    service, decision_run, experiment = build_service(tmp_path)
    artifact_path = "supervised/predictions.parquet"
    frame = pd.read_parquet(tmp_path / artifact_path)
    frame["prediction"] = frame["prediction"].map(lambda label: str(np.asarray([label])))
    frame["correct"] = False  # Legacy artifacts also stored incorrect correctness.
    (tmp_path / artifact_path).unlink()
    frame.to_parquet(tmp_path / artifact_path, index=False)

    model = service.compare(decision_run.id, experiment.id)["models"][0]

    assert model["supervised"]["metrics"]["accuracy"] == 0.75
    assert model["disagreement"]["both_correct"] == 2


def test_comparison_does_not_accept_arbitrary_prediction_strings(tmp_path: Path) -> None:
    service, decision_run, experiment = build_service(tmp_path)
    artifact_path = "supervised/predictions.parquet"
    frame = pd.read_parquet(tmp_path / artifact_path)
    frame.loc[0, "prediction"] = "['unknown']"
    (tmp_path / artifact_path).unlink()
    frame.to_parquet(tmp_path / artifact_path, index=False)

    with pytest.raises(DecisionBenchmarkError, match="class contract"):
        service.compare(decision_run.id, experiment.id)


def test_legacy_catboost_numeric_class_labels_are_normalized() -> None:
    predictions = np.asarray(["[0]", "[1]", "[2]"])

    assert DecisionComparisonService._normalize_supervised_predictions(
        predictions, classes=["0", "1", "2"]
    ).tolist() == ["0", "1", "2"]


def test_comparison_rejects_different_dataset(tmp_path: Path) -> None:
    service, decision_run, experiment = build_service(
        tmp_path, experiment_dataset_id="other-dataset"
    )

    with pytest.raises(DecisionBenchmarkError, match="same dataset"):
        service.compare(decision_run.id, experiment.id)


def test_comparison_rejects_duplicate_row_ids(tmp_path: Path) -> None:
    service, decision_run, experiment = build_service(tmp_path)
    artifact_path = f"decision/runs/{decision_run.id}/responses.parquet"
    (tmp_path / artifact_path).unlink()
    save_parquet(
        service.storage,
        artifact_path,
        [
            {
                "row_id": "0",
                "target": "yes",
                "prediction": "yes",
                "probability_yes": 0.9,
                "confidence": 0.9,
                "correct": True,
            },
            {
                "row_id": "0",
                "target": "yes",
                "prediction": "yes",
                "probability_yes": 0.9,
                "confidence": 0.9,
                "correct": True,
            },
        ],
    )

    with pytest.raises(DecisionBenchmarkError, match="duplicate row IDs"):
        service.compare(decision_run.id, experiment.id)


def test_binary_metrics_respect_explicit_positive_class() -> None:
    probabilities = np.asarray(
        [
            [0.6, 0.4],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    metrics = DecisionComparisonService._metrics(
        target=np.asarray(["approve", "decline", "decline", "decline"]),
        predictions=np.asarray(["approve", "approve", "decline", "decline"]),
        probabilities=probabilities,
        confidence=np.max(probabilities, axis=1),
        classes=["approve", "decline"],
        positive_class="approve",
    )

    assert metrics["pr_auc"] == pytest.approx(0.5)
