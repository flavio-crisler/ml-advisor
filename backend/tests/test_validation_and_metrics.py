import numpy as np
import pytest

from app.domain.exceptions import MetricEvaluationError, ValidationStrategyError
from app.domain.experiment.spec import (
    AveragingStrategy,
    MetricsSpec,
    RocAucStrategy,
    TaskType,
    ValidationSpec,
)
from app.ml.metrics.evaluator import MetricEvaluation, MetricEvaluator
from app.ml.metrics.leaderboard import LeaderboardBuilder, LeaderboardEntry
from app.ml.registries.catalog import build_default_registry_catalog
from app.ml.validation.splitters import ValidationSplitter


def test_stratified_folds_are_reproducible_and_balanced() -> None:
    splitter = ValidationSplitter()
    target = np.asarray([0, 1, 2] * 12)
    specification = ValidationSpec(
        strategy="stratified_kfold", folds=3, shuffle=True, random_seed=41
    )

    first = splitter.split(specification, TaskType.MULTICLASS_CLASSIFICATION, target)
    repeated = splitter.split(specification, TaskType.MULTICLASS_CLASSIFICATION, target)
    changed = splitter.split(
        specification.model_copy(update={"random_seed": 99}),
        TaskType.MULTICLASS_CLASSIFICATION,
        target,
    )

    assert first == repeated
    assert first != changed
    assert len(first.folds) == 3
    for fold in first.folds:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
        assert np.bincount(target[list(fold.test_indices)]).tolist() == [4, 4, 4]
    assert first.to_dict()["random_seed"] == 41


def test_train_test_and_regression_kfold_preserve_all_indices() -> None:
    splitter = ValidationSplitter()
    classification = splitter.split(
        ValidationSpec(strategy="train_test_split", test_size=0.25, random_seed=7),
        TaskType.BINARY_CLASSIFICATION,
        [0, 1] * 20,
    )
    fold = classification.folds[0]
    assert len(fold.train_indices) == 30
    assert len(fold.test_indices) == 10
    assert set(fold.train_indices) | set(fold.test_indices) == set(range(40))

    regression = splitter.split(
        ValidationSpec(strategy="kfold", folds=4, random_seed=7),
        TaskType.REGRESSION,
        np.linspace(0, 1, 20),
    )
    assert len(regression.folds) == 4
    assert sorted(index for split in regression.folds for index in split.test_indices) == list(
        range(20)
    )

    with pytest.raises(ValidationStrategyError, match="cannot be used for regression"):
        splitter.split(
            ValidationSpec(strategy="stratified_kfold"),
            TaskType.REGRESSION,
            np.linspace(0, 1, 20),
        )


def test_binary_metrics_use_explicit_positive_label_and_probability_order() -> None:
    evaluator = MetricEvaluator(build_default_registry_catalog().metrics)
    result = evaluator.evaluate(
        task=TaskType.BINARY_CLASSIFICATION,
        specification=MetricsSpec(
            primary="f1",
            secondary=["accuracy", "precision", "recall", "roc_auc", "pr_auc", "log_loss"],
            positive_label="yes",
        ),
        target=["no", "yes", "yes", "no"],
        predictions=["no", "yes", "no", "no"],
        probabilities=np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.7, 0.3],
            ]
        ),
        probability_classes=("no", "yes"),
    )

    assert result.primary_metric == "f1"
    assert result.values["accuracy"] == pytest.approx(0.75)
    assert result.values["precision"] == pytest.approx(1.0)
    assert result.values["recall"] == pytest.approx(0.5)
    assert result.values["f1"] == pytest.approx(2 / 3)
    assert result.values["roc_auc"] == pytest.approx(1.0)

    with pytest.raises(MetricEvaluationError, match="require probabilities"):
        evaluator.evaluate(
            task=TaskType.BINARY_CLASSIFICATION,
            specification=MetricsSpec(primary="roc_auc", positive_label="yes"),
            target=["no", "yes"],
            predictions=["no", "yes"],
        )


def test_multiclass_metrics_use_explicit_averaging_and_roc_strategy() -> None:
    evaluator = MetricEvaluator(build_default_registry_catalog().metrics)
    target = ["a", "b", "c", "a", "b", "c"]
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    result = evaluator.evaluate(
        task=TaskType.MULTICLASS_CLASSIFICATION,
        specification=MetricsSpec(
            primary="f1",
            secondary=["precision", "recall", "roc_auc", "pr_auc", "log_loss"],
            averaging=AveragingStrategy.WEIGHTED,
            roc_auc_strategy=RocAucStrategy.OVR,
        ),
        target=target,
        predictions=target,
        probabilities=probabilities,
        probability_classes=("a", "b", "c"),
    )

    assert result.values["f1"] == pytest.approx(1.0)
    assert result.values["precision"] == pytest.approx(1.0)
    assert result.values["recall"] == pytest.approx(1.0)
    assert result.values["roc_auc"] == pytest.approx(1.0)
    assert result.values["pr_auc"] == pytest.approx(1.0)
    assert result.values["log_loss"] > 0


def test_regression_metrics_aggregate_folds_and_reject_zero_mape() -> None:
    evaluator = MetricEvaluator(build_default_registry_catalog().metrics)
    result = evaluator.evaluate(
        task=TaskType.REGRESSION,
        specification=MetricsSpec(primary="rmse", secondary=["mae", "mse", "r2", "mape"]),
        target=[1.0, 2.0, 4.0],
        predictions=[1.0, 3.0, 3.0],
    )
    assert result.values["mae"] == pytest.approx(2 / 3)
    assert result.values["mse"] == pytest.approx(2 / 3)
    assert result.values["rmse"] == pytest.approx(np.sqrt(2 / 3))

    means, deviations = evaluator.aggregate(
        [
            MetricEvaluation(primary_metric="rmse", values={"rmse": 1.0}),
            MetricEvaluation(primary_metric="rmse", values={"rmse": 3.0}),
        ]
    )
    assert means == {"rmse": 2.0}
    assert deviations == {"rmse": 1.0}

    with pytest.raises(MetricEvaluationError, match="undefined"):
        evaluator.evaluate(
            task=TaskType.REGRESSION,
            specification=MetricsSpec(primary="mape"),
            target=[0.0, 1.0],
            predictions=[0.1, 1.0],
        )


def test_leaderboard_respects_metric_direction_and_stable_ties() -> None:
    builder = LeaderboardBuilder(build_default_registry_catalog().metrics)
    entries = [
        LeaderboardEntry("run-b", "forest", {"f1": 0.9, "rmse": 2.0}, 2.0),
        LeaderboardEntry("run-a", "linear", {"f1": 0.9, "rmse": 1.0}, 1.0),
        LeaderboardEntry("run-c", "boost", {"f1": 0.8, "rmse": 0.5}, 0.5),
    ]

    classification = builder.build("f1", entries)
    regression = builder.build("rmse", entries)

    assert classification.direction == "maximize"
    assert [entry.run_id for entry in classification.entries] == ["run-a", "run-b", "run-c"]
    assert regression.direction == "minimize"
    assert [entry.run_id for entry in regression.entries] == ["run-c", "run-a", "run-b"]
