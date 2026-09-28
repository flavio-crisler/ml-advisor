import json
from pathlib import Path

import pandas as pd
import pytest

from app.domain.exceptions import PipelineCompilationError
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    PreprocessingStepSpec,
    TaskType,
    ValidationSpec,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.ml.preprocessing.compiler import PipelineCompiler
from app.ml.preprocessing.numerical import NumericScaler
from app.ml.registries.metadata import ModelCapabilities


def _spec(
    features: list[str],
    preprocessing: list[PreprocessingStepSpec],
) -> ExperimentSpec:
    return ExperimentSpec(
        dataset_id="dataset",
        dataset_hash="sha256:dataset",
        profile_version="1.0",
        name="Preprocessing test",
        objective="Verify deterministic feature transformations",
        task=TaskType.BINARY_CLASSIFICATION,
        target="target",
        features=features,
        column_roles={
            **{feature: ColumnRole.FEATURE for feature in features},
            "target": ColumnRole.TARGET,
        },
        models=[ModelSpec(id="logistic_regression")],
        preprocessing=preprocessing,
        validation=ValidationSpec(strategy="stratified_kfold"),
        metrics=MetricsSpec(primary="f1", positive_label=1),
    )


def test_tfidf_is_explicit_bounded_and_fitted_only_on_training_rows() -> None:
    spec = _spec(
        ["message"],
        [PreprocessingStepSpec(id="text_vectorization", strategy="tfidf")],
    )
    compiler = PipelineCompiler()
    with pytest.raises(PipelineCompilationError, match="require text_vectorization"):
        compiler.compile(
            _spec(["message"], []),
            column_types={"message": "text"},
            capabilities=ModelCapabilities(),
        )
    pipeline = compiler.compile(
        spec, column_types={"message": "text"}, capabilities=ModelCapabilities()
    )
    train, _ = pipeline.fit_transform(
        pd.DataFrame({"message": ["good service", "bad service"]}),
        pd.Series([1, 0]),
    )
    holdout = pipeline.transform(pd.DataFrame({"message": ["unseenword service"]}))
    assert list(train.columns) == list(holdout.columns)
    assert not any("unseenword" in name for name in train.columns)
    assert train.shape[1] <= 256
    assert pipeline.get_config()["feature_transformers"][-2]["id"] == "text_vectorization"

    with_missing = compiler.compile(
        spec, column_types={"message": "text"}, capabilities=ModelCapabilities()
    )
    with_missing.fit_transform(pd.DataFrame({"message": ["good service", None]}), pd.Series([1, 0]))
    missing_config = with_missing.get_config()["feature_transformers"][0]
    assert missing_config["fill_values"]["message"] == ""

    with pytest.raises(PipelineCompilationError, match="max_features"):
        compiler.compile(
            _spec(
                ["message"],
                [
                    PreprocessingStepSpec(
                        id="text_vectorization", strategy="tfidf", parameters={"max_features": 5000}
                    )
                ],
            ),
            column_types={"message": "text"},
            capabilities=ModelCapabilities(),
        )

    override = compiler.compile(
        _spec(
            ["message"],
            [
                PreprocessingStepSpec(
                    id="text_vectorization",
                    strategy="tfidf",
                    parameters={"columns": ["message"]},
                )
            ],
        ),
        column_types={"message": "categorical"},
        capabilities=ModelCapabilities(),
    )
    overridden, _ = override.fit_transform(
        pd.DataFrame({"message": ["short note", "other note"]}), pd.Series([0, 1])
    )
    assert all(column.startswith("tfidf__message__") for column in overridden)


def test_compiler_builds_complete_pipeline_and_persists_fitted_config(tmp_path: Path) -> None:
    features = ["age", "income", "segment", "active", "created_at", "constant"]
    spec = _spec(
        features,
        [
            PreprocessingStepSpec(id="missing_values", strategy="median"),
            PreprocessingStepSpec(id="categorical_encoding", strategy="one_hot"),
            PreprocessingStepSpec(id="boolean_encoding", strategy="binary"),
            PreprocessingStepSpec(id="numeric_scaling", strategy="standard"),
            PreprocessingStepSpec(
                id="outliers",
                strategy="quantile_clip",
                parameters={"lower_quantile": 0.05, "upper_quantile": 0.95},
            ),
            PreprocessingStepSpec(id="datetime", strategy="components"),
            PreprocessingStepSpec(id="imbalance", strategy="class_weight"),
            PreprocessingStepSpec(
                id="feature_selection",
                strategy="variance_threshold",
                parameters={"threshold": 0.0},
            ),
        ],
    )
    pipeline = PipelineCompiler().compile(
        spec,
        column_types={
            "age": "numeric",
            "income": "numeric",
            "segment": "categorical",
            "active": "boolean",
            "created_at": "datetime",
            "constant": "numeric",
        },
        capabilities=ModelCapabilities(
            supports_proba=True,
            supports_imbalance_weighting=True,
            requires_scaling=True,
        ),
    )
    train = pd.DataFrame(
        {
            "age": [20.0, None, 40.0, 60.0],
            "income": [100.0, 120.0, 140.0, 10_000.0],
            "segment": ["A", "B", "A", "B"],
            "active": ["yes", "no", "yes", "no"],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "constant": [1, 1, 1, 1],
        }
    )
    transformed_train, transformed_target = pipeline.fit_transform(
        train, pd.Series([0, 0, 0, 1], name="target")
    )

    assert len(transformed_train) == len(transformed_target) == 4
    assert not transformed_train.isna().any().any()
    assert "constant" not in transformed_train
    assert {"segment_A", "segment_B"}.issubset(transformed_train.columns)
    assert "created_at_day" in transformed_train
    assert "created_at_year" not in transformed_train
    assert pipeline.get_model_options() == {"class_weight": {0: 2 / 3, 1: 2.0}}

    test = pd.DataFrame(
        {
            "age": [None, 1000.0],
            "income": [90.0, 20_000.0],
            "segment": ["C", "A"],
            "active": ["no", "yes"],
            "created_at": ["2027-02-01", "2027-02-02"],
            "constant": [1, 1],
        }
    )
    transformed_test = pipeline.transform(test)
    assert len(transformed_test) == 2
    assert tuple(transformed_test.columns) == pipeline.output_features
    assert transformed_test.loc[0, "segment_A"] == 0
    assert transformed_test.loc[0, "segment_B"] == 0

    storage = LocalFilesystemStorage(tmp_path / "artifacts")
    artifact = pipeline.persist_config(storage, "experiments/test/preprocessing.json")
    persisted = json.loads(storage.load(artifact))
    assert persisted["resolved_strategies"]["categorical_encoding"] == "one_hot"
    assert persisted["class_weighting"]["class_weights"] == {"0": 2 / 3, "1": 2.0}
    assert persisted["output_features"] == list(pipeline.output_features)


def test_pipeline_fits_statistics_only_on_training_data() -> None:
    spec = _spec(
        ["value"],
        [
            PreprocessingStepSpec(id="missing_values", strategy="median"),
            PreprocessingStepSpec(id="numeric_scaling", strategy="standard"),
        ],
    )
    pipeline = PipelineCompiler().compile(
        spec,
        column_types={"value": "numeric"},
        capabilities=ModelCapabilities(requires_scaling=True),
    )
    train = pd.DataFrame({"value": [0.0, None, 2.0]})
    pipeline.fit_transform(train, pd.Series([0, 1, 0]))
    transformed_test = pipeline.transform(pd.DataFrame({"value": [100.0, None]}))
    config = pipeline.get_config()
    scaler_config = next(
        item for item in config["feature_transformers"] if item["id"] == "numeric_scaling"
    )

    assert scaler_config["fitted"]["mean"] == [1.0]
    assert transformed_test.iloc[0, 0] > 100
    assert transformed_test.iloc[1, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("strategy", ["standard", "min_max", "robust"])
def test_numeric_scaler_supports_integer_columns(strategy: str) -> None:
    features = pd.DataFrame(
        {
            "quantity": [1, 2, 3],
            "order_year": [2022, 2023, 2024],
            "segment": ["A", "B", "A"],
        },
        index=[10, 20, 30],
    )
    scaler = NumericScaler(["quantity", "order_year"], strategy)

    scaler.fit(features)
    transformed = scaler.transform(features)

    assert transformed.index.tolist() == [10, 20, 30]
    assert transformed["segment"].tolist() == ["A", "B", "A"]
    assert transformed["quantity"].dtype == "float64"
    assert transformed["order_year"].dtype == "float64"
    assert features["quantity"].dtype == "int64"
    assert transformed["quantity"].tolist() != features["quantity"].tolist()


def test_compiler_rejects_raw_text_features_before_model_fit() -> None:
    spec = _spec(["message"], [])

    with pytest.raises(PipelineCompilationError, match="require text_vectorization:tfidf"):
        PipelineCompiler().compile(
            spec,
            column_types={"message": "text"},
            capabilities=ModelCapabilities(),
        )


def test_row_transformer_keeps_target_aligned_without_dropping_test_rows() -> None:
    spec = _spec(
        ["value"],
        [PreprocessingStepSpec(id="missing_values", strategy="drop_rows")],
    )
    pipeline = PipelineCompiler().compile(
        spec,
        column_types={"value": "numeric"},
        capabilities=ModelCapabilities(),
    )
    transformed, target = pipeline.fit_transform(
        pd.DataFrame({"value": [1.0, None, 3.0]}, index=[10, 11, 12]),
        pd.Series(["a", "b", "c"], index=[10, 11, 12]),
    )

    assert transformed.index.tolist() == [10, 12]
    assert target.to_dict() == {10: "a", 12: "c"}
    test = pipeline.transform(pd.DataFrame({"value": [None, 9.0]}))
    assert len(test) == 2
    assert test.iloc[0, 0] == 2.0


def test_compiler_enforces_model_capabilities_and_resolves_supported_fallbacks() -> None:
    categorical_passthrough = _spec(
        ["segment"],
        [PreprocessingStepSpec(id="categorical_encoding", strategy="passthrough")],
    )
    with pytest.raises(PipelineCompilationError, match="supports_categorical"):
        PipelineCompiler().compile(
            categorical_passthrough,
            column_types={"segment": "categorical"},
            capabilities=ModelCapabilities(),
        )

    class_weight = PipelineCompiler().compile(
        _spec(
            ["value"],
            [PreprocessingStepSpec(id="imbalance", strategy="class_weight")],
        ),
        column_types={"value": "numeric"},
        capabilities=ModelCapabilities(),
    )
    class_weight.fit_transform(
        pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}),
        pd.Series([0, 0, 0, 1]),
    )

    assert class_weight.requested_strategies["imbalance"] == "class_weight"
    assert class_weight.resolved_strategies["imbalance"] == "none"
    assert class_weight.get_model_options() == {}
    assert class_weight.get_config()["capability_adjustments"] == [
        {
            "component": "imbalance",
            "requested": "class_weight",
            "resolved": "none",
            "reason": "model_does_not_support_imbalance_weighting",
        }
    ]

    scaling = PipelineCompiler().compile(
        _spec(["value"], []),
        column_types={"value": "numeric"},
        capabilities=ModelCapabilities(requires_scaling=True),
    )
    assert scaling.requested_strategies["numeric_scaling"] == "passthrough"
    assert scaling.resolved_strategies["numeric_scaling"] == "standard"
    assert scaling.capability_adjustments == [
        {
            "component": "numeric_scaling",
            "requested": "passthrough",
            "resolved": "standard",
            "reason": "model_requires_scaling",
        }
    ]
