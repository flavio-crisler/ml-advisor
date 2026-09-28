from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.domain.exceptions import ModelTaskIncompatibleError
from app.domain.experiment.spec import TaskType
from app.ml.registries.catalog import build_default_registry_catalog


class FakeTrial:
    def suggest_int(self, name: str, low: int, high: int) -> int:
        del name, high
        return low

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        del name, high, log
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        del name
        return choices[0]


CLASSIFIER_PARAMETERS = {
    "random_forest_classifier": {"n_estimators": 10},
    "xgboost_classifier": {"n_estimators": 10, "max_depth": 2},
    "catboost_classifier": {"iterations": 10, "depth": 2},
    "lightgbm_classifier": {"n_estimators": 10, "num_leaves": 4},
    "knn_classifier": {"n_neighbors": 3},
}
REGRESSOR_PARAMETERS = {
    "random_forest_regressor": {"n_estimators": 10},
    "xgboost_regressor": {"n_estimators": 10, "max_depth": 2},
    "catboost_regressor": {"iterations": 10, "depth": 2},
    "lightgbm_regressor": {"n_estimators": 10, "num_leaves": 4},
    "knn_regressor": {"n_neighbors": 3},
}
FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "model_id",
    [
        "logistic_regression",
        "random_forest_classifier",
        "xgboost_classifier",
        "catboost_classifier",
        "lightgbm_classifier",
        "knn_classifier",
        "svm_classifier",
    ],
)
def test_classifier_adapters_share_fit_predict_and_probability_contract(model_id: str) -> None:
    catalog = build_default_registry_catalog()
    adapter = catalog.model_adapters.create(
        model_id,
        TaskType.BINARY_CLASSIFICATION,
        CLASSIFIER_PARAMETERS.get(model_id),
    )
    features = pd.DataFrame(
        {
            "first": np.linspace(-2.0, 2.0, 24),
            "second": np.tile([0.0, 1.0, 2.0], 8),
        }
    )
    target = pd.Series([0] * 12 + [1] * 12)
    fit_options = (
        {"class_weight": {0: 1.0, 1: 1.5}}
        if adapter.get_capabilities().supports_imbalance_weighting
        else None
    )

    adapter.fit(features, target, fit_options=fit_options)
    predictions = adapter.predict(features)
    probabilities = adapter.predict_proba(features)

    assert len(predictions) == len(features)
    assert probabilities.shape == (len(features), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert adapter.get_capabilities() == catalog.models.get(model_id).capabilities
    assert set(adapter.get_default_params()).issubset(adapter.get_params())
    assert isinstance(adapter.get_search_space(FakeTrial()), dict)


def test_calibrated_svm_accepts_class_weights_for_string_targets() -> None:
    adapter = build_default_registry_catalog().model_adapters.create(
        "svm_classifier", TaskType.BINARY_CLASSIFICATION
    )
    features = pd.DataFrame({"value": np.linspace(-2.0, 2.0, 30)})
    target = pd.Series(["no"] * 15 + ["yes"] * 15)

    adapter.fit(
        features,
        target,
        fit_options={"class_weight": {"no": 1.0, "yes": 2.0}},
    )

    assert set(adapter.predict(features)).issubset({"no", "yes"})
    assert adapter.get_classes() == ("no", "yes")
    assert np.allclose(adapter.predict_proba(features).sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "model_id",
    [
        "linear_regression",
        "random_forest_regressor",
        "xgboost_regressor",
        "catboost_regressor",
        "lightgbm_regressor",
        "knn_regressor",
        "svr",
    ],
)
def test_regressor_adapters_share_fit_predict_and_parameter_contract(model_id: str) -> None:
    catalog = build_default_registry_catalog()
    adapter = catalog.model_adapters.create(
        model_id,
        TaskType.REGRESSION,
        REGRESSOR_PARAMETERS.get(model_id),
    )
    features = pd.DataFrame(
        {
            "first": np.linspace(-2.0, 2.0, 24),
            "second": np.tile([0.0, 1.0, 2.0], 8),
        }
    )
    target = 3 * features["first"] - 0.5 * features["second"] + 2

    adapter.fit(features, target)
    predictions = adapter.predict(features)

    assert len(predictions) == len(features)
    assert np.isfinite(predictions).all()
    assert adapter.get_capabilities() == catalog.models.get(model_id).capabilities
    defaults = adapter.get_default_params()
    assert set(defaults).issubset(adapter.get_params())
    adapter.set_params({key: value for key, value in list(defaults.items())[:1]})
    assert isinstance(adapter.get_search_space(FakeTrial()), dict)
    with pytest.raises(NotImplementedError):
        adapter.predict_proba(features)


@pytest.mark.parametrize("model_id", ["catboost_classifier", "lightgbm_classifier"])
def test_native_categorical_adapters_handle_unseen_values(model_id: str) -> None:
    adapter = build_default_registry_catalog().model_adapters.create(
        model_id,
        TaskType.BINARY_CLASSIFICATION,
        {"iterations": 10, "depth": 2}
        if model_id == "catboost_classifier"
        else {"n_estimators": 10, "num_leaves": 4},
    )
    train = pd.DataFrame(
        {
            "value": np.arange(12, dtype=float),
            "segment": ["A", "B"] * 6,
        }
    )
    target = pd.Series([0, 1] * 6)

    adapter.fit(train, target)
    predictions = adapter.predict(pd.DataFrame({"value": [2.0], "segment": ["C"]}))

    assert predictions.shape == (1,)


def test_model_registry_rejects_incompatible_task_and_exposes_all_adapters() -> None:
    catalog = build_default_registry_catalog()

    assert len(catalog.models.all()) == len(catalog.model_adapters.all()) == 14
    with pytest.raises(ModelTaskIncompatibleError, match="does not support task regression"):
        catalog.model_adapters.create("logistic_regression", TaskType.REGRESSION)


@pytest.mark.parametrize(
    "model_id",
    [
        "logistic_regression",
        "random_forest_classifier",
        "xgboost_classifier",
        "catboost_classifier",
        "lightgbm_classifier",
        "knn_classifier",
        "svm_classifier",
    ],
)
def test_every_classifier_trains_on_kaggle_iris_data(model_id: str) -> None:
    frame = pd.read_csv(FIXTURE_DIRECTORY / "kaggle_iris.csv")
    features = frame.loc[
        :99,
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    ]
    target = frame.loc[:99, "Species"]
    adapter = build_default_registry_catalog().model_adapters.create(
        model_id,
        TaskType.BINARY_CLASSIFICATION,
        CLASSIFIER_PARAMETERS.get(model_id),
    )

    adapter.fit(features, target)
    predictions = adapter.predict(features)

    assert len(predictions) == 100
    assert set(predictions).issubset(set(target))


@pytest.mark.parametrize("numeric_labels", [False, True])
def test_catboost_multiclass_predictions_are_scalar_labels(numeric_labels: bool) -> None:
    frame = pd.read_csv(FIXTURE_DIRECTORY / "kaggle_iris.csv")
    features = frame[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    target = frame["Species"]
    if numeric_labels:
        target = target.map({label: index for index, label in enumerate(target.unique())})
    adapter = build_default_registry_catalog().model_adapters.create(
        "catboost_classifier",
        TaskType.MULTICLASS_CLASSIFICATION,
        CLASSIFIER_PARAMETERS["catboost_classifier"],
    )

    adapter.fit(features, target)
    predictions = adapter.predict(features)

    assert predictions.shape == (len(frame),)
    assert set(predictions).issubset(set(target))


@pytest.mark.parametrize(
    "model_id",
    [
        "linear_regression",
        "random_forest_regressor",
        "xgboost_regressor",
        "catboost_regressor",
        "lightgbm_regressor",
        "knn_regressor",
        "svr",
    ],
)
def test_every_regressor_trains_on_kaggle_salary_data(model_id: str) -> None:
    frame = pd.read_csv(FIXTURE_DIRECTORY / "kaggle_salary.csv")
    features = frame.loc[:, ["YearsExperience"]]
    target = frame["Salary"]
    adapter = build_default_registry_catalog().model_adapters.create(
        model_id,
        TaskType.REGRESSION,
        REGRESSOR_PARAMETERS.get(model_id),
    )

    adapter.fit(features, target)
    predictions = adapter.predict(features)

    assert len(predictions) == len(frame)
    assert np.isfinite(predictions).all()
