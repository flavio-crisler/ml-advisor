import numpy as np
import pandas as pd
import pytest

from app.domain.experiment.spec import TaskType
from app.ml.analytics.generator import ResultArtifactGenerator
from app.ml.registries.catalog import build_default_registry_catalog

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


def test_classification_artifacts_include_interactive_plots_and_shap() -> None:
    features = pd.DataFrame(
        {
            "first": np.linspace(-3.0, 3.0, 30),
            "second": np.tile([0.0, 1.0, 2.0], 10),
        }
    )
    target = pd.Series([0] * 15 + [1] * 15)
    model = build_default_registry_catalog().model_adapters.create(
        "logistic_regression", TaskType.BINARY_CLASSIFICATION
    )
    model.fit(features, target)

    artifact = ResultArtifactGenerator(max_explanation_rows=4).build_run(
        task=TaskType.BINARY_CLASSIFICATION,
        target=target.tolist(),
        predictions=model.predict(features).tolist(),
        probabilities=model.predict_proba(features).tolist(),
        probability_classes=model.get_classes(),
        positive_label=1,
        fitted_features=features,
        model=model,
        random_seed=41,
        learning_curve=[
            {"training_rows": 10, "validation_value": 0.7},
            {"training_rows": 20, "validation_value": 0.9},
        ],
    )

    assert artifact["schema_version"] == "1.0"
    assert set(artifact["plots"]) == {
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
        "probability_distribution",
        "learning_curve",
        "feature_importance",
        "shap_summary",
    }
    assert all("data" in plot and "layout" in plot for plot in artifact["plots"].values())
    assert artifact["explainability"]["status"] == "available"
    assert len(artifact["explainability"]["feature_importance"]) == 2
    assert (
        artifact["plots"]["feature_importance"]["data"] != artifact["plots"]["shap_summary"]["data"]
    )
    assert artifact["plots"]["shap_summary"]["data"][0]["type"] == "scatter"
    assert artifact["plots"]["confusion_matrix"]["data"][0]["texttemplate"] == "%{text}"
    assert artifact["plots"]["roc_curve"]["data"][-1]["meta"]["role"] == "baseline"


def test_regression_and_comparison_artifacts_are_plotly_serializable() -> None:
    features = pd.DataFrame({"experience": np.linspace(0.0, 10.0, 24)})
    target = 2.5 * features["experience"] + 4.0
    model = build_default_registry_catalog().model_adapters.create(
        "linear_regression", TaskType.REGRESSION
    )
    model.fit(features, target)
    generator = ResultArtifactGenerator(max_explanation_rows=4)

    artifact = generator.build_run(
        task=TaskType.REGRESSION,
        target=target.tolist(),
        predictions=model.predict(features).tolist(),
        probabilities=None,
        probability_classes=None,
        positive_label=None,
        fitted_features=features,
        model=model,
        random_seed=41,
        learning_curve=[],
    )
    comparison = generator.build_comparison(
        primary_metric="rmse",
        runs=[
            {"run_id": "run-1", "model_id": "linear", "metrics": {"rmse": 0.2}},
            {"run_id": "run-2", "model_id": "forest", "metrics": {"rmse": 0.4}},
        ],
    )

    assert set(artifact["plots"]) >= {
        "actual_vs_predicted",
        "residual_plot",
        "residual_distribution",
        "shap_summary",
    }
    assert comparison["primary_metric"] == "rmse"
    assert comparison["plot"]["data"][0]["x"] == ["linear", "forest"]


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
def test_shap_supports_every_classifier_adapter(model_id: str) -> None:
    features = pd.DataFrame(
        {
            "first": np.linspace(-2.0, 2.0, 24),
            "second": np.tile([0.0, 1.0, 2.0], 8),
        }
    )
    target = pd.Series([0] * 12 + [1] * 12)
    model = build_default_registry_catalog().model_adapters.create(
        model_id,
        TaskType.BINARY_CLASSIFICATION,
        CLASSIFIER_PARAMETERS.get(model_id),
    )
    model.fit(features, target)

    explanation = ResultArtifactGenerator(
        max_explanation_rows=2,
        max_background_rows=6,
    )._explain(
        task=TaskType.BINARY_CLASSIFICATION,
        features=features,
        model=model,
        random_seed=41,
    )

    assert explanation["status"] == "available", explanation


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
def test_shap_supports_every_regressor_adapter(model_id: str) -> None:
    features = pd.DataFrame(
        {
            "first": np.linspace(-2.0, 2.0, 24),
            "second": np.tile([0.0, 1.0, 2.0], 8),
        }
    )
    target = 3 * features["first"] - 0.5 * features["second"] + 2
    model = build_default_registry_catalog().model_adapters.create(
        model_id,
        TaskType.REGRESSION,
        REGRESSOR_PARAMETERS.get(model_id),
    )
    model.fit(features, target)

    explanation = ResultArtifactGenerator(
        max_explanation_rows=2,
        max_background_rows=6,
    )._explain(
        task=TaskType.REGRESSION,
        features=features,
        model=model,
        random_seed=41,
    )

    assert explanation["status"] == "available", explanation
