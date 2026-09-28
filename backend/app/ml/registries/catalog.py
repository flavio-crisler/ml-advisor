from dataclasses import dataclass

from app.domain.experiment.spec import AveragingStrategy, RocAucStrategy, TaskType
from app.ml.models.registry import ModelAdapterRegistry, build_default_model_adapter_registry
from app.ml.registries.metadata import (
    MetricMetadata,
    ModelCapabilities,
    ModelMetadata,
    PreprocessorMetadata,
    TunerMetadata,
    ValidationMetadata,
)
from app.ml.registries.registry import Registry
from app.ml.tuning.registry import TunerAdapterRegistry, build_default_tuner_adapter_registry

CLASSIFICATION_TASKS = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
)
ALL_TASKS = frozenset({*CLASSIFICATION_TASKS, TaskType.REGRESSION})


@dataclass(frozen=True, slots=True)
class RegistryCatalog:
    """Agrupa todos os catálogos e registros de componentes de Machine Learning da aplicação.

    Attributes:
        models (Registry[ModelMetadata]): Metadados de modelos cadastrados.
        model_adapters (ModelAdapterRegistry): Fábricas de adaptadores de modelos.
        metrics (Registry[MetricMetadata]): Metadados de métricas de avaliação.
        preprocessors (Registry[PreprocessorMetadata]): Metadados de transformadores.
        validators (Registry[ValidationMetadata]): Metadados de estratégias de validação.
        tuners (Registry[TunerMetadata]): Metadados de motores de tuning.
        tuner_adapters (TunerAdapterRegistry): Fábricas de adaptadores de tuners.
    """

    models: Registry[ModelMetadata]
    model_adapters: ModelAdapterRegistry
    metrics: Registry[MetricMetadata]
    preprocessors: Registry[PreprocessorMetadata]
    validators: Registry[ValidationMetadata]
    tuners: Registry[TunerMetadata]
    tuner_adapters: TunerAdapterRegistry


def build_default_registry_catalog() -> RegistryCatalog:
    """Constrói e inicializa o catálogo padrão com todos os componentes pré-registrados.

    Returns:
        RegistryCatalog: Instância consolidada contendo todos os registros de ML.
    """
    models = Registry(_model_metadata())
    return RegistryCatalog(
        models=models,
        model_adapters=build_default_model_adapter_registry(models.all()),
        metrics=Registry(_metric_metadata()),
        preprocessors=Registry(_preprocessor_metadata()),
        validators=Registry(_validation_metadata()),
        tuners=Registry([TunerMetadata(id="optuna", display_name="Optuna", supports_pruning=True)]),
        tuner_adapters=build_default_tuner_adapter_registry(),
    )


def _model_metadata() -> list[ModelMetadata]:
    """Gera a lista de metadados padrão para todos os algoritmos de modelos suportados.

    Returns:
        list[ModelMetadata]: Lista de metadados de modelos (classificadores e regressores).
    """
    definitions = [
        (
            "logistic_regression",
            "Logistic Regression",
            CLASSIFICATION_TASKS,
            True,
            True,
            False,
            False,
            True,
        ),
        (
            "random_forest_classifier",
            "Random Forest Classifier",
            CLASSIFICATION_TASKS,
            True,
            True,
            False,
            False,
            False,
        ),
        (
            "xgboost_classifier",
            "XGBoost Classifier",
            CLASSIFICATION_TASKS,
            True,
            True,
            False,
            True,
            False,
        ),
        (
            "catboost_classifier",
            "CatBoost Classifier",
            CLASSIFICATION_TASKS,
            True,
            True,
            True,
            True,
            False,
        ),
        (
            "lightgbm_classifier",
            "LightGBM Classifier",
            CLASSIFICATION_TASKS,
            True,
            True,
            True,
            True,
            False,
        ),
        ("knn_classifier", "KNN Classifier", CLASSIFICATION_TASKS, True, False, False, False, True),
        ("svm_classifier", "SVM Classifier", CLASSIFICATION_TASKS, True, True, False, False, True),
        (
            "linear_regression",
            "Linear Regression",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            False,
            False,
            True,
        ),
        (
            "random_forest_regressor",
            "Random Forest Regressor",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "xgboost_regressor",
            "XGBoost Regressor",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            False,
            True,
            False,
        ),
        (
            "catboost_regressor",
            "CatBoost Regressor",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            True,
            True,
            False,
        ),
        (
            "lightgbm_regressor",
            "LightGBM Regressor",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            True,
            True,
            False,
        ),
        (
            "knn_regressor",
            "KNN Regressor",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            False,
            False,
            True,
        ),
        (
            "svr",
            "Support Vector Regressor",
            frozenset({TaskType.REGRESSION}),
            False,
            False,
            False,
            False,
            True,
        ),
    ]
    return [
        ModelMetadata(
            id=model_id,
            display_name=display_name,
            tasks=tasks,
            capabilities=ModelCapabilities(
                supports_proba=supports_proba,
                supports_imbalance_weighting=supports_imbalance_weighting,
                supports_categorical=supports_categorical,
                supports_gpu=supports_gpu,
                requires_scaling=requires_scaling,
            ),
        )
        for (
            model_id,
            display_name,
            tasks,
            supports_proba,
            supports_imbalance_weighting,
            supports_categorical,
            supports_gpu,
            requires_scaling,
        ) in definitions
    ]


def _metric_metadata() -> list[MetricMetadata]:
    """Gera a lista de metadados padrão para todas as métricas de avaliação do catálogo.

    Returns:
        list[MetricMetadata]: Lista de definições de métricas para classificação e regressão.
    """
    averaging = (AveragingStrategy.MACRO, AveragingStrategy.WEIGHTED)
    return [
        MetricMetadata(
            id="accuracy",
            display_name="Accuracy",
            tasks=CLASSIFICATION_TASKS,
            direction="maximize",
        ),
        MetricMetadata(
            id="precision",
            display_name="Precision",
            tasks=CLASSIFICATION_TASKS,
            direction="maximize",
            supported_averaging=averaging,
            default_averaging=AveragingStrategy.MACRO,
        ),
        MetricMetadata(
            id="recall",
            display_name="Recall",
            tasks=CLASSIFICATION_TASKS,
            direction="maximize",
            supported_averaging=averaging,
            default_averaging=AveragingStrategy.MACRO,
        ),
        MetricMetadata(
            id="f1",
            display_name="F1",
            tasks=CLASSIFICATION_TASKS,
            direction="maximize",
            supported_averaging=averaging,
            default_averaging=AveragingStrategy.MACRO,
        ),
        MetricMetadata(
            id="roc_auc",
            display_name="ROC-AUC",
            tasks=CLASSIFICATION_TASKS,
            direction="maximize",
            requires_proba=True,
            roc_auc_strategies=(RocAucStrategy.OVR, RocAucStrategy.OVO),
        ),
        MetricMetadata(
            id="pr_auc",
            display_name="PR-AUC",
            tasks=CLASSIFICATION_TASKS,
            direction="maximize",
            requires_proba=True,
        ),
        MetricMetadata(
            id="log_loss",
            display_name="Log Loss",
            tasks=CLASSIFICATION_TASKS,
            direction="minimize",
            requires_proba=True,
        ),
        MetricMetadata(
            id="mae",
            display_name="MAE",
            tasks=frozenset({TaskType.REGRESSION}),
            direction="minimize",
        ),
        MetricMetadata(
            id="mse",
            display_name="MSE",
            tasks=frozenset({TaskType.REGRESSION}),
            direction="minimize",
        ),
        MetricMetadata(
            id="rmse",
            display_name="RMSE",
            tasks=frozenset({TaskType.REGRESSION}),
            direction="minimize",
        ),
        MetricMetadata(
            id="r2",
            display_name="R²",
            tasks=frozenset({TaskType.REGRESSION}),
            direction="maximize",
        ),
        MetricMetadata(
            id="mape",
            display_name="MAPE",
            tasks=frozenset({TaskType.REGRESSION}),
            direction="minimize",
        ),
    ]


def _preprocessor_metadata() -> list[PreprocessorMetadata]:
    """Gera a lista de metadados padrão para todas as etapas de pré-processamento.

    Returns:
        list[PreprocessorMetadata]: Lista de metadados de transformadores e estratégias.
    """
    return [
        PreprocessorMetadata(
            id="text_vectorization",
            display_name="Text vectorization (TF-IDF)",
            category="text",
            transformer_kind="feature",
            strategies=("tfidf",),
        ),
        PreprocessorMetadata(
            id="missing_values",
            display_name="Missing values",
            category="missing",
            transformer_kind="composite",
            strategies=("drop_rows", "drop_columns", "mean", "median", "mode", "constant"),
        ),
        PreprocessorMetadata(
            id="categorical_encoding",
            display_name="Categorical encoding",
            category="categorical",
            transformer_kind="feature",
            strategies=("one_hot", "ordinal", "passthrough"),
        ),
        PreprocessorMetadata(
            id="boolean_encoding",
            display_name="Boolean encoding",
            category="boolean",
            transformer_kind="feature",
            strategies=("binary", "passthrough"),
        ),
        PreprocessorMetadata(
            id="numeric_scaling",
            display_name="Numeric scaling",
            category="numeric",
            transformer_kind="feature",
            strategies=("standard", "min_max", "robust", "passthrough"),
        ),
        PreprocessorMetadata(
            id="datetime",
            display_name="Date and time",
            category="datetime",
            transformer_kind="feature",
            strategies=("components", "ordinal", "drop"),
        ),
        PreprocessorMetadata(
            id="outliers",
            display_name="Outliers",
            category="outliers",
            transformer_kind="feature",
            strategies=("none", "quantile_clip", "iqr"),
        ),
        PreprocessorMetadata(
            id="imbalance",
            display_name="Class imbalance",
            category="imbalance",
            transformer_kind="feature",
            strategies=("none", "class_weight"),
        ),
        PreprocessorMetadata(
            id="feature_selection",
            display_name="Feature selection",
            category="feature_selection",
            transformer_kind="feature",
            strategies=("none", "variance_threshold"),
        ),
    ]


def _validation_metadata() -> list[ValidationMetadata]:
    """Gera a lista de metadados padrão para as estratégias de validação suportadas.

    Returns:
        list[ValidationMetadata]: Lista de estratégias de validação (splits e K-fold).
    """
    return [
        ValidationMetadata(
            id="train_test_split",
            display_name="Train/test split",
            tasks=ALL_TASKS,
            parameters=("test_size", "shuffle", "random_seed"),
        ),
        ValidationMetadata(
            id="stratified_kfold",
            display_name="Stratified K-Fold",
            tasks=CLASSIFICATION_TASKS,
            parameters=("folds", "shuffle", "random_seed"),
        ),
        ValidationMetadata(
            id="kfold",
            display_name="K-Fold",
            tasks=frozenset({TaskType.REGRESSION}),
            parameters=("folds", "shuffle", "random_seed"),
        ),
    ]
