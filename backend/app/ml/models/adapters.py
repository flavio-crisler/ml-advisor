from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from app.ml.models.base import ModelAdapter
from app.ml.registries.metadata import ModelCapabilities

SearchSpaceBuilder = Callable[[Any], Mapping[str, Any]]


class EstimatorModelAdapter(ModelAdapter):
    """Adaptador genérico para estimadores compatíveis com a interface do Scikit-Learn.

    Attributes:
        estimator_type (type[BaseEstimator]): Classe do estimador subjacente.
        default_params (dict[str, Any]): Parâmetros padrão do estimador.
        capabilities (ModelCapabilities): Metadados de capacidades suportadas.
        search_space (SearchSpaceBuilder): Função geradora do espaço de busca Optuna.
        estimator (BaseEstimator): Instância ativa do estimador configurado.
    """

    def __init__(
        self,
        estimator_type: type[BaseEstimator],
        *,
        default_params: Mapping[str, Any],
        capabilities: ModelCapabilities,
        search_space: SearchSpaceBuilder,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Inicializa o adaptador instanciando o estimador com os parâmetros mesclados.

        Args:
            estimator_type (type[BaseEstimator]): Classe do estimador.
            default_params (Mapping[str, Any]): Dicionário de parâmetros padrão.
            capabilities (ModelCapabilities): Capacidades operacionais do modelo.
            search_space (SearchSpaceBuilder): Construtor de espaço de busca.
            parameters (Mapping[str, Any] | None): Parâmetros customizados opcionais.
        """
        self.estimator_type = estimator_type
        self.default_params = dict(default_params)
        self.capabilities = capabilities
        self.search_space = search_space
        self.estimator = estimator_type(**{**self.default_params, **(parameters or {})})

    def fit(
        self,
        features: Any,
        target: Any,
        *,
        fit_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Ajusta o estimador nos dados de treino com suporte a opções de fit.

        Args:
            features (Any): Matriz de features ou DataFrame.
            target (Any): Vetor ou Série com os valores alvo.
            fit_options (Mapping[str, Any] | None): Opções adicionais de treino (ex: class_weight).

        Raises:
            ValueError: Se forem passadas opções de ajuste não suportadas.
        """
        prepared = self._prepare_features(features, fitting=True)
        options = dict(fit_options or {})
        class_weight = options.pop("class_weight", None)
        fit_arguments: dict[str, Any] = {}
        if class_weight:
            fit_arguments.update(self._apply_class_weight(class_weight, target))
        if options:
            raise ValueError(f"Unsupported model fit options: {sorted(options)}")
        self.estimator.fit(prepared, target, **fit_arguments)

    def predict(self, features: Any) -> Any:
        """Gera predições para as features fornecidas.

        Args:
            features (Any): Dados de entrada a serem avaliados.

        Returns:
            Any: Predições geradas pelo estimador.
        """
        return self.estimator.predict(self._prepare_features(features, fitting=False))

    def predict_proba(self, features: Any) -> Any:
        """Calcula probabilidades estimadas caso o modelo dê suporte.

        Args:
            features (Any): Dados de entrada.

        Returns:
            Any: Matriz de probabilidades preditas.

        Raises:
            NotImplementedError: Se o estimador não suportar cálculo de probabilidades.
        """
        if not self.capabilities.supports_proba:
            return super().predict_proba(features)
        return self.estimator.predict_proba(self._prepare_features(features, fitting=False))

    def get_classes(self) -> tuple[Any, ...]:
        """Retorna os rótulos de classes conhecidos pelo modelo treinado.

        Returns:
            tuple[Any, ...]: Tupla contendo os rótulos de classe do estimador.
        """
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            return super().get_classes()
        return tuple(classes.tolist())

    def get_params(self) -> Mapping[str, Any]:
        """Retorna os parâmetros de primeiro nível configurados no estimador.

        Returns:
            Mapping[str, Any]: Dicionário de parâmetros do modelo.
        """
        return self.estimator.get_params(deep=False)

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Atualiza parâmetros e reinstancia o estimador subjacente.

        Args:
            params (Mapping[str, Any]): Novos hiperparâmetros a definir.
        """
        updated = {**self.estimator.get_params(deep=False), **dict(params)}
        self.estimator = self.estimator_type(**updated)

    def get_default_params(self) -> Mapping[str, Any]:
        """Retorna cópia dos parâmetros padrão do estimador.

        Returns:
            Mapping[str, Any]: Dicionário com os hiperparâmetros padrão.
        """
        return dict(self.default_params)

    def get_search_space(self, trial: Any) -> Mapping[str, Any]:
        """Gera e retorna o espaço de busca amostrado para o trial fornecido.

        Args:
            trial (Any): Instância do trial de otimização Optuna.

        Returns:
            Mapping[str, Any]: Dicionário de hiperparâmetros sugeridos.
        """
        return dict(self.search_space(trial))

    def get_capabilities(self) -> ModelCapabilities:
        """Retorna os metadados de capacidades do modelo.

        Returns:
            ModelCapabilities: Capacidades suportadas pelo estimador.
        """
        return self.capabilities

    def _prepare_features(self, features: Any, *, fitting: bool) -> Any:
        """Pré-processa ou formata as features antes de enviar ao estimador.

        Args:
            features (Any): Dados de entrada brutos.
            fitting (bool): True se chamado durante fit, False durante inferência.

        Returns:
            Any: Dados preparados para consumo pelo estimador.
        """
        del fitting
        return features

    def _apply_class_weight(
        self, class_weight: Mapping[Any, float], target: Any
    ) -> Mapping[str, Any]:
        """Aplica pesos de classes ao estimador ou retorna argumentos para fit.

        Args:
            class_weight (Mapping[Any, float]): Dicionário associando classes a pesos.
            target (Any): Valores alvo do conjunto de treino.

        Returns:
            Mapping[str, Any]: Argumentos extras para serem passados ao fit.

        Raises:
            ValueError: Se o estimador não suportar o parâmetro class_weight.
        """
        del target
        if "class_weight" not in self.estimator.get_params(deep=False):
            raise ValueError("This adapter cannot translate class_weight")
        self.estimator.set_params(class_weight=dict(class_weight))
        return {}


class XGBoostAdapter(EstimatorModelAdapter):
    """Adaptador para modelos XGBoost com codificação automática de rótulos de classe.

    Attributes:
        label_encoder (LabelEncoder): Codificador para classes categóricas no target.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Inicializa o adaptador XGBoost com encoder de rótulos.

        Args:
            *args (Any): Argumentos posicionais repassados para EstimatorModelAdapter.
            **kwargs (Any): Argumentos nomeados repassados para EstimatorModelAdapter.
        """
        super().__init__(*args, **kwargs)
        self.label_encoder = LabelEncoder()
        self._target_classes: tuple[Any, ...] = ()

    def fit(
        self,
        features: Any,
        target: Any,
        *,
        fit_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Ajusta o modelo XGBoost codificando o target para inteiros.

        Args:
            features (Any): Dados de entrada de features.
            target (Any): Valores alvo brutos.
            fit_options (Mapping[str, Any] | None): Opções como class_weight.

        Raises:
            ValueError: Se forem fornecidas opções de fit não suportadas.
        """
        prepared = self._prepare_features(features, fitting=True)
        options = dict(fit_options or {})
        class_weight = options.pop("class_weight", None)
        if options:
            raise ValueError(f"Unsupported model fit options: {sorted(options)}")
        original_target = np.asarray(target)
        encoded_target = self.label_encoder.fit_transform(original_target)
        self._target_classes = tuple(self.label_encoder.classes_.tolist())
        fit_arguments = {}
        if class_weight:
            fit_arguments["sample_weight"] = np.asarray(
                [class_weight[label] for label in original_target], dtype=float
            )
        self.estimator.fit(prepared, encoded_target, **fit_arguments)

    def predict(self, features: Any) -> Any:
        """Gera predições invertendo os inteiros preditos para os rótulos originais.

        Args:
            features (Any): Dados de entrada para inferência.

        Returns:
            Any: Classes preditas nos rótulos de domínio originais.
        """
        encoded = np.asarray(super().predict(features), dtype=int)
        return self.label_encoder.inverse_transform(encoded)

    def get_classes(self) -> tuple[Any, ...]:
        """Retorna as classes originais decodificadas do target.

        Returns:
            tuple[Any, ...]: Classes do modelo na ordem codificada.
        """
        if not self._target_classes:
            return super().get_classes()
        return self._target_classes

    def _apply_class_weight(
        self, class_weight: Mapping[Any, float], target: Any
    ) -> Mapping[str, Any]:
        """Converte pesos de classe em sample_weight por linha para o XGBoost.

        Args:
            class_weight (Mapping[Any, float]): Dicionário de pesos por classe.
            target (Any): Vetor alvo original.

        Returns:
            Mapping[str, Any]: Dicionário com sample_weight calculado.
        """
        return {"sample_weight": np.asarray([class_weight[label] for label in target], dtype=float)}


class CatBoostAdapter(EstimatorModelAdapter):
    """Adaptador para estimadores CatBoost com detecção automática de features categóricas.

    Attributes:
        categorical_columns (tuple[str, ...]): Nomes das colunas identificadas como categóricas.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Inicializa o adaptador CatBoost.

        Args:
            *args (Any): Argumentos repassados para a classe base.
            **kwargs (Any): Argumentos nomeados repassados para a classe base.
        """
        super().__init__(*args, **kwargs)
        self.categorical_columns: tuple[str, ...] = ()

    def fit(
        self,
        features: Any,
        target: Any,
        *,
        fit_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Ajusta o estimador CatBoost identificando features categóricas nativas.

        Args:
            features (Any): Dados de entrada de treino.
            target (Any): Valores alvo.
            fit_options (Mapping[str, Any] | None): Opções adicionais como class_weight.

        Raises:
            ValueError: Se houver opções de fit desconhecidas.
        """
        prepared = self._prepare_features(features, fitting=True)
        options = dict(fit_options or {})
        class_weight = options.pop("class_weight", None)
        if class_weight:
            self.estimator.set_params(class_weights=dict(class_weight))
        if options:
            raise ValueError(f"Unsupported model fit options: {sorted(options)}")
        fit_arguments = (
            {"cat_features": list(self.categorical_columns)} if self.categorical_columns else {}
        )
        self.estimator.fit(prepared, target, **fit_arguments)

    def predict(self, features: Any) -> Any:
        """Gera predições achatando arrays bi-dimensionais (n, 1) para 1D se necessário.

        Args:
            features (Any): Dados de entrada para inferência.

        Returns:
            Any: Array unidimensional de predições.
        """
        predictions = np.asarray(super().predict(features))
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions[:, 0]
        return predictions

    def _prepare_features(self, features: Any, *, fitting: bool) -> Any:
        """Formata colunas categóricas como strings conforme esperado pelo CatBoost.

        Args:
            features (Any): DataFrame de features.
            fitting (bool): True se for durante fit, False na predição.

        Returns:
            Any: DataFrame com colunas categóricas normalizadas como string.
        """
        if not isinstance(features, pd.DataFrame):
            return features
        if fitting:
            self.categorical_columns = tuple(
                column
                for column in features
                if pd.api.types.is_object_dtype(features[column])
                or isinstance(features[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(features[column])
            )
        prepared = features.copy()
        for column in self.categorical_columns:
            prepared[column] = prepared[column].astype(str)
        return prepared


class LightGBMAdapter(EstimatorModelAdapter):
    """Adaptador para estimadores LightGBM com suporte a tipos categóricos do pandas.

    Attributes:
        category_levels (dict[str, list[Any]]): Categorias observadas durante o ajuste.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Inicializa o adaptador LightGBM.

        Args:
            *args (Any): Argumentos posicionais para a classe base.
            **kwargs (Any): Argumentos nomeados para a classe base.
        """
        super().__init__(*args, **kwargs)
        self.category_levels: dict[str, list[Any]] = {}

    def _prepare_features(self, features: Any, *, fitting: bool) -> Any:
        """Converte colunas categóricas para o tipo pd.Categorical preservando os níveis.

        Args:
            features (Any): DataFrame de entrada.
            fitting (bool): Se verdadeiro, armazena os níveis únicos observados.

        Returns:
            Any: DataFrame com colunas categóricas ajustadas.
        """
        if not isinstance(features, pd.DataFrame):
            return features
        prepared = features.copy()
        if fitting:
            categorical_columns = [
                column
                for column in prepared
                if pd.api.types.is_object_dtype(prepared[column])
                or isinstance(prepared[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(prepared[column])
            ]
            self.category_levels = {
                column: list(pd.Series(prepared[column]).dropna().unique())
                for column in categorical_columns
            }
        for column, categories in self.category_levels.items():
            values = prepared[column]
            prepared[column] = pd.Categorical(
                values.where(values.isin(categories)), categories=categories
            )
        return prepared


class CalibratedSVMAdapter(ModelAdapter):
    """Adaptador para SVM com calibração de probabilidades via CalibratedClassifierCV.

    Attributes:
        default_params (dict[str, Any]): Parâmetros padrão do SVC.
        parameters (dict[str, Any]): Parâmetros ativos configurados.
        estimator (CalibratedClassifierCV | None): Estimador calibrado treinado.
    """

    def __init__(self, parameters: Mapping[str, Any] | None = None) -> None:
        """Inicializa o adaptador calibrado para SVM.

        Args:
            parameters (Mapping[str, Any] | None): Hiperparâmetros customizados para o SVC.
        """
        self.default_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        self.parameters = {**self.default_params, **(parameters or {})}
        self.estimator: CalibratedClassifierCV | None = None

    def fit(
        self,
        features: Any,
        target: Any,
        *,
        fit_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Ajusta o classificador SVC com calibração sigmoide em validação cruzada 3-fold.

        Args:
            features (Any): Features de treino.
            target (Any): Classes alvo.
            fit_options (Mapping[str, Any] | None): Opções adicionais como class_weight.

        Raises:
            ValueError: Se houver opções de fit não reconhecidas.
        """
        options = dict(fit_options or {})
        class_weight = options.pop("class_weight", None)
        if options:
            raise ValueError(f"Unsupported model fit options: {sorted(options)}")
        sample_weight = (
            np.asarray([class_weight[label] for label in target], dtype=float)
            if class_weight
            else None
        )
        classifier = SVC(**self.parameters)
        self.estimator = CalibratedClassifierCV(
            classifier, method="sigmoid", cv=3, ensemble=False
        ).fit(features, target, sample_weight=sample_weight)

    def predict(self, features: Any) -> Any:
        """Gera predições de classes usando o estimador calibrado.

        Args:
            features (Any): Dados de entrada.

        Returns:
            Any: Rótulos de classes preditas.
        """
        return self._fitted().predict(features)

    def predict_proba(self, features: Any) -> Any:
        """Calcula probabilidades calibradas para cada classe.

        Args:
            features (Any): Dados de entrada.

        Returns:
            Any: Matriz de probabilidades calibradas.
        """
        return self._fitted().predict_proba(features)

    def get_classes(self) -> tuple[Any, ...]:
        """Retorna os rótulos de classe conhecidos pelo estimador calibrado.

        Returns:
            tuple[Any, ...]: Classes do modelo.
        """
        return tuple(self._fitted().classes_.tolist())

    def get_params(self) -> Mapping[str, Any]:
        """Retorna os parâmetros configurados para o estimador SVC base.

        Returns:
            Mapping[str, Any]: Dicionário de hiperparâmetros.
        """
        return dict(self.parameters)

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Atualiza os parâmetros do SVC e reseta o estimador ajustado.

        Args:
            params (Mapping[str, Any]): Novos hiperparâmetros a definir.
        """
        self.parameters.update(params)
        self.estimator = None

    def get_default_params(self) -> Mapping[str, Any]:
        """Retorna os hiperparâmetros padrão do SVC calibrado.

        Returns:
            Mapping[str, Any]: Hiperparâmetros padrão.
        """
        return dict(self.default_params)

    def get_search_space(self, trial: Any) -> Mapping[str, Any]:
        """Gera o espaço de busca Optuna para hiperparâmetros de SVM.

        Args:
            trial (Any): Trial do Optuna.

        Returns:
            Mapping[str, Any]: Hiperparâmetros sugeridos pelo trial.
        """
        return _svm_search_space(trial)

    def get_capabilities(self) -> ModelCapabilities:
        """Retorna capacidades suportadas pelo classificador SVM calibrado.

        Returns:
            ModelCapabilities: Metadados de capacidades da SVM.
        """
        return SVM_CAPABILITIES

    def _fitted(self) -> CalibratedClassifierCV:
        """Retorna o estimador calibrado garantindo que foi previamente treinado.

        Returns:
            CalibratedClassifierCV: Instância treinada do estimador.

        Raises:
            RuntimeError: Se o modelo ainda não tiver sido treinado via fit().
        """
        if self.estimator is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.estimator


CLASSIFICATION_CAPABILITIES = ModelCapabilities(
    supports_proba=True,
    supports_imbalance_weighting=True,
)
TREE_CAPABILITIES = ModelCapabilities(
    supports_proba=True,
    supports_imbalance_weighting=True,
)
BOOSTED_CAPABILITIES = ModelCapabilities(
    supports_proba=True,
    supports_imbalance_weighting=True,
    supports_gpu=True,
)
NATIVE_BOOSTED_CAPABILITIES = ModelCapabilities(
    supports_proba=True,
    supports_imbalance_weighting=True,
    supports_categorical=True,
    supports_gpu=True,
)
SCALING_CLASSIFIER_CAPABILITIES = ModelCapabilities(
    supports_proba=True,
    requires_scaling=True,
)
SVM_CAPABILITIES = ModelCapabilities(
    supports_proba=True,
    supports_imbalance_weighting=True,
    requires_scaling=True,
)
REGRESSION_CAPABILITIES = ModelCapabilities()
SCALING_REGRESSION_CAPABILITIES = ModelCapabilities(requires_scaling=True)
NATIVE_REGRESSION_CAPABILITIES = ModelCapabilities(
    supports_categorical=True,
    supports_gpu=True,
)
BOOSTED_REGRESSION_CAPABILITIES = ModelCapabilities(supports_gpu=True)


def logistic_regression(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para LogisticRegression com escalonamento obrigatório.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o estimador.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para regressão logística.
    """
    return EstimatorModelAdapter(
        LogisticRegression,
        default_params={"C": 1.0, "max_iter": 1000, "random_state": 41},
        capabilities=ModelCapabilities(
            supports_proba=True,
            supports_imbalance_weighting=True,
            requires_scaling=True,
        ),
        search_space=_logistic_search_space,
        parameters=parameters,
    )


def random_forest_classifier(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para RandomForestClassifier.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para Random Forest de classificação.
    """
    return EstimatorModelAdapter(
        RandomForestClassifier,
        default_params={"n_estimators": 100, "random_state": 41, "n_jobs": 1},
        capabilities=TREE_CAPABILITIES,
        search_space=_random_forest_search_space,
        parameters=parameters,
    )


def xgboost_classifier(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador XGBoostAdapter para XGBClassifier.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de XGBoostAdapter para classificação.
    """
    return XGBoostAdapter(
        XGBClassifier,
        default_params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 41,
            "n_jobs": 1,
            "tree_method": "hist",
        },
        capabilities=BOOSTED_CAPABILITIES,
        search_space=_boosted_search_space,
        parameters=parameters,
    )


def catboost_classifier(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador CatBoostAdapter para CatBoostClassifier.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de CatBoostAdapter para classificação.
    """
    return CatBoostAdapter(
        CatBoostClassifier,
        default_params={
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.05,
            "random_seed": 41,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
        },
        capabilities=NATIVE_BOOSTED_CAPABILITIES,
        search_space=_catboost_search_space,
        parameters=parameters,
    )


def lightgbm_classifier(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador LightGBMAdapter para LGBMClassifier.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de LightGBMAdapter para classificação.
    """
    return LightGBMAdapter(
        LGBMClassifier,
        default_params={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 41,
            "n_jobs": 1,
            "verbosity": -1,
        },
        capabilities=NATIVE_BOOSTED_CAPABILITIES,
        search_space=_lightgbm_search_space,
        parameters=parameters,
    )


def knn_classifier(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para KNeighborsClassifier.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para KNN classificador.
    """
    return EstimatorModelAdapter(
        KNeighborsClassifier,
        default_params={"n_neighbors": 5, "weights": "uniform", "n_jobs": 1},
        capabilities=SCALING_CLASSIFIER_CAPABILITIES,
        search_space=_knn_search_space,
        parameters=parameters,
    )


def svm_classifier(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador CalibratedSVMAdapter para classificação.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o SVC.

    Returns:
        ModelAdapter: Instância de CalibratedSVMAdapter.
    """
    return CalibratedSVMAdapter(parameters)


def linear_regression(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para LinearRegression.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para regressão linear.
    """
    return EstimatorModelAdapter(
        LinearRegression,
        default_params={"n_jobs": 1},
        capabilities=SCALING_REGRESSION_CAPABILITIES,
        search_space=_empty_search_space,
        parameters=parameters,
    )


def random_forest_regressor(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para RandomForestRegressor.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para Random Forest de regressão.
    """
    return EstimatorModelAdapter(
        RandomForestRegressor,
        default_params={"n_estimators": 100, "random_state": 41, "n_jobs": 1},
        capabilities=REGRESSION_CAPABILITIES,
        search_space=_random_forest_search_space,
        parameters=parameters,
    )


def xgboost_regressor(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para XGBRegressor.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para XGBoost de regressão.
    """
    return EstimatorModelAdapter(
        XGBRegressor,
        default_params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 41,
            "n_jobs": 1,
            "tree_method": "hist",
        },
        capabilities=BOOSTED_REGRESSION_CAPABILITIES,
        search_space=_boosted_search_space,
        parameters=parameters,
    )


def catboost_regressor(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador CatBoostAdapter para CatBoostRegressor.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de CatBoostAdapter para regressão.
    """
    return CatBoostAdapter(
        CatBoostRegressor,
        default_params={
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.05,
            "random_seed": 41,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
        },
        capabilities=NATIVE_REGRESSION_CAPABILITIES,
        search_space=_catboost_search_space,
        parameters=parameters,
    )


def lightgbm_regressor(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador LightGBMAdapter para LGBMRegressor.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de LightGBMAdapter para regressão.
    """
    return LightGBMAdapter(
        LGBMRegressor,
        default_params={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 41,
            "n_jobs": 1,
            "verbosity": -1,
        },
        capabilities=NATIVE_REGRESSION_CAPABILITIES,
        search_space=_lightgbm_search_space,
        parameters=parameters,
    )


def knn_regressor(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para KNeighborsRegressor.

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o modelo.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para KNN regressor.
    """
    return EstimatorModelAdapter(
        KNeighborsRegressor,
        default_params={"n_neighbors": 5, "weights": "uniform", "n_jobs": 1},
        capabilities=SCALING_REGRESSION_CAPABILITIES,
        search_space=_knn_search_space,
        parameters=parameters,
    )


def svr(parameters: Mapping[str, Any] | None = None) -> ModelAdapter:
    """Cria e configura um adaptador para SVR (Support Vector Regressor).

    Args:
        parameters (Mapping[str, Any] | None): Parâmetros customizados para o estimador SVR.

    Returns:
        ModelAdapter: Instância de EstimatorModelAdapter para SVR.
    """
    return EstimatorModelAdapter(
        SVR,
        default_params={"C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
        capabilities=SCALING_REGRESSION_CAPABILITIES,
        search_space=_svr_search_space,
        parameters=parameters,
    )


def _boosted_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para modelos baseados em gradiente boosted.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Parâmetros amostrados como n_estimators, max_depth, learning_rate.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def _empty_search_space(trial: Any) -> Mapping[str, Any]:
    """Define um espaço de busca vazio para estimadores sem hiperparâmetros a ajustar.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Dicionário vazio.
    """
    del trial
    return {}


def _logistic_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de regularização para a regressão logística.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Dicionário contendo o parâmetro 'C'.
    """
    return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}


def _random_forest_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para florestas aleatórias (Random Forest).

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Dicionário contendo n_estimators, max_depth, min_samples_split.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }


def _catboost_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para estimadores CatBoost.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Parâmetros amostrados (iterations, depth, learning_rate, l2_leaf_reg).
    """
    return {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }


def _lightgbm_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para estimadores LightGBM.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Parâmetros amostrados (num_leaves, n_estimators, min_child_samples).
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }


def _knn_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para o algoritmo K-Nearest Neighbors.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Parâmetros amostrados (n_neighbors, weights, p).
    """
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    }


def _svm_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para classificadores Support Vector Machine.

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Parâmetros amostrados (C, kernel, gamma).
    """
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _svr_search_space(trial: Any) -> Mapping[str, Any]:
    """Define o espaço de busca de hiperparâmetros para Support Vector Regression (SVR).

    Args:
        trial (Any): Trial do Optuna.

    Returns:
        Mapping[str, Any]: Parâmetros amostrados combinando SVM básico com epsilon.
    """
    return {
        **_svm_search_space(trial),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
    }
