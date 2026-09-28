from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from app.ml.registries.metadata import ModelCapabilities


class ModelAdapter(ABC):
    """Define a interface base unificada para adaptadores de modelos de Machine Learning.

    Isola os algoritmos (Scikit-Learn, XGBoost, LightGBM, CatBoost) das camadas
    superiores de treinamento e orquestração.
    """

    @abstractmethod
    def fit(
        self,
        features: Any,
        target: Any,
        *,
        fit_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Treina o modelo nos dados fornecidos sem vazar detalhes da biblioteca.

        Args:
            features (Any): Conjunto de dados de entrada/features.
            target (Any): Valores alvo correspondentes (classes ou valores contínuos).
            fit_options (Mapping[str, Any] | None): Opções adicionais de ajuste (ex: class_weight).
        """

    @abstractmethod
    def predict(self, features: Any) -> Any:
        """Gera predições pontuais para as features informadas.

        Args:
            features (Any): Dados de entrada a serem inferidos.

        Returns:
            Any: Predições geradas pelo modelo (classes ou valores contínuos).
        """

    def predict_proba(self, features: Any) -> Any:
        """Gera probabilidades estimadas para cada classe.

        Args:
            features (Any): Dados de entrada.

        Returns:
            Any: Matriz ou array de probabilidades preditas.

        Raises:
            NotImplementedError: Se o modelo não suportar cálculo de probabilidades.
        """
        raise NotImplementedError("This model does not support probability predictions")

    def get_classes(self) -> tuple[Any, ...]:
        """Retorna os rótulos de classes conhecidos pelo modelo treinado.

        Returns:
            tuple[Any, ...]: Rótulos das classes na ordem de predição.

        Raises:
            NotImplementedError: Se o modelo não for de classificação.
        """
        raise NotImplementedError("This model does not expose classification labels")

    @abstractmethod
    def get_params(self) -> Mapping[str, Any]:
        """Retorna os parâmetros e hiperparâmetros atuais configurados no modelo.

        Returns:
            Mapping[str, Any]: Dicionário de hiperparâmetros atuais.
        """

    @abstractmethod
    def set_params(self, params: Mapping[str, Any]) -> None:
        """Atualiza hiperparâmetros no estimador subjacente.

        Args:
            params (Mapping[str, Any]): Novos parâmetros a aplicar.
        """

    @abstractmethod
    def get_default_params(self) -> Mapping[str, Any]:
        """Retorna a parametrização padrão do modelo.

        Returns:
            Mapping[str, Any]: Hiperparâmetros padrão recomendados.
        """

    @abstractmethod
    def get_search_space(self, trial: Any) -> Mapping[str, Any]:
        """Amostra hiperparâmetros a partir de um trial de otimização (ex: Optuna).

        Args:
            trial (Any): Instância do trial de otimização.

        Returns:
            Mapping[str, Any]: Dicionário de hiperparâmetros sugeridos para o trial.
        """

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Retorna as capacidades suportadas por este modelo.

        Returns:
            ModelCapabilities: Metadados sobre capacidades do estimador.
        """
