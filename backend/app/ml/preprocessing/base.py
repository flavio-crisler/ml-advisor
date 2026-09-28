from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class FeatureTransformer(ABC):
    """Interface abstrata para transformadores de colunas/features em DataFrames.

    Methods:
        fit: Aprende parâmetros da transformação com base nas features de treino.
        transform: Aplica as transformações aprendidas sobre o conjunto de features.
        get_config: Retorna a configuração serializável do transformador.
    """

    @abstractmethod
    def fit(self, features: Any, target: Any | None = None) -> "FeatureTransformer":
        """Ajusta o transformador aos dados de treino informados.

        Args:
            features (Any): Dados de entrada (geralmente DataFrame pandas).
            target (Any | None): Vetor alvo opcional.

        Returns:
            FeatureTransformer: A própria instância ajustada.
        """

    @abstractmethod
    def transform(self, features: Any) -> Any:
        """Aplica a transformação sobre o conjunto de features.

        Args:
            features (Any): Dados de entrada a serem transformados.

        Returns:
            Any: Dados com as transformações aplicadas.
        """

    @abstractmethod
    def get_config(self) -> Mapping[str, Any]:
        """Retorna o dicionário de configurações e parâmetros aprendidos do transformador.

        Returns:
            Mapping[str, Any]: Dicionário com metadados e estado do transformador.
        """


class RowTransformer(ABC):
    """Interface abstrata para transformadores que realizam reamostragem ou filtragem de linhas.

    Methods:
        fit_resample: Filtra ou reamostra as linhas de features e target conjuntamente.
        get_config: Retorna a configuração serializável da reamostragem.
    """

    @abstractmethod
    def fit_resample(self, features: Any, target: Any) -> tuple[Any, Any]:
        """Ajusta e reamostra/filtra conjuntamente as matrizes de features e target.

        Args:
            features (Any): Conjunto de features original.
            target (Any): Valores alvo originais.

        Returns:
            tuple[Any, Any]: Tupla contendo features e target filtrados/reamostrados.
        """

    @abstractmethod
    def get_config(self) -> Mapping[str, Any]:
        """Retorna a configuração do transformador de linhas.

        Returns:
            Mapping[str, Any]: Dicionário de configuração da operação.
        """
