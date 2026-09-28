from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from app.ml.preprocessing.base import FeatureTransformer

BooleanStrategy = Literal["binary", "passthrough"]


class BooleanEncoder(FeatureTransformer):
    """Converte valores booleanos textuais ou numéricos em inteiros binários (0 ou 1).

    Attributes:
        columns (tuple[str, ...]): Colunas booleanas a serem tratadas.
        strategy (BooleanStrategy): Estratégia ('binary' ou 'passthrough').
        active_columns (tuple[str, ...]): Colunas ativas no DataFrame.
    """

    _true_values = {"true", "1", "yes", "y", "sim", "s"}
    _false_values = {"false", "0", "no", "n", "nao", "não"}

    def __init__(self, columns: list[str], strategy: BooleanStrategy) -> None:
        """Inicializa o codificador booleano.

        Args:
            columns (list[str]): Colunas booleanas a codificar.
            strategy (BooleanStrategy): Estratégia ('binary' ou 'passthrough').
        """
        self.columns = tuple(columns)
        self.strategy = strategy
        self.active_columns: tuple[str, ...] = ()

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "BooleanEncoder":
        """Valida e ajusta o codificador nas colunas booleanas presentes no DataFrame.

        Args:
            features (pd.DataFrame): DataFrame de treino.
            target (Any | None): Ignorado.

        Returns:
            BooleanEncoder: Instância ajustada do codificador booleano.

        Raises:
            ValueError: Se houver valores booleanos textuais desconhecidos.
        """
        del target
        self.active_columns = tuple(column for column in self.columns if column in features)
        if self.strategy == "binary":
            self._convert(features)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transforma os valores booleanos nas colunas em inteiros binários (0 ou 1).

        Args:
            features (pd.DataFrame): DataFrame com colunas booleanas a transformar.

        Returns:
            pd.DataFrame: DataFrame com os valores mapeados em 0 ou 1.
        """
        if self.strategy == "passthrough":
            return features.copy()
        transformed = features.copy()
        converted = self._convert(features)
        for column, values in converted.items():
            transformed[column] = values
        return transformed

    def get_config(self) -> Mapping[str, Any]:
        """Retorna o dicionário de configurações do codificador booleano.

        Returns:
            Mapping[str, Any]: Configuração contendo estratégia e colunas tratadas.
        """
        return {
            "id": "boolean_encoding",
            "strategy": self.strategy,
            "kind": "feature",
            "columns": list(self.active_columns),
        }

    def _convert(self, features: pd.DataFrame) -> dict[str, pd.Series]:
        """Converte strings e representações booleanas para 0 ou 1 estrito.

        Args:
            features (pd.DataFrame): DataFrame contendo as colunas a converter.

        Returns:
            dict[str, pd.Series]: Mapeamento de coluna para Série binária convertida.

        Raises:
            ValueError: Se forem identificados valores fora dos conjuntos de True/False aceitos.
        """
        converted: dict[str, pd.Series] = {}
        for column in self.active_columns:
            normalized = features[column].astype(str).str.strip().str.lower()
            unknown = set(normalized) - self._true_values - self._false_values
            if unknown:
                raise ValueError(
                    f"Column {column} contains invalid boolean values: {sorted(unknown)}"
                )
            converted[column] = normalized.map(lambda value: 1 if value in self._true_values else 0)
        return converted
