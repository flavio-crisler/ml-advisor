from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from app.ml.preprocessing.base import FeatureTransformer

DateTimeStrategy = Literal["components", "ordinal", "drop"]


class DateTimeTransformer(FeatureTransformer):
    """Extrai componentes ou converte colunas de data/hora para variáveis numéricas.

    Attributes:
        columns (tuple[str, ...]): Colunas datetime originais.
        strategy (DateTimeStrategy): Estratégia ('components', 'ordinal', 'drop').
        active_columns (tuple[str, ...]): Colunas presentes no conjunto de dados.
        output_columns (tuple[str, ...]): Nomes das colunas resultantes da transformação.
    """

    def __init__(self, columns: list[str], strategy: DateTimeStrategy) -> None:
        """Inicializa o transformador de data e hora.

        Args:
            columns (list[str]): Lista de colunas datetime a serem transformadas.
            strategy (DateTimeStrategy): Estratégia de decomposição ('components',
                'ordinal', 'drop').
        """
        self.columns = tuple(columns)
        self.strategy = strategy
        self.active_columns: tuple[str, ...] = ()
        self.output_columns: tuple[str, ...] = ()

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "DateTimeTransformer":
        """Calcula os nomes das colunas derivadas com base na estratégia selecionada.

        Args:
            features (pd.DataFrame): DataFrame de treino.
            target (Any | None): Ignorado.

        Returns:
            DateTimeTransformer: Instância ajustada do transformador.
        """
        del target
        self.active_columns = tuple(column for column in self.columns if column in features)
        if self.strategy == "components":
            self.output_columns = tuple(
                f"{column}_{component}"
                for column in self.active_columns
                for component in ("year", "month", "day", "dayofweek")
            )
        elif self.strategy == "ordinal":
            self.output_columns = tuple(f"{column}_ordinal" for column in self.active_columns)
        else:
            self.output_columns = ()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Decompõe ou converte as colunas datetime do DataFrame.

        Args:
            features (pd.DataFrame): DataFrame com colunas de data e hora a transformar.

        Returns:
            pd.DataFrame: DataFrame com as colunas originais removidas e derivadas adicionadas.

        Raises:
            ValueError: Se alguma coluna contiver datas inválidas não parseáveis.
        """
        transformed = features.drop(columns=list(self.active_columns)).copy()
        for column in self.active_columns:
            values = pd.to_datetime(features[column], errors="coerce", utc=True)
            if values.isna().any():
                raise ValueError(f"Column {column} contains invalid datetime values")
            if self.strategy == "components":
                transformed[f"{column}_year"] = values.dt.year
                transformed[f"{column}_month"] = values.dt.month
                transformed[f"{column}_day"] = values.dt.day
                transformed[f"{column}_dayofweek"] = values.dt.dayofweek
            elif self.strategy == "ordinal":
                transformed[f"{column}_ordinal"] = values.astype("int64") / 86_400_000_000_000
        return transformed

    def get_config(self) -> Mapping[str, Any]:
        """Retorna o dicionário de configurações da transformação datetime.

        Returns:
            Mapping[str, Any]: Configuração contendo colunas de entrada e saída geradas.
        """
        return {
            "id": "datetime",
            "strategy": self.strategy,
            "kind": "feature",
            "input_columns": list(self.active_columns),
            "output_columns": list(self.output_columns),
        }
