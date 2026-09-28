from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from app.ml.preprocessing.base import FeatureTransformer

CategoricalStrategy = Literal["one_hot", "ordinal", "passthrough"]


class CategoricalEncoder(FeatureTransformer):
    """Codifica colunas categóricas utilizando One-Hot Encoding ou Ordinal Encoding.

    Attributes:
        columns (tuple[str, ...]): Colunas categóricas originais.
        strategy (CategoricalStrategy): Estratégia ('one_hot', 'ordinal', 'passthrough').
        active_columns (tuple[str, ...]): Colunas presentes no conjunto ajustado.
        encoder (OneHotEncoder | OrdinalEncoder | None): Estimador do Scikit-Learn ajustado.
        output_columns (tuple[str, ...]): Nomes das novas colunas geradas após codificação.
    """

    def __init__(self, columns: list[str], strategy: CategoricalStrategy) -> None:
        """Inicializa o codificador categórico.

        Args:
            columns (list[str]): Colunas categóricas a serem codificadas.
            strategy (CategoricalStrategy): Tipo de codificação selecionada.
        """
        self.columns = tuple(columns)
        self.strategy = strategy
        self.active_columns: tuple[str, ...] = ()
        self.encoder: OneHotEncoder | OrdinalEncoder | None = None
        self.output_columns: tuple[str, ...] = ()

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "CategoricalEncoder":
        """Ajusta o codificador nas colunas categóricas presentes no DataFrame.

        Args:
            features (pd.DataFrame): DataFrame contendo as features categóricas de treino.
            target (Any | None): Ignorado.

        Returns:
            CategoricalEncoder: Instância ajustada do codificador.
        """
        del target
        self.active_columns = tuple(column for column in self.columns if column in features)
        if not self.active_columns or self.strategy == "passthrough":
            self.output_columns = self.active_columns
            return self
        values = features.loc[:, self.active_columns].astype(str)
        if self.strategy == "one_hot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder.fit(values)
        self.output_columns = tuple(self.encoder.get_feature_names_out(self.active_columns))
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aplica a codificação One-Hot ou Ordinal substituindo as colunas originais.

        Args:
            features (pd.DataFrame): DataFrame com colunas categóricas a transformar.

        Returns:
            pd.DataFrame: DataFrame com as novas colunas codificadas.
        """
        if not self.active_columns or self.strategy == "passthrough":
            return features.copy()
        assert self.encoder is not None
        encoded = self.encoder.transform(features.loc[:, self.active_columns].astype(str))
        encoded_frame = pd.DataFrame(
            encoded,
            index=features.index,
            columns=self.output_columns,
        )
        return pd.concat([features.drop(columns=list(self.active_columns)), encoded_frame], axis=1)

    def get_config(self) -> Mapping[str, Any]:
        """Retorna as configurações e mapeamento de colunas de entrada e saída.

        Returns:
            Mapping[str, Any]: Dicionário de configuração da etapa categórica.
        """
        return {
            "id": "categorical_encoding",
            "strategy": self.strategy,
            "kind": "feature",
            "input_columns": list(self.active_columns),
            "output_columns": list(self.output_columns),
        }
