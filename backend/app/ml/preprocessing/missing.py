from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.ml.preprocessing.base import FeatureTransformer, RowTransformer

MissingStrategy = Literal["drop_rows", "drop_columns", "mean", "median", "mode", "constant"]


def _plain_value(value: Any) -> Any:
    """Converte tipos escalares do NumPy ou pandas em tipos nativos do Python serializáveis em JSON.

    Args:
        value (Any): Valor escalar arbitrário (ex: np.generic, pd.Timestamp).

    Returns:
        Any: Valor equivalente convertido para tipo nativo primitivo.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class MissingRowDropper(RowTransformer):
    """Descarta linhas contendo valores nulos em features ou target (estratégia drop_rows)."""

    def fit_resample(
        self, features: pd.DataFrame, target: pd.Series | Any
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Remove observações com valores faltantes mantendo o alinhamento de índices.

        Args:
            features (pd.DataFrame): DataFrame de entrada de features.
            target (pd.Series | Any): Série ou vetor de target.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Features e target contendo apenas linhas completas.

        Raises:
            ValueError: Se o número de linhas de features e target for divergente.
        """
        target_series = (
            target.copy()
            if isinstance(target, pd.Series)
            else pd.Series(target, index=features.index, name="target")
        )
        if len(features) != len(target_series):
            raise ValueError("Features and target must have the same row count")
        keep = ~features.isna().any(axis=1) & ~target_series.isna()
        return features.loc[keep].copy(), target_series.loc[keep].copy()

    def get_config(self) -> Mapping[str, Any]:
        """Retorna a configuração da estratégia de remoção de linhas.

        Returns:
            Mapping[str, Any]: Dicionário identificador da etapa.
        """
        return {"id": "missing_values", "strategy": "drop_rows", "kind": "row"}


class MissingValueTransformer(FeatureTransformer):
    """Trata valores faltantes em colunas usando imputação ou descarte de colunas.

    Attributes:
        strategy (MissingStrategy): Estratégia ('drop_columns', 'mean', 'median',
            'mode', 'constant').
        constant_value (Any): Valor usado para imputação constante.
        text_columns (tuple[str, ...]): Nomes de colunas de texto livre.
        input_columns (tuple[str, ...]): Colunas presentes no fit.
        dropped_columns (tuple[str, ...]): Colunas eliminadas por conterem nulos.
        fill_values (dict[str, Any]): Valores de preenchimento aprendidos por coluna.
    """

    def __init__(
        self,
        strategy: MissingStrategy,
        *,
        constant_value: Any = "__missing__",
        text_columns: tuple[str, ...] = (),
    ) -> None:
        """Inicializa o transformador de imputação de valores ausentes.

        Args:
            strategy (MissingStrategy): Estratégia de preenchimento ou descarte.
            constant_value (Any): Valor constante usado quando a estratégia for 'constant'.
            text_columns (tuple[str, ...]): Colunas textuais (imputadas com string vazia).
        """
        self.strategy = strategy
        self.constant_value = constant_value
        self.text_columns = text_columns
        self.input_columns: tuple[str, ...] = ()
        self.dropped_columns: tuple[str, ...] = ()
        self.fill_values: dict[str, Any] = {}

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "MissingValueTransformer":
        """Aprende os valores de imputação (média, mediana, moda) ou colunas a descartar.

        Args:
            features (pd.DataFrame): DataFrame de treino.
            target (Any | None): Ignorado.

        Returns:
            MissingValueTransformer: Instância ajustada do transformador.
        """
        del target
        self.input_columns = tuple(features.columns)
        self.dropped_columns = (
            tuple(column for column in features if features[column].isna().any())
            if self.strategy == "drop_columns"
            else ()
        )
        self.fill_values = {
            column: self._fill_value(features[column], is_text=column in self.text_columns)
            for column in features
            if column not in self.dropped_columns
        }
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aplica a imputação ou remove colunas descartadas.

        Args:
            features (pd.DataFrame): DataFrame com valores faltantes.

        Returns:
            pd.DataFrame: DataFrame transformado sem valores nulos nas colunas ativas.

        Raises:
            ValueError: Se o DataFrame não contiver todas as colunas observadas no fit.
        """
        missing = set(self.input_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Input is missing fitted columns: {sorted(missing)}")
        transformed = features.loc[:, self.input_columns].drop(columns=list(self.dropped_columns))
        return transformed.fillna(self.fill_values)

    def get_config(self) -> Mapping[str, Any]:
        """Retorna o dicionário de configurações e valores de imputação calculados.

        Returns:
            Mapping[str, Any]: Metadados de configuração e valores por coluna.
        """
        return {
            "id": "missing_values",
            "strategy": self.strategy,
            "kind": "feature",
            "constant_value": _plain_value(self.constant_value),
            "dropped_columns": list(self.dropped_columns),
            "fill_values": {
                column: _plain_value(value) for column, value in self.fill_values.items()
            },
        }

    def _fill_value(self, values: pd.Series, *, is_text: bool = False) -> Any:
        """Determina o valor de imputação para uma coluna específica.

        Args:
            values (pd.Series): Série de valores da coluna.
            is_text (bool): Se verdadeiro, imputa com string vazia.

        Returns:
            Any: Valor de substituição a ser usado no fillna.
        """
        if self.strategy == "constant":
            return self.constant_value
        if is_text:
            return ""
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            if self.strategy == "mean":
                result = numeric.mean()
            elif self.strategy in {"median", "drop_rows", "drop_columns"}:
                result = numeric.median()
            else:
                modes = numeric.mode(dropna=True)
                result = modes.iloc[0] if not modes.empty else np.nan
            return self.constant_value if pd.isna(result) else result
        modes = values.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else self.constant_value
