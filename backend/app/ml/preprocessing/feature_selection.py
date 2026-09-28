from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from app.ml.preprocessing.base import FeatureTransformer

FeatureSelectionStrategy = Literal["none", "variance_threshold"]


class BasicFeatureSelector(FeatureTransformer):
    """Seleciona features removendo aquelas com variância zero ou abaixo do limiar configurado.

    Attributes:
        strategy (FeatureSelectionStrategy): Estratégia ('none', 'variance_threshold').
        threshold (float): Limiar de variância mínima permitida para manter uma feature.
        selected_columns (tuple[str, ...]): Colunas mantidas após a seleção.
        dropped_columns (tuple[str, ...]): Colunas descartadas por baixa variância.
    """

    def __init__(self, strategy: FeatureSelectionStrategy, *, threshold: float = 0.0) -> None:
        """Inicializa o seletor de features por variância.

        Args:
            strategy (FeatureSelectionStrategy): Estratégia ('none' ou 'variance_threshold').
            threshold (float): Limite de corte de variância mínima. Padrão 0.0.
        """
        self.strategy = strategy
        self.threshold = threshold
        self.selected_columns: tuple[str, ...] = ()
        self.dropped_columns: tuple[str, ...] = ()

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "BasicFeatureSelector":
        """Identifica quais colunas atendem ao critério mínimo de variância.

        Args:
            features (pd.DataFrame): DataFrame de treino.
            target (Any | None): Ignorado.

        Returns:
            BasicFeatureSelector: Instância ajustada do seletor.

        Raises:
            ValueError: Se a seleção remover todas as features disponíveis.
        """
        del target
        if self.strategy == "none":
            self.selected_columns = tuple(features.columns)
            self.dropped_columns = ()
            return self
        selected: list[str] = []
        dropped: list[str] = []
        for column in features:
            if pd.api.types.is_numeric_dtype(features[column]):
                keep = float(features[column].var(ddof=0)) > self.threshold
            else:
                keep = features[column].nunique(dropna=False) > 1
            (selected if keep else dropped).append(column)
        if not selected:
            raise ValueError("Feature selection removed every feature")
        self.selected_columns = tuple(selected)
        self.dropped_columns = tuple(dropped)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Retorna o DataFrame contendo apenas as colunas que foram selecionadas.

        Args:
            features (pd.DataFrame): DataFrame original.

        Returns:
            pd.DataFrame: DataFrame filtrado com as features selecionadas.
        """
        return features.loc[:, self.selected_columns].copy()

    def get_config(self) -> Mapping[str, Any]:
        """Retorna o relatório e configuração das colunas mantidas e descartadas.

        Returns:
            Mapping[str, Any]: Dicionário com threshold, selected_columns e dropped_columns.
        """
        return {
            "id": "feature_selection",
            "strategy": self.strategy,
            "kind": "feature",
            "threshold": self.threshold,
            "selected_columns": list(self.selected_columns),
            "dropped_columns": list(self.dropped_columns),
        }
