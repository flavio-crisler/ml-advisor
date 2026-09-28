from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from app.ml.preprocessing.base import FeatureTransformer

NumericStrategy = Literal["standard", "min_max", "robust", "passthrough"]
OutlierStrategy = Literal["none", "quantile_clip", "iqr"]


class OutlierClipper(FeatureTransformer):
    """Limita valores discrepantes (outliers) em colunas numéricas via quantis ou IQR.

    Attributes:
        columns (tuple[str, ...]): Colunas numéricas alvo da análise.
        strategy (OutlierStrategy): Estratégia de corte ('quantile_clip', 'iqr' ou 'none').
        lower_quantile (float): Limite inferior percentual para quantile_clip.
        upper_quantile (float): Limite superior percentual para quantile_clip.
        iqr_multiplier (float): Multiplicador do intervalo interquartil (IQR).
        active_columns (tuple[str, ...]): Colunas presentes no conjunto de dados ajustado.
        bounds (dict[str, tuple[float, float]]): Limites inferior e superior aprendidos por coluna.
    """

    def __init__(
        self,
        columns: list[str],
        strategy: OutlierStrategy,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        iqr_multiplier: float = 1.5,
    ) -> None:
        """Inicializa o limitador de outliers.

        Args:
            columns (list[str]): Lista de colunas a tratar.
            strategy (OutlierStrategy): Estratégia de tratamento de outliers.
            lower_quantile (float): Quantil inferior para corte. Padrão 0.01 (1%).
            upper_quantile (float): Quantil superior para corte. Padrão 0.99 (99%).
            iqr_multiplier (float): Fator do intervalo interquartil. Padrão 1.5.
        """
        self.columns = tuple(columns)
        self.strategy = strategy
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.iqr_multiplier = iqr_multiplier
        self.active_columns: tuple[str, ...] = ()
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "OutlierClipper":
        """Calcula os limites inferior e superior para cada coluna configurada.

        Args:
            features (pd.DataFrame): DataFrame de features de treino.
            target (Any | None): Ignorado.

        Returns:
            OutlierClipper: A própria instância com limites calculados.
        """
        del target
        self.active_columns = tuple(column for column in self.columns if column in features)
        self.bounds = {}
        if self.strategy == "none":
            return self
        for column in self.active_columns:
            values = pd.to_numeric(features[column], errors="raise")
            if self.strategy == "quantile_clip":
                lower = float(values.quantile(self.lower_quantile))
                upper = float(values.quantile(self.upper_quantile))
            else:
                first_quartile = float(values.quantile(0.25))
                third_quartile = float(values.quantile(0.75))
                interval = third_quartile - first_quartile
                lower = first_quartile - self.iqr_multiplier * interval
                upper = third_quartile + self.iqr_multiplier * interval
            self.bounds[column] = (lower, upper)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aplica o truncamento (clip) aos valores fora dos limites aprendidos.

        Args:
            features (pd.DataFrame): DataFrame com colunas numéricas a transformar.

        Returns:
            pd.DataFrame: DataFrame com valores limitados aos limites superior e inferior.
        """
        transformed = features.copy()
        for column, (lower, upper) in self.bounds.items():
            transformed[column] = transformed[column].clip(lower, upper)
        return transformed

    def get_config(self) -> Mapping[str, Any]:
        """Retorna as configurações e limites aprendidos do limitador de outliers.

        Returns:
            Mapping[str, Any]: Dicionário com estratégia e limites por coluna.
        """
        return {
            "id": "outliers",
            "strategy": self.strategy,
            "kind": "feature",
            "bounds": {
                column: {"lower": lower, "upper": upper}
                for column, (lower, upper) in self.bounds.items()
            },
        }


class NumericScaler(FeatureTransformer):
    """Padroniza ou normaliza colunas numéricas usando estimadores do Scikit-Learn.

    Attributes:
        columns (tuple[str, ...]): Colunas numéricas a escalar.
        strategy (NumericStrategy): Tipo de escala ('standard', 'min_max', 'robust', 'passthrough').
        active_columns (tuple[str, ...]): Colunas presentes no DataFrame.
        scaler (StandardScaler | MinMaxScaler | RobustScaler | None): Estimador de escala ajustado.
    """

    def __init__(self, columns: list[str], strategy: NumericStrategy) -> None:
        """Inicializa o escalonador numérico.

        Args:
            columns (list[str]): Colunas numéricas a transformar.
            strategy (NumericStrategy): Estratégia de escalonamento.
        """
        self.columns = tuple(columns)
        self.strategy = strategy
        self.active_columns: tuple[str, ...] = ()
        self.scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "NumericScaler":
        """Ajusta o estimador de escala nas colunas numéricas ativas.

        Args:
            features (pd.DataFrame): DataFrame de treino.
            target (Any | None): Ignorado.

        Returns:
            NumericScaler: Instância ajustada do escalonador.
        """
        del target
        self.active_columns = tuple(column for column in self.columns if column in features)
        if not self.active_columns or self.strategy == "passthrough":
            return self
        scalers = {
            "standard": StandardScaler,
            "min_max": MinMaxScaler,
            "robust": RobustScaler,
        }
        self.scaler = scalers[self.strategy]().fit(features.loc[:, self.active_columns])
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aplica a escala aprendida sobre as colunas numéricas.

        Args:
            features (pd.DataFrame): DataFrame com as colunas a escalar.

        Returns:
            pd.DataFrame: DataFrame transformado.
        """
        transformed = features.copy()
        if self.scaler is not None:
            scaled = self.scaler.transform(features.loc[:, self.active_columns])
            for index, column in enumerate(self.active_columns):
                transformed[column] = scaled[:, index]
        return transformed

    def get_config(self) -> Mapping[str, Any]:
        """Retorna os hiperparâmetros e estatísticas aprendidas pelo escalonador.

        Returns:
            Mapping[str, Any]: Dicionário com médias, desvios ou mínimos/máximos.
        """
        fitted: dict[str, Any] = {}
        if isinstance(self.scaler, StandardScaler):
            fitted = {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
        elif isinstance(self.scaler, MinMaxScaler):
            fitted = {
                "minimum": self.scaler.data_min_.tolist(),
                "maximum": self.scaler.data_max_.tolist(),
            }
        elif isinstance(self.scaler, RobustScaler):
            fitted = {
                "center": self.scaler.center_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
        return {
            "id": "numeric_scaling",
            "strategy": self.strategy,
            "kind": "feature",
            "columns": list(self.active_columns),
            "fitted": fitted,
        }
