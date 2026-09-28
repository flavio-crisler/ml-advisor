import math
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.domain.experiment.spec import ColumnRole


class ColumnType(StrEnum):
    """Enumeração dos tipos semânticos inferidos para colunas de datasets."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    EMPTY = "empty"


class ValueCount(BaseModel):
    """Contagem e proporção de ocorrência de um valor específico em uma coluna.

    Attributes:
        value (str): Representação textual do valor.
        count (int): Quantidade absoluta de ocorrências.
        percentage (float): Proporção percentual frente aos valores válidos (não nulos).
    """

    model_config = ConfigDict(frozen=True)

    value: str
    count: int
    percentage: float


class HistogramBucket(BaseModel):
    """Intervalo de valores e contagem de frequência para histogramas numéricos.

    Attributes:
        lower (float): Limite inferior do bucket.
        upper (float): Limite superior do bucket.
        count (int): Quantidade de observações contidas neste intervalo.
    """

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    count: int


class NumericStatistics(BaseModel):
    """Estatísticas descritivas calculadas para uma coluna numérica contínua.

    Attributes:
        minimum (float): Menor valor observado.
        maximum (float): Maior valor observado.
        mean (float): Média aritmética dos valores.
        median (float): Mediana (percentil 50).
        standard_deviation (float): Desvio padrão amostral/populacional.
        first_quartile (float): Primeiro quartil (percentil 25).
        third_quartile (float): Terceiro quartil (percentil 75).
        outlier_count (int): Quantidade de outliers detectados via critério de 1.5 * IQR.
    """

    model_config = ConfigDict(frozen=True)

    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float
    first_quartile: float
    third_quartile: float
    outlier_count: int


class ColumnProfile(BaseModel):
    """Perfil estatístico e semântico detalhado de uma coluna individual.

    Attributes:
        name (str): Nome original da coluna.
        source_dtype (str): Tipo primitivo no DataFrame fonte.
        inferred_type (ColumnType): Tipo semântico inferido pelo profiler.
        suggested_role (ColumnRole): Papel sugerido (feature, target, id, ignored).
        non_null_count (int): Total de valores preenchidos.
        missing_count (int): Total de valores ausentes (nulos ou NaN).
        missing_percentage (float): Percentual de valores faltantes.
        cardinality (int): Quantidade de valores únicos distintos.
        uniqueness_ratio (float): Razão entre valores únicos e total não-nulo.
        semantic_type (Literal["free_text"] | None): Classificação semântica especial.
        average_length (float | None): Comprimento médio de strings (para texto).
        numeric_statistics (NumericStatistics | None): Estatísticas descritivas (se numérica).
        histogram (list[HistogramBucket]): Histograma de distribuição dos dados.
        top_values (list[ValueCount]): Valores mais frequentes e suas contagens.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    source_dtype: str
    inferred_type: ColumnType
    suggested_role: ColumnRole
    non_null_count: int
    missing_count: int
    missing_percentage: float
    cardinality: int
    uniqueness_ratio: float
    semantic_type: Literal["free_text"] | None = None
    average_length: float | None = Field(default=None, ge=0)
    numeric_statistics: NumericStatistics | None = None
    histogram: list[HistogramBucket] = Field(default_factory=list)
    top_values: list[ValueCount] = Field(default_factory=list)


class ProfileWarning(BaseModel):
    """Alerta ou diagnóstico gerado durante a análise do perfil de dados.

    Attributes:
        code (str): Código identificador do aviso.
        severity (Literal["info", "warning"]): Nível de severidade do diagnóstico.
        column (str | None): Nome da coluna associada ao alerta, se houver.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    severity: Literal["info", "warning"]
    column: str | None = None


class DatasetProfile(BaseModel):
    """Perfil estatístico consolidado do conjunto de dados completo.

    Attributes:
        profile_version (str): Versão do esquema do perfil.
        rows (int): Quantidade total de linhas.
        columns (int): Quantidade total de colunas.
        size_bytes (int): Tamanho bruto do arquivo em bytes.
        encoding (str): Codificação de caracteres do arquivo.
        delimiter (str): Delimitador utilizado para separar colunas.
        column_names (list[str]): Nomes de todas as colunas na ordem original.
        numeric_columns (list[str]): Colunas identificadas como numéricas.
        categorical_columns (list[str]): Colunas categóricas.
        boolean_columns (list[str]): Colunas booleanas.
        datetime_columns (list[str]): Colunas de data e hora.
        text_columns (list[str]): Colunas de texto livre.
        candidate_targets (list[str]): Colunas elegíveis como alvos de modelagem.
        candidate_target_distributions (dict[str, list[ValueCount]]): Distribuições dos alvos.
        high_cardinality_columns (list[str]): Colunas com alta cardinalidade.
        possible_id_columns (list[str]): Colunas com forte indício de serem identificadores.
        constant_columns (list[str]): Colunas com um único valor (constantes).
        near_constant_columns (list[str]): Colunas quase constantes (>= 98% mesmo valor).
        possible_leakage_columns (list[str]): Colunas suspeitas de vazamento de dados.
        missing_values (dict[str, int]): Contagem de ausentes por coluna.
        missing_percentages (dict[str, float]): Percentual de ausentes por coluna.
        duplicates (int): Quantidade de linhas totalmente duplicadas.
        estimated_memory_bytes (int): Memória estimada consumida em RAM pelo DataFrame.
        column_profiles (list[ColumnProfile]): Lista de perfis individuais de cada coluna.
        preview (list[dict[str, Any]]): Amostra representativa das primeiras linhas.
        warnings (list[ProfileWarning]): Diagnósticos de qualidade gerados pelo profiler.
    """

    model_config = ConfigDict(frozen=True)

    profile_version: str = "1.1"
    rows: int
    columns: int
    size_bytes: int
    encoding: str
    delimiter: str
    column_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]
    text_columns: list[str] = Field(default_factory=list)
    candidate_targets: list[str]
    candidate_target_distributions: dict[str, list[ValueCount]]
    high_cardinality_columns: list[str]
    possible_id_columns: list[str]
    constant_columns: list[str]
    near_constant_columns: list[str]
    possible_leakage_columns: list[str]
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    duplicates: int
    estimated_memory_bytes: int
    column_profiles: list[ColumnProfile]
    preview: list[dict[str, Any]]
    warnings: list[ProfileWarning]


class DeterministicDatasetProfiler:
    """Analisador determinístico para geração de perfis e diagnósticos de dados tabulares.

    Attributes:
        preview_rows (int): Quantidade de linhas incluídas na amostra de preview.
    """

    _boolean_values = {"0", "1", "false", "n", "nao", "não", "no", "s", "sim", "true", "yes"}
    _target_names = {
        "alvo",
        "class",
        "classe",
        "label",
        "outcome",
        "resultado",
        "target",
        "y",
    }
    _leakage_tokens = {"after", "final", "future", "leak", "post", "prediction", "predicted"}

    def __init__(self, preview_rows: int = 20) -> None:
        """Inicializa o profiler com o número de linhas para a pré-visualização.

        Args:
            preview_rows: Quantidade de linhas para o preview. Padrão 20.
        """
        self.preview_rows = preview_rows

    def profile(
        self,
        frame: pd.DataFrame,
        *,
        size_bytes: int,
        encoding: str,
        delimiter: str,
    ) -> DatasetProfile:
        """Gera o perfil estatístico completo e determinístico do DataFrame.

        Analisa cada coluna para inferência de tipos, contagens de valores nulos e únicos,
        detecção de possíveis identificadores, colunas constantes ou vazamentos, além de
        calcular estatísticas numéricas, histogramas e listar alvos candidatos.

        Args:
            frame: DataFrame do Pandas com os dados carregados do dataset.
            size_bytes: Tamanho do arquivo original em bytes.
            encoding: Codificação de caracteres detectada.
            delimiter: Caractere delimitador de colunas identificado.

        Returns:
            DatasetProfile contendo os metadados consolidados e perfis por coluna.
        """
        row_count = len(frame)
        column_names = [str(column) for column in frame.columns]
        profiles: list[ColumnProfile] = []
        possible_ids: list[str] = []
        high_cardinality: list[str] = []
        constants: list[str] = []
        near_constants: list[str] = []
        leakage: list[str] = []

        inferred_types: dict[str, ColumnType] = {}
        for name in column_names:
            series = frame[name]
            inferred_type = self._infer_type(series)
            inferred_types[name] = inferred_type
            non_null_count = int(series.notna().sum())
            missing_count = row_count - non_null_count
            cardinality = int(series.nunique(dropna=True))
            uniqueness_ratio = cardinality / non_null_count if non_null_count else 0.0
            normalized_name = self._normalized_name(name)

            is_id = self._is_possible_id(
                normalized_name,
                inferred_type,
                non_null_count,
                uniqueness_ratio,
            )
            is_high_cardinality = (
                inferred_type == ColumnType.CATEGORICAL
                and cardinality > 20
                and (cardinality > 50 or uniqueness_ratio > 0.5)
            )
            is_constant = cardinality <= 1
            dominant_ratio = self._dominant_ratio(series)
            is_near_constant = not is_constant and dominant_ratio >= 0.98
            has_leakage_name = any(
                token in normalized_name.split("_") for token in self._leakage_tokens
            )

            if is_id:
                possible_ids.append(name)
            if is_high_cardinality:
                high_cardinality.append(name)
            if is_constant:
                constants.append(name)
            if is_near_constant:
                near_constants.append(name)
            if has_leakage_name:
                leakage.append(name)

            role = (
                ColumnRole.ID
                if is_id
                else ColumnRole.IGNORED
                if is_constant
                else ColumnRole.FEATURE
            )
            numeric_statistics, histogram = self._numeric_profile(series, inferred_type)
            profiles.append(
                ColumnProfile(
                    name=name,
                    source_dtype=str(series.dtype),
                    inferred_type=inferred_type,
                    suggested_role=role,
                    non_null_count=non_null_count,
                    missing_count=missing_count,
                    missing_percentage=self._percentage(missing_count, row_count),
                    cardinality=cardinality,
                    uniqueness_ratio=round(uniqueness_ratio, 4),
                    semantic_type=("free_text" if inferred_type == ColumnType.TEXT else None),
                    average_length=(
                        self._average_length(series) if inferred_type == ColumnType.TEXT else None
                    ),
                    numeric_statistics=numeric_statistics,
                    histogram=histogram,
                    top_values=self._top_values(series),
                )
            )

        candidate_targets = self._candidate_targets(
            frame,
            inferred_types,
            possible_ids,
            high_cardinality,
            constants,
        )
        profile_by_name = {profile.name: profile for profile in profiles}
        profiles = [
            profile.model_copy(
                update={"suggested_role": ColumnRole.TARGET}
                if profile.name in candidate_targets
                else {}
            )
            for profile in profiles
        ]

        duplicates = int(frame.duplicated().sum())
        warnings = self._warnings(
            profiles,
            possible_ids=possible_ids,
            high_cardinality=high_cardinality,
            constants=constants,
            near_constants=near_constants,
            leakage=leakage,
            candidate_targets=candidate_targets,
            duplicates=duplicates,
        )
        target_distributions = {
            target: self._top_values(
                frame[target],
                limit=min(profile_by_name[target].cardinality, 20),
            )
            for target in candidate_targets
        }

        return DatasetProfile(
            rows=row_count,
            columns=len(column_names),
            size_bytes=size_bytes,
            encoding=encoding,
            delimiter=delimiter,
            column_names=column_names,
            numeric_columns=self._columns_of_type(inferred_types, ColumnType.NUMERIC),
            categorical_columns=self._columns_of_type(inferred_types, ColumnType.CATEGORICAL),
            boolean_columns=self._columns_of_type(inferred_types, ColumnType.BOOLEAN),
            datetime_columns=self._columns_of_type(inferred_types, ColumnType.DATETIME),
            text_columns=self._columns_of_type(inferred_types, ColumnType.TEXT),
            candidate_targets=candidate_targets,
            candidate_target_distributions=target_distributions,
            high_cardinality_columns=high_cardinality,
            possible_id_columns=possible_ids,
            constant_columns=constants,
            near_constant_columns=near_constants,
            possible_leakage_columns=leakage,
            missing_values={profile.name: profile.missing_count for profile in profiles},
            missing_percentages={profile.name: profile.missing_percentage for profile in profiles},
            duplicates=duplicates,
            estimated_memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
            column_profiles=profiles,
            preview=self._preview(frame),
            warnings=warnings,
        )

    def _infer_type(self, series: pd.Series) -> ColumnType:
        """Infere deterministicamente o tipo semântico da coluna a partir dos valores válidos.

        Args:
            series: Série do Pandas contendo os valores da coluna.

        Returns:
            ColumnType correspondente ao tipo semântico inferido.
        """
        non_null = series.dropna()
        if non_null.empty:
            return ColumnType.EMPTY
        if pd.api.types.is_bool_dtype(series.dtype):
            return ColumnType.BOOLEAN
        normalized = {str(value).strip().casefold() for value in non_null.unique()}
        if len(normalized) <= 2 and normalized <= self._boolean_values:
            return ColumnType.BOOLEAN
        if pd.api.types.is_numeric_dtype(series.dtype):
            return ColumnType.NUMERIC
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return ColumnType.DATETIME
        parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if float(parsed_dates.notna().mean()) >= 0.9:
            return ColumnType.DATETIME
        string_values = non_null.astype(str)
        average_length = float(string_values.str.len().mean())
        uniqueness_ratio = float(string_values.nunique() / len(string_values))
        if average_length >= 40 or (int(string_values.nunique()) > 20 and uniqueness_ratio >= 0.5):
            return ColumnType.TEXT
        return ColumnType.CATEGORICAL

    def _numeric_profile(
        self, series: pd.Series, inferred_type: ColumnType
    ) -> tuple[NumericStatistics | None, list[HistogramBucket]]:
        """Calcula estatísticas descritivas e histograma para coluna numérica contínua.

        Args:
            series: Série contendo os valores numéricos brutos.
            inferred_type: Tipo semântico inferido previamente.

        Returns:
            Tupla contendo as estatísticas numéricas (ou None) e a lista de buckets do histograma.
        """
        if inferred_type != ColumnType.NUMERIC:
            return None, []
        numeric = (
            pd.to_numeric(series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if numeric.empty:
            return None, []
        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        outlier_count = 0
        if iqr > 0:
            outlier_count = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
        statistics = NumericStatistics(
            minimum=float(numeric.min()),
            maximum=float(numeric.max()),
            mean=float(numeric.mean()),
            median=float(numeric.median()),
            standard_deviation=float(numeric.std(ddof=0)),
            first_quartile=q1,
            third_quartile=q3,
            outlier_count=outlier_count,
        )
        if float(numeric.min()) == float(numeric.max()):
            return statistics, []
        counts, edges = np.histogram(numeric, bins=min(10, max(2, int(math.sqrt(len(numeric))))))
        histogram = [
            HistogramBucket(
                lower=float(edges[index]), upper=float(edges[index + 1]), count=int(count)
            )
            for index, count in enumerate(counts)
        ]
        return statistics, histogram

    def _candidate_targets(
        self,
        frame: pd.DataFrame,
        inferred_types: dict[str, ColumnType],
        possible_ids: list[str],
        high_cardinality: list[str],
        constants: list[str],
    ) -> list[str]:
        """Identifica colunas elegíveis como alvos potenciais para modelos preditivos.

        Args:
            frame: DataFrame completo com as colunas avaliadas.
            inferred_types: Mapeamento de tipos inferidos por coluna.
            possible_ids: Lista de colunas identificadas como chaves/IDs.
            high_cardinality: Lista de colunas com alta cardinalidade.
            constants: Lista de colunas com valores constantes.

        Returns:
            Lista de nomes de colunas candidatos a target.
        """
        excluded = set(possible_ids + high_cardinality + constants)
        named: list[str] = []
        low_cardinality: list[str] = []
        for name in inferred_types:
            if name in excluded or inferred_types[name] in {
                ColumnType.DATETIME,
                ColumnType.EMPTY,
                ColumnType.TEXT,
            }:
                continue
            normalized_name = self._normalized_name(name)
            cardinality = int(frame[name].nunique(dropna=True))
            if normalized_name in self._target_names or normalized_name.endswith("_target"):
                named.append(name)
            elif 2 <= cardinality <= 20:
                low_cardinality.append(name)
        return list(dict.fromkeys(named + low_cardinality))[:10]

    def _warnings(
        self,
        profiles: list[ColumnProfile],
        *,
        possible_ids: list[str],
        high_cardinality: list[str],
        constants: list[str],
        near_constants: list[str],
        leakage: list[str],
        candidate_targets: list[str],
        duplicates: int,
    ) -> list[ProfileWarning]:
        """Gera a lista de alertas diagnósticos sobre a qualidade dos dados tabulares.

        Args:
            profiles: Lista de perfis individuais de colunas.
            possible_ids: Nomes de colunas identificadas como IDs.
            high_cardinality: Nomes de colunas com alta cardinalidade.
            constants: Nomes de colunas com valores constantes.
            near_constants: Nomes de colunas quase constantes.
            leakage: Nomes de colunas suspeitas de vazamento.
            candidate_targets: Nomes de alvos candidatos identificados.
            duplicates: Quantidade de linhas duplicadas.

        Returns:
            Lista de alertas diagnósticos identificados no dataset.
        """
        warnings: list[ProfileWarning] = []
        if duplicates:
            warnings.append(ProfileWarning(code="duplicate_rows", severity="warning"))
        for profile in profiles:
            if profile.missing_count:
                warnings.append(
                    ProfileWarning(code="missing_values", severity="warning", column=profile.name)
                )
            if profile.numeric_statistics and profile.numeric_statistics.outlier_count:
                warnings.append(
                    ProfileWarning(code="possible_outliers", severity="info", column=profile.name)
                )
        warnings.extend(self._column_warnings("possible_id", possible_ids, "info"))
        warnings.extend(self._column_warnings("high_cardinality", high_cardinality, "warning"))
        warnings.extend(self._column_warnings("constant_column", constants, "warning"))
        warnings.extend(self._column_warnings("near_constant_column", near_constants, "warning"))
        warnings.extend(self._column_warnings("possible_leakage", leakage, "warning"))
        profile_by_name = {profile.name: profile for profile in profiles}
        for target in candidate_targets:
            counts = [value.count for value in profile_by_name[target].top_values]
            if len(counts) >= 2 and min(counts) / max(counts) < 0.2:
                warnings.append(
                    ProfileWarning(
                        code="possible_target_imbalance", severity="warning", column=target
                    )
                )
        return warnings

    @staticmethod
    def _column_warnings(
        code: str, columns: Iterable[str], severity: Literal["info", "warning"]
    ) -> list[ProfileWarning]:
        """Cria lista de ProfileWarning para as colunas fornecidas.

        Args:
            code: Código textual padronizado do alerta.
            columns: Coleção de nomes de colunas afetadas.
            severity: Severidade do aviso ('info' ou 'warning').

        Returns:
            Lista de objetos ProfileWarning criados.
        """
        return [ProfileWarning(code=code, severity=severity, column=column) for column in columns]

    @staticmethod
    def _top_values(series: pd.Series, limit: int = 10) -> list[ValueCount]:
        """Calcula os valores mais frequentes e suas respectivas porcentagens.

        Args:
            series: Série contendo os valores da coluna.
            limit: Quantidade máxima de valores mais frequentes a retornar. Padrão 10.

        Returns:
            Lista de ValueCount com valor, contagem e porcentagem.
        """
        non_null_count = int(series.notna().sum())
        if not non_null_count:
            return []
        counts = series.value_counts(dropna=True).head(limit)
        return [
            ValueCount(
                value=DeterministicDatasetProfiler._display_value(value),
                count=int(count),
                percentage=round(int(count) * 100 / non_null_count, 4),
            )
            for value, count in counts.items()
        ]

    def _preview(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Gera a pré-visualização tabular em formato serializável para JSON.

        Args:
            frame: DataFrame de onde extrair a amostra inicial.

        Returns:
            Lista de dicionários representando as primeiras linhas do DataFrame.
        """
        return [
            {str(key): self._json_value(value) for key, value in row.items()}
            for row in frame.head(self.preview_rows).to_dict(orient="records")
        ]

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Converte valores escalares, NaN e Timestamps em tipos serializáveis em JSON.

        Args:
            value: Valor bruto a ser sanitizado.

        Returns:
            Valor convertido ou None se for nulo/NaN.
        """
        if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _display_value(value: Any) -> str:
        """Formata valores de categorias e timestamps para representação textual simples.

        Args:
            value: Valor bruto para exibição.

        Returns:
            String formatada do valor.
        """
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _percentage(part: int, total: int) -> float:
        """Calcula o percentual proporcional arredondado a quatro casas decimais.

        Args:
            part: Quantidade parcial observada.
            total: Quantidade total de referência.

        Returns:
            Percentual calculado ou 0.0 se total for zero.
        """
        return round(part * 100 / total, 4) if total else 0.0

    @staticmethod
    def _dominant_ratio(series: pd.Series) -> float:
        """Calcula a proporção do valor mais representativo em relação ao total não-nulo.

        Args:
            series: Série do Pandas a analisar.

        Returns:
            Proporção float entre 0.0 e 1.0 do valor dominante.
        """
        counts = series.value_counts(dropna=True)
        return float(counts.iloc[0] / counts.sum()) if not counts.empty else 0.0

    @staticmethod
    def _average_length(series: pd.Series) -> float:
        """Calcula o comprimento médio de caracteres para strings de uma coluna.

        Args:
            series: Série contendo dados textuais.

        Returns:
            Comprimento médio em caracteres arredondado a duas casas decimais.
        """
        non_null = series.dropna().astype(str)
        return round(float(non_null.str.len().mean()), 2) if not non_null.empty else 0.0

    @staticmethod
    def _normalized_name(name: str) -> str:
        """Normaliza o nome da coluna para caracteres alfanuméricos em minúsculas.

        Args:
            name: Nome original da coluna.

        Returns:
            Nome normalizado com caracteres especiais substituídos por underscores.
        """
        return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")

    @staticmethod
    def _is_possible_id(
        name: str,
        inferred_type: ColumnType,
        non_null_count: int,
        uniqueness_ratio: float,
    ) -> bool:
        """Avalia se uma coluna possui características de chave primária ou índice.

        Args:
            name: Nome normalizado da coluna.
            inferred_type: Tipo semântico inferido da coluna.
            non_null_count: Quantidade de valores válidos.
            uniqueness_ratio: Razão de unicidade dos valores.

        Returns:
            True se a coluna apresentar padrões típicos de ID, False caso contrário.
        """
        name_looks_like_id = (
            name in {"id", "index"}
            or name.endswith("_id")
            or name.startswith("id_")
            or name.startswith("unnamed_")
        )
        if non_null_count < 10 or uniqueness_ratio < 0.9:
            return False
        return name_looks_like_id or (
            inferred_type == ColumnType.CATEGORICAL and uniqueness_ratio >= 0.98
        )

    @staticmethod
    def _columns_of_type(inferred_types: dict[str, ColumnType], expected: ColumnType) -> list[str]:
        """Filtra os nomes de colunas que correspondem ao tipo semântico esperado.

        Args:
            inferred_types: Mapeamento dos nomes de colunas para seus tipos inferidos.
            expected: Tipo semântico esperado para o filtro.

        Returns:
            Lista de nomes de colunas que coincidem com o tipo solicitado.
        """
        return [name for name, inferred_type in inferred_types.items() if inferred_type == expected]
