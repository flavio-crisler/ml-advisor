import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.domain.exceptions import PipelineCompilationError
from app.domain.experiment.spec import ExperimentSpec
from app.infrastructure.storage.base import ArtifactStorage
from app.ml.preprocessing.base import FeatureTransformer, RowTransformer
from app.ml.preprocessing.boolean import BooleanEncoder
from app.ml.preprocessing.categorical import CategoricalEncoder
from app.ml.preprocessing.datetime import DateTimeTransformer
from app.ml.preprocessing.feature_selection import BasicFeatureSelector
from app.ml.preprocessing.imbalance import ClassWeightResolver
from app.ml.preprocessing.missing import MissingRowDropper, MissingValueTransformer
from app.ml.preprocessing.numerical import NumericScaler, OutlierClipper
from app.ml.preprocessing.text import TextVectorizer
from app.ml.registries.metadata import ModelCapabilities


class CompiledPreprocessingPipeline:
    """Pipeline compilado contendo a sequência de transformações de pré-processamento.

    Attributes:
        original_features (tuple[str, ...]): Nomes das features originais esperadas na entrada.
        feature_transformers (list[FeatureTransformer]): Lista sequencial de transformadores de
            features.
        row_transformers (list[RowTransformer]): Lista de transformadores a nível de linhas
            (como remoção de ausentes).
        class_weights (ClassWeightResolver): Resolutor de balanceamento e pesos de classes.
        requested_strategies (dict[str, str]): Estratégias originalmente solicitadas pelo
            experimento.
        resolved_strategies (dict[str, str]): Estratégias ajustadas de acordo com as capacidades
            do modelo.
        capability_adjustments (list[dict[str, str]]): Histórico de ajustes de estratégias
            aplicados pelo compilador.
        capabilities (ModelCapabilities): Metadados de capacidades suportadas pelo modelo.
        output_features (tuple[str, ...]): Nomes das features geradas após o ajuste do pipeline.
    """

    def __init__(
        self,
        *,
        original_features: list[str],
        feature_transformers: list[FeatureTransformer],
        row_transformers: list[RowTransformer],
        class_weights: ClassWeightResolver,
        requested_strategies: Mapping[str, str],
        resolved_strategies: Mapping[str, str],
        capability_adjustments: list[Mapping[str, str]],
        capabilities: ModelCapabilities,
    ) -> None:
        """Inicializa o pipeline de pré-processamento compilado.

        Args:
            original_features: Lista de nomes das features originais da especificação.
            feature_transformers: Lista de transformadores de features.
            row_transformers: Lista de transformadores que modificam linhas/amostras.
            class_weights: Resolutor de balanceamento de classes.
            requested_strategies: Mapeamento de estratégias solicitadas pelo usuário/advisor.
            resolved_strategies: Mapeamento de estratégias finais após validação com o modelo.
            capability_adjustments: Registros de alterações nas estratégias demandadas pelo modelo.
            capabilities: Capacidades e restrições do modelo de machine learning alvo.
        """
        self.original_features = tuple(original_features)
        self.feature_transformers = feature_transformers
        self.row_transformers = row_transformers
        self.class_weights = class_weights
        self.requested_strategies = dict(requested_strategies)
        self.resolved_strategies = dict(resolved_strategies)
        self.capability_adjustments = [dict(adjustment) for adjustment in capability_adjustments]
        self.capabilities = capabilities
        self.output_features: tuple[str, ...] = ()
        self._fitted = False

    def fit_transform(
        self, features: pd.DataFrame, target: pd.Series | Any
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Ajusta todos os transformadores e aplica as transformações aos dados de treino e alvo.

        Args:
            features: DataFrame contendo as features de entrada.
            target: Série ou array contendo a variável alvo.

        Returns:
            Tupla contendo o DataFrame de features transformadas e a Series alvo processada.
        """
        transformed = self._select_features(features)
        transformed_target = (
            target.copy()
            if isinstance(target, pd.Series)
            else pd.Series(target, index=features.index, name="target")
        )
        for transformer in self.row_transformers:
            transformed, transformed_target = transformer.fit_resample(
                transformed, transformed_target
            )
        for transformer in self.feature_transformers:
            transformer.fit(transformed, transformed_target)
            transformed = transformer.transform(transformed)
        self.class_weights.fit(transformed_target)
        self.output_features = tuple(transformed.columns)
        self._fitted = True
        return transformed, transformed_target

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aplica as transformações aprendidas em novos dados de inferência/teste.

        Args:
            features: DataFrame com as features originais a serem transformadas.

        Returns:
            DataFrame com as features transformadas prontas para o modelo.

        Raises:
            RuntimeError: Se o pipeline ainda não tiver sido ajustado via fit_transform.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        transformed = self._select_features(features)
        for transformer in self.feature_transformers:
            transformed = transformer.transform(transformed)
        return transformed

    def get_model_options(self) -> Mapping[str, Any]:
        """Retorna parâmetros adicionais calculados pelo pipeline para serem passados ao modelo.

        Returns:
            Dicionário com opções de modelo, como class_weight.

        Raises:
            RuntimeError: Se o pipeline ainda não foi ajustado.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before model options are available")
        return self.class_weights.get_model_options()

    def get_config(self) -> dict[str, Any]:
        """Gera uma representação em dicionário de toda a configuração do pipeline compilado.

        Returns:
            Dicionário com detalhes dos transformadores, estratégias e metadados.
        """
        return {
            "original_features": list(self.original_features),
            "output_features": list(self.output_features),
            "requested_strategies": self.requested_strategies,
            "resolved_strategies": self.resolved_strategies,
            "capability_adjustments": self.capability_adjustments,
            "model_capabilities": self.capabilities.model_dump(mode="json"),
            "row_transformers": [
                dict(transformer.get_config()) for transformer in self.row_transformers
            ],
            "feature_transformers": [
                dict(transformer.get_config()) for transformer in self.feature_transformers
            ],
            "class_weighting": dict(self.class_weights.get_config()),
        }

    def persist_config(self, storage: ArtifactStorage, artifact_id: str) -> str:
        """Serializa em JSON e salva a configuração do pipeline compilado no storage.

        Args:
            storage: Instância do armazenamento de artefatos.
            artifact_id: Identificador único do artefato gerado.

        Returns:
            Caminho ou URI onde o artefato foi persistido.
        """
        content = json.dumps(
            self.get_config(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return storage.save(artifact_id, content)

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Valida a presença e filtra as colunas esperadas para o experimento.

        Args:
            features: DataFrame recebido para validação.

        Returns:
            Cópia do DataFrame contendo apenas as colunas das features originais.

        Raises:
            ValueError: Se alguma feature obrigatória estiver ausente no DataFrame.
        """
        missing = set(self.original_features) - set(features.columns)
        if missing:
            raise ValueError(f"Input is missing experiment features: {sorted(missing)}")
        return features.loc[:, self.original_features].copy()


class PipelineCompiler:
    """Compilador de especificações de experimentos em pipelines de pré-processamento executáveis.

    Attributes:
        _known_steps (set[str]): Conjunto de identificadores de etapas válidas de pré-processamento.
    """

    _known_steps = {
        "missing_values",
        "categorical_encoding",
        "boolean_encoding",
        "numeric_scaling",
        "datetime",
        "outliers",
        "imbalance",
        "feature_selection",
        "text_vectorization",
    }

    def compile(
        self,
        spec: ExperimentSpec,
        *,
        column_types: Mapping[str, str],
        capabilities: ModelCapabilities,
    ) -> CompiledPreprocessingPipeline:
        """Compila uma especificação de experimento em um pipeline de pré-processamento.

        Valida se os passos, colunas e tipos são compatíveis com os requisitos e capacidades
        do modelo selecionado, aplicando ajustes automáticos quando aplicável.

        Args:
            spec: Especificação do experimento com etapas de pré-processamento e features.
            column_types: Mapeamento dos nomes das colunas para seus tipos semânticos brutos.
            capabilities: Capacidades e limitações suportadas pelo modelo de machine learning.

        Returns:
            Instância de CompiledPreprocessingPipeline configurada e pronta para ajuste.

        Raises:
            PipelineCompilationError: Se houver etapas desconhecidas, colunas sem tipo mapeado,
                tipos de features não suportados ou parâmetros de vetorização inconsistentes.
        """
        steps = {step.id: step for step in spec.preprocessing}
        unknown = set(steps) - self._known_steps
        if unknown:
            raise PipelineCompilationError(f"Unknown preprocessing components: {sorted(unknown)}")
        missing_types = set(spec.features) - set(column_types)
        if missing_types:
            raise PipelineCompilationError(
                f"Column types are missing for features: {sorted(missing_types)}"
            )
        supported_types = {"numeric", "categorical", "boolean", "datetime", "text"}
        unsupported_types = [
            f"{column}:{column_types[column]}"
            for column in spec.features
            if str(column_types[column]).lower() not in supported_types
        ]
        if unsupported_types:
            raise PipelineCompilationError(
                "Unsupported raw feature types: "
                f"{unsupported_types}. Remove or convert these columns before training."
            )

        text_strategy = self._strategy(steps, "text_vectorization", "none")
        text_parameters = self._parameters(steps, "text_vectorization")
        explicit_text_columns = text_parameters.get("columns", [])
        if (
            not isinstance(explicit_text_columns, list)
            or any(not isinstance(column, str) for column in explicit_text_columns)
            or len(explicit_text_columns) != len(set(explicit_text_columns))
        ):
            raise PipelineCompilationError(
                "text_vectorization:columns must be a list of distinct column names"
            )
        invalid_overrides = [
            column
            for column in explicit_text_columns
            if column not in spec.features
            or str(column_types[column]).lower() not in {"text", "categorical"}
        ]
        if invalid_overrides:
            raise PipelineCompilationError(
                "text_vectorization:columns must be selected text or categorical features: "
                f"{invalid_overrides}"
            )
        text_columns = list(
            dict.fromkeys([*self._columns(spec, column_types, "text"), *explicit_text_columns])
        )
        if text_columns and text_strategy != "tfidf":
            raise PipelineCompilationError(
                f"Text features {text_columns} require text_vectorization:tfidf"
            )
        max_features = text_parameters.get("max_features", 256)
        if (
            isinstance(max_features, bool)
            or not isinstance(max_features, int)
            or not 1 <= max_features <= 2048
        ):
            raise PipelineCompilationError(
                "text_vectorization:max_features must be an integer from 1 to 2048"
            )
        if len(text_columns) * max_features > 2048:
            raise PipelineCompilationError(
                "text_vectorization:total_features exceeds 2048; "
                "reduce max_features or text columns"
            )

        requested_strategies = {
            "missing_values": self._strategy(steps, "missing_values", "median"),
            "categorical_encoding": self._strategy(steps, "categorical_encoding", "one_hot"),
            "boolean_encoding": self._strategy(steps, "boolean_encoding", "binary"),
            "numeric_scaling": self._strategy(steps, "numeric_scaling", "passthrough"),
            "datetime": self._strategy(steps, "datetime", "components"),
            "outliers": self._strategy(steps, "outliers", "none"),
            "imbalance": self._strategy(steps, "imbalance", "none"),
            "feature_selection": self._strategy(steps, "feature_selection", "none"),
            "text_vectorization": text_strategy,
        }
        strategies = dict(requested_strategies)
        capability_adjustments: list[dict[str, str]] = []
        if capabilities.requires_scaling and strategies["numeric_scaling"] == "passthrough":
            strategies["numeric_scaling"] = "standard"
            capability_adjustments.append(
                {
                    "component": "numeric_scaling",
                    "requested": "passthrough",
                    "resolved": "standard",
                    "reason": "model_requires_scaling",
                }
            )
        if (
            strategies["categorical_encoding"] == "passthrough"
            and any(
                str(column_types[column]).lower() == "categorical" and column not in text_columns
                for column in spec.features
            )
            and not capabilities.supports_categorical
        ):
            raise PipelineCompilationError(
                "Categorical passthrough requires supports_categorical=true"
            )
        if (
            strategies["imbalance"] == "class_weight"
            and not capabilities.supports_imbalance_weighting
        ):
            strategies["imbalance"] = "none"
            capability_adjustments.append(
                {
                    "component": "imbalance",
                    "requested": "class_weight",
                    "resolved": "none",
                    "reason": "model_does_not_support_imbalance_weighting",
                }
            )

        numeric = self._columns(spec, column_types, "numeric")
        categorical = [
            column
            for column in self._columns(spec, column_types, "categorical")
            if column not in text_columns
        ]
        boolean = self._columns(spec, column_types, "boolean")
        datetime = self._columns(spec, column_types, "datetime")
        missing_parameters = self._parameters(steps, "missing_values")
        outlier_parameters = self._parameters(steps, "outliers")
        selection_parameters = self._parameters(steps, "feature_selection")

        row_transformers: list[RowTransformer] = []
        if strategies["missing_values"] == "drop_rows":
            row_transformers.append(MissingRowDropper())
        feature_transformers: list[FeatureTransformer] = [
            MissingValueTransformer(
                strategies["missing_values"],
                constant_value=missing_parameters.get("constant_value", "__missing__"),
                text_columns=tuple(text_columns),
            ),
            DateTimeTransformer(datetime, strategies["datetime"]),
            BooleanEncoder(boolean, strategies["boolean_encoding"]),
            OutlierClipper(
                numeric,
                strategies["outliers"],
                lower_quantile=float(outlier_parameters.get("lower_quantile", 0.01)),
                upper_quantile=float(outlier_parameters.get("upper_quantile", 0.99)),
                iqr_multiplier=float(outlier_parameters.get("iqr_multiplier", 1.5)),
            ),
            NumericScaler(numeric, strategies["numeric_scaling"]),
            CategoricalEncoder(categorical, strategies["categorical_encoding"]),
            TextVectorizer(text_columns, max_features=max_features),
            BasicFeatureSelector(
                strategies["feature_selection"],
                threshold=float(selection_parameters.get("threshold", 0.0)),
            ),
        ]
        return CompiledPreprocessingPipeline(
            original_features=spec.features,
            feature_transformers=feature_transformers,
            row_transformers=row_transformers,
            class_weights=ClassWeightResolver(strategies["imbalance"]),
            requested_strategies=requested_strategies,
            resolved_strategies=strategies,
            capability_adjustments=capability_adjustments,
            capabilities=capabilities,
        )

    @staticmethod
    def _strategy(steps: Mapping[str, Any], step_id: str, default: str) -> str:
        """Obtém o nome da estratégia configurada para uma etapa ou o valor padrão.

        Args:
            steps: Mapeamento de passos de pré-processamento por identificador.
            step_id: Identificador da etapa a ser consultada.
            default: Estratégia padrão a retornar caso a etapa não esteja presente.

        Returns:
            Nome da estratégia a ser utilizada.
        """
        return steps[step_id].strategy if step_id in steps else default

    @staticmethod
    def _parameters(steps: Mapping[str, Any], step_id: str) -> Mapping[str, Any]:
        """Obtém os parâmetros adicionais configurados para uma etapa de pré-processamento.

        Args:
            steps: Mapeamento de passos de pré-processamento por identificador.
            step_id: Identificador da etapa a ser consultada.

        Returns:
            Dicionário com os parâmetros configurados ou dicionário vazio se ausente.
        """
        return steps[step_id].parameters if step_id in steps else {}

    @staticmethod
    def _columns(
        spec: ExperimentSpec, column_types: Mapping[str, str], expected_type: str
    ) -> list[str]:
        """Filtra as features da especificação que correspondem a um tipo semântico esperado.

        Args:
            spec: Especificação do experimento.
            column_types: Mapeamento das colunas com seus respectivos tipos.
            expected_type: Tipo semântico esperado (ex: 'numeric', 'categorical').

        Returns:
            Lista de nomes de colunas que correspondem ao tipo requisitado.
        """
        return [
            column for column in spec.features if str(column_types[column]).lower() == expected_type
        ]
