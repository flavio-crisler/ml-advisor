import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from app.domain.experiment.repositories import ExperimentRepository
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    PreprocessingStepSpec,
    TaskType,
)
from app.ml.analytics.generator import ResultArtifactGenerator
from app.ml.metrics.evaluator import MetricEvaluator
from app.ml.preprocessing.compiler import PipelineCompiler
from app.ml.preprocessing.semantic import SemanticFeatureTransformer
from app.ml.registries.catalog import RegistryCatalog
from app.ml.validation.splitters import ValidationSplitter

logger = logging.getLogger(__name__)


class SemanticImpactEvaluator:
    """Avalia o impacto preditivo da adição de features semânticas em experimentos.

    Attributes:
        experiments (ExperimentRepository): Repositório para consulta de experimentos.
        catalog (RegistryCatalog): Catálogo de adaptadores de modelo e métricas.
        column_types (dict[str, str]): Mapeamento de tipos inferidos das colunas.
        splitter (ValidationSplitter): Utilitário para divisão dos folds de validação.
        compiler (PipelineCompiler): Compilador de pipelines de pré-processamento.
        metrics (MetricEvaluator): Avaliador de métricas estatísticas e de desempenho.
        artifacts (ResultArtifactGenerator): Gerador de artefatos de análise e importância.
    """

    def __init__(
        self,
        *,
        experiments: ExperimentRepository,
        catalog: RegistryCatalog,
        column_types: dict[str, str],
    ) -> None:
        """Inicializa o avaliador de impacto semântico.

        Args:
            experiments (ExperimentRepository): Repositório de experimentos.
            catalog (RegistryCatalog): Catálogo de componentes de ML.
            column_types (dict[str, str]): Tipos inferidos das colunas do dataset.
        """
        self.experiments = experiments
        self.catalog = catalog
        self.column_types = column_types
        self.splitter = ValidationSplitter()
        self.compiler = PipelineCompiler()
        self.metrics = MetricEvaluator(catalog.metrics)
        self.artifacts = ResultArtifactGenerator(max_explanation_rows=16, max_background_rows=32)

    def evaluate(
        self,
        *,
        experiment_id: str,
        source_frame: pd.DataFrame,
        semantic_frame: pd.DataFrame,
        semantic_columns: list[str],
        text_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Avalia o impacto comparativo de features semânticas em relação ao baseline estruturado.

        Args:
            experiment_id (str): Identificador do experimento base de referência.
            source_frame (pd.DataFrame): DataFrame original contendo os dados brutos.
            semantic_frame (pd.DataFrame): DataFrame com features semânticas extraídas.
            semantic_columns (list[str]): Nomes das features semânticas a incorporar.
            text_columns (list[str] | None): Colunas de texto opcionais para baseline TF-IDF.

        Returns:
            dict[str, Any]: Dicionário com métricas comparativas e metadados da validação.

        Raises:
            ValueError: Se o experimento não for encontrado, tiver tuning ativo ou
                dados inconsistentes.
        """
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(f"Source experiment {experiment_id} was not found")
        spec = ExperimentSpec.model_validate(experiment.spec)
        if spec.tuning.enabled:
            raise ValueError(
                "Semantic impact requires a source experiment with tuning disabled so both "
                "arms use the same fixed budget"
            )
        if experiment.dataset_id != spec.dataset_id:
            raise ValueError("Source experiment dataset is inconsistent")
        transformer = SemanticFeatureTransformer(semantic_frame)
        augmented_frame = transformer.transform(source_frame)
        metadata = transformer.get_metadata()
        augmented_roles = dict(spec.column_roles)
        augmented_roles.update({name: ColumnRole.FEATURE for name in semantic_columns})
        augmented_spec = spec.model_copy(
            update={
                "features": [*spec.features, *semantic_columns],
                "column_roles": augmented_roles,
            }
        )
        augmented_types = {**self.column_types, **metadata.column_types}
        eligible_text_columns = list(
            dict.fromkeys(
                column
                for column in (text_columns or [])
                if column in source_frame
                and self.column_types.get(column) == "text"
                and column not in spec.features
                and column != spec.target
            )
        )
        text_spec = None
        if eligible_text_columns and len(eligible_text_columns) <= 128:
            text_roles = dict(spec.column_roles)
            text_roles.update({name: ColumnRole.FEATURE for name in eligible_text_columns})
            text_spec = spec.model_copy(
                update={
                    "features": [*spec.features, *eligible_text_columns],
                    "column_roles": text_roles,
                    "preprocessing": [
                        *[step for step in spec.preprocessing if step.id != "text_vectorization"],
                        PreprocessingStepSpec(
                            id="text_vectorization",
                            strategy="tfidf",
                            parameters={
                                "max_features": min(256, 2048 // len(eligible_text_columns))
                            },
                        ),
                    ],
                }
            )
        target = source_frame[spec.target]
        validation = self.splitter.split(spec.validation, spec.task, target)
        validation_payload = validation.to_dict()
        models: list[dict[str, Any]] = []
        for model_spec in [model for model in spec.models if model.enabled]:
            started = time.perf_counter()
            baseline = self._evaluate_model(
                model_id=model_spec.id,
                parameters=model_spec.parameters,
                spec=spec,
                features=source_frame.loc[:, spec.features],
                target=target,
                column_types=self.column_types,
                validation=validation,
            )
            baseline_duration = time.perf_counter() - started
            augmented_started = time.perf_counter()
            augmented = self._evaluate_model(
                model_id=model_spec.id,
                parameters=model_spec.parameters,
                spec=augmented_spec,
                features=augmented_frame.loc[:, augmented_spec.features],
                target=target,
                column_types=augmented_types,
                validation=validation,
                explain=True,
                semantic_columns=set(semantic_columns),
            )
            augmented_duration = time.perf_counter() - augmented_started
            text_started = time.perf_counter()
            text_baseline = None
            text_baseline_status = "not_applicable"
            if text_spec is not None:
                try:
                    text_baseline = self._evaluate_model(
                        model_id=model_spec.id,
                        parameters=model_spec.parameters,
                        spec=text_spec,
                        features=source_frame.loc[:, text_spec.features],
                        target=target,
                        column_types=self.column_types,
                        validation=validation,
                    )
                    text_baseline_status = "available"
                except Exception:
                    logger.exception(
                        "Optional local text baseline failed for model %s", model_spec.id
                    )
                    text_baseline_status = "unavailable"
            elif eligible_text_columns:
                text_baseline_status = "too_many_text_columns"
            text_duration = time.perf_counter() - text_started if text_baseline else None
            primary = spec.metrics.primary
            baseline_value = baseline["metrics"][primary]
            augmented_value = augmented["metrics"][primary]
            models.append(
                {
                    "model_id": model_spec.id,
                    "parameters": model_spec.parameters,
                    "baseline_metrics": baseline["metrics"],
                    "augmented_metrics": augmented["metrics"],
                    "text_baseline_metrics": text_baseline["metrics"] if text_baseline else None,
                    "text_baseline_status": text_baseline_status,
                    "delta_semantic_vs_text": (
                        augmented_value - text_baseline["metrics"][primary]
                        if text_baseline
                        else None
                    ),
                    "primary_metric": primary,
                    "metric_direction": self.catalog.metrics.get(primary).direction,
                    "delta_absolute": augmented_value - baseline_value,
                    "delta_percent": (
                        (augmented_value - baseline_value) / abs(baseline_value) * 100
                        if baseline_value
                        else None
                    ),
                    "semantic_feature_importance": augmented["semantic_feature_importance"],
                    "baseline_duration_seconds": baseline_duration,
                    "text_baseline_duration_seconds": text_duration,
                    "augmented_duration_seconds": augmented_duration,
                    "duration_seconds": time.perf_counter() - started,
                }
            )
        return {
            "schema_version": "1.1",
            "source_experiment_id": experiment_id,
            "comparison_contract": {
                "same_dataset": True,
                "same_validation": True,
                "same_random_seed": True,
                "same_models": True,
                "same_parameters": True,
                "same_tuning_budget": True,
                "tuning_trials_per_arm": 0,
                "only_difference": "semantic_features",
                "arms": [
                    "structured",
                    *(
                        ["structured_tfidf"]
                        if any(model["text_baseline_metrics"] is not None for model in models)
                        else []
                    ),
                    "structured_semantic",
                ],
            },
            "validation": validation_payload,
            "semantic_features": semantic_columns,
            "text_columns": eligible_text_columns,
            "models": models,
        }

    def _evaluate_model(
        self,
        *,
        model_id: str,
        parameters: dict[str, Any],
        spec: ExperimentSpec,
        features: pd.DataFrame,
        target: pd.Series,
        column_types: dict[str, str],
        validation: Any,
        explain: bool = False,
        semantic_columns: set[str] | None = None,
    ) -> dict[str, Any]:
        """Treina e avalia um modelo sob folds de validação cruzada para um braço experimental.

        Args:
            model_id (str): Identificador do modelo no catálogo.
            parameters (dict[str, Any]): Hiperparâmetros do modelo.
            spec (ExperimentSpec): Especificação do experimento.
            features (pd.DataFrame): DataFrame de variáveis preditoras.
            target (pd.Series): Série com os rótulos ou valores alvo.
            column_types (dict[str, str]): Mapeamento de tipos das colunas.
            validation (Any): Estrutura de validação contendo os folds.
            explain (bool): Se True, calcula a importância de features semânticas.
            semantic_columns (set[str] | None): Conjunto de colunas semânticas de interesse.

        Returns:
            dict[str, Any]: Dicionário com métricas agregadas e importâncias calculadas.
        """
        evaluations = []
        for fold in validation.folds:
            adapter = self.catalog.model_adapters.create(model_id, spec.task, parameters)
            pipeline = self.compiler.compile(
                spec,
                column_types=column_types,
                capabilities=adapter.get_capabilities(),
            )
            train_features, train_target = pipeline.fit_transform(
                features.iloc[list(fold.train_indices)], target.iloc[list(fold.train_indices)]
            )
            test_features = pipeline.transform(features.iloc[list(fold.test_indices)])
            test_target = target.iloc[list(fold.test_indices)]
            adapter.fit(train_features, train_target, fit_options=pipeline.get_model_options())
            predictions = adapter.predict(test_features)
            metric_ids = [spec.metrics.primary, *spec.metrics.secondary]
            needs_probability = any(
                self.catalog.metrics.get(metric_id).requires_proba for metric_id in metric_ids
            )
            probabilities = adapter.predict_proba(test_features) if needs_probability else None
            classes = adapter.get_classes() if needs_probability else None
            evaluations.append(
                self.metrics.evaluate(
                    task=spec.task,
                    specification=spec.metrics,
                    target=test_target,
                    predictions=predictions,
                    probabilities=probabilities,
                    probability_classes=classes,
                )
            )
        values, _ = self.metrics.aggregate(evaluations)
        importance: list[dict[str, Any]] = []
        if explain:
            adapter = self.catalog.model_adapters.create(model_id, spec.task, parameters)
            pipeline = self.compiler.compile(
                spec,
                column_types=column_types,
                capabilities=adapter.get_capabilities(),
            )
            fitted_features, fitted_target = pipeline.fit_transform(features, target)
            adapter.fit(fitted_features, fitted_target, fit_options=pipeline.get_model_options())
            predictions = adapter.predict(fitted_features)
            probabilities = (
                adapter.predict_proba(fitted_features) if spec.task != TaskType.REGRESSION else None
            )
            classes = adapter.get_classes() if probabilities is not None else None
            analysis = self.artifacts.build_run(
                task=spec.task,
                target=np.asarray(fitted_target).tolist(),
                predictions=np.asarray(predictions).tolist(),
                probabilities=(
                    np.asarray(probabilities, dtype=float).tolist()
                    if probabilities is not None
                    else None
                ),
                probability_classes=classes,
                positive_label=spec.metrics.positive_label,
                fitted_features=fitted_features,
                model=adapter,
                random_seed=spec.validation.random_seed,
                learning_curve=[],
            )
            all_importance = analysis["explainability"].get("feature_importance", [])
            semantic_columns = semantic_columns or set()
            importance = [
                {**item, "kind": "semantic"}
                for item in all_importance
                if item["feature"] in semantic_columns
            ]
        return {"metrics": values, "semantic_feature_importance": importance}
