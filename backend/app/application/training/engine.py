import hashlib
import io
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.application.datasets.profiling import DatasetProfile
from app.application.experiments.validation import ExperimentSpecValidator
from app.domain.common import new_id, sanitize_for_json
from app.domain.dataset.repositories import DatasetRepository
from app.domain.exceptions import EntityNotFoundError, ExperimentExecutionError
from app.domain.experiment.entities import (
    ExperimentStatus,
    ModelRun,
    ModelRunStatus,
    TuningTrial,
    TuningTrialStatus,
)
from app.domain.experiment.repositories import (
    ExperimentRepository,
    ModelRunRepository,
    TuningTrialRepository,
)
from app.domain.experiment.spec import ExperimentSpec, ModelSpec, TaskType
from app.infrastructure.storage.base import ArtifactStorage
from app.ml.analytics.generator import ResultArtifactGenerator
from app.ml.metrics.evaluator import MetricEvaluation, MetricEvaluator
from app.ml.metrics.leaderboard import LeaderboardBuilder, LeaderboardEntry
from app.ml.preprocessing.compiler import PipelineCompiler
from app.ml.registries.catalog import RegistryCatalog
from app.ml.tuning.base import TuningTrialPruned, TuningTrialResult
from app.ml.validation.splitters import ValidationFold, ValidationPlan, ValidationSplitter


@dataclass(frozen=True, slots=True)
class ExperimentExecutionResult:
    """Resultado consolidado da execução de uma rodada completa de experimento.

    Attributes:
        experiment_id (str): Identificador do experimento executado.
        execution_id (str): Identificador único da rodada de execução gerada.
        runs (tuple[ModelRun, ...]): Coleção de execuções de modelos concluídas.
        leaderboard_path (str): Caminho no storage para o artefato JSON do leaderboard.
    """

    experiment_id: str
    execution_id: str
    runs: tuple[ModelRun, ...]
    leaderboard_path: str


@dataclass(frozen=True, slots=True)
class EvaluationObservations:
    """Observações agregadas da validação cruzada para explicabilidade e predições pareadas.

    Attributes:
        row_ids (tuple[str, ...]): Identificadores únicos de cada linha amostrada.
        fold_indices (tuple[int, ...]): Índice do fold de validação de cada observação.
        target (tuple[Any, ...]): Rótulos ou valores reais observados.
        predictions (tuple[Any, ...]): Previsões de classes ou valores contínuos gerados.
        probabilities (tuple[tuple[float, ...], ...] | None): Probabilidades por classe.
        probability_classes (tuple[Any, ...] | None): Ordem das classes das probabilidades.
    """

    row_ids: tuple[str, ...]
    fold_indices: tuple[int, ...]
    target: tuple[Any, ...]
    predictions: tuple[Any, ...]
    probabilities: tuple[tuple[float, ...], ...] | None
    probability_classes: tuple[Any, ...] | None


class ExperimentEngine:
    """Motor central de treinamento, validação, tuning e persistência de artefatos de ML.

    Attributes:
        datasets (DatasetRepository): Repositório para consulta de datasets.
        experiments (ExperimentRepository): Repositório para controle de experimentos.
        runs (ModelRunRepository): Repositório para registro dos runs de modelos.
        tuning_trials (TuningTrialRepository): Repositório para tentativas de tuning.
        dataset_storage (ArtifactStorage): Storage onde os arquivos de datasets estão salvos.
        artifact_storage (ArtifactStorage): Storage onde os artefatos de treino são gerados.
        catalog (RegistryCatalog): Catálogo de modelos, pré-processamento e métricas.
        spec_validator (ExperimentSpecValidator): Validador semântico das especificações.
        splitter (ValidationSplitter): Gerador das partições e folds de validação.
        pipeline_compiler (PipelineCompiler): Compilador dos pipelines de pré-processamento.
        metric_evaluator (MetricEvaluator): Avaliador de métricas de desempenho.
        leaderboard_builder (LeaderboardBuilder): Construtor de tabelas de ranqueamento.
        artifact_generator (ResultArtifactGenerator): Gerador de artefatos analíticos e SHAP.
    """

    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        experiments: ExperimentRepository,
        runs: ModelRunRepository,
        tuning_trials: TuningTrialRepository,
        dataset_storage: ArtifactStorage,
        artifact_storage: ArtifactStorage,
        catalog: RegistryCatalog,
    ) -> None:
        """Inicializa o motor de experimentos com dependências de persistência e catálogos.

        Args:
            datasets: Repositório de datasets.
            experiments: Repositório de experimentos.
            runs: Repositório de runs de modelos.
            tuning_trials: Repositório de histórico de trials.
            dataset_storage: Storage de arquivos originais.
            artifact_storage: Storage de artefatos gerados.
            catalog: Catálogo de componentes e algoritmos de ML.
        """
        self.datasets = datasets
        self.experiments = experiments
        self.runs = runs
        self.tuning_trials = tuning_trials
        self.dataset_storage = dataset_storage
        self.artifact_storage = artifact_storage
        self.catalog = catalog
        self.spec_validator = ExperimentSpecValidator(catalog)
        self.splitter = ValidationSplitter()
        self.pipeline_compiler = PipelineCompiler()
        self.metric_evaluator = MetricEvaluator(catalog.metrics)
        self.leaderboard_builder = LeaderboardBuilder(catalog.metrics)
        self.artifact_generator = ResultArtifactGenerator()

    def execute(
        self,
        experiment_id: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExperimentExecutionResult:
        """Executa a rodada completa de treinamento, validação e geração de artefatos.

        Coordena a divisão dos dados, compilação dos pipelines, treinamento e avaliação de
        cada modelo habilitado, otimização de hiperparâmetros (se ativa), geração de gráficos
        Plotly, cálculos de SHAP e consolidação do leaderboard.

        Args:
            experiment_id: Identificador do experimento a ser executado.
            on_event: Callback opcional para receber eventos de progresso em tempo real.

        Returns:
            ExperimentExecutionResult com o resumo da execução e caminhos de artefatos.

        Raises:
            EntityNotFoundError: Se o experimento não for localizado no repositório.
            ExperimentExecutionError: Se houver inconsistências de hash ou dados nulos no alvo.
        """
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
        specification = self.spec_validator.validate(ExperimentSpec.model_validate(experiment.spec))
        dataset, profile, frame = self._load_dataset(specification)
        execution_id = new_id()
        prefix = f"{experiment.id}/executions/{execution_id}"
        completed_runs: list[ModelRun] = []
        current_run: ModelRun | None = None

        self.experiments.update_status(experiment.id, ExperimentStatus.RUNNING)
        self._emit(
            on_event,
            "training.started",
            experiment_id=experiment.id,
            progress=0,
            message_key="training.started",
        )
        try:
            target = frame[specification.target]
            if target.isna().any():
                raise ExperimentExecutionError(
                    "Target contains missing values and cannot be trained safely"
                )
            validation = self.splitter.split(specification.validation, specification.task, target)
            self._save_json(f"{prefix}/validation.json", validation.to_dict())
            self._emit(
                on_event,
                "validation.completed",
                experiment_id=experiment.id,
                progress=15,
                message_key="training.validation.completed",
            )
            self._save_json(
                f"{prefix}/experiment.json",
                {
                    "experiment_id": experiment.id,
                    "execution_id": execution_id,
                    "dataset_id": dataset.id,
                    "dataset_hash": dataset.content_hash,
                    "spec": specification.model_dump(mode="json"),
                    "runtime": self._runtime_versions(),
                },
            )

            column_types = {
                column.name: column.inferred_type.value for column in profile.column_profiles
            }
            enabled_models = [model for model in specification.models if model.enabled]
            for model_index, model_specification in enumerate(enabled_models):
                adapter = self.catalog.model_adapters.create(
                    model_specification.id,
                    specification.task,
                    model_specification.parameters,
                )
                current_run = self.runs.create(
                    experiment_id=experiment.id,
                    model_id=model_specification.id,
                    random_seed=specification.validation.random_seed,
                    parameters=adapter.get_params(),
                    execution_id=execution_id,
                )
                self.runs.transition(current_run.id, ModelRunStatus.RUNNING)
                completed_runs.append(
                    self._execute_model(
                        experiment_id=experiment.id,
                        execution_prefix=prefix,
                        run=current_run,
                        model_specification=model_specification,
                        specification=specification,
                        frame=frame,
                        validation=validation,
                        column_types=column_types,
                        on_event=on_event,
                        model_index=model_index,
                        model_count=len(enabled_models),
                    )
                )
                current_run = None

            leaderboard = self.leaderboard_builder.build(
                specification.metrics.primary,
                [
                    LeaderboardEntry(
                        run_id=run.id,
                        model_id=run.model_id,
                        metrics=self._metric_values(run),
                        duration_seconds=run.duration_seconds or 0.0,
                    )
                    for run in completed_runs
                ],
            )
            comparison_path = self._save_json(
                f"{prefix}/comparison.json",
                self.artifact_generator.build_comparison(
                    primary_metric=specification.metrics.primary,
                    runs=[
                        {
                            "run_id": entry.run_id,
                            "model_id": entry.model_id,
                            "metrics": entry.metrics,
                        }
                        for entry in leaderboard.entries
                    ],
                ),
            )
            leaderboard_path = self._save_json(
                f"{prefix}/leaderboard.json",
                {
                    "primary_metric": leaderboard.primary_metric,
                    "direction": leaderboard.direction,
                    "comparison_path": comparison_path,
                    "entries": [
                        {
                            "run_id": entry.run_id,
                            "model_id": entry.model_id,
                            "metrics": entry.metrics,
                            "duration_seconds": entry.duration_seconds,
                        }
                        for entry in leaderboard.entries
                    ],
                },
            )
            self.experiments.update_status(experiment.id, ExperimentStatus.COMPLETED)
            self._emit(
                on_event,
                "training.completed",
                experiment_id=experiment.id,
                progress=100,
                message_key="training.completed",
            )
            return ExperimentExecutionResult(
                experiment_id=experiment.id,
                execution_id=execution_id,
                runs=tuple(completed_runs),
                leaderboard_path=leaderboard_path,
            )
        except Exception as error:
            if current_run is not None:
                self.runs.transition(
                    current_run.id,
                    ModelRunStatus.FAILED,
                    error_message=f"{type(error).__name__}: {error}",
                )
            self.experiments.update_status(experiment.id, ExperimentStatus.FAILED)
            raise

    def _execute_model(
        self,
        *,
        experiment_id: str,
        execution_prefix: str,
        run: ModelRun,
        model_specification: ModelSpec,
        specification: ExperimentSpec,
        frame: pd.DataFrame,
        validation: ValidationPlan,
        column_types: dict[str, str],
        on_event: Callable[[dict[str, Any]], None] | None,
        model_index: int,
        model_count: int,
    ) -> ModelRun:
        """Executa o ciclo completo de treinamento, tuning e avaliação de um único modelo.

        Args:
            experiment_id: Identificador do experimento pai.
            execution_prefix: Prefixo no storage para salvar artefatos da rodada.
            run: Entidade ModelRun inicializada para o modelo.
            model_specification: Especificação dos parâmetros e algoritmo do modelo.
            specification: Especificação técnica global do experimento.
            frame: DataFrame completo carregado com features e target.
            validation: Plano de particionamento em folds de validação.
            column_types: Mapeamento de tipos semânticos brutos das colunas.
            on_event: Callback de emissão de eventos de progresso.
            model_index: Índice ordinal do modelo na lista de habilitados.
            model_count: Quantidade total de modelos habilitados a serem executados.

        Returns:
            ModelRun concluído contendo métricas calculadas e caminhos de artefatos.

        Raises:
            ExperimentExecutionError: Se o tuning estiver ativo sem métrica de otimização.
        """
        features = frame.loc[:, specification.features]
        target = frame[specification.target]
        model_progress_start = 15 + int(75 * model_index / model_count)
        model_progress_end = 15 + int(75 * (model_index + 1) / model_count)
        self._emit(
            on_event,
            "model.started",
            experiment_id=experiment_id,
            run_id=run.id,
            model=model_specification.id,
            progress=model_progress_start,
            message_key="training.model.started",
        )

        effective_parameters = dict(model_specification.parameters)
        tuning_payload: dict[str, Any] | None = None
        if specification.tuning.enabled:
            tuning_metric = specification.tuning.metric
            if tuning_metric is None:
                raise ExperimentExecutionError("Tuning requires an optimization metric")
            search_adapter = self.catalog.model_adapters.create(
                model_specification.id,
                specification.task,
                model_specification.parameters,
            )
            tuner = self.catalog.tuner_adapters.create(specification.tuning.engine)
            direction = self.catalog.metrics.get(tuning_metric).direction

            def objective(trial: Any, parameters: dict[str, Any]) -> float:
                observed_values: list[float] = []

                def report_fold(
                    _fold_index: int,
                    _fold_count: int,
                    evaluation: MetricEvaluation,
                ) -> None:
                    observed_values.append(evaluation.values[tuning_metric])
                    trial.report(
                        sum(observed_values) / len(observed_values),
                        step=len(observed_values) - 1,
                    )
                    if trial.should_prune():
                        raise TuningTrialPruned(
                            f"Trial {trial.number} was pruned after {len(observed_values)} folds"
                        )

                evaluations, _, _ = self._evaluate_parameters(
                    model_specification=model_specification,
                    parameters=parameters,
                    specification=specification,
                    features=features,
                    target=target,
                    validation=validation,
                    column_types=column_types,
                    on_fold=report_fold,
                )
                values, _ = self.metric_evaluator.aggregate(evaluations)
                return values[tuning_metric]

            def persist_trial(result: TuningTrialResult) -> None:
                status = {
                    "completed": TuningTrialStatus.COMPLETED,
                    "pruned": TuningTrialStatus.PRUNED,
                    "failed": TuningTrialStatus.FAILED,
                }[result.status]
                self.tuning_trials.add(
                    TuningTrial.create(
                        run_id=run.id,
                        trial_number=result.number,
                        parameters=result.parameters,
                        status=status,
                        metric_value=result.value,
                    )
                )
                self._emit(
                    on_event,
                    "tuning.progress",
                    experiment_id=experiment_id,
                    run_id=run.id,
                    model=model_specification.id,
                    trial=result.number + 1,
                    trials=specification.tuning.trials,
                    status=result.status,
                    value=result.value,
                    progress=model_progress_start
                    + int(
                        (model_progress_end - model_progress_start)
                        * 0.5
                        * (result.number + 1)
                        / specification.tuning.trials
                    ),
                    message_key="training.tuning.in_progress",
                )

            tuning_result = tuner.optimize(
                search_space=search_adapter.get_search_space,
                objective=objective,
                direction=direction,
                trials=specification.tuning.trials,
                random_seed=specification.validation.random_seed,
                fixed_parameters=model_specification.parameters,
                on_trial=persist_trial,
            )
            effective_parameters = tuning_result.best_parameters
            tuning_payload = {
                "engine": specification.tuning.engine,
                "metric": tuning_metric,
                "direction": direction,
                "best_value": tuning_result.best_value,
                "best_parameters": tuning_result.best_parameters,
                "trials": len(tuning_result.trials),
            }

        def report_training_fold(
            fold_index: int,
            fold_count: int,
            _evaluation: MetricEvaluation,
        ) -> None:
            self._emit(
                on_event,
                "training.progress",
                experiment_id=experiment_id,
                run_id=run.id,
                model=model_specification.id,
                progress=model_progress_start
                + int((model_progress_end - model_progress_start) * (fold_index + 1) / fold_count),
                message_key="training.model.in_progress",
                fold=fold_index + 1,
                folds=fold_count,
            )

        fold_evaluations, fold_payloads, observations = self._evaluate_parameters(
            model_specification=model_specification,
            parameters=effective_parameters,
            specification=specification,
            features=features,
            target=target,
            validation=validation,
            column_types=column_types,
            on_fold=report_training_fold,
            collect_observations=True,
        )

        metric_values, metric_deviations = self.metric_evaluator.aggregate(fold_evaluations)
        final_adapter = self.catalog.model_adapters.create(
            model_specification.id,
            specification.task,
            effective_parameters,
        )
        final_pipeline = self.pipeline_compiler.compile(
            specification,
            column_types=column_types,
            capabilities=final_adapter.get_capabilities(),
        )
        final_features, final_target = final_pipeline.fit_transform(features, target)
        final_adapter.fit(
            final_features,
            final_target,
            fit_options=final_pipeline.get_model_options(),
        )

        learning_curve = self._build_learning_curve(
            model_specification=model_specification,
            parameters=effective_parameters,
            specification=specification,
            features=features,
            target=target,
            validation=validation,
            column_types=column_types,
        )

        run_prefix = f"{execution_prefix}/runs/{run.id}"
        analysis_path = self._save_json(
            f"{run_prefix}/analysis.json",
            self.artifact_generator.build_run(
                task=specification.task,
                target=observations.target,
                predictions=observations.predictions,
                probabilities=observations.probabilities,
                probability_classes=observations.probability_classes,
                positive_label=specification.metrics.positive_label,
                fitted_features=final_features,
                model=final_adapter,
                random_seed=specification.validation.random_seed,
                learning_curve=learning_curve,
            ),
        )
        predictions_path = self._save_predictions(
            f"{run_prefix}/predictions.parquet",
            observations=observations,
            task=specification.task,
        )
        metrics_payload = {
            "primary_metric": specification.metrics.primary,
            "values": metric_values,
            "standard_deviations": metric_deviations,
            "folds": fold_payloads,
            "effective_parameters": dict(final_adapter.get_params()),
            "analysis_path": analysis_path,
            "predictions_path": predictions_path,
        }
        if tuning_payload is not None:
            metrics_payload["tuning"] = tuning_payload
            self._save_json(f"{run_prefix}/tuning.json", tuning_payload)
        self._save_json(f"{run_prefix}/metrics.json", metrics_payload)
        self._save_json(f"{run_prefix}/preprocessing.json", final_pipeline.get_config())
        model_path = self._save_model(
            f"{run_prefix}/model.joblib",
            {
                "experiment_id": experiment_id,
                "run_id": run.id,
                "spec_version": specification.spec_version,
                "pipeline": final_pipeline,
                "model": final_adapter,
            },
        )
        completed = self.runs.transition(
            run.id,
            ModelRunStatus.COMPLETED,
            metrics=metrics_payload,
            artifact_path=model_path,
        )
        self._emit(
            on_event,
            "model.completed",
            experiment_id=experiment_id,
            run_id=run.id,
            model=model_specification.id,
            progress=model_progress_end,
            message_key="training.model.completed",
        )
        return completed

    def _evaluate_parameters(
        self,
        *,
        model_specification: ModelSpec,
        parameters: dict[str, Any],
        specification: ExperimentSpec,
        features: pd.DataFrame,
        target: pd.Series,
        validation: ValidationPlan,
        column_types: dict[str, str],
        on_fold: Callable[[int, int, MetricEvaluation], None] | None = None,
        collect_observations: bool = False,
    ) -> tuple[
        list[MetricEvaluation],
        list[dict[str, Any]],
        EvaluationObservations,
    ]:
        """Avalia um conjunto de parâmetros treinando o modelo nos folds de validação cruzada.

        Args:
            model_specification: Especificação do modelo com o ID do algoritmo.
            parameters: Configuração de hiperparâmetros a avaliar.
            specification: Especificação técnica global do experimento.
            features: DataFrame com as features de entrada.
            target: Série do Pandas contendo a variável alvo.
            validation: Plano de particionamento dos folds.
            column_types: Mapeamento dos tipos semânticos das colunas.
            on_fold: Callback opcional executado a cada conclusão de fold.
            collect_observations: Se True, acumula predições e probabilidades detalhadas.

        Returns:
            Tupla contendo a lista de avaliações de métricas por fold, payloads detalhados
            por fold e o objeto EvaluationObservations com as predições coletadas.
        """
        evaluations: list[MetricEvaluation] = []
        payloads: list[dict[str, Any]] = []
        observed_target: list[Any] = []
        observed_predictions: list[Any] = []
        observed_row_ids: list[str] = []
        observed_fold_indices: list[int] = []
        observed_probabilities: list[tuple[float, ...]] = []
        observed_classes: tuple[Any, ...] | None = None
        for fold in validation.folds:
            adapter = self.catalog.model_adapters.create(
                model_specification.id,
                specification.task,
                parameters,
            )
            pipeline = self.pipeline_compiler.compile(
                specification,
                column_types=column_types,
                capabilities=adapter.get_capabilities(),
            )
            train_features, train_target = pipeline.fit_transform(
                features.iloc[list(fold.train_indices)],
                target.iloc[list(fold.train_indices)],
            )
            test_features = pipeline.transform(features.iloc[list(fold.test_indices)])
            test_target = target.iloc[list(fold.test_indices)]
            adapter.fit(
                train_features,
                train_target,
                fit_options=pipeline.get_model_options(),
            )
            predictions = adapter.predict(test_features)
            metric_ids = [
                specification.metrics.primary,
                *specification.metrics.secondary,
            ]
            metric_needs_probabilities = any(
                self.catalog.metrics.get(metric_id).requires_proba for metric_id in metric_ids
            )
            collect_probabilities = (
                collect_observations and specification.task != TaskType.REGRESSION
            )
            needs_probabilities = metric_needs_probabilities or collect_probabilities
            probabilities = adapter.predict_proba(test_features) if needs_probabilities else None
            classes = adapter.get_classes() if needs_probabilities else None
            evaluation = self.metric_evaluator.evaluate(
                task=specification.task,
                specification=specification.metrics,
                target=test_target,
                predictions=predictions,
                probabilities=probabilities,
                probability_classes=classes,
            )
            evaluations.append(evaluation)
            payloads.append(
                {
                    "fold_index": fold.fold_index,
                    "metrics": evaluation.values,
                    "train_rows": len(train_features),
                    "test_rows": len(test_features),
                }
            )
            if collect_observations:
                observed_row_ids.extend(str(index) for index in fold.test_indices)
                observed_fold_indices.extend([fold.fold_index] * len(fold.test_indices))
                observed_target.extend(np.asarray(test_target).tolist())
                observed_predictions.extend(np.asarray(predictions).tolist())
                if probabilities is not None:
                    fold_probabilities = np.asarray(probabilities, dtype=float)
                    observed_probabilities.extend(
                        tuple(float(value) for value in row) for row in fold_probabilities
                    )
                    fold_classes = tuple(classes or ())
                    if observed_classes is not None and observed_classes != fold_classes:
                        raise ExperimentExecutionError(
                            "Probability class order changed between validation folds"
                        )
                    observed_classes = fold_classes
            if on_fold is not None:
                on_fold(fold.fold_index, len(validation.folds), evaluation)
        return (
            evaluations,
            payloads,
            EvaluationObservations(
                row_ids=tuple(observed_row_ids),
                fold_indices=tuple(observed_fold_indices),
                target=tuple(observed_target),
                predictions=tuple(observed_predictions),
                probabilities=(tuple(observed_probabilities) if observed_probabilities else None),
                probability_classes=observed_classes,
            ),
        )

    def _build_learning_curve(
        self,
        *,
        model_specification: ModelSpec,
        parameters: dict[str, Any],
        specification: ExperimentSpec,
        features: pd.DataFrame,
        target: pd.Series,
        validation: ValidationPlan,
        column_types: dict[str, str],
    ) -> list[dict[str, float | int]]:
        """Gera pontos da curva de aprendizado variando o volume de treinamento.

        Args:
            model_specification (ModelSpec): Especificação do modelo.
            parameters (dict[str, Any]): Hiperparâmetros do modelo.
            specification (ExperimentSpec): Especificação completa do experimento.
            features (pd.DataFrame): DataFrame com features de entrada.
            target (pd.Series): Série com os valores-alvo.
            validation (ValidationPlan): Plano de validação cruzada.
            column_types (dict[str, str]): Mapeamento de coluna para tipo inferido.

        Returns:
            list[dict[str, float | int]]: Lista de pontos contendo tamanho e métrica primária.
        """
        if not validation.folds:
            return []
        source_fold = validation.folds[0]
        train_indices = np.asarray(source_fold.train_indices, dtype=int)
        rng = np.random.default_rng(specification.validation.random_seed)
        points: list[dict[str, float | int]] = []
        seen_sizes: set[int] = set()
        for fraction in (0.5, 0.75, 1.0):
            if specification.task == TaskType.REGRESSION:
                shuffled = rng.permutation(train_indices)
                count = max(2, int(round(len(shuffled) * fraction)))
                subset = shuffled[:count]
            else:
                selected: list[int] = []
                train_target = target.iloc[train_indices]
                for label in train_target.unique():
                    group = train_indices[np.asarray(train_target == label)]
                    shuffled_group = rng.permutation(group)
                    count = max(1, int(round(len(group) * fraction)))
                    selected.extend(int(index) for index in shuffled_group[:count])
                subset = np.asarray(sorted(selected), dtype=int)
            if len(subset) in seen_sizes:
                continue
            seen_sizes.add(len(subset))
            curve_validation = ValidationPlan(
                strategy="learning_curve",
                random_seed=specification.validation.random_seed,
                shuffle=True,
                folds=(
                    ValidationFold(
                        fold_index=0,
                        train_indices=tuple(int(index) for index in subset),
                        test_indices=source_fold.test_indices,
                    ),
                ),
            )
            try:
                evaluations, _, _ = self._evaluate_parameters(
                    model_specification=model_specification,
                    parameters=parameters,
                    specification=specification,
                    features=features,
                    target=target,
                    validation=curve_validation,
                    column_types=column_types,
                )
            except (ValueError, RuntimeError):
                continue
            values, _ = self.metric_evaluator.aggregate(evaluations)
            points.append(
                {
                    "training_rows": len(subset),
                    "validation_value": values[specification.metrics.primary],
                }
            )
        return sorted(points, key=lambda point: int(point["training_rows"]))

    def _load_dataset(
        self, specification: ExperimentSpec
    ) -> tuple[Any, DatasetProfile, pd.DataFrame]:
        """Carrega e valida a integridade do dataset e perfil do experimento.

        Args:
            specification (ExperimentSpec): Especificação do experimento a ser executado.

        Returns:
            tuple[Any, DatasetProfile, pd.DataFrame]: Entidade do dataset, perfil e DataFrame.

        Raises:
            ExperimentExecutionError: Se o dataset não existir, falhar na integridade hash,
                estiver sem perfil ou faltar colunas obrigatórias.
        """
        dataset = self.datasets.get(specification.dataset_id)
        if dataset is None:
            raise ExperimentExecutionError(f"Dataset {specification.dataset_id} was not found")
        if dataset.content_hash != specification.dataset_hash:
            raise ExperimentExecutionError(
                "Experiment dataset hash does not match persisted dataset"
            )
        content = self.dataset_storage.load(dataset.storage_path)
        if hashlib.sha256(content).hexdigest() != dataset.content_hash:
            raise ExperimentExecutionError(
                "Persisted dataset content failed integrity verification"
            )
        if dataset.profile_path is None:
            raise ExperimentExecutionError("Dataset profile is unavailable")
        profile = DatasetProfile.model_validate_json(
            self.dataset_storage.load(dataset.profile_path)
        )
        if profile.profile_version != specification.profile_version:
            raise ExperimentExecutionError(
                "Experiment profile version does not match dataset profile"
            )
        frame = pd.read_csv(
            io.BytesIO(content),
            encoding=profile.encoding,
            sep=profile.delimiter,
            low_memory=False,
        )
        required_columns = {specification.target, *specification.features}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ExperimentExecutionError(
                f"Dataset is missing experiment columns: {sorted(missing_columns)}"
            )
        return dataset, profile, frame

    @staticmethod
    def _metric_values(run: ModelRun) -> dict[str, float]:
        """Extrai os valores numéricos das métricas agregadas de uma execução.

        Args:
            run (ModelRun): Execução do modelo contendo payload de métricas.

        Returns:
            dict[str, float]: Dicionário associando nome da métrica ao valor numérico float.

        Raises:
            ExperimentExecutionError: Se a execução não contiver valores válidos de métricas.
        """
        metrics = run.metrics
        if metrics is None or not isinstance(metrics.get("values"), dict):
            raise ExperimentExecutionError(f"Run {run.id} has no metric values")
        return {key: float(value) for key, value in metrics["values"].items()}

    def _save_json(self, artifact_id: str, payload: dict[str, Any]) -> str:
        """Serializa e persiste um payload arbitrário em formato JSON no storage.

        Args:
            artifact_id (str): Identificador relativo do artefato a salvar.
            payload (dict[str, Any]): Dicionário com dados a serem persistidos.

        Returns:
            str: Caminho de persistência retornado pelo storage.
        """
        return self.artifact_storage.save(
            artifact_id,
            json.dumps(
                sanitize_for_json(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )

    def _save_model(self, artifact_id: str, payload: dict[str, Any]) -> str:
        """Serializa e persiste artefatos de modelo treinado com joblib no storage.

        Args:
            artifact_id (str): Identificador relativo do artefato a salvar.
            payload (dict[str, Any]): Dicionário com componentes do modelo treinado.

        Returns:
            str: Caminho de persistência retornado pelo storage.
        """
        buffer = io.BytesIO()
        joblib.dump(payload, buffer, compress=3)
        return self.artifact_storage.save(artifact_id, buffer.getvalue())

    def _save_predictions(
        self,
        artifact_id: str,
        *,
        observations: EvaluationObservations,
        task: TaskType,
    ) -> str:
        """Persiste observações e predições out-of-fold em formato Parquet no storage.

        Args:
            artifact_id (str): Identificador do artefato a ser salvo.
            observations (EvaluationObservations): Observações coletadas durante validação.
            task (TaskType): Tipo da tarefa de aprendizado supervisionado.

        Returns:
            str: Caminho de persistência retornado pelo storage.

        Raises:
            ExperimentExecutionError: Se houver identificadores de linha duplicados na validação.
        """
        if len(observations.row_ids) != len(set(observations.row_ids)):
            raise ExperimentExecutionError(
                "Validation produced duplicate row ids; paired comparison would be ambiguous"
            )
        probabilities = observations.probabilities or tuple(() for _ in observations.row_ids)
        classes = tuple(str(item) for item in (observations.probability_classes or ()))
        rows: list[dict[str, Any]] = []
        for row_id, fold_index, target, prediction, probability_values in zip(
            observations.row_ids,
            observations.fold_indices,
            observations.target,
            observations.predictions,
            probabilities,
            strict=True,
        ):
            probability_map = (
                {
                    class_name: float(value)
                    for class_name, value in zip(classes, probability_values, strict=True)
                }
                if classes
                else None
            )
            persisted_target = str(target) if task != TaskType.REGRESSION else float(target)
            persisted_prediction = (
                str(prediction) if task != TaskType.REGRESSION else float(prediction)
            )
            rows.append(
                {
                    "row_id": row_id,
                    "fold_index": fold_index,
                    "target": persisted_target,
                    "prediction": persisted_prediction,
                    "correct": (
                        persisted_prediction == persisted_target
                        if task != TaskType.REGRESSION
                        else None
                    ),
                    "confidence": (max(probability_map.values()) if probability_map else None),
                    "probabilities": probability_map,
                }
            )
        buffer = io.BytesIO()
        pd.DataFrame(rows).to_parquet(buffer, index=False)
        return self.artifact_storage.save(artifact_id, buffer.getvalue())

    @staticmethod
    def _runtime_versions() -> dict[str, str]:
        """Coleta as versões das dependências de runtime e bibliotecas de ML instaladas.

        Returns:
            dict[str, str]: Mapeamento de nome do pacote para string de versão.
        """
        packages = [
            "catboost",
            "lightgbm",
            "numpy",
            "optuna",
            "pandas",
            "plotly",
            "scikit-learn",
            "shap",
            "xgboost",
        ]
        versions = {"python": platform.python_version()}
        for package in packages:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not-installed"
        return versions

    @staticmethod
    def _emit(
        callback: Callable[[dict[str, Any]], None] | None,
        event_type: str,
        **payload: Any,
    ) -> None:
        """Emite evento de progresso se um callback tiver sido fornecido.

        Args:
            callback (Callable[[dict[str, Any]], None] | None): Função de callback para eventos.
            event_type (str): Identificador do tipo de evento emitido.
            **payload (Any): Atributos adicionais do evento.
        """
        if callback is not None:
            callback({"type": event_type, **payload})
