import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.domain.experiment.entities import Experiment, ModelRun, ModelRunStatus
from app.ml.registries.catalog import RegistryCatalog


@dataclass(frozen=True, slots=True)
class RoundSummary:
    """Resumo estatístico de uma rodada de execução (execution_id) do experimento.

    Attributes:
        execution_id (str): Identificador único da rodada de execução.
        started_at (datetime | None): Data e hora de início do primeiro run da rodada.
        finished_at (datetime | None): Data e hora de término do último run da rodada.
        total_models (int): Quantidade total de modelos executados na rodada.
        completed_models (int): Quantidade de modelos concluídos com sucesso.
    """

    execution_id: str
    started_at: datetime | None
    finished_at: datetime | None
    total_models: int
    completed_models: int


@dataclass(frozen=True, slots=True)
class RankedModel:
    """Dados de ranqueamento e métricas de desempenho de um modelo concluído na rodada.

    Attributes:
        model_id (str): Identificador do modelo de machine learning.
        run_id (str): Identificador único da execução (run).
        run_index (int): Posição ordinal do modelo na especificação do experimento.
        rank (int): Posição no ranqueamento da rodada (1 para melhor desempenho).
        primary_metric_value (float): Pontuação obtida na métrica primária.
        delta_to_best (float): Diferença absoluta em relação ao melhor modelo.
        standard_deviation (float | None): Desvio padrão entre folds da validação, se houver.
        duration_seconds (float | None): Tempo total de execução do treinamento em segundos.
        secondary_metrics (dict[str, float]): Métricas secundárias válidas calculadas.
    """

    model_id: str
    run_id: str
    run_index: int
    rank: int
    primary_metric_value: float
    delta_to_best: float
    standard_deviation: float | None
    duration_seconds: float | None
    secondary_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ExcludedModel:
    """Representa um modelo que não pôde ser ranqueado na comparação competitiva.

    Attributes:
        model_id (str): Identificador do modelo de machine learning.
        run_id (str): Identificador do run correspondente.
        run_index (int): Posição do modelo na ordem de execução.
        status (str): Status final do run (ex: failed, running).
        reason (str): Motivo de exclusão do ranqueamento.
    """

    model_id: str
    run_id: str
    run_index: int
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ComparisonProtocol:
    """Metadados do protocolo experimental compartilhado entre os modelos comparados.

    Attributes:
        dataset_id (str): Identificador do dataset utilizado.
        dataset_hash (str | None): Hash SHA-256 do dataset na execução.
        task (str): Tipo da tarefa de machine learning.
        target (str): Nome da coluna alvo predita.
        features (list[str]): Lista de features utilizadas pelos modelos.
        validation_strategy (str): Estratégia de validação empregada.
        random_seed (int): Semente de aleatoriedade para partição dos dados.
        folds (int | None): Número de folds da validação cruzada, se aplicável.
        test_size (float | None): Proporção de teste para divisão simples, se aplicável.
        preprocessing (list[str]): Lista com identificadores das etapas de pré-processamento.
        tuning_enabled (bool): Indica se a otimização de hiperparâmetros estava habilitada.
    """

    dataset_id: str
    dataset_hash: str | None
    task: str
    target: str
    features: list[str]
    validation_strategy: str
    random_seed: int
    folds: int | None
    test_size: float | None
    preprocessing: list[str]
    tuning_enabled: bool


@dataclass(frozen=True, slots=True)
class ExperimentComparison:
    """Estrutura consolidada de comparação de modelos de uma rodada do experimento.

    Attributes:
        experiment_id (str): Identificador do experimento comparado.
        available_rounds (list[RoundSummary]): Lista com todas as rodadas disponíveis.
        selected_round_id (str | None): Identificador da rodada atualmente analisada.
        primary_metric (str): Identificador da métrica primária de ordenação.
        metric_direction (str): Direção de otimização ('maximize' ou 'minimize').
        can_explain (bool): Indica se há modelos válidos suficientes para gerar explicação.
        explanation_block_reason (str | None): Motivo do bloqueio da explicação, se houver.
        best_model (RankedModel | None): Modelo vencedor da rodada.
        worst_model (RankedModel | None): Modelo com menor pontuação na métrica primária.
        ranked_models (list[RankedModel]): Lista ordenada dos modelos elegíveis.
        excluded_models (list[ExcludedModel]): Modelos descartados do ranqueamento.
        protocol (ComparisonProtocol): Protocolo de validação comum a todos os modelos.
    """

    experiment_id: str
    available_rounds: list[RoundSummary]
    selected_round_id: str | None
    primary_metric: str
    metric_direction: str
    can_explain: bool
    explanation_block_reason: str | None
    best_model: RankedModel | None
    worst_model: RankedModel | None
    ranked_models: list[RankedModel]
    excluded_models: list[ExcludedModel]
    protocol: ComparisonProtocol


def _extract_protocol(experiment: Experiment) -> ComparisonProtocol:
    """Extrai os metadados do protocolo experimental a partir da especificação do experimento.

    Args:
        experiment: Entidade do experimento contendo a especificação técnica.

    Returns:
        ComparisonProtocol populado com parâmetros de validação, dataset e features.
    """
    spec = experiment.spec if isinstance(experiment.spec, dict) else experiment.spec.model_dump()
    validation = spec.get("validation", {})
    tuning = spec.get("tuning", {})
    preprocessing_steps = [
        step.get("id") or step.get("strategy") or "unknown"
        for step in spec.get("preprocessing", [])
        if isinstance(step, dict)
    ]
    return ComparisonProtocol(
        dataset_id=str(spec.get("dataset_id") or experiment.dataset_id),
        dataset_hash=spec.get("dataset_hash"),
        task=str(spec.get("task")),
        target=str(spec.get("target")),
        features=list(spec.get("features", [])),
        validation_strategy=str(validation.get("strategy", "unknown")),
        random_seed=int(validation.get("random_seed", 42)),
        folds=validation.get("folds"),
        test_size=validation.get("test_size"),
        preprocessing=preprocessing_steps,
        tuning_enabled=bool(tuning.get("enabled", False)),
    )


def build_experiment_comparison(
    experiment: Experiment,
    runs: list[ModelRun],
    catalog: RegistryCatalog,
    *,
    execution_id: str | None = None,
) -> ExperimentComparison:
    """Constrói a comparação estruturada dos modelos de um experimento por rodada de execução.

    Agrupa os runs por rodada (execution_id), ranqueia os modelos elegíveis conforme a métrica
    primária e identifica modelos excluídos ou com falha.

    Args:
        experiment: Entidade do experimento contendo a especificação técnica.
        runs: Lista com todos os runs de modelos vinculados ao experimento.
        catalog: Catálogo de métricas para obter a direção de otimização (maximize/minimize).
        execution_id: Identificador opcional da rodada de execução a analisar.

    Returns:
        ExperimentComparison com os modelos ranqueados e resumo das rodadas disponíveis.
    """
    spec = experiment.spec if isinstance(experiment.spec, dict) else experiment.spec.model_dump()
    primary_metric = (spec.get("metrics") or {}).get("primary", "accuracy")
    direction = (
        catalog.metrics.get(primary_metric).direction
        if catalog.metrics.contains(primary_metric)
        else "maximize"
    )
    protocol = _extract_protocol(experiment)

    # Group runs by execution_id. Runs with execution_id is None are legacy and unlinked.
    runs_by_execution: dict[str, list[ModelRun]] = defaultdict(list)
    for run in runs:
        if run.execution_id:
            runs_by_execution[run.execution_id].append(run)

    available_rounds: list[RoundSummary] = []
    for exec_id, round_runs in runs_by_execution.items():
        start_candidates = [r.started_at for r in round_runs if r.started_at is not None]
        finish_candidates = [r.finished_at for r in round_runs if r.finished_at is not None]
        started_at = min(start_candidates) if start_candidates else None
        finished_at = max(finish_candidates) if finish_candidates else None
        completed_count = sum(1 for r in round_runs if r.status == ModelRunStatus.COMPLETED)
        available_rounds.append(
            RoundSummary(
                execution_id=exec_id,
                started_at=started_at,
                finished_at=finished_at,
                total_models=len(round_runs),
                completed_models=completed_count,
            )
        )

    # Sort rounds by finished_at descending (latest first), then started_at
    available_rounds.sort(
        key=lambda r: (
            r.finished_at.timestamp() if r.finished_at else 0.0,
            r.started_at.timestamp() if r.started_at else 0.0,
        ),
        reverse=True,
    )

    # Determine selected round
    selected_round_id: str | None = None
    if execution_id:
        if execution_id in runs_by_execution:
            selected_round_id = execution_id
    elif available_rounds:
        # Suggest the most recent round with completed models, or simply the latest round
        completed_round = next((r for r in available_rounds if r.completed_models > 0), None)
        selected_round_id = (
            completed_round.execution_id if completed_round else available_rounds[0].execution_id
        )

    if not selected_round_id or selected_round_id not in runs_by_execution:
        return ExperimentComparison(
            experiment_id=experiment.id,
            available_rounds=available_rounds,
            selected_round_id=None,
            primary_metric=primary_metric,
            metric_direction=direction,
            can_explain=False,
            explanation_block_reason="no_rounds" if not available_rounds else "round_not_found",
            best_model=None,
            worst_model=None,
            ranked_models=[],
            excluded_models=[],
            protocol=protocol,
        )

    selected_runs = runs_by_execution[selected_round_id]
    eligible_runs: list[tuple[ModelRun, float, float | None, dict[str, float]]] = []
    excluded_models: list[ExcludedModel] = []

    for run in selected_runs:
        if run.status != ModelRunStatus.COMPLETED:
            excluded_models.append(
                ExcludedModel(
                    model_id=run.model_id,
                    run_id=run.id,
                    run_index=run.run_index,
                    status=run.status.value,
                    reason=f"status_{run.status.value}",
                )
            )
            continue

        metrics = run.metrics
        metric_values = (metrics or {}).get("values") if isinstance(metrics, dict) else None
        if not isinstance(metric_values, dict) or primary_metric not in metric_values:
            excluded_models.append(
                ExcludedModel(
                    model_id=run.model_id,
                    run_id=run.id,
                    run_index=run.run_index,
                    status=run.status.value,
                    reason="missing_metric",
                )
            )
            continue

        raw_val = metric_values[primary_metric]
        try:
            fval = float(raw_val)
        except (ValueError, TypeError):
            fval = float("nan")

        if not math.isfinite(fval):
            excluded_models.append(
                ExcludedModel(
                    model_id=run.model_id,
                    run_id=run.id,
                    run_index=run.run_index,
                    status=run.status.value,
                    reason="non_finite_metric",
                )
            )
            continue

        std_devs = (metrics or {}).get("standard_deviations") if isinstance(metrics, dict) else None
        std_val: float | None = None
        if isinstance(std_devs, dict) and primary_metric in std_devs:
            try:
                s = float(std_devs[primary_metric])
                if math.isfinite(s):
                    std_val = s
            except (ValueError, TypeError):
                std_val = None

        secondary = {
            k: float(v)
            for k, v in metric_values.items()
            if k != primary_metric and isinstance(v, (int, float)) and math.isfinite(float(v))
        }

        eligible_runs.append((run, fval, std_val, secondary))

    # Stable sort:
    # If maximize, highest value first (multiplier = -1)
    # If minimize, lowest value first (multiplier = 1)
    multiplier = -1 if direction == "maximize" else 1
    eligible_runs.sort(
        key=lambda item: (
            multiplier * item[1],
            item[0].duration_seconds or 0.0,
            item[0].model_id,
            item[0].id,
        )
    )

    ranked_models: list[RankedModel] = []
    best_value = eligible_runs[0][1] if eligible_runs else 0.0

    current_rank = 1
    for index, (run, fval, std_val, secondary) in enumerate(eligible_runs):
        if index > 0:
            prev_val = eligible_runs[index - 1][1]
            if not math.isclose(fval, prev_val, rel_tol=1e-9, abs_tol=1e-12):
                current_rank = index + 1

        delta_to_best = abs(fval - best_value)
        ranked_models.append(
            RankedModel(
                model_id=run.model_id,
                run_id=run.id,
                run_index=run.run_index,
                rank=current_rank,
                primary_metric_value=fval,
                delta_to_best=delta_to_best,
                standard_deviation=std_val,
                duration_seconds=run.duration_seconds,
                secondary_metrics=secondary,
            )
        )

    best_model = ranked_models[0] if ranked_models else None
    worst_model = ranked_models[-1] if ranked_models else None

    can_explain = len(ranked_models) >= 2
    explanation_block_reason = None if can_explain else "insufficient_eligible_models"

    return ExperimentComparison(
        experiment_id=experiment.id,
        available_rounds=available_rounds,
        selected_round_id=selected_round_id,
        primary_metric=primary_metric,
        metric_direction=direction,
        can_explain=can_explain,
        explanation_block_reason=explanation_block_reason,
        best_model=best_model,
        worst_model=worst_model,
        ranked_models=ranked_models,
        excluded_models=excluded_models,
        protocol=protocol,
    )


def build_experiment_comparison_context(
    comparison: ExperimentComparison,
    experiment: Experiment,
) -> dict[str, Any]:
    """Extrai agregados de comparação e metadados seguros para contexto de prompts LLM.

    Gera uma visão resumida e segura do desempenho e protocolo sem expor dados brutos
    do dataset ou caminhos internos do sistema de arquivos.

    Args:
        comparison: Comparação estruturada dos modelos de uma rodada.
        experiment: Entidade do experimento associado.

    Returns:
        Dicionário formatado com dados seguros e limitações metodológicas.
    """
    direction_label = (
        "higher is better" if comparison.metric_direction == "maximize" else "lower is better"
    )
    return {
        "experiment_name": experiment.name,
        "objective": experiment.objective,
        "task": comparison.protocol.task,
        "target": comparison.protocol.target,
        "primary_metric": comparison.primary_metric,
        "metric_direction": comparison.metric_direction,
        "ranking_criterion": f"{comparison.primary_metric} ({direction_label})",
        "best_model": (
            {
                "model_id": comparison.best_model.model_id,
                "primary_metric_value": comparison.best_model.primary_metric_value,
            }
            if comparison.best_model
            else None
        ),
        "worst_model": (
            {
                "model_id": comparison.worst_model.model_id,
                "primary_metric_value": comparison.worst_model.primary_metric_value,
            }
            if comparison.worst_model
            else None
        ),
        "ranked_models": [
            {
                "model_id": m.model_id,
                "rank": m.rank,
                "primary_metric_value": m.primary_metric_value,
                "delta_to_best": m.delta_to_best,
                "standard_deviation": m.standard_deviation,
                "duration_seconds": m.duration_seconds,
                "secondary_metrics": m.secondary_metrics,
            }
            for m in comparison.ranked_models
        ],
        "excluded_models": [
            {
                "model_id": m.model_id,
                "status": m.status,
                "reason": m.reason,
            }
            for m in comparison.excluded_models
        ],
        "shared_protocol": {
            "validation_strategy": comparison.protocol.validation_strategy,
            "random_seed": comparison.protocol.random_seed,
            "folds": comparison.protocol.folds,
            "test_size": comparison.protocol.test_size,
            "features_count": len(comparison.protocol.features),
            "tuning_enabled": comparison.protocol.tuning_enabled,
        },
        "limitations": [
            "Deltas and fold standard deviations do not imply statistical significance.",
            "Observed differences reflect only this validation protocol and dataset sample.",
            "Trade-offs between primary and secondary metrics should be considered "
            "before model selection.",
            "No causal relationship or production generalization guarantee is established.",
        ],
    }
