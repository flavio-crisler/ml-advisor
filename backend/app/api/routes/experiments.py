import asyncio
import json
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import (
    get_artifact_storage,
    get_experiment_service,
    get_experiment_strategy_repository,
    get_explanation_service,
    get_job_executor,
    get_model_run_repository,
    get_registry_catalog,
    get_training_submission_service,
    get_tuning_trial_repository,
)
from app.application.experiments.comparison import (
    ExperimentComparison,
    build_experiment_comparison,
    build_experiment_comparison_context,
)
from app.application.experiments.service import ExperimentService
from app.application.explanations.models import ExplanationLanguage, ExplanationResult
from app.application.explanations.prompts import ExplanationContextError
from app.application.explanations.service import ExplanationService
from app.application.jobs.executor import TERMINAL_JOB_STATUSES, JobExecutor, JobSnapshot
from app.application.jobs.service import TrainingSubmissionService
from app.domain.exceptions import (
    DomainError,
    EntityNotFoundError,
    ExperimentDeletionError,
    ExperimentSpecValidationError,
    ImmutableExperimentError,
    JobAlreadyRunningError,
    JobNotFoundError,
    JobStateError,
)
from app.domain.experiment.entities import (
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    ModelRun,
    TuningTrial,
)
from app.domain.experiment.spec import ExperimentSpec
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyExperimentStrategyRepository,
    SqlAlchemyModelRunRepository,
    SqlAlchemyTuningTrialRepository,
)
from app.infrastructure.storage.base import ArtifactStorage
from app.llm.errors import LLMConfigurationError, LLMProviderError
from app.ml.registries.catalog import RegistryCatalog

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentCreateRequest(BaseModel):
    """Payload de entrada para criação manual ou assistida de um novo experimento.

    Attributes:
        mode (ExperimentMode): Modo de criação (manual ou assistido via Advisor).
        description (str | None): Descrição textual opcional sobre a intenção do experimento.
        spec (ExperimentSpec): Especificação técnica declarativa de dados, modelos e métricas.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ExperimentMode = ExperimentMode.ASSISTED
    description: str | None = None
    spec: ExperimentSpec


class ExperimentCloneRequest(BaseModel):
    """Payload de entrada para clonagem de um experimento existente.

    Attributes:
        name (str | None): Nome opcional para o novo experimento clonado.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None


class ExperimentUpdateRequest(BaseModel):
    """Payload de entrada para atualização da especificação de um experimento em rascunho.

    Attributes:
        spec (ExperimentSpec): Nova especificação técnica completa a ser salva.
    """

    model_config = ConfigDict(extra="forbid")

    spec: ExperimentSpec


class ExperimentRenameRequest(BaseModel):
    """Payload de entrada para renomeação de um experimento existente.

    Attributes:
        name (str): Novo nome do experimento (1 a 200 caracteres).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)


class ModelRunResponse(BaseModel):
    """Representação serializada de uma execução de treinamento de modelo individual.

    Attributes:
        id (str): Identificador único da execução do modelo.
        model_id (str): Identificador do algoritmo no catálogo de modelos.
        run_index (int): Índice ordinal de execução dentro do experimento.
        status (str): Status operacional da execução.
        started_at (datetime | None): Timestamp de início do treinamento.
        finished_at (datetime | None): Timestamp de conclusão do treinamento.
        duration_seconds (float | None): Tempo total de processamento em segundos.
        parameters: Hiperparâmetros utilizados no treinamento.
        metrics: Métricas calculadas na validação cruzada.
        artifact_path (str | None): Caminho do modelo e artefatos salvos.
        error_message (str | None): Descrição de erro caso a execução tenha falhado.
        execution_id (str | None): Identificador da rodada de execução associada.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    model_id: str
    run_index: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    parameters: dict[str, Any]
    metrics: dict[str, Any] | None
    artifact_path: str | None
    error_message: str | None
    execution_id: str | None = None


class RoundSummaryResponse(BaseModel):
    """Resumo de uma rodada de treinamento em lote de modelos.

    Attributes:
        execution_id (str): Identificador da rodada de execução.
        started_at (datetime | None): Data e hora de início da rodada.
        finished_at (datetime | None): Data e hora de término da rodada.
        total_models (int): Quantidade total de modelos planejados.
        completed_models (int): Modelos concluídos com sucesso.
    """

    model_config = ConfigDict(frozen=True)

    execution_id: str
    started_at: datetime | None
    finished_at: datetime | None
    total_models: int
    completed_models: int


class RankedModelResponse(BaseModel):
    """Modelo classificado no ranking comparativo da rodada de treinamento.

    Attributes:
        model_id (str): Identificador do algoritmo no catálogo.
        run_id (str): Identificador da execução individual do modelo.
        run_index (int): Índice de execução.
        rank (int): Posição no ranking (1 para o melhor modelo).
        primary_metric_value (float): Valor obtido na métrica primária.
        delta_to_best (float): Diferença relativa em relação ao melhor modelo.
        standard_deviation (float | None): Desvio padrão entre os folds de validação.
        duration_seconds (float | None): Duração do treino em segundos.
        secondary_metrics (dict[str, float]): Demais métricas secundárias calculadas.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    run_id: str
    run_index: int
    rank: int
    primary_metric_value: float
    delta_to_best: float
    standard_deviation: float | None
    duration_seconds: float | None
    secondary_metrics: dict[str, float]


class ExcludedModelResponse(BaseModel):
    """Modelo excluído do ranking comparativo por falha ou incompatibilidade.

    Attributes:
        model_id (str): Identificador do algoritmo.
        run_id (str): Identificador da execução individual.
        run_index (int): Índice de execução.
        status (str): Status final da execução.
        reason (str): Motivo pelo qual o modelo foi desconsiderado no ranking.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    run_id: str
    run_index: int
    status: str
    reason: str


class ComparisonProtocolResponse(BaseModel):
    """Protocolo experimental e configurações de validação compartilhadas no ranking.

    Attributes:
        dataset_id (str): Identificador do dataset base.
        dataset_hash (str | None): Hash SHA-256 do dataset utilizado.
        task (str): Tipo da tarefa de Machine Learning.
        target (str): Nome da coluna alvo.
        features (list[str]): Lista de variáveis preditoras selecionadas.
        validation_strategy (str): Estratégia de validação adotada (k-fold, estratificada, etc.).
        random_seed (int): Semente de aleatoriedade configurada.
        folds (int | None): Número de folds de validação.
        test_size (float | None): Proporção do conjunto de teste em splits simples.
        preprocessing (list[str]): Etapas de pré-processamento aplicadas.
        tuning_enabled (bool): Indica se otimização de hiperparâmetros estava ativa.
    """

    model_config = ConfigDict(frozen=True)

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


class ExperimentComparisonResponse(BaseModel):
    """Resposta com relatório de comparação e ranking de modelos de uma rodada.

    Attributes:
        experiment_id (str): Identificador do experimento avaliado.
        available_rounds (list[RoundSummaryResponse]): Rodadas de execução disponíveis.
        selected_round_id (str | None): Identificador da rodada atualmente selecionada.
        primary_metric (str): Métrica de avaliação primária do ranking.
        metric_direction (str): Direção de otimização da métrica (maximize/minimize).
        can_explain (bool): Se a rodada possui modelos suficientes para explicação LLM.
        explanation_block_reason (str | None): Motivo do bloqueio da explicação LLM, se houver.
        best_model (RankedModelResponse | None): Modelo no topo do ranking.
        worst_model (RankedModelResponse | None): Modelo na base do ranking.
        ranked_models (list[RankedModelResponse]): Lista ordenada de modelos elegíveis.
        excluded_models (list[ExcludedModelResponse]): Modelos descartados por erro.
        protocol (ComparisonProtocolResponse): Protocolo e splits de validação compartilhados.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    available_rounds: list[RoundSummaryResponse]
    selected_round_id: str | None
    primary_metric: str
    metric_direction: str
    can_explain: bool
    explanation_block_reason: str | None
    best_model: RankedModelResponse | None
    worst_model: RankedModelResponse | None
    ranked_models: list[RankedModelResponse]
    excluded_models: list[ExcludedModelResponse]
    protocol: ComparisonProtocolResponse


class ExperimentComparisonExplanationRequest(BaseModel):
    """Payload de solicitação de explicação em linguagem natural da comparação.

    Attributes:
        execution_id (str): Identificador da rodada de execução a ser explicada.
        language (ExplanationLanguage): Idioma de destino do texto (ex: 'pt-BR').
    """

    model_config = ConfigDict(extra="forbid")

    execution_id: str
    language: ExplanationLanguage = "pt-BR"


class ExperimentResponse(BaseModel):
    """Representação serializada completa de um experimento de ML.

    Attributes:
        id (str): Identificador único do experimento.
        name (str): Nome do experimento.
        description (str | None): Descrição textual.
        mode (ExperimentMode): Modo de criação (manual, assistido).
        status (ExperimentStatus): Status do ciclo de vida (draft, running, completed, etc.).
        dataset_id (str): Identificador do dataset associado.
        parent_experiment_id (str | None): ID do experimento pai caso clonado ou derivado.
        objective (str): Meta ou problema em linguagem natural.
        spec (ExperimentSpec): Especificação técnica declarativa.
        created_at (datetime): Data e hora de criação.
        updated_at (datetime): Data e hora da última modificação.
        runs (list[ModelRunResponse]): Lista de execuções de modelos registradas.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None
    mode: ExperimentMode
    status: ExperimentStatus
    dataset_id: str
    parent_experiment_id: str | None
    objective: str
    spec: ExperimentSpec
    created_at: datetime
    updated_at: datetime
    runs: list[ModelRunResponse]


class TuningTrialResponse(BaseModel):
    """Registro de um trial individual de otimização de hiperparâmetros (Optuna).

    Attributes:
        id (str): Identificador único do trial.
        trial_number (int): Número ordinal da tentativa.
        parameters: Combinação de hiperparâmetros testada.
        status (str): Status de conclusão do trial.
        metric_value (float | None): Valor alcançado na métrica de otimização.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    trial_number: int
    parameters: dict[str, Any]
    status: str
    metric_value: float | None


class ComparisonRunResponse(BaseModel):
    """Elemento comparativo entre execuções de diferentes experimentos.

    Attributes:
        experiment_id (str): Identificador do experimento de origem.
        experiment_name (str): Nome do experimento.
        parent_experiment_id (str | None): ID do experimento pai.
        run (ModelRunResponse): Dados da execução do modelo treinado.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    experiment_name: str
    parent_experiment_id: str | None
    run: ModelRunResponse


class JobResponse(BaseModel):
    """Representação serializada do estado de um job de treinamento assíncrono.

    Attributes:
        job_id (str): Identificador do job em background.
        experiment_id (str): Identificador do experimento que o originou.
        status (str): Status de execução (queued, running, completed, failed, cancelled).
        created_at (datetime): Timestamp de enfileiramento.
        started_at (datetime | None): Timestamp de início real.
        finished_at (datetime | None): Timestamp de encerramento.
        error (str | None): Detalhes de erro em caso de falha.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    experiment_id: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None


class ExperimentStrategyResponse(BaseModel):
    """Registro da estratégia analítica ou melhoria associada ao experimento.

    Attributes:
        experiment_id (str): Identificador do experimento.
        source (str): Origem da estratégia (ex: advisor, improvement).
        provider_id (str | None): Provedor LLM que gerou a proposta.
        strategy: Dicionário contendo a estratégia serializada.
        created_at (datetime): Timestamp de criação.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    source: str
    provider_id: str | None
    strategy: dict[str, Any]
    created_at: datetime


def _run_response(run: ModelRun) -> ModelRunResponse:
    """Converte a entidade ModelRun no modelo de resposta serializável correspondente.

    Args:
        run (ModelRun): Entidade da execução do modelo.

    Returns:
        ModelRunResponse: Objeto de resposta contendo métricas e status.
    """
    return ModelRunResponse(
        id=run.id,
        model_id=run.model_id,
        run_index=run.run_index,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_seconds=run.duration_seconds,
        parameters=run.parameters,
        metrics=run.metrics,
        artifact_path=run.artifact_path,
        error_message=run.error_message,
        execution_id=run.execution_id,
    )


def _comparison_response(comparison: ExperimentComparison) -> ExperimentComparisonResponse:
    """Converte a estrutura de comparação entre modelos para o modelo da API.

    Args:
        comparison (ExperimentComparison): Estrutura de dados de comparação calculada.

    Returns:
        ExperimentComparisonResponse: Resposta com ranking e protocolo da rodada.
    """
    return ExperimentComparisonResponse(
        experiment_id=comparison.experiment_id,
        available_rounds=[
            RoundSummaryResponse(
                execution_id=r.execution_id,
                started_at=r.started_at,
                finished_at=r.finished_at,
                total_models=r.total_models,
                completed_models=r.completed_models,
            )
            for r in comparison.available_rounds
        ],
        selected_round_id=comparison.selected_round_id,
        primary_metric=comparison.primary_metric,
        metric_direction=comparison.metric_direction,
        can_explain=comparison.can_explain,
        explanation_block_reason=comparison.explanation_block_reason,
        best_model=RankedModelResponse(
            model_id=comparison.best_model.model_id,
            run_id=comparison.best_model.run_id,
            run_index=comparison.best_model.run_index,
            rank=comparison.best_model.rank,
            primary_metric_value=comparison.best_model.primary_metric_value,
            delta_to_best=comparison.best_model.delta_to_best,
            standard_deviation=comparison.best_model.standard_deviation,
            duration_seconds=comparison.best_model.duration_seconds,
            secondary_metrics=comparison.best_model.secondary_metrics,
        )
        if comparison.best_model
        else None,
        worst_model=RankedModelResponse(
            model_id=comparison.worst_model.model_id,
            run_id=comparison.worst_model.run_id,
            run_index=comparison.worst_model.run_index,
            rank=comparison.worst_model.rank,
            primary_metric_value=comparison.worst_model.primary_metric_value,
            delta_to_best=comparison.worst_model.delta_to_best,
            standard_deviation=comparison.worst_model.standard_deviation,
            duration_seconds=comparison.worst_model.duration_seconds,
            secondary_metrics=comparison.worst_model.secondary_metrics,
        )
        if comparison.worst_model
        else None,
        ranked_models=[
            RankedModelResponse(
                model_id=m.model_id,
                run_id=m.run_id,
                run_index=m.run_index,
                rank=m.rank,
                primary_metric_value=m.primary_metric_value,
                delta_to_best=m.delta_to_best,
                standard_deviation=m.standard_deviation,
                duration_seconds=m.duration_seconds,
                secondary_metrics=m.secondary_metrics,
            )
            for m in comparison.ranked_models
        ],
        excluded_models=[
            ExcludedModelResponse(
                model_id=m.model_id,
                run_id=m.run_id,
                run_index=m.run_index,
                status=m.status,
                reason=m.reason,
            )
            for m in comparison.excluded_models
        ],
        protocol=ComparisonProtocolResponse(
            dataset_id=comparison.protocol.dataset_id,
            dataset_hash=comparison.protocol.dataset_hash,
            task=comparison.protocol.task,
            target=comparison.protocol.target,
            features=comparison.protocol.features,
            validation_strategy=comparison.protocol.validation_strategy,
            random_seed=comparison.protocol.random_seed,
            folds=comparison.protocol.folds,
            test_size=comparison.protocol.test_size,
            preprocessing=comparison.protocol.preprocessing,
            tuning_enabled=comparison.protocol.tuning_enabled,
        ),
    )


def _experiment_response(experiment: Experiment, runs: list[ModelRun]) -> ExperimentResponse:
    """Monta a resposta completa do experimento agregando suas execuções de modelos.

    Args:
        experiment (Experiment): Entidade do experimento.
        runs (list[ModelRun]): Lista de execuções de modelos vinculadas.

    Returns:
        ExperimentResponse: Objeto de resposta serializável.
    """
    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        mode=experiment.mode,
        status=experiment.status,
        dataset_id=experiment.dataset_id,
        parent_experiment_id=experiment.parent_experiment_id,
        objective=experiment.objective,
        spec=ExperimentSpec.model_validate(experiment.spec),
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        runs=[_run_response(run) for run in runs],
    )


def _job_response(snapshot: JobSnapshot) -> JobResponse:
    """Converte um JobSnapshot no modelo de resposta de jobs.

    Args:
        snapshot (JobSnapshot): Snapshot retornado pelo executor de jobs.

    Returns:
        JobResponse: Resposta com identificador e status do job.
    """
    return JobResponse(
        job_id=snapshot.job_id,
        experiment_id=snapshot.experiment_id,
        status=snapshot.status.value,
        created_at=snapshot.created_at,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
        error=snapshot.error,
    )


def _trial_response(trial: TuningTrial) -> TuningTrialResponse:
    """Converte a entidade TuningTrial para o modelo de resposta de trials.

    Args:
        trial (TuningTrial): Entidade do trial de otimização.

    Returns:
        TuningTrialResponse: Objeto serializável do trial.
    """
    return TuningTrialResponse(
        id=trial.id,
        trial_number=trial.trial_number,
        parameters=trial.parameters,
        status=trial.status.value,
        metric_value=trial.metric_value,
    )


def _domain_http_error(error: DomainError) -> HTTPException:
    """Converte erros de domínio de experimentos em exceções HTTPException com status adequado.

    Args:
        error (DomainError): Exceção de domínio levantada.

    Returns:
        HTTPException: Exceção HTTP com código e detalhes estruturados.
    """
    if isinstance(error, (EntityNotFoundError, JobNotFoundError)):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(
        error,
        (
            ExperimentDeletionError,
            ImmutableExperimentError,
            JobAlreadyRunningError,
            JobStateError,
        ),
    ):
        code = status.HTTP_409_CONFLICT
    else:
        code = status.HTTP_400_BAD_REQUEST
    detail: dict[str, Any] = {
        "code": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, ExperimentSpecValidationError):
        detail["issues"] = error.issues
    return HTTPException(status_code=code, detail=detail)


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    request: ExperimentCreateRequest,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> ExperimentResponse:
    """Cria e persiste um novo experimento com validação de sua especificação.

    Args:
        request (ExperimentCreateRequest): Dados de criação e especificação do experimento.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções de modelos.

    Returns:
        ExperimentResponse: Entidade do experimento criada com lista de runs vazia.

    Raises:
        HTTPException: Se a especificação for inválida ou o dataset não existir.
    """
    try:
        experiment = service.create(
            request.spec,
            mode=request.mode,
            description=request.description,
        )
    except DomainError as error:
        raise _domain_http_error(error) from error
    return _experiment_response(experiment, runs.list_for_experiment(experiment.id))


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> list[ExperimentResponse]:
    """Lista todos os experimentos ativos cadastrados na plataforma.

    Args:
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções de modelos.

    Returns:
        list[ExperimentResponse]: Lista completa de experimentos e suas execuções.
    """
    return [
        _experiment_response(item, runs.list_for_experiment(item.id)) for item in service.list_all()
    ]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: str,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> ExperimentResponse:
    """Recupera detalhes, especificações e histórico de execuções de um experimento.

    Args:
        experiment_id (str): Identificador do experimento.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.

    Returns:
        ExperimentResponse: Dados completos do experimento consultado.

    Raises:
        HTTPException: Se o experimento não for encontrado (404).
    """
    try:
        experiment = service.get(experiment_id)
    except DomainError as error:
        raise _domain_http_error(error) from error
    return _experiment_response(experiment, runs.list_for_experiment(experiment.id))


@router.get("/{experiment_id}/strategy", response_model=ExperimentStrategyResponse)
def get_experiment_strategy(
    experiment_id: str,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    strategies: Annotated[
        SqlAlchemyExperimentStrategyRepository,
        Depends(get_experiment_strategy_repository),
    ],
) -> ExperimentStrategyResponse:
    """Recupera a proposta ou estratégia analítica que deu origem ao experimento.

    Args:
        experiment_id (str): Identificador do experimento.
        service (ExperimentService): Serviço de experimentos injetado.
        strategies (SqlAlchemyExperimentStrategyRepository): Repositório de estratégias.

    Returns:
        ExperimentStrategyResponse: Estratégia persistida com origem e provedor.

    Raises:
        HTTPException: Se o experimento ou estratégia não existirem (404).
    """
    try:
        service.get(experiment_id)
    except DomainError as error:
        raise _domain_http_error(error) from error
    strategy = strategies.get_for_experiment(experiment_id)
    if strategy is None:
        raise _domain_http_error(
            EntityNotFoundError(f"No recorded strategy for experiment {experiment_id}")
        )
    return ExperimentStrategyResponse(
        experiment_id=strategy.experiment_id,
        source=strategy.source,
        provider_id=strategy.provider_id,
        strategy=strategy.strategy,
        created_at=strategy.created_at,
    )


@router.put("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment(
    experiment_id: str,
    request: ExperimentUpdateRequest,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> ExperimentResponse:
    """Atualiza a especificação de um experimento antes de seu primeiro treinamento.

    Args:
        experiment_id (str): Identificador do experimento a modificar.
        request (ExperimentUpdateRequest): Nova especificação a ser validada e aplicada.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.

    Returns:
        ExperimentResponse: Experimento atualizado.

    Raises:
        HTTPException: Se o experimento já tiver sido executado (409) ou for inválido (400).
    """
    try:
        experiment = service.update_spec(experiment_id, request.spec)
    except DomainError as error:
        raise _domain_http_error(error) from error
    return _experiment_response(experiment, runs.list_for_experiment(experiment.id))


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
def rename_experiment(
    experiment_id: str,
    request: ExperimentRenameRequest,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> ExperimentResponse:
    """Altera o nome descritivo de um experimento existente.

    Args:
        experiment_id (str): Identificador do experimento.
        request (ExperimentRenameRequest): Novo nome fornecido.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.

    Returns:
        ExperimentResponse: Experimento com nome renomeado.

    Raises:
        HTTPException: Se o experimento não for encontrado (404).
    """
    try:
        experiment = service.rename(experiment_id, request.name)
    except DomainError as error:
        raise _domain_http_error(error) from error
    return _experiment_response(experiment, runs.list_for_experiment(experiment.id))


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(
    experiment_id: str,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
) -> Response:
    """Exclui logicamente ou fisicamente um experimento da plataforma.

    Args:
        experiment_id (str): Identificador do experimento a ser removido.
        service (ExperimentService): Serviço de experimentos injetado.

    Returns:
        Response: Resposta vazia com HTTP 204 No Content.

    Raises:
        HTTPException: Se houver jobs em andamento ou restrição de deleção (409).
    """
    try:
        service.delete(experiment_id)
    except DomainError as error:
        raise _domain_http_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs/{run_id}", response_model=ModelRunResponse)
def get_model_run(
    run_id: str,
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> ModelRunResponse:
    """Recupera detalhes e métricas de treinamento de uma execução de modelo.

    Args:
        run_id (str): Identificador da execução individual do modelo.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções injetado.

    Returns:
        ModelRunResponse: Métricas, hiperparâmetros e status da execução.

    Raises:
        HTTPException: Se a execução não for encontrada (404).
    """
    run = runs.get(run_id)
    if run is None:
        raise _domain_http_error(EntityNotFoundError(f"Model run {run_id} was not found"))
    return _run_response(run)


@router.get("/runs/{run_id}/tuning-trials", response_model=list[TuningTrialResponse])
def list_tuning_trials(
    run_id: str,
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
    trials: Annotated[SqlAlchemyTuningTrialRepository, Depends(get_tuning_trial_repository)],
) -> list[TuningTrialResponse]:
    """Lista todos os trials de otimização de hiperparâmetros da execução de um modelo.

    Args:
        run_id (str): Identificador da execução do modelo.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.
        trials (SqlAlchemyTuningTrialRepository): Repositório de trials injetado.

    Returns:
        list[TuningTrialResponse]: Lista de tentativas com parâmetros e métricas.

    Raises:
        HTTPException: Se a execução do modelo não existir (404).
    """
    if runs.get(run_id) is None:
        raise _domain_http_error(EntityNotFoundError(f"Model run {run_id} was not found"))
    return [_trial_response(trial) for trial in trials.list_for_run(run_id)]


@router.get("/runs/{run_id}/analysis", response_model=dict[str, Any])
def get_run_analysis(
    run_id: str,
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_artifact_storage)],
) -> dict[str, Any]:
    """Recupera o artefato de análise aprofundada (SHAP, curvas, resíduos) do modelo.

    Args:
        run_id (str): Identificador da execução do modelo.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.
        storage (ArtifactStorage): Storage para leitura do artefato JSON.

    Returns:
        dict[str, Any]: Estrutura com explainability, feature importance e diagnósticos.

    Raises:
        HTTPException: Se o modelo não for encontrado (404) ou não possuir análise (409).
    """
    run = runs.get(run_id)
    if run is None:
        raise _domain_http_error(EntityNotFoundError(f"Model run {run_id} was not found"))
    analysis_path = (run.metrics or {}).get("analysis_path")
    if not isinstance(analysis_path, str):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "AnalysisUnavailable",
                "message_key": "experiments.run.analysisUnavailable",
            },
        )
    return json.loads(storage.load(analysis_path))


@router.get(
    "/{experiment_id}/compare/{other_experiment_id}",
    response_model=list[ComparisonRunResponse],
)
def compare_experiments(
    experiment_id: str,
    other_experiment_id: str,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> list[ComparisonRunResponse]:
    """Retorna execuções de dois experimentos para análise comparativa cruzada.

    Args:
        experiment_id (str): Identificador do primeiro experimento.
        other_experiment_id (str): Identificador do segundo experimento.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.

    Returns:
        list[ComparisonRunResponse]: Lista de execuções de ambos os experimentos.

    Raises:
        HTTPException: Se qualquer um dos experimentos não for encontrado (404).
    """
    try:
        experiments = [
            service.get_including_deleted(experiment_id),
            service.get_including_deleted(other_experiment_id),
        ]
    except DomainError as error:
        raise _domain_http_error(error) from error
    return [
        ComparisonRunResponse(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            parent_experiment_id=experiment.parent_experiment_id,
            run=_run_response(run),
        )
        for experiment in experiments
        for run in runs.list_for_experiment(experiment.id)
    ]


@router.post(
    "/{experiment_id}/clone",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def clone_experiment(
    experiment_id: str,
    request: ExperimentCloneRequest,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
) -> ExperimentResponse:
    """Clona um experimento existente mantendo sua especificação como novo rascunho.

    Args:
        experiment_id (str): Identificador do experimento de origem.
        request (ExperimentCloneRequest): Parâmetros opcionais (ex: novo nome).
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.

    Returns:
        ExperimentResponse: Novo experimento derivado criado em status DRAFT.

    Raises:
        HTTPException: Se o experimento original não existir (404).
    """
    try:
        clone = service.clone(experiment_id, name=request.name)
    except DomainError as error:
        raise _domain_http_error(error) from error
    return _experiment_response(clone, runs.list_for_experiment(clone.id))


@router.post(
    "/{experiment_id}/run",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_experiment(
    experiment_id: str,
    training: Annotated[
        TrainingSubmissionService,
        Depends(get_training_submission_service),
    ],
) -> JobResponse:
    """Submete um experimento para execução e treinamento de modelos em segundo plano.

    Args:
        experiment_id (str): Identificador do experimento a executar.
        training (TrainingSubmissionService): Serviço de submissão de treinamento.

    Returns:
        JobResponse: Snapshot do job assíncrono enfileirado.

    Raises:
        HTTPException: Se já houver job em execução (409) ou o experimento não existir (404).
    """
    try:
        snapshot = training.submit(experiment_id)
    except DomainError as error:
        raise _domain_http_error(error) from error
    return _job_response(snapshot)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(
    job_id: str,
    executor: Annotated[JobExecutor, Depends(get_job_executor)],
) -> JobResponse:
    """Consulta o status atual de um job de treinamento assíncrono.

    Args:
        job_id (str): Identificador do job em background.
        executor (JobExecutor): Executor de jobs injetado.

    Returns:
        JobResponse: Snapshot atualizado com status, timestamps e eventuais erros.

    Raises:
        HTTPException: Se o job não for encontrado (404).
    """
    try:
        return _job_response(executor.status(job_id))
    except DomainError as error:
        raise _domain_http_error(error) from error


@router.post("/{experiment_id}/cancel", response_model=JobResponse)
def cancel_experiment(
    experiment_id: str,
    executor: Annotated[JobExecutor, Depends(get_job_executor)],
) -> JobResponse:
    """Interrompe e cancela um job de treinamento ativo de um experimento.

    Args:
        experiment_id (str): Identificador do experimento cujo job será cancelado.
        executor (JobExecutor): Executor assíncrono de jobs.

    Returns:
        JobResponse: Snapshot do job com status atualizado para cancelado.

    Raises:
        HTTPException: Se nenhum job ativo for localizado para o experimento (404).
    """
    try:
        snapshot = executor.find_for_experiment(experiment_id)
        if snapshot is None:
            raise JobNotFoundError(f"No job was found for experiment {experiment_id}")
        return _job_response(executor.cancel(snapshot.job_id))
    except DomainError as error:
        raise _domain_http_error(error) from error


@router.get(
    "/{experiment_id}/comparison",
    response_model=ExperimentComparisonResponse,
)
def get_experiment_comparison(
    experiment_id: str,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
    catalog: Annotated[RegistryCatalog, Depends(get_registry_catalog)],
    execution_id: Annotated[str | None, Query()] = None,
) -> ExperimentComparisonResponse:
    """Gera ranking comparativo de modelos treinados para uma rodada do experimento.

    Args:
        experiment_id (str): Identificador do experimento.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.
        catalog (RegistryCatalog): Catálogo de métricas e modelos.
        execution_id (str | None): Identificador opcional da rodada desejada.

    Returns:
        ExperimentComparisonResponse: Ranking com delta to best, protocolo e métricas.

    Raises:
        HTTPException: Se o experimento não existir (404).
    """
    try:
        experiment = service.get_including_deleted(experiment_id)
    except DomainError as error:
        raise _domain_http_error(error) from error
    experiment_runs = runs.list_for_experiment(experiment_id)
    comparison = build_experiment_comparison(
        experiment,
        experiment_runs,
        catalog,
        execution_id=execution_id,
    )
    return _comparison_response(comparison)


@router.post(
    "/{experiment_id}/explanation",
    response_model=ExplanationResult,
)
def explain_experiment_comparison(
    experiment_id: str,
    request: ExperimentComparisonExplanationRequest,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
    runs: Annotated[SqlAlchemyModelRunRepository, Depends(get_model_run_repository)],
    catalog: Annotated[RegistryCatalog, Depends(get_registry_catalog)],
    explanation_service: Annotated[ExplanationService, Depends(get_explanation_service)],
) -> ExplanationResult:
    """Gera explicação textual interpretativa do ranking de modelos via LLM.

    Args:
        experiment_id (str): Identificador do experimento.
        request (ExperimentComparisonExplanationRequest): Rodada e idioma solicitados.
        service (ExperimentService): Serviço de experimentos injetado.
        runs (SqlAlchemyModelRunRepository): Repositório de execuções.
        catalog (RegistryCatalog): Catálogo de componentes ML.
        explanation_service (ExplanationService): Serviço de explicações por LLM.

    Returns:
        ExplanationResult: Justificativa estruturada e narrativa dos resultados.

    Raises:
        HTTPException: Se os modelos forem insuficientes (409), o contexto for muito
            grande (413), a LLM estiver desconfigurada (503) ou o provedor falhar (502).
    """
    try:
        experiment = service.get_including_deleted(experiment_id)
    except DomainError as error:
        raise _domain_http_error(error) from error
    experiment_runs = runs.list_for_experiment(experiment_id)
    comparison = build_experiment_comparison(
        experiment,
        experiment_runs,
        catalog,
        execution_id=request.execution_id,
    )
    if not comparison.can_explain:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": comparison.explanation_block_reason or "insufficient_eligible_models"},
        )
    context = build_experiment_comparison_context(comparison, experiment)
    try:
        return explanation_service.explain(
            scope="experiment_comparison",
            language=request.language,
            context=context,
        )
    except ExplanationContextError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": error.code},
        ) from error
    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": error.code},
        ) from error
    except LLMProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": error.code},
        ) from error


@router.get("/{experiment_id}/events")
def experiment_events(
    experiment_id: str,
    executor: Annotated[JobExecutor, Depends(get_job_executor)],
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> StreamingResponse:
    """Transmite eventos em tempo real do job de treinamento via Server-Sent Events (SSE).

    Args:
        experiment_id (str): Identificador do experimento.
        executor (JobExecutor): Executor de background jobs.
        after (int): Número de sequência inicial para busca de eventos retroativos.
        last_event_id (int | None): Header Last-Event-ID para suporte a reconexão automática.

    Returns:
        StreamingResponse: Stream SSE com eventos em formato texto/event-stream.

    Raises:
        HTTPException: Se nenhum job for localizado para o experimento (404).
    """
    snapshot = executor.find_for_experiment(experiment_id)
    if snapshot is None:
        raise _domain_http_error(
            JobNotFoundError(f"No job was found for experiment {experiment_id}")
        )
    cursor = max(after, last_event_id or 0)

    async def stream():
        nonlocal cursor
        while True:
            events = executor.events(snapshot.job_id, after=cursor)
            for event in events:
                cursor = event.sequence
                payload = {
                    "sequence": event.sequence,
                    "type": event.type,
                    "timestamp": event.timestamp.isoformat(),
                    **event.payload,
                }
                yield (
                    f"id: {event.sequence}\n"
                    f"event: {event.type}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )
            current = executor.status(snapshot.job_id)
            if current.status in TERMINAL_JOB_STATUSES and not executor.events(
                snapshot.job_id, after=cursor
            ):
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
