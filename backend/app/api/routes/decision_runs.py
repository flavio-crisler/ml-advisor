import asyncio
import json
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import (
    get_artifact_storage,
    get_calibration_run_repository,
    get_decision_benchmark_service,
    get_decision_comparison_service,
    get_decision_job_executor,
    get_decision_run_repository,
)
from app.application.datasets.errors import DatasetIngestionError
from app.application.decision_benchmarks.comparison import DecisionComparisonService
from app.application.decision_benchmarks.service import DecisionBenchmarkService
from app.application.jobs.executor import TERMINAL_JOB_STATUSES, JobExecutor
from app.domain.decision.runs import CalibrationRun, DecisionRun, ZeroShotDecisionSpec
from app.domain.exceptions import DomainError, EntityNotFoundError, JobNotFoundError
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyCalibrationRunRepository,
    SqlAlchemyDecisionRunRepository,
)
from app.infrastructure.storage.base import ArtifactStorage

router = APIRouter(tags=["decision-runs"])


class DecisionRunCreateRequest(BaseModel):
    """Payload de entrada para criação e submissão de benchmark de decisão.

    Attributes:
        spec (ZeroShotDecisionSpec): Especificação completa do benchmark zero-shot.
        external_processing_acknowledged (bool): Consentimento para provedores externos.
        experiment_id (str | None): ID de experimento de ML para pareamento comparativo.
    """

    model_config = ConfigDict(extra="forbid")

    spec: ZeroShotDecisionSpec
    external_processing_acknowledged: bool = False
    experiment_id: str | None = None


class DecisionJobResponse(BaseModel):
    """Representação simplificada do job em background de um benchmark de decisão.

    Attributes:
        decision_run_id (str): Identificador da execução de decisão criada.
        job_id (str): Identificador do job assíncrono enfileirado.
        status (str): Status operacional atual do job.
    """

    model_config = ConfigDict(frozen=True)

    decision_run_id: str
    job_id: str
    status: str


class DecisionRunResponse(BaseModel):
    """Representação detalhada de uma execução de benchmark de decisão e suas métricas.

    Attributes:
        id (str): Identificador único da execução.
        dataset_id (str): Identificador do dataset avaliado.
        experiment_id (str | None): Experimento ML de referência pareado.
        provider_id (str): Provedor de inferência utilizado.
        requested_model (str): Nome do modelo solicitado.
        resolved_model (str | None): Modelo exato resolvido pelo provedor.
        question_spec_hash (str): Hash canônico das perguntas avaliadas.
        status (str): Status operacional da execução.
        row_count (int): Linhas totais do dataset.
        processed_rows (int): Linhas processadas até o momento.
        request_count (int): Total de requisições enviadas ao provedor.
        cache_hits (int): Quantidade de respostas servidas via cache local.
        input_tokens (int | None): Total de tokens de entrada consumidos.
        estimated_cost (float | None): Custo financeiro estimado da execução.
        latency_ms_total (float | None): Latência acumulada de inferência em ms.
        artifact_path (str | None): Caminho do artefato com respostas e métricas salvas.
        calibration_run_id (str | None): ID da calibração associada, se houver.
        error_message (str | None): Mensagem de erro em caso de falha.
        created_at (datetime): Data e hora de criação do benchmark.
        started_at (datetime | None): Data e hora de início de processamento.
        finished_at (datetime | None): Data e hora de conclusão.
        spec (ZeroShotDecisionSpec): Especificação das perguntas e regras.
        metrics (dict[str, Any] | None): Métricas avaliadas carregadas do artefato.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    dataset_id: str
    experiment_id: str | None
    provider_id: str
    requested_model: str
    resolved_model: str | None
    question_spec_hash: str
    status: str
    row_count: int
    processed_rows: int
    request_count: int
    cache_hits: int
    input_tokens: int | None
    estimated_cost: float | None
    latency_ms_total: float | None
    artifact_path: str | None
    calibration_run_id: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    spec: ZeroShotDecisionSpec
    metrics: dict[str, Any] | None = None


class CalibrationResponse(BaseModel):
    """Resposta com detalhes de calibração de probabilidades de um benchmark.

    Attributes:
        id (str): Identificador da calibração.
        decision_run_id (str): Identificador da execução de decisão calibrada.
        task (str): Tipo da tarefa preditiva.
        metrics (dict[str, Any]): Métricas de calibração (ECE, Brier score, etc.).
        artifact_path (str): Caminho do artefato persistido.
        created_at (datetime): Data e hora da calibração.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    decision_run_id: str
    task: str
    metrics: dict[str, Any]
    artifact_path: str
    created_at: datetime


class ComparedApproachResponse(BaseModel):
    """Métricas e custos de uma das abordagens na comparação (Supervisionada vs Zero-Shot).

    Attributes:
        metrics (dict[str, float | None]): Dicionário de métricas de acurácia/desempenho.
        latency_p95_ms (float | None): Percentil 95 de latência de predição em ms.
        estimated_cost (float | None): Custo estimado para a amostra avaliada.
        training_duration_seconds (float | None): Duração do treinamento supervisionado.
        request_count (int | None): Requisições de inferência realizadas.
        cache_hits (int | None): Hits no cache de decisão.
        full_run_estimated_cost (float | None): Custo projetado para todo o dataset.
    """

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float | None]
    latency_p95_ms: float | None = None
    estimated_cost: float | None = None
    training_duration_seconds: float | None = None
    request_count: int | None = None
    cache_hits: int | None = None
    full_run_estimated_cost: float | None = None


class DisagreementResponse(BaseModel):
    """Matriz de concordância e discordância entre predições supervisionadas e zero-shot.

    Attributes:
        both_correct (int): Amostras onde ambos os modelos acertaram.
        supervised_only_correct (int): Amostras onde apenas o modelo ML acertou.
        zero_shot_only_correct (int): Amostras onde apenas a LLM zero-shot acertou.
        both_wrong (int): Amostras onde ambos erraram.
        prediction_disagreements (int): Quantidade total de discordâncias de predição.
    """

    model_config = ConfigDict(frozen=True)

    both_correct: int
    supervised_only_correct: int
    zero_shot_only_correct: int
    both_wrong: int
    prediction_disagreements: int


class ComparedModelResponse(BaseModel):
    """Comparação detalhada de um modelo supervisionado contra a decisão zero-shot.

    Attributes:
        model_run_id (str): Identificador da execução do modelo treinado.
        model_id (str): Identificador do algoritmo de ML.
        run_index (int): Índice ordinal da execução.
        sample_size (int): Quantidade de instâncias avaliadas no pareamento.
        classes (list[str]): Classes categóricas avaliadas.
        supervised (ComparedApproachResponse): Desempenho da abordagem supervisionada.
        zero_shot (ComparedApproachResponse): Desempenho da abordagem zero-shot.
        metric_deltas (dict[str, float | None]): Diferença de métricas (zero-shot - supervisionado).
        disagreement (DisagreementResponse): Métricas de discordância amostral.
    """

    model_config = ConfigDict(frozen=True)

    model_run_id: str
    model_id: str
    run_index: int
    sample_size: int
    classes: list[str]
    supervised: ComparedApproachResponse
    zero_shot: ComparedApproachResponse
    metric_deltas: dict[str, float | None]
    disagreement: DisagreementResponse


class UnavailableModelResponse(BaseModel):
    """Registro de modelo de ML que não pôde ser comparado ao benchmark.

    Attributes:
        run_id (str): Identificador da execução do modelo.
        model_id (str): Identificador do algoritmo.
        reason (str): Motivo pelo qual a comparação não pôde ser efetuada.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    model_id: str
    reason: str


class DecisionComparisonResponse(BaseModel):
    """Relatório estruturado de comparação entre modelos de ML e benchmark de decisão LLM.

    Attributes:
        schema_version (str): Versão do schema da comparação.
        decision_run_id (str): Identificador da execução de decisão.
        experiment_id (str): Identificador do experimento de ML comparado.
        experiment_name (str): Nome do experimento.
        contract (dict[str, Any]): Metadados do contrato de pareamento amostral.
        models (list[ComparedModelResponse]): Lista de modelos com comparação completa.
        unavailable_models (list[UnavailableModelResponse]): Modelos não elegíveis para comparação.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str
    decision_run_id: str
    experiment_id: str
    experiment_name: str
    contract: dict[str, Any]
    models: list[ComparedModelResponse]
    unavailable_models: list[UnavailableModelResponse]


def _http_error(error: Exception) -> HTTPException:
    """Mapeia erros de domínio ou ingestão de decision runs para HTTPException correspondente.

    Args:
        error (Exception): Exceção levantada durante a execução.

    Returns:
        HTTPException: Exceção HTTP com status e detalhe estruturado.
    """
    if isinstance(error, (EntityNotFoundError, JobNotFoundError)):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, DatasetIngestionError):
        code = (
            status.HTTP_404_NOT_FOUND
            if error.code == "dataset_not_found"
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    else:
        code = status.HTTP_409_CONFLICT
    return HTTPException(
        status_code=code,
        detail={"code": type(error).__name__, "message": str(error)},
    )


def _response(run: DecisionRun, storage: ArtifactStorage) -> DecisionRunResponse:
    """Converte uma entidade DecisionRun e carrega suas métricas para montar a resposta.

    Args:
        run (DecisionRun): Entidade de execução de decisão.
        storage (ArtifactStorage): Storage para leitura do JSON de métricas.

    Returns:
        DecisionRunResponse: Payload completo de detalhes da execução.
    """
    metrics = None
    if run.artifact_path:
        metrics = json.loads(storage.load(run.artifact_path))
    return DecisionRunResponse(
        id=run.id,
        dataset_id=run.dataset_id,
        experiment_id=run.experiment_id,
        provider_id=run.provider_id,
        requested_model=run.requested_model,
        resolved_model=run.resolved_model,
        question_spec_hash=run.question_spec_hash,
        status=run.status.value,
        row_count=run.row_count,
        processed_rows=run.processed_rows,
        request_count=run.request_count,
        cache_hits=run.cache_hits,
        input_tokens=run.input_tokens,
        estimated_cost=run.estimated_cost,
        latency_ms_total=run.latency_ms_total,
        artifact_path=run.artifact_path,
        calibration_run_id=run.calibration_run_id,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        spec=run.specification,
        metrics=metrics,
    )


def _calibration_response(run: CalibrationRun) -> CalibrationResponse:
    """Converte a entidade CalibrationRun para o modelo serializável da API.

    Args:
        run (CalibrationRun): Entidade de calibração.

    Returns:
        CalibrationResponse: Objeto de resposta de calibração.
    """
    return CalibrationResponse(
        id=run.id,
        decision_run_id=run.decision_run_id,
        task=run.task,
        metrics=run.metrics,
        artifact_path=run.artifact_path,
        created_at=run.created_at,
    )


def _submit(
    request: DecisionRunCreateRequest,
    service: DecisionBenchmarkService,
) -> DecisionJobResponse:
    """Submete uma requisição de benchmark de decisão ao serviço e trata erros.

    Args:
        request (DecisionRunCreateRequest): Parâmetros da requisição.
        service (DecisionBenchmarkService): Serviço de benchmark de decisão.

    Returns:
        DecisionJobResponse: Snapshot com IDs da execução e do job enfileirado.

    Raises:
        HTTPException: Em caso de falha de validação ou erro de domínio.
    """
    try:
        result = service.submit(
            request.spec,
            external_processing_acknowledged=request.external_processing_acknowledged,
            experiment_id=request.experiment_id,
        )
    except (DatasetIngestionError, DomainError) as error:
        raise _http_error(error) from error
    return DecisionJobResponse(
        decision_run_id=result.run.id,
        job_id=result.job.job_id,
        status=result.job.status.value,
    )


@router.post(
    "/decision-runs", response_model=DecisionJobResponse, status_code=status.HTTP_202_ACCEPTED
)
def create_decision_run(
    request: DecisionRunCreateRequest,
    service: Annotated[DecisionBenchmarkService, Depends(get_decision_benchmark_service)],
) -> DecisionJobResponse:
    """Cria e enfileira uma nova execução de benchmark de decisão LLM.

    Args:
        request (DecisionRunCreateRequest): Especificação do benchmark e opções.
        service (DecisionBenchmarkService): Serviço injetado de benchmarks.

    Returns:
        DecisionJobResponse: Identificador do job assíncrono em processamento.
    """
    return _submit(request, service)


@router.post(
    "/decision-benchmarks",
    response_model=DecisionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_decision_benchmark(
    request: DecisionRunCreateRequest,
    service: Annotated[DecisionBenchmarkService, Depends(get_decision_benchmark_service)],
) -> DecisionJobResponse:
    """Cria e enfileira um benchmark de decisão (rota alias para /decision-runs).

    Args:
        request (DecisionRunCreateRequest): Especificação do benchmark.
        service (DecisionBenchmarkService): Serviço injetado de benchmarks.

    Returns:
        DecisionJobResponse: Identificador do job assíncrono enfileirado.
    """
    return _submit(request, service)


@router.get("/decision-runs", response_model=list[DecisionRunResponse])
def list_decision_runs(
    runs: Annotated[SqlAlchemyDecisionRunRepository, Depends(get_decision_run_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_artifact_storage)],
) -> list[DecisionRunResponse]:
    """Lista todas as execuções de benchmark de decisão registradas.

    Args:
        runs (SqlAlchemyDecisionRunRepository): Repositório de benchmarks.
        storage (ArtifactStorage): Storage para carregamento de métricas.

    Returns:
        list[DecisionRunResponse]: Lista de execuções com métricas e status.
    """
    return [_response(run, storage) for run in runs.list_all()]


@router.get("/decision-runs/{run_id}", response_model=DecisionRunResponse)
@router.get("/decision-benchmarks/{run_id}", response_model=DecisionRunResponse)
def get_decision_run(
    run_id: str,
    runs: Annotated[SqlAlchemyDecisionRunRepository, Depends(get_decision_run_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_artifact_storage)],
) -> DecisionRunResponse:
    """Recupera detalhes e métricas de uma execução de benchmark específica.

    Args:
        run_id (str): Identificador da execução.
        runs (SqlAlchemyDecisionRunRepository): Repositório de benchmarks.
        storage (ArtifactStorage): Storage de artefatos.

    Returns:
        DecisionRunResponse: Detalhes, status e métricas da execução.

    Raises:
        HTTPException: Se a execução não for encontrada (404).
    """
    run = runs.get(run_id)
    if run is None:
        raise _http_error(EntityNotFoundError(f"Decision run {run_id} was not found"))
    return _response(run, storage)


@router.get("/decision-benchmarks/{run_id}/calibration", response_model=CalibrationResponse)
def get_calibration(
    run_id: str,
    runs: Annotated[SqlAlchemyDecisionRunRepository, Depends(get_decision_run_repository)],
    calibrations: Annotated[
        SqlAlchemyCalibrationRunRepository, Depends(get_calibration_run_repository)
    ],
) -> CalibrationResponse:
    """Retorna dados de calibração de probabilidades para o benchmark informado.

    Args:
        run_id (str): Identificador do benchmark de decisão.
        runs (SqlAlchemyDecisionRunRepository): Repositório de benchmarks.
        calibrations (SqlAlchemyCalibrationRunRepository): Repositório de calibrações.

    Returns:
        CalibrationResponse: Métricas e artefato da calibração.

    Raises:
        HTTPException: Se o benchmark ou a calibração não forem localizados (404).
    """
    if runs.get(run_id) is None:
        raise _http_error(EntityNotFoundError(f"Decision run {run_id} was not found"))
    calibration = calibrations.get_for_decision_run(run_id)
    if calibration is None:
        raise _http_error(EntityNotFoundError(f"Calibration for {run_id} was not found"))
    return _calibration_response(calibration)


@router.get(
    "/decision-benchmarks/{run_id}/compare/{experiment_id}",
    response_model=DecisionComparisonResponse,
)
def compare_decision_run(
    run_id: str,
    experiment_id: str,
    service: Annotated[DecisionComparisonService, Depends(get_decision_comparison_service)],
) -> DecisionComparisonResponse:
    """Gera relatório de comparação entre a decisão zero-shot e modelos de ML supervisionados.

    Args:
        run_id (str): Identificador do benchmark de decisão.
        experiment_id (str): Identificador do experimento de ML treinado.
        service (DecisionComparisonService): Serviço de comparação injetado.

    Returns:
        DecisionComparisonResponse: Comparação estatística, métricas e discordâncias.

    Raises:
        HTTPException: Se entidades não forem encontradas ou forem incompatíveis.
    """
    try:
        comparison = service.compare(run_id, experiment_id)
    except DomainError as error:
        raise _http_error(error) from error
    return DecisionComparisonResponse.model_validate(comparison)


@router.post("/decision-runs/{run_id}/cancel", response_model=DecisionJobResponse)
def cancel_decision_run(
    run_id: str,
    executor: Annotated[JobExecutor, Depends(get_decision_job_executor)],
) -> DecisionJobResponse:
    """Cancela a execução de um job de benchmark de decisão em andamento.

    Args:
        run_id (str): Identificador da execução a cancelar.
        executor (JobExecutor): Executor de jobs injetado.

    Returns:
        DecisionJobResponse: Snapshot atualizado com status cancelado.

    Raises:
        HTTPException: Se o job não for encontrado (404) ou não puder ser cancelado.
    """
    try:
        snapshot = executor.find_for_experiment(run_id)
        if snapshot is None:
            raise JobNotFoundError(f"No job was found for decision run {run_id}")
        cancelled = executor.cancel(snapshot.job_id)
    except DomainError as error:
        raise _http_error(error) from error
    return DecisionJobResponse(
        decision_run_id=run_id,
        job_id=cancelled.job_id,
        status=cancelled.status.value,
    )


@router.get("/decision-runs/{run_id}/events")
def decision_run_events(
    run_id: str,
    executor: Annotated[JobExecutor, Depends(get_decision_job_executor)],
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> StreamingResponse:
    """Transmite eventos de progresso do benchmark de decisão via Server-Sent Events (SSE).

    Args:
        run_id (str): Identificador da execução.
        executor (JobExecutor): Executor assíncrono de jobs.
        after (int): Número de sequência inicial para busca de eventos retroativos.
        last_event_id (int | None): Header Last-Event-ID para reconexão automática SSE.

    Returns:
        StreamingResponse: Stream SSE com eventos em formato texto/event-stream.

    Raises:
        HTTPException: Se o job não for localizado (404).
    """
    snapshot = executor.find_for_experiment(run_id)
    if snapshot is None:
        raise _http_error(JobNotFoundError(f"No job was found for decision run {run_id}"))
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

    return StreamingResponse(stream(), media_type="text/event-stream")
