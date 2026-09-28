import asyncio
import json
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import (
    get_database,
    get_explanation_service,
    get_registry_catalog,
    get_semantic_feature_proposer,
    get_semantic_feature_service,
    get_semantic_feature_set_repository,
    get_semantic_job_executor,
)
from app.application.datasets.errors import DatasetIngestionError
from app.application.explanations.models import ExplanationLanguage, ExplanationResult
from app.application.explanations.prompts import ExplanationContextError
from app.application.explanations.service import ExplanationService
from app.application.jobs.executor import TERMINAL_JOB_STATUSES, JobExecutor
from app.application.semantic_features.impact_explanation import build_semantic_impact_context
from app.application.semantic_features.proposer import (
    SemanticFeatureProposer,
    SemanticProposalResult,
)
from app.application.semantic_features.service import SemanticFeatureService
from app.domain.decision.semantic_features import (
    SemanticDiscoveryMode,
    SemanticFeatureSet,
    SemanticQuestionProposal,
)
from app.domain.exceptions import DomainError, EntityNotFoundError, JobNotFoundError
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyExperimentRepository,
    SqlAlchemySemanticFeatureSetRepository,
)
from app.llm.errors import LLMConfigurationError, LLMProviderError
from app.ml.registries.catalog import RegistryCatalog

router = APIRouter(prefix="/semantic-features", tags=["semantic-features"])


class SemanticProposalRequest(BaseModel):
    """Payload de entrada para proposição de perguntas e features semânticas.

    Attributes:
        dataset_id (str): Identificador do dataset de origem.
        objective (str): Meta ou problema em linguagem natural.
        selected_columns (list[str]): Colunas textuais selecionadas para enriquecimento.
        language (Literal["pt-BR", "en-US"]): Idioma das instruções geradas.
        discovery_mode (SemanticDiscoveryMode): Modo de descoberta semântica.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=2000)
    selected_columns: list[str] = Field(min_length=1, max_length=20)
    language: Literal["pt-BR", "en-US"] = "pt-BR"
    discovery_mode: SemanticDiscoveryMode = SemanticDiscoveryMode.TARGET_AGNOSTIC


class SemanticProposalResponse(BaseModel):
    """Resposta contendo as perguntas semânticas propostas e estimativas de custo.

    Attributes:
        proposal (SemanticQuestionProposal): Lista de perguntas e especificações sugeridas.
        provider_id (str): Provedor LLM que gerou a proposta.
        discovery_mode (SemanticDiscoveryMode): Modo de descoberta utilizado.
        sample_sent (bool): Indica se uma amostra do dataset foi enviada.
        sample_rows (int): Quantidade de linhas enviadas ao LLM.
        estimated_requests (int): Número estimado de requisições necessárias.
        estimated_input_tokens (int | None): Total projetado de tokens de entrada.
        estimated_cost (float | None): Custo financeiro estimado da extração.
    """

    model_config = ConfigDict(frozen=True)

    proposal: SemanticQuestionProposal
    provider_id: str
    discovery_mode: SemanticDiscoveryMode
    sample_sent: bool
    sample_rows: int
    estimated_requests: int
    estimated_input_tokens: int | None = None
    estimated_cost: float | None = None


class SemanticGenerateRequest(BaseModel):
    """Payload de entrada para geração e materialização de features semânticas.

    Attributes:
        dataset_id (str): Identificador do dataset base.
        name (str): Nome amigável do conjunto de features semânticas.
        discovery_mode (SemanticDiscoveryMode): Modo de descoberta adotado.
        proposal (SemanticQuestionProposal): Perguntas semânticas a serem executadas.
        provider_id (str): Provedor de inferência de decisão selecionado.
        model (str): Nome do modelo a ser utilizado.
        external_processing_acknowledged (bool): Consentimento para provedores externos.
        source_experiment_id (str | None): Experimento opcional para cálculo de impacto ML.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    discovery_mode: SemanticDiscoveryMode = SemanticDiscoveryMode.TARGET_AGNOSTIC
    proposal: SemanticQuestionProposal
    provider_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    external_processing_acknowledged: bool = False
    source_experiment_id: str | None = None


class SemanticJobResponse(BaseModel):
    """Representação simplificada do job em background de extração semântica.

    Attributes:
        semantic_feature_set_id (str): Identificador do conjunto semântico criado.
        job_id (str): Identificador do job assíncrono enfileirado.
        status (str): Status operacional do job.
    """

    model_config = ConfigDict(frozen=True)

    semantic_feature_set_id: str
    job_id: str
    status: str


class SemanticImpactExplanationRequest(BaseModel):
    """Payload de solicitação de explicação interpretativa do impacto semântico.

    Attributes:
        language (ExplanationLanguage): Idioma de destino da explicação ('pt-BR' ou 'en-US').
    """

    model_config = ConfigDict(extra="forbid")

    language: ExplanationLanguage


class SemanticFeatureSetResponse(BaseModel):
    """Detalhes completos e estado operacional de um conjunto de features semânticas.

    Attributes:
        id (str): Identificador único do conjunto.
        dataset_id (str): Dataset base associado.
        source_experiment_id (str | None): Experimento ML pareado para avaliação de impacto.
        name (str): Nome amigável do conjunto.
        version (str): Versão do conjunto.
        status (str): Status de processamento (queued, running, completed, etc.).
        discovery_mode (SemanticDiscoveryMode): Modo de descoberta semântica utilizado.
        questions_frozen (bool): Indica se a lista de perguntas está congelada.
        proposal (SemanticQuestionProposal): Perguntas que originaram as features.
        provider_id (str): Provedor de inferência executado.
        requested_model (str): Modelo solicitado.
        resolved_model (str | None): Modelo exato resolvido.
        row_count (int): Quantidade total de linhas do dataset.
        processed_rows (int): Linhas concluídas até o momento.
        request_count (int): Requisições disparadas.
        cache_hits (int): Respostas reaproveitadas do cache local.
        input_tokens (int | None): Tokens consumidos.
        estimated_cost (float | None): Custo estimado acumulado.
        latency_ms_total (float | None): Latência acumulada de inferência em ms.
        artifact_path (str | None): Caminho do arquivo Parquet com as features.
        impact (dict[str, Any] | None): Relatório comparativo de impacto preditivo.
        error_message (str | None): Mensagem de erro caso tenha falhado.
        created_at (datetime): Timestamp de criação.
        started_at (datetime | None): Timestamp de início.
        finished_at (datetime | None): Timestamp de conclusão.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    dataset_id: str
    source_experiment_id: str | None
    name: str
    version: str
    status: str
    discovery_mode: SemanticDiscoveryMode
    questions_frozen: bool
    proposal: SemanticQuestionProposal
    provider_id: str
    requested_model: str
    resolved_model: str | None
    row_count: int
    processed_rows: int
    request_count: int
    cache_hits: int
    input_tokens: int | None
    estimated_cost: float | None
    latency_ms_total: float | None
    artifact_path: str | None
    impact: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def _http_error(error: Exception) -> HTTPException:
    """Mapeia erros de domínio e infraestrutura de features semânticas para HTTPException.

    Args:
        error (Exception): Exceção original levantada.

    Returns:
        HTTPException: Exceção HTTP com código e mensagem estruturada.
    """
    if isinstance(error, EntityNotFoundError | JobNotFoundError):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, DatasetIngestionError):
        code = (
            status.HTTP_404_NOT_FOUND
            if error.code == "dataset_not_found"
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    elif isinstance(error, LLMConfigurationError):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(error, LLMProviderError):
        code = status.HTTP_502_BAD_GATEWAY
    else:
        code = status.HTTP_409_CONFLICT
    return HTTPException(
        status_code=code,
        detail={"code": type(error).__name__, "message": str(error)},
    )


def _response(feature_set: SemanticFeatureSet) -> SemanticFeatureSetResponse:
    """Converte a entidade SemanticFeatureSet no modelo serializável da API.

    Args:
        feature_set (SemanticFeatureSet): Entidade do conjunto semântico.

    Returns:
        SemanticFeatureSetResponse: Objeto de resposta completo.
    """
    return SemanticFeatureSetResponse(
        id=feature_set.id,
        dataset_id=feature_set.dataset_id,
        source_experiment_id=feature_set.source_experiment_id,
        name=feature_set.name,
        version=feature_set.version,
        status=feature_set.status.value,
        discovery_mode=feature_set.discovery_mode,
        questions_frozen=feature_set.questions_frozen,
        proposal=SemanticQuestionProposal(questions=feature_set.questions),
        provider_id=feature_set.provider_id,
        requested_model=feature_set.requested_model,
        resolved_model=feature_set.resolved_model,
        row_count=feature_set.row_count,
        processed_rows=feature_set.processed_rows,
        request_count=feature_set.request_count,
        cache_hits=feature_set.cache_hits,
        input_tokens=feature_set.input_tokens,
        estimated_cost=feature_set.estimated_cost,
        latency_ms_total=feature_set.latency_ms_total,
        artifact_path=feature_set.artifact_path,
        impact=feature_set.impact,
        error_message=feature_set.error_message,
        created_at=feature_set.created_at,
        started_at=feature_set.started_at,
        finished_at=feature_set.finished_at,
    )


@router.post("/propose", response_model=SemanticProposalResponse)
def propose_semantic_features(
    request: SemanticProposalRequest,
    proposer: Annotated[SemanticFeatureProposer, Depends(get_semantic_feature_proposer)],
) -> SemanticProposalResponse:
    """Propõe novas features semânticas com base no objetivo e colunas selecionadas.

    Args:
        request (SemanticProposalRequest): Parâmetros da requisição de proposta semântica.
        proposer (SemanticFeatureProposer): Serviço proponente de features semânticas.

    Returns:
        SemanticProposalResponse: Proposta de features semânticas gerada com estimativas.

    Raises:
        HTTPException: Se o dataset não existir ou se houver falha de validação ou LLM.
    """
    try:
        result: SemanticProposalResult = proposer.propose(
            dataset_id=request.dataset_id,
            objective=request.objective,
            selected_columns=request.selected_columns,
            language=request.language,
            discovery_mode=request.discovery_mode,
        )
        dataset = proposer.datasets.get_dataset(request.dataset_id)
    except (
        DatasetIngestionError,
        DomainError,
        LLMConfigurationError,
        LLMProviderError,
    ) as error:
        raise _http_error(error) from error
    groups = {tuple(sorted(question.input_columns)) for question in result.proposal.questions}
    return SemanticProposalResponse(
        **result.model_dump(),
        estimated_requests=dataset.row_count * len(groups),
    )


@router.post("/generate", response_model=SemanticJobResponse, status_code=status.HTTP_202_ACCEPTED)
def generate_semantic_features(
    request: SemanticGenerateRequest,
    service: Annotated[SemanticFeatureService, Depends(get_semantic_feature_service)],
) -> SemanticJobResponse:
    """Submete um job para geração e materialização de features semânticas.

    Args:
        request (SemanticGenerateRequest): Dados da solicitação de geração semântica.
        service (SemanticFeatureService): Serviço de ciclo de vida de features semânticas.

    Returns:
        SemanticJobResponse: Identificadores do conjunto semântico e do job assíncrono.

    Raises:
        HTTPException: Se o dataset não for encontrado ou se houver violação de domínio.
    """
    try:
        submission = service.submit(
            dataset_id=request.dataset_id,
            name=request.name,
            discovery_mode=request.discovery_mode,
            proposal=request.proposal,
            provider_id=request.provider_id,
            model=request.model,
            external_processing_acknowledged=request.external_processing_acknowledged,
            source_experiment_id=request.source_experiment_id,
        )
    except (DatasetIngestionError, DomainError) as error:
        raise _http_error(error) from error
    return SemanticJobResponse(
        semantic_feature_set_id=submission.feature_set.id,
        job_id=submission.job.job_id,
        status=submission.job.status.value,
    )


@router.get("", response_model=list[SemanticFeatureSetResponse])
def list_semantic_feature_sets(
    repository: Annotated[
        SqlAlchemySemanticFeatureSetRepository,
        Depends(get_semantic_feature_set_repository),
    ],
) -> list[SemanticFeatureSetResponse]:
    """Lista todos os conjuntos de features semânticas persistidos.

    Args:
        repository (SqlAlchemySemanticFeatureSetRepository): Repositório de feature sets.

    Returns:
        list[SemanticFeatureSetResponse]: Lista detalhada dos conjuntos existentes.
    """
    return [_response(item) for item in repository.list_all()]


@router.get("/{feature_set_id}", response_model=SemanticFeatureSetResponse)
def get_semantic_feature_set(
    feature_set_id: str,
    repository: Annotated[
        SqlAlchemySemanticFeatureSetRepository,
        Depends(get_semantic_feature_set_repository),
    ],
) -> SemanticFeatureSetResponse:
    """Recupera os detalhes de um conjunto específico de features semânticas.

    Args:
        feature_set_id (str): Identificador do conjunto de features semânticas.
        repository (SqlAlchemySemanticFeatureSetRepository): Repositório de feature sets.

    Returns:
        SemanticFeatureSetResponse: Detalhes do conjunto encontrado.

    Raises:
        HTTPException: Se o conjunto semântico não for localizado.
    """
    feature_set = repository.get(feature_set_id)
    if feature_set is None:
        raise _http_error(
            EntityNotFoundError(f"Semantic feature set {feature_set_id} was not found")
        )
    return _response(feature_set)


@router.get("/{feature_set_id}/impact", response_model=dict[str, Any])
def get_semantic_feature_impact(
    feature_set_id: str,
    repository: Annotated[
        SqlAlchemySemanticFeatureSetRepository,
        Depends(get_semantic_feature_set_repository),
    ],
) -> dict[str, Any]:
    """Recupera a análise de impacto calculada para o conjunto semântico.

    Args:
        feature_set_id (str): Identificador do conjunto de features semânticas.
        repository (SqlAlchemySemanticFeatureSetRepository): Repositório de feature sets.

    Returns:
        dict[str, Any]: Dicionário com métricas de impacto no baseline e modelos.

    Raises:
        HTTPException: Se o conjunto não existir ou o impacto ainda não estiver disponível.
    """
    feature_set = repository.get(feature_set_id)
    if feature_set is None:
        raise _http_error(
            EntityNotFoundError(f"Semantic feature set {feature_set_id} was not found")
        )
    if feature_set.impact is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "SemanticImpactUnavailable"},
        )
    return feature_set.impact


@router.post("/{feature_set_id}/explanation", response_model=ExplanationResult)
def explain_semantic_impact(
    feature_set_id: str,
    request: SemanticImpactExplanationRequest,
    repository: Annotated[
        SqlAlchemySemanticFeatureSetRepository,
        Depends(get_semantic_feature_set_repository),
    ],
    service: Annotated[ExplanationService, Depends(get_explanation_service)],
    catalog: Annotated[RegistryCatalog, Depends(get_registry_catalog)],
    database: Annotated[Database, Depends(get_database)],
) -> ExplanationResult:
    """Gera uma explicação em linguagem natural do impacto das features semânticas.

    Args:
        feature_set_id (str): Identificador do conjunto semântico.
        request (SemanticImpactExplanationRequest): Parâmetros como o idioma da explicação.
        repository (SqlAlchemySemanticFeatureSetRepository): Repositório de feature sets.
        service (ExplanationService): Serviço de geração de explicações textuais.
        catalog (RegistryCatalog): Catálogo de algoritmos e métricas registrados.
        database (Database): Conexão com o banco de dados para buscar o experimento.

    Returns:
        ExplanationResult: Texto explicativo e metadados de geração via LLM.

    Raises:
        HTTPException: Se o conjunto não for achado, estiver incompleto ou sem experimento base.
    """
    feature_set = repository.get(feature_set_id)
    if feature_set is None:
        raise HTTPException(status_code=404, detail={"code": "SemanticFeatureSetNotFound"})
    if feature_set.status.value != "completed" or not (feature_set.impact or {}).get("models"):
        raise HTTPException(status_code=409, detail={"code": "SemanticImpactUnavailable"})
    experiment = (
        SqlAlchemyExperimentRepository(database).get(feature_set.source_experiment_id)
        if feature_set.source_experiment_id
        else None
    )
    if experiment is None:
        raise HTTPException(status_code=409, detail={"code": "SourceExperimentUnavailable"})
    context = build_semantic_impact_context(feature_set, experiment, catalog)
    try:
        return service.explain(scope="semantic_impact", language=request.language, context=context)
    except ExplanationContextError as error:
        raise HTTPException(status_code=413, detail={"code": error.code}) from error
    except (LLMConfigurationError, LLMProviderError) as error:
        raise _http_error(error) from error


@router.post("/{feature_set_id}/cancel", response_model=SemanticJobResponse)
def cancel_semantic_features(
    feature_set_id: str,
    executor: Annotated[JobExecutor, Depends(get_semantic_job_executor)],
) -> SemanticJobResponse:
    """Cancela o processamento de um job de features semânticas em execução.

    Args:
        feature_set_id (str): Identificador do conjunto cujas features estão sendo geradas.
        executor (JobExecutor): Executor de background jobs do subsistema semântico.

    Returns:
        SemanticJobResponse: Status atualizado do job cancelado.

    Raises:
        HTTPException: Se o job não for encontrado ou se houver erro de cancelamento.
    """
    try:
        snapshot = executor.find_for_experiment(feature_set_id)
        if snapshot is None:
            raise JobNotFoundError(f"No job was found for semantic set {feature_set_id}")
        cancelled = executor.cancel(snapshot.job_id)
    except DomainError as error:
        raise _http_error(error) from error
    return SemanticJobResponse(
        semantic_feature_set_id=feature_set_id,
        job_id=cancelled.job_id,
        status=cancelled.status.value,
    )


@router.get("/{feature_set_id}/events")
def semantic_feature_events(
    feature_set_id: str,
    executor: Annotated[JobExecutor, Depends(get_semantic_job_executor)],
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> StreamingResponse:
    """Transmite eventos SSE em tempo real do processamento de features semânticas.

    Args:
        feature_set_id (str): Identificador do conjunto de features semânticas.
        executor (JobExecutor): Executor de jobs semânticos.
        after (int): Ponto inicial de sequência de eventos a transmitir. Padrão 0.
        last_event_id (int | None): Último ID recebido pelo cliente para reconexão.

    Returns:
        StreamingResponse: Stream Server-Sent Events (SSE) contendo os eventos do job.

    Raises:
        HTTPException: Se o job não for encontrado.
    """
    snapshot = executor.find_for_experiment(feature_set_id)
    if snapshot is None:
        raise _http_error(JobNotFoundError(f"No job was found for {feature_set_id}"))
    cursor = max(after, last_event_id or 0)

    async def stream():
        nonlocal cursor
        while True:
            for event in executor.events(snapshot.job_id, after=cursor):
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
