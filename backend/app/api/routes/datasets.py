from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import get_dataset_service
from app.application.datasets.errors import (
    DatasetIngestionError,
    DatasetNotFoundError,
    DatasetTooLargeError,
)
from app.application.datasets.profiling import DatasetProfile
from app.application.datasets.service import DatasetIngestionResult, DatasetService
from app.application.semantic_features.columns import (
    SemanticColumnOptions,
    semantic_column_options,
)
from app.domain.dataset.entities import Dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetSummaryResponse(BaseModel):
    """Representação resumida dos metadados de um dataset.

    Attributes:
        id (str): Identificador único do dataset.
        name (str): Nome amigável do dataset.
        original_filename (str): Nome do arquivo CSV original enviado no upload.
        hash (str): Hash SHA-256 de integridade e deduplicação do conteúdo.
        row_count (int): Quantidade total de linhas de dados.
        column_count (int): Quantidade de colunas presentes.
        created_at (datetime): Data e hora de criação do registro.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    original_filename: str
    hash: str
    row_count: int
    column_count: int
    created_at: datetime


class DatasetUploadResponse(BaseModel):
    """Payload de retorno após o upload e profilagem de um dataset.

    Attributes:
        dataset (DatasetSummaryResponse): Metadados resumidos do dataset criado.
        profile (DatasetProfile): Perfil estrutural e estatístico das colunas.
        semantic_columns (SemanticColumnOptions): Colunas elegíveis para features semânticas.
        reused (bool): True se o arquivo foi deduplicado por hash idêntico pré-existente.
    """

    model_config = ConfigDict(frozen=True)

    dataset: DatasetSummaryResponse
    profile: DatasetProfile
    semantic_columns: SemanticColumnOptions
    reused: bool


class DatasetDetailResponse(BaseModel):
    """Detalhes completos de um dataset, incluindo perfil e opções semânticas.

    Attributes:
        dataset (DatasetSummaryResponse): Resumo cadastral do dataset.
        profile (DatasetProfile): Perfil analítico e inferência de tipos.
        semantic_columns (SemanticColumnOptions): Colunas candidatas a perguntas semânticas.
    """

    model_config = ConfigDict(frozen=True)

    dataset: DatasetSummaryResponse
    profile: DatasetProfile
    semantic_columns: SemanticColumnOptions


def _summary(dataset: Dataset) -> DatasetSummaryResponse:
    """Converte a entidade de domínio Dataset para o modelo de resumo da API.

    Args:
        dataset (Dataset): Entidade de domínio do dataset.

    Returns:
        DatasetSummaryResponse: Objeto de resposta serializado.
    """
    return DatasetSummaryResponse(
        id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        hash=dataset.content_hash,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        created_at=dataset.created_at,
    )


def _http_error(error: DatasetIngestionError) -> HTTPException:
    """Converte erros de ingestão de dataset em exceções HTTPException apropriadas.

    Args:
        error (DatasetIngestionError): Erro de domínio ou ingestão levantado.

    Returns:
        HTTPException: Exceção HTTP com código e mensagem estruturada.
    """
    status_code = status.HTTP_400_BAD_REQUEST
    if isinstance(error, DatasetTooLargeError):
        status_code = status.HTTP_413_CONTENT_TOO_LARGE
    elif isinstance(error, DatasetNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    return HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message_key": error.message_key},
    )


@router.post("", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    response: Response,
    file: Annotated[UploadFile, File()],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
    name: Annotated[str | None, Form()] = None,
) -> DatasetUploadResponse:
    """Faz o upload de um arquivo CSV, valida schema, gera perfil e persiste metadados.

    Args:
        response (Response): Objeto de resposta para ajuste do status code (200 se reusado).
        file (UploadFile): Arquivo CSV recebido via multipart/form-data.
        service (DatasetService): Serviço de datasets injetado via Depends.
        name (str | None): Nome opcional customizado para o dataset.

    Returns:
        DatasetUploadResponse: Resumo do dataset e relatório completo de profilagem.

    Raises:
        HTTPException: Se o arquivo for inválido (400) ou exceder o limite de tamanho (413).
    """
    try:
        result: DatasetIngestionResult = service.ingest(
            filename=file.filename or "",
            name=name,
            stream=file.file,
        )
    except DatasetIngestionError as error:
        raise _http_error(error) from error
    if result.reused:
        response.status_code = status.HTTP_200_OK
    return DatasetUploadResponse(
        dataset=_summary(result.dataset),
        profile=result.profile,
        semantic_columns=semantic_column_options(result.profile),
        reused=result.reused,
    )


@router.get("", response_model=list[DatasetSummaryResponse])
def list_datasets(
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[DatasetSummaryResponse]:
    """Lista todos os datasets persistidos na plataforma.

    Args:
        service (DatasetService): Serviço de datasets injetado.

    Returns:
        list[DatasetSummaryResponse]: Lista ordenada de resumos cadastrais dos datasets.
    """
    return [_summary(dataset) for dataset in service.list_datasets()]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(
    dataset_id: str,
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetDetailResponse:
    """Retorna detalhes e o perfil estrutural de um dataset específico.

    Args:
        dataset_id (str): Identificador do dataset.
        service (DatasetService): Serviço de datasets injetado.

    Returns:
        DatasetDetailResponse: Detalhes do dataset, perfil estatístico e opções semânticas.

    Raises:
        HTTPException: Se o dataset não for encontrado (404).
    """
    try:
        dataset = service.get_dataset(dataset_id)
        profile = service.get_profile(dataset_id)
    except DatasetIngestionError as error:
        raise _http_error(error) from error
    return DatasetDetailResponse(
        dataset=_summary(dataset),
        profile=profile,
        semantic_columns=semantic_column_options(profile),
    )


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
def get_dataset_profile(
    dataset_id: str,
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetProfile:
    """Retorna exclusivamente o perfil estatístico e tipagem das colunas do dataset.

    Args:
        dataset_id (str): Identificador do dataset.
        service (DatasetService): Serviço de datasets injetado.

    Returns:
        DatasetProfile: Objeto com sumário descritivo e metadados por coluna.

    Raises:
        HTTPException: Se o dataset ou seu arquivo de perfil não forem encontrados (404).
    """
    try:
        return service.get_profile(dataset_id)
    except DatasetIngestionError as error:
        raise _http_error(error) from error
