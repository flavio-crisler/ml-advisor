from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import (
    get_cascade_run_repository,
    get_confidence_cascade_service,
)
from app.application.confidence_cascade.service import ConfidenceCascadeService
from app.domain.decision.cascade import CascadePolicy, CascadeRun
from app.domain.exceptions import DomainError, EntityNotFoundError
from app.infrastructure.repositories.sqlalchemy import SqlAlchemyCascadeRunRepository

router = APIRouter(prefix="/confidence-cascades", tags=["confidence-cascades"])


class CascadeRunResponse(BaseModel):
    """Representação serializada de uma execução de cascata de confiança.

    Attributes:
        id (str): Identificador único da execução da cascata.
        decision_run_id (str): Identificador do benchmark de decisão associado.
        policy (CascadePolicy): Política de threshold e fallback aplicada.
        metrics (dict[str, Any]): Métricas de cobertura, custo e desempenho obtidas.
        artifact_path (str): Caminho para os artefatos de predições resultantes.
        created_at (datetime): Data e hora de criação da execução.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    decision_run_id: str
    policy: CascadePolicy
    metrics: dict[str, Any]
    artifact_path: str
    created_at: datetime


def _response(run: CascadeRun) -> CascadeRunResponse:
    """Converte a entidade de domínio de cascata no modelo de resposta da API.

    Args:
        run (CascadeRun): Entidade de execução da cascata.

    Returns:
        CascadeRunResponse: Objeto de resposta serializável.
    """
    return CascadeRunResponse(
        id=run.id,
        decision_run_id=run.decision_run_id,
        policy=run.policy,
        metrics=run.metrics,
        artifact_path=run.artifact_path,
        created_at=run.created_at,
    )


@router.post("", response_model=CascadeRunResponse, status_code=status.HTTP_201_CREATED)
def create_confidence_cascade(
    policy: CascadePolicy,
    service: Annotated[ConfidenceCascadeService, Depends(get_confidence_cascade_service)],
) -> CascadeRunResponse:
    """Executa e persiste uma nova avaliação de política de cascata de confiança.

    Args:
        policy (CascadePolicy): Parâmetros da política de confiança e fallback.
        service (ConfidenceCascadeService): Serviço de cascatas injetado via Depends.

    Returns:
        CascadeRunResponse: Resultados e métricas da política avaliada.

    Raises:
        HTTPException: Se a execução de benchmark não existir (404) ou violar regras (409).
    """
    try:
        return _response(service.create(policy))
    except DomainError as error:
        code = (
            status.HTTP_404_NOT_FOUND
            if isinstance(error, EntityNotFoundError)
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(
            status_code=code,
            detail={"code": type(error).__name__, "message": str(error)},
        ) from error


@router.get("", response_model=list[CascadeRunResponse])
def list_confidence_cascades(
    repository: Annotated[SqlAlchemyCascadeRunRepository, Depends(get_cascade_run_repository)],
) -> list[CascadeRunResponse]:
    """Lista todas as execuções de cascata de confiança persistidas no sistema.

    Args:
        repository (SqlAlchemyCascadeRunRepository): Repositório injetado de cascatas.

    Returns:
        list[CascadeRunResponse]: Lista de execuções de cascata registradas.
    """
    return [_response(run) for run in repository.list_all()]


@router.get("/{cascade_id}", response_model=CascadeRunResponse)
def get_confidence_cascade(
    cascade_id: str,
    repository: Annotated[SqlAlchemyCascadeRunRepository, Depends(get_cascade_run_repository)],
) -> CascadeRunResponse:
    """Recupera detalhes de uma execução de cascata de confiança pelo identificador.

    Args:
        cascade_id (str): Identificador da execução de cascata.
        repository (SqlAlchemyCascadeRunRepository): Repositório injetado de cascatas.

    Returns:
        CascadeRunResponse: Detalhes e métricas da cascata consultada.

    Raises:
        HTTPException: Se a cascata não for encontrada (404).
    """
    run = repository.get(cascade_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CascadeNotFound"},
        )
    return _response(run)
