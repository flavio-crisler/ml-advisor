from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_improvement_proposal_service
from app.application.improvements.models import (
    ImprovementDraftRequest,
    ImprovementDraftResult,
    ImprovementProposalRequest,
    SupervisedImprovementProposal,
)
from app.application.improvements.prompts import ImprovementContextError
from app.application.improvements.service import ImprovementProposalService
from app.domain.exceptions import EntityNotFoundError, ExperimentSpecValidationError
from app.llm.errors import LLMConfigurationError, LLMProviderError

router = APIRouter(prefix="/improvement-proposals", tags=["improvement-proposals"])


def _proposal_error(error: Exception) -> HTTPException:
    """Mapeia exceções de domínio, contexto e LLM em HTTPException apropriadas.

    Args:
        error (Exception): Exceção original levantada no serviço de melhorias.

    Returns:
        HTTPException: Exceção HTTP com código e detalhes estruturados.
    """
    if isinstance(error, EntityNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": type(error).__name__},
        )
    if isinstance(error, ExperimentSpecValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": type(error).__name__, "issues": list(error.issues)},
        )
    if isinstance(error, ImprovementContextError):
        return HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": error.code},
        )
    if isinstance(error, LLMConfigurationError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": error.code},
        )
    if isinstance(error, LLMProviderError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": error.code},
        )
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": "improvement_proposal_invalid"},
    )


@router.post("/supervised", response_model=SupervisedImprovementProposal)
def propose_supervised_improvement(
    request: ImprovementProposalRequest,
    service: Annotated[
        ImprovementProposalService,
        Depends(get_improvement_proposal_service),
    ],
) -> SupervisedImprovementProposal:
    """Solicita propostas de melhoria estruturada para um modelo supervisionado via LLM.

    Args:
        request (ImprovementProposalRequest): Experimento de origem, modelo e contexto de métricas.
        service (ImprovementProposalService): Serviço de melhorias injetado via Depends.

    Returns:
        SupervisedImprovementProposal: Proposta contendo recomendações e diagnósticos.

    Raises:
        HTTPException: Se o experimento não existir (404), a spec for inválida (422),
            o contexto for excessivo (413) ou houver falha de IA (502/503).
    """
    try:
        return service.propose(
            language=request.language,
            source_experiment_id=request.source_experiment_id,
            source_model_run_id=request.source_model_run_id,
            context=request.context,
        )
    except (
        EntityNotFoundError,
        ExperimentSpecValidationError,
        ImprovementContextError,
        LLMProviderError,
        ValueError,
    ) as error:
        raise _proposal_error(error) from error


@router.post(
    "/supervised/draft",
    response_model=ImprovementDraftResult,
    status_code=status.HTTP_201_CREATED,
)
def create_supervised_improvement_draft(
    request: ImprovementDraftRequest,
    service: Annotated[
        ImprovementProposalService,
        Depends(get_improvement_proposal_service),
    ],
) -> ImprovementDraftResult:
    """Cria um novo experimento rascunho a partir de uma proposta de melhoria aprovada.

    Args:
        request (ImprovementDraftRequest): Identificadores de origem e recomendação selecionada.
        service (ImprovementProposalService): Serviço de melhorias injetado via Depends.

    Returns:
        ImprovementDraftResult: Novo experimento rascunho com a nova especificação ajustada.

    Raises:
        HTTPException: Se os dados base não forem encontrados (404) ou a spec for inválida (422).
    """
    try:
        return service.create_draft(
            source_experiment_id=request.source_experiment_id,
            source_model_run_id=request.source_model_run_id,
            source_spec_hash=request.source_spec_hash,
            provider_id=request.provider_id,
            recommendation=request.recommendation,
        )
    except (EntityNotFoundError, ExperimentSpecValidationError, ValueError) as error:
        raise _proposal_error(error) from error
