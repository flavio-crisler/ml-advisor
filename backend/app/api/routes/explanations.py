import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import get_explanation_service
from app.application.explanations.models import (
    ExplanationAvailability,
    ExplanationLanguage,
    ExplanationResult,
    ExplanationScope,
)
from app.application.explanations.prompts import ExplanationContextError
from app.application.explanations.service import ExplanationService
from app.llm.errors import LLMConfigurationError, LLMProviderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explanations", tags=["explanations"])


class ExplanationRequest(BaseModel):
    """Payload de entrada para requisição de explicação interpretativa com LLM.

    Attributes:
        scope (ExplanationScope): Escopo temático (ex: run_diagnostics, experiment_comparison).
        language (ExplanationLanguage): Código do idioma de destino ('pt-BR' ou 'en-US').
        context (dict[str, Any]): Dicionário com métricas agregadas e parâmetros necessários.
    """

    model_config = ConfigDict(extra="forbid")

    scope: ExplanationScope
    language: ExplanationLanguage
    context: dict[str, Any]


@router.get("/status", response_model=ExplanationAvailability)
def explanation_status(
    service: Annotated[ExplanationService, Depends(get_explanation_service)],
) -> ExplanationAvailability:
    """Verifica a disponibilidade operacional do serviço de explicações e do LLM.

    Args:
        service (ExplanationService): Serviço de explicações injetado.

    Returns:
        ExplanationAvailability: Status de suporte, disponibilidade e provedor ativo.
    """
    return service.availability()


@router.post("", response_model=ExplanationResult)
def explain_result(
    request: ExplanationRequest,
    service: Annotated[ExplanationService, Depends(get_explanation_service)],
) -> ExplanationResult:
    """Gera uma explicação em linguagem natural para um resultado ou comparação.

    Args:
        request (ExplanationRequest): Escopo, idioma e dados de contexto agregado.
        service (ExplanationService): Serviço de explicações injetado.

    Returns:
        ExplanationResult: Justificativa estruturada e narrativa textual produzida pelo LLM.

    Raises:
        HTTPException: Se o contexto exceder o limite (413), a LLM estiver desconfigurada (503)
            ou o provedor de IA falhar (502).
    """
    try:
        return service.explain(
            scope=request.scope,
            language=request.language,
            context=request.context,
        )
    except ExplanationContextError as error:
        logger.warning("Explanation context rejected: %s", error)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": error.code},
        ) from error
    except LLMConfigurationError as error:
        logger.warning("LLM configuration error during explanation: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": error.code},
        ) from error
    except LLMProviderError as error:
        logger.exception("LLM provider error during explanation: %s", error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": error.code},
        ) from error
