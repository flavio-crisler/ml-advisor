from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_advisor_service
from app.application.advisor.errors import AdvisorRecommendationError
from app.application.advisor.models import AdvisorResult
from app.application.advisor.service import AdvisorService
from app.application.datasets.errors import DatasetIngestionError, DatasetNotFoundError
from app.llm.errors import LLMConfigurationError, LLMProviderError

router = APIRouter(prefix="/advisor", tags=["advisor"])


class AdvisorRecommendationRequest(BaseModel):
    """Payload de entrada para solicitação de recomendação ao Advisor.

    Attributes:
        dataset_id (str): Identificador do dataset a ser analisado.
        objective (str): Descrição do problema e meta analítica em linguagem natural.
        name (str | None): Nome sugerido para a especificação do experimento.
        language (Literal["pt-BR", "en-US"]): Idioma para as justificativas geradas.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=2000)
    name: str | None = Field(default=None, max_length=200)
    language: Literal["pt-BR", "en-US"] = "pt-BR"


@router.post("/recommend", response_model=AdvisorResult)
def recommend(
    request: AdvisorRecommendationRequest,
    service: Annotated[AdvisorService, Depends(get_advisor_service)],
) -> AdvisorResult:
    """Solicita ao Advisor uma recomendação completa de modelagem baseada em LLM.

    Args:
        request (AdvisorRecommendationRequest): Parâmetros do dataset e objetivo.
        service (AdvisorService): Serviço injetado do Advisor.

    Returns:
        AdvisorResult: Especificação de experimento validada e justificativa em texto.

    Raises:
        HTTPException: Se o dataset não for encontrado (404), houver falha de ingestão (400),
            a recomendação violar o contrato de validação (422), a LLM estiver
            desconfigurada (503) ou o provedor falhar (502).
    """
    try:
        return service.recommend(
            dataset_id=request.dataset_id,
            objective=request.objective,
            name=request.name,
            language=request.language,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": exc.code, "message_key": exc.message_key},
        ) from exc
    except DatasetIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message_key": exc.message_key},
        ) from exc
    except AdvisorRecommendationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": exc.code, "issues": exc.issues},
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": exc.code, "message_key": f"advisor.errors.{exc.code}"},
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": exc.code, "message_key": f"advisor.errors.{exc.code}"},
        ) from exc
