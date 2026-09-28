from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_auto_workflow_service
from app.application.advisor.errors import AdvisorRecommendationError
from app.application.advisor.models import RecommendationSpec
from app.application.auto.service import AutoWorkflowService
from app.application.datasets.errors import DatasetIngestionError, DatasetNotFoundError
from app.domain.exceptions import DomainError
from app.domain.experiment.spec import ExperimentSpec
from app.llm.errors import LLMConfigurationError, LLMProviderError

router = APIRouter(prefix="/auto", tags=["auto"])


class AutoRunRequest(BaseModel):
    """Payload de entrada para disparo do fluxo autônomo ponta a ponta.

    Attributes:
        dataset_id (str): Identificador do dataset a ser utilizado.
        objective (str): Meta de negócio ou pergunta analítica a resolver.
        name (str | None): Nome opcional atribuído ao experimento.
        language (Literal["pt-BR", "en-US"]): Idioma para geração das justificativas.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=2000)
    name: str | None = Field(default=None, max_length=200)
    language: Literal["pt-BR", "en-US"] = "pt-BR"


class AutoRunResponse(BaseModel):
    """Resposta com detalhes do experimento e job iniciados no modo autônomo.

    Attributes:
        experiment_id (str): Identificador do experimento registrado.
        job_id (str): Identificador do job de treinamento em background.
        job_status (str): Status operacional inicial do job.
        recommendation (RecommendationSpec): Recomendação analítica gerada pelo Advisor.
        proposal (ExperimentSpec): Especificação técnica do experimento criado.
        provider_id (str): Identificador do provedor LLM consultado.
        sample_sent (bool): Indica se uma amostra do dataset foi enviada ao LLM.
        sample_rows (int): Quantidade de linhas enviadas na amostra.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    job_id: str
    job_status: str
    recommendation: RecommendationSpec
    proposal: ExperimentSpec
    provider_id: str
    sample_sent: bool
    sample_rows: int


@router.post("/run", response_model=AutoRunResponse, status_code=status.HTTP_202_ACCEPTED)
def run_auto_workflow(
    request: AutoRunRequest,
    service: Annotated[AutoWorkflowService, Depends(get_auto_workflow_service)],
) -> AutoRunResponse:
    """Executa o pipeline autônomo completo (recomendação -> criação -> treinamento).

    Args:
        request (AutoRunRequest): Parâmetros do dataset e objetivo.
        service (AutoWorkflowService): Serviço orquestrador injetado via Depends.

    Returns:
        AutoRunResponse: Metadados do experimento e job assíncrono em execução.

    Raises:
        HTTPException: Se o dataset não existir (404), houver erro de parsing (400),
            a recomendação for inválida (422), a LLM estiver desconfigurada (503),
            o provedor externo falhar (502) ou ocorrer conflito de domínio (409).
    """
    try:
        result = service.execute(
            dataset_id=request.dataset_id,
            objective=request.objective,
            language=request.language,
            name=request.name,
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
    except DomainError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": type(exc).__name__, "message": str(exc)},
        ) from exc
    return AutoRunResponse(
        experiment_id=result.experiment.id,
        job_id=result.job.job_id,
        job_status=result.job.status.value,
        recommendation=result.advisor.recommendation,
        proposal=result.advisor.proposal,
        provider_id=result.advisor.provider_id,
        sample_sent=result.advisor.sample_sent,
        sample_rows=result.advisor.sample_rows,
    )
