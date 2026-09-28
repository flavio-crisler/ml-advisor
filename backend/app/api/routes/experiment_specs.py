from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import get_experiment_service
from app.application.experiments.service import ExperimentService
from app.domain.exceptions import (
    EntityNotFoundError,
    ExperimentExecutionError,
    ExperimentSpecValidationError,
)
from app.domain.experiment.spec import ExperimentSpec

router = APIRouter(prefix="/experiment-specs", tags=["experiment-specs"])


class ExperimentSpecValidationResponse(BaseModel):
    """Resposta confirmando a validação bem-sucedida de uma especificação de experimento.

    Attributes:
        valid (Literal[True]): Flag booleana estrita indicando sucesso na validação.
        spec (ExperimentSpec): Especificação canônica após validação sintática e semântica.
    """

    model_config = ConfigDict(frozen=True)

    valid: Literal[True]
    spec: ExperimentSpec


@router.post("/validate", response_model=ExperimentSpecValidationResponse)
def validate_experiment_spec(
    spec: ExperimentSpec,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
) -> ExperimentSpecValidationResponse:
    """Valida uma especificação de experimento contra o catálogo de modelos e o dataset.

    Args:
        spec (ExperimentSpec): Especificação completa a ser verificada.
        service (ExperimentService): Serviço de experimentos injetado via Depends.

    Returns:
        ExperimentSpecValidationResponse: Confirmação de validação e spec normalizada.

    Raises:
        HTTPException: Se a especificação violar regras de ML (400), o dataset não
            existir (404) ou o hash de conteúdo for divergente (400).
    """
    try:
        validated = service.validate_specification(spec)
    except ExperimentSpecValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_experiment_spec", "issues": error.issues},
        ) from error
    except EntityNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "dataset_not_found", "message": str(error)},
        ) from error
    except ExperimentExecutionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "dataset_hash_mismatch", "message": str(error)},
        ) from error
    return ExperimentSpecValidationResponse(valid=True, spec=validated)
