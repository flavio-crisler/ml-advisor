from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config.settings import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Payload de resposta para verificação de integridade e liveness do serviço.

    Attributes:
        status (Literal["ok"]): Indicador textual de disponibilidade do sistema.
        version (str): Versão da aplicação atualmente em execução.
    """

    status: Literal["ok"]
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verifica a integridade da API e retorna o status atual e versão instalada.

    Returns:
        HealthResponse: Objeto com status operacional e número de versão.
    """
    return HealthResponse(status="ok", version=get_settings().app_version)
