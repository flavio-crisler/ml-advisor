from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_decision_provider_registry,
    get_decision_provider_runtime_config,
)
from app.application.decisions.errors import DecisionProviderConfigurationError
from app.application.decisions.registry import (
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
    DecisionProviderSummary,
)
from app.domain.decision.provider import ProviderHealth

router = APIRouter(prefix="/decision-providers", tags=["decision-providers"])
RegistryDependency = Annotated[
    DecisionProviderRegistry,
    Depends(get_decision_provider_registry),
]
ConfigDependency = Annotated[
    DecisionProviderRuntimeConfig,
    Depends(get_decision_provider_runtime_config),
]


@router.get("", response_model=list[DecisionProviderSummary])
def list_decision_providers(
    registry: RegistryDependency,
    config: ConfigDependency,
) -> tuple[DecisionProviderSummary, ...]:
    """Lista todos os provedores de decisão registrados e seus status de configuração.

    Args:
        registry (RegistryDependency): Registro injetado de provedores de decisão.
        config (ConfigDependency): Configuração de runtime atual.

    Returns:
        tuple[DecisionProviderSummary, ...]: Tupla com os resumos dos provedores.
    """
    return registry.summaries(config)


@router.post("/test", response_model=ProviderHealth)
async def test_decision_provider(
    registry: RegistryDependency,
    config: ConfigDependency,
) -> ProviderHealth:
    """Realiza teste de conectividade e liveness com o provedor de decisão configurado.

    Args:
        registry (RegistryDependency): Registro injetado de provedores.
        config (ConfigDependency): Configuração de runtime atual.

    Returns:
        ProviderHealth: Estado de saúde operacional e métricas de conectividade.

    Raises:
        HTTPException: Se as features estiverem desativadas ou o provedor desconfigurado (409).
    """
    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "decision_features_disabled"},
        )
    try:
        provider = registry.create(config.provider_id, config)
    except DecisionProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "decision_provider_not_configured"},
        ) from exc
    return await provider.health_check()
