from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.application.decisions.errors import DecisionProviderConfigurationError
from app.domain.decision.provider import (
    DecisionProvider,
    DecisionProviderCapabilities,
    ProviderHealthStatus,
)


class DecisionProviderRuntimeConfig(BaseModel):
    """Configurações em tempo de execução para clientes de provedores de decisão.

    Attributes:
        enabled (bool): Habilita ou desabilita o uso de provedores de decisão.
        provider_id (str): Identificador do provedor configurado.
        model (str): Nome ou caminho do modelo utilizado no provedor.
        base_url (str): URL base para chamadas de API do provedor.
        api_key (SecretStr): Chave de API protegida para autenticação.
        timeout_seconds (float): Tempo limite em segundos por requisição.
        max_retries (int): Número máximo de tentativas em caso de erro transiente.
        concurrency (int): Quantidade de requisições concorrentes permitidas.
        cache_enabled (bool): Ativa cache de respostas idênticas.
        external_processing_ack_required (bool): Exige consentimento para envio externo de dados.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    provider_id: str = Field(default="typesafe_jev", pattern=r"^[a-z0-9][a-z0-9_-]*$")
    model: str = ""
    base_url: str = ""
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    concurrency: int = Field(default=8, ge=1, le=128)
    cache_enabled: bool = True
    external_processing_ack_required: bool = True


DecisionProviderFactory = Callable[[DecisionProviderRuntimeConfig], DecisionProvider]


@dataclass(frozen=True)
class DecisionProviderRegistration:
    """Metadados de registro e fábrica de instanciação de um provedor de decisão.

    Attributes:
        provider_id (str): Identificador único do provedor.
        display_name (str): Nome amigável para exibição na interface.
        capabilities (DecisionProviderCapabilities): Capacidades operacionais do provedor.
        factory (DecisionProviderFactory | None): Função fábrica para criar o cliente do provedor.
    """

    provider_id: str
    display_name: str
    capabilities: DecisionProviderCapabilities
    factory: DecisionProviderFactory | None = None


class DecisionProviderSummary(BaseModel):
    """Resumo consolidado do status, credenciais e saúde de um provedor de decisão.

    Attributes:
        schema_version (Literal["1.0"]): Versão do schema de resumo.
        id (str): Identificador do provedor.
        display_name (str): Nome de exibição.
        capabilities (DecisionProviderCapabilities): Capacidades suportadas.
        is_external (bool): Indica se o processamento é executado externamente.
        feature_enabled (bool): Indica se a funcionalidade está globalmente ativa.
        selected (bool): Indica se este provedor é o selecionado no momento.
        configured (bool): Indica se o provedor possui credenciais válidas preenchidas.
        available (bool): Indica se o provedor está pronto para uso e instanciável.
        model (str | None): Nome do modelo configurado, se selecionado.
        health_status (ProviderHealthStatus): Estado de saúde e conectividade.
        health_message_key (str | None): Chave i18n para detalhamento da saúde.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    id: str
    display_name: str
    capabilities: DecisionProviderCapabilities
    is_external: bool
    feature_enabled: bool
    selected: bool
    configured: bool
    available: bool
    model: str | None = None
    health_status: ProviderHealthStatus
    health_message_key: str | None = None


class DecisionProviderRegistry:
    """Registro e gerenciador de fábricas de instâncias de provedores de decisão.

    Attributes:
        _registrations (dict[str, DecisionProviderRegistration]): Dicionário interno de registros.
    """

    def __init__(self) -> None:
        """Inicializa o registro vazio de provedores de decisão."""
        self._registrations: dict[str, DecisionProviderRegistration] = {}

    def register(self, registration: DecisionProviderRegistration) -> None:
        """Registra um provedor de decisão no catálogo interno.

        Args:
            registration (DecisionProviderRegistration): Metadados e fábrica do provedor.

        Raises:
            DecisionProviderConfigurationError: Se o ID do provedor já estiver registrado.
        """
        if registration.provider_id in self._registrations:
            raise DecisionProviderConfigurationError(
                f"Decision provider '{registration.provider_id}' is already registered."
            )
        self._registrations[registration.provider_id] = registration

    def ids(self) -> tuple[str, ...]:
        """Retorna a tupla com todos os identificadores de provedores cadastrados.

        Returns:
            tuple[str, ...]: Identificadores registrados.
        """
        return tuple(self._registrations)

    def create(
        self,
        provider_id: str,
        config: DecisionProviderRuntimeConfig,
    ) -> DecisionProvider:
        """Cria e inicializa a instância do provedor usando a configuração fornecida.

        Args:
            provider_id (str): Identificador do provedor desejado.
            config (DecisionProviderRuntimeConfig): Configurações de execução e credenciais.

        Returns:
            DecisionProvider: Instância operacional do provedor.

        Raises:
            DecisionProviderConfigurationError: Se o provedor não for suportado ou não existir.
        """
        registration = self._get(provider_id)
        if registration.factory is None:
            raise DecisionProviderConfigurationError(
                f"Decision provider '{provider_id}' is not available in this build."
            )
        return registration.factory(config)

    def summaries(
        self, config: DecisionProviderRuntimeConfig
    ) -> tuple[DecisionProviderSummary, ...]:
        """Gera os resumos de status e disponibilidade para todos os provedores registrados.

        Args:
            config (DecisionProviderRuntimeConfig): Configuração atual do ambiente.

        Returns:
            tuple[DecisionProviderSummary, ...]: Lista de resumos de cada provedor.
        """
        selected_id = _canonical_provider_id(config.provider_id)
        return tuple(
            self._summary(registration, config, selected_id)
            for registration in self._registrations.values()
        )

    def _get(self, provider_id: str) -> DecisionProviderRegistration:
        """Recupera o registro do provedor resolvendo eventuais aliases.

        Args:
            provider_id (str): Identificador ou alias do provedor.

        Returns:
            DecisionProviderRegistration: Registro associado.

        Raises:
            DecisionProviderConfigurationError: Se o provedor não for encontrado.
        """
        canonical_id = _canonical_provider_id(provider_id)
        try:
            return self._registrations[canonical_id]
        except KeyError as exc:
            raise DecisionProviderConfigurationError(
                f"Decision provider '{provider_id}' is not registered."
            ) from exc

    @staticmethod
    def _summary(
        registration: DecisionProviderRegistration,
        config: DecisionProviderRuntimeConfig,
        selected_id: str,
    ) -> DecisionProviderSummary:
        """Constrói o objeto de resumo avaliando credenciais e status de seleção.

        Args:
            registration (DecisionProviderRegistration): Registro do provedor.
            config (DecisionProviderRuntimeConfig): Configuração ativa no sistema.
            selected_id (str): Identificador do provedor atualmente selecionado.

        Returns:
            DecisionProviderSummary: Resumo com flags e estado de saúde inferido.
        """
        selected = registration.provider_id == selected_id
        has_credentials = bool(config.model.strip() and config.api_key.get_secret_value())
        configured = bool(config.enabled and selected and has_credentials)
        available = bool(configured and registration.factory is not None)

        if not config.enabled:
            health_status = "disabled"
            health_message_key = "decisionProviders.status.featureDisabled"
        elif not selected:
            health_status = "unknown"
            health_message_key = "decisionProviders.status.notSelected"
        elif not has_credentials:
            health_status = "unavailable"
            health_message_key = "decisionProviders.status.notConfigured"
        elif registration.factory is None:
            health_status = "unavailable"
            health_message_key = "decisionProviders.status.adapterUnavailable"
        else:
            health_status = "unknown"
            health_message_key = "decisionProviders.status.notChecked"

        return DecisionProviderSummary(
            id=registration.provider_id,
            display_name=registration.display_name,
            capabilities=registration.capabilities,
            is_external=registration.capabilities.is_external,
            feature_enabled=config.enabled,
            selected=selected,
            configured=configured,
            available=available,
            model=config.model if selected and config.model else None,
            health_status=health_status,
            health_message_key=health_message_key,
        )


def _canonical_provider_id(provider_id: str) -> str:
    """Normaliza aliases conhecidos de provedores para seus identificadores canônicos.

    Args:
        provider_id (str): Identificador ou alias do provedor.

    Returns:
        str: Identificador canônico correspondente.
    """
    aliases = {"typesafe": "typesafe_jev", "openrouter": "openrouter_jev"}
    return aliases.get(provider_id, provider_id)


def build_default_decision_provider_registry(
    *,
    typesafe_factory: DecisionProviderFactory | None = None,
    openrouter_factory: DecisionProviderFactory | None = None,
    typesafe_capabilities: DecisionProviderCapabilities | None = None,
    openrouter_capabilities: DecisionProviderCapabilities | None = None,
) -> DecisionProviderRegistry:
    """Constrói o registro padrão inicializado com os provedores TypeSafe e OpenRouter.

    Args:
        typesafe_factory (DecisionProviderFactory | None): Fábrica para o provedor TypeSafe.
        openrouter_factory (DecisionProviderFactory | None): Fábrica para o provedor OpenRouter.
        typesafe_capabilities (DecisionProviderCapabilities | None): Capacidades customizadas.
        openrouter_capabilities (DecisionProviderCapabilities | None): Capacidades customizadas.

    Returns:
        DecisionProviderRegistry: Instância do registro populada com as definições padrão.
    """
    registry = DecisionProviderRegistry()
    planned_capabilities = DecisionProviderCapabilities(
        supports_noul=True,
        supports_choice=True,
        supports_score=True,
        supports_batch=False,
        supports_model_version_resolution=False,
        is_external=True,
    )
    registry.register(
        DecisionProviderRegistration(
            provider_id="typesafe_jev",
            display_name="TypeSafe Jev",
            capabilities=typesafe_capabilities or planned_capabilities,
            factory=typesafe_factory,
        )
    )
    registry.register(
        DecisionProviderRegistration(
            provider_id="openrouter_jev",
            display_name="OpenRouter Jev",
            capabilities=openrouter_capabilities or planned_capabilities,
            factory=openrouter_factory,
        )
    )
    return registry
