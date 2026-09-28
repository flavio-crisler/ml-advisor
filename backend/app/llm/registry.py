from collections.abc import Callable

from app.llm.base import LLMProvider, LLMProviderConfig
from app.llm.errors import LLMConfigurationError
from app.llm.providers import OllamaProvider, OpenRouterProvider, VLLMProvider
from app.llm.transport import JsonTransport

ProviderFactory = Callable[[LLMProviderConfig, JsonTransport | None], LLMProvider]


class LLMProviderRegistry:
    """Gerencia o registro e ciclo de vida de instanciação de provedores de LLM.

    Attributes:
        _factories (dict[str, ProviderFactory]): Dicionário com funções construtoras por provedor.
    """

    def __init__(self) -> None:
        """Inicializa o registro de provedores vazio."""
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider_id: str, factory: ProviderFactory) -> None:
        """Registra uma função fábrica para determinado identificador de provedor.

        Args:
            provider_id (str): Identificador único do provedor.
            factory (ProviderFactory): Função de instanciação do adaptador de provedor.

        Raises:
            LLMConfigurationError: Se o provedor já estiver registrado.
        """
        if provider_id in self._factories:
            raise LLMConfigurationError(f"LLM provider '{provider_id}' is already registered.")
        self._factories[provider_id] = factory

    def create(
        self,
        config: LLMProviderConfig,
        transport: JsonTransport | None = None,
    ) -> LLMProvider:
        """Instancia um adaptador de provedor a partir de suas configurações.

        Args:
            config (LLMProviderConfig): Configurações e credenciais do provedor.
            transport (JsonTransport | None): Cliente HTTP customizado opcional.

        Returns:
            LLMProvider: Instância inicializada e pronta do provedor.

        Raises:
            LLMConfigurationError: Se o provedor não constar no registro.
        """
        try:
            factory = self._factories[config.provider]
        except KeyError as exc:
            raise LLMConfigurationError(
                f"LLM provider '{config.provider}' is not registered."
            ) from exc
        return factory(config, transport)

    def ids(self) -> tuple[str, ...]:
        """Lista os identificadores de todos os provedores registrados.

        Returns:
            tuple[str, ...]: Tupla contendo os IDs dos provedores disponíveis.
        """
        return tuple(self._factories)


def build_default_llm_provider_registry() -> LLMProviderRegistry:
    """Constrói o catálogo padrão com os provedores nativos Ollama, vLLM e OpenRouter.

    Returns:
        LLMProviderRegistry: Registro pré-configurado com os provedores padrão.
    """
    registry = LLMProviderRegistry()
    registry.register(OllamaProvider.id, OllamaProvider)
    registry.register(VLLMProvider.id, VLLMProvider)
    registry.register(OpenRouterProvider.id, OpenRouterProvider)
    return registry
