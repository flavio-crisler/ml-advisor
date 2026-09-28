from abc import ABC, abstractmethod
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr

StructuredResponse = TypeVar("StructuredResponse", bound=BaseModel)


class LLMCapabilities(BaseModel):
    """Descreve as capacidades funcionais suportadas por determinado provedor de LLM.

    Attributes:
        supports_reasoning (bool): Indica suporte a parametrização explícita de reasoning effort.
        supports_verbosity (bool): Indica controle de verbosidade de pensamentos do modelo.
        supports_json_schema (bool): Indica suporte nativo a respostas estruturadas por JSON Schema.
        supports_temperature (bool): Indica se o modelo permite ajuste de temperatura estocástica.
        is_local (bool): Indica se a inferência executa localmente sem chamadas remotas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    supports_reasoning: bool = False
    supports_verbosity: bool = False
    supports_json_schema: bool = False
    supports_temperature: bool = True
    is_local: bool = False


class LLMRequest(BaseModel):
    """Representa uma solicitação de inferência com prompts e parâmetros para o LLM.

    Attributes:
        system_prompt (str): Instruções contextuais e regras de sistema atribuídas ao modelo.
        user_prompt (str): Conteúdo da requisição ou dados de análise enviados pelo usuário.
        language (str): Código da localidade linguística adotada para a resposta (ex: 'pt-BR').
        temperature (float | None): Temperatura de amostragem estocástica ou None para default.
        seed (int | None): Semente de aleatoriedade para reproduzibilidade quando suportada.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    language: str = Field(default="pt-BR", min_length=2)
    temperature: float | None = Field(default=None, ge=0, le=2)
    seed: int | None = None


class LLMProviderConfig(BaseModel):
    """Configurações de conexão e credenciais de acesso a um provedor de LLM.

    Attributes:
        provider (Literal["ollama", "vllm", "openrouter"]): Identificador do tipo de provedor.
        model (str): Nome ou caminho do modelo no provedor.
        base_url (str): Endereço HTTP base para comunicação com a API.
        api_key (SecretStr): Chave de autenticação secreta da API.
        reasoning_model (bool): Flag indicando se o modelo possui capacidade de raciocínio.
        reasoning (str): Nível de esforço de raciocínio.
        verbosity (str): Nível de verbosidade do fluxo de pensamento.
        openrouter_providers (tuple[str, ...]): Ordem de preferência de provedores no OpenRouter.
        timeout_seconds (float): Tempo limite da chamada em segundos.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["ollama", "vllm", "openrouter"]
    model: str = Field(min_length=1)
    base_url: str = ""
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    reasoning_model: bool = False
    reasoning: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    verbosity: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    openrouter_providers: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=60, gt=0, le=600)


class LLMProvider(ABC):
    """Classe base abstrata para adaptadores de provedores de modelos de linguagem.

    Attributes:
        id (str): Identificador exclusivo do provedor.
        capabilities (LLMCapabilities): Matriz de funcionalidades suportadas.
    """

    id: str
    capabilities: LLMCapabilities

    @abstractmethod
    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        """Executa a inferência e valida a resposta contra o modelo estruturado Pydantic fornecido.

        Args:
            request (LLMRequest): Dados da solicitação contendo prompts e hiperparâmetros.
            response_model (type[StructuredResponse]): Classe do modelo esperado na resposta.

        Returns:
            StructuredResponse: Instância validada do modelo estruturado.

        Raises:
            LLMError: Se houver falha de rede, timeout ou incompatibilidade de schema.
        """


class UnavailableLLMProvider(LLMProvider):
    """Representa um adaptador de provedor intencionalmente indisponível ou mal configurado.

    Attributes:
        capabilities (LLMCapabilities): Capacidades padrão vazias.
        id (str): Identificador do provedor indisponível.
        reason (str): Motivo descritivo da indisponibilidade.
    """

    capabilities = LLMCapabilities()

    def __init__(self, provider_id: str, reason: str) -> None:
        """Inicializa o provedor indisponível com o motivo correspondente.

        Args:
            provider_id (str): Identificador do provedor.
            reason (str): Descrição do erro de configuração ou ausência de chave.
        """
        self.id = provider_id
        self.reason = reason

    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        """Lança erro de configuração ao tentar executar chamada em provedor indisponível.

        Args:
            request (LLMRequest): Solicitação de inferência.
            response_model (type[StructuredResponse]): Modelo de resposta esperado.

        Returns:
            StructuredResponse: Nunca retorna diretamente.

        Raises:
            LLMConfigurationError: Sempre lançado detalhando o motivo da indisponibilidade.
        """
        del request, response_model
        from app.llm.errors import LLMConfigurationError

        raise LLMConfigurationError(self.reason)
