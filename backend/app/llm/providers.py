import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.base import (
    LLMCapabilities,
    LLMProvider,
    LLMProviderConfig,
    LLMRequest,
    StructuredResponse,
)
from app.llm.errors import LLMConfigurationError, LLMResponseError
from app.llm.transport import JsonTransport, UrllibJsonTransport


def _schema_name(response_model: type[BaseModel]) -> str:
    """Gera um nome limpo em snake_case a partir do nome da classe do modelo de resposta.

    Args:
        response_model (type[BaseModel]): Classe do modelo Pydantic.

    Returns:
        str: Nome normalizado em snake_case para o JSON Schema.
    """
    name = "".join(
        character.lower() if character.isalnum() else "_" for character in response_model.__name__
    )
    return name.strip("_") or "structured_response"


def _validate_response(
    raw_value: object,
    response_model: type[StructuredResponse],
) -> StructuredResponse:
    """Valida e converte uma resposta bruta no modelo estruturado Pydantic solicitado.

    Args:
        raw_value (object): Conteúdo retornado (texto JSON ou mapeamento já decodificado).
        response_model (type[StructuredResponse]): Classe do modelo estruturado de saída.

    Returns:
        StructuredResponse: Instância validada do modelo.

    Raises:
        LLMResponseError: Se o conteúdo não puder ser validado ou não obedecer ao schema.
    """
    try:
        if isinstance(raw_value, str):
            return response_model.model_validate_json(raw_value)
        if isinstance(raw_value, Mapping):
            return response_model.model_validate(dict(raw_value))
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise LLMResponseError(
            "LLM provider returned content that does not match the requested schema."
        ) from exc
    raise LLMResponseError("LLM provider returned no structured content.")


class _BaseProvider(LLMProvider):
    """Classe base com transporte de rede e construção de mensagens para provedores de LLM.

    Attributes:
        default_base_url (str): Endereço padrão caso não seja informado na configuração.
        config (LLMProviderConfig): Parâmetros de configuração e credenciais.
        transport (JsonTransport): Cliente de transporte HTTP para envio e recebimento de JSON.
        base_url (str): URL base resolvida da API.
    """

    default_base_url: str

    def __init__(
        self,
        config: LLMProviderConfig,
        transport: JsonTransport | None = None,
    ) -> None:
        """Inicializa o cliente do provedor verificando a correspondência do identificador.

        Args:
            config (LLMProviderConfig): Configuração do provedor.
            transport (JsonTransport | None): Cliente de transporte HTTP opcional.

        Raises:
            LLMConfigurationError: Se a configuração não corresponder a este provedor.
        """
        if config.provider != self.id:
            raise LLMConfigurationError(
                f"Provider configuration '{config.provider}' cannot create '{self.id}'."
            )
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")

    @staticmethod
    def _messages(request: LLMRequest) -> list[dict[str, str]]:
        """Monta o formato de lista de mensagens para system e user prompts.

        Args:
            request (LLMRequest): Solicitação com os prompts.

        Returns:
            list[dict[str, str]]: Lista de dicionários com papéis e conteúdos.
        """
        return [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

    def _post(
        self, endpoint: str, payload: Mapping[str, Any], headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        """Executa uma requisição HTTP POST enviando payload JSON para a API.

        Args:
            endpoint (str): Caminho relativo do endpoint.
            payload (Mapping[str, Any]): Corpo da requisição a ser serializado.
            headers (Mapping[str, str]): Cabeçalhos HTTP da requisição.

        Returns:
            Mapping[str, Any]: Resposta desserializada em JSON.
        """
        return self.transport.post_json(
            f"{self.base_url}{endpoint}",
            headers,
            payload,
            self.config.timeout_seconds,
        )


class _OpenAICompatibleProvider(_BaseProvider):
    """Provedor base compatível com o endpoint /chat/completions da especificação OpenAI."""

    def _headers(self) -> dict[str, str]:
        """Gera os cabeçalhos HTTP adicionando o token Bearer caso configurado.

        Returns:
            dict[str, str]: Cabeçalhos incluindo Content-Type e Authorization.
        """
        headers = {"Content-Type": "application/json"}
        api_key = self.config.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _extra_payload(self) -> Mapping[str, Any]:
        """Permite que subclasses injetem parâmetros customizados no corpo da requisição.

        Returns:
            Mapping[str, Any]: Dicionário com campos adicionais.
        """
        return {}

    def _temperature(self, request: LLMRequest) -> float | None:
        """Resolve a temperatura de amostragem respeitando as capacidades do modelo.

        Args:
            request (LLMRequest): Solicitação de inferência.

        Returns:
            float | None: Valor de temperatura resolvido ou None.
        """
        if self.capabilities.supports_temperature:
            return request.temperature
        return None

    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        """Envia a requisição estruturada via JSON Schema para a rota de completions.

        Args:
            request (LLMRequest): Requisição contendo mensagens e parâmetros.
            response_model (type[StructuredResponse]): Classe do modelo de resposta esperado.

        Returns:
            StructuredResponse: Objeto validado com os campos extraídos.

        Raises:
            LLMResponseError: Se a resposta da API vier malformada ou violar o contrato.
        """
        schema = response_model.model_json_schema()
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages(request),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(response_model),
                    "strict": True,
                    "schema": schema,
                },
            },
            **self._extra_payload(),
        }
        temperature = self._temperature(request)
        if temperature is not None:
            payload["temperature"] = temperature
        if request.seed is not None:
            payload["seed"] = request.seed

        response = self._post("/chat/completions", payload, self._headers())
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM provider returned an unexpected response shape.") from exc
        if not isinstance(message, Mapping):
            raise LLMResponseError("LLM provider returned an unexpected response shape.")
        return _validate_response(message.get("parsed", message.get("content")), response_model)


class OllamaProvider(_BaseProvider):
    """Provedor local de inferência integrado com o daemon local do Ollama.

    Attributes:
        id (str): Identificador fixo 'ollama'.
        default_base_url (str): Endereço local padrão http://ollama:11434.
        capabilities (LLMCapabilities): Capacidades declaradas do provedor Ollama.
    """

    id = "ollama"
    default_base_url = "http://ollama:11434"
    capabilities = LLMCapabilities(
        supports_json_schema=True,
        supports_temperature=True,
        is_local=True,
    )

    def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        """Envia solicitação estruturada com formato JSON Schema para o endpoint /api/chat.

        Args:
            request (LLMRequest): Solicitação de inferência com prompts.
            response_model (type[StructuredResponse]): Modelo Pydantic de destino.

        Returns:
            StructuredResponse: Instância validada do modelo.

        Raises:
            LLMResponseError: Se o Ollama retornar conteúdo malformado.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages(request),
            "stream": False,
            "format": response_model.model_json_schema(),
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.seed is not None:
            options["seed"] = request.seed
        if options:
            payload["options"] = options

        response = self._post("/api/chat", payload, {"Content-Type": "application/json"})
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError("LLM provider returned an unexpected response shape.") from exc
        return _validate_response(content, response_model)


class VLLMProvider(_OpenAICompatibleProvider):
    """Provedor de inferência de alto desempenho em servidor local vLLM.

    Attributes:
        id (str): Identificador fixo 'vllm'.
        default_base_url (str): Endereço local http://vllm:8000/v1.
        capabilities (LLMCapabilities): Capacidades de schema e inferência local do vLLM.
    """

    id = "vllm"
    default_base_url = "http://vllm:8000/v1"
    capabilities = LLMCapabilities(
        supports_json_schema=True,
        supports_temperature=True,
        is_local=True,
    )


class OpenRouterProvider(_OpenAICompatibleProvider):
    """Provedor remoto de acesso a modelos de fronteira via API do OpenRouter.

    Attributes:
        id (str): Identificador fixo 'openrouter'.
        default_base_url (str): Endereço https://openrouter.ai/api/v1.
        capabilities (LLMCapabilities): Suporte a reasoning, verbosidade e schemas.
    """

    id = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    capabilities = LLMCapabilities(
        supports_reasoning=True,
        supports_verbosity=True,
        supports_json_schema=True,
        supports_temperature=True,
        is_local=False,
    )

    def __init__(
        self,
        config: LLMProviderConfig,
        transport: JsonTransport | None = None,
    ) -> None:
        """Inicializa o cliente OpenRouter validando a presença obrigatória da chave de API.

        Args:
            config (LLMProviderConfig): Configuração do provedor.
            transport (JsonTransport | None): Cliente de transporte opcional.

        Raises:
            LLMConfigurationError: Se a chave LLM_API_KEY estiver ausente.
        """
        if not config.api_key.get_secret_value():
            raise LLMConfigurationError("OpenRouter requires LLM_API_KEY.")
        super().__init__(config, transport)

    def _extra_payload(self) -> Mapping[str, Any]:
        """Constrói as diretivas específicas de roteamento e esforço de raciocínio do OpenRouter.

        Returns:
            Mapping[str, Any]: Parâmetros de ordenação de provedores e reasoning.
        """
        provider: dict[str, Any] = {"require_parameters": True}
        if self.config.openrouter_providers:
            provider.update(
                {
                    "order": list(self.config.openrouter_providers),
                    "allow_fallbacks": False,
                }
            )
        payload: dict[str, Any] = {"provider": provider}
        if self.config.reasoning_model:
            payload["reasoning"] = {"effort": self.config.reasoning}
            payload["verbosity"] = self.config.verbosity
        return payload

    def _temperature(self, request: LLMRequest) -> float | None:
        """Determina a temperatura adequada para o OpenRouter (0.0 para modelos determinísticos).

        Args:
            request (LLMRequest): Solicitação de inferência.

        Returns:
            float | None: 0.0 para modelos padrão ou None para modelos de raciocínio.
        """
        del request
        if self.config.reasoning_model:
            return None
        return 0.0
