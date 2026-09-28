class LLMProviderError(RuntimeError):
    """Classe base de exceções de LLM, segura contra vazamento de payloads e credenciais.

    Attributes:
        code (str): Código identificador do tipo de erro de provedor.
    """

    code = "llm_provider_error"


class LLMConfigurationError(LLMProviderError):
    """Indica que o provedor de LLM está incorretamente configurado ou com credenciais ausentes.

    Attributes:
        code (str): Código identificador de erro de configuração.
    """

    code = "llm_configuration_error"


class LLMTransportError(LLMProviderError):
    """Indica falha de rede, conexão recusada ou timeout ao comunicar com a API do modelo.

    Attributes:
        code (str): Código identificador de erro de transporte de rede.
    """

    code = "llm_transport_error"


class LLMResponseError(LLMProviderError):
    """Indica resposta do LLM vazia, corrompida ou incompatível com o schema esperado.

    Attributes:
        code (str): Código identificador de erro estrutural de resposta.
    """

    code = "llm_response_error"
