class DecisionProviderError(RuntimeError):
    """Exceção base para falhas operacionais na integração com provedores de decisão."""


class DecisionProviderConfigurationError(DecisionProviderError):
    """Exceção levantada quando um provedor não pode ser configurado com segurança."""


class DecisionProviderTransportError(DecisionProviderError):
    """Exceção levantada quando ocorre falha de transporte ou rede com o provedor.

    Attributes:
        retryable (bool): Indica se a requisição pode ser retentada.
        status_code (int | None): Código HTTP retornado, se aplicável.
        retry_after_seconds (float | None): Tempo sugerido de espera antes de retentar.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        """Inicializa a exceção com parâmetros de transporte e reexecução.

        Args:
            message (str): Descrição do erro ocorrido.
            retryable (bool): Sinaliza se o erro é transiente.
            status_code (int | None): Código de resposta HTTP.
            retry_after_seconds (float | None): Período recomendado em segundos.
        """
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class DecisionProviderAuthenticationError(DecisionProviderTransportError):
    """Exceção levantada quando credenciais do provedor faltam ou são rejeitadas."""


class DecisionProviderResponseError(DecisionProviderError):
    """Exceção levantada quando o provedor retorna resposta inconsistente ou malformada."""
