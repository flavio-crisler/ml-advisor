import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.application.decisions.errors import (
    DecisionProviderAuthenticationError,
    DecisionProviderResponseError,
    DecisionProviderTransportError,
)


class DecisionJsonTransport(Protocol):
    """Protocolo que define transporte assíncrono para requisições JSON.

    Methods:
        request_json: Executa requisição HTTP retornando o JSON decodificado.
    """

    async def request_json(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Executa requisição HTTP e retorna o payload decodificado como JSON.

        Args:
            method (str): Método HTTP (GET, POST, etc.).
            url (str): URL de destino da requisição.
            headers (Mapping[str, str]): Cabeçalhos HTTP da requisição.
            payload (Mapping[str, Any] | None): Corpo JSON da requisição ou None.
            timeout_seconds (float): Tempo limite em segundos.

        Returns:
            Mapping[str, Any]: Resposta recebida mapeada como dicionário.
        """
        ...


class UrllibDecisionJsonTransport:
    """Implementação do transporte JSON assíncrono baseada em urllib.request.

    Utiliza asyncio.to_thread para evitar bloqueio do loop de eventos.
    """

    async def request_json(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Despacha uma requisição HTTP via threadpool e decodifica a resposta JSON.

        Args:
            method (str): Método HTTP a ser executado.
            url (str): Endereço URL da requisição.
            headers (Mapping[str, str]): Cabeçalhos HTTP a enviar.
            payload (Mapping[str, Any] | None): Dados serializáveis em JSON para envio.
            timeout_seconds (float): Tempo limite em segundos para o socket.

        Returns:
            Mapping[str, Any]: Dicionário resultante da decodificação do JSON.

        Raises:
            DecisionProviderAuthenticationError: Se a API retornar erro HTTP 401.
            DecisionProviderTransportError: Se ocorrer erro de conexão ou timeout.
            DecisionProviderResponseError: Se o retorno não for um JSON válido.
        """
        return await asyncio.to_thread(
            self._request_json,
            method,
            url,
            headers,
            payload,
            timeout_seconds,
        )

    @staticmethod
    def _request_json(
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Executa requisição síncrona com urllib.request e faz tratamento de exceções.

        Args:
            method (str): Método HTTP.
            url (str): URL de destino.
            headers (Mapping[str, str]): Cabeçalhos HTTP.
            payload (Mapping[str, Any] | None): Conteúdo serializado ou None.
            timeout_seconds (float): Timeout em segundos.

        Returns:
            Mapping[str, Any]: Objeto JSON decodificado.

        Raises:
            DecisionProviderAuthenticationError: Para respostas HTTP 401.
            DecisionProviderTransportError: Para falhas HTTP e de transporte.
            DecisionProviderResponseError: Para respostas malformadas ou não dicionários.
        """
        request = Request(
            url,
            data=(
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if payload is not None
                else None
            ),
            headers=dict(headers),
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
            if exc.code == 401:
                raise DecisionProviderAuthenticationError(
                    "Decision Provider rejected the configured credentials.",
                    status_code=exc.code,
                ) from exc
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504, 520, 529}
            raise DecisionProviderTransportError(
                f"Decision Provider request failed with HTTP {exc.code}.",
                retryable=retryable,
                status_code=exc.code,
                retry_after_seconds=retry_after,
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise DecisionProviderTransportError(
                "Decision Provider could not be reached.",
                retryable=True,
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecisionProviderResponseError("Decision Provider returned invalid JSON.") from exc

        if not isinstance(decoded, dict):
            raise DecisionProviderResponseError(
                "Decision Provider returned an unexpected response shape."
            )
        return decoded


def _parse_retry_after(value: str | None) -> float | None:
    """Extrai e calcula o tempo de espera a partir do cabeçalho Retry-After.

    Args:
        value (str | None): Valor bruto do cabeçalho HTTP Retry-After.

    Returns:
        float | None: Intervalo em segundos para aguardar ou None se ausente/inválido.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(tz=UTC)).total_seconds())


Sleep = Callable[[float], Awaitable[None]]
RetryObserver = Callable[[int, float], None]


class RetryPolicy:
    """Define a política de repetições com backoff exponencial para chamadas HTTP.

    Attributes:
        max_retries (int): Número máximo de tentativas de repetição.
        base_delay_seconds (float): Intervalo inicial base em segundos.
        max_delay_seconds (float): Intervalo máximo limite em segundos.
        sleep (Sleep): Função assíncrona para pausa de execução.
    """

    def __init__(
        self,
        *,
        max_retries: int,
        base_delay_seconds: float = 0.25,
        max_delay_seconds: float = 8,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """Inicializa a política de tentativas.

        Args:
            max_retries (int): Limite máximo de retentativas.
            base_delay_seconds (float): Atraso base inicial. Padrão 0.25s.
            max_delay_seconds (float): Atraso teto em segundos. Padrão 8.0s.
            sleep (Sleep): Injeção da função sleep. Padrão asyncio.sleep.
        """
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.sleep = sleep

    async def execute(
        self,
        operation: Callable[[], Awaitable[Mapping[str, Any]]],
        on_retry: RetryObserver | None = None,
    ) -> Mapping[str, Any]:
        """Executa a operação com retentativas conforme a política configurada.

        Args:
            operation (Callable[[], Awaitable[Mapping[str, Any]]]): Ação assíncrona a executar.
            on_retry (RetryObserver | None): Callback chamado antes de cada nova tentativa.

        Returns:
            Mapping[str, Any]: Resultado da operação bem-sucedida.

        Raises:
            DecisionProviderTransportError: Se esgotar as tentativas ou erro não for retentável.
        """
        attempt = 0
        while True:
            try:
                return await operation()
            except DecisionProviderTransportError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                attempt += 1
                delay = exc.retry_after_seconds
                if delay is None:
                    delay = min(
                        self.max_delay_seconds,
                        self.base_delay_seconds * (2 ** (attempt - 1)),
                    )
                if on_retry is not None:
                    on_retry(attempt, delay)
                await self.sleep(delay)


class DecisionHttpClient:
    """Cliente HTTP com suporte a retry e decodificação JSON para provedores de decisão.

    Attributes:
        transport (DecisionJsonTransport): Mecanismo de transporte HTTP.
        retry_policy (RetryPolicy): Política de repetições em caso de falha transitória.
    """

    def __init__(
        self,
        *,
        transport: DecisionJsonTransport | None = None,
        retry_policy: RetryPolicy,
    ) -> None:
        """Inicializa o cliente HTTP de decisões.

        Args:
            transport (DecisionJsonTransport | None): Transporte HTTP opcional.
            retry_policy (RetryPolicy): Política de repetição a aplicar.
        """
        self.transport = transport or UrllibDecisionJsonTransport()
        self.retry_policy = retry_policy

    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
        on_retry: RetryObserver | None = None,
    ) -> Mapping[str, Any]:
        """Envia requisição POST com corpo JSON respeitando a política de retry.

        Args:
            url (str): Endereço URL de destino.
            headers (Mapping[str, str]): Cabeçalhos HTTP adicionais.
            payload (Mapping[str, Any]): Dicionário enviado no corpo da requisição.
            timeout_seconds (float): Tempo limite por tentativa.
            on_retry (RetryObserver | None): Observador opcional chamado em repetições.

        Returns:
            Mapping[str, Any]: Resposta JSON decodificada do servidor.
        """
        return await self.retry_policy.execute(
            lambda: self.transport.request_json("POST", url, headers, payload, timeout_seconds),
            on_retry,
        )

    async def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        on_retry: RetryObserver | None = None,
    ) -> Mapping[str, Any]:
        """Envia requisição GET com decodificação JSON e política de retry.

        Args:
            url (str): Endereço URL de destino.
            headers (Mapping[str, str]): Cabeçalhos HTTP adicionais.
            timeout_seconds (float): Tempo limite por tentativa.
            on_retry (RetryObserver | None): Observador opcional de repetições.

        Returns:
            Mapping[str, Any]: Resposta JSON decodificada.
        """
        return await self.retry_policy.execute(
            lambda: self.transport.request_json("GET", url, headers, None, timeout_seconds),
            on_retry,
        )
