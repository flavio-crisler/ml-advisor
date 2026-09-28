import json
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.llm.errors import LLMTransportError


class JsonTransport(Protocol):
    """Define o protocolo de transporte HTTP para envio e recebimento de payloads JSON."""

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Envia requisição POST com corpo JSON e retorna a resposta decodificada.

        Args:
            url (str): Endereço URL completo do endpoint de destino.
            headers (Mapping[str, str]): Cabeçalhos HTTP a serem incluídos.
            payload (Mapping[str, Any]): Corpo da requisição a ser serializado.
            timeout_seconds (float): Tempo limite da operação em segundos.

        Returns:
            Mapping[str, Any]: Conteúdo decodificado da resposta JSON.
        """
        ...


class UrllibJsonTransport:
    """Implementa o transporte HTTP de payloads JSON utilizando a biblioteca padrão urllib."""

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Executa a requisição POST síncrona via urllib com tratamento padronizado de exceções.

        Args:
            url (str): URL de destino da chamada HTTP.
            headers (Mapping[str, str]): Dicionário de cabeçalhos HTTP.
            payload (Mapping[str, Any]): Dados a serem enviados como JSON.
            timeout_seconds (float): Tempo máximo de espera pela resposta em segundos.

        Returns:
            Mapping[str, Any]: Dicionário desserializado da resposta JSON.

        Raises:
            LLMTransportError: Se houver erro de HTTP, falha de rede, timeout ou JSON inválido.
        """
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMTransportError(
                f"LLM provider rejected the request with HTTP {exc.code}."
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise LLMTransportError("LLM provider could not be reached.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMTransportError("LLM provider returned invalid JSON.") from exc

        if not isinstance(decoded, dict):
            raise LLMTransportError("LLM provider returned an unexpected response shape.")
        return decoded
