import json
import math
import numbers
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def new_id() -> str:
    """Gera um identificador único universal (UUIDv4) no formato string.

    Returns:
        str: Identificador UUIDv4 recém-gerado.
    """
    return str(uuid4())


def utc_now() -> datetime:
    """Retorna o timestamp corrente no fuso horário UTC.

    Returns:
        datetime: Data e hora atuais com timezone UTC definido.
    """
    return datetime.now(UTC)


def sanitize_for_json(value: Any) -> Any:
    """Higieniza recursivamente valores para serialização segura em formato JSON.

    Substitui valores especiais de ponto flutuante como NaN e infinito por None
    e normaliza dicionários e listas.

    Args:
        value (Any): Valor arbitrário a ser higienizado.

    Returns:
        Any: Estrutura equivalente convertida e segura para JSON.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real):
        if math.isnan(value) or math.isinf(value):
            return None
        if isinstance(value, (int, float)):
            return value
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    return value


_sanitize_for_json = sanitize_for_json


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serializa um mapeamento em JSON canônico determinístico com chaves ordenadas.

    Args:
        payload (Mapping[str, Any]): Dicionário ou mapeamento com os dados.

    Returns:
        str: String JSON canônica sem espaços redundantes e com chaves ordenadas.
    """
    sanitized = sanitize_for_json(payload)
    return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def json_object(payload_json: str) -> dict[str, Any]:
    """Desserializa uma string JSON assegurando que a raiz seja um objeto dict.

    Args:
        payload_json (str): Conteúdo JSON em formato de texto.

    Returns:
        dict[str, Any]: Dicionário resultante da desserialização.

    Raises:
        ValueError: Se a raiz desserializada não for um dicionário.
    """
    value = json.loads(payload_json)
    if not isinstance(value, dict):
        raise ValueError("JSON payload must be an object")
    return value


def ensure_utc(value: datetime) -> datetime:
    """Assegura que uma data/hora possua fuso horário associado em UTC.

    Args:
        value (datetime): Objeto datetime ingênuo (naive) ou ciente de fuso (aware).

    Returns:
        datetime: Objeto datetime garantidamente no fuso UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
