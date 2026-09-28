import json
import logging
from datetime import UTC, datetime
from typing import Any

CONTEXT_FIELDS = (
    "event",
    "experiment_id",
    "run_id",
    "job_id",
    "model",
    "progress",
    "decision_run_id",
    "semantic_feature_set_id",
    "calibration_run_id",
    "cascade_run_id",
    "provider_id",
    "requested_model",
    "resolved_model",
)


class JsonFormatter(logging.Formatter):
    """Formata registros de log em strings JSON estruturadas com campos de contexto operacional."""

    def format(self, record: logging.LogRecord) -> str:
        """Converte um registro de log em um objeto JSON serializado.

        Args:
            record (logging.LogRecord): Registro de evento emitido pelo subsistema de logging.

        Returns:
            str: Linha JSON estruturada contendo timestamp UTC, nível, logger e contexto.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    """Configura o logger raiz da aplicação com saída estruturada em JSON no stdout.

    Args:
        level (str): Nível mínimo de severidade dos logs (ex: 'DEBUG', 'INFO', 'WARNING').
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
