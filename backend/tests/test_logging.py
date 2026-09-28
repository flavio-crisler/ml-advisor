import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_preserves_correlation_context() -> None:
    record = logging.LogRecord(
        name="app.jobs",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="model.completed",
        args=(),
        exc_info=None,
    )
    record.event = "model.completed"
    record.experiment_id = "experiment-1"
    record.run_id = "run-1"
    record.job_id = "job-1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["event"] == "model.completed"
    assert payload["experiment_id"] == "experiment-1"
    assert payload["run_id"] == "run-1"
    assert payload["job_id"] == "job-1"
