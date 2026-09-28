import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/decision-runs/missing/events",
        "/api/v1/semantic-features/missing/events",
        "/api/v1/experiments/missing/events",
    ],
)
@pytest.mark.parametrize("cursor", ["invalid", "-1"])
def test_event_stream_rejects_invalid_last_event_id(path: str, cursor: str) -> None:
    client = TestClient(app)

    response = client.get(path, headers={"Last-Event-ID": cursor})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["header", "Last-Event-ID"]
