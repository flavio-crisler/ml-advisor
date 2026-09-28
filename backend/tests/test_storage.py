from pathlib import Path

import pytest

from app.domain.exceptions import ArtifactAlreadyExistsError
from app.infrastructure.storage.local import LocalFilesystemStorage


def test_local_storage_persists_without_overwriting(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path)
    artifact_id = "experiments/experiment-id/metrics/result.json"

    assert storage.save(artifact_id, b'{"f1":0.82}') == artifact_id
    assert LocalFilesystemStorage(tmp_path).load(artifact_id) == b'{"f1":0.82}'

    with pytest.raises(ArtifactAlreadyExistsError):
        storage.save(artifact_id, b"replacement")

    assert storage.load(artifact_id) == b'{"f1":0.82}'
    storage.delete(artifact_id)
    assert not (tmp_path / artifact_id).exists()


@pytest.mark.parametrize("artifact_id", ["../secret", "/absolute/path", "nested\\escape"])
def test_local_storage_rejects_unsafe_paths(tmp_path: Path, artifact_id: str) -> None:
    with pytest.raises(ValueError):
        LocalFilesystemStorage(tmp_path).save(artifact_id, b"unsafe")
