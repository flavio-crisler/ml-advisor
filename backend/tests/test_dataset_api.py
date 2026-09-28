from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api.dependencies import get_dataset_service
from app.application.datasets.importers import CsvDatasetImporter
from app.application.datasets.profiling import DeterministicDatasetProfiler
from app.application.datasets.service import DatasetService
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import SqlAlchemyDatasetRepository
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.main import app


def test_profiler_recognizes_exported_dataframe_index_as_identifier() -> None:
    frame = pd.DataFrame(
        {
            "Unnamed: 0": list(range(12)),
            "value": list(range(12, 24)),
            "target": ["yes", "no"] * 6,
        }
    )

    profile = DeterministicDatasetProfiler().profile(
        frame,
        size_bytes=100,
        encoding="utf-8",
        delimiter=",",
    )

    assert profile.possible_id_columns == ["Unnamed: 0"]
    exported_index = next(
        column for column in profile.column_profiles if column.name == "Unnamed: 0"
    )
    assert exported_index.suggested_role == "id"


def _service(
    database: Database, root: Path, *, max_upload_bytes: int = 1_000_000
) -> DatasetService:
    return DatasetService(
        repository=SqlAlchemyDatasetRepository(database),
        storage=LocalFilesystemStorage(root),
        importer=CsvDatasetImporter(),
        profiler=DeterministicDatasetProfiler(preview_rows=5),
        working_directory=root / ".incoming",
        max_upload_bytes=max_upload_bytes,
    )


def test_csv_upload_profile_listing_and_reuse(migrated_database: Database, tmp_path: Path) -> None:
    service = _service(migrated_database, tmp_path / "datasets")
    app.dependency_overrides[get_dataset_service] = lambda: service
    csv_content = b"""customer_id;active;created_at;segment;target;amount;constant
1;true;2026-01-01;A;yes;10;same
2;false;2026-01-02;A;no;11;same
3;true;2026-01-03;A;yes;12;same
4;false;2026-01-04;B;no;13;same
5;true;2026-01-05;A;yes;;same
6;false;2026-01-06;B;no;15;same
7;true;2026-01-07;A;yes;16;same
8;false;2026-01-08;B;no;17;same
9;true;2026-01-09;A;yes;18;same
10;false;2026-01-10;B;no;19;same
11;true;2026-01-11;A;yes;20;same
12;false;2026-01-12;B;no;1000;same
"""

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/datasets",
                files={"file": ("attendance.csv", csv_content, "text/csv")},
                data={"name": "Attendance"},
            )
            assert response.status_code == 201
            payload = response.json()
            dataset_id = payload["dataset"]["id"]
            profile = payload["profile"]
            assert payload["reused"] is False
            assert profile["rows"] == 12
            assert profile["columns"] == 7
            assert profile["delimiter"] == ";"
            assert profile["possible_id_columns"] == ["customer_id"]
            assert "active" in profile["boolean_columns"]
            assert "created_at" in profile["datetime_columns"]
            assert "target" in profile["candidate_targets"]
            amount_distribution = profile["candidate_target_distributions"]["amount"]
            assert len(amount_distribution) == 11
            assert {item["value"] for item in amount_distribution} == {
                "10.0",
                "11.0",
                "12.0",
                "13.0",
                "15.0",
                "16.0",
                "17.0",
                "18.0",
                "19.0",
                "20.0",
                "1000.0",
            }
            assert profile["missing_values"]["amount"] == 1
            assert profile["constant_columns"] == ["constant"]
            assert len(profile["preview"]) == 5

            stored = tmp_path / "datasets" / dataset_id
            assert (stored / "original.csv").read_bytes() == csv_content
            assert (stored / "profile.json").is_file()

            duplicate = client.post(
                "/api/v1/datasets",
                files={"file": ("attendance.csv", csv_content, "text/csv")},
            )
            assert duplicate.status_code == 200
            assert duplicate.json()["reused"] is True
            assert duplicate.json()["dataset"]["id"] == dataset_id

            listing = client.get("/api/v1/datasets")
            assert listing.status_code == 200
            assert len(listing.json()) == 1

            detail = client.get(f"/api/v1/datasets/{dataset_id}")
            assert detail.status_code == 200
            assert detail.json()["profile"]["profile_version"] == "1.1"
            assert "segment" in detail.json()["semantic_columns"]["optional"]
            assert "customer_id" not in detail.json()["semantic_columns"]["optional"]

            profile_response = client.get(f"/api/v1/datasets/{dataset_id}/profile")
            assert profile_response.status_code == 200
            assert profile_response.json()["rows"] == 12
    finally:
        app.dependency_overrides.clear()


def test_upload_limits_format_and_missing_dataset_errors(
    migrated_database: Database, tmp_path: Path
) -> None:
    service = _service(migrated_database, tmp_path / "datasets", max_upload_bytes=10)
    app.dependency_overrides[get_dataset_service] = lambda: service

    try:
        with TestClient(app) as client:
            too_large = client.post(
                "/api/v1/datasets",
                files={"file": ("large.csv", b"column\nmore-than-ten-bytes", "text/csv")},
            )
            assert too_large.status_code == 413
            assert too_large.json()["detail"]["code"] == "dataset_too_large"

            unsupported = client.post(
                "/api/v1/datasets",
                files={"file": ("dataset.xlsx", b"not-a-csv", "application/octet-stream")},
            )
            assert unsupported.status_code == 400
            assert unsupported.json()["detail"]["code"] == "dataset_format_unsupported"

            missing = client.get("/api/v1/datasets/missing")
            assert missing.status_code == 404
            assert missing.json()["detail"]["code"] == "dataset_not_found"
    finally:
        app.dependency_overrides.clear()
