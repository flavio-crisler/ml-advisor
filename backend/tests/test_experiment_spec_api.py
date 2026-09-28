from fastapi.testclient import TestClient

from app.api.dependencies import get_experiment_service
from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.domain.dataset.entities import Dataset
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    TaskType,
    ValidationSpec,
)
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyExperimentRepository,
)
from app.main import app
from app.ml.registries.catalog import build_default_registry_catalog


def test_validate_experiment_spec_checks_persisted_dataset_hash(
    migrated_database: Database,
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    dataset = datasets.add(
        Dataset(
            name="Validation dataset",
            original_filename="dataset.csv",
            content_hash="actual-hash",
            row_count=10,
            column_count=2,
            storage_path="dataset/original.csv",
            profile_path="dataset/profile.json",
        )
    )
    service = ExperimentService(
        experiments=SqlAlchemyExperimentRepository(migrated_database),
        datasets=datasets,
        validator=ExperimentSpecValidator(build_default_registry_catalog()),
    )
    spec = ExperimentSpec(
        dataset_id=dataset.id,
        dataset_hash="actual-hash",
        profile_version="1.0",
        name="Hash validation",
        objective="Validate a complete experiment contract",
        task=TaskType.BINARY_CLASSIFICATION,
        target="target",
        features=["value"],
        column_roles={"value": ColumnRole.FEATURE, "target": ColumnRole.TARGET},
        models=[ModelSpec(id="logistic_regression")],
        validation=ValidationSpec(strategy="train_test_split"),
        metrics=MetricsSpec(primary="f1", positive_label=1),
    )
    app.dependency_overrides[get_experiment_service] = lambda: service
    try:
        client = TestClient(app)
        valid = client.post(
            "/api/v1/experiment-specs/validate", json=spec.model_dump(mode="json")
        )
        assert valid.status_code == 200
        assert valid.json()["valid"] is True

        stale = spec.model_copy(update={"dataset_hash": "stale-hash"})
        invalid = client.post(
            "/api/v1/experiment-specs/validate", json=stale.model_dump(mode="json")
        )
        assert invalid.status_code == 400
        assert invalid.json()["detail"]["code"] == "dataset_hash_mismatch"
    finally:
        app.dependency_overrides.clear()
