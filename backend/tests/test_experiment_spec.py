from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import get_experiment_service
from app.application.datasets.profiling import DeterministicDatasetProfiler
from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.domain.dataset.entities import Dataset
from app.domain.exceptions import (
    DuplicateRegistryItemError,
    ExperimentSpecValidationError,
)
from app.domain.experiment.entities import Experiment, ExperimentMode
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    PreprocessingStepSpec,
    TaskType,
    TuningSpec,
    ValidationSpec,
)
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyExperimentRepository,
)
from app.main import app
from app.ml.registries.catalog import build_default_registry_catalog
from app.ml.registries.metadata import ModelMetadata
from app.ml.registries.registry import Registry


def _binary_spec(**overrides: object) -> ExperimentSpec:
    values: dict[str, object] = {
        "dataset_id": "dataset-id",
        "dataset_hash": "sha256:attendance",
        "profile_version": "1.1",
        "name": "Attendance prediction",
        "objective": "Predict whether a customer misses an appointment",
        "task": TaskType.BINARY_CLASSIFICATION,
        "target": "missed",
        "features": ["age", "segment"],
        "column_roles": {
            "age": ColumnRole.FEATURE,
            "segment": ColumnRole.FEATURE,
            "missed": ColumnRole.TARGET,
        },
        "models": [ModelSpec(id="logistic_regression")],
        "preprocessing": [
            PreprocessingStepSpec(id="missing_values", strategy="median"),
            PreprocessingStepSpec(id="categorical_encoding", strategy="one_hot"),
        ],
        "validation": ValidationSpec(strategy="stratified_kfold"),
        "metrics": MetricsSpec(primary="f1", secondary=["roc_auc"], positive_label="yes"),
        "tuning": TuningSpec(enabled=True, metric="f1", trials=10),
    }
    values.update(overrides)
    return ExperimentSpec.model_validate(values)


def test_structural_contract_rejects_ambiguous_specs() -> None:
    with pytest.raises(ValidationError, match="target cannot also be a feature"):
        _binary_spec(features=["age", "missed"])

    with pytest.raises(ValidationError, match="multiclass.*require averaging"):
        _binary_spec(
            task=TaskType.MULTICLASS_CLASSIFICATION,
            metrics=MetricsSpec(primary="f1"),
        )

    with pytest.raises(ValidationError, match="enabled tuning requires"):
        _binary_spec(tuning=TuningSpec(enabled=True))


def test_semantic_validation_uses_registry_capabilities() -> None:
    validator = ExperimentSpecValidator(build_default_registry_catalog())
    assert validator.validate(_binary_spec()).spec_version == "1.0"

    incompatible = _binary_spec(
        models=[ModelSpec(id="linear_regression"), ModelSpec(id="unknown_model")],
        validation=ValidationSpec(strategy="kfold"),
    )
    with pytest.raises(ExperimentSpecValidationError) as error:
        validator.validate(incompatible)

    assert error.value.issues == (
        "model_task_incompatible:linear_regression:binary_classification",
        "unknown_model:unknown_model",
        "metric_requires_proba:roc_auc:linear_regression",
        "validator_task_incompatible:kfold:binary_classification",
    )


def test_registry_is_extensible_and_rejects_duplicate_ids() -> None:
    registry = Registry[ModelMetadata]()
    custom = ModelMetadata(
        id="custom_classifier",
        display_name="Custom Classifier",
        tasks=frozenset({TaskType.BINARY_CLASSIFICATION}),
    )

    registry.register(custom)
    assert registry.get("custom_classifier") == custom

    with pytest.raises(DuplicateRegistryItemError):
        registry.register(custom)


def test_validated_spec_round_trips_through_repository(
    migrated_database: Database, tmp_path: Path
) -> None:
    dataset = SqlAlchemyDatasetRepository(migrated_database).add(
        Dataset(
            name="Attendance",
            original_filename="attendance.csv",
            content_hash="sha256:attendance",
            row_count=12,
            column_count=3,
            storage_path="datasets/attendance/original.csv",
        )
    )
    spec = _binary_spec(dataset_id=dataset.id)
    ExperimentSpecValidator(build_default_registry_catalog()).validate(spec)
    experiment = Experiment.create(
        name=spec.name,
        mode=ExperimentMode.EXPERT,
        dataset_id=dataset.id,
        objective=spec.objective,
        spec_version=spec.spec_version,
        spec=spec.model_dump(mode="json"),
    )
    SqlAlchemyExperimentRepository(migrated_database).add(experiment)

    database_path = migrated_database.engine.url.database
    assert database_path is not None
    reopened = Database(f"sqlite:///{database_path}")
    try:
        loaded = SqlAlchemyExperimentRepository(reopened).get(experiment.id)
        assert loaded is not None
        assert ExperimentSpec.model_validate(loaded.spec) == spec
    finally:
        reopened.dispose()


def test_experiment_service_rejects_columns_absent_from_dataset_profile(
    migrated_database: Database,
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    experiments = SqlAlchemyExperimentRepository(migrated_database)
    dataset = datasets.add(
        Dataset(
            name="Attendance",
            original_filename="attendance.csv",
            content_hash="sha256:attendance",
            row_count=4,
            column_count=3,
            storage_path="datasets/attendance/original.csv",
        )
    )
    profile = DeterministicDatasetProfiler().profile(
        pd.DataFrame(
            {
                "age": [20, 30, 40, 50],
                "segment": ["a", "b", "a", "b"],
                "missed": ["yes", "no", "yes", "no"],
            }
        ),
        size_bytes=100,
        encoding="utf-8",
        delimiter=",",
    )
    service = ExperimentService(
        experiments=experiments,
        datasets=datasets,
        validator=ExperimentSpecValidator(build_default_registry_catalog()),
        load_profile=lambda _: profile,
    )
    invalid = _binary_spec(
        dataset_id=dataset.id,
        target="missing_target",
        column_roles={
            "age": ColumnRole.FEATURE,
            "segment": ColumnRole.FEATURE,
            "missing_target": ColumnRole.TARGET,
        },
    )

    with pytest.raises(ExperimentSpecValidationError) as error:
        service.create(invalid, mode=ExperimentMode.EXPERT)

    assert error.value.issues == (
        "unknown_target:missing_target",
        "unknown_column_role:missing_target",
    )


def test_experiment_service_preflights_text_before_draft_creation(
    migrated_database: Database,
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    dataset = datasets.add(
        Dataset(
            name="Messages",
            original_filename="messages.csv",
            content_hash="sha256:attendance",
            row_count=4,
            column_count=3,
            storage_path="datasets/messages/original.csv",
        )
    )
    profile = DeterministicDatasetProfiler().profile(
        pd.DataFrame(
            {
                "age": [20, 30, 40, 50],
                "message": [
                    f"A long and distinct message about customer support {i}" for i in range(4)
                ],
                "missed": ["yes", "no", "yes", "no"],
            }
        ),
        size_bytes=400,
        encoding="utf-8",
        delimiter=",",
    )
    assert profile.text_columns == ["message"]
    service = ExperimentService(
        experiments=SqlAlchemyExperimentRepository(migrated_database),
        datasets=datasets,
        validator=ExperimentSpecValidator(build_default_registry_catalog()),
        load_profile=lambda _: profile,
    )
    raw_text = _binary_spec(
        dataset_id=dataset.id,
        features=["age", "message"],
        column_roles={
            "age": ColumnRole.FEATURE,
            "message": ColumnRole.FEATURE,
            "missed": ColumnRole.TARGET,
        },
    )
    with pytest.raises(ExperimentSpecValidationError, match="require text_vectorization"):
        service.create(raw_text, mode=ExperimentMode.EXPERT)

    processed = raw_text.model_copy(
        update={
            "preprocessing": [
                *raw_text.preprocessing,
                PreprocessingStepSpec(id="text_vectorization", strategy="tfidf"),
            ]
        }
    )
    assert service.create(processed, mode=ExperimentMode.EXPERT).id


def test_registry_and_spec_validation_api_contracts(
    migrated_database: Database,
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    dataset = datasets.add(
        Dataset(
            name="Attendance",
            original_filename="attendance.csv",
            content_hash="sha256:attendance",
            row_count=12,
            column_count=3,
            storage_path="datasets/attendance/original.csv",
        )
    )
    service = ExperimentService(
        experiments=SqlAlchemyExperimentRepository(migrated_database),
        datasets=datasets,
        validator=ExperimentSpecValidator(build_default_registry_catalog()),
    )
    valid_payload = _binary_spec(dataset_id=dataset.id).model_dump(mode="json")
    app.dependency_overrides[get_experiment_service] = lambda: service

    try:
        with TestClient(app) as client:
            expected_counts = {
                "models": 14,
                "metrics": 12,
                "preprocessors": 9,
                "validators": 3,
                "tuners": 1,
            }
            for endpoint, expected_count in expected_counts.items():
                response = client.get(f"/api/v1/{endpoint}")
                assert response.status_code == 200
                assert len(response.json()) == expected_count

            valid = client.post("/api/v1/experiment-specs/validate", json=valid_payload)
            assert valid.status_code == 200
            assert valid.json()["valid"] is True

            semantic_payload = {**valid_payload, "models": [{"id": "unknown_model"}]}
            semantic = client.post("/api/v1/experiment-specs/validate", json=semantic_payload)
            assert semantic.status_code == 400
            assert semantic.json()["detail"]["issues"] == ["unknown_model:unknown_model"]

            structural_payload = {
                key: value for key, value in valid_payload.items() if key != "target"
            }
            structural = client.post("/api/v1/experiment-specs/validate", json=structural_payload)
            assert structural.status_code == 422
    finally:
        app.dependency_overrides.clear()
