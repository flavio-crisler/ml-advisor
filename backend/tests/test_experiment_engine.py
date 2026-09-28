import io
import json
from pathlib import Path

import joblib
import pandas as pd

from app.application.datasets.importers import CsvDatasetImporter
from app.application.datasets.profiling import DeterministicDatasetProfiler
from app.application.datasets.service import DatasetService
from app.application.training.engine import ExperimentEngine
from app.domain.experiment.entities import Experiment, ExperimentMode, ExperimentStatus
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
    SqlAlchemyModelRunRepository,
    SqlAlchemyTuningTrialRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.ml.registries.catalog import build_default_registry_catalog


def _create_engine_fixture(
    migrated_database: Database, tmp_path: Path, *, tuning: bool = False
) -> tuple[ExperimentEngine, str, LocalFilesystemStorage, Path]:
    dataset_storage = LocalFilesystemStorage(tmp_path / "datasets")
    artifact_storage = LocalFilesystemStorage(tmp_path / "experiments")
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    experiments = SqlAlchemyExperimentRepository(migrated_database)
    fixture_path = Path(__file__).parent / "fixtures" / "attendance.csv"
    dataset = (
        DatasetService(
            repository=datasets,
            storage=dataset_storage,
            importer=CsvDatasetImporter(),
            profiler=DeterministicDatasetProfiler(preview_rows=5),
            working_directory=tmp_path / ".incoming",
            max_upload_bytes=1_000_000,
        )
        .ingest(
            filename="attendance.csv",
            name="Attendance",
            stream=io.BytesIO(fixture_path.read_bytes()),
        )
        .dataset
    )
    features = ["active", "created_at", "segment", "amount"]
    specification = ExperimentSpec(
        dataset_id=dataset.id,
        dataset_hash=dataset.content_hash,
        profile_version="1.1",
        name="Attendance prediction",
        objective="Predict appointment attendance",
        task=TaskType.BINARY_CLASSIFICATION,
        target="target",
        features=features,
        column_roles={
            **{feature: ColumnRole.FEATURE for feature in features},
            "customer_id": ColumnRole.ID,
            "constant": ColumnRole.IGNORED,
            "target": ColumnRole.TARGET,
        },
        models=[
            ModelSpec(id="logistic_regression"),
            ModelSpec(id="random_forest_classifier", parameters={"n_estimators": 10}),
        ],
        preprocessing=[
            PreprocessingStepSpec(id="missing_values", strategy="median"),
            PreprocessingStepSpec(id="categorical_encoding", strategy="one_hot"),
            PreprocessingStepSpec(id="boolean_encoding", strategy="binary"),
            PreprocessingStepSpec(id="numeric_scaling", strategy="standard"),
            PreprocessingStepSpec(id="datetime", strategy="components"),
            PreprocessingStepSpec(id="imbalance", strategy="class_weight"),
            PreprocessingStepSpec(id="feature_selection", strategy="variance_threshold"),
        ],
        validation=ValidationSpec(strategy="train_test_split", test_size=0.34, random_seed=41),
        metrics=MetricsSpec(
            primary="f1",
            secondary=["accuracy", "roc_auc"],
            positive_label="yes",
        ),
        tuning=TuningSpec(enabled=tuning, metric="f1" if tuning else None, trials=2),
    )
    experiment = experiments.add(
        Experiment.create(
            name=specification.name,
            mode=ExperimentMode.EXPERT,
            dataset_id=dataset.id,
            objective=specification.objective,
            spec_version=specification.spec_version,
            spec=specification.model_dump(mode="json"),
        )
    )
    engine = ExperimentEngine(
        datasets=datasets,
        experiments=experiments,
        runs=SqlAlchemyModelRunRepository(migrated_database),
        tuning_trials=SqlAlchemyTuningTrialRepository(migrated_database),
        dataset_storage=dataset_storage,
        artifact_storage=artifact_storage,
        catalog=build_default_registry_catalog(),
    )
    return engine, experiment.id, artifact_storage, fixture_path


def test_full_experiment_executes_persists_artifacts_and_reproduces(
    migrated_database: Database, tmp_path: Path
) -> None:
    engine, experiment_id, storage, fixture_path = _create_engine_fixture(
        migrated_database, tmp_path
    )

    first = engine.execute(experiment_id)

    assert len(first.runs) == 2
    assert all(run.status.value == "completed" for run in first.runs)
    assert all(run.metrics and "values" in run.metrics for run in first.runs)
    assert storage.load(first.leaderboard_path)
    leaderboard = json.loads(storage.load(first.leaderboard_path))
    assert storage.load(leaderboard["comparison_path"])
    stored_experiment = SqlAlchemyExperimentRepository(migrated_database).get(experiment_id)
    assert stored_experiment is not None
    assert stored_experiment.status == ExperimentStatus.COMPLETED

    model_run = first.runs[0]
    assert model_run.metrics is not None
    analysis = json.loads(storage.load(model_run.metrics["analysis_path"]))
    assert analysis["schema_version"] == "1.0"
    assert set(analysis["plots"]) >= {
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
        "feature_importance",
        "shap_summary",
    }
    assert analysis["explainability"]["status"] == "available"
    persisted_predictions = pd.read_parquet(
        io.BytesIO(storage.load(model_run.metrics["predictions_path"]))
    )
    assert {
        "row_id",
        "fold_index",
        "target",
        "prediction",
        "correct",
        "confidence",
        "probabilities",
    } <= set(persisted_predictions.columns)
    assert persisted_predictions["row_id"].is_unique
    assert len(persisted_predictions) == sum(
        fold["test_rows"] for fold in model_run.metrics["folds"]
    )
    assert model_run.artifact_path is not None
    bundle = joblib.load(io.BytesIO(storage.load(model_run.artifact_path)))
    raw = pd.read_csv(fixture_path, sep=";")
    transformed = bundle["pipeline"].transform(
        raw.loc[:, ["active", "created_at", "segment", "amount"]].head(2)
    )
    predictions = bundle["model"].predict(transformed)
    assert len(predictions) == 2

    second = engine.execute(experiment_id)
    all_runs = SqlAlchemyModelRunRepository(migrated_database).list_for_experiment(experiment_id)
    assert second.execution_id != first.execution_id
    assert len(all_runs) == 4
    assert sorted(run.run_index for run in all_runs) == [1, 1, 2, 2]
    first_metrics = {run.model_id: run.metrics["values"] for run in first.runs if run.metrics}
    second_metrics = {run.model_id: run.metrics["values"] for run in second.runs if run.metrics}
    assert first_metrics == second_metrics


def test_engine_tunes_persists_trials_and_retrains_best_parameters(
    migrated_database: Database, tmp_path: Path
) -> None:
    engine, experiment_id, storage, _ = _create_engine_fixture(
        migrated_database, tmp_path, tuning=True
    )

    result = engine.execute(experiment_id)

    experiment = SqlAlchemyExperimentRepository(migrated_database).get(experiment_id)
    assert experiment is not None
    assert experiment.status == ExperimentStatus.COMPLETED
    assert len(result.runs) == 2
    trials = SqlAlchemyTuningTrialRepository(migrated_database)
    for run in result.runs:
        persisted = trials.list_for_run(run.id)
        assert len(persisted) == 2
        assert all(trial.status.value in {"completed", "pruned", "failed"} for trial in persisted)
        assert run.metrics is not None
        assert run.metrics["tuning"]["best_parameters"]
        assert run.metrics["effective_parameters"]
        assert run.artifact_path is not None
        bundle = joblib.load(io.BytesIO(storage.load(run.artifact_path)))
        for key, value in run.metrics["tuning"]["best_parameters"].items():
            assert bundle["model"].get_params()[key] == value
