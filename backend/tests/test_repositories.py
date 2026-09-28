from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.domain.common import canonical_json
from app.domain.dataset.entities import Dataset
from app.domain.decision.questions import NoulQuestionSpec
from app.domain.decision.runs import DecisionRun, DecisionRunStatus, ZeroShotDecisionSpec
from app.domain.exceptions import ImmutableExperimentError, ImmutableModelRunError
from app.domain.experiment.entities import (
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    ModelRunStatus,
    TuningTrial,
    TuningTrialStatus,
)
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyDecisionRunRepository,
    SqlAlchemyExperimentRepository,
    SqlAlchemyModelRunRepository,
    SqlAlchemySettingRepository,
    SqlAlchemyTuningTrialRepository,
)


def test_interrupted_decision_runs_are_failed_without_losing_progress(
    migrated_database: Database,
) -> None:
    dataset = SqlAlchemyDatasetRepository(migrated_database).add(
        Dataset(
            name="Interrupted benchmark",
            original_filename="attendance.csv",
            content_hash="sha256:interrupted-benchmark",
            row_count=3,
            column_count=2,
            storage_path="datasets/interrupted/original.csv",
        )
    )
    spec = ZeroShotDecisionSpec(
        dataset_id=dataset.id,
        target="target",
        task="binary_classification",
        positive_class="yes",
        input_columns=["amount"],
        question=NoulQuestionSpec(instructions="Will this row be positive?"),
        provider="openrouter_jev",
        model="typesafe/jev-1.13",
    )
    runs = SqlAlchemyDecisionRunRepository(migrated_database)
    queued = runs.add(DecisionRun.create(spec, row_count=3, question_spec_hash="test"))
    running = runs.add(DecisionRun.create(spec, row_count=3, question_spec_hash="test"))
    finished = runs.add(DecisionRun.create(spec, row_count=3, question_spec_hash="test"))
    runs.transition(running.id, DecisionRunStatus.RUNNING, processed_rows=2)
    runs.transition(finished.id, DecisionRunStatus.COMPLETED)

    assert runs.fail_interrupted() == 2
    for run_id in (queued.id, running.id):
        recovered = runs.get(run_id)
        assert recovered is not None
        assert recovered.status == DecisionRunStatus.FAILED
        assert recovered.finished_at is not None
        assert recovered.error_message == "decision_job_interrupted"
    assert runs.get(running.id).processed_rows == 2
    assert runs.get(finished.id).status == DecisionRunStatus.COMPLETED
    assert runs.fail_interrupted() == 0


def test_repository_history_and_invariants_persist(
    migrated_database: Database, tmp_path: Path
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    experiments = SqlAlchemyExperimentRepository(migrated_database)
    runs = SqlAlchemyModelRunRepository(migrated_database)
    trials = SqlAlchemyTuningTrialRepository(migrated_database)
    settings = SqlAlchemySettingRepository(migrated_database)

    dataset = datasets.add(
        Dataset(
            name="Attendance",
            original_filename="attendance.csv",
            content_hash="sha256:test",
            row_count=100,
            column_count=5,
            storage_path="datasets/dataset/original.csv",
        )
    )
    experiment = experiments.add(
        Experiment.create(
            name="Absence prediction",
            mode=ExperimentMode.EXPERT,
            dataset_id=dataset.id,
            objective="Predict missed appointments",
            spec_version="1.0",
            spec={"target": "missed", "models": ["logistic_regression"]},
        )
    )

    edited = experiments.replace_spec(
        experiment.id,
        spec_version="1.0",
        spec={"target": "missed", "models": ["random_forest"]},
    )
    assert edited.spec["models"] == ["random_forest"]

    first_run = runs.create(
        experiment_id=experiment.id,
        model_id="random_forest",
        random_seed=41,
        parameters={"n_estimators": 100},
    )
    second_run = runs.create(
        experiment_id=experiment.id,
        model_id="random_forest",
        random_seed=41,
        parameters={"n_estimators": 100},
    )
    assert (first_run.run_index, second_run.run_index) == (1, 2)

    with pytest.raises(ImmutableExperimentError):
        experiments.replace_spec(experiment.id, spec_version="1.1", spec={"target": "changed"})

    running = runs.transition(first_run.id, ModelRunStatus.RUNNING)
    completed = runs.transition(
        first_run.id,
        ModelRunStatus.COMPLETED,
        metrics={"f1": 0.82},
        artifact_path="experiments/experiment/models/random-forest.joblib",
    )
    assert running.started_at is not None
    assert completed.metrics == {"f1": 0.82}
    assert completed.duration_seconds is not None

    with pytest.raises(ImmutableModelRunError):
        runs.transition(first_run.id, ModelRunStatus.FAILED, error_message="late failure")

    with pytest.raises(IntegrityError, match="model_run_config_is_immutable"):
        with migrated_database.session() as session:
            session.execute(
                text("UPDATE experiment_runs SET random_seed = 99 WHERE id = :run_id"),
                {"run_id": second_run.id},
            )

    with pytest.raises(IntegrityError, match="experiment_history_cannot_be_deleted"):
        with migrated_database.session() as session:
            session.execute(
                text("DELETE FROM experiments WHERE id = :experiment_id"),
                {"experiment_id": experiment.id},
            )

    trial = TuningTrial.create(
        run_id=second_run.id,
        trial_number=0,
        parameters={"max_depth": 4},
        status=TuningTrialStatus.COMPLETED,
        metric_value=0.8,
    )
    trials.add(trial)
    assert trials.list_for_run(second_run.id) == [trial]

    settings.set("ui_language", "pt-BR")
    settings.set("ui_language", "en-US")
    assert settings.get("ui_language") == "en-US"

    experiments.update_status(experiment.id, ExperimentStatus.RUNNING)
    database_path = tmp_path / "app.db"
    migrated_database.dispose()
    reopened = Database(f"sqlite:///{database_path}")
    try:
        persisted = SqlAlchemyExperimentRepository(reopened).get(experiment.id)
        assert persisted is not None
        assert persisted.status == ExperimentStatus.RUNNING
        assert len(SqlAlchemyModelRunRepository(reopened).list_for_experiment(experiment.id)) == 2
    finally:
        reopened.dispose()


def test_cloned_experiment_gets_new_identity_and_parent() -> None:
    original = Experiment.create(
        name="Original",
        mode=ExperimentMode.ASSISTED,
        dataset_id="dataset-id",
        objective="Classify outcome",
        spec_version="1.0",
        spec={"target": "outcome"},
    )

    clone = original.clone(name="Adjusted clone")

    assert clone.id != original.id
    assert clone.parent_experiment_id == original.id
    assert clone.spec_json == original.spec_json
    assert clone.status == ExperimentStatus.DRAFT


def test_model_run_parameters_and_metrics_sanitize_nan_and_inf(
    migrated_database: Database,
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    experiments = SqlAlchemyExperimentRepository(migrated_database)
    runs = SqlAlchemyModelRunRepository(migrated_database)
    trials = SqlAlchemyTuningTrialRepository(migrated_database)

    raw_payload = {
        "missing": float("nan"),
        "inf_val": float("inf"),
        "neg_inf": float("-inf"),
        "valid": 42,
        "flag": True,
        "nested": {"inner_nan": float("nan"), "inner_val": "ok"},
    }
    expected = (
        '{"flag":true,"inf_val":null,"missing":null,"neg_inf":null,'
        '"nested":{"inner_nan":null,"inner_val":"ok"},"valid":42}'
    )
    encoded = canonical_json(raw_payload)
    assert encoded == expected

    dataset = datasets.add(
        Dataset(
            name="Test Dataset",
            original_filename="test.csv",
            content_hash="sha256:test-nan",
            row_count=50,
            column_count=3,
            storage_path="datasets/test/nan.csv",
        )
    )
    experiment = experiments.add(
        Experiment.create(
            name="XGBoost NaN experiment",
            mode=ExperimentMode.EXPERT,
            dataset_id=dataset.id,
            objective="Predict target",
            spec_version="1.0",
            spec={"target": "target", "models": ["xgboost_regressor"]},
        )
    )

    run = runs.create(
        experiment_id=experiment.id,
        model_id="xgboost_regressor",
        random_seed=42,
        parameters={"missing": float("nan"), "n_estimators": 100},
    )
    assert run.parameters["missing"] is None
    assert run.parameters["n_estimators"] == 100

    completed = runs.transition(
        run.id,
        ModelRunStatus.COMPLETED,
        metrics={"loss": float("nan"), "rmse": 0.45},
    )
    assert completed.metrics == {"loss": None, "rmse": 0.45}

    trial = TuningTrial.create(
        run_id=run.id,
        trial_number=0,
        parameters={"missing": float("nan"), "learning_rate": 0.05},
        status=TuningTrialStatus.COMPLETED,
        metric_value=0.45,
    )
    saved_trial = trials.add(trial)
    assert saved_trial.parameters["missing"] is None
