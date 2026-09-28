from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_artifact_storage,
    get_experiment_service,
    get_job_executor,
    get_model_run_repository,
    get_tuning_trial_repository,
)
from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.application.jobs.executor import (
    JobEvent,
    JobExecutor,
    JobSnapshot,
    JobStatus,
    TrainingJob,
)
from app.domain.dataset.entities import Dataset
from app.domain.experiment.entities import (
    ModelRunStatus,
    TuningTrial,
    TuningTrialStatus,
)
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
    SqlAlchemyModelRunRepository,
    SqlAlchemyTuningTrialRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.main import app
from app.ml.registries.catalog import build_default_registry_catalog


class FakeJobExecutor(JobExecutor):
    def __init__(self) -> None:
        self.snapshots: dict[str, JobSnapshot] = {}
        self.event_log: dict[str, tuple[JobEvent, ...]] = {}

    def submit(self, job: TrainingJob) -> JobSnapshot:
        now = datetime.now(UTC)
        snapshot = JobSnapshot(
            job_id=job.job_id,
            experiment_id=job.experiment_id,
            status=JobStatus.COMPLETED,
            created_at=now,
            started_at=now,
            finished_at=now,
        )
        self.snapshots[job.job_id] = snapshot
        self.event_log[job.job_id] = (
            JobEvent(
                sequence=1,
                type="training.progress",
                timestamp=now,
                payload={
                    "experiment_id": job.experiment_id,
                    "progress": 50,
                    "message_key": "training.in_progress",
                },
            ),
            JobEvent(
                sequence=2,
                type="job.completed",
                timestamp=now,
                payload={"experiment_id": job.experiment_id},
            ),
        )
        return snapshot

    def cancel(self, job_id: str) -> JobSnapshot:
        snapshot = self.snapshots[job_id]
        cancelled = JobSnapshot(
            job_id=snapshot.job_id,
            experiment_id=snapshot.experiment_id,
            status=JobStatus.CANCELLED,
            created_at=snapshot.created_at,
            started_at=snapshot.started_at,
            finished_at=datetime.now(UTC),
        )
        self.snapshots[job_id] = cancelled
        return cancelled

    def status(self, job_id: str) -> JobSnapshot:
        return self.snapshots[job_id]

    def events(self, job_id: str, *, after: int = 0) -> tuple[JobEvent, ...]:
        return tuple(event for event in self.event_log[job_id] if event.sequence > after)

    def find_for_experiment(self, experiment_id: str) -> JobSnapshot | None:
        return next(
            (
                snapshot
                for snapshot in self.snapshots.values()
                if snapshot.experiment_id == experiment_id
            ),
            None,
        )


def _spec(dataset: Dataset) -> ExperimentSpec:
    return ExperimentSpec(
        dataset_id=dataset.id,
        dataset_hash=dataset.content_hash,
        profile_version="1.0",
        name="API experiment",
        objective="Verify the asynchronous experiment API",
        task=TaskType.BINARY_CLASSIFICATION,
        target="target",
        features=["value"],
        column_roles={"value": ColumnRole.FEATURE, "target": ColumnRole.TARGET},
        models=[ModelSpec(id="logistic_regression")],
        validation=ValidationSpec(strategy="train_test_split"),
        metrics=MetricsSpec(primary="f1", positive_label=1),
    )


def test_experiment_crud_run_status_and_sse_contract(
    migrated_database: Database, tmp_path: Path
) -> None:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    experiments = SqlAlchemyExperimentRepository(migrated_database)
    runs = SqlAlchemyModelRunRepository(migrated_database)
    trials = SqlAlchemyTuningTrialRepository(migrated_database)
    artifact_storage = LocalFilesystemStorage(tmp_path / "artifacts")
    dataset = datasets.add(
        Dataset(
            name="Dataset",
            original_filename="dataset.csv",
            content_hash="sha256-api",
            row_count=10,
            column_count=2,
            storage_path="dataset/original.csv",
            profile_path="dataset/profile.json",
        )
    )
    service = ExperimentService(
        experiments=experiments,
        datasets=datasets,
        validator=ExperimentSpecValidator(build_default_registry_catalog()),
    )
    executor = FakeJobExecutor()
    app.dependency_overrides[get_experiment_service] = lambda: service
    app.dependency_overrides[get_model_run_repository] = lambda: runs
    app.dependency_overrides[get_tuning_trial_repository] = lambda: trials
    app.dependency_overrides[get_artifact_storage] = lambda: artifact_storage
    app.dependency_overrides[get_job_executor] = lambda: executor

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/experiments",
                json={
                    "mode": "expert",
                    "description": "API contract",
                    "spec": _spec(dataset).model_dump(mode="json"),
                },
            )
            assert created.status_code == 201
            experiment_id = created.json()["id"]
            assert created.json()["status"] == "draft"
            assert created.json()["runs"] == []

            run = runs.create(
                experiment_id=experiment_id,
                model_id="logistic_regression",
                random_seed=41,
                parameters={"C": 1.0},
            )
            analysis_path = artifact_storage.save(
                f"{run.id}/analysis.json",
                b'{"schema_version":"1.0","plots":{}}',
            )
            runs.transition(run.id, ModelRunStatus.RUNNING)
            runs.transition(
                run.id,
                ModelRunStatus.COMPLETED,
                metrics={"analysis_path": analysis_path},
                artifact_path=f"{run.id}/model.joblib",
            )
            trials.add(
                TuningTrial.create(
                    run_id=run.id,
                    trial_number=0,
                    parameters={"C": 0.5},
                    status=TuningTrialStatus.COMPLETED,
                    metric_value=0.8,
                )
            )
            trial_response = client.get(f"/api/v1/experiments/runs/{run.id}/tuning-trials")
            assert trial_response.status_code == 200
            assert trial_response.json()[0]["parameters"] == {"C": 0.5}
            assert trial_response.json()[0]["metric_value"] == 0.8
            analysis_response = client.get(f"/api/v1/experiments/runs/{run.id}/analysis")
            assert analysis_response.status_code == 200
            assert analysis_response.json() == {"schema_version": "1.0", "plots": {}}
            run_response = client.get(f"/api/v1/experiments/runs/{run.id}")
            assert run_response.status_code == 200
            assert run_response.json()["parameters"] == {"C": 1.0}

            listing = client.get("/api/v1/experiments")
            assert listing.status_code == 200
            assert [item["id"] for item in listing.json()] == [experiment_id]

            clone = client.post(
                f"/api/v1/experiments/{experiment_id}/clone",
                json={"name": "Cloned experiment"},
            )
            assert clone.status_code == 201
            assert clone.json()["parent_experiment_id"] == experiment_id
            clone_id = clone.json()["id"]

            updated_spec = _spec(dataset).model_copy(
                update={
                    "name": "Adjusted clone",
                    "models": [ModelSpec(id="logistic_regression", parameters={"C": 0.5})],
                }
            )
            updated = client.put(
                f"/api/v1/experiments/{clone_id}",
                json={"spec": updated_spec.model_dump(mode="json")},
            )
            assert updated.status_code == 200
            assert updated.json()["name"] == "Adjusted clone"
            assert updated.json()["spec"]["models"][0]["parameters"] == {"C": 0.5}

            clone_run = runs.create(
                experiment_id=clone_id,
                model_id="logistic_regression",
                random_seed=41,
                parameters={"C": 0.5},
            )
            runs.transition(clone_run.id, ModelRunStatus.RUNNING)
            runs.transition(
                clone_run.id,
                ModelRunStatus.COMPLETED,
                metrics={"values": {"f1": 0.9}},
                artifact_path=f"{clone_run.id}/model.joblib",
            )
            comparison = client.get(f"/api/v1/experiments/{experiment_id}/compare/{clone_id}")
            assert comparison.status_code == 200
            assert {item["experiment_id"] for item in comparison.json()} == {
                experiment_id,
                clone_id,
            }

            renamed = client.patch(
                f"/api/v1/experiments/{experiment_id}",
                json={"name": "Renamed experiment"},
            )
            assert renamed.status_code == 200
            assert renamed.json()["name"] == "Renamed experiment"
            assert renamed.json()["spec"]["name"] == "API experiment"

            deleted = client.delete(f"/api/v1/experiments/{clone_id}")
            assert deleted.status_code == 204
            assert client.get(f"/api/v1/experiments/{clone_id}").status_code == 404
            assert [item["id"] for item in client.get("/api/v1/experiments").json()] == [
                experiment_id
            ]
            historical_comparison = client.get(
                f"/api/v1/experiments/{experiment_id}/compare/{clone_id}"
            )
            assert historical_comparison.status_code == 200
            assert {item["experiment_id"] for item in historical_comparison.json()} == {
                experiment_id,
                clone_id,
            }

            submitted = client.post(f"/api/v1/experiments/{experiment_id}/run")
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            assert submitted.json()["status"] == "completed"

            active_delete = client.delete(f"/api/v1/experiments/{experiment_id}")
            assert active_delete.status_code == 409
            assert active_delete.json()["detail"]["code"] == "ExperimentDeletionError"

            job_status = client.get(f"/api/v1/experiments/jobs/{job_id}")
            assert job_status.status_code == 200
            assert job_status.json()["status"] == "completed"

            events = client.get(f"/api/v1/experiments/{experiment_id}/events")
            assert events.status_code == 200
            assert events.headers["content-type"].startswith("text/event-stream")
            assert "event: training.progress" in events.text
            assert '"progress": 50' in events.text
            assert "event: job.completed" in events.text

            detail = client.get(f"/api/v1/experiments/{experiment_id}")
            assert detail.status_code == 200
            assert detail.json()["status"] == "queued"
    finally:
        app.dependency_overrides.clear()
