from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import (
    get_auto_workflow_service,
    get_experiment_service,
    get_experiment_strategy_repository,
)
from app.application.advisor.models import AdvisorResult, RecommendationSpec
from app.application.auto.service import AutoWorkflowService
from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.application.jobs.executor import JobEvent, JobExecutor, JobSnapshot, JobStatus, TrainingJob
from app.application.jobs.service import TrainingSubmissionService
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
    SqlAlchemyExperimentStrategyRepository,
)
from app.main import app
from app.ml.registries.catalog import build_default_registry_catalog


class RecordingExecutor(JobExecutor):
    def __init__(self) -> None:
        self.jobs: list[TrainingJob] = []
        self.snapshots: dict[str, JobSnapshot] = {}

    def submit(self, job: TrainingJob) -> JobSnapshot:
        self.jobs.append(job)
        snapshot = JobSnapshot(
            job_id=job.job_id,
            experiment_id=job.experiment_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(UTC),
        )
        self.snapshots[job.job_id] = snapshot
        return snapshot

    def cancel(self, job_id: str) -> JobSnapshot:
        return self.snapshots[job_id]

    def status(self, job_id: str) -> JobSnapshot:
        return self.snapshots[job_id]

    def events(self, job_id: str, *, after: int = 0) -> tuple[JobEvent, ...]:
        return ()

    def find_for_experiment(self, experiment_id: str) -> JobSnapshot | None:
        return next(
            (
                snapshot
                for snapshot in self.snapshots.values()
                if snapshot.experiment_id == experiment_id
            ),
            None,
        )


class StaticAdvisor:
    def __init__(self, result: AdvisorResult) -> None:
        self.result = result
        self.calls = 0

    def recommend(self, **_: str | None) -> AdvisorResult:
        self.calls += 1
        return self.result


def _workflow(
    migrated_database: Database,
    tmp_path: Path,
) -> tuple[
    AutoWorkflowService,
    ExperimentService,
    SqlAlchemyExperimentStrategyRepository,
    RecordingExecutor,
]:
    datasets = SqlAlchemyDatasetRepository(migrated_database)
    dataset = datasets.add(
        Dataset(
            id="auto-dataset",
            name="Auto dataset",
            original_filename="auto.csv",
            content_hash="sha256:auto",
            row_count=20,
            column_count=2,
            storage_path="auto/original.csv",
            profile_path="auto/profile.json",
        )
    )
    proposal = ExperimentSpec(
        dataset_id=dataset.id,
        dataset_hash=dataset.content_hash,
        profile_version="1.0",
        name="Automatic experiment",
        objective="Predict the target",
        task=TaskType.BINARY_CLASSIFICATION,
        target="target",
        features=["amount"],
        column_roles={"amount": ColumnRole.FEATURE, "target": ColumnRole.TARGET},
        models=[ModelSpec(id="logistic_regression")],
        validation=ValidationSpec(strategy="stratified_kfold"),
        metrics=MetricsSpec(primary="f1", positive_label="yes"),
    )
    recommendation = RecommendationSpec(
        task=proposal.task,
        target=proposal.target,
        features=proposal.features,
        models=proposal.models,
        validation=proposal.validation,
        metrics=proposal.metrics,
        explanation="Test a reproducible baseline before comparing measured results.",
    )
    advisor_result = AdvisorResult(
        recommendation=recommendation,
        proposal=proposal,
        provider_id="mock-provider",
        sample_sent=False,
        sample_rows=0,
    )
    experiments = ExperimentService(
        experiments=SqlAlchemyExperimentRepository(migrated_database),
        datasets=datasets,
        validator=ExperimentSpecValidator(build_default_registry_catalog()),
    )
    strategies = SqlAlchemyExperimentStrategyRepository(migrated_database)
    executor = RecordingExecutor()
    training = TrainingSubmissionService(
        experiments=experiments,
        executor=executor,
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        dataset_path=str(tmp_path / "datasets"),
        artifact_path=str(tmp_path / "artifacts"),
    )
    workflow = AutoWorkflowService(
        advisor=StaticAdvisor(advisor_result),  # type: ignore[arg-type]
        experiments=experiments,
        strategies=strategies,
        training=training,
    )
    return workflow, experiments, strategies, executor


def test_auto_workflow_creates_runs_and_records_immutable_strategy(
    migrated_database: Database,
    tmp_path: Path,
) -> None:
    workflow, experiments, strategies, executor = _workflow(migrated_database, tmp_path)

    result = workflow.execute(
        dataset_id="auto-dataset",
        objective="Predict the target",
        language="en-US",
    )

    persisted = experiments.get(result.experiment.id)
    assert persisted.mode == "auto"
    assert persisted.status == "queued"
    assert executor.jobs[0].experiment_id == persisted.id
    recorded = strategies.get_for_experiment(persisted.id)
    assert recorded is not None
    assert recorded.provider_id == "mock-provider"
    assert recorded.strategy["models"][0]["id"] == "logistic_regression"

    with pytest.raises(IntegrityError, match="experiment_strategy_is_immutable"):
        with migrated_database.session() as session:
            session.execute(
                text(
                    "UPDATE experiment_strategies SET provider_id = 'changed' "
                    "WHERE experiment_id = :experiment_id"
                ),
                {"experiment_id": persisted.id},
            )


def test_auto_api_returns_job_and_strategy_can_be_retrieved(
    migrated_database: Database,
    tmp_path: Path,
) -> None:
    workflow, experiments, strategies, _ = _workflow(migrated_database, tmp_path)
    app.dependency_overrides[get_auto_workflow_service] = lambda: workflow
    app.dependency_overrides[get_experiment_service] = lambda: experiments
    app.dependency_overrides[get_experiment_strategy_repository] = lambda: strategies
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/auto/run",
                json={
                    "dataset_id": "auto-dataset",
                    "objective": "Predict the target",
                    "language": "en-US",
                },
            )
            strategy = client.get(
                f"/api/v1/experiments/{response.json()['experiment_id']}/strategy"
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    assert response.json()["job_status"] == "queued"
    assert response.json()["proposal"]["target"] == "target"
    assert strategy.status_code == 200
    assert strategy.json()["source"] == "auto"
    assert strategy.json()["strategy"]["explanation"].startswith("Test a reproducible")
