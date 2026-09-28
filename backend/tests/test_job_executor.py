import time
from typing import Any

import pytest

from app.application.jobs.executor import (
    JobStatus,
    LocalProcessExecutor,
    TrainingJob,
)
from app.domain.exceptions import JobAlreadyRunningError, JobStateError


def _slow_worker(job: TrainingJob, emit: Any) -> dict[str, Any]:
    emit(
        {
            "type": "training.progress",
            "experiment_id": job.experiment_id,
            "progress": 25,
            "message_key": "training.in_progress",
        }
    )
    time.sleep(0.4)
    emit(
        {
            "type": "training.progress",
            "experiment_id": job.experiment_id,
            "progress": 75,
            "message_key": "training.in_progress",
        }
    )
    return {"experiment_id": job.experiment_id, "execution_id": "execution"}


def _failing_worker(job: TrainingJob, emit: Any) -> dict[str, Any]:
    del job, emit
    raise RuntimeError("intentional worker failure")


def _job(experiment_id: str = "experiment") -> TrainingJob:
    return TrainingJob(
        experiment_id=experiment_id,
        database_url="sqlite:////tmp/job-executor-unused.db",
        dataset_path="/tmp/job-executor-datasets",
        artifact_path="/tmp/job-executor-artifacts",
    )


def _wait_for_terminal(executor: LocalProcessExecutor, job_id: str, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = executor.status(job_id)
        if snapshot.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return snapshot
        time.sleep(0.02)
    raise AssertionError("Job did not reach a terminal state")


def test_process_executor_returns_immediately_and_streams_progress() -> None:
    executor = LocalProcessExecutor(
        worker=_slow_worker,
        cancellation_handler=lambda job: None,
    )
    started = time.monotonic()
    snapshot = executor.submit(_job())

    assert time.monotonic() - started < 0.3
    assert snapshot.status == JobStatus.QUEUED
    assert executor.status(snapshot.job_id).status in {
        JobStatus.QUEUED,
        JobStatus.RUNNING,
    }

    completed = _wait_for_terminal(executor, snapshot.job_id)
    events = executor.events(snapshot.job_id)
    assert completed.status == JobStatus.COMPLETED
    assert [event.type for event in events] == [
        "job.queued",
        "job.started",
        "training.progress",
        "training.progress",
        "job.completed",
    ]
    assert events[2].payload["progress"] == 25
    assert executor.events(snapshot.job_id, after=events[2].sequence)[0].payload["progress"] == 75


def test_process_executor_rejects_duplicates_and_cancels_running_job() -> None:
    executor = LocalProcessExecutor(
        worker=_slow_worker,
        cancellation_handler=lambda job: None,
    )
    submitted = executor.submit(_job("same-experiment"))

    with pytest.raises(JobAlreadyRunningError):
        executor.submit(_job("same-experiment"))

    cancelled = executor.cancel(submitted.job_id)
    assert cancelled.status == JobStatus.CANCELLED
    assert executor.events(submitted.job_id)[-1].type == "job.cancelled"
    with pytest.raises(JobStateError):
        executor.cancel(submitted.job_id)


def test_process_executor_surfaces_worker_errors() -> None:
    executor = LocalProcessExecutor(
        worker=_failing_worker,
        cancellation_handler=lambda job: None,
    )
    submitted = executor.submit(_job("failure"))

    failed = _wait_for_terminal(executor, submitted.job_id)

    assert failed.status == JobStatus.FAILED
    assert failed.error == "RuntimeError: intentional worker failure"
    assert executor.events(submitted.job_id)[-1].type == "job.failed"
