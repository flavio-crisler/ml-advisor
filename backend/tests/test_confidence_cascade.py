import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.application.confidence_cascade.service import (
    ConfidenceCascadeService,
    ConfidenceCascadeSimulator,
)
from app.domain.dataset.entities import Dataset
from app.domain.decision.cascade import CascadePolicy
from app.domain.decision.questions import NoulQuestionSpec
from app.domain.decision.runs import DecisionRun, DecisionRunStatus, ZeroShotDecisionSpec
from app.domain.decision.specifications import DecisionSpec
from app.domain.experiment.spec import TaskType
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyCascadeRunRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyDecisionRunRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage


def test_policy_validates_threshold_order() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        CascadePolicy(
            source_decision_run_id="run",
            auto_accept_threshold=0.7,
            review_threshold=0.8,
        )


def test_simulation_is_deterministic_and_threshold_changes_need_no_provider() -> None:
    simulator = ConfidenceCascadeSimulator()
    calls = 0
    inputs = {
        "confidences": [0.95, 0.85, 0.65, 0.55],
        "correct": [True, True, False, True],
        "decision_cost": 0.2,
    }
    first = simulator.simulate(
        policy=CascadePolicy(
            source_decision_run_id="run",
            auto_accept_threshold=0.9,
            review_threshold=0.6,
            manual_review_unit_cost=1,
        ),
        **inputs,
    )
    second = simulator.simulate(
        policy=CascadePolicy(
            source_decision_run_id="run",
            auto_accept_threshold=0.8,
            review_threshold=0.6,
            manual_review_unit_cost=1,
        ),
        **inputs,
    )

    assert calls == 0
    assert first["coverage"] == 0.25
    assert second["coverage"] == 0.5
    assert first["review_rate"] == 0.75
    assert first["accuracy_auto"] == 1
    assert len(first["threshold_sensitivity"]) == 21


def test_cascade_uses_persisted_decision_artifact_and_can_be_reopened(
    migrated_database: Any, tmp_path: Path
) -> None:
    storage = LocalFilesystemStorage(tmp_path / "artifacts")
    dataset = SqlAlchemyDatasetRepository(migrated_database).add(
        Dataset(
            id="cascade-dataset",
            name="Cascade",
            original_filename="cascade.csv",
            content_hash="cascade-hash",
            row_count=3,
            column_count=2,
            storage_path="unused.csv",
        )
    )
    spec = ZeroShotDecisionSpec(
        dataset_id=dataset.id,
        target="target",
        task=TaskType.BINARY_CLASSIFICATION,
        positive_class="yes",
        input_columns=["message"],
        question=NoulQuestionSpec(instructions="Will this happen?"),
        provider="typesafe_jev",
        model="jev-latest",
    )
    question_hash = DecisionSpec(
        provider=spec.provider,
        model=spec.model,
        questions={"prediction": spec.question},
    ).question_spec_hash()
    decision_runs = SqlAlchemyDecisionRunRepository(migrated_database)
    run = decision_runs.add(DecisionRun.create(spec, row_count=3, question_spec_hash=question_hash))
    decision_runs.transition(run.id, DecisionRunStatus.RUNNING)
    decision_runs.transition(
        run.id,
        DecisionRunStatus.COMPLETED,
        processed_rows=3,
        resolved_model="jev-1.13.0",
        estimated_cost=0.03,
        artifact_path=f"decision/runs/{run.id}/summary.json",
    )
    frame = pd.DataFrame(
        {
            "row_id": ["0", "1", "2"],
            "confidence": [0.95, 0.8, 0.55],
            "correct": [True, False, True],
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    storage.save(f"decision/runs/{run.id}/responses.parquet", buffer.getvalue())
    cascades = SqlAlchemyCascadeRunRepository(migrated_database)
    service = ConfidenceCascadeService(
        decision_runs=decision_runs,
        cascades=cascades,
        storage=storage,
    )

    created = service.create(
        CascadePolicy(
            source_decision_run_id=run.id,
            auto_accept_threshold=0.9,
            review_threshold=0.6,
        )
    )
    reopened = cascades.get(created.id)

    assert reopened is not None
    assert reopened.policy.auto_accept_threshold == 0.9
    assert reopened.metrics["coverage"] == pytest.approx(1 / 3)
    assert storage.load(reopened.artifact_path)
