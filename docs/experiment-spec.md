# ExperimentSpec v1

`ExperimentSpec` is the reproducibility contract shared by Auto, Assisted, Expert, the API, persistence, and the engine. The canonical Pydantic model is `backend/app/domain/experiment/spec.py`; [examples/experiment-spec.binary.json](../examples/experiment-spec.binary.json) is a copy-ready structural example.

## Required decisions

- Dataset identity: `dataset_id`, SHA-256 `dataset_hash`, and `profile_version`.
- Intent: `name`, `objective`, and task (`binary_classification`, `multiclass_classification`, or `regression`).
- Columns: one target, a unique feature list, and an explicit role for every referenced column.
- Execution: enabled models, preprocessing strategies, validation, metrics, and optional tuning.

Validate a payload before creating an experiment:

```bash
curl -sS -X POST http://localhost:8000/api/v1/experiment-specs/validate \
  -H 'Content-Type: application/json' \
  --data @examples/experiment-spec.binary.json
```

Replace the example dataset fields and column names with values returned by the dataset API before `POST /api/v1/experiments`.

## Invariants

The target cannot be a feature; feature/model/metric/preprocessing IDs are unique; exactly one target role and at least one enabled model are required. Binary precision/recall/F1/ROC-AUC/PR-AUC require `positive_label`. Multiclass precision/recall/F1 require `averaging`; multiclass ROC-AUC requires `roc_auc_strategy`. Probability metrics reject adapters without probability support. Enabled tuning requires a selected optimization metric.

Creation also verifies that the dataset exists, its hash matches, the profile version is current, and all referenced columns exist. After execution starts, the database makes the contract immutable. Clone the experiment, edit the draft clone, and retrain instead of mutating history.

Artifacts include the exact spec, runtime package versions, validation indices/configuration, effective model parameters, preprocessing config, metrics, analysis, serialized model pipeline, and leaderboard.
