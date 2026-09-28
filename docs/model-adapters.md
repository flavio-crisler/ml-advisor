# Model Adapters

Model libraries are isolated behind `ModelAdapter` in `backend/app/ml/models/base.py`. An adapter standardizes `fit`, `predict`, optional probabilities/classes, parameter access, Optuna search space, and declared capabilities.

## Built-in IDs

Classification: `logistic_regression`, `random_forest_classifier`, `xgboost_classifier`, `catboost_classifier`, `lightgbm_classifier`, `knn_classifier`, and `svm_classifier`.

Regression: `linear_regression`, `random_forest_regressor`, `xgboost_regressor`, `catboost_regressor`, `lightgbm_regressor`, `knn_regressor`, and `svr`.

Call `GET /api/v1/models` for current task support and capabilities. The registry is authoritative for probability support, class weighting, native categorical input, GPU availability, and required scaling. A declared GPU capability does not enable GPU parameters automatically.

## Add an adapter

1. Implement `ModelAdapter` in `backend/app/ml/models/` without leaking its library into the experiment engine.
2. Define deterministic defaults, including `random_state`/thread limits where supported, and a bounded search space.
3. Add capability metadata in `ml/registries/catalog.py`.
4. Register the factory in `ml/models/registry.py`. Metadata and factory ID sets must match.
5. Add isolated tests to `backend/tests/test_model_adapters.py`, including defaults, parameter round-trip, fit/predict, probability behavior, and incompatible tasks.

Do not silently coerce an unsupported metric or preprocessing strategy. Let registry/spec validation reject incompatible combinations before training. For probability metrics, the adapter must return probabilities with a stable class order from `get_classes()`.
