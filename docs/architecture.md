# Architecture

## Runtime shape

The application is a local-first modular monolith:

```text
React/Vite SPA -> Nginx -> FastAPI routes -> application services
                                      |-> domain contracts
                                      |-> ML/LLM adapters and registries
                                      |-> SQLAlchemy repositories -> SQLite
                                      `-> local storage -> datasets/artifacts
```

Flyway migrates the SQLite database before the backend starts. Nginx serves the production SPA and proxies `/api/` to FastAPI. A local process executor runs each training job outside the API process; the frontend receives job events through Server-Sent Events.

## Backend boundaries

- `api/`: HTTP schemas, status mapping, dependency assembly, and SSE.
- `application/`: dataset, Advisor, experiment, decision, semantic-feature, job, and training use cases.
- `domain/`: framework-free entities, repository protocols, and `ExperimentSpec`.
- `domain/decision/`: provider-neutral questions, responses, runs, semantic sets, and cascades.
- `ml/`: preprocessing compiler, model/tuner adapters, validation, metrics, and artifacts.
- `llm/`: provider contract, structured-output implementations, and transport.
- `infrastructure/`: SQLAlchemy repositories, database session, and filesystem storage.

Dependencies point inward: domain code does not import FastAPI, SQLAlchemy, pandas, or concrete providers. Registries expose stable component IDs and capability metadata, so orchestration never branches on ML libraries.

## Experiment lifecycle

CSV bytes are hashed and stored once. Deterministic profiling persists a versioned profile and preview. Auto, Assisted, and Expert all produce the same validated spec. Training verifies the dataset hash, compiles preprocessing per model capability, executes validation and optional Optuna trials, persists one `ModelRun` per model, then writes comparison and analysis artifacts.

Executed configurations and historical rows are immutable. Parameter or pipeline changes create a child experiment through clone/retrain, preserving `parent_experiment_id`.

## Decision lifecycle

Decision features are optional and disabled by default. A reviewed question and selected state
columns are frozen before a local process calls the configured `DecisionProvider`. The worker
verifies dataset bytes against their SHA-256 hash, sends neither target nor unselected columns,
checkpoints progress, and reuses a content-addressed row cache. SQLite stores lifecycle and summary
metadata; high-volume decision and semantic rows are immutable Parquet artifacts.

Calibration and confidence cascades operate only on persisted probabilities. Semantic feature
proposal uses the existing `LLMProvider`, while feature generation uses `DecisionProvider`; the ML
transformer performs no network calls. Optional impact evaluation holds dataset, split, seed,
models, parameters, validation, and tuning budget constant between baseline and augmented arms.

## Persistence and operations

`data/database/app.db`, `data/datasets/`, and `data/experiments/` must share persistent storage. Job
queues and SSE cursors are process-local, but datasets, experiment/decision history, row cache,
metrics, trials, and artifacts survive restarts. A retry creates a new immutable run and can resume
completed work through the persisted cache. JSON logs include correlation fields for experiment,
run, and job IDs; `LOG_LEVEL` controls verbosity. External observability, multi-user authentication,
distributed workers, and model serving are outside the MVP.
