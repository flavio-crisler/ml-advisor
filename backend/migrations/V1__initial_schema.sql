CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    column_count INTEGER NOT NULL CHECK (column_count >= 0),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    storage_path TEXT NOT NULL,
    profile_path TEXT
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    mode TEXT NOT NULL CHECK (mode IN ('auto', 'assisted', 'expert')),
    status TEXT NOT NULL CHECK (status IN ('draft', 'queued', 'running', 'completed', 'failed', 'cancelled')),
    dataset_id TEXT NOT NULL,
    parent_experiment_id TEXT,
    objective TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    experiment_spec_json TEXT NOT NULL CHECK (json_valid(experiment_spec_json)),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE RESTRICT,
    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(id) ON DELETE RESTRICT
);

CREATE TABLE experiment_runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    run_index INTEGER NOT NULL CHECK (run_index > 0),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    started_at DATETIME,
    finished_at DATETIME,
    duration_seconds REAL CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
    random_seed INTEGER NOT NULL,
    parameters_json TEXT NOT NULL CHECK (json_valid(parameters_json)),
    metrics_json TEXT CHECK (metrics_json IS NULL OR json_valid(metrics_json)),
    artifact_path TEXT,
    error_message TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE RESTRICT,
    UNIQUE (experiment_id, model_id, run_index)
);

CREATE TABLE tuning_trials (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trial_number INTEGER NOT NULL CHECK (trial_number >= 0),
    parameters_json TEXT NOT NULL CHECK (json_valid(parameters_json)),
    metric_value REAL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'pruned', 'failed')),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE RESTRICT,
    UNIQUE (run_id, trial_number)
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_experiments_dataset_id ON experiments(dataset_id);
CREATE INDEX idx_experiments_created_at ON experiments(created_at);
CREATE INDEX idx_experiment_runs_experiment_id ON experiment_runs(experiment_id);
CREATE INDEX idx_tuning_trials_run_id ON tuning_trials(run_id);

CREATE TRIGGER prevent_experiment_config_update_after_run
BEFORE UPDATE OF mode, dataset_id, parent_experiment_id, objective, spec_version, experiment_spec_json
ON experiments
FOR EACH ROW
WHEN EXISTS (SELECT 1 FROM experiment_runs WHERE experiment_id = OLD.id)
BEGIN
    SELECT RAISE(ABORT, 'experiment_config_immutable_after_first_run');
END;

CREATE TRIGGER prevent_terminal_run_update
BEFORE UPDATE ON experiment_runs
FOR EACH ROW
WHEN OLD.status IN ('completed', 'failed', 'cancelled')
BEGIN
    SELECT RAISE(ABORT, 'terminal_model_run_is_immutable');
END;

CREATE TRIGGER prevent_model_run_delete
BEFORE DELETE ON experiment_runs
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'model_run_history_cannot_be_deleted');
END;
