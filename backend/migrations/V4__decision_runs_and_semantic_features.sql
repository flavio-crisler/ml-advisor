CREATE TABLE decision_runs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    experiment_id TEXT,
    provider_id TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    resolved_model TEXT,
    decision_spec_version TEXT NOT NULL,
    question_spec_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL CHECK (json_valid(spec_json)),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    processed_rows INTEGER NOT NULL DEFAULT 0 CHECK (processed_rows >= 0),
    request_count INTEGER NOT NULL DEFAULT 0 CHECK (request_count >= 0),
    cache_hits INTEGER NOT NULL DEFAULT 0 CHECK (cache_hits >= 0),
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    estimated_cost REAL CHECK (estimated_cost IS NULL OR estimated_cost >= 0),
    latency_ms_total REAL CHECK (latency_ms_total IS NULL OR latency_ms_total >= 0),
    artifact_path TEXT,
    calibration_run_id TEXT,
    error_message TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    finished_at DATETIME,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE RESTRICT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE RESTRICT
);

CREATE TABLE calibration_runs (
    id TEXT PRIMARY KEY,
    decision_run_id TEXT NOT NULL UNIQUE,
    task TEXT NOT NULL,
    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
    artifact_path TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_run_id) REFERENCES decision_runs(id) ON DELETE RESTRICT
);

CREATE TABLE decision_cache (
    context_key TEXT PRIMARY KEY,
    cache_key TEXT NOT NULL UNIQUE,
    dataset_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL,
    question_spec_hash TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    resolved_model TEXT NOT NULL,
    response_json TEXT NOT NULL CHECK (json_valid(response_json)),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE semantic_feature_sets (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    source_experiment_id TEXT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    discovery_mode TEXT NOT NULL CHECK (discovery_mode IN ('target_agnostic', 'target_aware_development')),
    questions_frozen INTEGER NOT NULL CHECK (questions_frozen IN (0, 1)),
    question_spec_json TEXT NOT NULL CHECK (json_valid(question_spec_json)),
    provider_id TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    resolved_model TEXT,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    processed_rows INTEGER NOT NULL DEFAULT 0 CHECK (processed_rows >= 0),
    request_count INTEGER NOT NULL DEFAULT 0 CHECK (request_count >= 0),
    cache_hits INTEGER NOT NULL DEFAULT 0 CHECK (cache_hits >= 0),
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    estimated_cost REAL CHECK (estimated_cost IS NULL OR estimated_cost >= 0),
    latency_ms_total REAL CHECK (latency_ms_total IS NULL OR latency_ms_total >= 0),
    artifact_path TEXT,
    impact_json TEXT CHECK (impact_json IS NULL OR json_valid(impact_json)),
    error_message TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    finished_at DATETIME,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE RESTRICT,
    FOREIGN KEY (source_experiment_id) REFERENCES experiments(id) ON DELETE RESTRICT
);

CREATE INDEX idx_decision_runs_dataset ON decision_runs(dataset_id);
CREATE INDEX idx_decision_runs_created ON decision_runs(created_at);
CREATE INDEX idx_decision_cache_lookup ON decision_cache(
    dataset_hash, row_hash, question_spec_hash, provider_id, requested_model
);
CREATE INDEX idx_semantic_feature_sets_dataset ON semantic_feature_sets(dataset_id);

CREATE TRIGGER prevent_decision_run_spec_update
BEFORE UPDATE OF dataset_id, experiment_id, provider_id, requested_model,
    decision_spec_version, question_spec_hash, spec_json, row_count
ON decision_runs
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'decision_run_spec_is_immutable');
END;

CREATE TRIGGER prevent_calibration_run_update
BEFORE UPDATE ON calibration_runs
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'calibration_run_is_immutable');
END;

CREATE TRIGGER prevent_semantic_feature_questions_update
BEFORE UPDATE OF dataset_id, source_experiment_id, version, discovery_mode,
    questions_frozen, question_spec_json, provider_id, requested_model, row_count
ON semantic_feature_sets
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'semantic_feature_questions_are_frozen');
END;
