CREATE TABLE cascade_runs (
    id TEXT PRIMARY KEY,
    decision_run_id TEXT NOT NULL,
    policy_json TEXT NOT NULL CHECK (json_valid(policy_json)),
    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
    artifact_path TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_run_id) REFERENCES decision_runs(id) ON DELETE RESTRICT
);

CREATE INDEX idx_cascade_runs_decision_run ON cascade_runs(decision_run_id);

CREATE TRIGGER prevent_cascade_run_update
BEFORE UPDATE ON cascade_runs
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'cascade_run_is_immutable');
END;

CREATE TRIGGER prevent_cascade_run_delete
BEFORE DELETE ON cascade_runs
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'cascade_run_history_cannot_be_deleted');
END;
