ALTER TABLE experiment_runs ADD COLUMN execution_id TEXT;

CREATE INDEX idx_experiment_runs_execution_id ON experiment_runs(execution_id);
