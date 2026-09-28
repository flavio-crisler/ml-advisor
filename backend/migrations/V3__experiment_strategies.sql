CREATE TABLE experiment_strategies (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL CHECK (source IN ('auto', 'assisted')),
    provider_id TEXT,
    strategy_json TEXT NOT NULL CHECK (json_valid(strategy_json)),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE RESTRICT
);

CREATE INDEX idx_experiment_strategies_experiment_id
ON experiment_strategies(experiment_id);

CREATE TRIGGER prevent_experiment_strategy_update
BEFORE UPDATE ON experiment_strategies
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'experiment_strategy_is_immutable');
END;

CREATE TRIGGER prevent_experiment_strategy_delete
BEFORE DELETE ON experiment_strategies
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'experiment_strategy_history_cannot_be_deleted');
END;
