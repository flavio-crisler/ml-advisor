CREATE TRIGGER prevent_model_run_config_update
BEFORE UPDATE OF experiment_id, model_id, run_index, random_seed, parameters_json
ON experiment_runs
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'model_run_config_is_immutable');
END;

CREATE TRIGGER prevent_experiment_delete
BEFORE DELETE ON experiments
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'experiment_history_cannot_be_deleted');
END;

