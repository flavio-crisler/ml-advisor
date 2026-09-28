import pytest

from app.domain.exceptions import ExperimentExecutionError
from app.ml.tuning.base import TuningTrialPruned
from app.ml.tuning.optuna_adapter import OptunaTunerAdapter


def test_optuna_is_reproducible_and_reports_completed_trials() -> None:
    def optimize():
        observed = []
        result = OptunaTunerAdapter().optimize(
            search_space=lambda trial: {"x": trial.suggest_int("x", 0, 10)},
            objective=lambda _trial, parameters: float((parameters["x"] - 3) ** 2),
            direction="minimize",
            trials=8,
            random_seed=41,
            fixed_parameters={"fixed": True},
            on_trial=observed.append,
        )
        return result, observed

    first, first_observed = optimize()
    second, second_observed = optimize()

    assert first == second
    assert tuple(first_observed) == first.trials
    assert tuple(second_observed) == second.trials
    assert first.best_parameters["fixed"] is True
    assert all(trial.status == "completed" for trial in first.trials)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(ValueError("broken"), "failed"), (TuningTrialPruned("slow"), "pruned")],
)
def test_optuna_records_unsuccessful_trials_and_requires_a_completed_trial(
    error: Exception, expected_status: str
) -> None:
    observed = []

    def fail(_trial, _parameters):
        raise error

    with pytest.raises(ExperimentExecutionError, match="All tuning trials"):
        OptunaTunerAdapter().optimize(
            search_space=lambda trial: {"x": trial.suggest_int("x", 0, 1)},
            objective=fail,
            direction="maximize",
            trials=2,
            random_seed=41,
            on_trial=observed.append,
        )

    assert len(observed) == 2
    assert all(trial.status == expected_status for trial in observed)
