from collections.abc import Mapping
from typing import Any

import optuna

from app.domain.exceptions import ExperimentExecutionError
from app.ml.tuning.base import (
    SearchSpace,
    TrialCallback,
    TunerAdapter,
    TuningDirection,
    TuningObjective,
    TuningResult,
    TuningTrialPruned,
    TuningTrialResult,
)


class OptunaTunerAdapter(TunerAdapter):
    """Adaptador para otimização de hiperparâmetros utilizando o framework Optuna."""

    def optimize(
        self,
        *,
        search_space: SearchSpace,
        objective: TuningObjective,
        direction: TuningDirection,
        trials: int,
        random_seed: int,
        fixed_parameters: Mapping[str, Any] | None = None,
        on_trial: TrialCallback | None = None,
    ) -> TuningResult:
        """Executa o estudo Optuna utilizando amostrador TPE e podador por mediana.

        Args:
            search_space: Função geradora do espaço de busca para cada trial do Optuna.
            objective: Função de avaliação que treina o modelo e retorna o score.
            direction: Sentido de otimização ('maximize' ou 'minimize').
            trials: Quantidade de tentativas a executar.
            random_seed: Semente pseudoaleatória para reprodutibilidade do amostrador TPE.
            fixed_parameters: Parâmetros fixos a serem combinados com as sugestões.
            on_trial: Callback opcional executado a cada tentativa avaliada.

        Returns:
            TuningResult com os melhores hiperparâmetros encontrados e histórico.

        Raises:
            ExperimentExecutionError: Se todos os trials falharem ou forem podados.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1),
        )
        fixed = dict(fixed_parameters or {})
        completed: list[TuningTrialResult] = []

        def evaluate(trial: optuna.Trial) -> float:
            parameters = {**fixed, **dict(search_space(trial))}
            try:
                return float(objective(trial, parameters))
            except TuningTrialPruned as error:
                raise optuna.TrialPruned(str(error)) from error

        def record(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            status_by_state = {
                optuna.trial.TrialState.COMPLETE: "completed",
                optuna.trial.TrialState.PRUNED: "pruned",
                optuna.trial.TrialState.FAIL: "failed",
            }
            status = status_by_state.get(trial.state)
            if status is None:
                return
            result = TuningTrialResult(
                number=trial.number,
                parameters={**fixed, **dict(trial.params)},
                status=status,
                value=float(trial.value) if trial.value is not None else None,
            )
            completed.append(result)
            if on_trial is not None:
                on_trial(result)

        study.optimize(
            evaluate,
            n_trials=trials,
            callbacks=[record],
            catch=(Exception,),
            show_progress_bar=False,
        )
        successful = [trial for trial in completed if trial.status == "completed"]
        if not successful:
            raise ExperimentExecutionError("All tuning trials failed or were pruned")
        return TuningResult(
            best_parameters={**fixed, **dict(study.best_params)},
            best_value=float(study.best_value),
            trials=tuple(completed),
        )
