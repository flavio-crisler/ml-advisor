from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

TuningDirection = Literal["maximize", "minimize"]
SearchSpace = Callable[[Any], Mapping[str, Any]]
TuningObjective = Callable[[Any, Mapping[str, Any]], float]
TrialCallback = Callable[["TuningTrialResult"], None]


class TuningTrialPruned(Exception):
    """Sinaliza que um trial de otimização pode ser encerrado antecipadamente."""


@dataclass(frozen=True, slots=True)
class TuningTrialResult:
    """Resultado de uma tentativa individual (trial) de ajuste de hiperparâmetros.

    Attributes:
        number (int): Número sequencial de identificação do trial.
        parameters (dict[str, Any]): Configuração de hiperparâmetros testada na tentativa.
        status (Literal["completed", "pruned", "failed"]): Status final da tentativa.
        value (float | None): Valor numérico da métrica atingido, ou None se descartado.
    """

    number: int
    parameters: dict[str, Any]
    status: Literal["completed", "pruned", "failed"]
    value: float | None


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Resultado consolidado da otimização de hiperparâmetros de um modelo.

    Attributes:
        best_parameters (dict[str, Any]): Conjunto de parâmetros que obteve o melhor desempenho.
        best_value (float): Melhor pontuação alcançada para a métrica objetivo.
        trials (tuple[TuningTrialResult, ...]): Histórico de todas as tentativas executadas.
    """

    best_parameters: dict[str, Any]
    best_value: float
    trials: tuple[TuningTrialResult, ...]


class TunerAdapter(ABC):
    """Interface abstrata para adaptadores de bibliotecas de busca de hiperparâmetros."""

    @abstractmethod
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
        """Executa a otimização de hiperparâmetros sobre o espaço de busca fornecido.

        Args:
            search_space: Função geradora das sugestões de parâmetros para cada trial.
            objective: Função de avaliação que treina e pontua a configuração proposta.
            direction: Direção de otimização ('maximize' ou 'minimize').
            trials: Quantidade de tentativas a executar no estudo.
            random_seed: Semente pseudoaleatória para reprodutibilidade.
            fixed_parameters: Mapeamento de hiperparâmetros fixos a incluir na avaliação.
            on_trial: Callback opcional executado ao final de cada trial.

        Returns:
            TuningResult consolidado com os melhores parâmetros e histórico dos trials.
        """
