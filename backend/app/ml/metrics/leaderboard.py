from dataclasses import dataclass

from app.domain.exceptions import MetricEvaluationError
from app.ml.registries.metadata import MetricMetadata
from app.ml.registries.registry import Registry


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """Representa o resultado de desempenho de uma execução (run) para exibição no leaderboard.

    Attributes:
        run_id (str): Identificador único da execução do experimento.
        model_id (str): Identificador do modelo de machine learning avaliado.
        metrics (dict[str, float]): Mapeamento das métricas computadas para este modelo.
        duration_seconds (float): Tempo total de execução do treinamento em segundos.
    """

    run_id: str
    model_id: str
    metrics: dict[str, float]
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """Tabela de classificação consolidada de execuções de experimentos.

    Attributes:
        primary_metric (str): Identificador da métrica primária utilizada para classificação.
        direction (str): Direção de otimização da métrica ('maximize' ou 'minimize').
        entries (tuple[LeaderboardEntry, ...]): Entradas ordenadas da melhor para a pior.
    """

    primary_metric: str
    direction: str
    entries: tuple[LeaderboardEntry, ...]


class LeaderboardBuilder:
    """Construtor e ordenador de leaderboards com base em métricas e tempo de execução.

    Attributes:
        registry (Registry[MetricMetadata]): Catálogo de metadados das métricas.
    """

    def __init__(self, registry: Registry[MetricMetadata]) -> None:
        """Inicializa o construtor com o catálogo de métricas.

        Args:
            registry: Registro de metadados contendo a direção de otimização das métricas.
        """
        self.registry = registry

    def build(self, primary_metric: str, entries: list[LeaderboardEntry]) -> Leaderboard:
        """Constrói e ordena o leaderboard com base na métrica primária e tempo de execução.

        Args:
            primary_metric: Identificador da métrica primária para critério de ranqueamento.
            entries: Lista de entradas de execuções a serem avaliadas e ranqueadas.

        Returns:
            Leaderboard contendo as entradas ordenadas do melhor desempenho para o pior.

        Raises:
            MetricEvaluationError: Se a métrica primária não estiver presente em
                alguma das entradas.
        """
        metadata = self.registry.get(primary_metric)
        missing = [entry.run_id for entry in entries if primary_metric not in entry.metrics]
        if missing:
            raise MetricEvaluationError(
                f"Primary metric {primary_metric} is missing for runs: {missing}"
            )
        multiplier = -1 if metadata.direction == "maximize" else 1
        ordered = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    multiplier * entry.metrics[primary_metric],
                    entry.duration_seconds,
                    entry.model_id,
                    entry.run_id,
                ),
            )
        )
        return Leaderboard(
            primary_metric=primary_metric,
            direction=metadata.direction,
            entries=ordered,
        )
