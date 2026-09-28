import io
import json
from typing import Any

import numpy as np
import pandas as pd

from app.domain.common import canonical_json, new_id, utc_now
from app.domain.decision.cascade import BelowReviewAction, CascadePolicy, CascadeRun
from app.domain.decision.repositories import CascadeRunRepository, DecisionRunRepository
from app.domain.decision.runs import DecisionRunStatus
from app.domain.exceptions import DecisionBenchmarkError, EntityNotFoundError
from app.infrastructure.storage.base import ArtifactStorage


class ConfidenceCascadeSimulator:
    """Simulador estatístico para políticas de cascata baseadas em confiança."""

    def simulate(
        self,
        *,
        policy: CascadePolicy,
        confidences: list[float],
        correct: list[bool],
        decision_cost: float | None,
    ) -> dict[str, Any]:
        """Calcula métricas de cobertura, acurácia e sensibilidade para a política.

        Args:
            policy (CascadePolicy): Política com limiares de aceitação automática e revisão.
            confidences (list[float]): Vetor de pontuações de confiança das predições.
            correct (list[bool]): Vetor booleano indicando acerto/erro de cada predição.
            decision_cost (float | None): Custo base estimado das decisões do modelo.

        Returns:
            dict[str, Any]: Dicionário com taxas operacionais, custos e curva de sensibilidade.

        Raises:
            DecisionBenchmarkError: Se os vetores forem vazios, divergentes ou com valores fora
                do intervalo [0, 1].
        """
        if not confidences or len(confidences) != len(correct):
            raise DecisionBenchmarkError(
                "Cascade requires equal non-empty confidence and correctness vectors"
            )
        confidence = np.asarray(confidences, dtype=float)
        correctness = np.asarray(correct, dtype=bool)
        if np.any((confidence < 0) | (confidence > 1)):
            raise DecisionBenchmarkError("Cascade confidence must be between zero and one")
        current = self._point(
            threshold=policy.auto_accept_threshold,
            review_threshold=policy.review_threshold,
            action=policy.below_review_action,
            manual_review_unit_cost=policy.manual_review_unit_cost,
            confidence=confidence,
            correctness=correctness,
            decision_cost=decision_cost,
        )
        thresholds = [
            self._point(
                threshold=float(threshold),
                review_threshold=min(policy.review_threshold, float(threshold)),
                action=policy.below_review_action,
                manual_review_unit_cost=policy.manual_review_unit_cost,
                confidence=confidence,
                correctness=correctness,
                decision_cost=decision_cost,
            )
            for threshold in np.linspace(0.5, 1.0, 21)
        ]
        counts, edges = np.histogram(confidence, bins=10, range=(0, 1))
        return {
            "schema_version": "1.0",
            **current,
            "decision_cost": decision_cost,
            "threshold_sensitivity": thresholds,
            "confidence_distribution": [
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(count),
                }
                for index, count in enumerate(counts)
            ],
        }

    @staticmethod
    def _point(
        *,
        threshold: float,
        review_threshold: float,
        action: BelowReviewAction,
        manual_review_unit_cost: float | None,
        confidence: np.ndarray[Any, Any],
        correctness: np.ndarray[Any, Any],
        decision_cost: float | None,
    ) -> dict[str, Any]:
        """Calcula as métricas operacionais para um limiar específico de decisão.

        Args:
            threshold (float): Limiar para aceitação automática da predição.
            review_threshold (float): Limiar abaixo do qual ocorre ação de revisão/abstenção.
            action (BelowReviewAction): Ação adotada abaixo do limiar de revisão.
            manual_review_unit_cost (float | None): Custo financeiro unitário de revisão humana.
            confidence (np.ndarray[Any, Any]): Array com confianças calculadas.
            correctness (np.ndarray[Any, Any]): Array com flags de corretude.
            decision_cost (float | None): Custo base de inferência das decisões.

        Returns:
            dict[str, Any]: Métricas calculadas para o ponto de corte avaliado.
        """
        auto = confidence >= threshold
        low = confidence < review_threshold
        review = ~auto
        abstain = np.zeros_like(auto)
        if action == BelowReviewAction.ABSTAIN:
            abstain = low
            review = review & ~low
        auto_count = int(auto.sum())
        review_count = int(review.sum())
        total = len(confidence)
        accuracy = float(correctness[auto].mean()) if auto_count else None
        estimated_total_cost = None
        if decision_cost is not None:
            estimated_total_cost = decision_cost
            if manual_review_unit_cost is not None:
                estimated_total_cost += manual_review_unit_cost * review_count
        return {
            "threshold": threshold,
            "coverage": auto_count / total,
            "auto_decision_rate": auto_count / total,
            "review_rate": review_count / total,
            "abstain_rate": int(abstain.sum()) / total,
            "accuracy_auto": accuracy,
            "error_rate_auto": (1 - accuracy) if accuracy is not None else None,
            "estimated_total_cost": estimated_total_cost,
        }


class ConfidenceCascadeService:
    """Gerencia a simulação e persistência de políticas de cascata de confiança.

    Attributes:
        decision_runs (DecisionRunRepository): Repositório para consulta de benchmarks.
        cascades (CascadeRunRepository): Repositório para persistência de execuções de cascata.
        storage (ArtifactStorage): Armazenamento de arquivos e métricas serializadas.
        simulator (ConfidenceCascadeSimulator): Simulador estatístico de limiares.
    """

    def __init__(
        self,
        *,
        decision_runs: DecisionRunRepository,
        cascades: CascadeRunRepository,
        storage: ArtifactStorage,
        simulator: ConfidenceCascadeSimulator | None = None,
    ) -> None:
        """Inicializa o serviço de cascata de confiança com os repositórios e storage.

        Args:
            decision_runs (DecisionRunRepository): Repositório de execuções de benchmark.
            cascades (CascadeRunRepository): Repositório para execuções de cascata.
            storage (ArtifactStorage): Serviço de persistência de artefatos.
            simulator (ConfidenceCascadeSimulator | None): Instância do simulador de cascata.
        """
        self.decision_runs = decision_runs
        self.cascades = cascades
        self.storage = storage
        self.simulator = simulator or ConfidenceCascadeSimulator()

    def create(self, policy: CascadePolicy) -> CascadeRun:
        """Executa a simulação da política sobre o benchmark e persiste os resultados.

        Args:
            policy (CascadePolicy): Política contendo limiares e identificador do benchmark.

        Returns:
            CascadeRun: Entidade persistida com métricas e caminhos de artefatos.

        Raises:
            EntityNotFoundError: Se o benchmark de decisão de origem não for encontrado.
            DecisionBenchmarkError: Se o benchmark não estiver concluído ou faltar colunas.
        """
        run = self.decision_runs.get(policy.source_decision_run_id)
        if run is None:
            raise EntityNotFoundError(f"Decision run {policy.source_decision_run_id} was not found")
        if run.status != DecisionRunStatus.COMPLETED:
            raise DecisionBenchmarkError("Cascade requires a completed DecisionRun")
        responses_path = f"decision/runs/{run.id}/responses.parquet"
        frame = pd.read_parquet(io.BytesIO(self.storage.load(responses_path)))
        if "confidence" not in frame or "correct" not in frame:
            raise DecisionBenchmarkError("Decision artifact lacks cascade inputs")
        metrics = self.simulator.simulate(
            policy=policy,
            confidences=frame["confidence"].astype(float).tolist(),
            correct=frame["correct"].astype(bool).tolist(),
            decision_cost=run.estimated_cost,
        )
        cascade_id = new_id()
        prefix = f"decision/cascades/{cascade_id}"
        self.storage.save(
            f"{prefix}/policy.json",
            json.dumps(
                policy.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
            ).encode(),
        )
        metrics_path = self.storage.save(
            f"{prefix}/metrics.json",
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True).encode(),
        )
        return self.cascades.add(
            CascadeRun(
                id=cascade_id,
                decision_run_id=run.id,
                policy_json=canonical_json(policy.model_dump(mode="json")),
                metrics_json=canonical_json(metrics),
                artifact_path=metrics_path,
                created_at=utc_now(),
            )
        )
