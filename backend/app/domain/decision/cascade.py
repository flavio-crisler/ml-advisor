from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.common import canonical_json, json_object, new_id, utc_now


class BelowReviewAction(StrEnum):
    """Representa a ação operacional adotada quando a confiança fica abaixo do limiar de revisão."""

    MANUAL_REVIEW = "manual_review"
    ABSTAIN = "abstain"


class CascadePolicy(BaseModel):
    """Define as regras e limiares de decisão de uma política de cascata de confiança.

    Attributes:
        version (Literal["1.0"]): Versão do contrato da política.
        source_decision_run_id (str): Identificador da execução de decisão avaliada.
        auto_accept_threshold (float): Limiar mínimo para aceitação automática da predição.
        review_threshold (float): Limiar abaixo do qual ocorre revisão humana ou abstenção.
        below_review_action (BelowReviewAction): Comportamento quando abaixo do limiar de revisão.
        manual_review_unit_cost (float | None): Custo estimado por item em revisão manual.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["1.0"] = "1.0"
    source_decision_run_id: str = Field(min_length=1)
    auto_accept_threshold: float = Field(default=0.9, ge=0.5, le=1)
    review_threshold: float = Field(default=0.6, ge=0, le=1)
    below_review_action: BelowReviewAction = BelowReviewAction.MANUAL_REVIEW
    manual_review_unit_cost: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "CascadePolicy":
        """Valida se a relação de ordem entre os limiares é matematicamente consistente.

        Returns:
            CascadePolicy: Própria instância validada.

        Raises:
            ValueError: Se o limiar de revisão for estritamente maior que o de aceitação.
        """
        if self.review_threshold > self.auto_accept_threshold:
            raise ValueError("review_threshold cannot exceed auto_accept_threshold")
        return self


@dataclass(frozen=True, slots=True)
class CascadeRun:
    """Representa uma execução avaliada de política de cascata sobre inferências.

    Attributes:
        decision_run_id (str): Identificador da execução de decisão de origem.
        policy_json (str): Definição da política serializada em JSON.
        metrics_json (str): Métricas consolidadas da cascata serializadas em JSON.
        artifact_path (str): Caminho físico do artefato com as decisões roteadas.
        id (str): Identificador único universal da execução da cascata.
        created_at (datetime): Data e hora de execução em UTC.
    """

    decision_run_id: str
    policy_json: str
    metrics_json: str
    artifact_path: str
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        policy: CascadePolicy,
        metrics: Mapping[str, Any],
        artifact_path: str,
    ) -> "CascadeRun":
        """Cria uma nova execução de cascata com política e métricas serializadas.

        Args:
            policy (CascadePolicy): Política de cortes de confiança avaliada.
            metrics (Mapping[str, Any]): Dicionário com métricas calculadas.
            artifact_path (str): Caminho para gravação das previsões roteadas.

        Returns:
            CascadeRun: Instância da execução de cascata persistível.
        """
        return cls(
            decision_run_id=policy.source_decision_run_id,
            policy_json=canonical_json(policy.model_dump(mode="json")),
            metrics_json=canonical_json(metrics),
            artifact_path=artifact_path,
        )

    @property
    def policy(self) -> CascadePolicy:
        """Desserializa e reconstitui a entidade de política de cascata.

        Returns:
            CascadePolicy: Instância validada da política.
        """
        return CascadePolicy.model_validate(json_object(self.policy_json))

    @property
    def metrics(self) -> dict[str, Any]:
        """Desserializa as métricas apuradas na execução da cascata.

        Returns:
            dict[str, Any]: Mapeamento contendo cobertura, taxas de revisão e custos.
        """
        return json_object(self.metrics_json)
