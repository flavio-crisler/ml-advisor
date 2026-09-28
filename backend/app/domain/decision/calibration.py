from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ReliabilityBin(BaseModel):
    """Representa um intervalo discreto de confiança em uma curva de calibração.

    Attributes:
        lower (float): Limite inferior do intervalo de confiança [0, 1].
        upper (float): Limite superior do intervalo de confiança [0, 1].
        count (int): Quantidade de previsões alocadas neste intervalo.
        mean_confidence (float): Média das probabilidades estimadas no intervalo.
        empirical_accuracy (float): Acurácia real observada nas amostras do intervalo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0, le=1)
    empirical_accuracy: float = Field(ge=0, le=1)


class ThresholdPoint(BaseModel):
    """Representa métricas de corte operacional sob determinado limiar de confiança.

    Attributes:
        threshold (float): Limiar mínimo de confiança para aceitação automática.
        coverage (float): Proporção de amostras com confiança maior ou igual ao limiar.
        accuracy (float | None): Acurácia entre as previsões aceitas.
        error_rate (float | None): Taxa de erro complementar observada.
        review_rate (float): Taxa de encaminhamento para revisão humana (1 - coverage).
        estimated_cost (float | None): Custo financeiro estimado da operação sob o limiar.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    review_rate: float = Field(ge=0, le=1)
    estimated_cost: float | None = Field(default=None, ge=0)


class CalibrationSummary(BaseModel):
    """Contém as métricas consolidadas de calibração de probabilidade de um modelo.

    Attributes:
        schema_version (str): Versão estrutural do esquema de sumarização.
        brier_score (float): Pontuação de Brier média avaliada.
        log_loss (float): Perda logarítmica (cross-entropy) apurada.
        expected_calibration_error (float): Erro Esperado de Calibração (ECE).
        maximum_calibration_error (float): Erro Máximo de Calibração (MCE).
        confidence_mean (float): Média global dos níveis de confiança emitidos.
        reliability (list[ReliabilityBin]): Partições da curva de calibração de confiabilidade.
        confidence_histogram (list[dict[str, Any]]): Histograma da distribuição de confianças.
        thresholds (list[ThresholdPoint]): Pontos calculados para a curva de limiares.
        latency_ms (list[float]): Amostras de latência de resposta em milissegundos.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    brier_score: float = Field(ge=0)
    log_loss: float = Field(ge=0)
    expected_calibration_error: float = Field(ge=0, le=1)
    maximum_calibration_error: float = Field(ge=0, le=1)
    confidence_mean: float = Field(ge=0, le=1)
    reliability: list[ReliabilityBin]
    confidence_histogram: list[dict[str, Any]]
    thresholds: list[ThresholdPoint]
    latency_ms: list[float]


def reliability_bins(
    confidences: list[float], correct: list[bool], *, bins: int = 10
) -> tuple[list[ReliabilityBin], float, float]:
    """Calcula os intervalos de confiabilidade, o ECE ponderado e o MCE máximo.

    Args:
        confidences (list[float]): Vetor de probabilidades ou confianças preditas [0, 1].
        correct (list[bool]): Vetor booleano indicando acertos da predição.
        bins (int): Quantidade de intervalos uniformes a subdividir.

    Returns:
        tuple[list[ReliabilityBin], float, float]: Lista de bins calculados, valor ECE e valor MCE.

    Raises:
        ValueError: Se as listas tiverem comprimentos distintos ou forem vazias.
    """
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidence and correctness vectors must have equal non-zero length")
    confidence_array = np.asarray(confidences, dtype=float)
    correct_array = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[ReliabilityBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            mask = (confidence_array >= lower) & (confidence_array < upper)
        count = int(mask.sum())
        if not count:
            continue
        mean_confidence = float(confidence_array[mask].mean())
        accuracy = float(correct_array[mask].mean())
        gap = abs(mean_confidence - accuracy)
        weighted_gap += gap * count / len(confidence_array)
        maximum_gap = max(maximum_gap, gap)
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                empirical_accuracy=accuracy,
            )
        )
    return result, float(weighted_gap), float(maximum_gap)


def threshold_curve(
    confidences: list[float],
    correct: list[bool],
    *,
    total_cost: float | None = None,
    thresholds: list[float] | None = None,
) -> list[ThresholdPoint]:
    """Computa a curva de cobertura, acurácia e taxa de revisão sob diferentes limiares.

    O provedor avalia cada linha antes do corte de confiança, logo o custo
    total incorrido não é reduzido a posteriori.

    Args:
        confidences (list[float]): Vetor de confianças preditas pelas inferências.
        correct (list[bool]): Vetor booleano de acertos empíricos observados.
        total_cost (float | None): Custo total financeiro apurado do provedor.
        thresholds (list[float] | None): Lista opcional de limiares a avaliar.

    Returns:
        list[ThresholdPoint]: Pontos calculados da curva de corte de decisão.
    """
    confidence_array = np.asarray(confidences, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    points: list[ThresholdPoint] = []
    for threshold in thresholds or [round(value, 2) for value in np.linspace(0.5, 1.0, 11)]:
        accepted = confidence_array >= threshold
        accepted_count = int(accepted.sum())
        coverage = accepted_count / len(confidence_array) if len(confidence_array) else 0.0
        accuracy = float(correct_array[accepted].mean()) if accepted_count else None
        points.append(
            ThresholdPoint(
                threshold=threshold,
                coverage=coverage,
                accuracy=accuracy,
                error_rate=(1 - accuracy) if accuracy is not None else None,
                review_rate=1 - coverage,
                estimated_cost=total_cost,
            )
        )
    return points
