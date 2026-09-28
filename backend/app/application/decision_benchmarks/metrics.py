from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from app.domain.decision.calibration import (
    CalibrationSummary,
    reliability_bins,
    threshold_curve,
)
from app.domain.experiment.spec import AveragingStrategy


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluation:
    """Resultado consolidado da avaliação estatística e calibração de um benchmark.

    Attributes:
        metrics (dict[str, Any]): Dicionário com métricas clássicas e de calibração calculadas.
        calibration (CalibrationSummary): Sumário com histogramas, confiabilidade e curvas.
    """

    metrics: dict[str, Any]
    calibration: CalibrationSummary


def evaluate_binary(
    *,
    targets: list[str],
    positive_class: str,
    probabilities: list[float],
    confidences: list[float],
    latencies_ms: list[float],
    total_cost: float | None,
) -> BenchmarkEvaluation:
    """Calcula métricas de desempenho e calibração para problemas de classificação binária.

    Args:
        targets (list[str]): Rótulos verdadeiros observados na validação.
        positive_class (str): Identificador da classe considerada positiva.
        probabilities (list[float]): Probabilidades preditas para a classe positiva.
        confidences (list[float]): Escores de confiança associados a cada predição.
        latencies_ms (list[float]): Tempos de resposta em milissegundos por inferência.
        total_cost (float | None): Custo financeiro total acumulado na inferência.

    Returns:
        BenchmarkEvaluation: Avaliação contendo métricas calculadas e sumário de calibração.
    """
    truth = np.asarray([str(value) == positive_class for value in targets], dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    predictions = (probability >= 0.5).astype(int)
    correct = predictions == truth
    reliability, ece, mce = reliability_bins(confidences, correct.tolist())
    loss = float(log_loss(truth, np.column_stack([1 - probability, probability]), labels=[0, 1]))
    brier = float(brier_score_loss(truth, probability))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": _safe_metric(roc_auc_score, truth, probability),
        "pr_auc": _safe_metric(average_precision_score, truth, probability),
        "log_loss": loss,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "confidence_mean": float(np.mean(confidences)),
    }
    return BenchmarkEvaluation(
        metrics=metrics,
        calibration=_calibration(
            brier=brier,
            loss=loss,
            ece=ece,
            mce=mce,
            confidences=confidences,
            correct=correct.tolist(),
            latencies_ms=latencies_ms,
            total_cost=total_cost,
            reliability=reliability,
        ),
    )


def evaluate_multiclass(
    *,
    targets: list[str],
    classes: list[str],
    probabilities: list[list[float]],
    predictions: list[str],
    confidences: list[float],
    latencies_ms: list[float],
    total_cost: float | None,
    averaging: AveragingStrategy,
) -> BenchmarkEvaluation:
    """Calcula métricas de desempenho e calibração para classificação multiclasse.

    Args:
        targets (list[str]): Rótulos verdadeiros observados na validação.
        classes (list[str]): Lista ordenada de classes do problema.
        probabilities (list[list[float]]): Matriz de probabilidades preditas para cada classe.
        predictions (list[str]): Classes preditas pelo modelo ou provedor.
        confidences (list[float]): Escores de confiança das predições.
        latencies_ms (list[float]): Tempos de resposta em milissegundos por chamada.
        total_cost (float | None): Custo financeiro total incorrido.
        averaging (AveragingStrategy): Estratégia de agregação ('macro', 'weighted', etc.).

    Returns:
        BenchmarkEvaluation: Avaliação consolidada com métricas globais e por classe.
    """
    ordered_indices = sorted(range(len(classes)), key=lambda index: classes[index])
    classes = [classes[index] for index in ordered_indices]
    truth = np.asarray([str(value) for value in targets])
    probability = np.asarray(probabilities, dtype=float)[:, ordered_indices]
    predicted = np.asarray(predictions)
    correct = predicted == truth
    reliability, ece, mce = reliability_bins(confidences, correct.tolist())
    one_hot = label_binarize(truth, classes=classes)
    if len(classes) == 2 and one_hot.shape[1] == 1:
        one_hot = np.column_stack([1 - one_hot[:, 0], one_hot[:, 0]])
    brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    loss = float(log_loss(truth, probability, labels=classes))
    average = averaging.value
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, average=average, zero_division=0)),
        "recall": float(recall_score(truth, predicted, average=average, zero_division=0)),
        "f1": float(f1_score(truth, predicted, average=average, zero_division=0)),
        "averaging": average,
        "roc_auc": _safe_metric(
            roc_auc_score,
            truth,
            probability,
            labels=classes,
            multi_class="ovr",
            average=average,
        ),
        "pr_auc": _safe_metric(
            average_precision_score,
            one_hot,
            probability,
            average=average,
        ),
        "log_loss": loss,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "confidence_mean": float(np.mean(confidences)),
        "calibration_by_class": _class_calibration(one_hot, probability, classes),
    }
    return BenchmarkEvaluation(
        metrics=metrics,
        calibration=_calibration(
            brier=brier,
            loss=loss,
            ece=ece,
            mce=mce,
            confidences=confidences,
            correct=correct.tolist(),
            latencies_ms=latencies_ms,
            total_cost=total_cost,
            reliability=reliability,
        ),
    )


def _calibration(
    *,
    brier: float,
    loss: float,
    ece: float,
    mce: float,
    confidences: list[float],
    correct: list[bool],
    latencies_ms: list[float],
    total_cost: float | None,
    reliability: list[Any],
) -> CalibrationSummary:
    """Compila o sumário de calibração incluindo histogramas de confiança e curvas.

    Args:
        brier (float): Escore Brier calculado.
        loss (float): Log loss observado.
        ece (float): Erro de calibração esperado (ECE).
        mce (float): Erro de calibração máximo (MCE).
        confidences (list[float]): Vetor de confianças preditas.
        correct (list[bool]): Vetor booleano de acertos.
        latencies_ms (list[float]): Vetor de latências em ms.
        total_cost (float | None): Custo total do benchmark.
        reliability (list[Any]): Bins de confiabilidade calculados.

    Returns:
        CalibrationSummary: Sumário estruturado de calibração.
    """
    counts, edges = np.histogram(confidences, bins=10, range=(0, 1))
    histogram = [
        {"lower": float(edges[index]), "upper": float(edges[index + 1]), "count": int(count)}
        for index, count in enumerate(counts)
    ]
    return CalibrationSummary(
        brier_score=brier,
        log_loss=loss,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        confidence_mean=float(np.mean(confidences)),
        reliability=reliability,
        confidence_histogram=histogram,
        thresholds=threshold_curve(confidences, correct, total_cost=total_cost),
        latency_ms=latencies_ms,
    )


def _class_calibration(
    truth: np.ndarray[Any, Any], probability: np.ndarray[Any, Any], classes: list[str]
) -> dict[str, Any]:
    """Calcula erros de calibração e bins de confiabilidade isolados para cada classe.

    Args:
        truth (np.ndarray[Any, Any]): Matriz binária one-hot dos alvos verdadeiros.
        probability (np.ndarray[Any, Any]): Matriz de probabilidades preditas por classe.
        classes (list[str]): Nomes das classes avaliadas.

    Returns:
        dict[str, Any]: Mapeamento de nome de classe para ECE e bins de confiabilidade.
    """
    result: dict[str, Any] = {}
    for index, class_name in enumerate(classes):
        bins, ece, _ = reliability_bins(
            probability[:, index].tolist(), truth[:, index].astype(bool).tolist()
        )
        result[class_name] = {
            "expected_calibration_error": ece,
            "reliability": [item.model_dump(mode="json") for item in bins],
        }
    return result


def _safe_metric(metric: Any, *args: Any, **kwargs: Any) -> float | None:
    """Executa o cálculo de uma métrica de forma segura capturando exceções ValueError.

    Args:
        metric (Any): Função métrica do scikit-learn.
        *args (Any): Argumentos posicionais para a métrica.
        **kwargs (Any): Argumentos nomeados para a métrica.

    Returns:
        float | None: Valor numérico da métrica ou None se o cálculo falhar.
    """
    try:
        return float(metric(*args, **kwargs))
    except ValueError:
        return None
