from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from app.domain.exceptions import MetricEvaluationError
from app.domain.experiment.spec import MetricsSpec, TaskType
from app.ml.registries.metadata import MetricMetadata
from app.ml.registries.registry import Registry


@dataclass(frozen=True, slots=True)
class MetricEvaluation:
    """Representa o resultado da avaliação de métricas primárias e secundárias.

    Attributes:
        primary_metric (str): Identificador da métrica primária avaliada.
        values (dict[str, float]): Mapeamento dos identificadores das métricas para seus valores.
    """

    primary_metric: str
    values: dict[str, float]


class MetricEvaluator:
    """Avaliador de métricas de desempenho para modelos de regressão e classificação.

    Attributes:
        registry (Registry[MetricMetadata]): Registro de catálogo de métricas suportadas.
    """

    def __init__(self, registry: Registry[MetricMetadata]) -> None:
        """Inicializa o avaliador com o catálogo de métricas.

        Args:
            registry: Registro de metadados com as métricas disponíveis.
        """
        self.registry = registry

    def evaluate(
        self,
        *,
        task: TaskType,
        specification: MetricsSpec,
        target: Any,
        predictions: Any,
        probabilities: Any | None = None,
        probability_classes: tuple[Any, ...] | None = None,
    ) -> MetricEvaluation:
        """Calcula as métricas primárias e secundárias especificadas para as predições fornecidas.

        Args:
            task: Tipo da tarefa de machine learning (regressão, classificação binária
                ou multiclasse).
            specification: Especificação das métricas primárias e secundárias desejadas.
            target: Valores reais do conjunto de teste ou validação.
            predictions: Valores preditos pelo modelo.
            probabilities: Matriz de probabilidades preditas (necessário para métricas
                de ranking/loss).
            probability_classes: Ordem das classes correspondentes às colunas de probabilidades.

        Returns:
            MetricEvaluation contendo os valores calculados de cada métrica solicitada.

        Raises:
            MetricEvaluationError: Se alguma métrica for incompatível com a tarefa ou se faltarem
                probabilidades necessárias.
        """
        metric_ids = [specification.primary, *specification.secondary]
        for metric_id in metric_ids:
            metadata = self.registry.get(metric_id)
            if task not in metadata.tasks:
                raise MetricEvaluationError(f"Metric {metric_id} is incompatible with {task.value}")
        if task == TaskType.REGRESSION:
            values = self._regression(metric_ids, target, predictions)
        else:
            values = self._classification(
                task,
                metric_ids,
                specification,
                target,
                predictions,
                probabilities,
                probability_classes,
            )
        return MetricEvaluation(primary_metric=specification.primary, values=values)

    def aggregate(self, folds: list[MetricEvaluation]) -> tuple[dict[str, float], dict[str, float]]:
        """Consolida os resultados de múltiplos folds calculando médias e desvios padrão.

        Args:
            folds: Lista com os resultados de avaliação de cada fold da validação cruzada.

        Returns:
            Tupla de dicionários contendo (médias_por_métrica, desvios_padrão_por_métrica).

        Raises:
            MetricEvaluationError: Se a lista de folds estiver vazia ou com métricas inconsistentes.
        """
        if not folds:
            raise MetricEvaluationError("At least one fold evaluation is required")
        metric_ids = tuple(folds[0].values)
        if any(tuple(fold.values) != metric_ids for fold in folds):
            raise MetricEvaluationError("Fold evaluations must contain the same metrics")
        means = {
            metric_id: float(np.mean([fold.values[metric_id] for fold in folds]))
            for metric_id in metric_ids
        }
        standard_deviations = {
            metric_id: float(np.std([fold.values[metric_id] for fold in folds]))
            for metric_id in metric_ids
        }
        return means, standard_deviations

    @staticmethod
    def _regression(metric_ids: list[str], target: Any, predictions: Any) -> dict[str, float]:
        """Calcula métricas de regressão para os valores observados e preditos.

        Args:
            metric_ids: Lista de identificadores das métricas a computar.
            target: Valores reais do alvo numérico.
            predictions: Previsões contínuas geradas pelo modelo.

        Returns:
            Dicionário mapeando os identificadores para os valores numéricos computados.

        Raises:
            MetricEvaluationError: Se houver métrica não suportada ou divisão por zero em MAPE.
        """
        target_values = np.asarray(target, dtype=float)
        predicted_values = np.asarray(predictions, dtype=float)
        values: dict[str, float] = {}
        for metric_id in metric_ids:
            if metric_id == "mae":
                value = mean_absolute_error(target_values, predicted_values)
            elif metric_id == "mse":
                value = mean_squared_error(target_values, predicted_values)
            elif metric_id == "rmse":
                value = np.sqrt(mean_squared_error(target_values, predicted_values))
            elif metric_id == "r2":
                value = r2_score(target_values, predicted_values)
            elif metric_id == "mape":
                if np.any(target_values == 0):
                    raise MetricEvaluationError("MAPE is undefined when target contains zero")
                value = mean_absolute_percentage_error(target_values, predicted_values)
            else:
                raise MetricEvaluationError(f"Unsupported regression metric: {metric_id}")
            values[metric_id] = float(value)
        return values

    @staticmethod
    def _classification(
        task: TaskType,
        metric_ids: list[str],
        specification: MetricsSpec,
        target: Any,
        predictions: Any,
        probabilities: Any | None,
        probability_classes: tuple[Any, ...] | None,
    ) -> dict[str, float]:
        """Calcula métricas de classificação binária ou multiclasse.

        Args:
            task: Tipo de tarefa (classificação binária ou multiclasse).
            metric_ids: Identificadores das métricas solicitadas.
            specification: Especificação de métricas com averaging, positive_label e estratégias.
            target: Rótulos reais da variável alvo.
            predictions: Rótulos preditos pelo modelo.
            probabilities: Matriz de probabilidades preditas (opcional para algumas métricas).
            probability_classes: Tupla com a sequência de classes correspondentes às probabilidades.

        Returns:
            Dicionário com os valores das métricas calculadas.

        Raises:
            MetricEvaluationError: Se métricas probabilísticas não receberem probabilidades
                ou se a métrica não for suportada.
        """
        target_values = np.asarray(target)
        predicted_values = np.asarray(predictions)
        needs_probabilities = bool({"roc_auc", "pr_auc", "log_loss"}.intersection(metric_ids))
        if needs_probabilities and (probabilities is None or probability_classes is None):
            raise MetricEvaluationError(
                "Selected ranking or loss metrics require probabilities and class labels"
            )
        probability_values = np.asarray(probabilities) if probabilities is not None else None
        classes = tuple(probability_classes or ())
        average = (
            specification.averaging.value
            if task == TaskType.MULTICLASS_CLASSIFICATION and specification.averaging is not None
            else "binary"
        )
        values: dict[str, float] = {}
        for metric_id in metric_ids:
            if metric_id == "accuracy":
                value = accuracy_score(target_values, predicted_values)
            elif metric_id in {"precision", "recall", "f1"}:
                scorer = {
                    "precision": precision_score,
                    "recall": recall_score,
                    "f1": f1_score,
                }[metric_id]
                arguments: dict[str, Any] = {"average": average, "zero_division": 0}
                if average == "binary":
                    arguments["pos_label"] = specification.positive_label
                value = scorer(target_values, predicted_values, **arguments)
            elif metric_id == "roc_auc":
                assert probability_values is not None
                if task == TaskType.BINARY_CLASSIFICATION:
                    positive_index = MetricEvaluator._positive_index(
                        classes, specification.positive_label
                    )
                    binary_target = target_values == specification.positive_label
                    value = roc_auc_score(binary_target, probability_values[:, positive_index])
                else:
                    value = roc_auc_score(
                        target_values,
                        probability_values,
                        labels=list(classes),
                        multi_class=specification.roc_auc_strategy.value,
                        average=average,
                    )
            elif metric_id == "pr_auc":
                assert probability_values is not None
                if task == TaskType.BINARY_CLASSIFICATION:
                    positive_index = MetricEvaluator._positive_index(
                        classes, specification.positive_label
                    )
                    binary_target = target_values == specification.positive_label
                    value = average_precision_score(
                        binary_target, probability_values[:, positive_index]
                    )
                else:
                    binary_target = label_binarize(target_values, classes=list(classes))
                    value = average_precision_score(
                        binary_target, probability_values, average=average
                    )
            elif metric_id == "log_loss":
                assert probability_values is not None
                value = log_loss(target_values, probability_values, labels=list(classes))
            else:
                raise MetricEvaluationError(f"Unsupported classification metric: {metric_id}")
            values[metric_id] = float(value)
        return values

    @staticmethod
    def _positive_index(classes: tuple[Any, ...], positive_label: Any) -> int:
        """Determina o índice da classe positiva dentro da tupla de classes.

        Args:
            classes: Tupla com os identificadores de todas as classes.
            positive_label: Identificador do rótulo considerado positivo.

        Returns:
            Índice numérico da classe positiva.

        Raises:
            MetricEvaluationError: Se o positive_label não estiver presente na tupla de classes.
        """
        try:
            return classes.index(positive_label)
        except ValueError as error:
            raise MetricEvaluationError(
                f"Positive label {positive_label!r} is not present in probability classes"
            ) from error
