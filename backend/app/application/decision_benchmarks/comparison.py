import io
from typing import Any

import numpy as np
import pandas as pd
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

from app.domain.decision.calibration import reliability_bins
from app.domain.decision.repositories import DecisionRunRepository
from app.domain.decision.runs import DecisionRunStatus
from app.domain.exceptions import DecisionBenchmarkError, EntityNotFoundError
from app.domain.experiment.entities import ModelRunStatus
from app.domain.experiment.repositories import ExperimentRepository, ModelRunRepository
from app.domain.experiment.spec import TaskType
from app.infrastructure.storage.base import ArtifactStorage


class DecisionComparisonService:
    """Compara modelos supervisionados e provedores zero-shot nas mesmas linhas avaliadas.

    Attributes:
        decision_runs (DecisionRunRepository): Repositório de execuções de benchmark.
        experiments (ExperimentRepository): Repositório de experimentos supervisionados.
        model_runs (ModelRunRepository): Repositório de execuções de modelos.
        storage (ArtifactStorage): Armazenamento de arquivos Parquet e artefatos de predição.
    """

    def __init__(
        self,
        *,
        decision_runs: DecisionRunRepository,
        experiments: ExperimentRepository,
        model_runs: ModelRunRepository,
        storage: ArtifactStorage,
    ) -> None:
        """Inicializa o serviço de comparação pareada entre modelos e benchmarks.

        Args:
            decision_runs (DecisionRunRepository): Repositório de benchmarks.
            experiments (ExperimentRepository): Repositório de experimentos.
            model_runs (ModelRunRepository): Repositório de execuções de modelos.
            storage (ArtifactStorage): Serviço de armazenamento de artefatos.
        """
        self.decision_runs = decision_runs
        self.experiments = experiments
        self.model_runs = model_runs
        self.storage = storage

    def compare(self, decision_run_id: str, experiment_id: str) -> dict[str, Any]:
        """Gera a comparação pareada entre uma execução de decisão e os modelos do experimento.

        Args:
            decision_run_id (str): Identificador da execução do benchmark de decisão.
            experiment_id (str): Identificador do experimento supervisionado.

        Returns:
            dict[str, Any]: Dicionário com alinhamento de linhas, métricas e deltas pareados.

        Raises:
            EntityNotFoundError: Se o benchmark ou experimento não for encontrado.
            DecisionBenchmarkError: Se o benchmark não estiver concluído, os datasets divergirem
                ou as predições forem incompatíveis.
        """
        decision_run = self.decision_runs.get(decision_run_id)
        if decision_run is None:
            raise EntityNotFoundError(f"Decision run {decision_run_id} was not found")
        if decision_run.status != DecisionRunStatus.COMPLETED:
            raise DecisionBenchmarkError("Comparison requires a completed DecisionRun")
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise EntityNotFoundError(f"Experiment {experiment_id} was not found")

        decision_spec = decision_run.specification
        experiment_spec = experiment.spec
        compatibility = {
            "same_dataset": experiment.dataset_id == decision_run.dataset_id,
            "same_target": experiment_spec.get("target") == decision_spec.target,
            "same_task": experiment_spec.get("task") == decision_spec.task.value,
        }
        if not all(compatibility.values()):
            raise DecisionBenchmarkError(
                "Comparison requires the same dataset, target, and classification task"
            )
        if decision_spec.task not in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            raise DecisionBenchmarkError("Decision comparison supports classification only")

        decision_frame = self._load_frame(f"decision/runs/{decision_run.id}/responses.parquet")
        self._require_columns(
            decision_frame,
            {"row_id", "target", "prediction", "confidence", "correct"},
            "DecisionRun",
        )
        decision_frame["row_id"] = decision_frame["row_id"].astype(str)
        self._require_unique_row_ids(decision_frame, "DecisionRun")
        classes = (
            sorted(str(item) for item in decision_spec.question.criteria)
            if decision_spec.task == TaskType.MULTICLASS_CLASSIFICATION
            else sorted(decision_frame["target"].astype(str).unique().tolist())
        )
        if len(classes) < 2 or (
            decision_spec.positive_class is not None and decision_spec.positive_class not in classes
        ):
            raise DecisionBenchmarkError("DecisionRun has an invalid target class contract")

        models: list[dict[str, Any]] = []
        unavailable: list[dict[str, str]] = []
        for model_run in self.model_runs.list_for_experiment(experiment.id):
            if model_run.status != ModelRunStatus.COMPLETED:
                continue
            predictions_path = (model_run.metrics or {}).get("predictions_path")
            if not isinstance(predictions_path, str):
                unavailable.append(
                    {
                        "run_id": model_run.id,
                        "model_id": model_run.model_id,
                        "reason": "predictions_not_persisted",
                    }
                )
                continue
            supervised_frame = self._load_frame(predictions_path)
            self._require_columns(
                supervised_frame,
                {"row_id", "target", "prediction", "confidence", "correct"},
                "ModelRun",
            )
            supervised_frame["row_id"] = supervised_frame["row_id"].astype(str)
            self._require_unique_row_ids(supervised_frame, "ModelRun")
            models.append(
                self._compare_model(
                    decision_frame=decision_frame,
                    supervised_frame=supervised_frame,
                    classes=classes,
                    positive_class=decision_spec.positive_class,
                    model_run=model_run,
                    decision_run=decision_run,
                )
            )

        return {
            "schema_version": "1.0",
            "decision_run_id": decision_run.id,
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "contract": {
                **compatibility,
                "row_alignment": "intersection_by_row_id",
                "methodology": "supervised_validation_vs_zero_shot",
                "like_for_like_training": False,
            },
            "models": models,
            "unavailable_models": unavailable,
        }

    def _compare_model(
        self,
        *,
        decision_frame: pd.DataFrame,
        supervised_frame: pd.DataFrame,
        classes: list[str],
        positive_class: str | None,
        model_run: Any,
        decision_run: Any,
    ) -> dict[str, Any]:
        """Calcula as métricas e matriz de discordância pareadas para um modelo específico.

        Args:
            decision_frame (pd.DataFrame): DataFrame com predições do benchmark de decisão.
            supervised_frame (pd.DataFrame): DataFrame com predições da execução supervisionada.
            classes (list[str]): Classes válidas do problema.
            positive_class (str | None): Identificador da classe positiva, se binário.
            model_run (Any): Entidade da execução do modelo avaliado.
            decision_run (Any): Entidade da execução do benchmark de decisão.

        Returns:
            dict[str, Any]: Estatísticas pareadas, deltas de métricas e tabela de discordância.

        Raises:
            DecisionBenchmarkError: Se não houver linhas compartilhadas ou os alvos divergirem.
        """
        shared_ids = sorted(
            set(decision_frame["row_id"]) & set(supervised_frame["row_id"]),
            key=self._row_sort_key,
        )
        if not shared_ids:
            raise DecisionBenchmarkError(
                f"Model run {model_run.id} has no evaluation rows in common with the DecisionRun"
            )
        decision = decision_frame.set_index("row_id").loc[shared_ids].copy()
        supervised = supervised_frame.set_index("row_id").loc[shared_ids].copy()
        decision_targets = decision["target"].astype(str).to_numpy()
        supervised_targets = supervised["target"].astype(str).to_numpy()
        if not np.array_equal(decision_targets, supervised_targets):
            raise DecisionBenchmarkError(
                f"Model run {model_run.id} target values differ on aligned rows"
            )
        if not set(decision_targets).issubset(classes):
            raise DecisionBenchmarkError("Aligned targets do not match the class contract")

        decision_predictions = self._normalize_binary_predictions(
            decision["prediction"].astype(str).to_numpy(),
            classes=classes,
            positive_class=positive_class,
        )
        supervised_predictions = self._normalize_supervised_predictions(
            supervised["prediction"].astype(str).to_numpy(), classes=classes
        )
        if not set(decision_predictions).issubset(classes) or not set(
            supervised_predictions
        ).issubset(classes):
            raise DecisionBenchmarkError("Aligned predictions do not match the class contract")
        decision_probabilities = self._probability_matrix(
            decision,
            classes=classes,
            positive_class=positive_class,
        )
        supervised_probabilities = self._probability_matrix(
            supervised,
            classes=classes,
            positive_class=None,
        )
        decision_confidence = decision["confidence"].astype(float).to_numpy()
        supervised_confidence = supervised["confidence"].astype(float).to_numpy()
        decision_correct = decision_predictions == decision_targets
        supervised_correct = supervised_predictions == decision_targets
        supervised_metrics = self._metrics(
            target=decision_targets,
            predictions=supervised_predictions,
            probabilities=supervised_probabilities,
            confidence=supervised_confidence,
            classes=classes,
            positive_class=positive_class,
        )
        zero_shot_metrics = self._metrics(
            target=decision_targets,
            predictions=decision_predictions,
            probabilities=decision_probabilities,
            confidence=decision_confidence,
            classes=classes,
            positive_class=positive_class,
        )

        return {
            "model_run_id": model_run.id,
            "model_id": model_run.model_id,
            "run_index": model_run.run_index,
            "sample_size": len(shared_ids),
            "classes": classes,
            "supervised": {
                "metrics": supervised_metrics,
                "training_duration_seconds": model_run.duration_seconds,
                "latency_p95_ms": None,
                "estimated_cost": None,
            },
            "zero_shot": {
                "metrics": zero_shot_metrics,
                **self._decision_operations(decision, decision_run),
            },
            "metric_deltas": {
                name: (
                    None
                    if supervised_metrics[name] is None or zero_shot_metrics[name] is None
                    else supervised_metrics[name] - zero_shot_metrics[name]
                )
                for name in supervised_metrics
            },
            "disagreement": {
                "both_correct": int(np.sum(supervised_correct & decision_correct)),
                "supervised_only_correct": int(np.sum(supervised_correct & ~decision_correct)),
                "zero_shot_only_correct": int(np.sum(~supervised_correct & decision_correct)),
                "both_wrong": int(np.sum(~supervised_correct & ~decision_correct)),
                "prediction_disagreements": int(
                    np.sum(supervised_predictions != decision_predictions)
                ),
            },
        }

    @staticmethod
    def _metrics(
        *,
        target: np.ndarray[Any, Any],
        predictions: np.ndarray[Any, Any],
        probabilities: np.ndarray[Any, Any],
        confidence: np.ndarray[Any, Any],
        classes: list[str],
        positive_class: str | None,
    ) -> dict[str, float | None]:
        """Calcula métricas padronizadas para um subconjunto pareado de predições.

        Args:
            target (np.ndarray[Any, Any]): Vetor de classes verdadeiras.
            predictions (np.ndarray[Any, Any]): Vetor de predições nominais.
            probabilities (np.ndarray[Any, Any]): Matriz de probabilidades preditas.
            confidence (np.ndarray[Any, Any]): Vetor de confiança de cada predição.
            classes (list[str]): Lista ordenada de classes do problema.
            positive_class (str | None): Classe positiva em problemas binários.

        Returns:
            dict[str, float | None]: Dicionário contendo métricas de classificação e calibração.
        """
        correct = predictions == target
        _, ece, _ = reliability_bins(confidence.tolist(), correct.tolist())
        result: dict[str, float | None] = {
            "accuracy": float(accuracy_score(target, predictions)),
            "precision": float(
                precision_score(target, predictions, average="macro", zero_division=0)
            ),
            "recall": float(recall_score(target, predictions, average="macro", zero_division=0)),
            "f1": float(f1_score(target, predictions, average="macro", zero_division=0)),
            "log_loss": DecisionComparisonService._safe_metric(
                log_loss, target, probabilities, labels=classes
            ),
            "expected_calibration_error": ece,
        }
        if len(classes) == 2:
            positive = positive_class or classes[1]
            positive_index = classes.index(positive)
            truth = target == positive
            positive_probability = probabilities[:, positive_index]
            result.update(
                brier_score=DecisionComparisonService._safe_metric(
                    brier_score_loss, truth.astype(int), positive_probability
                ),
                roc_auc=DecisionComparisonService._safe_metric(
                    roc_auc_score, truth.astype(int), positive_probability
                ),
                pr_auc=DecisionComparisonService._safe_metric(
                    average_precision_score, truth.astype(int), positive_probability
                ),
            )
        else:
            one_hot = label_binarize(target, classes=classes)
            result.update(
                brier_score=float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
                roc_auc=DecisionComparisonService._safe_metric(
                    roc_auc_score,
                    target,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                ),
                pr_auc=DecisionComparisonService._safe_metric(
                    average_precision_score,
                    one_hot,
                    probabilities,
                    average="macro",
                ),
            )
        return result

    @staticmethod
    def _decision_operations(frame: pd.DataFrame, run: Any) -> dict[str, Any]:
        """Extrai métricas operacionais da execução do benchmark (latência p95, custos, cache).

        Args:
            frame (pd.DataFrame): DataFrame com respostas e tempos do benchmark.
            run (Any): Entidade da execução contendo custo total acumulado.

        Returns:
            dict[str, Any]: Métricas operacionais consolidadas.
        """
        if not {"cache_hit", "request_count", "latency_ms"}.issubset(frame.columns):
            return {
                "latency_p95_ms": None,
                "estimated_cost": None,
                "request_count": None,
                "cache_hits": None,
            }
        cache_hit = frame["cache_hit"].astype(bool)
        provider_latency = frame.loc[~cache_hit, "latency_ms"].astype(float).to_numpy()
        costs = frame["estimated_cost"] if "estimated_cost" in frame else None
        estimated_cost = None
        if costs is not None and not costs.isna().any():
            estimated_cost = float(costs.astype(float).sum())
        return {
            "latency_p95_ms": (
                float(np.percentile(provider_latency, 95)) if len(provider_latency) else None
            ),
            "estimated_cost": estimated_cost,
            "request_count": int(frame["request_count"].sum()),
            "cache_hits": int(cache_hit.sum()),
            "full_run_estimated_cost": run.estimated_cost,
        }

    @staticmethod
    def _probability_matrix(
        frame: pd.DataFrame,
        *,
        classes: list[str],
        positive_class: str | None,
    ) -> np.ndarray[Any, Any]:
        """Constrói e valida a matriz 2D de probabilidades normalizadas para as classes.

        Args:
            frame (pd.DataFrame): DataFrame contendo coluna probabilities ou probability_yes.
            classes (list[str]): Classes esperadas na ordem das colunas da matriz.
            positive_class (str | None): Nome da classe positiva em casos binários simplificados.

        Returns:
            np.ndarray[Any, Any]: Matriz de probabilidades com soma 1.0 por linha.

        Raises:
            DecisionBenchmarkError: Se as probabilidades faltarem ou forem inconsistentes.
        """
        try:
            if "probabilities" in frame and frame["probabilities"].notna().all():
                matrix = np.asarray(
                    [
                        [float(probabilities[class_name]) for class_name in classes]
                        for probabilities in frame["probabilities"]
                    ],
                    dtype=float,
                )
            elif "probability_yes" in frame and positive_class is not None:
                positive_index = classes.index(positive_class)
                values = frame["probability_yes"].astype(float).to_numpy()
                matrix = np.column_stack([1 - values, values])
                if positive_index == 0:
                    matrix = matrix[:, ::-1]
            else:
                raise DecisionBenchmarkError("Comparison artifact lacks class probabilities")
        except (KeyError, TypeError, ValueError) as error:
            raise DecisionBenchmarkError("Comparison probabilities are malformed") from error
        if (
            matrix.shape != (len(frame), len(classes))
            or not np.isfinite(matrix).all()
            or (matrix < 0).any()
            or (matrix > 1).any()
            or not np.allclose(matrix.sum(axis=1), 1.0)
        ):
            raise DecisionBenchmarkError("Comparison probabilities are malformed")
        return matrix

    @staticmethod
    def _normalize_binary_predictions(
        predictions: np.ndarray[Any, Any],
        *,
        classes: list[str],
        positive_class: str | None,
    ) -> np.ndarray[Any, Any]:
        """Mapeia predições binárias com prefixo 'not:' para a classe negativa real.

        Args:
            predictions (np.ndarray[Any, Any]): Array de predições nominais.
            classes (list[str]): As duas classes do problema binário.
            positive_class (str | None): Identificador da classe positiva.

        Returns:
            np.ndarray[Any, Any]: Predições normalizadas para os nomes exatos das classes.
        """
        if positive_class is None or len(classes) != 2:
            return predictions
        negative_class = next(item for item in classes if item != positive_class)
        return np.asarray(
            [negative_class if value == f"not:{positive_class}" else value for value in predictions]
        )

    @staticmethod
    def _normalize_supervised_predictions(
        predictions: np.ndarray[Any, Any], *, classes: list[str]
    ) -> np.ndarray[Any, Any]:
        """Normaliza predições legadas serializadas como arrays singleton pelo CatBoost.

        Args:
            predictions (np.ndarray[Any, Any]): Array de predições supervisionadas.
            classes (list[str]): Conjunto de classes válidas esperadas.

        Returns:
            np.ndarray[Any, Any]: Predições normalizadas contra as classes válidas.
        """
        # Historical CatBoost artifacts persisted singleton arrays using str(array).
        # Match only representations of known classes; never parse arbitrary text.
        legacy_labels: dict[str, str | None] = {}
        for label in classes:
            representations = {str(np.asarray([label]))}
            for convert in (int, float):
                try:
                    numeric_label = convert(label)
                except ValueError:
                    continue
                if str(numeric_label) == label and (convert is int or np.isfinite(numeric_label)):
                    representations.add(str(np.asarray([numeric_label])))
            for representation in representations:
                if representation not in classes:
                    if representation not in legacy_labels:
                        legacy_labels[representation] = label
                    elif legacy_labels[representation] != label:
                        legacy_labels[representation] = None
        return np.asarray(
            [
                value if value in classes else legacy_labels.get(value) or value
                for value in predictions
            ]
        )

    def _load_frame(self, artifact_path: str) -> pd.DataFrame:
        """Carrega arquivo Parquet a partir do serviço de armazenamento de artefatos.

        Args:
            artifact_path (str): Caminho relativo do artefato no storage.

        Returns:
            pd.DataFrame: DataFrame pandas carregado a partir do Parquet.

        Raises:
            DecisionBenchmarkError: Se o arquivo não existir ou estiver corrompido.
        """
        try:
            return pd.read_parquet(io.BytesIO(self.storage.load(artifact_path)))
        except (FileNotFoundError, OSError, ValueError) as error:
            raise DecisionBenchmarkError(
                f"Comparison artifact {artifact_path} is unavailable"
            ) from error

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
        """Verifica se todas as colunas obrigatórias estão presentes no DataFrame.

        Args:
            frame (pd.DataFrame): DataFrame a ser verificado.
            columns (set[str]): Conjunto de colunas requeridas.
            source (str): Identificador da fonte para detalhamento em caso de erro.

        Raises:
            DecisionBenchmarkError: Se faltar qualquer coluna exigida.
        """
        missing = columns - set(frame.columns)
        if missing:
            raise DecisionBenchmarkError(
                f"{source} comparison artifact is missing columns: {sorted(missing)}"
            )

    @staticmethod
    def _require_unique_row_ids(frame: pd.DataFrame, source: str) -> None:
        """Garante a unicidade dos identificadores de linha para alinhamento pareado 1:1.

        Args:
            frame (pd.DataFrame): DataFrame contendo coluna 'row_id'.
            source (str): Identificador da fonte para mensagem de erro.

        Raises:
            DecisionBenchmarkError: Se houver linhas duplicadas.
        """
        if frame["row_id"].duplicated().any():
            raise DecisionBenchmarkError(f"{source} comparison artifact contains duplicate row IDs")

    @staticmethod
    def _row_sort_key(value: str) -> tuple[int, int | str]:
        """Gera chave de ordenação natural para identificadores numéricos ou alfanuméricos.

        Args:
            value (str): Identificador da linha.

        Returns:
            tuple[int, int | str]: Chave de ordenação estável.
        """
        return (0, int(value)) if value.isdigit() else (1, value)

    @staticmethod
    def _safe_metric(metric: Any, *args: Any, **kwargs: Any) -> float | None:
        """Executa função de métrica capturando eventuais exceções ValueError.

        Args:
            metric (Any): Função métrica do scikit-learn.
            *args (Any): Argumentos posicionais.
            **kwargs (Any): Argumentos nomeados.

        Returns:
            float | None: Valor numérico ou None em caso de erro.
        """
        try:
            return float(metric(*args, **kwargs))
        except ValueError:
            return None
