import asyncio
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from app.application.datasets.errors import DatasetIngestionError
from app.application.datasets.loading import load_verified_csv
from app.application.decision_benchmarks.metrics import (
    BenchmarkEvaluation,
    evaluate_binary,
    evaluate_multiclass,
)
from app.application.decisions.registry import (
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
)
from app.domain.common import canonical_json
from app.domain.dataset.repositories import DatasetRepository
from app.domain.decision.provider import DecisionProvider
from app.domain.decision.repositories import (
    CalibrationRunRepository,
    DecisionCacheRepository,
    DecisionRunRepository,
)
from app.domain.decision.responses import (
    ChoiceDecisionAnswer,
    DecisionResponse,
    NoulDecisionAnswer,
)
from app.domain.decision.runs import CalibrationRun, DecisionRunStatus
from app.domain.decision.specifications import DecisionState
from app.domain.exceptions import DecisionBenchmarkError, EntityNotFoundError
from app.domain.experiment.spec import TaskType
from app.infrastructure.storage.base import ArtifactStorage


class DecisionExecutionService:
    """Orquestra a execução de benchmarks de decisão contra provedores LLM externos.

    Attributes:
        datasets (DatasetRepository): Repositório para consulta de datasets.
        runs (DecisionRunRepository): Repositório para acompanhamento do benchmark.
        calibrations (CalibrationRunRepository): Repositório para resultados de calibração.
        cache (DecisionCacheRepository): Cache de respostas de provedores.
        dataset_storage (ArtifactStorage): Armazenamento de arquivos de dados brutos.
        artifact_storage (ArtifactStorage): Armazenamento de predições e métricas.
        providers (DecisionProviderRegistry): Registro de fábricas de provedores.
        provider_config (DecisionProviderRuntimeConfig): Configuração ativa de execução.
        chunk_size (int): Quantidade de linhas processadas por lote assíncrono.
    """

    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        runs: DecisionRunRepository,
        calibrations: CalibrationRunRepository,
        cache: DecisionCacheRepository,
        dataset_storage: ArtifactStorage,
        artifact_storage: ArtifactStorage,
        providers: DecisionProviderRegistry,
        provider_config: DecisionProviderRuntimeConfig,
        chunk_size: int = 50,
    ) -> None:
        """Inicializa o serviço de execução de benchmarks com suas dependências.

        Args:
            datasets (DatasetRepository): Repositório de datasets.
            runs (DecisionRunRepository): Repositório de execuções de decisão.
            calibrations (CalibrationRunRepository): Repositório de calibrações.
            cache (DecisionCacheRepository): Repositório de cache de respostas.
            dataset_storage (ArtifactStorage): Storage para leitura do CSV.
            artifact_storage (ArtifactStorage): Storage para escrita de artefatos.
            providers (DecisionProviderRegistry): Registro de provedores de IA.
            provider_config (DecisionProviderRuntimeConfig): Configuração de runtime do provedor.
            chunk_size (int): Tamanho do lote de linhas por iteração assíncrona.
        """
        self.datasets = datasets
        self.runs = runs
        self.calibrations = calibrations
        self.cache = cache
        self.dataset_storage = dataset_storage
        self.artifact_storage = artifact_storage
        self.providers = providers
        self.provider_config = provider_config
        self.chunk_size = chunk_size

    def execute(
        self,
        run_id: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Executa o benchmark de decisão avaliando as linhas do dataset via provedor.

        Args:
            run_id (str): Identificador da execução do benchmark de decisão.
            on_event (Callable[[dict[str, Any]], None] | None): Callback para emissão de eventos.

        Returns:
            dict[str, Any]: Metadados do resultado com caminhos dos artefatos persistidos.

        Raises:
            EntityNotFoundError: Se o benchmark ou o dataset associado não for encontrado.
            DecisionBenchmarkError: Se a integridade do CSV falhar ou o contrato for violado.
        """
        run = self.runs.get(run_id)
        if run is None:
            raise EntityNotFoundError(f"Decision run {run_id} was not found")
        dataset = self.datasets.get(run.dataset_id)
        if dataset is None:
            raise EntityNotFoundError(f"Dataset {run.dataset_id} was not found")
        specification = run.specification
        self.runs.transition(run_id, DecisionRunStatus.RUNNING)
        try:
            try:
                _, frame = load_verified_csv(dataset, self.dataset_storage)
            except DatasetIngestionError as error:
                raise DecisionBenchmarkError(str(error)) from error
            self._validate_frame_contract(frame, specification)
            provider_config = self.provider_config.model_copy(
                update={"provider_id": specification.provider, "model": specification.model}
            )
            provider = self.providers.create(specification.provider, provider_config)
            self._emit(on_event, "decision.started", run_id, 0, 0, len(frame))
            rows = asyncio.run(
                self._evaluate_rows(
                    run_id=run_id,
                    dataset_hash=dataset.content_hash,
                    frame=frame,
                    provider=provider,
                    on_event=on_event,
                )
            )
            return self._complete(run_id, rows, on_event)
        except Exception as error:
            self.runs.transition(
                run_id,
                DecisionRunStatus.FAILED,
                error_message=f"{type(error).__name__}: {error}",
            )
            raise

    @staticmethod
    def _validate_frame_contract(frame: pd.DataFrame, specification: Any) -> None:
        """Valida se o DataFrame atende a todas as colunas e classes requeridas pela especificação.

        Args:
            frame (pd.DataFrame): DataFrame com dados do dataset carregado.
            specification (Any): Especificação da tarefa de benchmark de decisão.

        Raises:
            DecisionBenchmarkError: Se houver colunas ausentes, valores nulos no alvo ou
                classes inválidas.
        """
        required = {specification.target, *specification.input_columns}
        missing = required - set(frame.columns)
        if missing:
            raise DecisionBenchmarkError(f"Dataset is missing columns: {sorted(missing)}")
        if frame[specification.target].isna().any():
            raise DecisionBenchmarkError("Target contains missing values")
        classes = {str(value) for value in frame[specification.target].unique()}
        if specification.task == TaskType.BINARY_CLASSIFICATION:
            if len(classes) != 2 or specification.positive_class not in classes:
                raise DecisionBenchmarkError("Binary target classes do not match the spec")
        elif classes != set(specification.question.criteria):
            raise DecisionBenchmarkError("Choice criteria do not match target classes")

    async def _evaluate_rows(
        self,
        *,
        run_id: str,
        dataset_hash: str,
        frame: pd.DataFrame,
        provider: DecisionProvider,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        """Avalia todas as linhas do dataset em chunks assíncronos respeitando a concorrência.

        Args:
            run_id (str): Identificador da execução em processamento.
            dataset_hash (str): Hash criptográfico do dataset para chave de cache.
            frame (pd.DataFrame): Dados tabulares completos a avaliar.
            provider (DecisionProvider): Instância conectada do provedor de decisão.
            on_event (Callable[[dict[str, Any]], None] | None): Callback de progresso.

        Returns:
            list[dict[str, Any]]: Lista de dicionários com as predições e metadados de cada linha.

        Raises:
            DecisionBenchmarkError: Se o provedor resolver modelos heterogêneos durante a execução.
        """
        run = self.runs.get(run_id)
        assert run is not None
        semaphore = asyncio.Semaphore(self.provider_config.concurrency)
        results: list[dict[str, Any]] = []
        request_count = 0
        cache_hits = 0
        input_tokens: int | None = 0
        estimated_cost: float | None = 0.0
        latency_total = 0.0
        for start in range(0, len(frame), self.chunk_size):
            chunk = frame.iloc[start : start + self.chunk_size]
            evaluated = await asyncio.gather(
                *[
                    self._evaluate_row(
                        dataset_hash=dataset_hash,
                        row_index=(
                            int(index) if isinstance(index, (int, np.integer)) else str(index)
                        ),
                        row=row,
                        run=run,
                        provider=provider,
                        semaphore=semaphore,
                    )
                    for index, row in chunk.iterrows()
                ]
            )
            results.extend(evaluated)
            for item in evaluated:
                usage = item["_usage"]
                request_count += usage["request_count"]
                cache_hits += int(usage["cache_hit"])
                if usage["input_tokens"] is None:
                    input_tokens = None
                elif input_tokens is not None:
                    input_tokens += usage["input_tokens"]
                if usage["estimated_cost"] is None:
                    estimated_cost = None
                elif estimated_cost is not None:
                    estimated_cost += usage["estimated_cost"]
                latency_total += item["latency_ms"]
            processed = len(results)
            self.runs.checkpoint(
                run_id,
                processed_rows=processed,
                request_count=request_count,
                cache_hits=cache_hits,
                input_tokens=input_tokens,
                estimated_cost=estimated_cost,
                latency_ms_total=latency_total,
                resolved_model=results[-1]["resolved_model"],
            )
            self._emit(
                on_event,
                "decision.progress",
                run_id,
                processed * 100 / len(frame),
                processed,
                len(frame),
            )
            if cache_hits:
                self._emit(
                    on_event,
                    "decision.cache_hit",
                    run_id,
                    processed * 100 / len(frame),
                    processed,
                    len(frame),
                    count=cache_hits,
                )
        resolved_models = {item["resolved_model"] for item in results}
        if len(resolved_models) != 1:
            raise DecisionBenchmarkError("Provider resolved multiple model versions in one run")
        return results

    async def _evaluate_row(
        self,
        *,
        dataset_hash: str,
        row_index: int | str,
        row: pd.Series,
        run: Any,
        provider: DecisionProvider,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """Processa a inferência para uma única linha com consulta e persistência em cache.

        Args:
            dataset_hash (str): Hash SHA-256 do dataset de origem.
            row_index (int | str): Identificador do índice da linha.
            row (pd.Series): Série com os valores de entrada da linha.
            run (Any): Entidade da execução do benchmark contendo a especificação.
            provider (DecisionProvider): Cliente de inferência do provedor configurado.
            semaphore (asyncio.Semaphore): Semáforo limitador de chamadas concorrentes.

        Returns:
            dict[str, Any]: Dicionário com predição, confiança, tempos de resposta e uso.

        Raises:
            DecisionBenchmarkError: Se o provedor retornar resposta com tipo de resposta inválido.
        """
        specification = run.specification
        values = {column: self._scalar(row[column]) for column in specification.input_columns}
        row_hash = self._hash(values)
        context = {
            "dataset_hash": dataset_hash,
            "row_hash": row_hash,
            "question_spec_hash": run.question_spec_hash,
            "provider_id": provider.id,
            "requested_model": specification.model,
        }
        context_key = self._hash(context)
        cached = self.cache.get(context_key) if self.provider_config.cache_enabled else None
        if cached is not None:
            response = DecisionResponse.model_validate(cached)
            response = response.model_copy(
                update={
                    "usage": response.usage.model_copy(
                        update={
                            "request_count": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "estimated_cost": 0.0,
                            "cache_hit": True,
                        }
                    ),
                    "latency_ms": 0.0,
                }
            )
        else:
            async with semaphore:
                response = await provider.evaluate(
                    DecisionState(row_id=str(row_index), values=values),
                    {"prediction": specification.question},
                )
            if self.provider_config.cache_enabled:
                resolved_model = response.resolved_model or response.requested_model
                cache_key = self._hash({**context, "resolved_model": resolved_model})
                self.cache.put(
                    context_key=context_key,
                    cache_key=cache_key,
                    dataset_hash=dataset_hash,
                    row_hash=row_hash,
                    question_spec_hash=run.question_spec_hash,
                    provider_id=provider.id,
                    requested_model=specification.model,
                    resolved_model=resolved_model,
                    response=response.model_dump(mode="json"),
                )
        answer = response.answers["prediction"]
        result: dict[str, Any] = {
            "row_id": str(row_index),
            "target": self._scalar(row[specification.target]),
            "confidence": answer.confidence,
            "latency_ms": response.latency_ms or 0.0,
            "resolved_model": response.resolved_model or response.requested_model,
            "_usage": response.usage.model_dump(mode="json"),
        }
        if isinstance(answer, NoulDecisionAnswer):
            predicted_positive = answer.probability_yes >= 0.5
            actual_positive = str(result["target"]) == specification.positive_class
            result.update(
                probability_yes=answer.probability_yes,
                prediction=(
                    specification.positive_class
                    if answer.probability_yes >= 0.5
                    else f"not:{specification.positive_class}"
                ),
                correct=predicted_positive == actual_positive,
            )
        elif isinstance(answer, ChoiceDecisionAnswer):
            result.update(
                prediction=answer.selected_choice,
                probabilities=answer.choice_probabilities,
                correct=answer.selected_choice == str(result["target"]),
            )
        else:
            raise DecisionBenchmarkError("Benchmark received an unsupported answer type")
        return result

    def _complete(
        self,
        run_id: str,
        rows: list[dict[str, Any]],
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        """Consolida métricas finais, calibração e persiste artefatos no storage.

        Args:
            run_id (str): Identificador da execução concluída.
            rows (list[dict[str, Any]]): Lista de predições coletadas de todas as linhas.
            on_event (Callable[[dict[str, Any]], None] | None): Callback para evento final.

        Returns:
            dict[str, Any]: Mapeamento contendo identificadores e caminhos dos artefatos.
        """
        run = self.runs.get(run_id)
        assert run is not None
        specification = run.specification
        costs = [item["_usage"]["estimated_cost"] for item in rows]
        total_cost = sum(costs) if all(cost is not None for cost in costs) else None
        provider_latencies = [
            float(item["latency_ms"]) for item in rows if not item["_usage"]["cache_hit"]
        ]
        if specification.task == TaskType.BINARY_CLASSIFICATION:
            evaluation = evaluate_binary(
                targets=[str(item["target"]) for item in rows],
                positive_class=specification.positive_class or "",
                probabilities=[float(item["probability_yes"]) for item in rows],
                confidences=[float(item["confidence"]) for item in rows],
                latencies_ms=provider_latencies,
                total_cost=total_cost,
            )
        else:
            classes = list(specification.question.criteria)
            evaluation = evaluate_multiclass(
                targets=[str(item["target"]) for item in rows],
                classes=classes,
                probabilities=[
                    [float(item["probabilities"][class_name]) for class_name in classes]
                    for item in rows
                ],
                predictions=[str(item["prediction"]) for item in rows],
                confidences=[float(item["confidence"]) for item in rows],
                latencies_ms=provider_latencies,
                total_cost=total_cost,
                averaging=specification.averaging,
            )
        metrics = self._summarize(evaluation, rows, total_cost)
        prefix = f"decision/runs/{run_id}"
        self._save_json(f"{prefix}/decision_spec.json", specification.model_dump(mode="json"))
        response_rows = []
        for item in rows:
            usage = item["_usage"]
            response_rows.append(
                {
                    **{key: value for key, value in item.items() if key != "_usage"},
                    "request_count": usage["request_count"],
                    "cache_hit": usage["cache_hit"],
                    "input_tokens": usage["input_tokens"],
                    "estimated_cost": usage["estimated_cost"],
                }
            )
        response_frame = pd.DataFrame(response_rows)
        parquet = io.BytesIO()
        response_frame.to_parquet(parquet, index=False)
        response_path = self.artifact_storage.save(
            f"{prefix}/responses.parquet", parquet.getvalue()
        )
        summary_path = self._save_json(f"{prefix}/summary.json", metrics)
        self._save_json(
            f"{prefix}/usage.json",
            {
                "request_count": sum(item["_usage"]["request_count"] for item in rows),
                "cache_hits": sum(int(item["_usage"]["cache_hit"]) for item in rows),
                "input_tokens": self._optional_sum(
                    [item["_usage"]["input_tokens"] for item in rows]
                ),
                "estimated_cost": total_cost,
            },
        )
        calibration_prefix = f"decision/calibration/{run_id}"
        calibration_payload = evaluation.calibration.model_dump(mode="json")
        calibration_path = self._save_json(
            f"{calibration_prefix}/metrics.json", calibration_payload
        )
        self._save_json(
            f"{calibration_prefix}/reliability.json",
            {"bins": calibration_payload["reliability"]},
        )
        calibration = self.calibrations.add(
            CalibrationRun.create(
                decision_run_id=run_id,
                task=specification.task.value,
                metrics=calibration_payload,
                artifact_path=calibration_path,
            )
        )
        resolved_model = rows[0]["resolved_model"]
        self.runs.transition(
            run_id,
            DecisionRunStatus.COMPLETED,
            processed_rows=len(rows),
            resolved_model=resolved_model,
            request_count=sum(item["_usage"]["request_count"] for item in rows),
            cache_hits=sum(int(item["_usage"]["cache_hit"]) for item in rows),
            input_tokens=self._optional_sum([item["_usage"]["input_tokens"] for item in rows]),
            estimated_cost=total_cost,
            latency_ms_total=sum(provider_latencies),
            artifact_path=summary_path,
            calibration_run_id=calibration.id,
        )
        self._emit(on_event, "decision.completed", run_id, 100, len(rows), len(rows))
        return {
            "decision_run_id": run_id,
            "responses_path": response_path,
            "summary_path": summary_path,
            "calibration_run_id": calibration.id,
        }

    @staticmethod
    def _summarize(
        evaluation: BenchmarkEvaluation,
        rows: list[dict[str, Any]],
        total_cost: float | None,
    ) -> dict[str, Any]:
        """Gera o resumo estatístico consolidando métricas, latências e custos.

        Args:
            evaluation (BenchmarkEvaluation): Avaliação estatística calculada.
            rows (list[dict[str, Any]]): Linhas processadas com registros de uso.
            total_cost (float | None): Custo financeiro estimado acumulado.

        Returns:
            dict[str, Any]: Resumo completo das métricas operacionais e avaliativas.
        """
        latencies = np.asarray(
            [item["latency_ms"] for item in rows if not item["_usage"]["cache_hit"]],
            dtype=float,
        )
        return {
            **evaluation.metrics,
            "latency_p50_ms": (float(np.percentile(latencies, 50)) if len(latencies) else None),
            "latency_p95_ms": (float(np.percentile(latencies, 95)) if len(latencies) else None),
            "request_count": sum(item["_usage"]["request_count"] for item in rows),
            "cache_hits": sum(int(item["_usage"]["cache_hit"]) for item in rows),
            "input_tokens": DecisionExecutionService._optional_sum(
                [item["_usage"]["input_tokens"] for item in rows]
            ),
            "estimated_cost": total_cost,
        }

    def _save_json(self, artifact_id: str, payload: Mapping[str, Any]) -> str:
        """Serializa e salva um payload JSON canônico no storage de artefatos.

        Args:
            artifact_id (str): Identificador relativo do artefato a salvar.
            payload (Mapping[str, Any]): Dicionário com dados a serem persistidos.

        Returns:
            str: Caminho do artefato gerado pelo storage.
        """
        return self.artifact_storage.save(
            artifact_id,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(),
        )

    @staticmethod
    def _hash(payload: Mapping[str, Any]) -> str:
        """Calcula o hash criptográfico SHA-256 canônico de um payload estruturado.

        Args:
            payload (Mapping[str, Any]): Dicionário a ser hasheado.

        Returns:
            str: Digest hexadecimal de 64 caracteres.
        """
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    @staticmethod
    def _scalar(value: Any) -> str | int | float | bool | None:
        """Converte valores escalares do pandas/numpy em tipos primitivos do Python.

        Args:
            value (Any): Valor bruto vindo do DataFrame ou Series.

        Returns:
            str | int | float | bool | None: Valor sanitizado como primitivo Python ou None.
        """
        if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _optional_sum(values: list[int | float | None]) -> int | float | None:
        """Soma valores numéricos somente se todos os elementos forem válidos (não None).

        Args:
            values (list[int | float | None]): Lista de valores a serem somados.

        Returns:
            int | float | None: Soma total ou None se houver elementos ausentes.
        """
        return (
            sum(value for value in values if value is not None)
            if all(value is not None for value in values)
            else None
        )

    @staticmethod
    def _emit(
        callback: Callable[[dict[str, Any]], None] | None,
        event_type: str,
        run_id: str,
        progress: float,
        processed: int,
        total: int,
        **extra: Any,
    ) -> None:
        """Dispara callback de progresso ou eventos de ciclo de vida se registrado.

        Args:
            callback (Callable[[dict[str, Any]], None] | None): Função de callback.
            event_type (str): Identificador do tipo de evento emitido.
            run_id (str): Identificador da execução do benchmark.
            progress (float): Percentual de conclusão entre 0 e 100.
            processed (int): Quantidade de linhas processadas até o momento.
            total (int): Quantidade total de linhas a serem processadas.
            **extra (Any): Atributos extras a anexar ao payload do evento.
        """
        if callback is not None:
            callback(
                {
                    "type": event_type,
                    "decision_run_id": run_id,
                    "processed_rows": processed,
                    "total_rows": total,
                    "progress": round(progress, 2),
                    "message_key": event_type,
                    **extra,
                }
            )
