import asyncio
import hashlib
import io
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from app.application.datasets.errors import DatasetIngestionError
from app.application.datasets.loading import load_verified_csv
from app.application.decisions.registry import (
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
)
from app.domain.common import canonical_json
from app.domain.dataset.repositories import DatasetRepository
from app.domain.decision.provider import DecisionProvider
from app.domain.decision.repositories import (
    DecisionCacheRepository,
    SemanticFeatureSetRepository,
)
from app.domain.decision.responses import (
    ChoiceDecisionAnswer,
    DecisionResponse,
    NoulDecisionAnswer,
    ScoreDecisionAnswer,
)
from app.domain.decision.runs import DecisionRunStatus
from app.domain.decision.semantic_features import SemanticFeatureSet, SemanticQuestionSpec
from app.domain.decision.specifications import DecisionSpec, DecisionState
from app.domain.exceptions import DecisionBenchmarkError, EntityNotFoundError
from app.infrastructure.storage.base import ArtifactStorage


class SemanticFeatureExecutionService:
    """Executa a geração e avaliação de features semânticas sobre datasets tabulares.

    Attributes:
        datasets (DatasetRepository): Repositório para acesso a dados dos datasets.
        feature_sets (SemanticFeatureSetRepository): Repositório de conjuntos de features.
        cache (DecisionCacheRepository): Repositório de cache de respostas LLM.
        dataset_storage (ArtifactStorage): Storage para leitura de arquivos CSV originais.
        artifact_storage (ArtifactStorage): Storage para persistência de artefatos gerados.
        providers (DecisionProviderRegistry): Registro de provedores de inferência de decisão.
        provider_config (DecisionProviderRuntimeConfig): Configurações de execução e concorrência.
        impact_evaluator (Any | None): Avaliador opcional de impacto de features semânticas.
        chunk_size (int): Tamanho do lote de linhas processadas por iteração assíncrona.
    """

    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        feature_sets: SemanticFeatureSetRepository,
        cache: DecisionCacheRepository,
        dataset_storage: ArtifactStorage,
        artifact_storage: ArtifactStorage,
        providers: DecisionProviderRegistry,
        provider_config: DecisionProviderRuntimeConfig,
        impact_evaluator: Any | None = None,
        chunk_size: int = 50,
    ) -> None:
        """Inicializa o serviço de execução de features semânticas.

        Args:
            datasets (DatasetRepository): Repositório de datasets.
            feature_sets (SemanticFeatureSetRepository): Repositório de conjuntos semânticos.
            cache (DecisionCacheRepository): Repositório de cache de inferência.
            dataset_storage (ArtifactStorage): Armazenamento de arquivos de dataset.
            artifact_storage (ArtifactStorage): Armazenamento de artefatos e relatórios.
            providers (DecisionProviderRegistry): Catálogo de provedores de decisão.
            provider_config (DecisionProviderRuntimeConfig): Configuração de runtime dos provedores.
            impact_evaluator (Any | None): Avaliador de impacto ML opcional.
            chunk_size (int): Quantidade de linhas por lote de processamento.
        """
        self.datasets = datasets
        self.feature_sets = feature_sets
        self.cache = cache
        self.dataset_storage = dataset_storage
        self.artifact_storage = artifact_storage
        self.providers = providers
        self.provider_config = provider_config
        self.impact_evaluator = impact_evaluator
        self.chunk_size = chunk_size

    def execute(
        self,
        feature_set_id: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Executa a extração assíncrona de features semânticas e salva os resultados.

        Args:
            feature_set_id (str): Identificador do conjunto de features semânticas.
            on_event (Callable[[dict[str, Any]], None] | None): Callback opcional para progresso.

        Returns:
            dict[str, Any]: Informações de conclusão com ID e caminho do arquivo Parquet.

        Raises:
            EntityNotFoundError: Se o conjunto semântico ou o dataset não forem encontrados.
            DecisionBenchmarkError: Se houver falha de validação ou erro de ingestão.
        """
        feature_set = self.feature_sets.get(feature_set_id)
        if feature_set is None:
            raise EntityNotFoundError(f"Semantic feature set {feature_set_id} was not found")
        dataset = self.datasets.get(feature_set.dataset_id)
        if dataset is None:
            raise EntityNotFoundError(f"Dataset {feature_set.dataset_id} was not found")
        self.feature_sets.transition(feature_set_id, DecisionRunStatus.RUNNING)
        try:
            try:
                _, frame = load_verified_csv(dataset, self.dataset_storage)
            except DatasetIngestionError as error:
                raise DecisionBenchmarkError(str(error)) from error
            self._validate_columns(frame, feature_set.questions)
            self._validate_output_schema(feature_set.questions)
            config = self.provider_config.model_copy(
                update={
                    "provider_id": feature_set.provider_id,
                    "model": feature_set.requested_model,
                }
            )
            provider = self.providers.create(feature_set.provider_id, config)
            self._emit(on_event, feature_set_id, 0, 0, len(frame))
            rows = asyncio.run(
                self._generate(
                    feature_set=feature_set,
                    dataset_hash=dataset.content_hash,
                    frame=frame,
                    provider=provider,
                    on_event=on_event,
                )
            )
            return self._complete(feature_set, frame, rows, on_event)
        except Exception as error:
            self.feature_sets.transition(
                feature_set_id,
                DecisionRunStatus.FAILED,
                error_message=f"{type(error).__name__}: {error}",
            )
            raise

    async def _generate(
        self,
        *,
        feature_set: SemanticFeatureSet,
        dataset_hash: str,
        frame: pd.DataFrame,
        provider: DecisionProvider,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        """Coordena o processamento assíncrono em lotes para gerar features semânticas.

        Args:
            feature_set (SemanticFeatureSet): Configuração das features semânticas.
            dataset_hash (str): Hash SHA-256 do conteúdo do dataset original.
            frame (pd.DataFrame): DataFrame completo para enriquecimento semântico.
            provider (DecisionProvider): Provedor de inferência para avaliação de decisões.
            on_event (Callable[[dict[str, Any]], None] | None): Callback para envio de progresso.

        Returns:
            list[dict[str, Any]]: Lista de linhas contendo as features geradas e métricas.

        Raises:
            DecisionBenchmarkError: Se o provedor resolver modelos inconsistentes.
        """
        semaphore = asyncio.Semaphore(self.provider_config.concurrency)
        groups = self._question_groups(feature_set.questions)
        results: list[dict[str, Any]] = []
        totals: dict[str, int | float | None] = {
            "request_count": 0,
            "cache_hits": 0,
            "input_tokens": 0,
            "estimated_cost": 0.0,
            "latency_ms_total": 0.0,
        }
        for start in range(0, len(frame), self.chunk_size):
            chunk = frame.iloc[start : start + self.chunk_size]
            generated = await asyncio.gather(
                *[
                    self._generate_row(
                        row_index=str(index),
                        row=row,
                        groups=groups,
                        feature_set=feature_set,
                        dataset_hash=dataset_hash,
                        provider=provider,
                        semaphore=semaphore,
                    )
                    for index, row in chunk.iterrows()
                ]
            )
            results.extend(generated)
            self._accumulate(totals, generated)
            processed = len(results)
            self.feature_sets.checkpoint(
                feature_set.id,
                processed_rows=processed,
                request_count=int(totals["request_count"] or 0),
                cache_hits=int(totals["cache_hits"] or 0),
                input_tokens=totals["input_tokens"],
                estimated_cost=totals["estimated_cost"],
                latency_ms_total=float(totals["latency_ms_total"] or 0),
                resolved_model=generated[-1]["_resolved_model"],
            )
            self._emit(
                on_event,
                feature_set.id,
                processed * 100 / len(frame),
                processed,
                len(frame),
            )
        if len({item["_resolved_model"] for item in results}) != 1:
            raise DecisionBenchmarkError("Provider resolved multiple model versions")
        return results

    async def _generate_row(
        self,
        *,
        row_index: str,
        row: pd.Series,
        groups: dict[tuple[str, ...], list[SemanticQuestionSpec]],
        feature_set: SemanticFeatureSet,
        dataset_hash: str,
        provider: DecisionProvider,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """Processa e codifica as features semânticas para uma única linha de dados.

        Args:
            row_index (str): Identificador ou índice ordinal da linha.
            row (pd.Series): Valores da linha correspondente no DataFrame.
            groups (dict[tuple[str, ...], list[SemanticQuestionSpec]]): Perguntas agrupadas.
            feature_set (SemanticFeatureSet): Configuração das features semânticas.
            dataset_hash (str): Hash SHA-256 do dataset original.
            provider (DecisionProvider): Provedor de inferência.
            semaphore (asyncio.Semaphore): Semáforo para controle de concorrência.

        Returns:
            dict[str, Any]: Dicionário com as features codificadas e metadados de execução.

        Raises:
            DecisionBenchmarkError: Se múltiplos modelos forem resolvidos na mesma linha.
        """
        output: dict[str, Any] = {"row_id": row_index}
        usages: list[dict[str, Any]] = []
        latencies: list[float] = []
        resolved_models: set[str] = set()
        for columns, questions in groups.items():
            values = {column: self._scalar(row[column]) for column in columns}
            question_map = {question.id: question.question for question in questions}
            question_hash = DecisionSpec(
                provider=feature_set.provider_id,
                model=feature_set.requested_model,
                questions=question_map,
            ).question_spec_hash()
            context = {
                "dataset_hash": dataset_hash,
                "row_hash": self._hash(values),
                "question_spec_hash": question_hash,
                "provider_id": provider.id,
                "requested_model": feature_set.requested_model,
            }
            context_key = self._hash(context)
            cached = self.cache.get(context_key) if self.provider_config.cache_enabled else None
            if cached is not None:
                cached_response = DecisionResponse.model_validate(cached)
                response = cached_response.model_copy(
                    update={
                        "usage": cached_response.usage.model_copy(
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
                        DecisionState(row_id=row_index, values=values), question_map
                    )
                if self.provider_config.cache_enabled:
                    resolved = response.resolved_model or response.requested_model
                    self.cache.put(
                        context_key=context_key,
                        cache_key=self._hash({**context, "resolved_model": resolved}),
                        dataset_hash=dataset_hash,
                        row_hash=context["row_hash"],
                        question_spec_hash=question_hash,
                        provider_id=provider.id,
                        requested_model=feature_set.requested_model,
                        resolved_model=resolved,
                        response=response.model_dump(mode="json"),
                    )
            resolved_models.add(response.resolved_model or response.requested_model)
            usages.append(response.usage.model_dump(mode="json"))
            latencies.append(response.latency_ms or 0.0)
            for question in questions:
                self._encode(output, question, response.answers[question.id])
        if len(resolved_models) != 1:
            raise DecisionBenchmarkError("Provider resolved multiple models for one row")
        output["_usages"] = usages
        output["_latency_ms"] = sum(latencies)
        output["_resolved_model"] = resolved_models.pop()
        return output

    def _complete(
        self,
        feature_set: SemanticFeatureSet,
        source_frame: pd.DataFrame,
        rows: list[dict[str, Any]],
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        """Finaliza a execução, salva artefatos em Parquet/JSON e avalia impacto semântico.

        Args:
            feature_set (SemanticFeatureSet): Entidade do conjunto semântico.
            source_frame (pd.DataFrame): Dados originais para avaliação de impacto.
            rows (list[dict[str, Any]]): Linhas processadas e enriquecidas.
            on_event (Callable[[dict[str, Any]], None] | None): Callback de envio de eventos.

        Returns:
            dict[str, Any]: Informações finais com ID do conjunto e caminho do artefato Parquet.
        """
        prefix = f"decision/semantic-features/{feature_set.id}"
        self._save_json(
            f"{prefix}/questions.json",
            {
                "version": feature_set.version,
                "questions": [item.model_dump(mode="json") for item in feature_set.questions],
            },
        )
        clean_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")} for row in rows
        ]
        feature_frame = pd.DataFrame(clean_rows)
        buffer = io.BytesIO()
        feature_frame.to_parquet(buffer, index=False)
        feature_path = self.artifact_storage.save(f"{prefix}/features.parquet", buffer.getvalue())
        metadata = self._feature_metadata(feature_set.questions, feature_frame.columns)
        self._save_json(f"{prefix}/feature_metadata.json", metadata)
        totals: dict[str, int | float | None] = {
            "request_count": 0,
            "cache_hits": 0,
            "input_tokens": 0,
            "estimated_cost": 0.0,
            "latency_ms_total": 0.0,
        }
        self._accumulate(totals, rows)
        impact = None
        if self.impact_evaluator is not None and feature_set.source_experiment_id:
            impact = self.impact_evaluator.evaluate(
                experiment_id=feature_set.source_experiment_id,
                source_frame=source_frame,
                semantic_frame=feature_frame,
                semantic_columns=[column for column in feature_frame if column != "row_id"],
                text_columns=list(
                    dict.fromkeys(
                        column
                        for question in feature_set.questions
                        for column in question.input_columns
                    )
                ),
            )
            self._save_json(f"{prefix}/impact.json", impact)
        self.feature_sets.transition(
            feature_set.id,
            DecisionRunStatus.COMPLETED,
            resolved_model=rows[0]["_resolved_model"],
            processed_rows=len(rows),
            request_count=int(totals["request_count"] or 0),
            cache_hits=int(totals["cache_hits"] or 0),
            input_tokens=totals["input_tokens"],
            estimated_cost=totals["estimated_cost"],
            latency_ms_total=float(totals["latency_ms_total"] or 0),
            artifact_path=feature_path,
            impact_json=(SemanticFeatureSet.serialize_impact(impact) if impact else None),
        )
        self._emit(on_event, feature_set.id, 100, len(rows), len(rows))
        return {"semantic_feature_set_id": feature_set.id, "artifact_path": feature_path}

    @staticmethod
    def _encode(output: dict[str, Any], question: SemanticQuestionSpec, answer: Any) -> None:
        """Codifica a resposta do provedor de decisão nas colunas do schema semântico.

        Args:
            output (dict[str, Any]): Dicionário acumulador de atributos da linha.
            question (SemanticQuestionSpec): Especificação da pergunta semântica.
            answer (Any): Resposta estruturada retornada pelo modelo.

        Raises:
            DecisionBenchmarkError: Se o tipo da resposta não for suportado.
        """
        prefix = f"semantic_{question.id}"
        if isinstance(answer, NoulDecisionAnswer):
            output[f"{prefix}_probability"] = answer.probability_yes
        elif isinstance(answer, ChoiceDecisionAnswer):
            for choice, probability in answer.choice_probabilities.items():
                output[f"{prefix}_{SemanticFeatureExecutionService._slug(choice)}"] = probability
        elif isinstance(answer, ScoreDecisionAnswer):
            output[f"{prefix}_expected_score"] = answer.score
            output[f"{prefix}_uncertainty"] = 1 - answer.confidence
        else:
            raise DecisionBenchmarkError("Unsupported semantic answer type")

    @staticmethod
    def _question_groups(
        questions: list[SemanticQuestionSpec],
    ) -> dict[tuple[str, ...], list[SemanticQuestionSpec]]:
        """Agrupa perguntas semânticas que utilizam o mesmo conjunto de colunas de entrada.

        Args:
            questions (list[SemanticQuestionSpec]): Lista de perguntas a serem agrupadas.

        Returns:
            dict[tuple[str, ...], list[SemanticQuestionSpec]]: Perguntas indexadas por colunas.
        """
        groups: dict[tuple[str, ...], list[SemanticQuestionSpec]] = defaultdict(list)
        for question in questions:
            groups[tuple(sorted(question.input_columns))].append(question)
        return dict(groups)

    @staticmethod
    def _validate_columns(frame: pd.DataFrame, questions: list[SemanticQuestionSpec]) -> None:
        """Verifica se todas as colunas necessárias para as perguntas existem no DataFrame.

        Args:
            frame (pd.DataFrame): Dados tabulares carregados.
            questions (list[SemanticQuestionSpec]): Perguntas que declaram dependências de coluna.

        Raises:
            DecisionBenchmarkError: Se alguma coluna obrigatória estiver ausente.
        """
        required = {column for question in questions for column in question.input_columns}
        missing = required - set(frame.columns)
        if missing:
            raise DecisionBenchmarkError(f"Dataset is missing columns: {sorted(missing)}")

    @staticmethod
    def _feature_metadata(questions: list[SemanticQuestionSpec], columns: Any) -> dict[str, Any]:
        """Gera metadados associando cada coluna gerada à respectiva pergunta semântica.

        Args:
            questions (list[SemanticQuestionSpec]): Perguntas que originaram as colunas.
            columns (Any): Lista ou índice de colunas semânticas geradas.

        Returns:
            dict[str, Any]: Dicionário estruturado com metadados das features.
        """
        question_ids = sorted((question.id for question in questions), key=len, reverse=True)
        return {
            "schema_version": "1.0",
            "features": [
                {
                    "name": column,
                    "kind": "semantic",
                    "question_id": next(
                        question_id
                        for question_id in question_ids
                        if column.startswith(f"semantic_{question_id}_")
                    ),
                }
                for column in columns
                if column != "row_id"
            ],
        }

    @staticmethod
    def _validate_output_schema(questions: list[SemanticQuestionSpec]) -> None:
        """Assegura que as perguntas semânticas não gerem colunas com nomes conflitantes.

        Args:
            questions (list[SemanticQuestionSpec]): Perguntas a serem inspecionadas.

        Raises:
            DecisionBenchmarkError: Se houver colisão de nomes de colunas de saída.
        """
        feature_names: list[str] = []
        for question in questions:
            prefix = f"semantic_{question.id}"
            if question.question.type == "noul":
                feature_names.append(f"{prefix}_probability")
            elif question.question.type == "score":
                feature_names.extend([f"{prefix}_expected_score", f"{prefix}_uncertainty"])
            else:
                feature_names.extend(
                    f"{prefix}_{SemanticFeatureExecutionService._slug(choice)}"
                    for choice in question.question.criteria
                )
        if len(feature_names) != len(set(feature_names)):
            raise DecisionBenchmarkError(
                "Semantic questions produce colliding feature column names"
            )

    @staticmethod
    def _accumulate(totals: dict[str, int | float | None], rows: list[dict[str, Any]]) -> None:
        """Acumula estatísticas de execução (latência, requisições, tokens e custo).

        Args:
            totals (dict[str, int | float | None]): Dicionário acumulador atualizado in-place.
            rows (list[dict[str, Any]]): Linhas processadas contendo registros de uso.
        """
        for row in rows:
            totals["latency_ms_total"] = float(totals["latency_ms_total"] or 0) + float(
                row["_latency_ms"]
            )
            for usage in row["_usages"]:
                totals["request_count"] = int(totals["request_count"] or 0) + int(
                    usage["request_count"]
                )
                totals["cache_hits"] = int(totals["cache_hits"] or 0) + int(usage["cache_hit"])
                for field in ("input_tokens", "estimated_cost"):
                    if usage[field] is None:
                        totals[field] = None
                    elif totals[field] is not None:
                        totals[field] = totals[field] + usage[field]

    def _save_json(self, artifact_id: str, payload: Mapping[str, Any]) -> str:
        """Serializa e salva um dicionário como arquivo JSON no storage de artefatos.

        Args:
            artifact_id (str): Identificador do artefato a ser salvo.
            payload (Mapping[str, Any]): Conteúdo estruturado para serialização.

        Returns:
            str: Caminho retornado pelo storage para o arquivo persistido.
        """
        return self.artifact_storage.save(
            artifact_id,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(),
        )

    @staticmethod
    def _scalar(value: Any) -> str | int | float | bool | None:
        """Normaliza tipos numpy ou pandas para valores primitivos escalares Python.

        Args:
            value (Any): Valor a ser convertido.

        Returns:
            str | int | float | bool | None: Escalar primitivo ou None para valores nulos.
        """
        if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
            return None
        return value.item() if isinstance(value, np.generic) else value

    @staticmethod
    def _hash(payload: Mapping[str, Any]) -> str:
        """Calcula o hash SHA-256 de um dicionário serializado em formato JSON canônico.

        Args:
            payload (Mapping[str, Any]): Dicionário com chaves e valores a hashear.

        Returns:
            str: Hexdigest SHA-256 de 64 caracteres.
        """
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    @staticmethod
    def _slug(value: str) -> str:
        """Converte uma string em um identificador alfanumérico seguro para nomes de coluna.

        Args:
            value (str): Texto de origem.

        Returns:
            str: Identificador normalizado em snake_case ou prefixo de hash.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        return slug or hashlib.sha256(value.encode()).hexdigest()[:12]

    @staticmethod
    def _emit(
        callback: Callable[[dict[str, Any]], None] | None,
        feature_set_id: str,
        progress_value: float,
        processed: int,
        total: int,
    ) -> None:
        """Emite notificação de progresso estruturada se o callback estiver configurado.

        Args:
            callback (Callable[[dict[str, Any]], None] | None): Função receptora do evento.
            feature_set_id (str): Identificador do conjunto de features semânticas.
            progress_value (float): Valor percentual base de progresso.
            processed (int): Quantidade de linhas concluídas até o momento.
            total (int): Quantidade total de linhas a serem processadas.
        """
        if callback is not None:
            progress = progress_value if progress_value <= 100 else processed * 100 / total
            callback(
                {
                    "type": "semantic_features.progress",
                    "semantic_feature_set_id": feature_set_id,
                    "processed_rows": processed,
                    "total_rows": total,
                    "progress": round(progress, 2),
                    "message_key": "semantic_features.progress",
                }
            )
