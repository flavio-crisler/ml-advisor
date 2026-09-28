from collections.abc import Mapping
from typing import Any, Protocol

from app.domain.decision.cascade import CascadeRun
from app.domain.decision.runs import CalibrationRun, DecisionRun, DecisionRunStatus
from app.domain.decision.semantic_features import SemanticFeatureSet


class DecisionRunRepository(Protocol):
    """Define o contrato de persistência para execuções de benchmark de decisão zero-shot."""

    def add(self, run: DecisionRun) -> DecisionRun:
        """Persiste uma nova execução de decisão no repositório.

        Args:
            run (DecisionRun): Entidade da execução a salvar.

        Returns:
            DecisionRun: Instância persistida da execução.
        """
        ...

    def get(self, run_id: str) -> DecisionRun | None:
        """Recupera uma execução de decisão pelo identificador único.

        Args:
            run_id (str): Identificador da execução.

        Returns:
            DecisionRun | None: Execução encontrada ou None se inexistente.
        """
        ...

    def list_all(self) -> list[DecisionRun]:
        """Lista todas as execuções de decisão cadastradas.

        Returns:
            list[DecisionRun]: Lista de execuções de decisão registradas.
        """
        ...

    def transition(
        self,
        run_id: str,
        status: DecisionRunStatus,
        **changes: Any,
    ) -> DecisionRun:
        """Altera o status da execução aplicando modificações incrementais de estado.

        Args:
            run_id (str): Identificador da execução.
            status (DecisionRunStatus): Novo status a ser atribuído.
            **changes (Any): Atributos adicionais modificados (ex: métricas, erros).

        Returns:
            DecisionRun: Instância atualizada da execução.
        """
        ...

    def checkpoint(self, run_id: str, **changes: Any) -> DecisionRun:
        """Salva um ponto intermediário de progresso da execução sem alterar o status.

        Args:
            run_id (str): Identificador da execução.
            **changes (Any): Campos incrementais a atualizar (ex: progresso, custos).

        Returns:
            DecisionRun: Execução com o checkpoint registrado.
        """
        ...


class CalibrationRunRepository(Protocol):
    """Define o contrato para persistência e recuperação de execuções de calibração."""

    def add(self, run: CalibrationRun) -> CalibrationRun:
        """Persiste uma execução de calibração no repositório.

        Args:
            run (CalibrationRun): Execução de calibração a salvar.

        Returns:
            CalibrationRun: Instância persistida.
        """
        ...

    def get(self, run_id: str) -> CalibrationRun | None:
        """Recupera uma execução de calibração por ID.

        Args:
            run_id (str): Identificador da calibração.

        Returns:
            CalibrationRun | None: Instância encontrada ou None.
        """
        ...

    def get_for_decision_run(self, decision_run_id: str) -> CalibrationRun | None:
        """Recupera a execução de calibração vinculada a um benchmark de decisão.

        Args:
            decision_run_id (str): Identificador do benchmark de decisão.

        Returns:
            CalibrationRun | None: Calibração associada ou None se inexistente.
        """
        ...


class DecisionCacheRepository(Protocol):
    """Define o contrato de cache determinístico de inferências de decisão."""

    def get(self, context_key: str) -> Mapping[str, Any] | None:
        """Consulta o cache em busca de uma inferência previamente calculada.

        Args:
            context_key (str): Chave composta contextual da inferência.

        Returns:
            Mapping[str, Any] | None: Resposta em cache ou None se ausente.
        """
        ...

    def put(
        self,
        *,
        context_key: str,
        cache_key: str,
        dataset_hash: str,
        row_hash: str,
        question_spec_hash: str,
        provider_id: str,
        requested_model: str,
        resolved_model: str,
        response: Mapping[str, Any],
    ) -> None:
        """Armazena o resultado de uma inferência de decisão no cache.

        Args:
            context_key (str): Chave contextual completa.
            cache_key (str): Chave canônica de identificação do conteúdo.
            dataset_hash (str): Hash do dataset de origem.
            row_hash (str): Hash do conteúdo da linha avaliada.
            question_spec_hash (str): Hash da especificação da questão.
            provider_id (str): Identificador do provedor que avaliou.
            requested_model (str): Nome do modelo originalmente solicitado.
            resolved_model (str): Nome da versão exata do modelo utilizada.
            response (Mapping[str, Any]): Dicionário com a resposta gerada.
        """
        ...


class SemanticFeatureSetRepository(Protocol):
    """Define o contrato de armazenamento para conjuntos de features semânticas descobertas."""

    def add(self, feature_set: SemanticFeatureSet) -> SemanticFeatureSet:
        """Persiste um conjunto de features semânticas no repositório.

        Args:
            feature_set (SemanticFeatureSet): Entidade a salvar.

        Returns:
            SemanticFeatureSet: Instância salva do conjunto.
        """
        ...

    def get(self, feature_set_id: str) -> SemanticFeatureSet | None:
        """Recupera um conjunto de features semânticas por ID.

        Args:
            feature_set_id (str): Identificador único do conjunto.

        Returns:
            SemanticFeatureSet | None: Conjunto localizado ou None.
        """
        ...

    def list_all(self) -> list[SemanticFeatureSet]:
        """Lista todos os conjuntos de features semânticas registrados.

        Returns:
            list[SemanticFeatureSet]: Relação de conjuntos encontrados.
        """
        ...

    def transition(
        self, feature_set_id: str, status: DecisionRunStatus, **changes: Any
    ) -> SemanticFeatureSet:
        """Atualiza o estado de processamento do conjunto de features.

        Args:
            feature_set_id (str): Identificador do conjunto de features.
            status (DecisionRunStatus): Novo estado da tarefa.
            **changes (Any): Propriedades complementares atualizadas.

        Returns:
            SemanticFeatureSet: Instância atualizada do conjunto.
        """
        ...

    def checkpoint(self, feature_set_id: str, **changes: Any) -> SemanticFeatureSet:
        """Grava checkpoint intermediário no processamento de features semânticas.

        Args:
            feature_set_id (str): Identificador do conjunto.
            **changes (Any): Dados intermediários atualizados.

        Returns:
            SemanticFeatureSet: Instância com progresso gravado.
        """
        ...


class CascadeRunRepository(Protocol):
    """Define o contrato de persistência para execuções de cascata de confiança."""

    def add(self, run: CascadeRun) -> CascadeRun:
        """Persiste uma execução de política de cascata.

        Args:
            run (CascadeRun): Instância da execução de cascata a salvar.

        Returns:
            CascadeRun: Instância persistida.
        """
        ...

    def get(self, run_id: str) -> CascadeRun | None:
        """Recupera uma execução de cascata por ID.

        Args:
            run_id (str): Identificador da execução.

        Returns:
            CascadeRun | None: Execução encontrada ou None.
        """
        ...

    def list_all(self) -> list[CascadeRun]:
        """Lista todas as execuções de cascata de confiança.

        Returns:
            list[CascadeRun]: Relação de execuções de cascata salvas.
        """
        ...
