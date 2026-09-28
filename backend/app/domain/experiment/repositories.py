from collections.abc import Mapping
from typing import Any, Protocol

from app.domain.experiment.entities import (
    Experiment,
    ExperimentStatus,
    ExperimentStrategy,
    ModelRun,
    ModelRunStatus,
    TuningTrial,
)


class ExperimentRepository(Protocol):
    """Define o contrato de persistência e consulta para experimentos de ML."""

    def add(self, experiment: Experiment) -> Experiment:
        """Persiste um novo experimento no repositório.

        Args:
            experiment (Experiment): Instância do experimento a salvar.

        Returns:
            Experiment: Instância persistida do experimento.
        """
        ...

    def get(self, experiment_id: str) -> Experiment | None:
        """Recupera um experimento ativo através do seu identificador.

        Args:
            experiment_id (str): Identificador único do experimento.

        Returns:
            Experiment | None: Entidade encontrada ou None se inexistente ou excluída.
        """
        ...

    def list_all(self) -> list[Experiment]:
        """Lista todos os experimentos ativos ordenados cronologicamente.

        Returns:
            list[Experiment]: Lista de experimentos não excluídos logicamente.
        """
        ...

    def replace_spec(
        self,
        experiment_id: str,
        *,
        spec_version: str,
        spec: Mapping[str, Any],
        name: str | None = None,
        dataset_id: str | None = None,
        objective: str | None = None,
    ) -> Experiment:
        """Atualiza a especificação técnica de um experimento em rascunho.

        Args:
            experiment_id (str): Identificador do experimento a modificar.
            spec_version (str): Nova versão do contrato da especificação.
            spec (Mapping[str, Any]): Conteúdo estruturado da nova especificação.
            name (str | None): Novo nome opcional.
            dataset_id (str | None): Novo dataset_id opcional.
            objective (str | None): Novo objetivo opcional.

        Returns:
            Experiment: Experimento atualizado.
        """
        ...

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> Experiment:
        """Atualiza o estado de ciclo de vida do experimento.

        Args:
            experiment_id (str): Identificador do experimento.
            status (ExperimentStatus): Novo status a ser atribuído.

        Returns:
            Experiment: Instância atualizada do experimento.
        """
        ...

    def rename(self, experiment_id: str, name: str) -> Experiment:
        """Modifica o nome de exibição de um experimento existente.

        Args:
            experiment_id (str): Identificador do experimento.
            name (str): Novo nome descritivo.

        Returns:
            Experiment: Instância do experimento com o nome atualizado.
        """
        ...

    def soft_delete(self, experiment_id: str) -> None:
        """Marca logicamente o experimento como excluído.

        Args:
            experiment_id (str): Identificador do experimento a ser removido.
        """
        ...


class ModelRunRepository(Protocol):
    """Define o contrato de persistência para execuções de treinamento de modelos."""

    def create(
        self,
        *,
        experiment_id: str,
        model_id: str,
        random_seed: int,
        parameters: Mapping[str, Any],
        execution_id: str | None = None,
    ) -> ModelRun:
        """Cria e persiste um novo registro de execução de modelo em fila.

        Args:
            experiment_id (str): Identificador do experimento pai.
            model_id (str): Identificador do modelo no catálogo.
            random_seed (int): Semente aleatória para reproduzibilidade.
            parameters (Mapping[str, Any]): Hiperparâmetros configurados.
            execution_id (str | None): Identificador do lote de execução associado.

        Returns:
            ModelRun: Instância da execução criada e salva.
        """
        ...

    def get(self, run_id: str) -> ModelRun | None:
        """Recupera uma execução de modelo pelo identificador.

        Args:
            run_id (str): Identificador único da execução.

        Returns:
            ModelRun | None: Execução localizada ou None se não existir.
        """
        ...

    def list_for_experiment(self, experiment_id: str) -> list[ModelRun]:
        """Lista todas as execuções de modelo pertencentes a um experimento.

        Args:
            experiment_id (str): Identificador do experimento.

        Returns:
            list[ModelRun]: Lista de execuções de modelo do experimento.
        """
        ...

    def transition(
        self,
        run_id: str,
        status: ModelRunStatus,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact_path: str | None = None,
        error_message: str | None = None,
    ) -> ModelRun:
        """Atualiza o status de execução de um modelo e anexa métricas ou erros.

        Args:
            run_id (str): Identificador da execução a transicionar.
            status (ModelRunStatus): Novo status de execução.
            metrics (Mapping[str, Any] | None): Métricas computadas se finalizado.
            artifact_path (str | None): Caminho do modelo serializado em disco.
            error_message (str | None): Detalhes do erro em caso de falha.

        Returns:
            ModelRun: Instância da execução com o novo estado gravado.
        """
        ...


class TuningTrialRepository(Protocol):
    """Define o contrato para persistência de tentativas de ajuste de hiperparâmetros."""

    def add(self, trial: TuningTrial) -> TuningTrial:
        """Persiste uma tentativa de sintonia no repositório.

        Args:
            trial (TuningTrial): Instância da tentativa a persistir.

        Returns:
            TuningTrial: Instância persistida da tentativa.
        """
        ...

    def list_for_run(self, run_id: str) -> list[TuningTrial]:
        """Lista todas as tentativas de sintonia vinculadas a uma execução de modelo.

        Args:
            run_id (str): Identificador da execução do modelo.

        Returns:
            list[TuningTrial]: Lista com as tentativas de otimização registradas.
        """
        ...


class SettingRepository(Protocol):
    """Define o contrato de persistência para pares chave-valor de configuração."""

    def get(self, key: str) -> str | None:
        """Busca o valor textual associado a uma chave de configuração.

        Args:
            key (str): Chave identificadora do parâmetro.

        Returns:
            str | None: Valor armazenado ou None se a chave não existir.
        """
        ...

    def set(self, key: str, value: str) -> None:
        """Salva ou substitui o valor associado a uma chave de configuração.

        Args:
            key (str): Chave identificadora do parâmetro.
            value (str): Novo valor textual a ser armazenado.
        """
        ...


class ExperimentStrategyRepository(Protocol):
    """Define o contrato para armazenamento e recuperação de estratégias de experimentos."""

    def add(self, strategy: ExperimentStrategy) -> ExperimentStrategy:
        """Salva uma nova estratégia associada a um experimento.

        Args:
            strategy (ExperimentStrategy): Instância de estratégia a salvar.

        Returns:
            ExperimentStrategy: Instância de estratégia persistida.
        """
        ...

    def get_for_experiment(self, experiment_id: str) -> ExperimentStrategy | None:
        """Obtém a proposta de estratégia registrada para determinado experimento.

        Args:
            experiment_id (str): Identificador do experimento.

        Returns:
            ExperimentStrategy | None: Estratégia encontrada ou None.
        """
        ...

    def add_experiment_with_strategy(
        self,
        experiment: Experiment,
        strategy: ExperimentStrategy,
    ) -> Experiment:
        """Persiste atomicamente um experimento e sua respectiva estratégia.

        Args:
            experiment (Experiment): Entidade do experimento a persistir.
            strategy (ExperimentStrategy): Proposta de estratégia associada.

        Returns:
            Experiment: Instância persistida do experimento.
        """
        ...
