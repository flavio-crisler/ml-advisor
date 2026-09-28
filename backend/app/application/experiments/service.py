from collections.abc import Callable

from app.application.datasets.profiling import DatasetProfile
from app.application.experiments.validation import ExperimentSpecValidator
from app.domain.dataset.repositories import DatasetRepository
from app.domain.exceptions import (
    EntityNotFoundError,
    ExperimentDeletionError,
    ExperimentExecutionError,
    ExperimentSpecValidationError,
    ImmutableExperimentError,
    InvalidExperimentNameError,
    PipelineCompilationError,
)
from app.domain.experiment.entities import Experiment, ExperimentMode, ExperimentStatus
from app.domain.experiment.repositories import ExperimentRepository
from app.domain.experiment.spec import ExperimentSpec, TaskType
from app.ml.preprocessing.compiler import PipelineCompiler


class ExperimentService:
    """Serviço de aplicação responsável pelo ciclo de vida e regras de experimentos de ML.

    Attributes:
        experiments (ExperimentRepository): Repositório para persistência de experimentos.
        datasets (DatasetRepository): Repositório de datasets associados aos experimentos.
        validator (ExperimentSpecValidator): Validador semântico das especificações.
        load_profile (Callable[[str], DatasetProfile] | None): Função para carregamento de perfis.
    """

    def __init__(
        self,
        *,
        experiments: ExperimentRepository,
        datasets: DatasetRepository,
        validator: ExperimentSpecValidator,
        load_profile: Callable[[str], DatasetProfile] | None = None,
    ) -> None:
        """Inicializa o serviço de gerenciamento de experimentos.

        Args:
            experiments: Repositório de experimentos.
            datasets: Repositório de datasets.
            validator: Validador de conformidade da especificação.
            load_profile: Função opcional para carregar o perfil do dataset.
        """
        self.experiments = experiments
        self.datasets = datasets
        self.validator = validator
        self.load_profile = load_profile

    def create(
        self,
        specification: ExperimentSpec,
        *,
        mode: ExperimentMode,
        description: str | None = None,
    ) -> Experiment:
        """Cria e persiste um novo experimento a partir de sua especificação validada.

        Args:
            specification: Especificação técnica completa do experimento.
            mode: Modo de criação do experimento (manual ou advisor).
            description: Descrição opcional informada pelo usuário.

        Returns:
            Entidade Experiment persistida em estado de rascunho.
        """
        self.validate_specification(specification)
        return self.experiments.add(
            Experiment.create(
                name=specification.name,
                mode=mode,
                dataset_id=specification.dataset_id,
                objective=specification.objective,
                spec_version=specification.spec_version,
                spec=specification.model_dump(mode="json"),
                description=description,
            )
        )

    def validate_specification(self, specification: ExperimentSpec) -> ExperimentSpec:
        """Valida integralmente uma especificação técnica sem persistir ou iniciar execução.

        Args:
            specification: Especificação a ser validada.

        Returns:
            A especificação validada.

        Raises:
            EntityNotFoundError: Se o dataset associado não for encontrado.
            ExperimentExecutionError: Se o hash do dataset divergir do registrado.
            ExperimentSpecValidationError: Se houver violação de contrato com o perfil do dataset.
        """
        self.validator.validate(specification)
        dataset = self.datasets.get(specification.dataset_id)
        if dataset is None:
            raise EntityNotFoundError(f"Dataset {specification.dataset_id} was not found")
        if dataset.content_hash != specification.dataset_hash:
            raise ExperimentExecutionError(
                "Experiment dataset hash does not match persisted dataset"
            )
        self._validate_dataset_contract(specification)
        return specification

    def get(self, experiment_id: str) -> Experiment:
        """Recupera um experimento ativo por ID.

        Args:
            experiment_id: Identificador único do experimento.

        Returns:
            Entidade Experiment correspondente.

        Raises:
            EntityNotFoundError: Se o experimento não existir ou estiver deletado.
        """
        experiment = self.experiments.get(experiment_id)
        if experiment is None or experiment.deleted_at is not None:
            raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
        return experiment

    def get_including_deleted(self, experiment_id: str) -> Experiment:
        """Carrega metadados de um experimento mesmo se soft-deleted para fins de histórico.

        Args:
            experiment_id: Identificador único do experimento.

        Returns:
            Entidade Experiment correspondente.

        Raises:
            EntityNotFoundError: Se o experimento não existir.
        """
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
        return experiment

    def list_all(self) -> list[Experiment]:
        """Lista todos os experimentos ativos cadastrados no repositório.

        Returns:
            Lista contendo as entidades de experimentos ativas.
        """
        return self.experiments.list_all()

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> Experiment:
        """Atualiza o status de execução de um experimento existente.

        Args:
            experiment_id: Identificador do experimento.
            status: Novo status a ser atribuído.

        Returns:
            Entidade Experiment com o status atualizado.
        """
        self.get(experiment_id)
        return self.experiments.update_status(experiment_id, status)

    def rename(self, experiment_id: str, name: str) -> Experiment:
        """Altera o nome de exibição de um experimento.

        Args:
            experiment_id: Identificador do experimento.
            name: Novo nome para o experimento.

        Returns:
            Entidade Experiment com o novo nome.

        Raises:
            InvalidExperimentNameError: Se o novo nome for vazio ou contiver apenas espaços.
        """
        self.get(experiment_id)
        normalized_name = name.strip()
        if not normalized_name:
            raise InvalidExperimentNameError("Experiment name cannot be empty")
        return self.experiments.rename(experiment_id, normalized_name)

    def delete(self, experiment_id: str) -> None:
        """Executa a exclusão lógica (soft-delete) de um experimento inativo.

        Args:
            experiment_id: Identificador do experimento.

        Raises:
            ExperimentDeletionError: Se o experimento estiver em execução ou na fila.
        """
        experiment = self.get(experiment_id)
        if experiment.status in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}:
            raise ExperimentDeletionError(
                "Queued or running experiments must be cancelled before deletion"
            )
        self.experiments.soft_delete(experiment_id)

    def clone(self, experiment_id: str, *, name: str | None = None) -> Experiment:
        """Clona um experimento existente gerando um novo rascunho com a mesma configuração.

        Args:
            experiment_id: Identificador do experimento original a ser clonado.
            name: Nome opcional customizado para a nova cópia.

        Returns:
            Nova entidade Experiment clonada e persistida.
        """
        original = self.get(experiment_id)
        return self.experiments.add(original.clone(name=name))

    def update_spec(self, experiment_id: str, specification: ExperimentSpec) -> Experiment:
        """Atualiza a especificação de um experimento que ainda se encontra em rascunho (DRAFT).

        Args:
            experiment_id: Identificador do experimento a modificar.
            specification: Nova especificação técnica a ser aplicada.

        Returns:
            Entidade Experiment com a especificação substituída.

        Raises:
            ImmutableExperimentError: Se o experimento já tiver sido executado.
        """
        experiment = self.get(experiment_id)
        if experiment.status != ExperimentStatus.DRAFT:
            raise ImmutableExperimentError(
                "Executed experiment configuration cannot be changed; clone it instead"
            )
        self.validate_specification(specification)
        return self.experiments.replace_spec(
            experiment_id,
            spec_version=specification.spec_version,
            spec=specification.model_dump(mode="json"),
            name=specification.name,
            dataset_id=specification.dataset_id,
            objective=specification.objective,
        )

    def _validate_dataset_contract(self, specification: ExperimentSpec) -> None:
        """Verifica a conformidade entre a especificação e o perfil estatístico do dataset.

        Valida colunas, papéis, consistência do alvo com a tarefa e viabilidade de compilação
        do pipeline de pré-processamento para cada modelo habilitado.

        Args:
            specification: Especificação técnica do experimento a ser validada.

        Raises:
            ExperimentSpecValidationError: Se houver qualquer incompatibilidade de contrato.
        """
        if self.load_profile is None:
            return
        profile = self.load_profile(specification.dataset_id)
        issues: list[str] = []
        columns = set(profile.column_names)
        if specification.profile_version != profile.profile_version:
            issues.append(
                "profile_version_mismatch:"
                f"{specification.profile_version}:{profile.profile_version}"
            )
        if specification.target not in columns:
            issues.append(f"unknown_target:{specification.target}")
        else:
            target_profile = next(
                column for column in profile.column_profiles if column.name == specification.target
            )
            if (
                specification.task == TaskType.BINARY_CLASSIFICATION
                and target_profile.cardinality != 2
            ):
                issues.append("binary_target_requires_two_classes")
            if (
                specification.task == TaskType.MULTICLASS_CLASSIFICATION
                and target_profile.cardinality < 3
            ):
                issues.append("multiclass_target_requires_three_classes")
            if (
                specification.task == TaskType.REGRESSION
                and target_profile.inferred_type.value != "numeric"
            ):
                issues.append("regression_target_must_be_numeric")
        issues.extend(
            f"unknown_feature:{feature}"
            for feature in specification.features
            if feature not in columns
        )
        issues.extend(
            f"unknown_column_role:{column}"
            for column in specification.column_roles
            if column not in columns
        )
        if not issues:
            column_types = {
                column.name: column.inferred_type.value for column in profile.column_profiles
            }
            for model in (model for model in specification.models if model.enabled):
                if not self.validator.catalog.models.contains(model.id):
                    continue
                try:
                    PipelineCompiler().compile(
                        specification,
                        column_types=column_types,
                        capabilities=self.validator.catalog.models.get(model.id).capabilities,
                    )
                except PipelineCompilationError as error:
                    issues.append(f"preprocessing_incompatible:{model.id}:{error}")
        if issues:
            raise ExperimentSpecValidationError(issues)
