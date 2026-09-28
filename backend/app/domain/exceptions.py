class DomainError(Exception):
    """Representa a classe base para violações de invariantes de domínio."""


class EntityNotFoundError(DomainError):
    """Indica que uma entidade solicitada não foi localizada no repositório."""


class ImmutableExperimentError(DomainError):
    """Indica tentativa indevida de alteração em experimento já executado."""


class ExperimentDeletionError(DomainError):
    """Indica que um experimento não pode ser removido do histórico visível."""


class InvalidExperimentNameError(DomainError):
    """Indica que o nome de exibição de um experimento é inválido ou vazio."""


class ImmutableModelRunError(DomainError):
    """Indica tentativa de alteração em uma execução de modelo finalizada."""


class ArtifactAlreadyExistsError(DomainError):
    """Indica conflito ao tentar sobrescrever um artefato histórico existente."""


class RegistryItemNotFoundError(DomainError):
    """Indica que um item de registro foi solicitado através de identificador desconhecido."""


class DuplicateRegistryItemError(DomainError):
    """Indica que dois componentes tentaram se registrar com o mesmo identificador."""


class ExperimentSpecValidationError(DomainError):
    """Indica que uma especificação de experimento referencia componentes incompatíveis.

    Attributes:
        issues (tuple[str, ...]): Coleção com os motivos de validação rejeitados.
    """

    def __init__(self, issues: list[str]) -> None:
        """Inicializa a exceção com a lista de inconsistências encontradas.

        Args:
            issues (list[str]): Lista com mensagens descritivas dos problemas.
        """
        super().__init__("; ".join(issues))
        self.issues = tuple(issues)


class PipelineCompilationError(DomainError):
    """Indica falha ao tentar compilar etapas de pré-processamento para o modelo."""


class ModelTaskIncompatibleError(DomainError):
    """Indica incompatibilidade entre o adaptador de modelo e a tarefa de ML."""


class MetricEvaluationError(DomainError):
    """Indica que as métricas selecionadas não puderam ser avaliadas."""


class ValidationStrategyError(DomainError):
    """Indica incompatibilidade da estratégia de validação com os dados ou tarefa."""


class ExperimentExecutionError(DomainError):
    """Indica falha crítica durante a execução reproduzível de um experimento."""


class DecisionBenchmarkError(DomainError):
    """Indica erro durante a execução de benchmark zero-shot de provedores."""


class JobNotFoundError(DomainError):
    """Indica que a tarefa em segundo plano requisitada não foi encontrada."""


class JobAlreadyRunningError(DomainError):
    """Indica que já existe um job em execução ativa para o experimento alvo."""


class JobStateError(DomainError):
    """Indica que uma operação solicitada é incompatível com o estado atual do job."""
