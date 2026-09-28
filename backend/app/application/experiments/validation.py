from app.domain.exceptions import ExperimentSpecValidationError
from app.domain.experiment.spec import ExperimentSpec
from app.ml.registries.catalog import RegistryCatalog


class ExperimentSpecValidator:
    """Validador semântico de especificações de experimentos contra o catálogo do sistema.

    Attributes:
        catalog (RegistryCatalog): Catálogo de componentes de ML (modelos, métricas, etc.).
    """

    def __init__(self, catalog: RegistryCatalog) -> None:
        """Inicializa o validador com o catálogo de componentes.

        Args:
            catalog: Catálogo de componentes de ML registrados no sistema.
        """
        self.catalog = catalog

    def validate(self, spec: ExperimentSpec) -> ExperimentSpec:
        """Valida a consistência de modelos, métricas, validadores, pré-processamento e tuning.

        Verifica se todos os componentes referenciados existem no catálogo e se são
        compatíveis com a tarefa de machine learning e entre si.

        Args:
            spec: Especificação do experimento a ser validada.

        Returns:
            A própria instância de ExperimentSpec validada.

        Raises:
            ExperimentSpecValidationError: Se houver componentes desconhecidos ou incompatíveis.
        """
        issues: list[str] = []
        enabled_models = [model for model in spec.models if model.enabled]

        for model in enabled_models:
            if not self.catalog.models.contains(model.id):
                issues.append(f"unknown_model:{model.id}")
                continue
            metadata = self.catalog.models.get(model.id)
            if spec.task not in metadata.tasks:
                issues.append(f"model_task_incompatible:{model.id}:{spec.task.value}")

        metric_ids = [spec.metrics.primary, *spec.metrics.secondary]
        for metric_id in metric_ids:
            if not self.catalog.metrics.contains(metric_id):
                issues.append(f"unknown_metric:{metric_id}")
                continue
            metric = self.catalog.metrics.get(metric_id)
            if spec.task not in metric.tasks:
                issues.append(f"metric_task_incompatible:{metric_id}:{spec.task.value}")
            if metric.requires_proba:
                incompatible = [
                    model.id
                    for model in enabled_models
                    if self.catalog.models.contains(model.id)
                    and not self.catalog.models.get(model.id).capabilities.supports_proba
                ]
                issues.extend(
                    f"metric_requires_proba:{metric_id}:{model_id}" for model_id in incompatible
                )

        if not self.catalog.validators.contains(spec.validation.strategy):
            issues.append(f"unknown_validator:{spec.validation.strategy}")
        elif spec.task not in self.catalog.validators.get(spec.validation.strategy).tasks:
            issues.append(
                f"validator_task_incompatible:{spec.validation.strategy}:{spec.task.value}"
            )

        for step in spec.preprocessing:
            if not self.catalog.preprocessors.contains(step.id):
                issues.append(f"unknown_preprocessor:{step.id}")
                continue
            if step.strategy not in self.catalog.preprocessors.get(step.id).strategies:
                issues.append(f"unknown_preprocessing_strategy:{step.id}:{step.strategy}")

        if spec.tuning.enabled:
            if not self.catalog.tuners.contains(spec.tuning.engine):
                issues.append(f"unknown_tuner:{spec.tuning.engine}")
            if spec.tuning.metric not in metric_ids:
                issues.append(f"tuning_metric_not_selected:{spec.tuning.metric}")

        if issues:
            raise ExperimentSpecValidationError(issues)
        return spec
