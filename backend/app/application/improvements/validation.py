import hashlib
import json

from app.application.experiments.validation import ExperimentSpecValidator
from app.application.improvements.capabilities import ImprovementCapabilityCatalog
from app.application.improvements.models import (
    ConfigurationChange,
    SupervisedImprovementRecommendation,
)
from app.domain.common import canonical_json
from app.domain.experiment.spec import ExperimentSpec

MUTABLE_FIELDS = ("models", "preprocessing", "validation", "metrics", "tuning")


def experiment_spec_hash(spec: ExperimentSpec) -> str:
    """Calcula o hash criptográfico SHA-256 canônico de uma especificação de experimento.

    Args:
        spec (ExperimentSpec): Especificação completa do experimento.

    Returns:
        str: Digest hexadecimal de 64 caracteres representando o hash da especificação.
    """
    payload = canonical_json(spec.model_dump(mode="json")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ImprovementProposalValidator:
    """Valida propostas de melhoria contra capacidades ML e regras de especificação.

    Attributes:
        capabilities (ImprovementCapabilityCatalog): Validador de capacidades e hiperparâmetros.
        spec_validator (ExperimentSpecValidator): Validador de consistência da especificação.
    """

    def __init__(
        self,
        *,
        capabilities: ImprovementCapabilityCatalog,
        spec_validator: ExperimentSpecValidator,
    ) -> None:
        """Inicializa o validador de propostas com catálogo e validador de especificações.

        Args:
            capabilities (ImprovementCapabilityCatalog): Catálogo de capacidades suportadas.
            spec_validator (ExperimentSpecValidator): Validador de especificações de experimentos.
        """
        self.capabilities = capabilities
        self.spec_validator = spec_validator

    def build_spec(
        self,
        recommendation: SupervisedImprovementRecommendation,
        *,
        source: ExperimentSpec,
        proposed_name: str,
    ) -> ExperimentSpec:
        """Constrói e valida uma nova especificação mesclando a base com a recomendação.

        Args:
            recommendation (SupervisedImprovementRecommendation): Recomendação proposta.
            source (ExperimentSpec): Especificação do experimento original de origem.
            proposed_name (str): Nome sugerido para a nova especificação.

        Returns:
            ExperimentSpec: Nova especificação validada pronta para persistência/execução.

        Raises:
            ExperimentSpecValidationError: Se a recomendação violar regras de ML ou negócio.
        """
        self.capabilities.validate_parameters(
            task=source.task,
            models=recommendation.models,
            preprocessing=recommendation.preprocessing,
        )
        proposed = source.model_copy(
            update={
                "name": proposed_name.strip()[:200],
                "models": recommendation.models,
                "preprocessing": recommendation.preprocessing,
                "validation": recommendation.validation,
                "metrics": recommendation.metrics,
                "tuning": recommendation.tuning,
            }
        )
        self.spec_validator.validate(proposed)
        return proposed

    @staticmethod
    def changes(
        source: ExperimentSpec,
        proposed: ExperimentSpec,
    ) -> list[ConfigurationChange]:
        """Calcula as diferenças nos campos mutáveis entre duas especificações.

        Args:
            source (ExperimentSpec): Especificação original antes das melhorias.
            proposed (ExperimentSpec): Nova especificação proposta com alterações.

        Returns:
            list[ConfigurationChange]: Lista das alterações detectadas nos campos mutáveis.
        """
        before = source.model_dump(mode="json")
        after = proposed.model_dump(mode="json")
        return [
            ConfigurationChange(
                path=field,
                before=json.dumps(before[field], ensure_ascii=False, sort_keys=True),
                after=json.dumps(after[field], ensure_ascii=False, sort_keys=True),
            )
            for field in MUTABLE_FIELDS
            if before[field] != after[field]
        ]
