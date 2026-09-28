from typing import Any, Protocol

from app.application.experiments.service import ExperimentService
from app.application.improvements.capabilities import ImprovementCapabilityCatalog
from app.application.improvements.models import (
    GeneratedSupervisedImprovementRecommendation,
    ImprovementDraftResult,
    ImprovementLanguage,
    SupervisedImprovementProposal,
    SupervisedImprovementRecommendation,
)
from app.application.improvements.prompts import ImprovementPromptBuilder
from app.application.improvements.validation import (
    ImprovementProposalValidator,
    experiment_spec_hash,
)
from app.domain.exceptions import (
    EntityNotFoundError,
    ExperimentSpecValidationError,
)
from app.domain.experiment.entities import Experiment, ExperimentMode, ExperimentStrategy
from app.domain.experiment.repositories import ModelRunRepository
from app.domain.experiment.spec import ExperimentSpec
from app.llm.base import LLMProvider


class ImprovementDraftPersistence(Protocol):
    """Protocolo de persistência atômica para experimentos e estratégias associadas."""

    def add_experiment_with_strategy(
        self,
        experiment: Experiment,
        strategy: ExperimentStrategy,
    ) -> Experiment:
        """Persiste um experimento com sua estratégia em uma transação atômica.

        Args:
            experiment (Experiment): Entidade do experimento a ser persistido.
            strategy (ExperimentStrategy): Estratégia de linhagem e proveniência associada.

        Returns:
            Experiment: Instância persistida do experimento.
        """
        ...


class ImprovementProposalService:
    """Coordena a proposição e criação de experimentos derivados a partir de LLM.

    Attributes:
        provider (LLMProvider): Provedor de inteligência artificial generativa.
        prompts (ImprovementPromptBuilder): Construtor de prompts para melhorias.
        experiments (ExperimentService): Serviço de domínio para gerência de experimentos.
        model_runs (ModelRunRepository): Repositório de execuções de modelos.
        capabilities (ImprovementCapabilityCatalog): Catálogo de capacidades suportadas.
        validator (ImprovementProposalValidator): Validador de propostas de melhoria.
        persistence (ImprovementDraftPersistence): Adaptador de persistência atômica.
        random_seed (int): Semente pseudoaleatória para reprodutibilidade.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        prompts: ImprovementPromptBuilder,
        experiments: ExperimentService,
        model_runs: ModelRunRepository,
        capabilities: ImprovementCapabilityCatalog,
        validator: ImprovementProposalValidator,
        persistence: ImprovementDraftPersistence,
        random_seed: int,
    ) -> None:
        """Inicializa o serviço de sugestão de melhorias supervisionadas.

        Args:
            provider (LLMProvider): Provedor LLM configurado.
            prompts (ImprovementPromptBuilder): Construtor de prompts de melhoria.
            experiments (ExperimentService): Serviço de manipulação de experimentos.
            model_runs (ModelRunRepository): Repositório de execuções de modelos.
            capabilities (ImprovementCapabilityCatalog): Catálogo de capacidades.
            validator (ImprovementProposalValidator): Validador de propostas e regras.
            persistence (ImprovementDraftPersistence): Mecanismo de persistência de rascunhos.
            random_seed (int): Semente pseudoaleatória padrão.
        """
        self.provider = provider
        self.prompts = prompts
        self.experiments = experiments
        self.model_runs = model_runs
        self.capabilities = capabilities
        self.validator = validator
        self.persistence = persistence
        self.random_seed = random_seed

    def propose(
        self,
        *,
        language: ImprovementLanguage,
        source_experiment_id: str,
        source_model_run_id: str | None,
        context: dict[str, Any],
    ) -> SupervisedImprovementProposal:
        """Gera uma proposta estruturada de melhorias para um experimento concluído.

        Args:
            language (ImprovementLanguage): Idioma das justificativas ('pt-BR' ou 'en-US').
            source_experiment_id (str): Identificador do experimento de origem.
            source_model_run_id (str | None): Identificador da execução específica, opcional.
            context (dict[str, Any]): Contexto agregado dos resultados.

        Returns:
            SupervisedImprovementProposal: Proposta com diferenças e nova especificação.

        Raises:
            ExperimentSpecValidationError: Se a IA não propuser alterações efetivas.
        """
        source_experiment, source_spec, source_metrics = self._source(
            source_experiment_id,
            source_model_run_id,
        )
        request = self.prompts.build(
            language=language,
            source_spec=source_spec,
            source_metrics=source_metrics,
            context=context,
            capabilities=self.capabilities.for_task(source_spec.task),
            seed=self.random_seed,
        )
        generated = self.provider.generate_structured(
            request,
            GeneratedSupervisedImprovementRecommendation,
        )
        recommendation = SupervisedImprovementRecommendation(
            **generated.model_dump(mode="python"),
        )
        proposed = self.validator.build_spec(
            recommendation,
            source=source_spec,
            proposed_name=self._draft_name(source_experiment),
        )
        changes = self.validator.changes(source_spec, proposed)
        if not changes:
            raise ExperimentSpecValidationError(["ai_proposal_has_no_changes"])
        return SupervisedImprovementProposal(
            source_experiment_id=source_experiment.id,
            source_model_run_id=source_model_run_id,
            source_spec_hash=experiment_spec_hash(source_spec),
            recommendation=recommendation,
            proposed_spec=proposed,
            changes=changes,
            provider_id=self.provider.id,
            is_local=self.provider.capabilities.is_local,
        )

    def create_draft(
        self,
        *,
        source_experiment_id: str,
        source_model_run_id: str | None,
        source_spec_hash: str,
        provider_id: str,
        recommendation: SupervisedImprovementRecommendation,
    ) -> ImprovementDraftResult:
        """Cria e persiste um novo experimento rascunho com base na recomendação aceita.

        Args:
            source_experiment_id (str): Identificador do experimento base.
            source_model_run_id (str | None): Execução de modelo associada de referência.
            source_spec_hash (str): Hash esperado da especificação original para concorrência.
            provider_id (str): Identificador do provedor que gerou a recomendação.
            recommendation (SupervisedImprovementRecommendation): Recomendação validada.

        Returns:
            ImprovementDraftResult: Resultado contendo identificador do rascunho criado.

        Raises:
            ExperimentSpecValidationError: Se o hash diferir ou houver conflito de provedor.
        """
        if provider_id != self.provider.id:
            raise ExperimentSpecValidationError(["improvement_provider_changed"])
        source_experiment, source_spec, _ = self._source(
            source_experiment_id,
            source_model_run_id,
        )
        if experiment_spec_hash(source_spec) != source_spec_hash:
            raise ExperimentSpecValidationError(["improvement_source_spec_changed"])
        proposed = self.validator.build_spec(
            recommendation,
            source=source_spec,
            proposed_name=self._draft_name(source_experiment),
        )
        self.experiments.validate_specification(proposed)
        draft = Experiment.create(
            name=proposed.name,
            mode=ExperimentMode.ASSISTED,
            dataset_id=proposed.dataset_id,
            objective=proposed.objective,
            spec_version=proposed.spec_version,
            spec=proposed.model_dump(mode="json"),
            description=recommendation.summary,
            parent_experiment_id=source_experiment.id,
        )
        strategy = ExperimentStrategy.create(
            experiment_id=draft.id,
            source="assisted",
            provider_id=self.provider.id,
            strategy={
                "kind": "ai_improvement",
                "schema_version": recommendation.schema_version,
                "source_experiment_id": source_experiment.id,
                "source_model_run_id": source_model_run_id,
                "source_spec_hash": source_spec_hash,
                "recommendation": recommendation.model_dump(mode="json"),
                "proposed_spec": proposed.model_dump(mode="json"),
                "changes": [
                    item.model_dump(mode="json")
                    for item in self.validator.changes(source_spec, proposed)
                ],
            },
        )
        persisted = self.persistence.add_experiment_with_strategy(draft, strategy)
        return ImprovementDraftResult(
            experiment_id=persisted.id,
            parent_experiment_id=source_experiment.id,
            name=persisted.name,
        )

    def _source(
        self,
        experiment_id: str,
        model_run_id: str | None,
    ) -> tuple[Experiment, ExperimentSpec, dict[str, Any] | None]:
        """Carrega e valida o experimento de origem e suas métricas de execução.

        Args:
            experiment_id (str): Identificador do experimento.
            model_run_id (str | None): Identificador da execução de modelo, se especificado.

        Returns:
            tuple[Experiment, ExperimentSpec, dict[str, Any] | None]: Tupla com experimento,
                especificação e payload de métricas.

        Raises:
            EntityNotFoundError: Se a execução especificada não for encontrada.
            ExperimentSpecValidationError: Se a execução não pertencer ao experimento de origem.
        """
        experiment = self.experiments.get(experiment_id)
        specification = ExperimentSpec.model_validate(experiment.spec)
        metrics = None
        if model_run_id is not None:
            model_run = self.model_runs.get(model_run_id)
            if model_run is None:
                raise EntityNotFoundError(f"Model run {model_run_id} was not found")
            if model_run.experiment_id != experiment_id:
                raise ExperimentSpecValidationError(
                    ["improvement_run_does_not_belong_to_source_experiment"]
                )
            metrics = model_run.metrics
        return experiment, specification, metrics

    @staticmethod
    def _draft_name(source: Experiment) -> str:
        """Gera um nome representativo para o experimento rascunho derivado da proposta.

        Args:
            source (Experiment): Experimento de origem.

        Returns:
            str: Nome gerado concatenando sufixo indicativo de proposta de IA.
        """
        return f"{source.name} · AI proposal"[:200]
