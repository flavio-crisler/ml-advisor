from dataclasses import dataclass

from app.application.advisor.models import AdvisorResult
from app.application.advisor.service import AdvisorService
from app.application.experiments.service import ExperimentService
from app.application.jobs.executor import JobSnapshot
from app.application.jobs.service import TrainingSubmissionService
from app.domain.experiment.entities import Experiment, ExperimentMode, ExperimentStrategy
from app.domain.experiment.repositories import ExperimentStrategyRepository


@dataclass(frozen=True, slots=True)
class AutoWorkflowResult:
    """Encapsula as entidades geradas na execução de ponta a ponta do modo Auto.

    Attributes:
        advisor (AdvisorResult): Resultado gerado pelo serviço de recomendação do Advisor.
        experiment (Experiment): Experimento persistido e configurado a partir da recomendação.
        job (JobSnapshot): Snapshot do job assíncrono de treinamento enfileirado.
        strategy (ExperimentStrategy): Registro de estratégia e proveniência associado.
    """

    advisor: AdvisorResult
    experiment: Experiment
    job: JobSnapshot
    strategy: ExperimentStrategy


class AutoWorkflowService:
    """Orquestra o fluxo completo do modo Auto: recomendação, criação e treinamento.

    Attributes:
        advisor (AdvisorService): Serviço gerador de recomendações de arquitetura de ML.
        experiments (ExperimentService): Serviço de domínio para criação de experimentos.
        strategies (ExperimentStrategyRepository): Repositório para registro de proveniência.
        training (TrainingSubmissionService): Serviço de submissão de jobs de treinamento.
    """

    def __init__(
        self,
        *,
        advisor: AdvisorService,
        experiments: ExperimentService,
        strategies: ExperimentStrategyRepository,
        training: TrainingSubmissionService,
    ) -> None:
        """Inicializa o orquestrador do modo Auto com os serviços dependentes.

        Args:
            advisor (AdvisorService): Serviço de recomendação do Advisor.
            experiments (ExperimentService): Serviço de persistência de experimentos.
            strategies (ExperimentStrategyRepository): Repositório de estratégias de IA.
            training (TrainingSubmissionService): Serviço de despacho de jobs de treino.
        """
        self.advisor = advisor
        self.experiments = experiments
        self.strategies = strategies
        self.training = training

    def execute(
        self,
        *,
        dataset_id: str,
        objective: str,
        language: str,
        name: str | None = None,
    ) -> AutoWorkflowResult:
        """Executa a cadeia completa: consulta Advisor, cria experimento e despacha job.

        Args:
            dataset_id (str): Identificador do dataset de treinamento.
            objective (str): Descrição do problema de negócio em linguagem natural.
            language (str): Código de idioma para recomendações ('pt-BR' ou 'en-US').
            name (str | None): Nome customizado opcional para o experimento criado.

        Returns:
            AutoWorkflowResult: Resultado consolidado com recomendação, experimento e job.
        """
        advisor_result = self.advisor.recommend(
            dataset_id=dataset_id,
            objective=objective,
            language=language,
            name=name,
        )
        experiment = self.experiments.create(
            advisor_result.proposal,
            mode=ExperimentMode.AUTO,
            description=advisor_result.recommendation.explanation,
        )
        strategy = self.strategies.add(
            ExperimentStrategy.create(
                experiment_id=experiment.id,
                source="auto",
                provider_id=advisor_result.provider_id,
                strategy=advisor_result.recommendation.model_dump(mode="json"),
            )
        )
        job = self.training.submit(experiment.id)
        return AutoWorkflowResult(
            advisor=advisor_result,
            experiment=self.experiments.get(experiment.id),
            job=job,
            strategy=strategy,
        )
