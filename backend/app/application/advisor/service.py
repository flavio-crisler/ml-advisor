import logging

from app.application.advisor.errors import AdvisorRecommendationError
from app.application.advisor.models import AdvisorResult, RecommendationSpec
from app.application.advisor.prompts import AdvisorPromptBuilder
from app.application.advisor.validation import AdvisorRecommendationValidator
from app.application.datasets.service import DatasetService
from app.llm.base import LLMProvider
from app.llm.errors import LLMResponseError
from app.ml.registries.catalog import RegistryCatalog

logger = logging.getLogger(__name__)


class AdvisorService:
    """Serviço de orquestração do Advisor de ML com LLM e validação automática.

    Attributes:
        datasets (DatasetService): Serviço de gerenciamento de datasets e perfis.
        provider (LLMProvider): Provedor de modelos de linguagem para geração estruturada.
        catalog (RegistryCatalog): Catálogo de componentes e algoritmos de ML.
        validator (AdvisorRecommendationValidator): Validador de propostas de experimentos.
        prompts (AdvisorPromptBuilder): Construtor de prompts do Advisor por idioma.
        send_sample (bool): Indica se uma amostra tabular é enviada ao provedor LLM.
        sample_size (int): Quantidade máxima de linhas enviadas na amostra.
        random_seed (int): Semente de aleatoriedade para reprodutibilidade.
    """

    def __init__(
        self,
        *,
        datasets: DatasetService,
        provider: LLMProvider,
        catalog: RegistryCatalog,
        validator: AdvisorRecommendationValidator,
        prompts: AdvisorPromptBuilder,
        send_sample: bool,
        sample_size: int,
        random_seed: int,
    ) -> None:
        """Inicializa o serviço do Advisor com suas dependências e parâmetros de amostragem.

        Args:
            datasets: Serviço de acesso aos datasets e seus perfis.
            provider: Provedor de LLM configurado.
            catalog: Catálogo de modelos, pré-processadores e métricas.
            validator: Validador de conformidade da recomendação gerada.
            prompts: Construtor de prompts localizados.
            send_sample: Flag que define se amostras de dados acompanham o prompt.
            sample_size: Quantidade limite de linhas de amostra enviadas.
            random_seed: Semente pseudoaleatória para a inferência.
        """
        self.datasets = datasets
        self.provider = provider
        self.catalog = catalog
        self.validator = validator
        self.prompts = prompts
        self.send_sample = send_sample
        self.sample_size = sample_size
        self.random_seed = random_seed

    def recommend(
        self,
        *,
        dataset_id: str,
        objective: str,
        language: str,
        name: str | None = None,
    ) -> AdvisorResult:
        """Gera uma recomendação de modelagem com suporte a repetição guiada por feedback.

        Args:
            dataset_id: Identificador do dataset no workspace.
            objective: Descrição textual do objetivo de negócio do modelo.
            language: Código do idioma para o prompt ('pt-BR' ou 'en-US').
            name: Nome opcional customizado para a proposta gerada.

        Returns:
            AdvisorResult contendo a recomendação e a proposta validada de ExperimentSpec.

        Raises:
            LLMResponseError: Se o provedor falhar em produzir JSON estruturado válido após retry.
            AdvisorRecommendationError: Se a recomendação persistir inválida após a correção.
        """
        dataset = self.datasets.get_dataset(dataset_id)
        profile = self.datasets.get_profile(dataset_id)
        request = self.prompts.build(
            objective=objective,
            profile=profile,
            catalog=self.catalog,
            language=language,
            send_sample=self.send_sample,
            sample_size=self.sample_size,
            random_seed=self.random_seed,
        )
        for attempt in range(2):
            try:
                recommendation = self.provider.generate_structured(request, RecommendationSpec)
            except LLMResponseError as exc:
                logger.warning(
                    "Advisor attempt %d for dataset %s failed with LLMResponseError: %s",
                    attempt,
                    dataset_id,
                    exc,
                )
                if attempt == 1:
                    raise
                request = self.prompts.with_validation_feedback(
                    request, ("invalid_structured_response",)
                )
                continue
            try:
                proposal = self.validator.to_proposal(
                    recommendation,
                    dataset=dataset,
                    profile=profile,
                    objective=objective,
                    name=name,
                )
                logger.info(
                    "Advisor successfully generated and validated proposal '%s' on attempt %d",
                    proposal.name,
                    attempt,
                )
                break
            except AdvisorRecommendationError as exc:
                logger.warning(
                    "Advisor attempt %d for dataset %s failed validation with issues: %s",
                    attempt,
                    dataset_id,
                    exc.issues,
                )
                if attempt == 1:
                    logger.error(
                        "Advisor recommendation failed after 2 attempts for dataset %s. "
                        "Final issues: %s",
                        dataset_id,
                        exc.issues,
                    )
                    raise
                request = self.prompts.with_validation_feedback(request, exc.issues)
        sample_rows = min(len(profile.preview), self.sample_size) if self.send_sample else 0
        return AdvisorResult(
            recommendation=recommendation,
            proposal=proposal,
            provider_id=self.provider.id,
            sample_sent=self.send_sample,
            sample_rows=sample_rows,
        )
