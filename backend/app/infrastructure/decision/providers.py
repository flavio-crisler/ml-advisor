import logging
import math
import time
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import ValidationError

from app.application.decisions.errors import (
    DecisionProviderConfigurationError,
    DecisionProviderError,
    DecisionProviderResponseError,
)
from app.application.decisions.registry import DecisionProviderRuntimeConfig
from app.domain.decision.provider import (
    DecisionProviderCapabilities,
    ProviderHealth,
)
from app.domain.decision.questions import (
    ChoiceQuestionSpec,
    DecisionQuestionSpec,
    NoulQuestionSpec,
    ScoreQuestionSpec,
)
from app.domain.decision.responses import (
    ChoiceDecisionAnswer,
    DecisionAnswer,
    DecisionResponse,
    DecisionUsage,
    NoulDecisionAnswer,
    ScoreDecisionAnswer,
)
from app.domain.decision.specifications import DecisionState
from app.infrastructure.decision.http import DecisionHttpClient, RetryPolicy

logger = logging.getLogger(__name__)


class _BaseJevProvider:
    """Classe base para provedores de decisão compatíveis com o protocolo Jev.

    Attributes:
        id (str): Identificador do provedor.
        accepted_config_ids (set[str]): Identificadores aceitos na configuração.
        default_base_url (str): URL base padrão do serviço.
        capabilities (DecisionProviderCapabilities): Capacidades declaradas do provedor.
        config (DecisionProviderRuntimeConfig): Configurações de tempo de execução.
        base_url (str): URL base normalizada da API.
        http (DecisionHttpClient): Cliente HTTP configurado para requisições.
    """

    id: ClassVar[str]
    accepted_config_ids: ClassVar[set[str]]
    default_base_url: ClassVar[str]
    capabilities: ClassVar[DecisionProviderCapabilities]

    def __init__(
        self,
        config: DecisionProviderRuntimeConfig,
        http_client: DecisionHttpClient | None = None,
    ) -> None:
        """Inicializa e valida a configuração do provedor de decisões.

        Args:
            config (DecisionProviderRuntimeConfig): Configuração de execução do provedor.
            http_client (DecisionHttpClient | None): Cliente HTTP opcional para injeção.

        Raises:
            DecisionProviderConfigurationError: Se o provedor, modelo ou chave forem inválidos.
        """
        if config.provider_id not in self.accepted_config_ids:
            raise DecisionProviderConfigurationError(
                f"Provider configuration '{config.provider_id}' cannot create '{self.id}'."
            )
        if not config.model.strip():
            raise DecisionProviderConfigurationError(
                "A Decision Provider model must be configured."
            )
        if not config.api_key.get_secret_value():
            raise DecisionProviderConfigurationError(
                "A Decision Provider API key must be configured."
            )
        self.config = config
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")
        self.http = http_client or DecisionHttpClient(
            retry_policy=RetryPolicy(max_retries=config.max_retries)
        )

    def get_capabilities(self) -> DecisionProviderCapabilities:
        """Retorna as capacidades operacionais suportadas pelo provedor.

        Returns:
            DecisionProviderCapabilities: Metadados de capacidades do provedor.
        """
        return self.capabilities

    async def evaluate(
        self,
        state: DecisionState,
        questions: dict[str, DecisionQuestionSpec],
    ) -> DecisionResponse:
        """Avalia um conjunto de perguntas de decisão dado o estado de entrada.

        Args:
            state (DecisionState): Dados contextuais da linha/instância.
            questions (dict[str, DecisionQuestionSpec]): Dicionário de perguntas de decisão.

        Returns:
            DecisionResponse: Resposta estruturada com respostas normalizadas e uso.

        Raises:
            DecisionProviderResponseError: Se questions estiver vazio ou a resposta for inválida.
        """
        if not questions:
            raise DecisionProviderResponseError("At least one decision question is required.")
        payload = {
            "model": self.config.model,
            "state": state.values,
            "questions": {
                question_id: self._serialize_question(question)
                for question_id, question in questions.items()
            },
        }
        started = time.perf_counter()
        raw = await self.http.post_json(
            f"{self.base_url}/v1/systemone",
            self._headers(),
            payload,
            self.config.timeout_seconds,
            self._log_retry,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return self._normalize_response(raw, questions, latency_ms)

    async def health_check(self) -> ProviderHealth:
        """Verifica a conectividade e disponibilidade do endpoint do provedor.

        Returns:
            ProviderHealth: Status de disponibilidade e latência do provedor.
        """
        started = time.perf_counter()
        try:
            raw = await self.http.get_json(
                f"{self.base_url}/v1/models",
                self._headers(),
                self.config.timeout_seconds,
                self._log_retry,
            )
            models = raw.get("models", raw.get("data"))
            if not isinstance(models, list):
                raise DecisionProviderResponseError(
                    "Decision Provider model listing has an unexpected shape."
                )
        except DecisionProviderError:
            return ProviderHealth(
                status="unavailable",
                message_key="decisionProviders.status.connectionFailed",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return ProviderHealth(
            status="available",
            message_key="decisionProviders.status.connectionAvailable",
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _headers(self) -> dict[str, str]:
        """Gera os cabeçalhos de autenticação e conteúdo para chamadas HTTP.

        Returns:
            dict[str, str]: Cabeçalhos incluindo Authorization Bearer e Content-Type.
        """
        return {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _log_retry(self, attempt: int, delay: float) -> None:
        """Registra no log aviso sobre repetição de requisição com delay.

        Args:
            attempt (int): Número sequencial da tentativa.
            delay (float): Tempo de espera em segundos até a próxima chamada.
        """
        logger.warning(
            "Decision Provider request will be retried.",
            extra={
                "event": "decision.retry",
                "provider_id": self.id,
                "requested_model": self.config.model,
                "attempt": attempt,
                "retry_delay_seconds": delay,
            },
        )

    @staticmethod
    def _serialize_question(question: DecisionQuestionSpec) -> dict[str, Any]:
        """Serializa a especificação de uma pergunta para o payload da API.

        Args:
            question (DecisionQuestionSpec): Pergunta noul, choice ou score.

        Returns:
            dict[str, Any]: Representação serializável em JSON.

        Raises:
            DecisionProviderResponseError: Se o tipo de pergunta não for suportado.
        """
        if isinstance(question, NoulQuestionSpec):
            payload: dict[str, Any] = {
                "type": "noul",
                "instructions": question.instructions,
            }
            if question.criteria is not None:
                payload["criteria"] = question.criteria.model_dump(exclude_none=True)
            return payload
        if isinstance(question, ChoiceQuestionSpec):
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.criteria,
            }
        if isinstance(question, ScoreQuestionSpec):
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": question.criteria,
            }
        raise DecisionProviderResponseError("Unsupported decision question type.")

    def _normalize_response(
        self,
        raw: Mapping[str, Any],
        questions: dict[str, DecisionQuestionSpec],
        latency_ms: float,
    ) -> DecisionResponse:
        """Converte o retorno da API em entidade DecisionResponse tipada.

        Args:
            raw (Mapping[str, Any]): Resposta bruta da API.
            questions (dict[str, DecisionQuestionSpec]): Perguntas originais enviadas.
            latency_ms (float): Tempo total de resposta em milissegundos.

        Returns:
            DecisionResponse: Resposta estruturada e tipada.

        Raises:
            DecisionProviderResponseError: Se a resposta não coincidir com as perguntas
                ou for inválida.
        """
        raw_answers = raw.get("answers")
        if not isinstance(raw_answers, Mapping) or set(raw_answers) != set(questions):
            raise DecisionProviderResponseError(
                "Decision Provider returned answers that do not match the request."
            )
        try:
            answers = {
                question_id: self._normalize_answer(
                    raw_answers[question_id],
                    question,
                )
                for question_id, question in questions.items()
            }
            usage_raw = raw.get("usage", {})
            if not isinstance(usage_raw, Mapping):
                raise TypeError("usage must be an object")
            resolved_model = raw.get("model")
            if not isinstance(resolved_model, str) or not resolved_model:
                raise TypeError("model must be a non-empty string")
            request_id = raw.get("id")
            return DecisionResponse(
                provider_id=self.id,
                requested_model=self.config.model,
                resolved_model=resolved_model,
                answers=answers,
                usage=DecisionUsage(
                    request_count=1,
                    input_tokens=usage_raw.get("input_tokens"),
                    output_tokens=usage_raw.get("output_tokens"),
                    estimated_cost=usage_raw.get("cost"),
                ),
                latency_ms=latency_ms,
                request_id=request_id if isinstance(request_id, str) else None,
            )
        except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            raise DecisionProviderResponseError(
                "Decision Provider returned a malformed typed response."
            ) from exc

    @staticmethod
    def _normalize_answer(
        raw: object,
        question: DecisionQuestionSpec,
    ) -> DecisionAnswer:
        """Normaliza e valida o formato da resposta de acordo com o tipo da pergunta.

        Args:
            raw (object): Objeto de resposta recebido da API para a pergunta.
            question (DecisionQuestionSpec): Especificação da pergunta correspondente.

        Returns:
            DecisionAnswer: Resposta normalizada (Noul, Choice ou Score).

        Raises:
            DecisionProviderResponseError: Se o formato ou tipo recebido for incompatível.
        """
        if not isinstance(raw, Mapping) or raw.get("type") != question.type:
            raise DecisionProviderResponseError(
                "Decision Provider returned an answer with the wrong question type."
            )
        if isinstance(question, NoulQuestionSpec):
            probability_yes = float(raw["noul"])
            return NoulDecisionAnswer(
                probability_yes=probability_yes,
                confidence=abs((2 * probability_yes) - 1),
            )
        if isinstance(question, ChoiceQuestionSpec):
            probabilities = _BaseJevProvider._normalize_probabilities(
                raw["probabilities"],
                expected_keys=set(question.criteria),
            )
            return ChoiceDecisionAnswer(
                selected_choice=raw["choice"],
                choice_probabilities=probabilities,
                confidence=raw["confidence"],
            )
        if isinstance(question, ScoreQuestionSpec):
            probabilities = {
                int(key): value
                for key, value in _BaseJevProvider._normalize_probabilities(
                    raw["probabilities"],
                    expected_keys={str(index) for index in range(len(question.criteria))},
                ).items()
            }
            legend = {int(key): value for key, value in raw.get("legend", {}).items()}
            return ScoreDecisionAnswer(
                score=sum(level * value for level, value in probabilities.items()),
                score_probabilities=probabilities,
                legend=legend,
                confidence=raw["confidence"],
            )
        raise DecisionProviderResponseError("Unsupported decision question type.")

    @staticmethod
    def _normalize_probabilities(
        raw: object,
        *,
        expected_keys: set[str],
    ) -> dict[str, float]:
        """Normaliza a distribuição de probabilidades para que somem exatamente 1.0.

        Args:
            raw (object): Mapeamento de chaves para probabilidades numéricas.
            expected_keys (set[str]): Conjunto de chaves esperadas.

        Returns:
            dict[str, float]: Dicionário de probabilidades normalizadas.

        Raises:
            ValueError: Se chaves não coincidirem ou valores forem inválidos/fora do intervalo.
        """
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise ValueError("probability keys do not match the question criteria")
        probabilities = {str(key): float(value) for key, value in raw.items()}
        if any(
            not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()
        ):
            raise ValueError("probabilities must be finite values between zero and one")
        total = sum(probabilities.values())
        rounding_tolerance = max(
            1e-6,
            min(0.05, (0.005 * len(probabilities)) + 1e-9),
        )
        if total <= 0 or not math.isclose(total, 1.0, abs_tol=rounding_tolerance):
            raise ValueError("probabilities must sum to one within rounding tolerance")
        return {key: value / total for key, value in probabilities.items()}


class TypeSafeJevProvider(_BaseJevProvider):
    """Provedor de decisões Jev para a infraestrutura TypeSafe AI."""

    id = "typesafe_jev"
    accepted_config_ids = {"typesafe", "typesafe_jev"}
    default_base_url = "https://api.typesafe.ai"
    capabilities = DecisionProviderCapabilities(
        supports_noul=True,
        supports_choice=True,
        supports_score=True,
        supports_batch=False,
        supports_model_version_resolution=True,
        is_external=True,
        max_context_tokens=64_000,
    )


class OpenRouterJevProvider(_BaseJevProvider):
    """Provedor de decisões Jev para modelos roteados via OpenRouter."""

    id = "openrouter_jev"
    accepted_config_ids = {"openrouter", "openrouter_jev"}
    default_base_url = "https://openrouter.ai/api"
    capabilities = DecisionProviderCapabilities(
        supports_noul=True,
        supports_choice=True,
        supports_score=True,
        supports_batch=False,
        supports_model_version_resolution=True,
        is_external=True,
        max_context_tokens=32_000,
    )
