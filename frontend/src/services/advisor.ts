import type { components } from "../types/api.generated";

export type AdvisorResult = components["schemas"]["AdvisorResult"];
export type AdvisorRequest =
  components["schemas"]["AdvisorRecommendationRequest"];
export type AutoRunResult = components["schemas"]["AutoRunResponse"];
export type AutoRunRequest = components["schemas"]["AutoRunRequest"];

type ApiErrorPayload = {
  detail?: {
    code?: string;
    message_key?: string;
    issues?: string[];
  };
};

/**
 * Erro customizado disparado em falhas de chamadas à API do Advisor e Auto ML.
 */
export class AdvisorApiError extends Error {
  /**
   * Inicializa o erro da API do Advisor.
   *
   * @param code - Código de erro retornado pela API.
   * @param messageKey - Chave i18n correspondente à mensagem de erro.
   * @param issues - Lista de problemas detalhados ou violações de validação.
   */
  constructor(
    public readonly code: string,
    public readonly messageKey?: string,
    public readonly issues: string[] = [],
  ) {
    super(code);
  }
}

/**
 * Solicita recomendações de modelagem e pré-processamento ao Advisor.
 *
 * @param request - Parâmetros da recomendação incluindo dataset, objetivo e restrições.
 * @returns Resultado detalhado contendo a recomendação e plano de experimento.
 * @throws {AdvisorApiError} Quando o backend retorna erro HTTP não-2xx.
 */
export async function recommendExperiment(
  request: AdvisorRequest,
): Promise<AdvisorResult> {
  const response = await fetch("/api/v1/advisor/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const payload = (await response
      .json()
      .catch(() => ({}))) as ApiErrorPayload;
    throw new AdvisorApiError(
      payload.detail?.code ?? "unknown",
      payload.detail?.message_key,
      payload.detail?.issues,
    );
  }
  return (await response.json()) as AdvisorResult;
}

/**
 * Executa o fluxo de Auto ML do Advisor gerando recomendação e iniciando o treino.
 *
 * @param request - Configuração da execução automática.
 * @returns Resumo da execução automática contendo o ID do experimento criado.
 * @throws {AdvisorApiError} Quando ocorre falha na requisição ou no treinamento.
 */
export async function runAutoExperiment(
  request: AutoRunRequest,
): Promise<AutoRunResult> {
  const response = await fetch("/api/v1/auto/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const payload = (await response
      .json()
      .catch(() => ({}))) as ApiErrorPayload;
    throw new AdvisorApiError(
      payload.detail?.code ?? "unknown",
      payload.detail?.message_key,
      payload.detail?.issues,
    );
  }
  return (await response.json()) as AutoRunResult;
}
