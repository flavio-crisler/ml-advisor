import type { components } from "../types/api.generated";

export type ExplanationRequest = components["schemas"]["ExplanationRequest"];
export type ExplanationScope = ExplanationRequest["scope"];
export type ExplanationLanguage = ExplanationRequest["language"];
export type ExplanationAvailability =
  components["schemas"]["ExplanationAvailability"];
export type ResultExplanation = components["schemas"]["ResultExplanation"];
export type ChartExplanation = components["schemas"]["ChartExplanation"];
export type ExplanationResult = components["schemas"]["ExplanationResult"];

/**
 * Trata respostas JSON do serviço de explicações ou dispara erro se a chamada falhar.
 *
 * @param response - Objeto Response do fetch.
 * @returns Objeto desempacotado no formato T.
 * @throws {Error} Se a resposta HTTP for não-OK.
 */
async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error("explanation_request_failed");
  return (await response.json()) as T;
}

/**
 * Consulta a disponibilidade e status do serviço de explicações e provedor de LLM configurado.
 *
 * @returns Status de disponibilidade do serviço de explicações.
 */
export async function getExplanationAvailability(): Promise<ExplanationAvailability> {
  return parse<ExplanationAvailability>(
    await fetch("/api/v1/explanations/status"),
  );
}

/**
 * Solicita uma explicação contextualizada gerada por IA para um resultado, gráfico ou comparação.
 *
 * @param request - Parâmetros da explicação contendo escopo, idioma e dados de contexto.
 * @returns Resultado da explicação com o texto gerado e metadados.
 */
export async function explainResult(
  request: ExplanationRequest,
): Promise<ExplanationResult> {
  return parse<ExplanationResult>(
    await fetch("/api/v1/explanations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}
