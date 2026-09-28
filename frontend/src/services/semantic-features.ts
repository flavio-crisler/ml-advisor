import type { components } from "../types/api.generated";
import type { ExplanationResult, ExplanationLanguage } from "./explanations";

export type SemanticProposalRequest =
  components["schemas"]["SemanticProposalRequest"];
export type SemanticProposalResponse =
  components["schemas"]["SemanticProposalResponse"];
export type SemanticGenerateRequest =
  components["schemas"]["SemanticGenerateRequest"];
export type SemanticJob = components["schemas"]["SemanticJobResponse"];
export type SemanticFeatureSet =
  components["schemas"]["SemanticFeatureSetResponse"];
export type SemanticQuestion = components["schemas"]["SemanticQuestionSpec"];

/**
 * Trata resposta HTTP fetch e converte em JSON ou lança erro genérico em caso de falha.
 *
 * @param response - Objeto Response do fetch.
 * @returns Objeto desempacotado no tipo T.
 * @throws {Error} Se o status for diferente de 2xx.
 */
async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error("semantic_feature_request_failed");
  return (await response.json()) as T;
}

/**
 * Propõe novas features semânticas a partir de colunas de texto de um dataset.
 *
 * @param request - Parâmetros da proposta com dataset, objetivo e colunas de texto.
 * @returns Proposta contendo perguntas estruturadas e estimativas de processamento.
 */
export async function proposeSemanticFeatures(
  request: SemanticProposalRequest,
): Promise<SemanticProposalResponse> {
  return parse<SemanticProposalResponse>(
    await fetch("/api/v1/semantic-features/propose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}

/**
 * Submete a geração e materialização de features semânticas em lote via LLM.
 *
 * @param request - Configuração da geração com proposta aprovada e provedor/modelo.
 * @returns Informações do conjunto criado e do job de processamento assíncrono.
 */
export async function generateSemanticFeatures(
  request: SemanticGenerateRequest,
): Promise<SemanticJob> {
  return parse<SemanticJob>(
    await fetch("/api/v1/semantic-features/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}

/**
 * Lista todos os conjuntos de features semânticas registrados.
 *
 * @returns Lista de conjuntos de features semânticas.
 */
export async function listSemanticFeatureSets(): Promise<SemanticFeatureSet[]> {
  return parse<SemanticFeatureSet[]>(await fetch("/api/v1/semantic-features"));
}

/**
 * Obtém os detalhes completos e métricas de impacto de um conjunto de features semânticas.
 *
 * @param featureSetId - Identificador único do conjunto de features semânticas.
 * @returns Metadados do conjunto, status e análise de impacto comparativa.
 */
export async function getSemanticFeatureSet(
  featureSetId: string,
): Promise<SemanticFeatureSet> {
  return parse<SemanticFeatureSet>(
    await fetch(`/api/v1/semantic-features/${featureSetId}`),
  );
}

/**
 * Cancela a execução em segundo plano da geração de features semânticas.
 *
 * @param featureSetId - Identificador único do conjunto a cancelar.
 * @returns Resposta com o status atualizado do job cancelado.
 */
export async function cancelSemanticFeatures(
  featureSetId: string,
): Promise<SemanticJob> {
  return parse<SemanticJob>(
    await fetch(`/api/v1/semantic-features/${featureSetId}/cancel`, {
      method: "POST",
    }),
  );
}

/**
 * Solicita uma explicação contextualizada por IA do impacto das features semânticas nos modelos.
 *
 * @param featureSetId - Identificador do conjunto de features semânticas.
 * @param language - Idioma selecionado para o relatório explicativo.
 * @returns Explicação gerada com análise comparativa de ganhos de performance.
 */
export async function explainSemanticImpact(
  featureSetId: string,
  language: ExplanationLanguage,
): Promise<ExplanationResult> {
  return parse<ExplanationResult>(
    await fetch(`/api/v1/semantic-features/${featureSetId}/explanation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language }),
    }),
  );
}
