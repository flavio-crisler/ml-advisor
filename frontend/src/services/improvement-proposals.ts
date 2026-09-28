import type { components } from "../types/api.generated";

export type ImprovementProposalRequest =
  components["schemas"]["ImprovementProposalRequest"];
export type ImprovementDraftRequest =
  components["schemas"]["ImprovementDraftRequest"];
export type ImprovementDraftResult =
  components["schemas"]["ImprovementDraftResult"];
export type SupervisedImprovementProposal =
  components["schemas"]["SupervisedImprovementProposal"];

/**
 * Converte resposta HTTP fetch em JSON do tipo T ou lança erro se status não for de sucesso.
 *
 * @param response - Objeto Response do fetch.
 * @returns Objeto desempacotado no tipo genérico T.
 * @throws {Error} Se o status for diferente de 2xx.
 */
async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error("improvement_proposal_request_failed");
  return (await response.json()) as T;
}

/**
 * Solicita à IA uma proposta de melhorias iterativas para um experimento supervisionado.
 *
 * @param request - Dados do experimento de origem e idioma desejado.
 * @returns Proposta estruturada contendo modificações recomendadas e justificativas.
 */
export async function proposeSupervisedImprovement(
  request: ImprovementProposalRequest,
): Promise<SupervisedImprovementProposal> {
  return parse<SupervisedImprovementProposal>(
    await fetch("/api/v1/improvement-proposals/supervised", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}

/**
 * Cria um rascunho de experimento a partir das melhorias propostas pela IA.
 *
 * @param request - Identificador da proposta e seleção de melhorias a aplicar.
 * @returns Rascunho com especificação técnica do novo experimento criado.
 */
export async function createImprovementDraft(
  request: ImprovementDraftRequest,
): Promise<ImprovementDraftResult> {
  return parse<ImprovementDraftResult>(
    await fetch("/api/v1/improvement-proposals/supervised/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}
