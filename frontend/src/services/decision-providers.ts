import type { components } from "../types/api.generated";

export type DecisionProviderSummary =
  components["schemas"]["DecisionProviderSummary"];
export type ProviderHealth = components["schemas"]["ProviderHealth"];

/**
 * Lista todos os provedores de decisão (modelos externos/LLMs) disponíveis no sistema.
 *
 * @returns Lista de sumários dos provedores configurados.
 * @throws {Error} Se a API não responder com sucesso.
 */
export async function listDecisionProviders(): Promise<
  DecisionProviderSummary[]
> {
  const response = await fetch("/api/v1/decision-providers");
  if (!response.ok) throw new Error("decision_providers_unavailable");
  return (await response.json()) as DecisionProviderSummary[];
}

/**
 * Testa a conectividade com o provedor de decisão configurado (ex: OpenRouter).
 *
 * @returns Status de integridade e detalhes da conexão do provedor.
 * @throws {Error} Se o teste de conexão falhar.
 */
export async function testDecisionProvider(): Promise<ProviderHealth> {
  const response = await fetch("/api/v1/decision-providers/test", {
    method: "POST",
  });
  if (!response.ok) throw new Error("decision_provider_connection_failed");
  return (await response.json()) as ProviderHealth;
}
