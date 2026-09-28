import type { components } from "../types/api.generated";

export type CascadePolicy = components["schemas"]["CascadePolicy"];
export type CascadeRun = components["schemas"]["CascadeRunResponse"];

/**
 * Desempacota a resposta JSON de uma requisição ou dispara erro se a resposta não for OK.
 *
 * @param response - Objeto Response do fetch.
 * @returns Dados convertidos para o tipo genérico T.
 * @throws {Error} Se response.ok for falso.
 */
async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error("confidence_cascade_request_failed");
  return (await response.json()) as T;
}

/**
 * Lista todas as execuções de cascata de confiança registradas.
 *
 * @returns Lista de execuções de cascata de confiança.
 */
export async function listConfidenceCascades(): Promise<CascadeRun[]> {
  return parse<CascadeRun[]>(await fetch("/api/v1/confidence-cascades"));
}

/**
 * Cria e avalia uma nova política de cascata de confiança entre modelos.
 *
 * @param policy - Configuração da política com modelos, limiares e métricas.
 * @returns Resultado da execução da cascata com métricas comparativas.
 */
export async function createConfidenceCascade(
  policy: CascadePolicy,
): Promise<CascadeRun> {
  return parse<CascadeRun>(
    await fetch("/api/v1/confidence-cascades", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(policy),
    }),
  );
}
