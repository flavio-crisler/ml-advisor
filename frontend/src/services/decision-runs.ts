import type { components } from "../types/api.generated";

export type DecisionRun = components["schemas"]["DecisionRunResponse"];
export type DecisionJob = components["schemas"]["DecisionJobResponse"];
export type CalibrationRun = components["schemas"]["CalibrationResponse"];
export type DecisionRunCreate =
  components["schemas"]["DecisionRunCreateRequest"];

export type ComparedApproach =
  components["schemas"]["ComparedApproachResponse"];
export type ComparedModel = components["schemas"]["ComparedModelResponse"];
export type DecisionComparison =
  components["schemas"]["DecisionComparisonResponse"];

/**
 * Trata e desempacota a resposta JSON de chamadas de benchmarking de decisão.
 *
 * @param response - Resposta HTTP fetch.
 * @returns Objeto desempacotado no tipo T.
 * @throws {Error} Se a resposta HTTP for diferente de 2xx.
 */
async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error("decision_run_request_failed");
  return (await response.json()) as T;
}

/**
 * Lista todas as execuções de benchmark de decisão persistidas.
 *
 * @returns Lista de execuções de benchmark.
 */
export async function listDecisionRuns(): Promise<DecisionRun[]> {
  return parse<DecisionRun[]>(await fetch("/api/v1/decision-runs"));
}

/**
 * Submete um novo benchmark de decisão (zero-shot/few-shot com LLMs).
 *
 * @param request - Configuração do benchmark com dataset, provedor, prompt e colunas.
 * @returns Resposta com identificador da execução e do job em segundo plano.
 */
export async function createDecisionBenchmark(
  request: DecisionRunCreate,
): Promise<DecisionJob> {
  return parse<DecisionJob>(
    await fetch("/api/v1/decision-benchmarks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}

/**
 * Obtém os detalhes completos de uma execução de benchmark de decisão.
 *
 * @param runId - Identificador único da execução.
 * @returns Dados da execução incluindo métricas e predições.
 */
export async function getDecisionRun(runId: string): Promise<DecisionRun> {
  return parse<DecisionRun>(
    await fetch(`/api/v1/decision-benchmarks/${runId}`),
  );
}

/**
 * Obtém os dados de calibração de probabilidade de um benchmark de decisão.
 *
 * @param runId - Identificador único da execução do benchmark.
 * @returns Métricas de calibração (ECE, Brier score, curvas de calibração).
 */
export async function getCalibration(runId: string): Promise<CalibrationRun> {
  return parse<CalibrationRun>(
    await fetch(`/api/v1/decision-benchmarks/${runId}/calibration`),
  );
}

/**
 * Compara os resultados de um benchmark de decisão com um experimento supervisionado.
 *
 * @param runId - Identificador da execução do benchmark.
 * @param experimentId - Identificador do experimento supervisionado a comparar.
 * @returns Métricas comparativas lado a lado entre os modelos supervisionados e LLMs.
 */
export async function getDecisionComparison(
  runId: string,
  experimentId: string,
): Promise<DecisionComparison> {
  return parse<DecisionComparison>(
    await fetch(`/api/v1/decision-benchmarks/${runId}/compare/${experimentId}`),
  );
}
