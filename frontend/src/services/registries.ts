import type { components } from "../types/api.generated";

export type ModelMetadata = components["schemas"]["ModelMetadata"];
export type MetricMetadata = components["schemas"]["MetricMetadata"];
export type PreprocessorMetadata =
  components["schemas"]["PreprocessorMetadata"];
export type ValidationMetadata = components["schemas"]["ValidationMetadata"];
export type TunerMetadata = components["schemas"]["TunerMetadata"];

/**
 * Catálogo consolidado contendo os metadados de todos os componentes registrados no sistema.
 */
export type RegistryCatalog = {
  models: ModelMetadata[];
  metrics: MetricMetadata[];
  preprocessors: PreprocessorMetadata[];
  validators: ValidationMetadata[];
  tuners: TunerMetadata[];
};

/**
 * Consulta um registro específico no backend pelo nome do recurso.
 *
 * @param name - Nome do recurso no backend (ex: models, metrics, tuners).
 * @returns Lista de metadados dos componentes registrados.
 * @throws {Error} Se a rota do registro retornar status diferente de 2xx.
 */
async function getRegistry<T>(name: string): Promise<T[]> {
  const response = await fetch(`/api/v1/${name}`);
  if (!response.ok) throw new Error(`registry_unavailable:${name}`);
  return (await response.json()) as T[];
}

/**
 * Carrega simultaneamente o catálogo completo de todos os registros da aplicação.
 *
 * @returns Objeto com listas de modelos, métricas, pré-processadores, validadores e tuners.
 */
export async function getRegistryCatalog(): Promise<RegistryCatalog> {
  const [models, metrics, preprocessors, validators, tuners] =
    await Promise.all([
      getRegistry<ModelMetadata>("models"),
      getRegistry<MetricMetadata>("metrics"),
      getRegistry<PreprocessorMetadata>("preprocessors"),
      getRegistry<ValidationMetadata>("validators"),
      getRegistry<TunerMetadata>("tuners"),
    ]);

  return { models, metrics, preprocessors, validators, tuners };
}

/**
 * Recupera apenas o registro de métricas de avaliação cadastradas.
 *
 * @returns Lista de metadados das métricas disponíveis.
 */
export async function getMetricRegistry(): Promise<MetricMetadata[]> {
  return getRegistry<MetricMetadata>("metrics");
}
