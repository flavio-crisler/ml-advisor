import type { components } from "../types/api.generated";

export type DatasetSummary = components["schemas"]["DatasetSummaryResponse"];
export type DatasetDetail = components["schemas"]["DatasetDetailResponse"];
export type DatasetUpload = components["schemas"]["DatasetUploadResponse"];
export type DatasetProfile = components["schemas"]["DatasetProfile"];
export type ColumnProfile = components["schemas"]["ColumnProfile"];
export type ProfileWarning = components["schemas"]["ProfileWarning"];

type ApiErrorPayload = {
  detail?: {
    code?: string;
    message_key?: string;
  };
};

/**
 * Representa um erro disparado durante requisições à API de datasets.
 */
export class ApiError extends Error {
  /**
   * Cria uma nova instância de ApiError.
   *
   * @param code - Código de erro retornado pela API ou identificado pelo cliente.
   */
  constructor(public readonly code: string) {
    super(code);
  }
}

/**
 * Trata e desempacota o payload JSON retornado pela API de datasets.
 *
 * @param response - Resposta HTTP fetch.
 * @returns Dados convertidos para o tipo T.
 * @throws {ApiError} Se o status for diferente de 2xx.
 */
async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response
      .json()
      .catch(() => ({}))) as ApiErrorPayload;
    throw new ApiError(payload.detail?.code ?? "unknown");
  }
  return (await response.json()) as T;
}

/**
 * Lista todos os datasets cadastrados na plataforma.
 *
 * @returns Lista com os resumos de todos os datasets disponíveis.
 */
export async function listDatasets(): Promise<DatasetSummary[]> {
  return parseResponse<DatasetSummary[]>(await fetch("/api/v1/datasets"));
}

/**
 * Obtém os detalhes e perfil completo de um dataset específico.
 *
 * @param datasetId - Identificador único do dataset.
 * @returns Detalhes, perfil e sumários das colunas do dataset.
 */
export async function getDataset(datasetId: string): Promise<DatasetDetail> {
  return parseResponse<DatasetDetail>(
    await fetch(`/api/v1/datasets/${datasetId}`),
  );
}

/**
 * Realiza o upload de um arquivo de dataset com acompanhamento de progresso percentual.
 *
 * @param file - Arquivo a ser enviado (ex: CSV).
 * @param onProgress - Callback disparado a cada atualização percentual do upload (0-100).
 * @returns Dados de confirmação do upload e perfil inicial do dataset.
 * @throws {ApiError} Em caso de erro na rede ou validação do arquivo pelo backend.
 */
export function uploadDataset(
  file: File,
  onProgress: (percentage: number) => void,
): Promise<DatasetUpload> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const body = new FormData();
    body.append("file", file);

    request.open("POST", "/api/v1/datasets");
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded * 100) / event.total));
      }
    });
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300) {
        onProgress(100);
        resolve(request.response as DatasetUpload);
        return;
      }
      const payload = request.response as ApiErrorPayload | null;
      reject(new ApiError(payload?.detail?.code ?? "unknown"));
    });
    request.addEventListener("error", () => reject(new ApiError("network")));
    request.send(body);
  });
}
