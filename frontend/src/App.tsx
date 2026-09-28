import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, useParams } from "react-router-dom";

import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Progress } from "./components/ui/progress";
import { StatePanel } from "./components/ui/state-panel";
import { StatusBadge } from "./components/status-badge";
import { WorkflowSteps } from "./components/workflow-steps";
import { WorkspaceLayout } from "./components/workspace-header";
import { HowItWorksButton } from "./components/how-it-works";
import { AdvisorPanel } from "./features/advisor/advisor-panel";
import {
  ApiError,
  getDataset,
  listDatasets,
  type DatasetDetail,
  type DatasetSummary,
  type ProfileWarning,
  uploadDataset,
} from "./services/datasets";
import { type Experiment, listExperiments } from "./services/experiments";
import {
  getRegistryCatalog,
  type RegistryCatalog,
} from "./services/registries";

const numberFormatter = new Intl.NumberFormat();

/**
 * Converte quantidade de bytes em string formatada com unidade legível (B, KB ou MB).
 *
 * @param bytes - Quantidade de bytes.
 * @returns String formatada com unidade.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

/**
 * Componente raiz da interface do usuário para gerenciamento de datasets e acionamento de experimentos.
 *
 * @returns Layout principal com upload de dataset, biblioteca lateral, perfil e guia de produtos.
 */
export function App() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { i18n, t } = useTranslation(["common", "datasets"]);
  const fileInput = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    document.title = t("brand.name");
  }, [i18n.language, t]);

  const datasets = useQuery({
    queryKey: ["datasets"],
    queryFn: listDatasets,
  });
  const registryCatalog = useQuery({
    queryKey: ["registry-catalog"],
    queryFn: getRegistryCatalog,
  });
  const experiments = useQuery({
    queryKey: ["experiments"],
    queryFn: listExperiments,
    enabled: !datasetId,
  });
  const detail = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: () => getDataset(datasetId!),
    enabled: Boolean(datasetId),
  });
  const upload = useMutation({
    mutationFn: (file: File) => uploadDataset(file, setProgress),
    onMutate: () => setProgress(0),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.setQueryData(["datasets", result.dataset.id], {
        dataset: result.dataset,
        profile: result.profile,
        semantic_columns: result.semantic_columns,
      } satisfies DatasetDetail);
      navigate(`/datasets/${result.dataset.id}`);
    },
  });

  const selectFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) upload.mutate(file);
    event.target.value = "";
  };

  const uploadError =
    upload.error instanceof ApiError
      ? t(`datasets:errors.${upload.error.code}`, {
          defaultValue: t("datasets:errors.unknown"),
        })
      : null;

  return (
    <WorkspaceLayout>
      <WorkflowSteps current={datasetId ? 2 : 1} />
      <section
        className={datasetId ? "" : "grid gap-6 lg:grid-cols-[1.3fr_0.7fr]"}
      >
        <div>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
                {t(
                  datasetId ? "datasets:selected.eyebrow" : "datasets:eyebrow",
                )}
              </p>
              <h1 className="max-w-4xl text-4xl font-semibold tracking-tight sm:text-5xl">
                {t(datasetId ? "datasets:selected.title" : "datasets:title")}
              </h1>
            </div>
            <HowItWorksButton screen="datasets" />
          </div>
          <p className="mt-4 max-w-2xl text-lg leading-8 text-muted">
            {t(
              datasetId
                ? "datasets:selected.description"
                : "datasets:description",
            )}
          </p>
        </div>
        {!datasetId ? (
          <Card className="flex flex-col justify-between gap-5">
            <div>
              <p className="font-semibold">{t("datasets:upload.title")}</p>
              <p className="mt-2 text-sm leading-6 text-muted">
                {t("datasets:upload.help")}
              </p>
            </div>
            <input
              ref={fileInput}
              className="sr-only"
              type="file"
              accept=".csv,text/csv"
              aria-label={t("datasets:upload.inputLabel")}
              onChange={selectFile}
            />
            <Button
              onClick={() => fileInput.current?.click()}
              disabled={upload.isPending}
            >
              {upload.isPending
                ? t("datasets:upload.processing")
                : t("datasets:upload.action")}
            </Button>
            {upload.isPending ? (
              <div className="space-y-2">
                <Progress value={progress} />
                <p className="text-xs text-muted">
                  {t("datasets:upload.progress", { progress })}
                </p>
              </div>
            ) : null}
            {upload.data?.reused ? (
              <p className="text-sm text-success">
                {t("datasets:upload.reused")}
              </p>
            ) : null}
            {uploadError ? (
              <p className="text-sm text-error">{uploadError}</p>
            ) : null}
          </Card>
        ) : null}
      </section>

      {!datasetId ? (
        <ProductGuide
          onImport={() => fileInput.current?.click()}
          onExperiments={() => navigate("/experiments")}
          onDecisions={() => navigate("/decision-lab")}
        />
      ) : null}

      {!datasetId ? (
        <RegistrySummary
          catalog={registryCatalog.data}
          loading={registryCatalog.isPending}
          error={registryCatalog.isError}
          onRetry={() => void registryCatalog.refetch()}
        />
      ) : null}

      {!datasetId ? (
        <RecentExperiments
          experiments={experiments.data ?? []}
          datasets={datasets.data ?? []}
          catalog={registryCatalog.data}
          loading={experiments.isPending}
          error={experiments.isError}
          onSelect={(id) => navigate(`/experiments/${id}`)}
          onRetry={() => void experiments.refetch()}
          onStart={() => fileInput.current?.click()}
        />
      ) : null}

      <section
        id="dataset-library"
        className="grid gap-6 xl:grid-cols-[260px_1fr]"
      >
        <DatasetLibrary
          datasets={datasets.data ?? []}
          selectedId={datasetId}
          loading={datasets.isPending}
          onSelect={(id) => navigate(`/datasets/${id}`)}
        />
        {datasetId ? (
          detail.data ? (
            <DatasetProfileView detail={detail.data} />
          ) : detail.isError ? (
            <StatePanel
              title={t("datasets:errors.load")}
              description={t("datasets:errors.loadDescription")}
              tone="error"
              actionLabel={t("datasets:errors.retry")}
              onAction={() => void detail.refetch()}
            />
          ) : (
            <StatePanel title={t("datasets:loading")} />
          )
        ) : (
          <StatePanel
            title={t("datasets:empty.title")}
            description={t("datasets:empty.description")}
            actionLabel={t("datasets:upload.action")}
            onAction={() => fileInput.current?.click()}
          />
        )}
      </section>
    </WorkspaceLayout>
  );
}

/**
 * Guia de boas-vindas com cartões introdutórios das três principais capacidades do workbench (treinar, entender e decidir).
 *
 * @param props.onImport - Ação para abrir diálogo de upload de dataset.
 * @param props.onExperiments - Navegação para lista de experimentos.
 * @param props.onDecisions - Navegação para o laboratório de decisões.
 * @returns Seção do guia do produto com opção de dispensar/exibir.
 */
function ProductGuide({
  onImport,
  onExperiments,
  onDecisions,
}: {
  onImport: () => void;
  onExperiments: () => void;
  onDecisions: () => void;
}) {
  const { t } = useTranslation("common");
  const [visible, setVisible] = useState(
    () => localStorage.getItem("mlAdvisorGuideDismissed") !== "true",
  );
  const items = [
    { key: "train", action: onImport },
    { key: "understand", action: onExperiments },
    { key: "decide", action: onDecisions },
  ] as const;

  if (!visible) {
    return (
      <Card className="flex flex-wrap items-center justify-between gap-3 py-4">
        <p className="text-sm text-muted">{t("home.guide.hidden")}</p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            localStorage.removeItem("mlAdvisorGuideDismissed");
            setVisible(true);
          }}
        >
          {t("home.guide.show")}
        </Button>
      </Card>
    );
  }

  return (
    <section aria-labelledby="product-guide-title">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="product-guide-title" className="text-xl font-semibold">
            {t("home.guide.title")}
          </h2>
          <p className="mt-1 text-sm text-muted">
            {t("home.guide.description")}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            localStorage.setItem("mlAdvisorGuideDismissed", "true");
            setVisible(false);
          }}
        >
          {t("home.guide.dismiss")}
        </Button>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {items.map(({ key, action }, index) => (
          <Card key={key} className="flex h-full flex-col items-start">
            <span className="rounded-full bg-primary/10 px-2.5 py-1 font-mono text-xs font-semibold text-primary">
              {index + 1}
            </span>
            <h3 className="mt-4 font-semibold">
              {t(`home.guide.${key}.title`)}
            </h3>
            <p className="mt-2 flex-1 text-sm leading-6 text-muted">
              {t(`home.guide.${key}.description`)}
            </p>
            <Button
              className="mt-4"
              variant="outline"
              size="sm"
              onClick={action}
            >
              {t(`home.guide.${key}.action`)}
            </Button>
          </Card>
        ))}
      </div>
    </section>
  );
}

/**
 * Exibe os experimentos mais recentes com status, melhor modelo e métrica principal.
 *
 * @param props.experiments - Lista de experimentos cadastrados.
 * @param props.datasets - Lista de resumos de datasets.
 * @param props.catalog - Catálogo de modelos e métricas.
 * @param props.loading - Indicador de carregamento.
 * @param props.error - Indicador de erro.
 * @param props.onSelect - Callback acionado ao clicar em um experimento.
 * @param props.onRetry - Callback para tentar recarregar os dados.
 * @param props.onStart - Callback para iniciar novo upload caso vazio.
 * @returns Seção de experimentos recentes.
 */
function RecentExperiments({
  experiments,
  datasets,
  catalog,
  loading,
  error,
  onSelect,
  onRetry,
  onStart,
}: {
  experiments: Experiment[];
  datasets: DatasetSummary[];
  catalog?: RegistryCatalog;
  loading: boolean;
  error: boolean;
  onSelect: (experimentId: string) => void;
  onRetry: () => void;
  onStart: () => void;
}) {
  const { t } = useTranslation("common");
  const datasetNames = new Map(
    datasets.map((dataset) => [dataset.id, dataset.name]),
  );

  return (
    <section aria-labelledby="recent-experiments-title">
      <div className="mb-4 flex items-end justify-between gap-4">
        <div>
          <h2 id="recent-experiments-title" className="text-xl font-semibold">
            {t("home.recent.title")}
          </h2>
          <p className="mt-1 text-sm text-muted">
            {t("home.recent.description")}
          </p>
        </div>
      </div>
      {loading ? (
        <StatePanel title={t("home.recent.loading")} />
      ) : error ? (
        <StatePanel
          title={t("home.recent.error")}
          description={t("home.recent.errorDescription")}
          tone="error"
          actionLabel={t("home.recent.retry")}
          onAction={onRetry}
        />
      ) : experiments.length === 0 ? (
        <StatePanel
          title={t("home.recent.emptyTitle")}
          description={t("home.recent.emptyDescription")}
          actionLabel={t("home.recent.start")}
          onAction={onStart}
        />
      ) : (
        <div className="grid gap-4 lg:grid-cols-3">
          {[...experiments]
            .sort(
              (first, second) =>
                new Date(second.updated_at).getTime() -
                new Date(first.updated_at).getTime(),
            )
            .slice(0, 3)
            .map((experiment) => {
              const bestRun = selectBestRun(experiment, catalog);
              const primaryMetric = experiment.spec.metrics.primary;
              const metricValue = bestRun
                ? getMetricValue(bestRun, primaryMetric)
                : null;
              return (
                <button
                  key={experiment.id}
                  className="text-left"
                  onClick={() => onSelect(experiment.id)}
                >
                  <Card className="h-full transition-colors hover:border-primary">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="font-semibold">{experiment.name}</h3>
                      <StatusBadge status={experiment.status} />
                    </div>
                    <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
                      <RecentField
                        label={t("home.recent.dataset")}
                        value={
                          datasetNames.get(experiment.dataset_id) ??
                          experiment.dataset_id
                        }
                      />
                      <RecentField
                        label={t("home.recent.task")}
                        value={t(`home.tasks.${experiment.spec.task}`)}
                      />
                      <RecentField
                        label={t("home.recent.bestModel")}
                        value={bestRun?.model_id ?? t("home.recent.pending")}
                      />
                      <RecentField
                        label={primaryMetric}
                        value={
                          metricValue == null
                            ? t("home.recent.pending")
                            : metricValue.toFixed(4)
                        }
                      />
                    </dl>
                    <p className="mt-4 text-sm font-semibold text-primary">
                      {t(`home.recent.actions.${experiment.status}`, {
                        defaultValue: t("home.recent.actions.open"),
                      })}
                      <span aria-hidden="true"> →</span>
                    </p>
                  </Card>
                </button>
              );
            })}
        </div>
      )}
    </section>
  );
}

/**
 * Renderiza um campo descritivo com rótulo e valor na listagem de experimentos recentes.
 *
 * @param props Propriedades do campo contendo label e value.
 * @returns Elemento JSX representando o par termo/definição.
 */
function RecentField({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-muted">{label}</dt>
      <dd className="mt-1 truncate font-mono font-medium">{value}</dd>
    </div>
  );
}

/**
 * Obtém com segurança o valor numérico de uma métrica de um run de experimento.
 *
 * @param run Instância do run do experimento.
 * @param metric Identificador da métrica a ser extraída.
 * @returns Valor numérico da métrica ou null se inexistente ou inválido.
 */
function getMetricValue(
  run: Experiment["runs"][number],
  metric: string,
): number | null {
  const values = run.metrics?.values;
  if (!values || typeof values !== "object") return null;
  const value = (values as Record<string, unknown>)[metric];
  return typeof value === "number" ? value : null;
}

/**
 * Seleciona a melhor execução (run) de um experimento com base na métrica primária.
 *
 * Considera a direção de otimização registrada no catálogo (maximizar ou minimizar).
 *
 * @param experiment Experimento contendo as execuções avaliadas.
 * @param catalog Catálogo de registros contendo metadados de direção da métrica.
 * @returns O melhor run concluído ou null se nenhum run qualificado for encontrado.
 */
function selectBestRun(
  experiment: Experiment,
  catalog?: RegistryCatalog,
): Experiment["runs"][number] | null {
  const metric = experiment.spec.metrics.primary;
  const direction =
    catalog?.metrics.find((item) => item.id === metric)?.direction ??
    "maximize";
  return experiment.runs.reduce<Experiment["runs"][number] | null>(
    (best, run) => {
      const value = getMetricValue(run, metric);
      if (run.status !== "completed" || value == null) return best;
      if (!best) return run;
      const bestValue = getMetricValue(best, metric);
      if (bestValue == null) return run;
      return direction === "minimize"
        ? value < bestValue
          ? run
          : best
        : value > bestValue
          ? run
          : best;
    },
    null,
  );
}

/**
 * Exibe um resumo dos componentes e algoritmos registrados no catálogo da plataforma.
 *
 * @param props Propriedades contendo o catálogo, estado de carregamento, erro e callback de repetição.
 * @returns Painel descritivo com contagem de modelos, métricas e componentes.
 */
function RegistrySummary({
  catalog,
  loading,
  error,
  onRetry,
}: {
  catalog?: RegistryCatalog;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
}) {
  const { t } = useTranslation("common");

  return (
    <Card className="flex flex-wrap items-center justify-between gap-4 py-4">
      <div>
        <p className="font-semibold">{t("catalog.title")}</p>
        <p className="mt-1 text-sm text-muted">{t("catalog.description")}</p>
      </div>
      {loading ? (
        <p className="text-sm text-muted">{t("catalog.loading")}</p>
      ) : error || !catalog ? (
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-sm text-error">{t("catalog.error")}</p>
          <Button variant="outline" size="sm" onClick={onRetry}>
            {t("catalog.retry")}
          </Button>
        </div>
      ) : (
        <dl
          className="flex flex-wrap gap-x-5 gap-y-2 text-sm"
          data-testid="registry-summary"
        >
          {Object.entries(catalog).map(([name, items]) => (
            <div key={name} className="flex items-baseline gap-1">
              <dt className="text-muted">{t(`catalog.${name}`)}</dt>
              <dd className="font-mono font-semibold">{items.length}</dd>
            </div>
          ))}
        </dl>
      )}
    </Card>
  );
}

type DatasetLibraryProps = {
  datasets: DatasetSummary[];
  selectedId?: string;
  loading: boolean;
  onSelect: (datasetId: string) => void;
};

/**
 * Lista os datasets disponíveis na biblioteca do usuário para seleção e inspeção.
 *
 * @param props Propriedades contendo a lista de datasets, id selecionado, estado de carregamento e callback de seleção.
 * @returns Card com a lista clicável de datasets e suas dimensões.
 */
function DatasetLibrary({
  datasets,
  selectedId,
  loading,
  onSelect,
}: DatasetLibraryProps) {
  const { t } = useTranslation("datasets");

  return (
    <Card className="h-fit p-3">
      <p className="px-3 pb-3 pt-2 text-xs font-semibold uppercase tracking-wider text-muted">
        {t("library.title")}
      </p>
      {loading ? (
        <p className="px-3 pb-3 text-sm text-muted">{t("loading")}</p>
      ) : null}
      {!loading && datasets.length === 0 ? (
        <p className="px-3 pb-3 text-sm text-muted">{t("library.empty")}</p>
      ) : null}
      <div className="space-y-1">
        {datasets.map((dataset) => (
          <button
            key={dataset.id}
            aria-current={selectedId === dataset.id ? "page" : undefined}
            className={`w-full rounded-lg px-3 py-3 text-left transition-colors ${
              selectedId === dataset.id
                ? "bg-primary text-on-primary"
                : "hover:bg-background"
            }`}
            onClick={() => onSelect(dataset.id)}
          >
            <span className="block truncate text-sm font-semibold">
              {dataset.name}
            </span>
            <span
              className={`mt-1 block text-xs ${
                selectedId === dataset.id ? "text-on-primary/80" : "text-muted"
              }`}
            >
              {t("library.dimensions", {
                rows: numberFormatter.format(dataset.row_count),
                columns: dataset.column_count,
              })}
            </span>
          </button>
        ))}
      </div>
    </Card>
  );
}

/**
 * Visualização completa do perfil analítico, alertas, colunas e amostra de dados de um dataset.
 *
 * @param props Propriedades contendo os detalhes completos do dataset e perfil estatístico.
 * @returns Estrutura JSX completa do dashboard de análise exploratória e assessoria.
 */
function DatasetProfileView({ detail }: { detail: DatasetDetail }) {
  const { t } = useTranslation("datasets");
  const navigate = useNavigate();
  const { dataset, profile } = detail;

  return (
    <div className="min-w-0 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-muted">{dataset.original_filename}</p>
          <h2 className="mt-1 text-2xl font-semibold">{dataset.name}</h2>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="outline"
            onClick={() => navigate(`/datasets/${dataset.id}/expert`)}
          >
            {t("expert.action")}
          </Button>
          <span className="rounded-full border border-border bg-surface px-3 py-1 font-mono text-xs text-muted">
            {t("profile.version", { version: profile.profile_version })}
          </span>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label={t("metrics.rows")}
          value={numberFormatter.format(profile.rows)}
        />
        <Metric
          label={t("metrics.columns")}
          value={numberFormatter.format(profile.columns)}
        />
        <Metric
          label={t("metrics.duplicates")}
          value={numberFormatter.format(profile.duplicates)}
        />
        <Metric
          label={t("metrics.size")}
          value={formatBytes(profile.size_bytes)}
        />
      </div>

      <AdvisorPanel key={detail.dataset.id} detail={detail} />

      <Card>
        <h3 className="font-semibold">{t("warnings.title")}</h3>
        {profile.warnings.length ? (
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {profile.warnings.map((warning, index) => (
              <WarningItem
                key={`${warning.code}-${warning.column ?? "dataset"}-${index}`}
                warning={warning}
              />
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-muted">{t("warnings.empty")}</p>
        )}
      </Card>

      <Card className="overflow-hidden p-0">
        <div className="border-b border-border px-6 py-5">
          <h3 className="font-semibold">{t("columns.title")}</h3>
          <p className="mt-1 text-sm text-muted">{t("columns.description")}</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] border-collapse text-sm">
            <thead className="bg-background text-left text-xs uppercase tracking-wider text-muted">
              <tr>
                <th className="px-6 py-3">{t("columns.name")}</th>
                <th className="px-4 py-3">{t("columns.type")}</th>
                <th className="px-4 py-3">{t("columns.role")}</th>
                <th className="px-4 py-3 text-right">
                  {t("columns.cardinality")}
                </th>
                <th className="px-6 py-3 text-right">{t("columns.missing")}</th>
              </tr>
            </thead>
            <tbody>
              {profile.column_profiles.map((column) => (
                <tr key={column.name} className="border-t border-border">
                  <td className="px-6 py-4 font-mono font-medium">
                    {column.name}
                  </td>
                  <td className="px-4 py-4">
                    {t(`types.${column.inferred_type}`)}
                  </td>
                  <td className="px-4 py-4">
                    {t(`roles.${column.suggested_role}`)}
                  </td>
                  <td className="px-4 py-4 text-right">
                    {numberFormatter.format(column.cardinality)}
                  </td>
                  <td className="px-6 py-4 text-right">
                    {column.missing_percentage.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card className="overflow-hidden p-0">
        <div className="border-b border-border px-6 py-5">
          <h3 className="font-semibold">{t("preview.title")}</h3>
          <p className="mt-1 text-sm text-muted">{t("preview.description")}</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-max border-collapse text-sm">
            <thead className="bg-background text-left font-mono text-xs text-muted">
              <tr>
                {profile.column_names.map((column) => (
                  <th key={column} className="px-4 py-3 font-medium">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {profile.preview.map((row, rowIndex) => (
                <tr key={rowIndex} className="border-t border-border">
                  {profile.column_names.map((column) => (
                    <td key={column} className="max-w-64 truncate px-4 py-3">
                      {row[column] == null ? (
                        <span className="text-muted">-</span>
                      ) : (
                        String(row[column])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/**
 * Exibe um card de métrica estatística com rótulo e valor de destaque.
 *
 * @param props Propriedades contendo o rótulo descritivo e o valor formatado.
 * @returns Card com tipografia otimizada para visualização rápida.
 */
function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card className="p-5">
      <p className="text-sm text-muted">{label}</p>
      <p className="mt-2 font-mono text-2xl font-semibold">{value}</p>
    </Card>
  );
}

/**
 * Renderiza um item individual de alerta ou aviso de qualidade detectado no dataset.
 *
 * @param props Propriedades contendo o objeto de aviso com severidade e mensagem formatada.
 * @returns Elemento de lista estilizado de acordo com a severidade do alerta.
 */
function WarningItem({ warning }: { warning: ProfileWarning }) {
  const { t } = useTranslation("datasets");
  return (
    <li
      className={`rounded-lg border px-3 py-2 text-sm ${
        warning.severity === "warning"
          ? "border-warning-border bg-warning-bg text-warning-text"
          : "border-info-border bg-info-bg text-info-text"
      }`}
    >
      {t(`warningMessages.${warning.code}`, { column: warning.column ?? "" })}
    </li>
  );
}
