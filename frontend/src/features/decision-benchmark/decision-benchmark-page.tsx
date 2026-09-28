import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkspaceLayout } from "../../components/workspace-header";
import {
  getDataset,
  listDatasets,
  type ColumnProfile,
  type DatasetDetail,
} from "../../services/datasets";
import {
  createDecisionBenchmark,
  listDecisionRuns,
} from "../../services/decision-runs";
import { listDecisionProviders } from "../../services/decision-providers";
import { DecisionLabNav } from "../decision-providers/decision-lab-nav";
import { HowItWorksButton } from "../../components/how-it-works";

/**
 * Página de configuração e execução de benchmarks de decisão zero-shot/few-shot com modelos LLM.
 *
 * @returns Interface de seleção de dataset, target, colunas e painel de histórico de benchmarks.
 */
export function DecisionBenchmarkPage() {
  const { t, i18n } = useTranslation("decisionLab");
  const queryClient = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const providers = useQuery({
    queryKey: ["decision-providers"],
    queryFn: listDecisionProviders,
  });
  const runs = useQuery({
    queryKey: ["decision-runs"],
    queryFn: listDecisionRuns,
    refetchInterval: (query) =>
      query.state.data?.some((item) =>
        ["queued", "running"].includes(item.status),
      )
        ? 1000
        : false,
  });
  const [datasetId, setDatasetId] = useState("");
  const [target, setTarget] = useState("");
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [instructions, setInstructions] = useState("");
  const [positiveClass, setPositiveClass] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const detail = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: () => getDataset(datasetId),
    enabled: Boolean(datasetId),
  });
  const selectedProvider = providers.data?.find((item) => item.selected);
  const targetProfile = detail.data?.profile.column_profiles.find(
    (column) => column.name === target,
  );
  const classes = useMemo(
    () => decisionTargetClasses(detail.data, target),
    [detail.data, target],
  );
  const targetOptions = useMemo(
    () => decisionTargetOptions(detail.data),
    [detail.data],
  );
  const binary = targetProfile?.cardinality === 2;
  const targetIsSupported = Boolean(
    targetProfile && isDecisionTargetSupported(targetProfile, classes),
  );
  const [recentRunId, setRecentRunId] = useState("");
  const submit = useMutation({
    mutationFn: createDecisionBenchmark,
    onSuccess: (result) => {
      setRecentRunId(result.decision_run_id);
      return queryClient.invalidateQueries({ queryKey: ["decision-runs"] });
    },
  });

  function selectDataset(value: string) {
    setDatasetId(value);
    setTarget("");
    setInputColumns([]);
    setPositiveClass("");
  }

  function selectTarget(value: string) {
    setTarget(value);
    setInputColumns((current) => current.filter((column) => column !== value));
    setPositiveClass(decisionTargetClasses(detail.data, value)[0] ?? "");
  }

  function toggleColumn(column: string) {
    setInputColumns((current) =>
      current.includes(column)
        ? current.filter((item) => item !== column)
        : [...current, column],
    );
  }

  function runBenchmark() {
    if (!selectedProvider?.model || !targetProfile) return;
    const choiceCriteria = Object.fromEntries(
      classes.map((item) => [item, null]),
    );
    submit.mutate({
      external_processing_acknowledged: acknowledged,
      spec: {
        spec_version: "1.0",
        dataset_id: datasetId,
        target,
        task: binary ? "binary_classification" : "multiclass_classification",
        positive_class: binary ? positiveClass : null,
        input_columns: inputColumns,
        question: binary
          ? {
              question_version: "1.0",
              type: "noul",
              instructions,
              locale: i18n.language,
            }
          : {
              question_version: "1.0",
              type: "choice",
              instructions,
              locale: i18n.language,
              criteria: choiceCriteria,
            },
        provider: selectedProvider.id,
        model: selectedProvider.model,
        averaging: "macro",
        evaluation_split: "final_test",
        question_frozen: true,
      },
    });
  }

  const canSubmit = Boolean(
    datasetId &&
    target &&
    inputColumns.length &&
    instructions.trim() &&
    selectedProvider?.available &&
    acknowledged &&
    targetIsSupported &&
    (binary ? positiveClass : classes.length >= 2),
  );

  return (
    <WorkspaceLayout>
      <DecisionLabNav />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            {t("benchmark.eyebrow")}
          </p>
          <h1 className="text-4xl font-semibold tracking-tight">
            {t("benchmark.title")}
          </h1>
          <p className="mt-3 max-w-3xl text-muted">
            {t("benchmark.description")}
          </p>
        </div>
        <HowItWorksButton screen="decisionBenchmark" />
      </section>

      <div className="grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
        <Card>
          <h2 className="text-xl font-semibold">{t("benchmark.configure")}</h2>
          <div className="mt-5 grid gap-5 sm:grid-cols-2">
            <Field label={t("benchmark.dataset")}>
              <select
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={datasetId}
                onChange={(event) => selectDataset(event.target.value)}
              >
                <option value="">{t("benchmark.selectDataset")}</option>
                {datasets.data?.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t("benchmark.target")}>
              <select
                aria-label={t("benchmark.target")}
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={target}
                disabled={!detail.data}
                onChange={(event) => selectTarget(event.target.value)}
              >
                <option value="">{t("benchmark.selectTarget")}</option>
                {targetOptions.suggested.length ? (
                  <optgroup label={t("benchmark.suggestedTargets")}>
                    {targetOptions.suggested.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </optgroup>
                ) : null}
                {targetOptions.other.length ? (
                  <optgroup label={t("benchmark.otherTargets")}>
                    {targetOptions.other.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </optgroup>
                ) : null}
              </select>
              <span className="mt-2 block text-xs font-normal text-muted">
                {t("benchmark.targetHelp")}
              </span>
            </Field>
          </div>

          {detail.data && target ? (
            <fieldset className="mt-5">
              <legend className="text-sm font-semibold">
                {t("benchmark.inputs")}
              </legend>
              <p className="mt-1 text-xs text-muted">
                {t("benchmark.targetExcluded")}
              </p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {detail.data.profile.column_names
                  .filter((column) => column !== target)
                  .map((column) => (
                    <label
                      key={column}
                      className="flex items-center gap-2 text-sm"
                    >
                      <input
                        type="checkbox"
                        checked={inputColumns.includes(column)}
                        onChange={() => toggleColumn(column)}
                      />
                      {column}
                    </label>
                  ))}
              </div>
            </fieldset>
          ) : null}

          <Field label={t("benchmark.question")} className="mt-5">
            <textarea
              className="min-h-28 w-full rounded-lg border border-border bg-background px-3 py-2"
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              placeholder={t("benchmark.questionPlaceholder")}
            />
          </Field>

          {binary && classes.length ? (
            <Field label={t("benchmark.positiveClass")} className="mt-5">
              <select
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={positiveClass}
                onChange={(event) => setPositiveClass(event.target.value)}
              >
                {classes.map((className) => (
                  <option key={className} value={className}>
                    {className}
                  </option>
                ))}
              </select>
            </Field>
          ) : null}

          <label className="mt-5 flex items-start gap-3 rounded-lg border border-warning/40 bg-warning/5 p-4 text-sm">
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(event) => setAcknowledged(event.target.checked)}
            />
            <span>{t("benchmark.externalAck")}</span>
          </label>
          <div className="mt-5 flex items-center gap-3">
            <Button
              disabled={!canSubmit || submit.isPending}
              onClick={runBenchmark}
            >
              {submit.isPending
                ? t("benchmark.submitting")
                : t("benchmark.run")}
            </Button>
            {submit.isSuccess ? (
              <span className="text-sm text-success">
                {t("benchmark.queued")}
              </span>
            ) : null}
            {submit.isError ? (
              <span className="text-sm text-error">{t("benchmark.error")}</span>
            ) : null}
          </div>
        </Card>

        <div className="grid content-start gap-5">
          <Card>
            <h2 className="font-semibold">{t("comparison.supervised")}</h2>
            <p className="mt-2 text-sm text-muted">
              {t("comparison.supervisedDescription")}
            </p>
            <Link
              className="mt-4 inline-block text-sm text-primary"
              to="/experiments"
            >
              {t("comparison.openExperiments")}
            </Link>
          </Card>
          <Card>
            <h2 className="font-semibold">{t("comparison.zeroShot")}</h2>
            <p className="mt-2 text-sm text-muted">
              {t("comparison.zeroShotDescription")}
            </p>
          </Card>
          <Card
            title={t("comparison.tooltip")}
            className="border-primary/30 bg-primary/5"
          >
            <p className="text-sm">{t("comparison.tooltip")}</p>
          </Card>
        </div>
      </div>

      <section>
        <h2 className="mb-4 text-xl font-semibold">{t("benchmark.history")}</h2>
        {runs.isPending ? (
          <StatePanel title={t("states.loading")} />
        ) : runs.isError ? (
          <StatePanel
            title={t("benchmark.historyError")}
            description={t("states.errorDescription")}
            tone="error"
            actionLabel={t("states.retry")}
            onAction={() => void runs.refetch()}
          />
        ) : runs.data?.length ? (
          <div className="grid gap-3">
            {runs.data.map((run) => {
              const active =
                run.status === "queued" || run.status === "running";
              const progress = run.row_count
                ? Math.min(
                    100,
                    Math.round((run.processed_rows / run.row_count) * 100),
                  )
                : 0;

              return (
                <Link
                  key={run.id}
                  to={`/decision-lab/calibration/${run.id}`}
                  className="block focus:outline-none"
                >
                  <Card
                    className={`py-4 transition-colors hover:border-primary ${
                      active ? "border-primary/60 bg-primary/5" : ""
                    } ${run.id === recentRunId ? "ring-2 ring-primary/30" : ""}`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="font-semibold">
                          {run.spec.question.instructions}
                        </p>
                        <p className="mt-1 text-xs text-muted">
                          {run.provider_id} ·{" "}
                          {run.resolved_model ?? run.requested_model}
                        </p>
                      </div>
                      <span
                        className={
                          active
                            ? "rounded-full bg-primary px-2.5 py-1 text-xs font-semibold text-on-primary"
                            : run.status === "completed"
                              ? "rounded-full border border-success/30 bg-success/10 px-2.5 py-1 text-xs font-medium text-success"
                              : "rounded-full border border-error/30 bg-error/10 px-2.5 py-1 text-xs font-medium text-error"
                        }
                      >
                        {t(`runStatus.${run.status}`, {
                          defaultValue: run.status,
                        })}
                      </span>
                    </div>

                    {active ? (
                      <div
                        className="mt-4 rounded-lg border border-primary/30 bg-surface p-3"
                        role="status"
                      >
                        <p className="text-sm font-semibold text-primary">
                          {run.status === "queued"
                            ? t("benchmark.waitingForWorker")
                            : progress === 100
                              ? t("benchmark.finalizing")
                              : t("benchmark.processingRows", {
                                  processed: formatNumber(
                                    run.processed_rows,
                                    i18n.language,
                                  ),
                                  total: formatNumber(
                                    run.row_count,
                                    i18n.language,
                                  ),
                                  percent: progress,
                                })}
                        </p>
                        <div
                          className="mt-2 h-2 overflow-hidden rounded-full bg-border"
                          role="progressbar"
                          aria-label={t("benchmark.progressLabel")}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={progress}
                        >
                          <div
                            className="h-full rounded-full bg-primary transition-[width]"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <p className="mt-2 text-xs text-muted">
                          {t("benchmark.progressHelp")}
                        </p>
                      </div>
                    ) : null}
                  </Card>
                </Link>
              );
            })}
          </div>
        ) : (
          <StatePanel
            title={t("benchmark.noRuns")}
            description={t("benchmark.noRunsDescription")}
          />
        )}
      </section>
    </WorkspaceLayout>
  );
}

/**
 * Campo de formulário simples com rótulo textual e estilo consistente.
 *
 * @param props.label - Rótulo do campo.
 * @param props.className - Classes CSS adicionais.
 * @param props.children - Elemento de entrada associado.
 * @returns Elemento label estruturado.
 */
function Field({
  label,
  className = "",
  children,
}: {
  label: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <label className={`block text-sm font-medium ${className}`}>
      <span className="mb-2 block">{label}</span>
      {children}
    </label>
  );
}

/**
 * Extrai as classes categóricas disponíveis para a coluna alvo a partir do perfil e prévia do dataset.
 *
 * @param detail - Detalhes e perfil estatístico do dataset.
 * @param target - Nome da coluna selecionada como alvo.
 * @returns Lista de valores de classes identificados para a tarefa.
 */
function decisionTargetClasses(
  detail: DatasetDetail | undefined,
  target: string,
): string[] {
  if (!detail || !target) return [];
  const targetProfile = detail.profile.column_profiles.find(
    (column) => column.name === target,
  );
  if (!targetProfile) return [];

  const profiledValues =
    detail.profile.candidate_target_distributions?.[target] ??
    targetProfile.top_values ??
    [];
  const classes = new Set(profiledValues.map((item) => item.value));
  if (classes.size === targetProfile.cardinality) return [...classes];

  for (const row of detail.profile.preview) {
    const value = row[target];
    if (value !== null && value !== undefined) {
      classes.add(formatPreviewClass(value, targetProfile));
    }
  }
  return classes.size === targetProfile.cardinality ? [...classes] : [];
}

/**
 * Agrupa colunas candidatas a alvo em sugeridas e outras alternativas suportadas.
 *
 * @param detail - Detalhes do dataset com listas de colunas e cardinalidades.
 * @returns Objeto com listas de targets sugeridos e outros alvos válidos.
 */
function decisionTargetOptions(detail: DatasetDetail | undefined): {
  suggested: string[];
  other: string[];
} {
  if (!detail) return { suggested: [], other: [] };
  const suggestedTargets = new Set(detail.profile.candidate_targets);
  const supported = detail.profile.column_profiles
    .filter((profile) =>
      isDecisionTargetSupported(
        profile,
        decisionTargetClasses(detail, profile.name),
      ),
    )
    .map((profile) => profile.name);
  return {
    suggested: supported.filter((name) => suggestedTargets.has(name)),
    other: supported.filter((name) => !suggestedTargets.has(name)),
  };
}

/**
 * Formata um valor amostral do preview na representação textual da classe.
 *
 * @param value - Valor da célula.
 * @param profile - Metadados de tipo e distribuição da coluna.
 * @returns String formatada para a classe categórica.
 */
function formatPreviewClass(value: unknown, profile: ColumnProfile): string {
  if (
    typeof value === "number" &&
    Number.isInteger(value) &&
    profile.source_dtype.toLowerCase().includes("float")
  ) {
    return value.toFixed(1);
  }
  return String(value);
}

/**
 * Verifica se uma coluna atende aos critérios para ser utilizada como alvo em benchmark de decisão (2 a 20 classes).
 *
 * @param profile - Perfil estatístico da coluna.
 * @param classes - Classes identificadas.
 * @returns Verdadeiro se a coluna for elegível para benchmark.
 */
function isDecisionTargetSupported(
  profile: ColumnProfile,
  classes: string[],
): boolean {
  return (
    profile.cardinality >= 2 &&
    profile.cardinality <= 20 &&
    classes.length === profile.cardinality
  );
}

/**
 * Formata valores numéricos de acordo com o idioma/localidade do usuário.
 *
 * @param value - Número ou nulo.
 * @param locale - Código do idioma (ex: 'pt-BR' ou 'en-US').
 * @returns String formatada com separadores locais ou traço quando nulo.
 */
function formatNumber(value: number | null, locale: string) {
  return value === null ? "—" : new Intl.NumberFormat(locale).format(value);
}
