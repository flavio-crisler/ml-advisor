import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import { WorkspaceLayout } from "../../components/workspace-header";
import { StatusBadge } from "../../components/status-badge";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { Progress } from "../../components/ui/progress";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkflowSteps } from "../../components/workflow-steps";
import { HowItWorksButton } from "../../components/how-it-works";
import { describeExperimentIssues } from "../../services/experiment-validation";
import { ExperimentModelComparison } from "./experiment-model-comparison";
import {
  cancelExperiment,
  cloneExperiment,
  compareExperiments,
  getExperiment,
  ExperimentApiError,
  runExperiment,
  type ExperimentSpec,
  type Job,
  updateExperiment,
} from "../../services/experiments";

type TrainingState = {
  progress: number;
  messageKey: string;
  terminal: boolean;
};

/**
 * Página de detalhes e acompanhamento da execução de um experimento de Machine Learning.
 *
 * @returns Interface com status em tempo real, SSE de progresso, comparativo de modelos e editor de especificação.
 */
export function ExperimentDetailPage() {
  const { experimentId } = useParams();
  const { t } = useTranslation(["experiments", "training"]);
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const activeJobId = (location.state as { activeJobId?: string } | null)
    ?.activeJobId;
  const [job, setJob] = useState<Pick<Job, "job_id"> | null>(
    activeJobId ? { job_id: activeJobId } : null,
  );
  const [training, setTraining] = useState<TrainingState | null>(
    activeJobId
      ? {
          progress: 0,
          messageKey: "training:training.started",
          terminal: false,
        }
      : null,
  );
  const experiment = useQuery({
    queryKey: ["experiments", experimentId],
    queryFn: () => getExperiment(experimentId!),
    enabled: Boolean(experimentId),
  });
  const comparison = useQuery({
    queryKey: [
      "experiment-comparison",
      experiment.data?.parent_experiment_id,
      experimentId,
    ],
    queryFn: () =>
      compareExperiments(experiment.data!.parent_experiment_id!, experimentId!),
    enabled: Boolean(experiment.data?.parent_experiment_id && experimentId),
  });
  const clone = useMutation({
    mutationFn: () =>
      cloneExperiment(
        experimentId!,
        t("experiments:actions.copyName", { name: experiment.data!.name }),
      ),
    onSuccess: (created) => navigate(`/experiments/${created.id}`),
  });
  const run = useMutation({
    mutationFn: () => runExperiment(experimentId!),
    onSuccess: (createdJob) => {
      setJob(createdJob);
      setTraining({
        progress: 0,
        messageKey: "training:training.started",
        terminal: false,
      });
    },
  });
  const cancel = useMutation({
    mutationFn: () => cancelExperiment(experimentId!),
    onSuccess: async () => {
      setTraining({
        progress: training?.progress ?? 0,
        messageKey: "training:training.cancelled",
        terminal: true,
      });
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["experiments", experimentId],
        }),
        queryClient.invalidateQueries({ queryKey: ["experiments"] }),
      ]);
    },
  });

  useEffect(() => {
    if (!job || training?.terminal) return;
    const source = new EventSource(
      `/api/v1/experiments/${experimentId}/events`,
    );
    const eventNames = [
      "training.started",
      "validation.completed",
      "model.started",
      "training.progress",
      "tuning.progress",
      "model.completed",
      "training.completed",
      "job.completed",
      "job.failed",
      "job.cancelled",
    ];
    const listener = (event: Event) => {
      const message = event as MessageEvent<string>;
      const payload = JSON.parse(message.data) as {
        type: string;
        progress?: number;
        message_key?: string;
      };
      const terminal = [
        "job.completed",
        "job.failed",
        "job.cancelled",
      ].includes(payload.type);
      setTraining((current) => ({
        progress:
          payload.progress ?? (terminal ? 100 : (current?.progress ?? 0)),
        messageKey: payload.message_key
          ? `training:${payload.message_key}`
          : `training:${payload.type}`,
        terminal,
      }));
      if (terminal) {
        source.close();
        void queryClient.invalidateQueries({
          queryKey: ["experiments", experimentId],
        });
        void queryClient.invalidateQueries({
          queryKey: ["experiment-model-comparison", experimentId],
        });
        void queryClient.invalidateQueries({ queryKey: ["experiments"] });
      }
    };
    eventNames.forEach((name) => source.addEventListener(name, listener));
    source.onerror = () => source.close();
    return () => source.close();
  }, [experimentId, job, queryClient, training?.terminal]);

  if (experiment.isPending) {
    return (
      <WorkspaceLayout>
        <StatePanel title={t("experiments:loading")} />
      </WorkspaceLayout>
    );
  }
  if (experiment.isError || !experiment.data) {
    return (
      <WorkspaceLayout>
        <StatePanel
          title={t("experiments:errors.load")}
          description={t("experiments:errors.loadDescription")}
          tone="error"
          actionLabel={t("experiments:actions.retry")}
          onAction={() => void experiment.refetch()}
        />
      </WorkspaceLayout>
    );
  }

  const data = experiment.data;
  return (
    <WorkspaceLayout>
      <WorkflowSteps current={data.status === "completed" ? 6 : job ? 5 : 4} />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <button
            className="text-sm text-primary"
            onClick={() => navigate("/experiments")}
          >
            {t("experiments:actions.backToHistory")}
          </button>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <h1 className="text-3xl font-semibold">{data.name}</h1>
            <StatusBadge status={data.status} />
          </div>
          <p className="mt-3 max-w-3xl text-muted">{data.objective}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <HowItWorksButton screen="experimentDetail" />
          <Button
            variant="outline"
            onClick={() => clone.mutate()}
            disabled={clone.isPending}
          >
            {t("experiments:actions.clone")}
          </Button>
          <Button
            onClick={() => run.mutate()}
            disabled={run.isPending || (!training?.terminal && Boolean(job))}
          >
            {t(
              data.runs.length
                ? "experiments:actions.retrain"
                : "experiments:actions.train",
            )}
          </Button>
        </div>
      </section>

      {run.isError ? (
        <p className="text-sm text-error" role="alert">
          {run.error instanceof ExperimentApiError && run.error.issues.length
            ? describeExperimentIssues(run.error.issues)
            : t("experiments:errors.trainPreflight")}
        </p>
      ) : null}

      {training ? (
        <Card className="space-y-3">
          <div className="flex items-center justify-between gap-4">
            <p className="font-medium">
              {t(training.messageKey, {
                defaultValue: t("training:training.inProgress"),
              })}
            </p>
            {!training.terminal ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => cancel.mutate()}
              >
                {t("experiments:actions.cancel")}
              </Button>
            ) : null}
          </div>
          <Progress value={training.progress} />
        </Card>
      ) : null}

      {data.status === "failed" ? (
        <Card className="border-error/30 bg-error/5">
          <h2 className="font-semibold text-error">
            {t("experiments:detail.failedTitle")}
          </h2>
          <p className="mt-2 text-sm text-muted">
            {t("experiments:detail.failedDescription")}
          </p>
        </Card>
      ) : null}

      <section className="grid gap-5 lg:grid-cols-[1fr_0.8fr]">
        <Card>
          <h2 className="text-lg font-semibold">
            {t("experiments:detail.configuration")}
          </h2>
          <dl className="mt-4 grid gap-4 sm:grid-cols-2 text-sm">
            <Field
              label={t("experiments:fields.mode")}
              value={t(`experiments:modes.${data.mode}`)}
            />
            <Field
              label={t("experiments:fields.task")}
              value={t(`experiments:tasks.${data.spec.task}`)}
            />
            <Field
              label={t("experiments:fields.target")}
              value={data.spec.target}
            />
            <Field
              label={t("experiments:fields.features")}
              value={String(data.spec.features.length)}
            />
            <Field
              label={t("experiments:fields.featureList")}
              value={data.spec.features.join(", ")}
            />
            <Field
              label={t("experiments:fields.textProcessing")}
              value={
                (data.spec.preprocessing ?? []).some(
                  (step) =>
                    step.id === "text_vectorization" &&
                    step.strategy === "tfidf",
                )
                  ? t("experiments:fields.tfidfActive")
                  : t("experiments:fields.noTextProcessing")
              }
            />
            <Field
              label={t("experiments:fields.validation")}
              value={data.spec.validation.strategy}
            />
            <Field
              label={t("experiments:fields.primaryMetric")}
              value={data.spec.metrics.primary}
            />
          </dl>
        </Card>
        <Card>
          <h2 className="text-lg font-semibold">
            {t("experiments:detail.lineage")}
          </h2>
          <p className="mt-3 text-sm text-muted">
            {data.parent_experiment_id
              ? t("experiments:detail.cloneOf", {
                  id: data.parent_experiment_id,
                })
              : t("experiments:detail.original")}
          </p>
          <p className="mt-3 text-xs font-mono text-muted">{data.id}</p>
        </Card>
      </section>

      {data.status === "draft" ? (
        <SpecEditor
          experimentId={data.id}
          initialSpec={data.spec}
          onSaved={(saved) => {
            queryClient.setQueryData(["experiments", data.id], saved);
          }}
        />
      ) : null}

      {data.runs.length > 0 ? (
        <ExperimentModelComparison experimentId={data.id} />
      ) : null}

      <section>
        <h2 className="mb-4 text-xl font-semibold">
          {t("experiments:detail.runs")}
        </h2>
        {data.runs.length ? (
          <div className="grid gap-4 lg:grid-cols-2">
            {data.runs.map((modelRun) => (
              <button
                key={modelRun.id}
                className="text-left"
                onClick={() =>
                  navigate(`/experiments/${data.id}/runs/${modelRun.id}`)
                }
              >
                <Card className="h-full hover:border-primary">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold">{modelRun.model_id}</p>
                      <p className="mt-1 text-xs text-muted">
                        {t("experiments:detail.runIndex", {
                          index: modelRun.run_index,
                        })}
                      </p>
                    </div>
                    <StatusBadge status={modelRun.status} />
                  </div>
                  <MetricSummary metrics={modelRun.metrics} />
                  <p className="mt-4 text-sm font-semibold text-primary">
                    {t(
                      modelRun.status === "failed"
                        ? "experiments:detail.openFailure"
                        : "experiments:detail.openRun",
                    )}
                    <span aria-hidden="true"> →</span>
                  </p>
                </Card>
              </button>
            ))}
          </div>
        ) : (
          <Card>
            <p className="text-sm text-muted">
              {t("experiments:detail.noRuns")}
            </p>
          </Card>
        )}
      </section>

      {data.parent_experiment_id ? (
        <ComparisonTable
          rows={comparison.data ?? []}
          loading={comparison.isPending}
        />
      ) : null}
    </WorkspaceLayout>
  );
}

/**
 * Campo descritivo chave-valor para metadados do experimento.
 *
 * @param props.label - Rótulo da propriedade.
 * @param props.value - Valor da propriedade.
 * @returns Bloco de definição dt e dd.
 */
function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-muted">{label}</dt>
      <dd className="mt-1 font-medium">{value}</dd>
    </div>
  );
}

/**
 * Resumo horizontal de métricas numéricas computadas para o modelo.
 *
 * @param props.metrics - Objeto contendo o mapa de valores de métricas.
 * @returns Lista de definições com as métricas formatadas ou null se vazio.
 */
function MetricSummary({
  metrics,
}: {
  metrics: Record<string, unknown> | null;
}) {
  const { t } = useTranslation("experiments");
  const values = metrics?.values;
  if (!values || typeof values !== "object") return null;
  return (
    <dl className="mt-4 flex flex-wrap gap-4 text-sm">
      {Object.entries(values).map(([name, value]) => (
        <div key={name}>
          <dt className="text-muted">{name}</dt>
          <dd className="font-mono font-semibold">
            {typeof value === "number"
              ? value.toFixed(4)
              : t("detail.notAvailable")}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Editor interativo da especificação JSON de um experimento em rascunho ou antes do treino.
 *
 * @param props.experimentId - Identificador único do experimento.
 * @param props.initialSpec - Especificação inicial do experimento.
 * @param props.onSaved - Callback invocado quando as alterações forem salvas com sucesso.
 * @returns Card contendo área de texto JSON e botão de salvar com validação de erros.
 */
function SpecEditor({
  experimentId,
  initialSpec,
  onSaved,
}: {
  experimentId: string;
  initialSpec: ExperimentSpec;
  onSaved: (experiment: Awaited<ReturnType<typeof updateExperiment>>) => void;
}) {
  const { t } = useTranslation("experiments");
  const [value, setValue] = useState(JSON.stringify(initialSpec, null, 2));
  const [parseError, setParseError] = useState(false);
  const update = useMutation({
    mutationFn: (spec: ExperimentSpec) => updateExperiment(experimentId, spec),
    onSuccess: onSaved,
  });
  const save = () => {
    try {
      const spec = JSON.parse(value) as ExperimentSpec;
      setParseError(false);
      update.mutate(spec);
    } catch {
      setParseError(true);
    }
  };
  return (
    <Card>
      <h2 className="text-lg font-semibold">{t("editor.title")}</h2>
      <p className="mt-2 text-sm text-muted">{t("editor.description")}</p>
      <textarea
        className="mt-4 min-h-80 w-full rounded-lg border border-border bg-background p-4 font-mono text-xs"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        aria-label={t("editor.inputLabel")}
      />
      <div className="mt-3 flex items-center gap-3">
        <Button onClick={save} disabled={update.isPending}>
          {t("actions.save")}
        </Button>
        {parseError || update.isError ? (
          <p className="text-sm text-error" role="alert">
            {update.error instanceof ExperimentApiError &&
            update.error.issues.length
              ? describeExperimentIssues(update.error.issues)
              : t("editor.invalid")}
          </p>
        ) : null}
      </div>
    </Card>
  );
}

/**
 * Tabela de comparação lado a lado entre execuções de experimentos pai e filho.
 *
 * @param props.rows - Linhas pareadas contendo experimentos, modelos, métricas e parâmetros.
 * @param props.loading - Indicador de carregamento dos dados comparativos.
 * @returns Seção com tabela comparativa estilizada.
 */
function ComparisonTable({
  rows,
  loading,
}: {
  rows: Awaited<ReturnType<typeof compareExperiments>>;
  loading: boolean;
}) {
  const { t } = useTranslation("experiments");
  return (
    <section>
      <h2 className="mb-4 text-xl font-semibold">{t("comparison.title")}</h2>
      <Card className="overflow-x-auto">
        {loading ? (
          <p className="text-muted">{t("loading")}</p>
        ) : (
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead>
              <tr className="border-b border-border text-muted">
                <th className="pb-3">{t("comparison.experiment")}</th>
                <th className="pb-3">{t("comparison.model")}</th>
                <th className="pb-3">{t("comparison.metrics")}</th>
                <th className="pb-3">{t("comparison.parameters")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.run.id}
                  className="border-b border-border last:border-0"
                >
                  <td className="py-3">{row.experiment_name}</td>
                  <td className="py-3 font-mono">{row.run.model_id}</td>
                  <td className="py-3 font-mono text-xs">
                    {JSON.stringify(row.run.metrics?.values ?? {})}
                  </td>
                  <td className="py-3 font-mono text-xs">
                    {JSON.stringify(
                      row.run.metrics?.effective_parameters ??
                        row.run.parameters,
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </section>
  );
}
