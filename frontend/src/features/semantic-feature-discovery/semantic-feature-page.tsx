import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { HelpLabel, InfoTooltip } from "../../components/ui/info-tooltip";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkspaceLayout } from "../../components/workspace-header";
import { getDataset, listDatasets } from "../../services/datasets";
import { listDecisionProviders } from "../../services/decision-providers";
import { listExperiments } from "../../services/experiments";
import { getMetricRegistry } from "../../services/registries";
import {
  cancelSemanticFeatures,
  generateSemanticFeatures,
  listSemanticFeatureSets,
  proposeSemanticFeatures,
  type SemanticProposalResponse,
  type SemanticQuestion,
} from "../../services/semantic-features";
import { DecisionLabNav } from "../decision-providers/decision-lab-nav";
import { HowItWorksButton } from "../../components/how-it-works";
import { SemanticImpactAI } from "./semantic-impact-ai";

type ImpactModel = {
  model_id: string;
  primary_metric: string;
  metric_direction?: "maximize" | "minimize";
  baseline_metrics: Record<string, number>;
  augmented_metrics: Record<string, number>;
  text_baseline_metrics?: Record<string, number> | null;
  text_baseline_status?:
    "available" | "unavailable" | "too_many_text_columns" | "not_applicable";
  delta_absolute: number;
  delta_semantic_vs_text?: number | null;
  delta_percent: number | null;
  duration_seconds: number;
  baseline_duration_seconds?: number;
  text_baseline_duration_seconds?: number | null;
  augmented_duration_seconds?: number;
  semantic_feature_importance: Array<{ feature: string; importance: number }>;
};

type Impact = { models?: ImpactModel[]; text_columns?: string[] };
type QuestionKind = SemanticQuestion["question"]["type"];

/**
 * Página para descoberta, proposta interativa, geração assíncrona e avaliação de features semânticas via LLMs.
 *
 * @returns Interface com seleção de colunas de texto, editor de perguntas/critérios, SSE de progresso e análise de impacto.
 */
export function SemanticFeaturePage() {
  const { t, i18n } = useTranslation("decisionLab");
  const queryClient = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const providers = useQuery({
    queryKey: ["decision-providers"],
    queryFn: listDecisionProviders,
  });
  const experiments = useQuery({
    queryKey: ["experiments"],
    queryFn: listExperiments,
  });
  const metrics = useQuery({
    queryKey: ["metrics"],
    queryFn: getMetricRegistry,
  });
  const history = useQuery({
    queryKey: ["semantic-feature-sets"],
    queryFn: listSemanticFeatureSets,
    refetchInterval: (query) =>
      query.state.data?.some((item) =>
        ["queued", "running"].includes(item.status),
      )
        ? 1000
        : false,
  });
  const [datasetId, setDatasetId] = useState("");
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [objective, setObjective] = useState("");
  const [name, setName] = useState("");
  const [sourceExperimentId, setSourceExperimentId] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const [providerId, setProviderId] = useState("");
  const [model, setModel] = useState("");
  const [newQuestionType, setNewQuestionType] = useState<QuestionKind>("noul");
  const [recentFeatureSetId, setRecentFeatureSetId] = useState("");
  const recentFeatureSet = history.data?.find(
    (item) => item.id === recentFeatureSetId,
  );
  const [criteriaErrors, setCriteriaErrors] = useState<Set<string>>(new Set());
  const [proposal, setProposal] = useState<SemanticProposalResponse | null>(
    null,
  );
  const detail = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: () => getDataset(datasetId),
    enabled: Boolean(datasetId),
  });
  const semanticColumns = detail.data?.semantic_columns;
  const availableProviders = useMemo(
    () =>
      providers.data?.filter(
        (item) => item.selected && item.available && item.model,
      ) ?? [],
    [providers.data],
  );
  const selectedProvider =
    availableProviders.find((item) => item.id === providerId) ??
    availableProviders[0];
  const selectedModel =
    selectedProvider?.id === providerId && model
      ? model
      : (selectedProvider?.model ?? "");
  const compatibleExperiments = useMemo(
    () =>
      experiments.data?.filter(
        (item) =>
          item.dataset_id === datasetId &&
          item.status === "completed" &&
          !item.spec.tuning?.enabled,
      ) ?? [],
    [datasetId, experiments.data],
  );
  const propose = useMutation({
    mutationFn: proposeSemanticFeatures,
    onSuccess: (result) => {
      setProposal(result);
      setCriteriaErrors(new Set());
      setName((current) => current || t("semantic.defaultName"));
    },
  });
  const generate = useMutation({
    mutationFn: generateSemanticFeatures,
    onSuccess: (result) => {
      setRecentFeatureSetId(result.semantic_feature_set_id);
      queryClient.invalidateQueries({ queryKey: ["semantic-feature-sets"] });
      setProposal(null);
    },
  });
  const cancel = useMutation({
    mutationFn: cancelSemanticFeatures,
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["semantic-feature-sets"] }),
  });

  function selectDataset(value: string) {
    setDatasetId(value);
    setSelectedColumns([]);
    setSourceExperimentId("");
    setProposal(null);
    setCriteriaErrors(new Set());
  }

  function toggleColumn(column: string) {
    setSelectedColumns((current) =>
      current.includes(column)
        ? current.filter((item) => item !== column)
        : [...current, column],
    );
    setProposal(null);
    setCriteriaErrors(new Set());
  }

  function requestProposal() {
    propose.mutate({
      dataset_id: datasetId,
      objective,
      selected_columns: selectedColumns,
      language: i18n.language === "pt-BR" ? "pt-BR" : "en-US",
      discovery_mode: "target_agnostic",
    });
  }

  function updateQuestion(
    index: number,
    updater: (question: SemanticQuestion) => SemanticQuestion,
  ) {
    setProposal((current) => {
      if (!current) return current;
      const questions = current.proposal.questions.map((question, itemIndex) =>
        itemIndex === index ? updater(question) : question,
      );
      return { ...current, proposal: { ...current.proposal, questions } };
    });
  }

  function updateInstructions(index: number, instructions: string) {
    updateQuestion(index, (item) => ({
      ...item,
      question: { ...item.question, instructions },
    }));
  }

  function toggleQuestionColumn(index: number, column: string) {
    updateQuestion(index, (item) => {
      const columns = item.input_columns.includes(column)
        ? item.input_columns.filter((entry) => entry !== column)
        : [...item.input_columns, column];
      return columns.length ? { ...item, input_columns: columns } : item;
    });
  }

  function updateCriteria(
    index: number,
    criteria: SemanticQuestion["question"]["criteria"],
  ) {
    updateQuestion(index, (item) => ({
      ...item,
      question: { ...item.question, criteria } as SemanticQuestion["question"],
    }));
  }

  function removeQuestion(index: number) {
    const removedId = proposal?.proposal.questions[index]?.id;
    if (removedId) {
      setCriteriaErrors((current) => {
        const next = new Set(current);
        next.delete(removedId);
        return next;
      });
    }
    setProposal((current) =>
      current
        ? {
            ...current,
            proposal: {
              ...current.proposal,
              questions: current.proposal.questions.filter(
                (_, itemIndex) => itemIndex !== index,
              ),
            },
          }
        : current,
    );
  }

  function addQuestion() {
    if (!selectedColumns.length) return;
    setProposal((current) => {
      if (!current) return current;
      const existing = new Set(
        current.proposal.questions.map((item) => item.id),
      );
      let suffix = current.proposal.questions.length + 1;
      while (existing.has(`semantic_signal_${suffix}`)) suffix += 1;
      const question = createQuestion(newQuestionType, i18n.language, t);
      return {
        ...current,
        proposal: {
          ...current.proposal,
          questions: [
            ...current.proposal.questions,
            {
              semantic_spec_version: "1.0",
              id: `semantic_signal_${suffix}`,
              input_columns: [selectedColumns[0]],
              question,
            },
          ],
        },
      };
    });
  }

  function submitGeneration() {
    if (!proposal || !selectedProvider || !selectedModel) return;
    generate.mutate({
      dataset_id: datasetId,
      name,
      discovery_mode: "target_agnostic",
      proposal: proposal.proposal,
      provider_id: selectedProvider.id,
      model: selectedModel,
      external_processing_acknowledged: acknowledged,
      source_experiment_id: sourceExperimentId || null,
    });
  }

  const canGenerate = Boolean(
    proposal?.proposal.questions.length &&
    proposal.proposal.questions.every((item) =>
      item.question.instructions.trim(),
    ) &&
    !criteriaErrors.size &&
    name.trim() &&
    selectedProvider?.available &&
    selectedModel &&
    acknowledged,
  );

  return (
    <WorkspaceLayout>
      <DecisionLabNav />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            {t("semantic.eyebrow")}
          </p>
          <h1 className="text-4xl font-semibold tracking-tight">
            {t("semantic.title")}
          </h1>
          <p className="mt-3 max-w-3xl text-muted">
            {t("semantic.description")}
          </p>
          <p className="mt-3 max-w-3xl text-sm text-muted">
            {t("semantic.flowHelp")}
          </p>
        </div>
        <HowItWorksButton screen="semanticFeatures" />
      </section>

      <Card>
        <h2 className="text-xl font-semibold">{t("semantic.discover")}</h2>
        <div className="mt-5 grid gap-5 md:grid-cols-2">
          <Field label={t("semantic.dataset")}>
            <select
              className="w-full rounded-lg border border-border bg-background px-3 py-2"
              value={datasetId}
              onChange={(event) => selectDataset(event.target.value)}
            >
              <option value="">{t("semantic.selectDataset")}</option>
              {datasets.data?.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label={t("semantic.objective")}>
            <input
              className="w-full rounded-lg border border-border bg-background px-3 py-2"
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              placeholder={t("semantic.objectivePlaceholder")}
            />
          </Field>
        </div>
        {detail.data && semanticColumns ? (
          <div className="mt-5">
            <h3 className="text-sm font-semibold">
              {t("semantic.textColumns")}
            </h3>
            <p className="mt-1 text-xs text-muted">
              {t("semantic.textColumnsHelp")}
            </p>
            {semanticColumns.recommended.length ? (
              <fieldset className="mt-4">
                <legend className="text-sm font-semibold">
                  {t("semantic.recommendedColumns")}
                </legend>
                <div className="mt-2 flex flex-wrap gap-4">
                  {semanticColumns.recommended.map((column) => (
                    <label
                      key={column}
                      className="flex items-center gap-2 text-sm"
                    >
                      <input
                        type="checkbox"
                        aria-label={column}
                        checked={selectedColumns.includes(column)}
                        onChange={() => toggleColumn(column)}
                      />
                      {column}
                    </label>
                  ))}
                </div>
              </fieldset>
            ) : null}
            {semanticColumns.optional.length ? (
              <fieldset className="mt-4">
                <legend className="text-sm font-semibold">
                  {t("semantic.optionalColumns")}
                </legend>
                <p className="mt-1 text-xs text-muted">
                  {t("semantic.optionalWarning")}
                </p>
                <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {semanticColumns.optional.map((column) => {
                    const profile = detail.data.profile.column_profiles.find(
                      (item) => item.name === column,
                    );
                    return (
                      <label
                        key={column}
                        className="flex items-start gap-2 rounded-md border border-border p-2 text-sm"
                      >
                        <input
                          type="checkbox"
                          aria-label={column}
                          checked={selectedColumns.includes(column)}
                          onChange={() => toggleColumn(column)}
                        />
                        <span>
                          <span className="block font-medium">{column}</span>
                          {profile ? (
                            <span className="block text-xs text-muted">
                              {t("semantic.columnStats", {
                                distinct: profile.cardinality,
                                missing: profile.missing_percentage,
                              })}
                            </span>
                          ) : null}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </fieldset>
            ) : null}
            {!semanticColumns.recommended.length &&
            !semanticColumns.optional.length ? (
              <p className="mt-3 text-sm text-muted">
                {t("semantic.noTextColumns")}
              </p>
            ) : null}
            {semanticColumns.excluded_ids.length ? (
              <p className="mt-4 text-xs text-muted">
                {t("semantic.idsExcluded", {
                  columns: semanticColumns.excluded_ids.join(", "),
                })}
              </p>
            ) : null}
          </div>
        ) : null}
        <div className="mt-5 flex items-center gap-3">
          <Button
            onClick={requestProposal}
            disabled={
              !datasetId ||
              !objective.trim() ||
              !selectedColumns.length ||
              propose.isPending
            }
          >
            {propose.isPending
              ? t("semantic.proposing")
              : t("semantic.propose")}
          </Button>
          {propose.isError ? (
            <span className="text-sm text-error">{t("semantic.error")}</span>
          ) : null}
        </div>
      </Card>

      {proposal ? (
        <Card>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">{t("semantic.review")}</h2>
              <p className="mt-1 text-xs text-muted">
                {t("semantic.estimate", {
                  requests: proposal.estimated_requests,
                })}{" "}
                {t("semantic.costUnavailable")}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <select
                aria-label={t("semantic.newQuestionType")}
                className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
                value={newQuestionType}
                onChange={(event) =>
                  setNewQuestionType(event.target.value as QuestionKind)
                }
              >
                <option value="noul">Noul</option>
                <option value="choice">Choice</option>
                <option value="score">Score</option>
              </select>
              <InfoTooltip
                label={t("semantic.newQuestionType")}
                content={t("semantic.questionTypesHelp")}
              />
              <Button variant="outline" size="sm" onClick={addQuestion}>
                {t("semantic.addQuestion")}
              </Button>
            </div>
          </div>
          <div className="mt-5 grid gap-4">
            {proposal.proposal.questions.map((item, index) => (
              <div
                key={item.id}
                className="rounded-lg border border-border p-4"
              >
                <div className="flex flex-wrap justify-between gap-3">
                  <div>
                    <p className="font-mono text-sm font-semibold">{item.id}</p>
                    <p className="mt-1 text-xs text-muted">
                      <HelpLabel
                        help={t(`semantic.typeHelp.${item.question.type}`)}
                      >
                        {item.question.type}
                      </HelpLabel>
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => removeQuestion(index)}
                  >
                    {t("semantic.removeQuestion")}
                  </Button>
                </div>
                <textarea
                  className="mt-3 min-h-20 w-full rounded-lg border border-border bg-background px-3 py-2"
                  aria-label={`${t("semantic.question")} ${item.id}`}
                  value={item.question.instructions}
                  onChange={(event) =>
                    updateInstructions(index, event.target.value)
                  }
                />
                <fieldset className="mt-3">
                  <legend className="text-xs font-semibold">
                    {t("semantic.questionColumns")}
                  </legend>
                  <div className="mt-2 flex flex-wrap gap-3">
                    {selectedColumns.map((column) => (
                      <label
                        key={column}
                        className="flex items-center gap-2 text-xs"
                      >
                        <input
                          type="checkbox"
                          checked={item.input_columns.includes(column)}
                          onChange={() => toggleQuestionColumn(index, column)}
                        />
                        {column}
                      </label>
                    ))}
                  </div>
                </fieldset>
                <CriteriaEditor
                  key={`${item.id}-${item.question.type}`}
                  question={item.question}
                  label={t("semantic.criteria")}
                  invalidLabel={t("semantic.invalidCriteria")}
                  onChange={(criteria) => updateCriteria(index, criteria)}
                  onValidityChange={(valid) =>
                    setCriteriaErrors((current) => {
                      const next = new Set(current);
                      if (valid) next.delete(item.id);
                      else next.add(item.id);
                      return next;
                    })
                  }
                />
              </div>
            ))}
          </div>
          <div className="mt-5 grid gap-5 md:grid-cols-2">
            <Field label={t("semantic.featureSetName")}>
              <input
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </Field>
            <Field label={t("semantic.sourceExperiment")}>
              <select
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={sourceExperimentId}
                onChange={(event) => setSourceExperimentId(event.target.value)}
              >
                <option value="">{t("semantic.noImpact")}</option>
                {compatibleExperiments.map((experiment) => (
                  <option key={experiment.id} value={experiment.id}>
                    {experiment.name}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="mt-5 grid gap-5 md:grid-cols-2">
            <Field label={t("semantic.provider")}>
              <select
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={selectedProvider?.id ?? ""}
                onChange={(event) => {
                  const next = availableProviders.find(
                    (item) => item.id === event.target.value,
                  );
                  setProviderId(event.target.value);
                  setModel(next?.model ?? "");
                }}
              >
                {!availableProviders.length ? (
                  <option value="">{t("semantic.noProvider")}</option>
                ) : null}
                {availableProviders.map((provider) => (
                  <option key={provider.id} value={provider.id}>
                    {provider.display_name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t("semantic.providerModel")}>
              <select
                className="w-full rounded-lg border border-border bg-background px-3 py-2"
                value={selectedModel}
                onChange={(event) => setModel(event.target.value)}
              >
                {selectedProvider?.model ? (
                  <option value={selectedProvider.model}>
                    {selectedProvider.model}
                  </option>
                ) : (
                  <option value="">{t("semantic.noProvider")}</option>
                )}
              </select>
            </Field>
          </div>
          <label className="mt-5 flex items-start gap-3 rounded-lg border border-warning/40 bg-warning/5 p-4 text-sm">
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(event) => setAcknowledged(event.target.checked)}
            />
            <span>{t("semantic.externalAck")}</span>
          </label>
          <div className="mt-5 flex items-center gap-3">
            <Button
              disabled={!canGenerate || generate.isPending}
              onClick={submitGeneration}
            >
              {generate.isPending
                ? t("semantic.generating")
                : t("semantic.generate")}
            </Button>
            {generate.isError ? (
              <span className="text-sm text-error">{t("semantic.error")}</span>
            ) : null}
          </div>
        </Card>
      ) : null}

      {generate.isSuccess &&
      (!recentFeatureSet ||
        ["queued", "running"].includes(recentFeatureSet.status)) ? (
        <p role="status" className="text-sm text-success">
          {t("semantic.queued")} {t("semantic.queuedNextStep")}
        </p>
      ) : null}

      <section>
        <h2 className="mb-4 text-xl font-semibold">{t("semantic.history")}</h2>
        {history.isPending ? (
          <StatePanel title={t("states.loading")} />
        ) : history.isError ? (
          <StatePanel
            title={t("semantic.historyError")}
            description={t("states.errorDescription")}
            tone="error"
            actionLabel={t("states.retry")}
            onAction={() => void history.refetch()}
          />
        ) : history.data?.length ? (
          <div className="grid gap-4">
            {history.data.map((item) => {
              const impact = item.impact as Impact | null;
              const active =
                item.status === "queued" || item.status === "running";
              const progress = item.row_count
                ? Math.min(
                    100,
                    Math.round((item.processed_rows / item.row_count) * 100),
                  )
                : 0;
              const datasetName =
                datasets.data?.find((dataset) => dataset.id === item.dataset_id)
                  ?.name ?? item.dataset_id;
              const sourceExperiment = experiments.data?.find(
                (experiment) => experiment.id === item.source_experiment_id,
              );
              return (
                <Card
                  key={item.id}
                  className={`py-4 ${active ? "border-primary/60 bg-primary/5" : ""} ${item.id === recentFeatureSetId ? "ring-2 ring-primary/30" : ""}`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-semibold">{item.name}</h3>
                    <span
                      className={
                        active
                          ? "rounded-full bg-primary px-2 py-0.5 text-xs font-semibold text-on-primary"
                          : item.status === "completed"
                            ? "rounded-full bg-success/10 px-2 py-0.5 text-xs font-semibold text-success"
                            : "rounded-full bg-error/10 px-2 py-0.5 text-xs font-semibold text-error"
                      }
                    >
                      {t("runStatus." + item.status, {
                        defaultValue: item.status,
                      })}
                    </span>
                    {active ? (
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={
                          cancel.isPending && cancel.variables === item.id
                        }
                        onClick={() => cancel.mutate(item.id)}
                      >
                        {cancel.isPending && cancel.variables === item.id
                          ? t("semantic.cancelling")
                          : t("semantic.cancel")}
                      </Button>
                    ) : null}
                  </div>
                  {cancel.isError && cancel.variables === item.id ? (
                    <p className="mt-2 text-sm text-error" role="alert">
                      {t("semantic.cancelError")}
                    </p>
                  ) : null}
                  <p className="mt-1 text-sm text-muted">
                    {datasetName} ·{" "}
                    {t("semantic.questionCount", {
                      count: item.proposal.questions.length,
                    })}
                  </p>
                  {active ? (
                    <div
                      className="mt-4 rounded-lg border border-primary/30 bg-surface p-3"
                      role="status"
                    >
                      <p className="text-sm font-semibold text-primary">
                        {item.status === "queued"
                          ? t("semantic.waitingForWorker")
                          : progress === 100
                            ? t("semantic.finalizing")
                            : t("semantic.processingRows", {
                                processed: formatTokens(
                                  item.processed_rows,
                                  i18n.language,
                                ),
                                total: formatTokens(
                                  item.row_count,
                                  i18n.language,
                                ),
                                percent: progress,
                              })}
                      </p>
                      <div
                        className="mt-2 h-2 overflow-hidden rounded-full bg-border"
                        role="progressbar"
                        aria-label={t("semantic.progressLabel")}
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
                        {t("semantic.progressHelp")}
                      </p>
                    </div>
                  ) : null}
                  {item.status === "completed" ? (
                    <p className="mt-3 text-sm text-muted">
                      {impact?.models?.length
                        ? t("semantic.impactAvailable", {
                            experiment:
                              sourceExperiment?.name ??
                              item.source_experiment_id,
                          })
                        : item.source_experiment_id
                          ? t("semantic.impactUnavailable")
                          : t("semantic.noImpactAvailable")}
                    </p>
                  ) : null}
                  {item.status === "failed" ? (
                    <p className="mt-3 text-sm text-error">
                      {t("semantic.generationFailed")}
                    </p>
                  ) : null}
                  <details className="mt-4 rounded-lg border border-border p-3">
                    <summary className="cursor-pointer text-sm font-semibold text-primary">
                      {t("semantic.viewQuestions")}
                    </summary>
                    <details className="mt-3 text-xs text-muted">
                      <summary className="cursor-pointer font-medium">
                        {t("semantic.technicalDetails")}
                      </summary>
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <p className="mt-1 text-xs text-muted">
                            {t(`runStatus.${item.status}`, {
                              defaultValue: item.status,
                            })}{" "}
                            · {item.processed_rows}/{item.row_count} ·{" "}
                            {item.proposal.questions.length}{" "}
                            {t("semantic.featureCount")} · {item.request_count}{" "}
                            {t("semantic.requests")} · {item.cache_hits}{" "}
                            {t("semantic.cacheHits")} · {t("semantic.tokens")}:{" "}
                            {formatTokens(item.input_tokens, i18n.language)} ·{" "}
                            {t("semantic.estimatedCost")}:{" "}
                            {formatCost(item.estimated_cost)} ·{" "}
                            {t("semantic.totalLatency")}:{" "}
                            {formatLatency(item.latency_ms_total)}
                          </p>
                        </div>
                        <span className="font-mono text-xs text-muted">
                          {item.resolved_model ?? item.requested_model}
                        </span>
                      </div>
                    </details>
                    <p className="mt-3 text-xs text-muted">
                      {t("semantic.questionsFrozenHelp")}
                    </p>
                    <ol className="mt-3 space-y-3">
                      {item.proposal.questions.map((question) => (
                        <li
                          key={question.id}
                          className="rounded-md bg-background p-3 text-sm"
                        >
                          <p className="font-medium">
                            {question.question.instructions}
                          </p>
                          <p className="mt-1 text-xs text-muted">
                            {t(
                              "semantic.questionKind." + question.question.type,
                            )}{" "}
                            ·{" "}
                            {t("semantic.questionInput", {
                              columns: question.input_columns.join(", "),
                            })}
                          </p>
                          {question.question.type === "choice" ||
                          question.question.type === "score" ? (
                            <p className="mt-1 text-xs text-muted">
                              {t("semantic.questionOptions", {
                                options: (question.question.type === "choice"
                                  ? Object.keys(question.question.criteria)
                                  : question.question.criteria
                                ).join(", "),
                              })}
                            </p>
                          ) : null}
                        </li>
                      ))}
                    </ol>
                    {item.artifact_path ? (
                      <div className="mt-4 text-sm">
                        <p className="font-semibold">
                          {t("semantic.outputFile")}
                        </p>
                        <p className="mt-1 text-xs text-muted">
                          {t("semantic.outputFileHelp")}
                        </p>
                        <code className="mt-2 block break-all rounded-md bg-background p-2 text-xs">
                          {item.artifact_path}
                        </code>
                      </div>
                    ) : null}
                  </details>
                  {impact?.models?.length ? (
                    <div className="mt-5">
                      <h4 className="font-semibold">
                        {t("semantic.comparisonTitle")}{" "}
                        <InfoTooltip
                          content={t(
                            impact.models.some(
                              (model) => model.text_baseline_metrics,
                            )
                              ? "semantic.comparisonHelpThreeArms"
                              : "semantic.comparisonHelp",
                          )}
                          label={t("semantic.comparisonTitle")}
                        />
                      </h4>
                      <p className="mt-2 text-sm text-muted">
                        {impact.models.some(
                          (model) => model.text_baseline_metrics,
                        )
                          ? t("semantic.comparisonHelpThreeArms")
                          : t("semantic.comparisonHelp")}
                      </p>
                      {!impact.models.some(
                        (model) => model.text_baseline_metrics,
                      ) ? (
                        <p className="mt-1 text-xs text-muted">
                          {t("semantic.noLocalTextBaseline")}
                        </p>
                      ) : null}
                      <div className="mt-3 overflow-x-auto">
                        <table className="w-full min-w-[1100px] text-left text-sm [&_td]:px-2 [&_th]:px-2">
                          <thead>
                            <tr className="text-muted">
                              <th className="py-2">{t("semantic.model")}</th>
                              <th>{t("semantic.metric")}</th>
                              <th>
                                {t("semantic.baseline")}{" "}
                                <InfoTooltip
                                  content={t("semantic.baselineHelp")}
                                  label={t("semantic.baseline")}
                                />
                              </th>
                              {impact.models.some(
                                (model) => model.text_baseline_metrics,
                              ) ? (
                                <th>{t("semantic.localTextBaseline")}</th>
                              ) : null}
                              <th>
                                {t("semantic.augmented")}{" "}
                                <InfoTooltip
                                  content={t("semantic.augmentedHelp")}
                                  label={t("semantic.augmented")}
                                />
                              </th>
                              <th>
                                {t("semantic.direction")}{" "}
                                <InfoTooltip
                                  content={t("semantic.directionHelp")}
                                  label={t("semantic.direction")}
                                />
                              </th>
                              <th>
                                {t("semantic.deltaAbsolute")}{" "}
                                <InfoTooltip
                                  content={t("semantic.deltaHelp")}
                                  label={t("semantic.deltaAbsolute")}
                                />
                              </th>
                              {impact.models.some(
                                (model) => model.text_baseline_metrics,
                              ) ? (
                                <th>
                                  {t("semantic.deltaVsTfidf")}{" "}
                                  <InfoTooltip
                                    content={t("semantic.deltaVsTfidfHelp")}
                                    label={t("semantic.deltaVsTfidf")}
                                  />
                                </th>
                              ) : null}
                              <th>
                                {t("semantic.deltaPercent")}{" "}
                                <InfoTooltip
                                  content={t("semantic.percentHelp")}
                                  label={t("semantic.deltaPercent")}
                                />
                              </th>
                              <th>{t("semantic.observedResult")}</th>
                              <th>
                                {t("semantic.evaluationTime")}{" "}
                                <InfoTooltip
                                  content={t("semantic.evaluationTimeHelp")}
                                  label={t("semantic.evaluationTime")}
                                />
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {impact.models.map((model) => {
                              const direction =
                                model.metric_direction ??
                                metrics.data?.find(
                                  (metric) =>
                                    metric.id === model.primary_metric,
                                )?.direction;
                              const result = observedResult(
                                model.delta_absolute,
                                direction,
                              );
                              return (
                                <tr
                                  key={model.model_id}
                                  className="border-t border-border"
                                >
                                  <td className="whitespace-nowrap py-2">
                                    {model.model_id}
                                  </td>
                                  <td className="whitespace-nowrap">
                                    {model.primary_metric}
                                  </td>
                                  <td>
                                    {formatNumber(
                                      model.baseline_metrics[
                                        model.primary_metric
                                      ],
                                    )}
                                  </td>
                                  {impact.models?.some(
                                    (item) => item.text_baseline_metrics,
                                  ) ? (
                                    <td>
                                      {model.text_baseline_metrics
                                        ? formatNumber(
                                            model.text_baseline_metrics[
                                              model.primary_metric
                                            ],
                                          )
                                        : t("semantic.tfidfUnavailable")}
                                    </td>
                                  ) : null}
                                  <td>
                                    {formatNumber(
                                      model.augmented_metrics[
                                        model.primary_metric
                                      ],
                                    )}
                                  </td>
                                  <td>
                                    {direction === "maximize"
                                      ? t("semantic.higher")
                                      : direction === "minimize"
                                        ? t("semantic.lower")
                                        : t("semantic.unknownDirection")}
                                  </td>
                                  <td>{formatNumber(model.delta_absolute)}</td>
                                  {impact.models?.some(
                                    (item) => item.text_baseline_metrics,
                                  ) ? (
                                    <td>
                                      {model.delta_semantic_vs_text == null
                                        ? t("semantic.tfidfUnavailable")
                                        : formatNumber(
                                            model.delta_semantic_vs_text,
                                          )}
                                    </td>
                                  ) : null}
                                  <td>{formatPercent(model.delta_percent)}</td>
                                  <td
                                    className={
                                      result === "improved"
                                        ? "text-success"
                                        : result === "worsened"
                                          ? "text-error"
                                          : ""
                                    }
                                  >
                                    {t(`semantic.${result}`)}
                                  </td>
                                  <td>
                                    {model.text_baseline_duration_seconds !=
                                    null
                                      ? `${t("semantic.structuredShort")}: ${formatDuration(model.baseline_duration_seconds ?? 0)} · TF-IDF: ${formatDuration(model.text_baseline_duration_seconds)} · JEV: ${formatDuration(model.augmented_duration_seconds ?? 0)}`
                                      : formatDuration(model.duration_seconds)}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <p className="mt-3 text-sm text-muted">
                        {t("semantic.comparisonCaution")}
                      </p>
                      {impact.models.map((model) => (
                        <details
                          key={model.model_id}
                          className="mt-3 rounded-md border border-border p-3 text-sm"
                        >
                          <summary className="cursor-pointer font-medium">
                            {model.model_id}: {t("semantic.shapDetails")}{" "}
                            <InfoTooltip
                              content={t("semantic.shapHelp")}
                              label={t("semantic.semanticShap")}
                            />
                          </summary>
                          {model.semantic_feature_importance.length ? (
                            <ul className="mt-2 list-disc pl-5">
                              {model.semantic_feature_importance.map(
                                (entry) => (
                                  <li key={entry.feature}>
                                    {entry.feature}:{" "}
                                    {formatNumber(entry.importance)}
                                  </li>
                                ),
                              )}
                            </ul>
                          ) : (
                            <p className="mt-2 text-muted">
                              {t("semantic.noShap")}
                            </p>
                          )}
                        </details>
                      ))}
                      <SemanticImpactAI featureSetId={item.id} />
                    </div>
                  ) : null}
                </Card>
              );
            })}
          </div>
        ) : (
          <StatePanel
            title={t("semantic.empty")}
            description={t("semantic.emptyDescription")}
          />
        )}
      </section>
    </WorkspaceLayout>
  );
}

/**
 * Rótulo com campo de entrada aninhado para formulários.
 *
 * @param props.label - Rótulo textual do campo.
 * @param props.children - Elemento filho do campo.
 * @returns Elemento label estruturado.
 */
function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-sm font-medium">
      <span className="mb-2 block">{label}</span>
      {children}
    </label>
  );
}

/**
 * Cria uma especificação inicial padrão de pergunta semântica para o tipo selecionado.
 *
 * @param type - Tipo de pergunta ('noul', 'choice' ou 'score').
 * @param locale - Localidade (ex: 'pt-BR' ou 'en-US').
 * @param translate - Função de tradução i18n.
 * @returns Especificação completa da pergunta com critérios padrão.
 */
function createQuestion(
  type: QuestionKind,
  locale: string,
  translate: (key: string) => string,
): SemanticQuestion["question"] {
  const base = {
    question_version: "1.0" as const,
    instructions: translate("semantic.newQuestion"),
    locale,
  };
  if (type === "choice") {
    return {
      ...base,
      type,
      criteria: { yes: null, no: null },
    };
  }
  if (type === "score") {
    return { ...base, type, criteria: ["low", "high"] };
  }
  return { ...base, type, criteria: null };
}

/**
 * Editor JSON de critérios de classificação de uma pergunta semântica com validação em tempo real.
 *
 * @param props.question - Pergunta cuja configuração de critérios está sendo editada.
 * @param props.label - Rótulo do campo.
 * @param props.invalidLabel - Mensagem de erro exibida quando o JSON não atende ao esquema.
 * @param props.onChange - Callback com os novos critérios parseados.
 * @param props.onValidityChange - Notificação de mudança de validade (válido ou inválido).
 * @returns Textarea monoespaçada com mensagem de erro condicional.
 */
function CriteriaEditor({
  question,
  label,
  invalidLabel,
  onChange,
  onValidityChange,
}: {
  question: SemanticQuestion["question"];
  label: string;
  invalidLabel: string;
  onChange: (criteria: SemanticQuestion["question"]["criteria"]) => void;
  onValidityChange: (valid: boolean) => void;
}) {
  const [value, setValue] = useState(
    JSON.stringify(question.criteria ?? null, null, 2),
  );
  const [invalid, setInvalid] = useState(false);

  function update(raw: string) {
    setValue(raw);
    try {
      const parsed = JSON.parse(
        raw,
      ) as SemanticQuestion["question"]["criteria"];
      const valid = validCriteria(question.type, parsed);
      setInvalid(!valid);
      onValidityChange(valid);
      if (valid) onChange(parsed);
    } catch {
      setInvalid(true);
      onValidityChange(false);
    }
  }

  return (
    <label className="mt-3 block text-xs font-semibold">
      <span>{label}</span>
      <textarea
        className="mt-2 min-h-20 w-full rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs"
        aria-label={`${label} ${question.type}`}
        aria-invalid={invalid}
        value={value}
        onChange={(event) => update(event.target.value)}
      />
      {invalid ? (
        <span className="mt-1 block font-sans font-normal text-error">
          {invalidLabel}
        </span>
      ) : null}
    </label>
  );
}

/**
 * Valida se os critérios JSON preenchidos são semanticamente compatíveis com o tipo da pergunta.
 *
 * @param type - Tipo da pergunta (noul, choice, score).
 * @param criteria - Critérios fornecidos.
 * @returns Verdadeiro se os critérios forem válidos para o tipo.
 */
function validCriteria(
  type: QuestionKind,
  criteria: SemanticQuestion["question"]["criteria"],
) {
  if (type === "noul") {
    return (
      criteria === null ||
      (typeof criteria === "object" &&
        !Array.isArray(criteria) &&
        (Boolean(criteria.true) || Boolean(criteria.false)))
    );
  }
  if (type === "choice") {
    return (
      typeof criteria === "object" &&
      criteria !== null &&
      !Array.isArray(criteria) &&
      Object.keys(criteria).filter((key) => key.trim()).length >= 2
    );
  }
  return (
    Array.isArray(criteria) &&
    criteria.length >= 2 &&
    criteria.every((entry) => typeof entry === "string" && entry.trim())
  );
}

/**
 * Formata um número com 4 casas decimais ou traço caso indefinido.
 *
 * @param value - Valor numérico ou indefinido.
 * @returns String formatada.
 */
function formatNumber(value: number | undefined) {
  return value === undefined ? "—" : value.toFixed(4);
}

/**
 * Determina o resultado qualitativo da variação da métrica (melhorou, piorou, inalterado).
 *
 * @param delta - Diferença absoluta da métrica com as features aumentadas.
 * @param direction - Direção de otimização da métrica ('maximize' ou 'minimize').
 * @returns Classificação do resultado observado.
 */
function observedResult(
  delta: number,
  direction: "maximize" | "minimize" | undefined,
): "improved" | "worsened" | "unchanged" | "unknownResult" {
  if (!direction || !Number.isFinite(delta)) return "unknownResult";
  if (Math.abs(delta) < 1e-12) return "unchanged";
  return (direction === "maximize" ? delta > 0 : delta < 0)
    ? "improved"
    : "worsened";
}

/**
 * Formata custo financeiro estimado em dólares ou traço se nulo.
 *
 * @param value - Custo numérico ou nulo.
 * @returns String formatada com 4 casas decimais.
 */
function formatCost(value: number | null) {
  return value === null ? "—" : value.toFixed(4);
}

/**
 * Formata contagem de tokens com separadores de milhares locais.
 *
 * @param value - Quantidade de tokens ou nulo.
 * @param locale - Código do idioma/localidade.
 * @returns String formatada.
 */
function formatTokens(value: number | null, locale: string) {
  return value === null ? "—" : new Intl.NumberFormat(locale).format(value);
}

/**
 * Formata tempo de latência em milissegundos.
 *
 * @param value - Latência em ms ou nulo.
 * @returns String formatada com unidade 'ms'.
 */
function formatLatency(value: number | null) {
  return value === null ? "—" : `${value.toFixed(0)} ms`;
}

/**
 * Formata percentual de ganho/perda com sinal explícito de positivo ou negativo.
 *
 * @param value - Valor percentual ou nulo.
 * @returns String formatada com sinal e símbolo %.
 */
function formatPercent(value: number | null) {
  return value === null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

/**
 * Formata duração em segundos com duas casas decimais.
 *
 * @param value - Duração em segundos.
 * @returns String formatada com unidade 's'.
 */
function formatDuration(value: number) {
  return `${value.toFixed(2)} s`;
}
