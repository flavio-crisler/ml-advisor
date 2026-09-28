import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { WorkflowSteps } from "../../components/workflow-steps";
import {
  AdvisorApiError,
  type AdvisorResult,
  recommendExperiment,
  runAutoExperiment,
} from "../../services/advisor";
import type { DatasetDetail } from "../../services/datasets";
import { describeExperimentIssues } from "../../services/experiment-validation";
import {
  createExperiment,
  ExperimentApiError,
  type ExperimentSpec,
} from "../../services/experiments";

/**
 * Painel interativo do ML Advisor com recomendação assistida por LLM e modo Auto ML.
 *
 * @param props.detail - Dados detalhados e perfil estatístico do dataset em análise.
 * @returns Painel completo com formulário de objetivo, sugestões de target e revisão de proposta JSON.
 */
export function AdvisorPanel({ detail }: { detail: DatasetDetail }) {
  const { t } = useTranslation("advisor");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [objective, setObjective] = useState("");
  const [proposalJson, setProposalJson] = useState("");
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [llmLanguage, setLlmLanguage] = useState<"pt-BR" | "en-US">(() =>
    localStorage.getItem("llmLanguage") === "en-US" ? "en-US" : "pt-BR",
  );
  const suggestedTargets = detail.profile.candidate_targets ?? [];
  const objectiveExamples = suggestedTargets.length
    ? suggestedTargets
        .slice(0, 2)
        .map((target) => t("objective.targetExample", { target }))
    : [t("objective.datasetExample", { dataset: detail.dataset.name })];

  const recommendation = useMutation({
    mutationFn: () =>
      recommendExperiment({
        dataset_id: detail.dataset.id,
        objective,
        language: llmLanguage,
      }),
    onSuccess: (result) => {
      setProposalJson(JSON.stringify(result.proposal, null, 2));
      setReviewError(null);
    },
  });
  const autoRun = useMutation({
    mutationFn: () =>
      runAutoExperiment({
        dataset_id: detail.dataset.id,
        objective,
        language: llmLanguage,
      }),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
      navigate(`/experiments/${result.experiment_id}`, {
        state: { activeJobId: result.job_id },
      });
    },
  });
  const createDraft = useMutation({
    mutationFn: ({
      spec,
      result,
    }: {
      spec: ExperimentSpec;
      result: AdvisorResult;
    }) =>
      createExperiment(spec, {
        mode: "assisted",
        description: result.recommendation.explanation,
      }),
    onSuccess: async (experiment) => {
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
      navigate(`/experiments/${experiment.id}`);
    },
  });

  const submitDraft = () => {
    if (!recommendation.data) return;
    try {
      const parsed = JSON.parse(proposalJson) as ExperimentSpec;
      setReviewError(null);
      createDraft.mutate({ spec: parsed, result: recommendation.data });
    } catch {
      setReviewError(t("review.invalidJson"));
    }
  };

  const advisorError =
    recommendation.error instanceof AdvisorApiError
      ? recommendation.error.code === "invalid_advisor_recommendation"
        ? t("errors.invalid_advisor_recommendation")
        : recommendation.error.issues.length
          ? describeExperimentIssues(recommendation.error.issues)
          : t(`errors.${recommendation.error.code}`, {
              defaultValue: t("errors.unknown"),
            })
      : null;
  const autoError =
    autoRun.error instanceof AdvisorApiError
      ? autoRun.error.code === "invalid_advisor_recommendation"
        ? t("errors.invalid_advisor_recommendation")
        : autoRun.error.issues.length
          ? describeExperimentIssues(autoRun.error.issues)
          : t(`errors.${autoRun.error.code}`, {
              defaultValue: t("errors.unknown"),
            })
      : null;
  const draftError =
    createDraft.error instanceof ExperimentApiError
      ? createDraft.error.issues.length
        ? describeExperimentIssues(createDraft.error.issues)
        : t("errors.invalidProposal")
      : null;

  return (
    <Card data-testid="advisor-panel">
      <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">
            {t("eyebrow")}
          </p>
          <h3 className="mt-2 text-xl font-semibold">{t("title")}</h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
            {t("description")}
          </p>
        </div>
        <div className="space-y-3">
          <label htmlFor="advisor-objective" className="text-sm font-semibold">
            {t("objective.label")}
          </label>
          <textarea
            id="advisor-objective"
            className="min-h-28 w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            value={objective}
            maxLength={2000}
            placeholder={t("objective.placeholder")}
            onChange={(event) => setObjective(event.target.value)}
          />
          <div>
            <p className="text-xs font-semibold text-muted">
              {t("objective.examples")}
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {objectiveExamples.map((example) => (
                <Button
                  key={example}
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setObjective(example)}
                >
                  {example}
                </Button>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted">
              {t("objective.examplesHelp")}
            </p>
          </div>
          <label className="block text-sm font-semibold">
            {t("llmLanguage.label")}
            <select
              className="mt-2 h-10 w-full rounded-md border border-border bg-surface px-3 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              value={llmLanguage}
              onChange={(event) => {
                const language = event.target.value as "pt-BR" | "en-US";
                setLlmLanguage(language);
                localStorage.setItem("llmLanguage", language);
              }}
            >
              <option value="pt-BR">{t("llmLanguage.pt")}</option>
              <option value="en-US">{t("llmLanguage.en")}</option>
            </select>
          </label>
          <div className="grid gap-2 sm:grid-cols-2">
            <Button
              variant="outline"
              disabled={
                objective.trim().length === 0 ||
                recommendation.isPending ||
                autoRun.isPending
              }
              onClick={() => recommendation.mutate()}
            >
              {recommendation.isPending
                ? t("action.pending")
                : t("action.recommend")}
            </Button>
            <Button
              disabled={
                objective.trim().length === 0 ||
                autoRun.isPending ||
                recommendation.isPending
              }
              onClick={() => autoRun.mutate()}
            >
              {autoRun.isPending ? t("auto.pending") : t("auto.action")}
            </Button>
          </div>
          <p className="text-xs leading-5 text-muted">{t("auto.help")}</p>
          {advisorError || autoError ? (
            <p className="text-sm text-error" role="alert">
              {advisorError ?? autoError}
            </p>
          ) : null}
        </div>
      </div>

      {recommendation.data ? (
        <div className="mt-6 border-t border-border pt-6">
          <WorkflowSteps current={3} />
          <div className="grid gap-4 sm:grid-cols-3">
            <Summary
              label={t("summary.task")}
              value={t(`tasks.${recommendation.data.proposal.task}`)}
            />
            <Summary
              label={t("summary.target")}
              value={recommendation.data.proposal.target}
            />
            <Summary
              label={t("summary.models")}
              value={recommendation.data.proposal.models
                .map((model) => model.id)
                .join(", ")}
            />
          </div>
          <div className="mt-5 rounded-lg bg-background p-4">
            <p className="text-sm font-semibold">{t("explanation")}</p>
            <p className="mt-2 text-sm leading-6 text-muted">
              {recommendation.data.recommendation.explanation}
            </p>
            <p className="mt-3 text-xs text-muted">
              {recommendation.data.sample_sent
                ? t("privacy.sampleSent", {
                    count: recommendation.data.sample_rows,
                  })
                : t("privacy.profileOnly")}
            </p>
          </div>
          <label
            htmlFor="advisor-proposal"
            className="mt-5 block text-sm font-semibold"
          >
            {t("review.label")}
          </label>
          <p className="mt-1 text-xs text-muted">{t("review.help")}</p>
          <textarea
            id="advisor-proposal"
            className="mt-3 min-h-80 w-full rounded-lg border border-border bg-code p-4 font-mono text-xs leading-5 text-code-text focus:border-primary focus:outline-none"
            value={proposalJson}
            spellCheck={false}
            onChange={(event) => setProposalJson(event.target.value)}
          />
          {reviewError || draftError ? (
            <p className="mt-2 text-sm text-error" role="alert">
              {reviewError ?? draftError}
            </p>
          ) : null}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted">{t("review.noTraining")}</p>
            <Button disabled={createDraft.isPending} onClick={submitDraft}>
              {createDraft.isPending
                ? t("review.creating")
                : t("review.createDraft")}
            </Button>
          </div>
        </div>
      ) : null}
    </Card>
  );
}

/**
 * Bloco de resumo com rótulo e valor monoespaçado destacado.
 *
 * @param props.label - Rótulo do campo ou métrica.
 * @param props.value - Valor exibido no bloco.
 * @returns Elemento div estilizado como cartão de sumário.
 */
function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-4">
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-2 break-words font-mono text-sm font-semibold">
        {value}
      </p>
    </div>
  );
}
