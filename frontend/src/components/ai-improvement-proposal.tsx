import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import {
  createImprovementDraft,
  proposeSupervisedImprovement,
} from "../services/improvement-proposals";
import type { ExplanationLanguage } from "../services/explanations";
import { Button } from "./ui/button";

/**
 * Referência de origem (experimento e model run) para cálculo de proposta de melhorias.
 */
export type ImprovementSource = {
  experimentId: string;
  modelRunId?: string;
};

/**
 * Componente que solicita, revisa e aplica propostas de melhoria de modelos supervisionados via IA.
 *
 * @param props.source - Origem da melhoria com IDs do experimento e model run.
 * @param props.language - Idioma para a resposta gerada.
 * @param props.context - Contexto das métricas e parâmetros atuais do experimento.
 * @returns Painel interativo com diffs de mudanças sugeridas e botão para criar rascunho.
 */
export function AIImprovementProposal({
  source,
  language,
  context,
}: {
  source: ImprovementSource;
  language: ExplanationLanguage;
  context: Record<string, unknown>;
}) {
  const { t } = useTranslation("common");
  const navigate = useNavigate();
  const proposal = useMutation({ mutationFn: proposeSupervisedImprovement });
  const draft = useMutation({
    mutationFn: createImprovementDraft,
    onSuccess: (result) => navigate(`/experiments/${result.experiment_id}`),
  });

  const active =
    proposal.variables?.source_experiment_id === source.experimentId &&
    proposal.variables?.source_model_run_id === (source.modelRunId ?? null)
      ? proposal.data
      : undefined;

  return (
    <section className="mt-6 border-t border-border pt-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-3xl">
          <h3 className="font-semibold">{t("aiImprovement.title")}</h3>
          <p className="mt-1 text-sm text-muted">
            {t("aiImprovement.description")}
          </p>
        </div>
        <Button
          variant="outline"
          disabled={proposal.isPending || draft.isPending}
          onClick={() =>
            proposal.mutate({
              language,
              source_experiment_id: source.experimentId,
              source_model_run_id: source.modelRunId ?? null,
              context,
            })
          }
        >
          {proposal.isPending
            ? t("aiImprovement.generating")
            : active
              ? t("aiImprovement.regenerate")
              : t("aiImprovement.generate")}
        </Button>
      </div>

      {proposal.isError ? (
        <p className="mt-4 text-sm text-error">{t("aiImprovement.error")}</p>
      ) : null}

      {active ? (
        <div className="mt-5 rounded-xl border border-border bg-background p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
            {t("aiImprovement.reviewLabel")}
          </p>
          <p className="mt-3 leading-7">{active.recommendation.summary}</p>

          {active.changes.length ? (
            <div className="mt-5 space-y-3">
              <h4 className="font-semibold">{t("aiImprovement.changes")}</h4>
              {active.changes.map((change) => (
                <details
                  key={change.path}
                  className="rounded-lg border border-border p-3"
                >
                  <summary className="cursor-pointer font-mono text-sm font-semibold">
                    {change.path}
                  </summary>
                  <div className="mt-3 grid gap-3 lg:grid-cols-2">
                    <DiffValue
                      label={t("aiImprovement.before")}
                      value={change.before}
                    />
                    <DiffValue
                      label={t("aiImprovement.after")}
                      value={change.after}
                    />
                  </div>
                </details>
              ))}
            </div>
          ) : null}

          <div className="mt-5 grid gap-5 lg:grid-cols-3">
            <ProposalList
              title={t("aiImprovement.rationale")}
              items={active.recommendation.rationale ?? []}
            />
            <ProposalList
              title={t("aiImprovement.cautions")}
              items={active.recommendation.cautions ?? []}
            />
            <ProposalList
              title={t("aiImprovement.unsupported")}
              items={active.recommendation.unsupported_suggestions ?? []}
            />
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-border pt-5">
            <Button
              disabled={draft.isPending}
              onClick={() =>
                draft.mutate({
                  source_experiment_id: active.source_experiment_id,
                  source_model_run_id: active.source_model_run_id,
                  source_spec_hash: active.source_spec_hash,
                  provider_id: active.provider_id,
                  recommendation: active.recommendation,
                })
              }
            >
              {draft.isPending
                ? t("aiImprovement.creatingDraft")
                : t("aiImprovement.createDraft")}
            </Button>
            <p className="text-xs text-muted">
              {t("aiImprovement.noTraining")}
            </p>
          </div>
          {draft.isError ? (
            <p className="mt-3 text-sm text-error">
              {t("aiImprovement.draftError")}
            </p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

/**
 * Exibe o valor antes ou depois de uma alteração de hiperparâmetro formatado como bloco de código.
 *
 * @param props.label - Rótulo indicador ('Antes' ou 'Depois').
 * @param props.value - Valor textual ou JSON da propriedade.
 * @returns Bloco formatado com quebra de linha e rolagem suave.
 */
function DiffValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <p className="text-xs font-semibold text-muted">{label}</p>
      <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-code p-3 text-xs text-code-text">
        {formatJson(value)}
      </pre>
    </div>
  );
}

/**
 * Formata com indentação uma string JSON se válida, mantendo a string original caso contrário.
 *
 * @param value - String contendo JSON bruto ou texto.
 * @returns String formatada com recuo de 2 espaços se for JSON válido.
 */
function formatJson(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

/**
 * Lista de argumentos de uma proposta de melhoria (racional, cuidados ou sugestões não suportadas).
 *
 * @param props.title - Título do agrupamento de itens.
 * @param props.items - Vetor com os itens da lista.
 * @returns Seção estruturada com lista de itens textuais.
 */
function ProposalList({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <section>
      <h4 className="font-semibold">{title}</h4>
      <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-muted">
        {items.map((item, index) => (
          <li key={`${index}-${item}`}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
