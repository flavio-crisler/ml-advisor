import { useMutation, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { Card } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkspaceLayout } from "../../components/workspace-header";
import {
  listDecisionProviders,
  testDecisionProvider,
  type DecisionProviderSummary,
  type ProviderHealth,
} from "../../services/decision-providers";
import { DecisionLabNav } from "./decision-lab-nav";
import { HowItWorksButton } from "../../components/how-it-works";

const capabilityKeys = [
  "supports_noul",
  "supports_choice",
  "supports_score",
  "supports_batch",
  "supports_model_version_resolution",
] as const;

/**
 * Página principal do Decision Lab com listagem de provedores de IA, capacidades e teste de conexão.
 *
 * @returns Visão geral dos provedores com cartões de configuração e monitoramento de conectividade.
 */
export function DecisionLabPage() {
  const { t } = useTranslation("decisionLab");
  const providers = useQuery({
    queryKey: ["decision-providers"],
    queryFn: listDecisionProviders,
  });
  const connection = useMutation({ mutationFn: testDecisionProvider });

  const featureEnabled = providers.data?.some(
    (provider) => provider.feature_enabled,
  );

  return (
    <WorkspaceLayout>
      <DecisionLabNav />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            {t("eyebrow")}
          </p>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            {t("title")}
          </h1>
          <p className="mt-4 max-w-3xl text-lg leading-8 text-muted">
            {t("description")}
          </p>
        </div>
        <HowItWorksButton screen="decisionProviders" />
      </section>

      <Card className="border-warning/40 bg-warning/5">
        <p className="font-semibold">{t("external.title")}</p>
        <p className="mt-2 text-sm leading-6 text-muted">
          {t("external.description")}
        </p>
      </Card>

      {providers.isPending ? (
        <StatePanel title={t("states.loading")} />
      ) : providers.isError ? (
        <StatePanel
          title={t("states.error")}
          description={t("states.errorDescription")}
          tone="error"
          actionLabel={t("states.retry")}
          onAction={() => void providers.refetch()}
        />
      ) : providers.data.length === 0 ? (
        <StatePanel
          title={t("states.empty")}
          description={t("states.emptyDescription")}
        />
      ) : (
        <section aria-labelledby="provider-list-title">
          <div className="mb-4">
            <h2 id="provider-list-title" className="text-xl font-semibold">
              {t("providers.title")}
            </h2>
            <p className="mt-1 text-sm text-muted">
              {t("providers.description")}
            </p>
          </div>
          {!featureEnabled ? (
            <div className="mb-4 rounded-xl border border-border bg-background p-4 text-sm text-muted">
              {t("states.disabled")}
            </div>
          ) : null}
          <div className="grid gap-4 lg:grid-cols-2">
            {providers.data.map((provider) => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                connection={provider.selected ? connection.data : undefined}
                testing={provider.selected && connection.isPending}
                connectionError={provider.selected && connection.isError}
                onTest={() => connection.mutate()}
              />
            ))}
          </div>
        </section>
      )}
    </WorkspaceLayout>
  );
}

/**
 * Cartão de apresentação detalhado de um provedor de decisão configurado.
 *
 * @param props.provider - Sumário dos metadados e capacidades do provedor.
 * @param props.connection - Resultado do teste de conexão, se executado.
 * @param props.testing - Se o teste de conexão está em andamento.
 * @param props.connectionError - Se ocorreu falha durante o teste.
 * @param props.onTest - Callback para disparar o teste de conectividade.
 * @returns Card contendo dados do provedor, badges de capacidades e botão de teste.
 */
function ProviderCard({
  provider,
  connection,
  testing,
  connectionError,
  onTest,
}: {
  provider: DecisionProviderSummary;
  connection?: ProviderHealth;
  testing: boolean;
  connectionError: boolean;
  onTest: () => void;
}) {
  const { t } = useTranslation("decisionLab");
  return (
    <Card data-testid="decision-provider-card">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">{provider.display_name}</h3>
          <p className="mt-1 font-mono text-xs text-muted">{provider.id}</p>
        </div>
        <span
          className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
            provider.available
              ? "border-success/30 bg-success/10 text-success"
              : "border-border bg-background text-muted"
          }`}
        >
          {t(`status.${provider.health_status}`, {
            defaultValue: provider.health_status,
          })}
        </span>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
        <ProviderField
          label={t("providers.processing")}
          value={
            provider.is_external
              ? t("providers.external")
              : t("providers.local")
          }
        />
        <ProviderField
          label={t("providers.model")}
          value={provider.model ?? t("providers.notConfigured")}
        />
      </dl>

      <div className="mt-5 flex flex-wrap gap-2">
        {capabilityKeys
          .filter((capability) => provider.capabilities[capability])
          .map((capability) => (
            <span
              key={capability}
              className="rounded-full bg-primary/10 px-2.5 py-1 text-xs text-primary"
            >
              {t(`capabilities.${capability}`)}
            </span>
          ))}
      </div>
      {provider.selected ? (
        <div className="mt-5 border-t border-border pt-5">
          <Button
            variant="outline"
            size="sm"
            disabled={!provider.configured || testing}
            onClick={onTest}
          >
            {testing ? t("connection.testing") : t("connection.action")}
          </Button>
          {connection ? (
            <p
              className={`mt-3 text-sm ${
                connection.status === "available"
                  ? "text-success"
                  : "text-error"
              }`}
            >
              {t(`connection.${connection.status}`)}
              {connection.latency_ms != null
                ? ` · ${connection.latency_ms.toFixed(0)} ms`
                : ""}
            </p>
          ) : null}
          {connectionError ? (
            <p className="mt-3 text-sm text-error">{t("connection.error")}</p>
          ) : null}
        </div>
      ) : null}
    </Card>
  );
}

/**
 * Exibe um campo com rótulo e valor na lista de propriedades do provedor.
 *
 * @param props.label - Rótulo do campo.
 * @param props.value - Valor textual do campo.
 * @returns Bloco de definição com dt e dd.
 */
function ProviderField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-1 break-words font-medium">{value}</dd>
    </div>
  );
}
