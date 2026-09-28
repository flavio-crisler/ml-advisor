import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import { WorkspaceLayout } from "../../components/workspace-header";
import { StatusBadge } from "../../components/status-badge";
import { Button } from "../../components/ui/button";
import { StatePanel } from "../../components/ui/state-panel";
import { Card } from "../../components/ui/card";
import { HowItWorksButton } from "../../components/how-it-works";
import { listDatasets } from "../../services/datasets";
import {
  deleteExperiment,
  listExperiments,
  renameExperiment,
  type Experiment,
} from "../../services/experiments";

const statuses = [
  "draft",
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
] as const;

/**
 * Página principal de listagem e gerenciamento de experimentos com busca, filtros por status/dataset, renomeação e exclusão.
 *
 * @returns Interface com cabeçalho, barra de filtros, tabela de experimentos e modais de edição.
 */
export function ExperimentsPage() {
  const { i18n, t } = useTranslation("experiments");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [editing, setEditing] = useState<Experiment | null>(null);
  const [editedName, setEditedName] = useState("");
  const [deleting, setDeleting] = useState<Experiment | null>(null);
  const experiments = useQuery({
    queryKey: ["experiments"],
    queryFn: listExperiments,
  });
  const datasets = useQuery({
    queryKey: ["datasets"],
    queryFn: listDatasets,
  });
  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      renameExperiment(id, name),
    onSuccess: async () => {
      setEditing(null);
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
  const remove = useMutation({
    mutationFn: deleteExperiment,
    onSuccess: async () => {
      setDeleting(null);
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
  const datasetNames = useMemo(
    () => new Map((datasets.data ?? []).map((item) => [item.id, item.name])),
    [datasets.data],
  );
  const filteredExperiments = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase(i18n.language);
    return (experiments.data ?? [])
      .filter((experiment) => !status || experiment.status === status)
      .filter((experiment) => !datasetId || experiment.dataset_id === datasetId)
      .filter((experiment) => {
        if (!normalizedQuery) return true;
        return [
          experiment.name,
          experiment.objective,
          datasetNames.get(experiment.dataset_id) ?? experiment.dataset_id,
        ].some((value) =>
          value.toLocaleLowerCase(i18n.language).includes(normalizedQuery),
        );
      })
      .sort(
        (first, second) =>
          new Date(second.updated_at).getTime() -
          new Date(first.updated_at).getTime(),
      );
  }, [datasetId, datasetNames, experiments.data, i18n.language, query, status]);
  const hasFilters = Boolean(query || status || datasetId);
  const clearFilters = () => {
    setQuery("");
    setStatus("");
    setDatasetId("");
  };
  const dateFormatter = new Intl.DateTimeFormat(i18n.language, {
    dateStyle: "medium",
    timeStyle: "short",
  });

  return (
    <WorkspaceLayout>
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            {t("history.eyebrow")}
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-4xl font-semibold tracking-tight">
              {t("history.title")}
            </h1>
            <HowItWorksButton screen="experiments" />
          </div>
          <p className="mt-3 max-w-2xl text-muted">
            {t("history.description")}
          </p>
        </div>
        <Button onClick={() => navigate("/")}>
          {t("actions.chooseDataset")}
        </Button>
      </section>

      {experiments.isError ? (
        <StatePanel
          title={t("errors.load")}
          description={t("errors.loadDescription")}
          tone="error"
          actionLabel={t("actions.retry")}
          onAction={() => void experiments.refetch()}
        />
      ) : experiments.isPending ? (
        <StatePanel title={t("loading")} />
      ) : experiments.data.length === 0 ? (
        <StatePanel
          title={t("history.emptyTitle")}
          description={t("history.emptyDescription")}
          actionLabel={t("actions.chooseDataset")}
          onAction={() => navigate("/")}
        />
      ) : (
        <>
          <Card className="space-y-4">
            <div className="grid gap-3 md:grid-cols-[minmax(0,1.5fr)_minmax(180px,0.7fr)_minmax(180px,0.7fr)_auto] md:items-end">
              <label className="text-sm font-semibold">
                {t("filters.search")}
                <input
                  className="mt-2 h-10 w-full rounded-md border border-border bg-surface px-3 font-normal focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  type="search"
                  value={query}
                  placeholder={t("filters.searchPlaceholder")}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              <label className="text-sm font-semibold">
                {t("filters.status")}
                <select
                  className="mt-2 h-10 w-full rounded-md border border-border bg-surface px-3 font-normal focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  value={status}
                  onChange={(event) => setStatus(event.target.value)}
                >
                  <option value="">{t("filters.allStatuses")}</option>
                  {statuses.map((item) => (
                    <option key={item} value={item}>
                      {t(`status.${item}`)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm font-semibold">
                {t("filters.dataset")}
                <select
                  className="mt-2 h-10 w-full rounded-md border border-border bg-surface px-3 font-normal focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  value={datasetId}
                  onChange={(event) => setDatasetId(event.target.value)}
                >
                  <option value="">{t("filters.allDatasets")}</option>
                  {(datasets.data ?? []).map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name}
                    </option>
                  ))}
                </select>
              </label>
              <Button
                variant="outline"
                disabled={!hasFilters}
                onClick={clearFilters}
              >
                {t("actions.clearFilters")}
              </Button>
            </div>
            <p className="text-sm text-muted" aria-live="polite">
              {t("filters.resultCount", { count: filteredExperiments.length })}
            </p>
          </Card>

          {filteredExperiments.length === 0 ? (
            <StatePanel
              title={t("filters.emptyTitle")}
              description={t("filters.emptyDescription")}
              actionLabel={t("actions.clearFilters")}
              onAction={clearFilters}
            />
          ) : (
            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {filteredExperiments.map((experiment) => {
                const deletionBlocked = ["queued", "running"].includes(
                  experiment.status,
                );
                return (
                  <Card key={experiment.id} className="flex h-full flex-col">
                    <button
                      className="flex-1 text-left"
                      onClick={() => navigate(`/experiments/${experiment.id}`)}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <h2 className="font-semibold">{experiment.name}</h2>
                        <StatusBadge status={experiment.status} />
                      </div>
                      <p className="mt-3 line-clamp-2 text-sm text-muted">
                        {experiment.objective}
                      </p>
                      <dl className="mt-5 grid grid-cols-2 gap-3 text-xs">
                        <div>
                          <dt className="text-muted">{t("fields.mode")}</dt>
                          <dd className="mt-1 font-medium">
                            {t(`modes.${experiment.mode}`)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-muted">{t("fields.runs")}</dt>
                          <dd className="mt-1 font-mono font-medium">
                            {experiment.runs.length}
                          </dd>
                        </div>
                      </dl>
                      <p className="mt-4 text-xs text-muted">
                        {dateFormatter.format(new Date(experiment.created_at))}
                      </p>
                      <p className="mt-4 text-sm font-semibold text-primary">
                        {t(`cardActions.${experiment.status}`, {
                          defaultValue: t("cardActions.open"),
                        })}
                        <span aria-hidden="true"> →</span>
                      </p>
                    </button>
                    <div className="mt-5 flex flex-wrap gap-2 border-t border-border pt-4">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          rename.reset();
                          setEditedName(experiment.name);
                          setEditing(experiment);
                        }}
                      >
                        {t("actions.rename")}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="text-error hover:border-error"
                        disabled={deletionBlocked}
                        title={
                          deletionBlocked
                            ? t("management.deleteBlocked")
                            : undefined
                        }
                        onClick={() => {
                          remove.reset();
                          setDeleting(experiment);
                        }}
                      >
                        {t("actions.delete")}
                      </Button>
                    </div>
                  </Card>
                );
              })}
            </section>
          )}
        </>
      )}

      {editing ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4">
          <Card
            className="w-full max-w-md"
            role="dialog"
            aria-modal="true"
            aria-labelledby="rename-experiment-title"
          >
            <form
              onSubmit={(event) => {
                event.preventDefault();
                rename.mutate({ id: editing.id, name: editedName.trim() });
              }}
            >
              <h2
                id="rename-experiment-title"
                className="text-xl font-semibold"
              >
                {t("management.renameTitle")}
              </h2>
              <p className="mt-2 text-sm text-muted">
                {t("management.renameDescription")}
              </p>
              <label className="mt-5 block text-sm font-semibold">
                {t("management.nameLabel")}
                <input
                  autoFocus
                  className="mt-2 h-10 w-full rounded-md border border-border bg-surface px-3 font-normal focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  maxLength={200}
                  value={editedName}
                  onChange={(event) => setEditedName(event.target.value)}
                />
              </label>
              {rename.isError ? (
                <p className="mt-3 text-sm text-error" role="alert">
                  {t("management.renameError")}
                </p>
              ) : null}
              <div className="mt-6 flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setEditing(null)}
                >
                  {t("actions.close")}
                </Button>
                <Button
                  type="submit"
                  disabled={!editedName.trim() || rename.isPending}
                >
                  {t("actions.saveName")}
                </Button>
              </div>
            </form>
          </Card>
        </div>
      ) : null}

      {deleting ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4">
          <Card
            className="w-full max-w-md"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="delete-experiment-title"
          >
            <h2 id="delete-experiment-title" className="text-xl font-semibold">
              {t("management.deleteTitle")}
            </h2>
            <p className="mt-2 text-sm text-muted">
              {t("management.deleteDescription", { name: deleting.name })}
            </p>
            <p className="mt-3 text-sm text-muted">
              {t("management.deleteRetention")}
            </p>
            {remove.isError ? (
              <p className="mt-3 text-sm text-error" role="alert">
                {t("management.deleteError")}
              </p>
            ) : null}
            <div className="mt-6 flex justify-end gap-2">
              <Button variant="outline" onClick={() => setDeleting(null)}>
                {t("actions.close")}
              </Button>
              <Button
                className="bg-error text-white hover:bg-error/90"
                disabled={remove.isPending}
                onClick={() => remove.mutate(deleting.id)}
              >
                {t("actions.confirmDelete")}
              </Button>
            </div>
          </Card>
        </div>
      ) : null}
    </WorkspaceLayout>
  );
}
