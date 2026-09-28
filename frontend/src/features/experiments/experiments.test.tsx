import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../../i18n";
import { ExperimentDetailPage } from "./experiment-detail-page";
import { ExperimentsPage } from "./experiments-page";
import { RunDetailPage } from "./run-detail-page";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-chart" />,
}));

const spec = {
  spec_version: "1.0",
  dataset_id: "dataset-1",
  dataset_hash: "hash-1",
  profile_version: "1.0",
  name: "Attendance model",
  objective: "Predict attendance",
  task: "binary_classification",
  target: "target",
  features: ["amount"],
  column_roles: { amount: "feature", target: "target" },
  models: [{ id: "logistic_regression", enabled: true, parameters: {} }],
  preprocessing: [],
  validation: {
    strategy: "train_test_split",
    test_size: 0.2,
    folds: 5,
    shuffle: true,
    random_seed: 41,
  },
  metrics: {
    primary: "f1",
    secondary: [],
    positive_label: 1,
  },
  tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
};

const modelRun = {
  id: "run-1",
  model_id: "logistic_regression",
  run_index: 1,
  status: "completed",
  started_at: "2026-09-21T12:00:00Z",
  finished_at: "2026-09-21T12:00:01Z",
  duration_seconds: 1,
  parameters: { C: 1 },
  metrics: {
    values: { f1: 0.91 },
    effective_parameters: { C: 0.5 },
    analysis_path: "analysis.json",
  },
  artifact_path: "model.joblib",
  error_message: null,
};

const experiment = {
  id: "experiment-1",
  name: "Attendance model",
  description: null,
  mode: "expert",
  status: "draft",
  dataset_id: "dataset-1",
  parent_experiment_id: null,
  objective: "Predict attendance",
  spec,
  created_at: "2026-09-21T12:00:00Z",
  updated_at: "2026-09-21T12:00:00Z",
  runs: [modelRun],
};

function renderRoute(
  path: string,
  element: React.ReactNode,
  routePattern = "*",
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path={routePattern} element={element} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function jsonResponse(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

describe("experiment history and results", () => {
  beforeEach(async () => {
    localStorage.clear();
    await i18n.changeLanguage("en-US");
  });

  afterEach(() => cleanup());

  it("lists persisted experiments", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) =>
        jsonResponse(
          input.endsWith("/experiments")
            ? [experiment]
            : input.endsWith("/datasets")
              ? []
              : { status: "ok" },
        ),
      ),
    );

    renderRoute("/experiments", <ExperimentsPage />);

    expect(await screen.findByText("Attendance model")).toBeInTheDocument();
    expect(screen.getAllByText("Draft")).toHaveLength(2);
    expect(screen.getByText("Training history")).toBeInTheDocument();
  });

  it("filters experiment history by search, status, and dataset", async () => {
    const completed = {
      ...experiment,
      id: "experiment-2",
      name: "Churn model",
      objective: "Predict churn",
      status: "completed",
      dataset_id: "dataset-2",
      updated_at: "2026-09-22T12:00:00Z",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) =>
        jsonResponse(
          input.endsWith("/experiments")
            ? [experiment, completed]
            : input.endsWith("/datasets")
              ? [
                  { id: "dataset-1", name: "Attendance" },
                  { id: "dataset-2", name: "Customers" },
                ]
              : { status: "ok" },
        ),
      ),
    );

    renderRoute("/experiments", <ExperimentsPage />);

    expect(await screen.findByText("Churn model")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Search"), {
      target: { value: "attendance" },
    });
    expect(screen.getByText("Attendance model")).toBeInTheDocument();
    expect(screen.queryByText("Churn model")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Search"), {
      target: { value: "" },
    });
    fireEvent.change(screen.getByLabelText("Status"), {
      target: { value: "completed" },
    });
    expect(screen.getByText("Churn model")).toBeInTheDocument();
    expect(screen.queryByText("Attendance model")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });
    expect(
      screen.getByText("No experiments match these filters."),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getAllByRole("button", { name: "Clear filters" })[1],
    );
    expect(screen.getByText("Attendance model")).toBeInTheDocument();
  });

  it("renames and removes an experiment from history", async () => {
    let storedExperiments = [experiment];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) => {
        if (input.endsWith("/datasets")) return jsonResponse([]);
        if (input.endsWith("/experiments") && !init?.method) {
          return jsonResponse(storedExperiments);
        }
        if (
          input.endsWith("/experiments/experiment-1") &&
          init?.method === "PATCH"
        ) {
          const { name } = JSON.parse(String(init.body)) as { name: string };
          storedExperiments = [{ ...storedExperiments[0], name }];
          return jsonResponse(storedExperiments[0]);
        }
        if (
          input.endsWith("/experiments/experiment-1") &&
          init?.method === "DELETE"
        ) {
          storedExperiments = [];
          return Promise.resolve({
            ok: true,
            status: 204,
            json: () => Promise.resolve(null),
          });
        }
        return jsonResponse({ status: "ok" });
      }),
    );

    renderRoute("/experiments", <ExperimentsPage />);
    await screen.findByText("Attendance model");

    fireEvent.click(screen.getByRole("button", { name: "Rename" }));
    const name = screen.getByLabelText("Experiment name");
    fireEvent.change(name, { target: { value: "Attendance baseline" } });
    fireEvent.click(screen.getByRole("button", { name: "Save name" }));
    expect(await screen.findByText("Attendance baseline")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(
      screen.getByRole("alertdialog", {
        name: "Remove experiment from history?",
      }),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Remove from history" }),
    );
    expect(await screen.findByText("No experiments yet.")).toBeInTheDocument();
  });

  it("shows an editable draft, runs, trials, and Plotly analysis", async () => {
    const explanationRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) => {
        if (input.endsWith("/explanations/status")) {
          return jsonResponse({
            available: true,
            provider_id: "ollama",
            is_local: true,
          });
        }
        if (input.endsWith("/explanations") && init?.method === "POST") {
          explanationRequests.push(
            JSON.parse(String(init.body)) as Record<string, unknown>,
          );
          return jsonResponse({
            provider_id: "ollama",
            is_local: true,
            explanation: {
              summary: "Overall run explanation.",
              findings: [],
              cautions: [],
              next_steps: [],
              chart_explanations: [
                {
                  chart_id: "confusion_matrix",
                  summary: "The diagonal contains most observations.",
                  findings: ["There are 17 correct predictions."],
                  cautions: ["Review the three off-diagonal errors."],
                },
              ],
            },
          });
        }
        if (input.endsWith("/analysis")) {
          return jsonResponse({
            schema_version: "1.0",
            task: "binary_classification",
            plots: {
              confusion_matrix: {
                data: [
                  {
                    type: "heatmap",
                    x: ["no", "yes"],
                    y: ["no", "yes"],
                    z: [
                      [8, 1],
                      [2, 9],
                    ],
                  },
                ],
                layout: {},
              },
            },
            explainability: { status: "available" },
          });
        }
        if (input.endsWith("/tuning-trials")) return jsonResponse([]);
        if (input.endsWith("/runs/run-1")) return jsonResponse(modelRun);
        if (input.endsWith("/experiments/experiment-1"))
          return jsonResponse(experiment);
        return jsonResponse({ status: "ok" });
      }),
    );

    const detail = renderRoute(
      "/experiments/experiment-1",
      <ExperimentDetailPage />,
      "/experiments/:experimentId",
    );
    expect(
      await screen.findByText("Edit draft specification"),
    ).toBeInTheDocument();
    expect(screen.getByText("logistic_regression")).toBeInTheDocument();
    detail.unmount();

    renderRoute(
      "/experiments/experiment-1/runs/run-1",
      <RunDetailPage />,
      "/experiments/:experimentId/runs/:runId",
    );
    expect(await screen.findByText("Effective parameters")).toBeInTheDocument();
    expect(await screen.findByText("Confusion matrix")).toBeInTheDocument();
    expect(screen.getByTestId("plotly-chart")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Deepen with AI" }),
    ).toBeEnabled();

    fireEvent.click(
      screen.getByRole("button", { name: "Generate explanation" }),
    );
    expect(
      await screen.findByText("The diagonal contains most observations."),
    ).toBeVisible();
    expect(explanationRequests[0]).toMatchObject({
      scope: "model_run",
      context: {
        charts: [
          {
            chart_id: "confusion_matrix",
            data_policy: "aggregate_only_no_raw_rows",
          },
        ],
      },
    });
  });

  it("refreshes the persisted experiment status after cancellation", async () => {
    let cancelled = false;
    vi.stubGlobal(
      "EventSource",
      class {
        onerror: (() => void) | null = null;

        addEventListener() {}

        close() {}
      },
    );
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) => {
        if (input.endsWith("/run") && init?.method === "POST") {
          return jsonResponse({ job_id: "job-1", status: "queued" });
        }
        if (input.endsWith("/cancel") && init?.method === "POST") {
          cancelled = true;
          return jsonResponse({ job_id: "job-1", status: "cancelled" });
        }
        if (input.endsWith("/experiments/experiment-1")) {
          return jsonResponse({
            ...experiment,
            status: cancelled ? "cancelled" : "draft",
          });
        }
        return jsonResponse({ status: "ok" });
      }),
    );

    renderRoute(
      "/experiments/experiment-1",
      <ExperimentDetailPage />,
      "/experiments/:experimentId",
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /^(Train|Retrain)$/ }),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(
      await screen.findByText("Cancelled", { exact: true }),
    ).toBeInTheDocument();
  });

  it("shows the failure cause and a recovery action for a failed run", async () => {
    const failedRun = {
      ...modelRun,
      status: "failed",
      metrics: null,
      error_message: "ValueError: invalid training data",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) => {
        if (input.endsWith("/tuning-trials")) return jsonResponse([]);
        if (input.endsWith("/runs/run-1")) return jsonResponse(failedRun);
        return jsonResponse({ status: "ok" });
      }),
    );

    renderRoute(
      "/experiments/experiment-1/runs/run-1",
      <RunDetailPage />,
      "/experiments/:experimentId/runs/:runId",
    );

    expect(await screen.findByText("This run failed")).toBeInTheDocument();
    expect(
      screen.getByText("ValueError: invalid training data"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Return and review experiment" }),
    ).toBeInTheDocument();
  });
});
