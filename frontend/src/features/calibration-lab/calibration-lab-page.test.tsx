import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../../i18n";
import { CalibrationLabPage } from "./calibration-lab-page";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plot" />,
}));

const spec = {
  spec_version: "1.0",
  dataset_id: "dataset-1",
  target: "label",
  task: "binary_classification",
  positive_class: "yes",
  input_columns: ["message"],
  question: {
    question_version: "1.0",
    type: "noul",
    instructions: "Will this cancel?",
    locale: "en-US",
  },
  provider: "typesafe_jev",
  model: "jev-latest",
  averaging: "macro",
  evaluation_split: "final_test",
  question_frozen: true,
};

const decisionRun = {
  id: "run-1",
  dataset_id: "dataset-1",
  experiment_id: null,
  provider_id: "typesafe_jev",
  requested_model: "jev-latest",
  resolved_model: "jev-1.13",
  question_spec_hash: "hash",
  status: "completed",
  row_count: 4,
  processed_rows: 4,
  request_count: 3,
  cache_hits: 1,
  input_tokens: 15,
  estimated_cost: 0.03,
  latency_ms_total: 60,
  artifact_path: "summary.json",
  calibration_run_id: "calibration-1",
  error_message: null,
  created_at: "2026-09-22T00:00:00Z",
  started_at: "2026-09-22T00:00:00Z",
  finished_at: "2026-09-22T00:00:01Z",
  spec,
  metrics: { latency_p95_ms: 29 },
};

const experiment = {
  id: "experiment-1",
  name: "Supervised baseline",
  description: null,
  mode: "assisted",
  status: "completed",
  dataset_id: "dataset-1",
  parent_experiment_id: null,
  objective: "Predict label",
  spec: {
    dataset_id: "dataset-1",
    target: "label",
    task: "binary_classification",
  },
  created_at: "2026-09-22T00:00:00Z",
  updated_at: "2026-09-22T00:00:00Z",
  runs: [],
};

function response(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

function renderPage() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter initialEntries={["/decision-lab/calibration/run-1"]}>
        <Routes>
          <Route
            path="/decision-lab/calibration/:runId"
            element={<CalibrationLabPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

let explanationRequests: Array<{
  scope: string;
  context: { charts?: Array<{ chart_id: string; data: unknown }> };
}> = [];

describe("Calibration Lab", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en-US");
    explanationRequests = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) => {
        if (input.endsWith("/explanations/status")) {
          return response({
            available: true,
            provider_id: "ollama",
            is_local: true,
          });
        }
        if (input.endsWith("/explanations") && init?.method === "POST") {
          const request = JSON.parse(
            String(init.body),
          ) as (typeof explanationRequests)[number];
          explanationRequests.push(request);
          const chartIds =
            request.context.charts?.map((chart) => chart.chart_id) ?? [];
          return response({
            provider_id: "ollama",
            is_local: true,
            explanation: {
              summary:
                request.scope === "chart_explanation"
                  ? "The selected chart was analyzed in depth."
                  : "The comparison is ready for review.",
              findings: [],
              cautions: [],
              next_steps: [],
              chart_explanations: chartIds.map((chartId) => ({
                chart_id: chartId,
                summary:
                  request.scope === "chart_explanation"
                    ? `Focused ${chartId} explanation.`
                    : `Overall ${chartId} explanation.`,
                findings: [],
                cautions: [],
              })),
            },
          });
        }
        if (input.endsWith("/calibration")) {
          return response({
            id: "calibration-1",
            decision_run_id: "run-1",
            task: "binary_classification",
            artifact_path: "calibration.json",
            created_at: "2026-09-22T00:00:00Z",
            metrics: {
              brier_score: 0.1,
              log_loss: 0.2,
              expected_calibration_error: 0.05,
              maximum_calibration_error: 0.1,
              confidence_mean: 0.8,
              reliability: [
                {
                  lower: 0.8,
                  upper: 0.9,
                  count: 4,
                  mean_confidence: 0.8,
                  empirical_accuracy: 0.75,
                },
              ],
              confidence_histogram: [{ lower: 0.8, upper: 0.9, count: 4 }],
              thresholds: [
                {
                  threshold: 0.8,
                  coverage: 0.75,
                  accuracy: 1,
                  error_rate: 0,
                  review_rate: 0.25,
                  estimated_cost: 0.03,
                },
              ],
              latency_ms: [],
            },
          });
        }
        if (input.includes("/compare/experiment-1")) {
          return response({
            schema_version: "1.0",
            decision_run_id: "run-1",
            experiment_id: "experiment-1",
            experiment_name: "Supervised baseline",
            contract: { row_alignment: "intersection_by_row_id" },
            unavailable_models: [],
            models: [
              {
                model_run_id: "model-run-1",
                model_id: "logistic_regression",
                run_index: 2,
                sample_size: 4,
                classes: ["no", "yes"],
                supervised: {
                  metrics: {
                    accuracy: 0.75,
                    f1: 0.7333,
                    roc_auc: 0.8,
                    pr_auc: 0.8,
                    brier_score: 0.2,
                    expected_calibration_error: 0.1,
                  },
                  latency_p95_ms: null,
                  estimated_cost: null,
                },
                zero_shot: {
                  metrics: {
                    accuracy: 0.5,
                    f1: 0.5,
                    roc_auc: 0.6,
                    pr_auc: 0.6,
                    brier_score: 0.3,
                    expected_calibration_error: 0.2,
                  },
                  latency_p95_ms: 29,
                  estimated_cost: 0.03,
                },
                metric_deltas: {
                  accuracy: 0.25,
                  f1: 0.2333,
                  roc_auc: 0.2,
                  pr_auc: 0.2,
                  brier_score: -0.1,
                  expected_calibration_error: -0.1,
                },
                disagreement: {
                  both_correct: 2,
                  supervised_only_correct: 1,
                  zero_shot_only_correct: 0,
                  both_wrong: 1,
                  prediction_disagreements: 1,
                },
              },
            ],
          });
        }
        if (input.endsWith("/experiments")) return response([experiment]);
        return response(decisionRun);
      }),
    );
  });

  afterEach(() => cleanup());

  it("shows the recorded provider error when a benchmark fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) =>
        response(
          input.endsWith("/decision-benchmarks/run-1")
            ? {
                ...decisionRun,
                status: "failed",
                processed_rows: 2,
                error_message:
                  "DecisionProviderTransportError: Decision Provider request failed with HTTP 520.",
              }
            : [],
        ),
      ),
    );
    renderPage();

    expect(
      await screen.findByText("The zero-shot benchmark failed."),
    ).toBeInTheDocument();
    expect(screen.getByText(/HTTP 520/)).toBeInTheDocument();
    expect(
      screen.queryByText("The zero-shot benchmark is still processing."),
    ).not.toBeInTheDocument();
  });

  it("shows provider operations and a paired supervised comparison", async () => {
    renderPage();

    expect(await screen.findByText("Real provider calls")).toBeInTheDocument();
    expect(screen.getByText("Cache hits")).toBeInTheDocument();
    expect(
      screen.getByText(/every response came from cache/i),
    ).toBeInTheDocument();

    fireEvent.change(
      screen.getByLabelText("Compatible supervised experiment"),
      { target: { value: "experiment-1" } },
    );

    expect(await screen.findByText("4 shared rows")).toBeInTheDocument();
    expect(
      screen.getByText("Supervised Models · logistic_regression"),
    ).toBeInTheDocument();
    expect(screen.getByText("Δ supervised − zero-shot")).toBeInTheDocument();
    expect(screen.getByText("logistic_regression · Run 2")).toBeInTheDocument();
    expect(screen.getByText("Both correct")).toBeInTheDocument();
    expect(screen.getByText("Different predictions")).toBeInTheDocument();

    expect(await screen.findByText(/context stays/i)).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Generate explanation" }),
    );
    expect(
      await screen.findByRole("button", {
        name: "Create improvement proposal",
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("Overall reliability explanation."),
    ).toBeVisible();
    expect(
      screen.getAllByRole("button", { name: "Deepen with AI" }),
    ).toHaveLength(4);
    expect(explanationRequests[0].context.charts).toHaveLength(4);
    expect(explanationRequests[0].context.charts?.[3]).toMatchObject({
      chart_id: "provider_latency",
      data: { count: 0, bins: [] },
    });

    fireEvent.click(
      screen.getAllByRole("button", { name: "Deepen with AI" })[0],
    );
    expect(
      await screen.findByText("Focused reliability explanation."),
    ).toBeVisible();
    expect(explanationRequests[1]).toMatchObject({
      scope: "chart_explanation",
      context: { charts: [{ chart_id: "reliability" }] },
    });
  });
});
