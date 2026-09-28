import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../../i18n";
import { ConfidenceCascadePage } from "./confidence-cascade-page";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plot" />,
}));

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/decision-lab/cascades"]}>
        <ConfidenceCascadePage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const decisionRun = {
  id: "run-1",
  dataset_id: "dataset-1",
  experiment_id: null,
  provider_id: "typesafe_jev",
  requested_model: "jev-latest",
  resolved_model: "jev-1.13.0",
  question_spec_hash: "hash",
  status: "completed",
  row_count: 2,
  processed_rows: 2,
  request_count: 2,
  cache_hits: 0,
  input_tokens: 10,
  estimated_cost: 0.02,
  latency_ms_total: 20,
  artifact_path: "summary.json",
  calibration_run_id: "calibration-1",
  error_message: null,
  created_at: "2026-09-21T00:00:00Z",
  started_at: "2026-09-21T00:00:00Z",
  finished_at: "2026-09-21T00:00:01Z",
  spec: { question: { instructions: "Will this customer churn?" } },
  metrics: {},
};

const cascadeRun = {
  id: "cascade-1",
  decision_run_id: "run-1",
  policy: {
    version: "1.0",
    source_decision_run_id: "run-1",
    auto_accept_threshold: 0.9,
    review_threshold: 0.6,
    below_review_action: "abstain",
    manual_review_unit_cost: null,
  },
  metrics: {
    threshold: 0.9,
    coverage: 0.5,
    auto_decision_rate: 0.5,
    review_rate: 0.5,
    abstain_rate: 0.25,
    accuracy_auto: 1,
    error_rate_auto: 0,
    estimated_total_cost: 0.02,
    decision_cost: 0.02,
    threshold_sensitivity: [
      {
        threshold: 0.5,
        coverage: 1,
        auto_decision_rate: 1,
        review_rate: 0,
        abstain_rate: 0.25,
        accuracy_auto: 0.5,
        error_rate_auto: 0.5,
        estimated_total_cost: 0.02,
      },
      {
        threshold: 0.9,
        coverage: 0.5,
        auto_decision_rate: 0.5,
        review_rate: 0.5,
        abstain_rate: 0.25,
        accuracy_auto: 1,
        error_rate_auto: 0,
        estimated_total_cost: 0.02,
      },
    ],
    confidence_distribution: [{ lower: 0.8, upper: 0.9, count: 2 }],
  },
  artifact_path: "metrics.json",
  created_at: "2026-09-21T00:00:00Z",
};

describe("Confidence Cascade", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en-US");
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) =>
        Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve(
              input.endsWith("/decision-runs")
                ? [decisionRun]
                : init?.method === "POST"
                  ? cascadeRun
                  : [],
            ),
        }),
      ),
    );
  });

  afterEach(() => cleanup());

  it("explores persisted thresholds without another API request", async () => {
    renderPage();
    const simulate = screen.getByRole("button", {
      name: "Simulate and save policy",
    });
    await waitFor(() => expect(simulate).toBeEnabled());
    fireEvent.click(simulate);
    expect(
      await screen.findByLabelText("Cascade simulation results"),
    ).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(
          ([url, init]) =>
            String(url).endsWith("/confidence-cascades") && !init?.method,
        ),
      ).toHaveLength(2),
    );
    const callsBeforeSlider = fetchMock.mock.calls.length;

    fireEvent.change(
      screen.getByRole("slider", {
        name: /^Explore auto-accept threshold locally/,
      }),
      { target: { value: "0.5" } },
    );

    expect(screen.getByText("100.0%")).toBeInTheDocument();
    expect(screen.getByText("Abstention rate")).toBeInTheDocument();
    expect(screen.getByText("25.0%")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(callsBeforeSlider);
  });
});
