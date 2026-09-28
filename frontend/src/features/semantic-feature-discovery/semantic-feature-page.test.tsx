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
import { SemanticFeaturePage } from "./semantic-feature-page";

let historyItems: unknown[] = [];

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/decision-lab/semantic-features"]}>
        <SemanticFeaturePage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Semantic Feature Discovery", () => {
  beforeEach(async () => {
    historyItems = [];
    await i18n.changeLanguage("en-US");
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) => {
        let payload: unknown = [];
        if (input.endsWith("/datasets")) {
          payload = [
            {
              id: "dataset-1",
              name: "Support",
              original_filename: "support.csv",
              hash: "hash",
              row_count: 10,
              column_count: 2,
              created_at: "2026-09-21T00:00:00Z",
            },
          ];
        } else if (input.endsWith("/datasets/dataset-1")) {
          payload = {
            dataset: { id: "dataset-1", name: "Support" },
            profile: {
              text_columns: ["message", "notes"],
              column_names: [
                "message",
                "notes",
                "Return_Reason",
                "Transaction_ID",
                "target",
              ],
              column_profiles: [
                {
                  name: "Return_Reason",
                  cardinality: 6,
                  missing_percentage: 89.6,
                },
              ],
            },
            semantic_columns: {
              recommended: ["message", "notes"],
              optional: ["Return_Reason"],
              excluded_ids: ["Transaction_ID"],
            },
          };
        } else if (input.endsWith("/decision-providers")) {
          payload = [
            {
              id: "typesafe_jev",
              selected: true,
              available: true,
              model: "jev-latest",
            },
          ];
        } else if (input.endsWith("/semantic-features")) {
          payload = historyItems;
        } else if (input.endsWith("/metrics")) {
          payload = [{ id: "rmse", direction: "minimize" }];
        } else if (input.endsWith("/explanations/status")) {
          payload = { available: true, provider_id: "fake", is_local: true };
        } else if (
          input.endsWith("/semantic-features/set-impact/explanation")
        ) {
          payload = {
            provider_id: "fake",
            is_local: true,
            explanation: {
              summary: "The metric declined in this evaluation.",
              findings: ["RMSE rose from 1.0 to 1.2."],
              cautions: ["Check prediction-time availability."],
              next_steps: ["Validate on new data."],
              chart_explanations: [],
            },
          };
        } else if (
          input.endsWith("/semantic-features/propose") &&
          init?.method === "POST"
        ) {
          payload = {
            provider_id: "openai_compatible",
            discovery_mode: "target_agnostic",
            sample_sent: true,
            sample_rows: 2,
            estimated_requests: 10,
            estimated_input_tokens: null,
            estimated_cost: null,
            proposal: {
              schema_version: "1.0",
              questions: [
                {
                  semantic_spec_version: "1.0",
                  id: "intent",
                  input_columns: ["message"],
                  question: {
                    question_version: "1.0",
                    type: "noul",
                    instructions: "Does the customer intend to cancel?",
                    locale: "en-US",
                  },
                },
              ],
            },
          };
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(payload),
        });
      }),
    );
  });

  afterEach(() => cleanup());

  it("proposes from selected text columns without sending the target", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Support" });
    fireEvent.change(await screen.findByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });
    fireEvent.click(await screen.findByLabelText("message"));
    fireEvent.change(screen.getByLabelText("Modeling objective"), {
      target: { value: "Predict churn" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Suggest questions with AI" }),
    );

    expect(
      await screen.findByDisplayValue("Does the customer intend to cancel?"),
    ).toBeInTheDocument();
    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url]) =>
        String(url).endsWith("/semantic-features/propose"),
      );
    expect(call).toBeDefined();
    const body = JSON.parse(String(call?.[1]?.body));
    expect(body.selected_columns).toEqual(["message"]);
    expect(body).not.toHaveProperty("target");
    await waitFor(() => expect(body.discovery_mode).toBe("target_agnostic"));
  });

  it("offers categorical text manually and excludes identifiers", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Support" });
    fireEvent.change(await screen.findByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });

    expect(await screen.findByLabelText("Return_Reason")).toBeInTheDocument();
    expect(screen.queryByLabelText("Transaction_ID")).not.toBeInTheDocument();
    expect(screen.getByText(/Identifiers excluded/)).toHaveTextContent(
      "Transaction_ID",
    );
    fireEvent.click(screen.getByLabelText("Return_Reason"));
    fireEvent.change(screen.getByLabelText("Modeling objective"), {
      target: { value: "Predict returns" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Suggest questions with AI" }),
    );
    await waitFor(() => {
      const call = vi
        .mocked(fetch)
        .mock.calls.find(([url]) =>
          String(url).endsWith("/semantic-features/propose"),
        );
      expect(JSON.parse(String(call?.[1]?.body)).selected_columns).toEqual([
        "Return_Reason",
      ]);
    });
  });

  it("reviews typed criteria and columns before freezing generation", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Support" });
    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });
    fireEvent.click(await screen.findByLabelText("message"));
    fireEvent.click(screen.getByLabelText("notes"));
    fireEvent.change(screen.getByLabelText("Modeling objective"), {
      target: { value: "Predict churn" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Suggest questions with AI" }),
    );
    await screen.findByDisplayValue("Does the customer intend to cancel?");

    fireEvent.change(screen.getByLabelText("New question type"), {
      target: { value: "choice" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    fireEvent.change(screen.getByLabelText("Criteria (JSON) choice"), {
      target: {
        value: JSON.stringify({ billing: "Billing", technical: null }),
      },
    });
    fireEvent.click(
      screen.getByLabelText(/I understand that only the selected columns/),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Generate reviewed variables" }),
    );

    await waitFor(() => {
      const call = vi
        .mocked(fetch)
        .mock.calls.find(([url]) =>
          String(url).endsWith("/semantic-features/generate"),
        );
      expect(call).toBeDefined();
      const body = JSON.parse(String(call?.[1]?.body));
      expect(body.provider_id).toBe("typesafe_jev");
      expect(body.model).toBe("jev-latest");
      expect(body.proposal.questions[1]).toEqual(
        expect.objectContaining({
          input_columns: ["message"],
          question: expect.objectContaining({
            type: "choice",
            criteria: { billing: "Billing", technical: null },
          }),
        }),
      );
    });
  });

  it("displays help tooltips explaining question types", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Support" });
    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });
    fireEvent.click(await screen.findByLabelText("message"));
    fireEvent.change(screen.getByLabelText("Modeling objective"), {
      target: { value: "Predict churn" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Suggest questions with AI" }),
    );
    await screen.findByDisplayValue("Does the customer intend to cancel?");

    const newTypeTooltip = screen.getByRole("button", {
      name: "More information about New question type",
    });
    expect(newTypeTooltip).toBeInTheDocument();
    fireEvent.click(newTypeTooltip);
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "Noul: Binary yes/no question producing 1 probability feature",
    );

    const questionTypeTooltip = screen.getByRole("button", {
      name: "More information about noul",
    });
    expect(questionTypeTooltip).toBeInTheDocument();
  });

  it("shows saved questions, output location, and the lack of training comparison", async () => {
    historyItems = [
      {
        id: "set-completed",
        dataset_id: "dataset-1",
        source_experiment_id: null,
        name: "Semantic variable set",
        status: "completed",
        row_count: 10,
        processed_rows: 10,
        request_count: 4,
        cache_hits: 6,
        input_tokens: 100,
        estimated_cost: 0.01,
        latency_ms_total: 500,
        requested_model: "jev-latest",
        resolved_model: "jev-1.13",
        artifact_path:
          "decision/semantic-features/set-completed/features.parquet",
        impact: null,
        proposal: {
          questions: [
            {
              id: "intent",
              input_columns: ["message"],
              question: {
                type: "noul",
                instructions: "Does the customer intend to cancel?",
                criteria: null,
              },
            },
          ],
        },
      },
    ];
    renderPage();

    expect(
      await screen.findByText("Semantic variable set"),
    ).toBeInTheDocument();
    expect(screen.getByText("Support · 1 question")).toBeInTheDocument();
    expect(
      screen.getByText(
        /no training comparison ran because no experiment was selected/,
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByText("View questions, file, and details"));
    expect(
      screen.getByText("Does the customer intend to cancel?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "decision/semantic-features/set-completed/features.parquet",
      ),
    ).toBeInTheDocument();
  });

  it("highlights running generation with an accessible progress indicator", async () => {
    historyItems = [
      {
        id: "set-running",
        dataset_id: "dataset-1",
        source_experiment_id: null,
        name: "Running semantic set",
        status: "running",
        row_count: 20,
        processed_rows: 10,
        request_count: 2,
        cache_hits: 8,
        input_tokens: null,
        estimated_cost: null,
        latency_ms_total: null,
        requested_model: "jev-latest",
        resolved_model: null,
        artifact_path: null,
        impact: null,
        proposal: { questions: [] },
      },
    ];
    renderPage();

    expect(await screen.findByText("Running semantic set")).toBeInTheDocument();
    expect(
      screen.getByText("Generating variables: 10 of 20 rows (50%)"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("progressbar", { name: "Variable generation progress" }),
    ).toHaveAttribute("aria-valuenow", "50");
  });

  it("explains a minimize metric without calling AI before opt-in", async () => {
    historyItems = [
      {
        id: "set-impact",
        dataset_id: "dataset-1",
        source_experiment_id: "experiment-1",
        name: "Impact set",
        status: "completed",
        row_count: 10,
        processed_rows: 10,
        request_count: 1,
        cache_hits: 0,
        input_tokens: 10,
        estimated_cost: 0,
        latency_ms_total: 100,
        requested_model: "jev-latest",
        resolved_model: "jev-latest",
        artifact_path: null,
        proposal: { questions: [] },
        impact: {
          models: [
            {
              model_id: "random_forest_regressor",
              primary_metric: "rmse",
              baseline_metrics: { rmse: 1 },
              augmented_metrics: { rmse: 1.2 },
              delta_absolute: 0.2,
              delta_percent: 20,
              duration_seconds: 24.26,
              semantic_feature_importance: [
                { feature: "semantic_score", importance: 0.1 },
              ],
            },
          ],
        },
      },
    ];
    renderPage();
    expect(await screen.findByText("Impact set")).toBeInTheDocument();
    expect(await screen.findByText("Numerically declined")).toBeInTheDocument();
    expect(screen.getByText("Comparison time")).toBeInTheDocument();
    expect(
      screen.queryByText(
        /Optional: clicking sends aggregate comparison context/,
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Provider fake is local/),
    ).not.toBeInTheDocument();
    expect(
      vi
        .mocked(fetch)
        .mock.calls.some(([url]) =>
          String(url).endsWith("/semantic-features/set-impact/explanation"),
        ),
    ).toBe(false);
    fireEvent.click(
      screen.getByRole("button", { name: "Explain this comparison with AI" }),
    );
    expect(
      await screen.findByText("The metric declined in this evaluation."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Check prediction-time availability."),
    ).toBeInTheDocument();
    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url]) =>
        String(url).endsWith("/semantic-features/set-impact/explanation"),
      );
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ language: "en-US" });
  });

  it("shows the local TF-IDF arm when a three-way impact is available", async () => {
    historyItems = [
      {
        id: "set-three-arms",
        dataset_id: "dataset-1",
        source_experiment_id: "experiment-1",
        name: "Three-way impact",
        status: "completed",
        row_count: 10,
        processed_rows: 10,
        request_count: 1,
        cache_hits: 0,
        input_tokens: 10,
        estimated_cost: 0,
        latency_ms_total: 100,
        requested_model: "jev-latest",
        resolved_model: "jev-latest",
        artifact_path: null,
        proposal: { questions: [] },
        impact: {
          models: [
            {
              model_id: "logistic_regression",
              primary_metric: "rmse",
              baseline_metrics: { rmse: 1.0 },
              text_baseline_metrics: { rmse: 0.9 },
              augmented_metrics: { rmse: 0.8 },
              delta_absolute: -0.2,
              delta_semantic_vs_text: -0.1,
              delta_percent: -20,
              duration_seconds: 3,
              baseline_duration_seconds: 1,
              text_baseline_duration_seconds: 1,
              augmented_duration_seconds: 1,
              semantic_feature_importance: [],
            },
          ],
        },
      },
    ];
    renderPage();
    expect(await screen.findByText("Three-way impact")).toBeInTheDocument();
    expect(screen.getByText("With local text (TF-IDF)")).toBeInTheDocument();
    expect(screen.getByText("JEV − TF-IDF")).toBeInTheDocument();
    expect(
      screen.getByText(/Three evaluations with the same folds/),
    ).toBeInTheDocument();
  });
});
