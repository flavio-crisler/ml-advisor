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
import { DecisionBenchmarkPage } from "./decision-benchmark-page";

function renderPage() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter initialEntries={["/decision-lab/benchmark"]}>
        <DecisionBenchmarkPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Zero-Shot Benchmark", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en-US");
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) => {
        let payload: unknown = [];
        if (input.endsWith("/datasets")) {
          payload = [
            {
              id: "dataset-1",
              name: "Churn",
              original_filename: "churn.csv",
              hash: "hash",
              row_count: 2,
              column_count: 2,
              created_at: "2026-09-21T00:00:00Z",
            },
          ];
        } else if (input.endsWith("/datasets/dataset-1")) {
          payload = {
            dataset: { id: "dataset-1", name: "Churn" },
            profile: {
              candidate_targets: ["label", "amount"],
              candidate_target_distributions: {
                label: [{ value: "yes" }, { value: "no" }],
                amount: [
                  { value: "10.0" },
                  { value: "11.0" },
                  { value: "12.0" },
                  { value: "13.0" },
                  { value: "15.0" },
                  { value: "16.0" },
                  { value: "17.0" },
                  { value: "18.0" },
                  { value: "19.0" },
                  { value: "20.0" },
                ],
              },
              column_names: ["message", "label", "amount", "Return_Flag"],
              column_profiles: [
                {
                  name: "label",
                  source_dtype: "object",
                  cardinality: 2,
                  top_values: [{ value: "yes" }, { value: "no" }],
                },
                {
                  name: "amount",
                  source_dtype: "float64",
                  cardinality: 11,
                  top_values: [
                    { value: "10.0" },
                    { value: "11.0" },
                    { value: "12.0" },
                    { value: "13.0" },
                    { value: "15.0" },
                    { value: "16.0" },
                    { value: "17.0" },
                    { value: "18.0" },
                    { value: "19.0" },
                    { value: "20.0" },
                  ],
                },
                {
                  name: "Return_Flag",
                  source_dtype: "object",
                  cardinality: 2,
                  top_values: [{ value: "Yes" }, { value: "No" }],
                },
              ],
              preview: [
                { amount: 10 },
                { amount: 11 },
                { amount: 12 },
                { amount: 13 },
                { amount: 15 },
                { amount: 16 },
                { amount: 17 },
                { amount: 18 },
                { amount: 19 },
                { amount: 20 },
                { amount: 1000 },
              ],
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
        } else if (init?.method === "POST") {
          payload = {
            decision_run_id: "run-1",
            job_id: "job-1",
            status: "queued",
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

  it("keeps the target out of provider input columns", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Churn" });
    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });
    await screen.findByRole("option", { name: "label" });
    fireEvent.change(
      await screen.findByLabelText("Outcome to predict (target)"),
      {
        target: { value: "label" },
      },
    );
    fireEvent.click(await screen.findByLabelText("message"));
    expect(screen.queryByLabelText("label")).not.toBeInTheDocument();
    fireEvent.change(
      screen.getByLabelText("Question the model should answer"),
      {
        target: { value: "Will this customer churn?" },
      },
    );
    fireEvent.click(
      screen.getByText(/I understand that the selected columns/)
        .previousElementSibling!,
    );
    const submit = screen.getByRole("button", { name: "Start comparison" });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);
    await screen.findByText("Comparison added to the queue.");

    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url, init]) =>
        Boolean(
          String(url).endsWith("/decision-benchmarks") &&
          init?.method === "POST",
        ),
      );
    const body = JSON.parse(String(call?.[1]?.body));
    expect(body.spec.input_columns).toEqual(["message"]);
    expect(body.spec.target).toBe("label");
  });

  it("offers compatible targets even when profiling did not suggest them", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Churn" });
    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });

    const returnFlag = await screen.findByRole("option", {
      name: "Return_Flag",
    });
    expect(returnFlag).toBeEnabled();
    expect(
      screen.getByRole("group", { name: "Other compatible columns" }),
    ).toContainElement(returnFlag);
  });

  it("recovers every class from a legacy truncated target profile", async () => {
    renderPage();
    await screen.findByRole("option", { name: "Churn" });
    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "dataset-1" },
    });
    await screen.findByRole("option", { name: "amount" });
    fireEvent.change(
      await screen.findByLabelText("Outcome to predict (target)"),
      {
        target: { value: "amount" },
      },
    );
    fireEvent.click(await screen.findByLabelText("message"));
    fireEvent.change(
      screen.getByLabelText("Question the model should answer"),
      {
        target: { value: "What is the amount?" },
      },
    );
    fireEvent.click(
      screen.getByText(/I understand that the selected columns/)
        .previousElementSibling!,
    );

    const submit = screen.getByRole("button", { name: "Start comparison" });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);
    await screen.findByText("Comparison added to the queue.");

    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url, init]) =>
        Boolean(
          String(url).endsWith("/decision-benchmarks") &&
          init?.method === "POST",
        ),
      );
    const body = JSON.parse(String(call?.[1]?.body));
    expect(body.spec.task).toBe("multiclass_classification");
    expect(Object.keys(body.spec.question.criteria)).toHaveLength(11);
    expect(body.spec.question.criteria).toHaveProperty("1000.0", null);
  });

  it("displays prominent active status and progress for running and queued benchmarks", async () => {
    const runningRun = {
      id: "run-running-1",
      dataset_id: "dataset-1",
      experiment_id: null,
      provider_id: "openrouter_jev",
      requested_model: "typesafe/jev-1.13-20260917",
      resolved_model: null,
      question_spec_hash: "hash",
      status: "running",
      row_count: 100,
      processed_rows: 45,
      request_count: 45,
      cache_hits: 5,
      input_tokens: 1200,
      estimated_cost: 0.05,
      latency_ms_total: 1500,
      artifact_path: null,
      calibration_run_id: null,
      error_message: null,
      created_at: "2026-09-24T00:00:00Z",
      started_at: "2026-09-24T00:01:00Z",
      finished_at: null,
      spec: {
        dataset_id: "dataset-1",
        target: "label",
        task: "binary_classification",
        input_columns: ["message"],
        positive_class: "yes",
        question: {
          kind: "noul",
          instructions: "Will this customer churn?",
          criteria: {},
        },
      },
    };

    const queuedRun = {
      ...runningRun,
      id: "run-queued-1",
      status: "queued",
      processed_rows: 0,
      spec: {
        ...runningRun.spec,
        question: {
          ...runningRun.spec.question,
          instructions: "Is this passenger a survivor?",
        },
      },
    };

    vi.mocked(fetch).mockImplementation((input: string | URL | Request) => {
      const url = String(input);
      let payload: unknown = [];
      if (url.endsWith("/datasets")) {
        payload = [{ id: "dataset-1", name: "Churn" }];
      } else if (url.endsWith("/decision-providers")) {
        payload = [
          {
            id: "typesafe_jev",
            selected: true,
            available: true,
            model: "jev-latest",
          },
        ];
      } else if (url.endsWith("/decision-runs")) {
        payload = [runningRun, queuedRun];
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(payload),
      } as Response);
    });

    renderPage();

    await screen.findByText("Will this customer churn?");
    expect(
      screen.getByText("Is this passenger a survivor?"),
    ).toBeInTheDocument();

    const runningBadge = screen.getByText("Running");
    expect(runningBadge).toHaveClass("bg-primary", "text-on-primary");

    const queuedBadge = screen.getByText("Queued");
    expect(queuedBadge).toHaveClass("bg-primary", "text-on-primary");

    expect(
      screen.getByText("Running comparison: 45 of 100 rows (45%)"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Waiting for comparison to start"),
    ).toBeInTheDocument();

    const progressbars = screen.getAllByRole("progressbar");
    expect(progressbars).toHaveLength(2);
    expect(progressbars[0]).toHaveAttribute("aria-valuenow", "45");
    expect(progressbars[1]).toHaveAttribute("aria-valuenow", "0");
  });
});
