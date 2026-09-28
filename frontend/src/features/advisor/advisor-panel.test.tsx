import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../../i18n";
import type { DatasetDetail } from "../../services/datasets";
import { AdvisorPanel } from "./advisor-panel";

const proposal = {
  spec_version: "1.0" as const,
  dataset_id: "dataset-1",
  dataset_hash: "sha256:test",
  profile_version: "1.0",
  name: "Attendance - target",
  objective: "Predict missed appointments",
  task: "binary_classification" as const,
  target: "target",
  features: ["amount"],
  column_roles: { amount: "feature" as const, target: "target" as const },
  models: [{ id: "logistic_regression", enabled: true }],
  validation: {
    strategy: "stratified_kfold",
    test_size: 0.2,
    folds: 5,
    shuffle: true,
    random_seed: 41,
  },
  metrics: { primary: "f1", positive_label: "yes" },
};

const detail = {
  dataset: {
    id: "dataset-1",
    name: "Attendance",
    original_filename: "attendance.csv",
    hash: "sha256:test",
    row_count: 12,
    column_count: 2,
    created_at: "2026-09-21T12:00:00Z",
  },
  profile: {},
} as DatasetDetail;

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <AdvisorPanel detail={detail} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AdvisorPanel", () => {
  beforeEach(async () => {
    cleanup();
    localStorage.clear();
    await i18n.changeLanguage("en-US");
  });

  it("requires review before creating an assisted draft", async () => {
    const fetchMock = vi.fn().mockImplementation((input: string) => {
      if (input.endsWith("/advisor/recommend")) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              recommendation: {
                task: "binary_classification",
                target: "target",
                features: ["amount"],
                models: [{ id: "logistic_regression", enabled: true }],
                validation: proposal.validation,
                metrics: proposal.metrics,
                explanation: "Test a transparent linear baseline first.",
              },
              proposal,
              provider_id: "openrouter",
              sample_sent: false,
              sample_rows: 0,
            }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ id: "experiment-1" }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPanel();

    fireEvent.change(screen.getByLabelText("What do you want to predict?"), {
      target: { value: "Predict missed appointments" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Generate recommendation" }),
    );

    expect(
      await screen.findByText("Test a transparent linear baseline first."),
    ).toBeInTheDocument();
    expect(screen.getByText(/no dataset rows were sent/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Review and edit the proposal")).toHaveValue(
      JSON.stringify(proposal, null, 2),
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);

    fireEvent.click(
      screen.getByRole("button", { name: "Create guided draft" }),
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/experiments");
    const createBody = JSON.parse(
      fetchMock.mock.calls[1][1].body as string,
    ) as {
      mode: string;
    };
    expect(createBody.mode).toBe("assisted");
  });

  it("starts the validated Auto workflow from dataset and objective", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          experiment_id: "experiment-auto",
          job_id: "job-auto",
          job_status: "queued",
          recommendation: {
            task: "binary_classification",
            target: "target",
            features: ["amount"],
            models: [{ id: "logistic_regression", enabled: true }],
            validation: proposal.validation,
            metrics: proposal.metrics,
            explanation: "Automatic strategy",
          },
          proposal,
          provider_id: "openrouter",
          sample_sent: false,
          sample_rows: 0,
        }),
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPanel();

    fireEvent.change(screen.getByLabelText("What do you want to predict?"), {
      target: { value: "Predict missed appointments" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Train automatically" }),
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/auto/run");
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      dataset_id: string;
      objective: string;
      language: string;
    };
    expect(body).toMatchObject({
      dataset_id: "dataset-1",
      objective: "Predict missed appointments",
      language: "pt-BR",
    });
  });

  it("fills an objective example without starting any request", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    renderPanel();

    fireEvent.click(
      screen.getByRole("button", {
        name: "Predict a relevant outcome in Attendance",
      }),
    );

    expect(screen.getByLabelText("What do you want to predict?")).toHaveValue(
      "Predict a relevant outcome in Attendance",
    );
    expect(fetchMock).not.toHaveBeenCalled();
    expect(
      screen.getByText(/Nothing runs until you choose an action/i),
    ).toBeInTheDocument();
  });

  it("shows a safe explanation rather than internal validation codes from the AI", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        json: () =>
          Promise.resolve({
            detail: {
              code: "invalid_advisor_recommendation",
              issues: [
                "unknown_preprocessor:message",
                "unknown_preprocessor:segment",
                "unknown_preprocessor:target",
              ],
            },
          }),
      }),
    );
    renderPanel();

    fireEvent.change(screen.getByLabelText("What do you want to predict?"), {
      target: { value: "Predict target using the other variables" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Generate recommendation" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("No training was started");
    expect(alert).not.toHaveTextContent("unknown_preprocessor");
  });
});
