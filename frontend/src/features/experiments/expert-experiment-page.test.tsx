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
import type { RegistryCatalog } from "../../services/registries";
import { ExpertExperimentForm } from "./expert-experiment-page";

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
  profile: {
    profile_version: "1.0",
    column_names: ["amount", "target"],
    candidate_targets: ["target"],
    candidate_target_distributions: {
      target: [
        { value: "yes", count: 6, percentage: 50 },
        { value: "no", count: 6, percentage: 50 },
      ],
    },
    column_profiles: [
      {
        name: "amount",
        inferred_type: "numeric",
        suggested_role: "feature",
      },
      {
        name: "target",
        inferred_type: "categorical",
        suggested_role: "target",
      },
    ],
    possible_id_columns: [],
    missing_values: { amount: 0, target: 0 },
    categorical_columns: ["target"],
    boolean_columns: [],
    numeric_columns: ["amount"],
    datetime_columns: [],
  },
} as unknown as DatasetDetail;

const catalog = {
  models: [
    {
      id: "logistic_regression",
      display_name: "Logistic Regression",
      tasks: ["binary_classification", "multiclass_classification"],
    },
    {
      id: "linear_regression",
      display_name: "Linear Regression",
      tasks: ["regression"],
    },
  ],
  metrics: [
    {
      id: "accuracy",
      display_name: "Accuracy",
      tasks: ["binary_classification", "multiclass_classification"],
      direction: "maximize",
      requires_proba: false,
      supported_averaging: [],
      roc_auc_strategies: [],
    },
    {
      id: "f1",
      display_name: "F1",
      tasks: ["binary_classification", "multiclass_classification"],
      direction: "maximize",
      requires_proba: false,
      supported_averaging: ["macro", "weighted"],
      roc_auc_strategies: [],
    },
    {
      id: "rmse",
      display_name: "RMSE",
      tasks: ["regression"],
      direction: "minimize",
      requires_proba: false,
      supported_averaging: [],
      roc_auc_strategies: [],
    },
  ],
  preprocessors: [
    {
      id: "text_vectorization",
      display_name: "Text vectorization (TF-IDF)",
      category: "text",
      transformer_kind: "feature",
      strategies: ["tfidf"],
    },
  ],
  validators: [
    {
      id: "stratified_kfold",
      display_name: "Stratified K-Fold",
      tasks: ["binary_classification", "multiclass_classification"],
      parameters: ["folds", "shuffle", "random_seed"],
    },
    {
      id: "kfold",
      display_name: "K-Fold",
      tasks: ["regression"],
      parameters: ["folds", "shuffle", "random_seed"],
    },
  ],
  tuners: [{ id: "optuna", display_name: "Optuna", supports_pruning: true }],
} as RegistryCatalog;

describe("ExpertExperimentForm", () => {
  beforeEach(async () => {
    cleanup();
    await i18n.changeLanguage("en-US");
  });

  it("creates a complete Expert draft without calling the advisor", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: "expert-1" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = new QueryClient({
      defaultOptions: { mutations: { retry: false } },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ExpertExperimentForm detail={detail} catalog={catalog} />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    fireEvent.change(screen.getByLabelText("Objective"), {
      target: { value: "Predict missed appointments" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "F1" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Create advanced draft" }),
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/experiments");
    expect(fetchMock.mock.calls[0][0]).not.toContain("advisor");
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      mode: string;
      spec: {
        target: string;
        features: string[];
        models: { id: string }[];
        metrics: { primary: string };
      };
    };
    expect(body.mode).toBe("expert");
    expect(body.spec).toMatchObject({
      target: "target",
      features: ["amount"],
      models: [{ id: "logistic_regression" }],
      metrics: { primary: "accuracy" },
    });
    expect(
      (
        JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
          spec: { metrics: { positive_label: unknown } };
        }
      ).spec.metrics.positive_label,
    ).toBe("yes");
  });

  it("infers multiclass and requires explicit TF-IDF before selecting text", () => {
    const semanticDetail = {
      ...detail,
      profile: {
        ...detail.profile,
        column_names: ["message", "segment", "target"],
        candidate_targets: ["target"],
        column_profiles: [
          {
            name: "message",
            inferred_type: "text",
            suggested_role: "feature",
          },
          {
            name: "segment",
            inferred_type: "categorical",
            suggested_role: "feature",
          },
          {
            name: "target",
            inferred_type: "categorical",
            suggested_role: "target",
            cardinality: 3,
          },
        ],
      },
    } as unknown as DatasetDetail;
    const queryClient = new QueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ExpertExperimentForm detail={semanticDetail} catalog={catalog} />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(screen.getByLabelText("Task")).toHaveValue(
      "multiclass_classification",
    );
    expect(
      screen.getByRole("checkbox", { name: "segment (categorical)" }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: "message (text: enable TF-IDF below)",
      }),
    ).toBeDisabled();
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Text vectorization (TF-IDF)" }),
    );
    const textColumn = screen.getByRole("checkbox", { name: "message (text)" });
    expect(textColumn).toBeEnabled();
    fireEvent.click(textColumn);
    expect(textColumn).toBeChecked();
  });
});
