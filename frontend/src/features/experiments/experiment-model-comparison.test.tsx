import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../../i18n";
import { ExperimentModelComparison } from "./experiment-model-comparison";

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  });
}

const mockComparisonRound1 = {
  experiment_id: "exp-123",
  selected_round_id: "exec-1",
  available_rounds: [
    {
      execution_id: "exec-1",
      started_at: "2026-09-24T10:00:00Z",
      finished_at: "2026-09-24T10:02:00Z",
      total_models: 3,
      completed_models: 2,
    },
    {
      execution_id: "exec-0",
      started_at: "2026-09-23T10:00:00Z",
      finished_at: "2026-09-23T10:01:00Z",
      total_models: 1,
      completed_models: 1,
    },
  ],
  primary_metric: "roc_auc",
  metric_direction: "maximize",
  protocol: {
    strategy: "k_fold",
    folds: 5,
    holdout_ratio: null,
    random_seed: 42,
  },
  ranked_models: [
    {
      model_id: "lightgbm_classifier",
      run_id: "run-lgbm",
      run_index: 0,
      rank: 1,
      primary_metric_value: 0.8421,
      delta_to_best: 0.0,
      standard_deviation: 0.015,
      duration_seconds: 12.4,
      secondary_metrics: { accuracy: 0.785 },
      parameters: { n_estimators: 100 },
    },
    {
      model_id: "logistic_regression",
      run_id: "run-lr",
      run_index: 1,
      rank: 2,
      primary_metric_value: 0.7915,
      delta_to_best: -0.0506,
      standard_deviation: 0.021,
      duration_seconds: 4.2,
      secondary_metrics: { accuracy: 0.742 },
      parameters: { C: 1.0 },
    },
  ],
  excluded_models: [
    {
      model_id: "random_forest_classifier",
      run_id: "run-rf",
      run_index: 2,
      status: "failed",
      reason: "status_failed",
    },
  ],
  can_explain: true,
};

const mockComparisonRound0 = {
  experiment_id: "exp-123",
  selected_round_id: "exec-0",
  available_rounds: [
    {
      execution_id: "exec-1",
      started_at: "2026-09-24T10:00:00Z",
      finished_at: "2026-09-24T10:02:00Z",
      total_models: 3,
      completed_models: 2,
    },
    {
      execution_id: "exec-0",
      started_at: "2026-09-23T10:00:00Z",
      finished_at: "2026-09-23T10:01:00Z",
      total_models: 1,
      completed_models: 1,
    },
  ],
  primary_metric: "roc_auc",
  metric_direction: "maximize",
  protocol: {
    strategy: "k_fold",
    folds: 5,
    holdout_ratio: null,
    random_seed: 42,
  },
  ranked_models: [
    {
      model_id: "logistic_regression",
      run_id: "run-lr-old",
      run_index: 0,
      rank: 1,
      primary_metric_value: 0.772,
      delta_to_best: 0.0,
      standard_deviation: 0.019,
      duration_seconds: 3.5,
      secondary_metrics: {},
      parameters: {},
    },
  ],
  excluded_models: [],
  can_explain: false,
};

const mockAvailability = {
  available: true,
  provider_id: "ollama",
  is_local: true,
};

const mockExplanationResult = {
  provider_id: "ollama",
  is_local: true,
  explanation: {
    summary:
      "O modelo LightGBM superou a Regressão Logística em 0.0506 de ROC AUC.",
    findings: [
      "LightGBM obteve melhor discriminação global (0.8421 vs 0.7915).",
      "A variabilidade entre os 5 folds foi baixa (desvio padrão 0.0150).",
    ],
    next_steps: [
      "Testar ajuste fino de hiperparâmetros no LightGBM.",
      "Avaliar calibração de probabilidades antes do deploy.",
    ],
    cautions: [
      "Resultados baseados em validação cruzada no dataset atual sem split de teste cego.",
    ],
  },
};

function renderComponent() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <ExperimentModelComparison experimentId="exp-123" />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ExperimentModelComparison", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("pt-BR");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders deterministic comparison table without calling LLM", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/explanations/status")) {
        return jsonResponse(mockAvailability);
      }
      if (url.includes("/comparison")) {
        return jsonResponse(mockComparisonRound1);
      }
      return jsonResponse({});
    });
    vi.stubGlobal("fetch", fetchMock);

    renderComponent();

    // Verify title and primary metric badge
    expect(
      await screen.findByText("Comparação de modelos"),
    ).toBeInTheDocument();
    expect(screen.getByText("roc_auc")).toBeInTheDocument();
    expect(screen.getByText("Maior é melhor")).toBeInTheDocument();

    // Verify ranked models rows
    expect(
      screen.getByTestId("comparison-row-lightgbm_classifier"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("comparison-row-logistic_regression"),
    ).toBeInTheDocument();
    expect(screen.getByText("0.8421")).toBeInTheDocument();
    expect(screen.getByText("-0.0506")).toBeInTheDocument();
    expect(screen.getByText("±0.0150")).toBeInTheDocument();
    expect(screen.getByText("12.4s")).toBeInTheDocument();
    expect(screen.getByText("Melhor")).toBeInTheDocument();

    // Verify excluded models toggle exists
    expect(
      screen.getByRole("button", { name: /Modelos desconsiderados/i }),
    ).toBeInTheDocument();

    // Verify LLM explanation was NOT called on mount
    expect(
      fetchMock.mock.calls.some(
        (call) =>
          String(call[0]).includes("/explanation") &&
          !String(call[0]).includes("/status"),
      ),
    ).toBe(false);
  });

  it("switches round via dropdown and updates comparison data", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/explanations/status")) {
        return jsonResponse(mockAvailability);
      }
      if (url.includes("execution_id=exec-0")) {
        return jsonResponse(mockComparisonRound0);
      }
      if (url.includes("/comparison")) {
        return jsonResponse(mockComparisonRound1);
      }
      return jsonResponse({});
    });
    vi.stubGlobal("fetch", fetchMock);

    renderComponent();

    await screen.findByText("Comparação de modelos");
    const select = screen.getByTestId("comparison-round-select");
    expect(select).toBeInTheDocument();

    // Change round to exec-0
    await act(async () => {
      fireEvent.change(select, { target: { value: "exec-0" } });
    });

    // In round 0 there is only 1 eligible model -> shows insufficient banner
    expect(
      await screen.findByTestId("insufficient-models-banner"),
    ).toBeInTheDocument();

    // AI button should be disabled because can_explain is false
    const explainButton = screen.getByTestId("explain-comparison-button");
    expect(explainButton).toBeDisabled();
  });

  it("expands excluded models section on click", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/explanations/status")) {
        return jsonResponse(mockAvailability);
      }
      if (url.includes("/comparison")) {
        return jsonResponse(mockComparisonRound1);
      }
      return jsonResponse({});
    });
    vi.stubGlobal("fetch", fetchMock);

    renderComponent();

    await screen.findByText("Comparação de modelos");
    const toggleButton = screen.getByRole("button", {
      name: /Modelos desconsiderados/i,
    });
    expect(
      screen.queryByTestId("excluded-row-random_forest_classifier"),
    ).not.toBeInTheDocument();

    fireEvent.click(toggleButton);

    expect(
      screen.getByTestId("excluded-row-random_forest_classifier"),
    ).toBeInTheDocument();
    expect(screen.getByText("Treinamento falhou")).toBeInTheDocument();
  });

  it("calls AI explanation on click and displays structured output", async () => {
    let explanationCalled = false;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/explanations/status")) {
        return jsonResponse(mockAvailability);
      }
      if (url.includes("/comparison")) {
        return jsonResponse(mockComparisonRound1);
      }
      if (url.includes("/explanation")) {
        explanationCalled = true;
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body));
        expect(body.execution_id).toBe("exec-1");
        expect(body.language).toBe("pt-BR");
        return jsonResponse(mockExplanationResult);
      }
      return jsonResponse({});
    });
    vi.stubGlobal("fetch", fetchMock);

    renderComponent();

    await screen.findByText("Comparação de modelos");
    const explainButton = screen.getByTestId("explain-comparison-button");
    expect(explainButton).toBeEnabled();

    await act(async () => {
      fireEvent.click(explainButton);
    });

    expect(explanationCalled).toBe(true);

    // Verify structured explanation components
    expect(
      await screen.findByTestId("ai-comparison-result"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "O modelo LightGBM superou a Regressão Logística em 0.0506 de ROC AUC.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "LightGBM obteve melhor discriminação global (0.8421 vs 0.7915).",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Resultados baseados em validação cruzada no dataset atual sem split de teste cego.",
      ),
    ).toBeInTheDocument();

    // Button should now show "Regenerar explicação"
    expect(screen.getByText("Regenerar explicação")).toBeInTheDocument();
  });

  it("resets explanation when language changes", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/explanations/status")) {
        return jsonResponse(mockAvailability);
      }
      if (url.includes("/comparison")) {
        return jsonResponse(mockComparisonRound1);
      }
      if (url.includes("/explanation")) {
        const body = JSON.parse(String(init?.body));
        return jsonResponse({
          ...mockExplanationResult,
          explanation: {
            ...mockExplanationResult.explanation,
            summary:
              body.language === "en-US"
                ? "LightGBM outperformed Logistic Regression."
                : mockExplanationResult.explanation.summary,
          },
        });
      }
      return jsonResponse({});
    });
    vi.stubGlobal("fetch", fetchMock);

    renderComponent();

    await screen.findByText("Comparação de modelos");
    const explainButton = screen.getByTestId("explain-comparison-button");

    await act(async () => {
      fireEvent.click(explainButton);
    });

    expect(
      await screen.findByTestId("ai-comparison-result"),
    ).toBeInTheDocument();

    // Change language to en-US -> explanation should reset
    await act(async () => {
      await i18n.changeLanguage("en-US");
    });

    expect(
      screen.queryByTestId("ai-comparison-result"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("explain-comparison-button")).toHaveTextContent(
      "Explain comparison with AI",
    );
  });

  it("displays error state when explanation request fails", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/explanations/status")) {
        return jsonResponse(mockAvailability);
      }
      if (url.includes("/comparison")) {
        return jsonResponse(mockComparisonRound1);
      }
      if (url.includes("/explanation")) {
        return jsonResponse({ detail: { code: "llm_timeout" } }, 502);
      }
      return jsonResponse({});
    });
    vi.stubGlobal("fetch", fetchMock);

    renderComponent();

    await screen.findByText("Comparação de modelos");
    const explainButton = screen.getByTestId("explain-comparison-button");

    await act(async () => {
      fireEvent.click(explainButton);
    });

    expect(
      await screen.findByTestId("ai-comparison-error"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Não foi possível gerar a explicação comparativa."),
    ).toBeInTheDocument();
  });
});
