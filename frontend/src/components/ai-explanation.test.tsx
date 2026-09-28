import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import i18n from "../i18n";
import { AIExplanation } from "./ai-explanation";

function response(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

function renderPanel() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <AIExplanation
        scope="model_run"
        context={{ model: "logistic_regression", accuracy: 0.91 }}
      />
    </QueryClientProvider>,
  );
}

describe("AIExplanation", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("pt-BR");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("sends the aggregate context only after an explicit click in the active language", async () => {
    const fetchMock = vi.fn((input: string, init?: RequestInit) => {
      if (input.endsWith("/status")) {
        return response({
          available: true,
          provider_id: "ollama",
          is_local: true,
        });
      }
      expect(init?.method).toBe("POST");
      return response({
        provider_id: "ollama",
        is_local: true,
        explanation: {
          summary: "O modelo teve bom desempenho.",
          findings: ["A acurácia foi 0,91."],
          cautions: ["Confirme em dados novos."],
          next_steps: ["Revise os erros."],
        },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPanel();

    expect(await screen.findByText(/contexto permanece/i)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Gerar explicação" }));

    expect(
      await screen.findByText("O modelo teve bom desempenho."),
    ).toBeInTheDocument();
    const request = JSON.parse(
      String(fetchMock.mock.calls[1]?.[1]?.body),
    ) as Record<string, unknown>;
    expect(request).toMatchObject({
      scope: "model_run",
      language: "pt-BR",
      context: { model: "logistic_regression", accuracy: 0.91 },
    });

    await act(() => i18n.changeLanguage("en-US"));
    expect(
      screen.getByRole("button", { name: "Generate explanation" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("O modelo teve bom desempenho."),
    ).not.toBeInTheDocument();
  });

  it("keeps the action disabled when no provider is configured", async () => {
    const fetchMock = vi.fn(() =>
      response({
        available: false,
        provider_id: "openrouter",
        is_local: false,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    renderPanel();

    expect(
      await screen.findByText(/configure LLM_PROVIDER e LLM_MODEL/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Gerar explicação" }),
    ).toBeDisabled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("requires proposal review and creates a draft without starting training", async () => {
    await i18n.changeLanguage("en-US");
    const recommendation = {
      schema_version: "1.0",
      scope: "supervised_experiment",
      summary: "Compare a regularized baseline.",
      rationale: ["C stays within the supported range."],
      models: [
        {
          id: "logistic_regression",
          enabled: true,
          parameters: { C: 0.5 },
        },
      ],
      preprocessing: [],
      validation: {
        strategy: "stratified_kfold",
        test_size: 0.2,
        folds: 5,
        shuffle: true,
        random_seed: 41,
      },
      metrics: {
        primary: "f1",
        secondary: [],
        averaging: "macro",
        roc_auc_strategy: null,
        positive_label: "yes",
      },
      tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
      cautions: ["Compare on the same split."],
      unsupported_suggestions: ["Isotonic calibration is unavailable."],
    };
    const fetchMock = vi.fn((input: string, init?: RequestInit) => {
      if (input.endsWith("/status")) {
        return response({
          available: true,
          provider_id: "ollama",
          is_local: true,
        });
      }
      if (input.endsWith("/explanations")) {
        return response({
          provider_id: "ollama",
          is_local: true,
          explanation: {
            summary: "The result can be improved.",
            findings: [],
            cautions: [],
            next_steps: [],
          },
        });
      }
      if (input.endsWith("/supervised/draft")) {
        return response({
          experiment_id: "draft-1",
          parent_experiment_id: "experiment-1",
          name: "AI proposal",
          status: "draft",
        });
      }
      if (input.endsWith("/supervised")) {
        return response({
          schema_version: "1.0",
          source_experiment_id: "experiment-1",
          source_model_run_id: "run-1",
          source_spec_hash: "hash",
          recommendation,
          proposed_spec: {},
          changes: [
            { path: "models", before: '[{"C":1}]', after: '[{"C":0.5}]' },
          ],
          provider_id: "ollama",
          is_local: true,
        });
      }
      throw new Error(`Unexpected request: ${input} ${init?.method ?? "GET"}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <MemoryRouter initialEntries={["/runs/run-1"]}>
          <Routes>
            <Route
              path="/runs/run-1"
              element={
                <AIExplanation
                  scope="model_run"
                  context={{ accuracy: 0.8 }}
                  improvementSource={{
                    experimentId: "experiment-1",
                    modelRunId: "run-1",
                  }}
                />
              }
            />
            <Route
              path="/experiments/draft-1"
              element={<p>Draft destination</p>}
            />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText(/context stays/i)).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Generate explanation" }),
    );
    expect(
      await screen.findByText("The result can be improved."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Compare a regularized baseline."),
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Create improvement proposal" }),
    );
    expect(
      await screen.findByText("Compare a regularized baseline."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Isotonic calibration is unavailable."),
    ).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).endsWith("/run")),
    ).toBe(false);

    fireEvent.click(
      screen.getByRole("button", { name: "Create experiment draft" }),
    );
    expect(await screen.findByText("Draft destination")).toBeInTheDocument();
    const draftCall = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/supervised/draft"),
    );
    expect(draftCall?.[1]?.method).toBe("POST");
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).endsWith("/run")),
    ).toBe(false);
  });
});
