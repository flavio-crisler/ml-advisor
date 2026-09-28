import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../i18n";
import { AIChartExplanation } from "./ai-chart-explanation";

function response(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

describe("AIChartExplanation", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en-US");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows the global result and requests a focused deep dive for one chart", async () => {
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
          summary: "Focused summary.",
          findings: [],
          cautions: [],
          next_steps: [],
          chart_explanations: [
            {
              chart_id: "roc_curve",
              summary: "Focused ROC interpretation.",
              findings: ["AUC is 0.94."],
              cautions: ["The sample is small."],
            },
          ],
        },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <AIChartExplanation
          chartId="roc_curve"
          context={{
            model_run: { metrics: { roc_auc: 0.94 } },
            charts: [{ chart_id: "roc_curve", data: { auc: 0.94 } }],
          }}
          globalExplanation={{
            chart_id: "roc_curve",
            summary: "Overall ROC interpretation.",
            findings: [],
            cautions: [],
          }}
        />
      </QueryClientProvider>,
    );

    expect(
      await screen.findByText("Overall ROC interpretation."),
    ).toBeVisible();
    const deepenButton = screen.getByRole("button", {
      name: "Deepen with AI",
    });
    await waitFor(() => expect(deepenButton).toBeEnabled());
    fireEvent.click(deepenButton);
    expect(
      await screen.findByText("Focused ROC interpretation."),
    ).toBeVisible();

    const request = JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body)) as {
      scope: string;
      context: { charts: Array<{ chart_id: string }> };
    };
    expect(request.scope).toBe("chart_explanation");
    expect(request.context.charts).toEqual([
      { chart_id: "roc_curve", data: { auc: 0.94 } },
    ]);
    expect(screen.getByRole("button", { name: "Deepen again" })).toBeEnabled();
  });
});
