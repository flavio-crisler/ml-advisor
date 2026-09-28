import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import i18n from "./i18n";

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Routes>
          <Route path="*" element={<App />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("App", () => {
  beforeEach(async () => {
    localStorage.clear();
    await i18n.changeLanguage("en-US");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: string) => {
        const registrySizes: Record<string, number> = {
          models: 14,
          metrics: 12,
          preprocessors: 8,
          validators: 3,
          tuners: 1,
        };
        const registry = Object.entries(registrySizes).find(([name]) =>
          input.endsWith(`/${name}`),
        );
        const body =
          input.endsWith("/datasets") || input.endsWith("/experiments")
            ? []
            : registry
              ? Array.from({ length: registry[1] }, (_, index) => ({
                  id: `${registry[0]}-${index}`,
                }))
              : { status: "ok", version: "0.1.0" };
        return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
      }),
    );
  });

  afterEach(() => cleanup());

  it("renders the localized empty dataset state and healthy API", async () => {
    renderApp();

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Start by importing or choosing a dataset.",
    );
    expect(
      screen.getByRole("button", { name: "ML Advisor" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "What you can do" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "AI Decisions" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("No datasets imported yet."),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId("health-status")).toHaveTextContent(
        "API available",
      ),
    );
    expect(await screen.findByTestId("registry-summary")).toHaveTextContent(
      "Models14",
    );
    expect(screen.getByTestId("registry-summary")).toHaveTextContent(
      "Optimization methods1",
    );
  });

  it("lets a first-time user dismiss and reopen the starter guide", async () => {
    renderApp();
    expect(
      screen.getByRole("heading", { name: "What you can do" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Hide guide" }));
    expect(
      screen.queryByRole("heading", { name: "What you can do" }),
    ).not.toBeInTheDocument();
    expect(localStorage.getItem("mlAdvisorGuideDismissed")).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Show starter guide" }));
    expect(
      screen.getByRole("heading", { name: "What you can do" }),
    ).toBeInTheDocument();
  });
});
