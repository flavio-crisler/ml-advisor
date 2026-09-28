import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "../../i18n";
import { DecisionLabPage } from "./decision-lab-page";

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/decision-lab"]}>
        <DecisionLabPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Decision Lab provider registry", () => {
  beforeEach(async () => {
    localStorage.clear();
    await i18n.changeLanguage("en-US");
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) =>
        Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve(
              input.endsWith("/decision-providers")
                ? [
                    {
                      schema_version: "1.0",
                      id: "typesafe_jev",
                      display_name: "TypeSafe Jev",
                      capabilities: {
                        schema_version: "1.0",
                        supports_noul: true,
                        supports_choice: true,
                        supports_score: true,
                        supports_batch: false,
                        supports_model_version_resolution: false,
                        is_external: true,
                        max_context_tokens: null,
                      },
                      is_external: true,
                      feature_enabled: false,
                      selected: true,
                      configured: false,
                      available: false,
                      model: null,
                      health_status: "disabled",
                      health_message_key:
                        "decisionProviders.status.featureDisabled",
                    },
                  ]
                : { status: "ok" },
            ),
        }),
      ),
    );
  });

  afterEach(() => cleanup());

  it("lists providers from the registry without rendering credentials", async () => {
    renderPage();

    expect(await screen.findByText("TypeSafe Jev")).toBeInTheDocument();
    expect(screen.getByText("Feature disabled")).toBeInTheDocument();
    expect(screen.getByText("External processing")).toBeInTheDocument();
    expect(screen.getByText("Yes/no answer (Noul)")).toBeInTheDocument();
    expect(screen.queryByText(/api key/i)).not.toBeInTheDocument();
  });

  it("tests the selected configured provider explicitly", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string, init?: RequestInit) =>
        Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve(
              init?.method === "POST"
                ? {
                    schema_version: "1.0",
                    status: "available",
                    message_key: "decisionProviders.status.connectionAvailable",
                    latency_ms: 12,
                  }
                : input.endsWith("/decision-providers")
                  ? [
                      {
                        schema_version: "1.0",
                        id: "typesafe_jev",
                        display_name: "TypeSafe Jev",
                        capabilities: {
                          schema_version: "1.0",
                          supports_noul: true,
                          supports_choice: true,
                          supports_score: true,
                          supports_batch: false,
                          supports_model_version_resolution: true,
                          is_external: true,
                          max_context_tokens: 64000,
                        },
                        is_external: true,
                        feature_enabled: true,
                        selected: true,
                        configured: true,
                        available: true,
                        model: "jev-latest",
                        health_status: "unknown",
                        health_message_key:
                          "decisionProviders.status.notChecked",
                      },
                    ]
                  : { status: "ok" },
            ),
        }),
      ),
    );
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: "Test connection" }),
    );

    expect(await screen.findByText(/Connection available/)).toHaveTextContent(
      "12 ms",
    );
  });
});
