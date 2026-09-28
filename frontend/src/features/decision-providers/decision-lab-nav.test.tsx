import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import i18n from "../../i18n";
import { DecisionLabNav } from "./decision-lab-nav";

describe("DecisionLabNav", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("pt-BR");
  });

  afterEach(() => {
    cleanup();
  });

  it("highlights Comparar zero-shot when on /decision-lab/benchmark", () => {
    render(
      <MemoryRouter initialEntries={["/decision-lab/benchmark"]}>
        <DecisionLabNav />
      </MemoryRouter>,
    );

    const benchmarkLink = screen.getByRole("link", {
      name: /comparar zero-shot/i,
    });
    expect(benchmarkLink.className).toContain("bg-primary text-white");
  });

  it("keeps Comparar zero-shot highlighted when on /decision-lab/calibration/:runId", () => {
    render(
      <MemoryRouter initialEntries={["/decision-lab/calibration/run-123"]}>
        <DecisionLabNav />
      </MemoryRouter>,
    );

    const benchmarkLink = screen.getByRole("link", {
      name: /comparar zero-shot/i,
    });
    expect(benchmarkLink.className).toContain("bg-primary text-white");
  });

  it("highlights Conexões when on /decision-lab", () => {
    render(
      <MemoryRouter initialEntries={["/decision-lab"]}>
        <DecisionLabNav />
      </MemoryRouter>,
    );

    const providersLink = screen.getByRole("link", {
      name: /conexões/i,
    });
    expect(providersLink.className).toContain("bg-primary text-white");

    const benchmarkLink = screen.getByRole("link", {
      name: /comparar zero-shot/i,
    });
    expect(benchmarkLink.className).not.toContain("bg-primary text-white");
  });
});
