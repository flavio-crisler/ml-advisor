import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import i18n from "../../i18n";
import { HowItWorksButton } from "./how-it-works-button";

describe("HowItWorks component", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("pt-BR");
  });

  afterEach(() => {
    cleanup();
  });

  it("renders button and opens modal on click in pt-BR", () => {
    render(<HowItWorksButton screen="decisionBenchmark" />);

    const button = screen.getByRole("button", { name: /como funciona/i });
    expect(button).toBeInTheDocument();

    fireEvent.click(button);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(
      screen.getByText(/como funciona: comparar zero-shot/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/avaliação de modelos de decisão \(jev\)/i),
    ).toBeInTheDocument();
  });

  it("switches tabs between overview, technical, and steps", () => {
    render(<HowItWorksButton screen="decisionBenchmark" />);

    fireEvent.click(screen.getByRole("button", { name: /como funciona/i }));

    const technicalTab = screen.getByRole("button", {
      name: /detalhes técnicos/i,
    });
    fireEvent.click(technicalTab);
    expect(
      screen.getByText(/executa um benchmark zero-shot em lote/i),
    ).toBeInTheDocument();

    const stepsTab = screen.getByRole("button", { name: /passo a passo/i });
    fireEvent.click(stepsTab);
    expect(
      screen.getByText(/selecione o dataset que contém/i),
    ).toBeInTheDocument();
  });

  it("closes modal when clicking close button", () => {
    render(<HowItWorksButton screen="decisionBenchmark" />);

    fireEvent.click(screen.getByRole("button", { name: /como funciona/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    const closeButtons = screen.getAllByRole("button", { name: /fechar/i });
    fireEvent.click(closeButtons[0]);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders in en-US correctly", async () => {
    await i18n.changeLanguage("en-US");
    render(<HowItWorksButton screen="datasets" />);

    const button = screen.getByRole("button", { name: /how it works/i });
    expect(button).toBeInTheDocument();

    fireEvent.click(button);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(
      screen.getByText(/how it works: data & profiling/i),
    ).toBeInTheDocument();
  });
});
