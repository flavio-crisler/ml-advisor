import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import i18n from "../../i18n";
import { InfoTooltip } from "./info-tooltip";

describe("InfoTooltip", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en-US");
  });

  it("opens on click, associates the description, and closes with Escape", () => {
    render(<InfoTooltip label="Brier score" content="Lower is better." />);

    const trigger = screen.getByRole("button", {
      name: "More information about Brier score",
    });
    expect(trigger).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(trigger);
    const tooltip = screen.getByRole("tooltip");
    expect(tooltip).toHaveTextContent("Lower is better.");
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(trigger).toHaveAttribute("aria-describedby", tooltip.id);

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
  });

  it("closes a pinned tooltip on a second click", () => {
    render(<InfoTooltip content="Threshold guidance." />);

    const trigger = screen.getByRole("button", { name: "More information" });
    fireEvent.click(trigger);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    fireEvent.click(trigger);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("uses the active interface language in its accessible label", async () => {
    await i18n.changeLanguage("pt-BR");
    render(<InfoTooltip label="Acurácia" content="Quanto maior, melhor." />);

    expect(
      screen.getByRole("button", {
        name: "Mais informações sobre Acurácia",
      }),
    ).toBeInTheDocument();
  });
});
