import { expect, test } from "@playwright/test";
import path from "node:path";

const attendanceCsv = path.resolve("../backend/tests/fixtures/attendance.csv");

test("main screens render and navigation has no JavaScript or API 5xx errors", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(`JavaScript: ${error.message}`));
  page.on("response", (response) => {
    if (response.url().includes("/api/") && response.status() >= 500) {
      errors.push(
        `HTTP ${response.status()}: ${new URL(response.url()).pathname}`,
      );
    }
  });
  await page.addInitScript(() => {
    localStorage.setItem("uiLanguage", "en-US");
    localStorage.setItem("llmLanguage", "en-US");
  });

  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      name: "Start by importing or choosing a dataset.",
    }),
  ).toBeVisible();
  await page.getByLabel("Select CSV file").setInputFiles(attendanceCsv);
  await expect(page).toHaveURL(/\/datasets\/[\w-]+$/);
  await expect(page.getByText("Column profile")).toBeVisible();

  await page.getByRole("button", { name: "Configure manually" }).click();
  await expect(
    page.getByRole("heading", { name: "Configure every experiment stage" }),
  ).toBeVisible();
  await page.getByLabel("Objective").fill("Smoke-test missed appointments");
  await page.getByRole("button", { name: "Create advanced draft" }).click();
  await expect(page).toHaveURL(/\/experiments\/[\w-]+$/);
  await expect(page.getByRole("button", { name: "Train" })).toBeVisible();
  await page.getByRole("button", { name: "Train" }).click();
  await expect(
    page.getByText("Completed", { exact: true }).first(),
  ).toBeVisible({
    timeout: 90_000,
  });
  await page
    .getByRole("button")
    .filter({ hasText: "logistic_regression" })
    .click();
  await expect(page.locator("#metrics")).toBeVisible();
  await expect(
    page.getByRole("navigation", { name: "Result sections" }),
  ).toBeVisible();

  await page.goto("/experiments");
  await expect(
    page.getByRole("heading", { name: "Experiments and results" }),
  ).toBeVisible();
  await page.goto("/decision-lab");
  await expect(
    page.getByRole("heading", { name: "AI Decisions" }),
  ).toBeVisible();
  await page.getByRole("link", { name: /Compare zero-shot/ }).click();
  await expect(
    page.getByRole("heading", { name: "Compare a zero-shot model" }),
  ).toBeVisible();
  await page.getByLabel("Dataset").selectOption({ label: "attendance" });
  await page.getByLabel("Outcome to predict (target)").selectOption("target");
  await page.getByRole("checkbox", { name: "active", exact: true }).check();
  await page
    .getByLabel("Question the model should answer")
    .fill("Is this customer active?");
  await page.getByLabel(/I understand that the selected columns/).check();
  await page.getByRole("button", { name: "Start comparison" }).click();
  await expect(page.getByText("Completed", { exact: true })).toBeVisible({
    timeout: 90_000,
  });
  await page
    .getByRole("link")
    .filter({ hasText: "Is this customer active?" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Calibration Lab" }),
  ).toBeVisible();
  await expect(page.getByText("Brier Score")).toBeVisible();

  await page.goto("/decision-lab/semantic-features");
  await expect(
    page.getByRole("heading", { name: "Create variables with AI" }),
  ).toBeVisible();
  await page.goto("/decision-lab/cascades");
  await expect(
    page.getByRole("heading", { name: "Set a confidence policy" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});
