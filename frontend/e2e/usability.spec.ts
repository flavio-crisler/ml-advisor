import { expect, test, type Page } from "@playwright/test";

async function expectNoPageOverflow(page: Page) {
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.documentElement.scrollWidth <=
          document.documentElement.clientWidth,
      ),
    )
    .toBe(true);
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("uiLanguage", "en-US");
    localStorage.removeItem("mlAdvisorGuideDismissed");
    localStorage.removeItem("mlAdvisorLastWorkspacePath");
  });
});

test("guides first access and remembers the last product area", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page).toHaveTitle("ML Advisor");
  await expect(
    page.getByRole("heading", {
      name: "Start by importing or choosing a dataset.",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "What you can do" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Import CSV" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "View experiments" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Explore decisions" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Hide guide" }).click();
  await expect(
    page.getByRole("button", { name: "Show starter guide" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Experiments" }).click();
  await expect(
    page.getByRole("heading", { name: "Experiments and results" }),
  ).toBeVisible();
  await expect(page.getByLabel("Search")).toBeVisible();
  await expect(page.getByLabel("Status")).toBeVisible();
  await expect(page.getByLabel("Dataset")).toBeVisible();

  await page.getByRole("button", { name: "Data" }).click();
  await expect(
    page.getByRole("button", { name: "Resume: Experiments" }),
  ).toBeVisible();
  await expectNoPageOverflow(page);
});

test("keeps navigation usable on a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expectNoPageOverflow(page);

  await page.getByRole("button", { name: "AI Decisions" }).click();
  await expect(
    page.getByRole("heading", { name: "AI Decisions" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /Connections/ })).toBeVisible();
  await expect(page.getByText("Provider registry")).toBeVisible();
  await expect(page.getByText("Zero-shot benchmark")).toBeVisible();
  await expectNoPageOverflow(page);

  await page.getByRole("link", { name: /Create AI variables/ }).click();
  await expect(
    page.getByRole("heading", { name: "Create variables with AI" }),
  ).toBeVisible();
  await expectNoPageOverflow(page);
});
