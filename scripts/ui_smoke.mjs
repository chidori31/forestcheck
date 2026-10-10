import { chromium } from "../frontend/node_modules/@playwright/test/index.mjs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
const blocked = [];
await page.route("**/*", async (route) => {
  const request = route.request();
  const url = new URL(request.url());
  const allowedRead =
    request.method() === "GET" ||
    request.method() === "OPTIONS" ||
    (request.method() === "POST" && /\/records\/query$/.test(url.pathname));
  if (url.hostname !== "127.0.0.1" || !allowedRead) {
    blocked.push("unexpected_request");
    await route.abort();
    return;
  }
  await route.continue();
});
page.on("pageerror", (e) => errors.push(e.message));
try {
  await page.goto("http://127.0.0.1:5174");
  await page.getByLabel("Локальный API").fill("http://127.0.0.1:8124");
  await page
    .getByLabel("Токен оператора")
    .fill(process.env.FORESTCHECK_OPERATOR_TOKEN);
  await page.getByRole("button", { name: "Открыть рабочее место" }).click();
  await page
    .getByLabel("Анализ")
    .locator("option")
    .nth(1)
    .waitFor({ state: "attached" });
  const id = await page
    .getByLabel("Анализ")
    .locator("option")
    .nth(1)
    .getAttribute("value");
  await page.getByLabel("Анализ").selectOption(id);
  await page.getByRole("grid").waitFor({ state: "visible" });
  await page.locator(".signal-cell").first().click();
  await page
    .getByRole("heading", { name: "Решение специалиста" })
    .waitFor({ state: "visible" });
  await page.locator(".evidence-panel").evaluate((el) => (el.scrollTop = 0));
  if (await page.getByText("Синтетические данные", { exact: true }).count())
    throw Error("Wrong mode");
  await page.screenshot({
    path: join(process.env.FORESTCHECK_UI_REPORT, "real-1440.png"),
  });
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.screenshot({
    path: join(process.env.FORESTCHECK_UI_REPORT, "real-1280.png"),
  });
  await page
    .locator(".signal-cell")
    .filter({ hasText: "Необычное значение" })
    .first()
    .click();
  await page
    .getByRole("heading", { name: "Решение специалиста" })
    .waitFor({ state: "visible" });
  await page.locator(".evidence-panel").evaluate((el) => (el.scrollTop = 0));
  await page.screenshot({
    path: join(process.env.FORESTCHECK_UI_REPORT, "real-anomaly-1280.png"),
  });
  const result = {
    anomaly_opened: true,
    status: "passed",
    page_errors: errors,
    mode: "real",
    analysis_id: id,
    visible_rows: await page.locator(".grid-row").count(),
    pagination: await page.locator(".pagination span").innerText(),
    holdout: "NOT_RUN",
    reviews: "NOT_MODIFIED",
    source: "NOT_MODIFIED",
  };
  if (errors.length || blocked.length)
    throw Error("Browser errors or unexpected mutation");
  await writeFile(
    join(process.env.FORESTCHECK_UI_REPORT, "browser.json"),
    JSON.stringify(result, null, 2),
  );
  console.log("REAL_UI_BROWSER_PASSED");
} finally {
  await browser.close();
}
