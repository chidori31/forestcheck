import { test, expect } from "@playwright/test";
import { resolve } from "node:path";
test("настоящий API: upload → worker → evidence → решения → версия → экспорт", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByLabel("Локальный API").fill("http://127.0.0.1:8123");
  await page
    .getByLabel("Токен оператора")
    .fill(process.env.FORESTCHECK_OPERATOR_TOKEN!);
  await page.getByRole("button", { name: "Открыть рабочее место" }).click();
  await expect(
    page.getByText("Синтетические данные", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Наборов пока нет")).toBeVisible();
  await page.getByRole("button", { name: "＋ Загрузить набор" }).click();
  const uploaded = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/v1/datasets") && r.request().method() === "POST",
  );
  await page
    .getByLabel("CSV файл")
    .setInputFiles(resolve("public/demo/synthetic.csv"));
  const uploadResponse = await uploaded;
  expect(uploadResponse.status()).toBe(201);
  await expect(page.getByLabel("Mapping JSON")).toBeEnabled();
  await page
    .getByLabel("Mapping JSON")
    .setInputFiles(resolve("public/demo/mapping.json"));
  await expect(
    page.getByLabel(
      "Сверены позиции, единицы и уровень записи; ограничения сохранены",
    ),
  ).toBeVisible({ timeout: 10000 });
  await page
    .getByLabel(
      "Сверены позиции, единицы и уровень записи; ограничения сохранены",
    )
    .check();
  await page
    .getByRole("button", { name: "Подтвердить mapping и импортировать" })
    .click();
  await expect(
    page.getByRole("button", { name: "Запустить анализ" }),
  ).toBeEnabled({ timeout: 30000 });
  await page.getByRole("button", { name: "Запустить анализ" }).click();
  await expect(page.getByRole("grid")).toBeVisible({ timeout: 30000 });
  await expect(page.getByText("1–100 из 120")).toBeVisible();
  expect(await page.getByRole("row").count()).toBeLessThan(40);
  await page.locator(".signal-cell").first().click();
  await expect(
    page.getByRole("heading", { name: "Решение специалиста" }),
  ).toBeVisible();
  await expect(
    page.getByText(/Подходящая таблица роста не применена/),
  ).toBeVisible();
  await page.getByLabel("Наличие ошибки").selectOption("confirmed_error");
  await page.getByLabel("Решение по предложению").selectOption("accepted");
  await page
    .getByLabel("Комментарий специалиста")
    .fill("Синтетическая проверка арифметики");
  await page.getByRole("button", { name: "Сохранить решение" }).click();
  await expect(
    page.getByText("Решение сохранено. Исходник не изменён."),
  ).toBeVisible();
  await page.locator(".evidence-panel").evaluate((el) => (el.scrollTop = 0));
  await page.screenshot({ path: "../work/ui-e2e/atlas-1440.png" });
  await page.setViewportSize({ width: 1280, height: 720 });
  await expect
    .poll(async () => {
      const cell = await page.locator(".active-cell").boundingBox();
      const viewport = await page.getByRole("grid").boundingBox();
      return (
        !!cell &&
        !!viewport &&
        cell.x + cell.width <= viewport.x + viewport.width + 1
      );
    })
    .toBe(true);
  await page.screenshot({ path: "../work/ui-e2e/atlas-1280.png" });
  await page.locator(".signal-cell").nth(1).click();
  await expect(
    page.getByRole("heading", { name: "Решение специалиста" }),
  ).toBeVisible();
  await page.getByLabel("Наличие ошибки").selectOption("no_error");
  await page.getByLabel("Решение по предложению").selectOption("rejected");
  await page.getByRole("button", { name: "Сохранить решение" }).click();
  await expect(
    page.getByText("Решение сохранено. Исходник не изменён."),
  ).toBeVisible();
  const report = page.waitForEvent("download");
  await page.getByRole("button", { name: "Экспорт", exact: true }).click();
  await (await report).saveAs("../work/ui-e2e/report.json");
  await page
    .getByRole("button", { name: "Создать версию из выбранных решений (1)" })
    .click();
  await expect(page.getByText("Версия 2 · исправленная")).toBeAttached({
    timeout: 30000,
  });
  await page.getByLabel("Формат экспорта").selectOption("xlsx");
  const corrected = page.waitForEvent("download");
  await page.getByRole("button", { name: "Экспорт", exact: true }).click();
  await (await corrected).saveAs("../work/ui-e2e/corrected.xlsx");
  expect(errors).toEqual([]);
});

test("пустой фильтр, сетевой сбой, настоящий конфликт Review и неподдерживаемая загрузка", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Локальный API").fill("http://127.0.0.1:8123");
  await page
    .getByLabel("Токен оператора")
    .fill(process.env.FORESTCHECK_OPERATOR_TOKEN!);
  await page.getByRole("button", { name: "Открыть рабочее место" }).click();
  await expect(page.getByLabel("Анализ").locator("option")).toHaveCount(2);
  const id = await page
    .getByLabel("Анализ")
    .locator("option")
    .nth(1)
    .getAttribute("value");
  await page.getByLabel("Анализ").selectOption(id!);
  await expect(page.getByRole("grid")).toBeVisible();
  const detailResponse = page.waitForResponse(
    (r) => /\/findings\/[^?]+$/.test(r.url()) && r.request().method() === "GET",
  );
  await page.locator(".signal-cell").first().click();
  const detail = await (await detailResponse).json();
  await page
    .getByLabel("Комментарий специалиста")
    .fill("Сохранить черновик при конфликте");
  const response = await page.request.post(
    "http://127.0.0.1:8123/api/v1/reviews",
    {
      headers: {
        Authorization: "Bearer " + process.env.FORESTCHECK_OPERATOR_TOKEN,
        "Idempotency-Key": crypto.randomUUID(),
      },
      data: {
        schema_version: "1.0",
        finding_id: detail.finding.id,
        expected_version: detail.review_version,
        error_assessment: "uncertain",
        suggestion_id: null,
        suggestion_decision: "not_considered",
        comment: "Другое сохранённое решение синтетического теста",
      },
    },
  );
  expect(response.status()).toBe(201);
  await page.getByRole("button", { name: "Сохранить решение" }).click();
  await expect(
    page.getByText("Версия рассмотрения изменилась. Ваш ввод сохранён."),
  ).toBeVisible();
  await expect(page.getByLabel("Комментарий специалиста")).toHaveValue(
    "Сохранить черновик при конфликте",
  );
  await page.getByRole("button", { name: "Сверить свежую версию" }).click();
  await expect(
    page.getByText("Другое сохранённое решение синтетического теста"),
  ).toBeVisible();
  await expect(page.getByLabel("Комментарий специалиста")).toHaveValue(
    "Сохранить черновик при конфликте",
  );
  await page.getByRole("button", { name: "Сохранить решение" }).click();
  await expect(
    page.getByText("Решение сохранено. Исходник не изменён."),
  ).toBeVisible();
  await page.getByLabel("Поиск по ID или коду поля").fill("no-such-field");
  await expect(
    page.getByText("Нет записей для выбранных условий."),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Выберите ячейку" }),
  ).toBeVisible();
  await page.route("**/api/v1/findings?**", (route) => route.abort("failed"));
  await page.getByLabel("Поиск по ID или коду поля").fill("total");
  await expect(page.getByText(/Сервер недоступен/)).toBeVisible();
  await expect(page.getByRole("grid")).toHaveCount(0);
  await page.unroute("**/api/v1/findings?**");
  await page.getByLabel("Поиск по ID или коду поля").fill("");
  await expect(page.getByRole("grid")).toBeVisible();
  await page.getByRole("grid").focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.locator(".grid-row.selected")).toHaveCount(1);
  await page.getByRole("button", { name: "＋ Загрузить набор" }).click();
  await page.getByLabel("CSV файл").setInputFiles({
    name: "unsupported.xlsm",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("synthetic"),
  });
  await expect(page.locator(".notice.unsupported")).toBeVisible();
});

test("начальная загрузка не позволяет загрузить файл поверх устаревшего списка", async ({
  page,
}) => {
  let release = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/v1/datasets?limit=200", async (route) => {
    await gate;
    await route.continue();
  });
  await page.goto("/");
  await page.getByLabel("Локальный API").fill("http://127.0.0.1:8123");
  await page
    .getByLabel("Токен оператора")
    .fill(process.env.FORESTCHECK_OPERATOR_TOKEN!);
  await page.getByRole("button", { name: "Открыть рабочее место" }).click();
  try {
    await expect(
      page.getByRole("button", { name: "＋ Загрузить набор" }),
    ).toBeDisabled({ timeout: 1000 });
  } finally {
    release();
  }
  await expect(
    page.getByRole("button", { name: "＋ Загрузить набор" }),
  ).toBeEnabled();
  await page.unrouteAll({ behavior: "wait" });
});
