import { defineConfig } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
process.env.FORESTCHECK_OPERATOR_TOKEN ??= randomUUID() + randomUUID();
process.env.FORESTCHECK_BACKEND_ROOT ??= mkdtempSync(
  join(tmpdir(), "forestcheck-ui-synthetic-"),
);
export default defineConfig({
  testDir: "e2e",
  timeout: 90000,
  workers: 1,
  fullyParallel: false,
  reporter: "list",
  outputDir: "../work/ui-e2e",
  use: {
    baseURL: "http://127.0.0.1:5173",
    viewport: { width: 1440, height: 900 },
    trace: "off",
    screenshot: "off",
  },
  webServer: [
    {
      command:
        "..\\.venv\\Scripts\\python.exe -B ..\\scripts\\ui_demo.py --port 8123",
      url: "http://127.0.0.1:8123/health",
      reuseExistingServer: false,
      timeout: 30000,
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
      timeout: 30000,
    },
  ],
});
