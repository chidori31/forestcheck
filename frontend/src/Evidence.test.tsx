import { afterEach, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { Evidence } from "./Evidence";
import type { Detail } from "./types";
afterEach(cleanup);
it("показывает объект AnomalyScore из контракта, шкалу и направление без процентов", () => {
  const detail: Detail = JSON.parse(
    JSON.stringify({
      finding: {
        id: "f",
        record_id: "r",
        field_name: "height",
        signal_kind: "anomaly",
        severity: "warning",
        priority: 2,
        anomaly_score: {
          value: 2.5,
          scale: "robust_deviation",
          higher_is_worse: true,
        },
        evidence: [{ code: "synthetic", message: "Синтетический сигнал" }],
      },
      review_version: 0,
      latest_review: null,
      suggestions: [],
    }),
  );
  render(
    <Evidence
      record={{
        id: "r",
        revision: 1,
        business_key: null,
        fields: [
          {
            name: "height",
            original_value: "10",
            normalized_value: 10,
            unit: "m",
            state: "observed",
            source: { row: 1, column: "column_1", table: "synthetic" },
          },
        ],
      }}
      field="height"
      detail={detail}
      columns={[
        {
          position: 1,
          source_header: "Синтетическая высота",
          field: "height",
          unit: "m",
          transform: "decimal_point",
        },
      ]}
      draft={{
        assessment: "uncertain",
        decision: "not_considered",
        comment: "",
        suggestionId: null,
      }}
      onDraft={vi.fn()}
      onSave={vi.fn()}
      onRefresh={vi.fn()}
    />,
  );
  expect(screen.getByText(/Anomaly score: 2.5/).textContent).toContain(
    "robust_deviation",
  );
  expect(screen.getByText(/Anomaly score: 2.5/).textContent).toContain(
    "не вероятность",
  );
});
