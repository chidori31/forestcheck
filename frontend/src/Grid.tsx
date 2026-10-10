import { useEffect, useRef, useState } from "react";
import type { Column, DataRecord, FindingItem } from "./types";
import { display, signals, reviewLabel } from "./components";
export type Selection = {
  recordId: string;
  field: string;
  findingId?: string;
  rowKey: string;
};
export function Grid({
  records,
  items,
  columns,
  selection,
  onSelect,
  mode,
  total,
  offset,
}: {
  records: DataRecord[];
  items: FindingItem[];
  columns: Column[];
  selection: Selection | null;
  onSelect: (s: Selection) => void;
  mode: "findings" | "records";
  total: number;
  offset: number;
}) {
  const viewport = useRef<HTMLDivElement>(null),
    [scroll, setScroll] = useState(0),
    [height, setHeight] = useState(500);
  useEffect(() => {
    const node = viewport.current;
    if (!node) return;
    const observer = new ResizeObserver(() => {
      setHeight(node.clientHeight);
      const cell = node.querySelector(".active-cell");
      if (cell) {
        const bounds = node.getBoundingClientRect(),
          active = cell.getBoundingClientRect();
        if (active.right > bounds.right)
          node.scrollLeft += active.right - bounds.right;
        else if (active.left < bounds.left + 224)
          node.scrollLeft -= bounds.left + 224 - active.left;
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (viewport.current) viewport.current.scrollTop = 0;
    setScroll(0);
  }, [mode, offset, records]);
  const byId = new Map(records.map((r) => [r.id, r]));
  const rows =
    mode === "findings"
      ? items.map((item) => ({
          key: item.finding.id,
          record: byId.get(item.finding.record_id),
          item,
        }))
      : records.map((record) => ({
          key: record.id,
          record,
          item: undefined as FindingItem | undefined,
        }));
  const start = Math.max(0, Math.floor((scroll - 72) / 38) - 3),
    end = Math.min(rows.length, start + Math.ceil(height / 38) + 8);
  const template = `54px 170px ${columns.map(() => "150px").join(" ")}`;
  function choose(index: number, field?: string) {
    const row = rows[index];
    if (!row?.record) return;
    const name =
      field ?? row.item?.finding.field_name ?? columns[0]?.field ?? "";
    const columnIndex = columns.findIndex((c) => c.field === name);
    const node = viewport.current;
    if (node && columnIndex >= 0) {
      const left = 224 + columnIndex * 150;
      if (left < node.scrollLeft + 224) node.scrollLeft = left - 224;
      else if (left + 150 > node.scrollLeft + node.clientWidth)
        node.scrollLeft = left + 150 - node.clientWidth;
    }
    onSelect({
      rowKey: row.key,
      recordId: row.record.id,
      field: name,
      findingId:
        row.item &&
        (row.item.finding.field_name === name || !row.item.finding.field_name)
          ? row.item.finding.id
          : undefined,
    });
  }
  return (
    <div
      ref={viewport}
      className="grid-viewport"
      role="grid"
      aria-label="Лесотаксационная таблица"
      aria-rowcount={total + 1}
      aria-colcount={columns.length + 2}
      tabIndex={0}
      onScroll={(e) => setScroll(e.currentTarget.scrollTop)}
      onKeyDown={(e) => {
        if (
          ![
            "ArrowDown",
            "ArrowUp",
            "ArrowLeft",
            "ArrowRight",
            "Home",
            "End",
          ].includes(e.key)
        )
          return;
        e.preventDefault();
        const current = Math.max(
          0,
          rows.findIndex((r) => r.key === selection?.rowKey),
        );
        const next =
          e.key === "Home"
            ? 0
            : e.key === "End"
              ? rows.length - 1
              : Math.max(
                  0,
                  Math.min(
                    rows.length - 1,
                    current +
                      (e.key === "ArrowDown"
                        ? 1
                        : e.key === "ArrowUp"
                          ? -1
                          : 0),
                  ),
                );
        const col = Math.max(
          0,
          columns.findIndex((c) => c.field === selection?.field),
        );
        choose(
          next,
          e.key === "ArrowLeft" || e.key === "ArrowRight"
            ? columns[
                Math.max(
                  0,
                  Math.min(
                    columns.length - 1,
                    col + (e.key === "ArrowRight" ? 1 : -1),
                  ),
                )
              ]?.field
            : undefined,
        );
        const node = viewport.current;
        if (
          node &&
          (next * 38 < node.scrollTop ||
            next * 38 > node.scrollTop + height - 130)
        )
          node.scrollTop = Math.max(0, next * 38 - 100);
      }}
    >
      <div
        className="grid-sheet"
        style={{
          width: 224 + 150 * columns.length,
          minHeight: 72 + 38 * rows.length,
        }}
      >
        <div
          className="grid-heading"
          role="row"
          style={{ gridTemplateColumns: template }}
        >
          <div role="columnheader">№</div>
          <div role="columnheader">Сигнал / решение</div>
          {columns.map((c) => (
            <div role="columnheader" key={c.field} title={c.source_header}>
              <span>{c.source_header || `Столбец ${c.position}`}</span>
              <small>{c.unit ?? "ед. не установлена"}</small>
            </div>
          ))}
        </div>
        {rows.slice(start, end).map((row, local) => {
          const index = start + local;
          return (
            <div
              role="row"
              aria-rowindex={offset + index + 2}
              aria-selected={selection?.rowKey === row.key}
              className={`grid-row ${selection?.rowKey === row.key ? "selected" : ""}`}
              key={row.key}
              style={{ top: 72 + index * 38, gridTemplateColumns: template }}
            >
              <div role="rowheader">{offset + index + 1}</div>
              <button
                role="gridcell"
                className="signal-cell"
                title={row.item ? reviewLabel(row.item) : "Исходная строка"}
                onClick={() => choose(index)}
              >
                {row.item ? (
                  <>
                    <span>
                      {signals[row.item.finding.signal_kind] ??
                        row.item.finding.signal_kind}
                    </span>
                    <small>{reviewLabel(row.item)}</small>
                  </>
                ) : (
                  "Исходная строка"
                )}
              </button>
              {columns.map((c) => {
                const f = row.record?.fields.find((f) => f.name === c.field);
                const signal =
                  row.item?.finding.field_name === c.field
                    ? row.item.finding.signal_kind
                    : undefined;
                return (
                  <button
                    role="gridcell"
                    className={`${signal ? "signal " + signal : ""} ${selection?.rowKey === row.key && selection.field === c.field ? "active-cell" : ""}`}
                    aria-label={`${c.source_header}, строка ${offset + index + 1}: ${display(f?.state === "observed" ? f.normalized_value : f?.original_value)}`}
                    title={`${display(f?.state === "observed" ? f.normalized_value : f?.original_value)}${signal ? " · " + (signals[signal] ?? signal) : ""}`}
                    key={c.field}
                    onClick={() => choose(index, c.field)}
                  >
                    {signal && (
                      <span aria-hidden="true">
                        {signal === "missing"
                          ? "∅"
                          : signal === "anomaly"
                            ? "△"
                            : "!"}{" "}
                      </span>
                    )}
                    {display(
                      f?.state === "observed"
                        ? f.normalized_value
                        : f?.original_value,
                    )}
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
