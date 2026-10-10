import type { Column, DataRecord, Detail, Draft } from "./types";
import {
  display,
  ReviewEditor,
  reviewLabel,
  signals,
  StatusNotice,
} from "./components";
export function Evidence({
  record,
  field,
  detail,
  columns,
  draft,
  onDraft,
  onSave,
  onRefresh,
}: {
  record: DataRecord;
  field: string;
  detail: Detail | null;
  columns: Column[];
  draft: Draft;
  onDraft: (draft: Draft) => void;
  onSave: () => Promise<void>;
  onRefresh: () => Promise<void>;
}) {
  const cell = record.fields.find((f) => f.name === field),
    label = columns.find((c) => c.field === field)?.source_header ?? field;
  const recommendation =
    detail?.suggestions.find((r) => r.suggestion.id === draft.suggestionId) ??
    detail?.suggestions[0];
  const suggestion = recommendation?.suggestion;
  const name = (key: string) =>
    columns.find((c) => c.field === key)?.source_header ?? key;
  if (suggestion && suggestion.schema_version !== "2.0")
    return <StatusNotice state="unsupported" />;
  return (
    <>
      <div className="evidence-title">
        <span className="eyebrow">
          ДОКАЗАТЕЛЬСТВА / {cell?.source.row ?? "—"}
        </span>
        <h2>{label}</h2>
        <p>{detail ? reviewLabel(detail) : "Исходная ячейка"}</p>
      </div>
      <div className="evidence-body">
        <div className="comparison">
          <div>
            <small>Исходное значение</small>
            <strong>{display(cell?.original_value)}</strong>
            <span>{cell?.unit ?? "Единица не установлена"}</span>
          </div>
          <div>
            <small>Предложение</small>
            <strong>{display(suggestion?.proposed_value)}</strong>
            <span>{suggestion?.unit ?? "Не предложено"}</span>
          </div>
        </div>
        <dl className="metadata">
          <dt>Состояние</dt>
          <dd>{cell?.state}</dd>
          <dt>Нормализовано</dt>
          <dd>{display(cell?.normalized_value)}</dd>
          <dt>Координаты источника</dt>
          <dd>
            Строка {cell?.source.row}, столбец {cell?.source.column}
          </dd>
        </dl>
        {detail && (
          <section>
            <h3>{signals[detail.finding.signal_kind] ?? "Сигнал"}</h3>
            <p>
              Серьёзность: {detail.finding.severity} · приоритет{" "}
              {detail.finding.priority}
            </p>
            {detail.finding.anomaly_score != null && (
              <p>
                Anomaly score: {detail.finding.anomaly_score.value} ·{" "}
                {detail.finding.anomaly_score.scale} ·{" "}
                {detail.finding.anomaly_score.higher_is_worse
                  ? "больше — необычнее"
                  : "меньше — необычнее"}{" "}
                (не вероятность)
              </p>
            )}
            {detail.finding.evidence.map((e, i) => (
              <div className="evidence-block" key={i}>
                <p>{e.message}</p>
                <small>
                  {e.code} · {e.reference_id}
                </small>
              </div>
            ))}
          </section>
        )}
        {detail && detail.suggestions.length > 1 && (
          <label>
            Вариант предложения
            <select
              value={draft.suggestionId ?? ""}
              onChange={(e) =>
                onDraft({
                  ...draft,
                  suggestionId: e.target.value,
                  decision: "deferred",
                })
              }
            >
              {detail.suggestions.map((r) => (
                <option key={r.suggestion.id} value={r.suggestion.id}>
                  {name(r.suggestion.field_name)} ·{" "}
                  {r.suggestion.method ?? r.suggestion.status}
                </option>
              ))}
            </select>
          </label>
        )}
        {suggestion && (
          <>
            <section>
              <h3>Оценка возможного значения</h3>
              <p>{suggestion.explanation.observed}</p>
              <p>
                {suggestion.candidate_range
                  ? `Диапазон: ${suggestion.candidate_range.lower}–${suggestion.candidate_range.upper} ${suggestion.unit ?? ""}`
                  : suggestion.explanation.considered_range}
              </p>
              <p>{suggestion.uncertainty.basis}</p>
              {suggestion.refusal_reason && (
                <p className="warning">
                  Число не предложено: {suggestion.refusal_reason}
                </p>
              )}
              {suggestion.alternatives.map((a, i) => (
                <p key={i}>
                  Вариант {i + 1}: {display(a.value)} [{display(a.lower)};{" "}
                  {display(a.upper)}] {a.unit} · {a.support.method}
                </p>
              ))}
              {suggestion.unit_hypotheses.map((h, i) => (
                <p key={i}>
                  Гипотеза единицы: {h.observed_unit} → {h.candidate_unit}.{" "}
                  {h.next_action}
                </p>
              ))}
            </section>
            <section>
              <h3>Источники и применимость</h3>
              {suggestion.evidence.map((s, i) => (
                <details key={i}>
                  <summary>
                    {s.method} · {s.status}
                  </summary>
                  <p>{s.basis}</p>
                  <p>
                    Версия {s.version} · {s.source_id}
                  </p>
                  <p>Входы: {s.input_fields.map(name).join(", ") || "—"}</p>
                  <p>Ограничения: {s.limitations.join("; ") || "Не указаны"}</p>
                  <small>
                    Общие корни данных: {s.dependency_roots.join(", ")}
                  </small>
                </details>
              ))}
              <p className="muted">
                Методы на одних данных не являются независимыми подтверждениями.
              </p>
            </section>
            <section>
              <h3>Что проверить специалисту</h3>
              <ul>
                {[
                  ...suggestion.explanation.specialist_actions,
                  ...suggestion.explanation.missing,
                  ...suggestion.limitations,
                ].map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </section>
            <details>
              <summary>Версии и проверка кандидата</summary>
              <p>Набор: ревизия {suggestion.binding.dataset_revision}</p>
              <p className="mono">
                Mapping: {suggestion.binding.mapping_sha256}
              </p>
              <p className="mono">
                Snapshot: {suggestion.binding.snapshot_sha256}
              </p>
              <pre>{JSON.stringify(recommendation?.validation, null, 2)}</pre>
            </details>
          </>
        )}
        {suggestion?.evidence.some(
          (s) => s.method === "growth" && s.status === "available",
        ) ? (
          <StatusNotice
            state="unsupported"
            text="Эталон указан в evidence. API не предоставляет точки кривой и границы применимости для графика; график не построен."
          />
        ) : (
          <StatusNotice state="missing_reference" />
        )}
        {detail && (
          <>
            <div className="current-review">
              <h3>Последнее сохранённое решение</h3>
              {detail.latest_review ? (
                <>
                  <p>{reviewLabel(detail)}</p>
                  <p>
                    {detail.latest_review.operator} ·{" "}
                    {new Date(
                      detail.latest_review.review.reviewed_at,
                    ).toLocaleString("ru-RU")}
                  </p>
                  <p>{detail.latest_review.review.note || "Без комментария"}</p>
                </>
              ) : (
                <p>Ещё не рассмотрено</p>
              )}
            </div>
            <ReviewEditor
              draft={draft}
              onChange={onDraft}
              canAccept={!!recommendation?.patch}
              onSave={onSave}
              onRefresh={onRefresh}
            />
          </>
        )}
        <details>
          <summary>Исходная строка · {record.id}</summary>
          <dl>
            {record.fields.map((f) => (
              <div key={f.name}>
                <dt>{name(f.name)}</dt>
                <dd>
                  {display(f.original_value)} {f.unit}
                </dd>
              </div>
            ))}
          </dl>
        </details>
      </div>
    </>
  );
}
