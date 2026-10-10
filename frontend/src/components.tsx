import { useState } from "react";
import { ApiError, message } from "./api";
import type { Draft, FindingItem, Scalar } from "./types";
export const display = (value: Scalar | undefined) =>
  value === null || value === undefined
    ? "—"
    : value === ""
      ? '""'
      : String(value);
export const signals: Record<string, string> = {
  missing: "∅ Пропуск",
  anomaly: "△ Необычное значение",
  parse_error: "! Формат",
  contradiction: "≠ Противоречие",
  inconsistency: "≠ Противоречие",
  reference_mismatch: "↔ Несоответствие эталону",
  invalid_code: "? Код",
  duplicate: "↔ Ключ",
};
export const jobs: Record<string, string> = {
  queued: "В очереди",
  running: "Выполняется",
  succeeded: "Завершён",
  failed: "Ошибка анализа",
  cancelled: "Отменён",
};
export function reviewLabel(item: FindingItem) {
  const review = item.latest_review;
  if (!review || review.action === "revoke") return "Подозрение";
  const values = [];
  if (review.review.error_assessment === "confirmed_error")
    values.push("Подтверждённая ошибка");
  else if (review.review.error_assessment === "no_error")
    values.push("Допустимое исключение");
  else values.push("Не удалось проверить");
  if (review.review.suggestion_decision === "rejected")
    values.push("Предложение отклонено");
  if (review.review.suggestion_decision === "accepted")
    values.push("Значение принято");
  return values.join(" · ");
}
const states = {
  loading: "Загрузка данных…",
  empty: "Нет записей для выбранных условий.",
  error: "Не удалось получить данные. Повторите запрос.",
  unsupported: "Формат или версия не поддерживается.",
  missing_reference:
    "Подходящая таблица роста не применена. Кривая не строится: нет подтверждённого эталона.",
  analysis_failed:
    "Анализ завершился ошибкой. Результаты не подменяются демонстрационными.",
};
export function StatusNotice({
  state,
  text,
}: {
  state: keyof typeof states;
  text?: string;
}) {
  return (
    <div
      className={`notice ${state}`}
      role={
        state === "error" || state === "analysis_failed" ? "alert" : "status"
      }
    >
      <span aria-hidden="true">{state === "loading" ? "◌" : "!"}</span>
      <p>{text ?? states[state]}</p>
    </div>
  );
}
export function ReviewEditor({
  draft,
  onChange,
  canAccept,
  onSave,
  onRefresh,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
  canAccept: boolean;
  onSave: () => Promise<void>;
  onRefresh: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [conflict, setConflict] = useState(false),
    [saved, setSaved] = useState(false);
  function change(next: Draft) {
    setSaved(false);
    onChange(next);
  }
  async function save() {
    setBusy(true);
    setError("");
    setSaved(false);
    try {
      await onSave();
      setSaved(true);
    } catch (e) {
      setConflict(e instanceof ApiError && e.status === 409);
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="review-form" aria-label="Решение специалиста">
      <h3>Решение специалиста</h3>
      <label>
        Наличие ошибки
        <select
          aria-label="Наличие ошибки"
          value={draft.assessment}
          onChange={(e) =>
            change({
              ...draft,
              assessment: e.target.value as Draft["assessment"],
              decision:
                draft.decision === "accepted" ? "deferred" : draft.decision,
            })
          }
        >
          <option value="uncertain">Не удалось проверить</option>
          <option value="confirmed_error">Подтверждённая ошибка</option>
          <option value="no_error">Допустимое исключение — ошибки нет</option>
        </select>
      </label>
      <label>
        Предложенное значение
        <select
          aria-label="Решение по предложению"
          value={draft.decision}
          onChange={(e) =>
            change({
              ...draft,
              decision: e.target.value as Draft["decision"],
            })
          }
        >
          <option value="not_considered">Не рассматривалось</option>
          <option value="deferred" disabled={!draft.suggestionId}>
            Отложить
          </option>
          <option
            value="accepted"
            disabled={!canAccept || draft.assessment !== "confirmed_error"}
          >
            Принять предложенное значение
          </option>
          <option value="rejected" disabled={!draft.suggestionId}>
            Отклонить предложение
          </option>
        </select>
      </label>
      <label>
        Комментарий специалиста
        <textarea
          value={draft.comment}
          maxLength={2000}
          onChange={(e) => change({ ...draft, comment: e.target.value })}
        />
      </label>
      {conflict ? (
        <div role="alert" className="conflict">
          <b>Версия рассмотрения изменилась. Ваш ввод сохранён.</b>
          <p>{error}</p>
          <button
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await onRefresh();
                setConflict(false);
                setError("");
              } catch (e) {
                setError(message(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            Сверить свежую версию
          </button>
        </div>
      ) : error ? (
        <StatusNotice state="error" text={error} />
      ) : null}
      <button className="primary" disabled={busy || conflict} onClick={save}>
        {busy ? "Сохранение…" : "Сохранить решение"}
      </button>
      {saved && <p role="status">Решение сохранено. Исходник не изменён.</p>}
      <small>Принятие значения отдельно от создания новой версии данных.</small>
    </section>
  );
}
