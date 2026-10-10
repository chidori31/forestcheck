import { useState } from "react";
import { Api, ApiError, message } from "./api";
import type { Dataset, Job, Mapping, Schema } from "./types";
import { StatusNotice } from "./components";
export function Upload({
  api,
  dataset,
  onDataset,
  onJob,
  onClose,
}: {
  api: Api;
  dataset: Dataset | null;
  onDataset: (d: Dataset) => void;
  onJob: (j: Job) => Promise<void>;
  onClose: () => void;
}) {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [unsupported, setUnsupported] = useState(false),
    [mapping, setMapping] = useState<Mapping | null>(null),
    [checked, setChecked] = useState(false),
    [headers, setHeaders] = useState<string[]>([]);
  async function upload(file?: File) {
    if (!file) return;
    setBusy(true);
    setError("");
    setUnsupported(false);
    try {
      const result = await api.upload(file);
      onDataset(result);
    } catch (e) {
      setUnsupported(e instanceof ApiError && e.status === 415);
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function readMapping(file?: File) {
    if (!file) return;
    setMapping(null);
    setChecked(false);
    setBusy(true);
    setError("");
    try {
      if (file.size > 2 * 1024 ** 2)
        throw new ApiError(415, "mapping_too_large");
      const value = JSON.parse(await file.text()) as Mapping;
      if (
        value.schema_version !== "1.0" ||
        !Array.isArray(value.columns) ||
        !value.columns.length
      )
        throw new ApiError(415, "unsupported_mapping");
      if (!dataset) throw new ApiError(422, "upload_first");
      const schema = await api.request<Schema>(
        `/datasets/${dataset.id}/schema`,
        {
          schema_version: "1.0",
          encoding: value.encoding,
          delimiter: value.delimiter,
        },
      );
      if (
        JSON.stringify(schema.headers) !==
        JSON.stringify(value.columns.map((c) => c.source_header))
      )
        throw new ApiError(422, "mapping_header_mismatch");
      setHeaders(schema.headers);
      setMapping(value);
    } catch (e) {
      setUnsupported(e instanceof ApiError && e.status === 415);
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function confirm() {
    if (!dataset || !mapping) return;
    setBusy(true);
    setError("");
    try {
      const j = await api.request<Job>(`/datasets/${dataset.id}/mapping`, {
        schema_version: "1.0",
        expected_version: dataset.etag,
        mapping,
      });
      await onJob(j);
      onClose();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="upload-panel" aria-label="Загрузка и mapping">
      <header>
        <div>
          <span className="eyebrow">ИМПОРТ / ЯВНЫЙ КОНТРАКТ</span>
          <h2>Набор и единицы измерения</h2>
        </div>
        <button onClick={onClose} aria-label="Закрыть загрузку">
          ×
        </button>
      </header>
      <p>
        Источник сохраняется неизменным. Поддерживается CSV до 64 MiB; архивы и
        книги с макросами не принимаются.
      </p>
      <div className="upload-steps">
        <label>
          <b>01 · Исходная таблица</b>
          <input
            aria-label="CSV файл"
            type="file"
            accept=".csv"
            disabled={busy}
            onChange={(e) => upload(e.target.files?.[0])}
          />
        </label>
        <label>
          <b>02 · Mapping JSON</b>
          <input
            aria-label="Mapping JSON"
            type="file"
            accept=".json"
            disabled={busy || !dataset || dataset.state !== "uploaded"}
            onChange={(e) => readMapping(e.target.files?.[0])}
          />
        </label>
      </div>
      {mapping && (
        <>
          <p>
            Уровень: <b>{mapping.record_level}</b> · mapping{" "}
            {mapping.mapping_version} · {mapping.encoding}
          </p>
          <div className="mapping-scroll">
            <table>
              <thead>
                <tr>
                  <th>№</th>
                  <th>Заголовок источника</th>
                  <th>Поле</th>
                  <th>Единица</th>
                  <th>Преобразование</th>
                </tr>
              </thead>
              <tbody>
                {mapping.columns.map((c, i) => (
                  <tr key={c.position}>
                    <td>{c.position}</td>
                    <td>{headers[i]}</td>
                    <td>{c.field}</td>
                    <td>{c.unit ?? "Не установлена"}</td>
                    <td>{c.transform}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {mapping.analysis_blockers?.length > 0 && (
            <p className="warning">
              Ограничения: {mapping.analysis_blockers.join("; ")}
            </p>
          )}
          <label className="check">
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
            />
            Сверены позиции, единицы и уровень записи; ограничения сохранены
          </label>
          <button
            className="primary"
            disabled={busy || !checked}
            onClick={confirm}
          >
            Подтвердить mapping и импортировать
          </button>
        </>
      )}
      {busy && (
        <StatusNotice
          state="loading"
          text="Загрузка или импорт. Ожидание worker…"
        />
      )}
      {error && (
        <StatusNotice
          state={unsupported ? "unsupported" : "error"}
          text={error}
        />
      )}
    </section>
  );
}
