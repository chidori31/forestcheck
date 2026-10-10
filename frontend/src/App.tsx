import { useCallback, useEffect, useRef, useState } from "react";
import { Api, ApiError, message } from "./api";
import { Evidence } from "./Evidence";
import { Grid } from "./Grid";
import type { Selection } from "./Grid";
import { jobs, StatusNotice } from "./components";
import { Upload } from "./Upload";
import type {
  Dataset,
  Schema,
  DataRecord,
  Detail,
  Draft,
  FindingItem,
  Job,
  Page,
  Profile,
  Review,
  Version,
} from "./types";
type Session = { operator: string; data_mode: "real" | "synthetic" };
const blank: Draft = {
  assessment: "uncertain",
  decision: "not_considered",
  comment: "",
  suggestionId: null,
};
export default function App() {
  const [connection, setConnection] = useState<{
      api: Api;
      session: Session;
    } | null>(null),
    [base, setBase] = useState("http://127.0.0.1:8000"),
    [token, setToken] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  if (connection)
    return (
      <Workspace
        api={connection.api}
        session={connection.session}
        disconnect={() => setConnection(null)}
      />
    );
  return (
    <div className="login">
      <div className="login-rail">ЛК</div>
      <main>
        <span className="eyebrow">FORESTCHECK / ЛОКАЛЬНЫЙ АТЛАС</span>
        <h1>ЛесКонтроль</h1>
        <p>Рабочее место проверки лесотаксационных сведений</p>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            setError("");
            try {
              const api = new Api(base, token);
              const session = await api.request<Session>("/session");
              setConnection({ api, session });
              setToken("");
            } catch (e) {
              setError(message(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          <label>
            Локальный API
            <input
              aria-label="Локальный API"
              value={base}
              onChange={(e) => setBase(e.target.value)}
              autoComplete="off"
            />
          </label>
          <label>
            Токен оператора
            <input
              aria-label="Токен оператора"
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              autoComplete="off"
              required
            />
          </label>
          <button className="primary" disabled={busy}>
            {busy ? "Подключение…" : "Открыть рабочее место"}
          </button>
        </form>
        {error && <StatusNotice state="error" text={error} />}
        <small>
          Токен хранится только в памяти вкладки. Нужны запущенные backend и
          worker.
          <br />
          Синтетический demo запускается отдельно; при сбое сервера он не
          подставляется.
        </small>
      </main>
    </div>
  );
}
function Workspace({
  api,
  session,
  disconnect,
}: {
  api: Api;
  session: Session;
  disconnect: () => void;
}) {
  const [booting, setBooting] = useState(true);
  const [datasets, setDatasets] = useState<Dataset[]>([]),
    [datasetId, setDatasetId] = useState(""),
    [schema, setSchema] = useState<Schema | null>(null),
    [versions, setVersions] = useState<Version[]>([]),
    [version, setVersion] = useState("");
  const [profiles, setProfiles] = useState<Profile[]>([]),
    [profileId, setProfileId] = useState(""),
    [useModels, setUseModels] = useState(true),
    [analyses, setAnalyses] = useState<Job[]>([]),
    [analysisId, setAnalysisId] = useState(""),
    [queue, setQueue] = useState<Job[]>([]);
  const [mode, setMode] = useState<"findings" | "records">("findings"),
    [offset, setOffset] = useState(0),
    [total, setTotal] = useState(0),
    [items, setItems] = useState<FindingItem[]>([]),
    [records, setRecords] = useState<DataRecord[]>([]),
    [selection, setSelection] = useState<Selection | null>(null),
    [detail, setDetail] = useState<Detail | null>(null);
  const [search, setSearch] = useState(""),
    [query, setQuery] = useState(""),
    [severity, setSeverity] = useState(""),
    [fieldFilter, setFieldFilter] = useState(""),
    [reviewed, setReviewed] = useState(""),
    [sort, setSort] = useState("priority"),
    [descending, setDescending] = useState(false);
  const [loading, setLoading] = useState(false),
    [detailLoading, setDetailLoading] = useState(false),
    [error, setError] = useState(""),
    [detailError, setDetailError] = useState(""),
    [upload, setUpload] = useState(false),
    [revision, setRevision] = useState(0),
    [working, setWorking] = useState(false),
    [panel, setPanel] = useState<"table" | "jobs">("table");
  const [drafts, setDrafts] = useState<Record<string, Draft>>({}),
    [basket, setBasket] = useState<Record<string, Review>>({}),
    [exportKind, setExportKind] = useState("report"),
    [format, setFormat] = useState("json");
  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const analysisRef = useRef(analysisId);
  analysisRef.current = analysisId;
  const evidenceViewport = useRef<HTMLElement>(null);
  useEffect(() => {
    if (evidenceViewport.current) evidenceViewport.current.scrollTop = 0;
  }, [selection?.rowKey, selection?.field]);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const dataset = datasets.find((d) => d.id === datasetId),
    analysis = analyses.find((j) => j.id === analysisId),
    columns = schema?.mapping?.columns ?? [];
  const compatible = profiles.filter(
    (p) => p.mapping_sha256 === schema?.mapping_sha256,
  );
  const reload = useCallback(async () => {
    const [ds, ps, aa, jj] = await Promise.all([
      api.request<Page<Dataset>>("/datasets?limit=200"),
      api.request<Page<Profile>>("/analyses/profiles?limit=200"),
      api.request<Page<Job>>("/analyses?limit=200"),
      api.request<Page<Job>>("/jobs?limit=200"),
    ]);
    if (!alive.current) return;
    setDatasets(ds.items);
    setProfiles(ps.items);
    setAnalyses(aa.items);
    setQueue(jj.items);
  }, [api]);
  useEffect(() => {
    reload()
      .catch((e) => setError(message(e)))
      .finally(() => setBooting(false));
  }, [reload]);
  useEffect(() => {
    if (!datasetId && datasets.length) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);
  useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(search.trim());
      setOffset(0);
    }, 250);
    return () => clearTimeout(timer);
  }, [search]);
  useEffect(() => {
    setSchema(null);
    setVersion("");
    setAnalysisId("");
    setVersions([]);
    setSelection(null);
    setBasket({});
    setOffset(0);
    if (!datasetId) return;
    const controller = new AbortController();
    const current = datasets.find((d) => d.id === datasetId);
    if (!current?.current_version_id) return;
    Promise.all([
      api.request<Schema>(
        `/datasets/${datasetId}/schema`,
        undefined,
        controller.signal,
      ),
      api.request<Page<Version>>(
        `/datasets/${datasetId}/versions?limit=200`,
        undefined,
        controller.signal,
      ),
    ])
      .then(([s, v]) => {
        setSchema(s);
        setVersions(v.items);
        setVersion(current.current_version_id!);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      });
    return () => controller.abort();
  }, [api, datasetId, datasets]);
  useEffect(() => {
    setProfileId(compatible[0]?.id ?? "");
  }, [schema, profiles]); // exact server mapping binding
  useEffect(() => {
    setSelection(null);
    setDetail(null);
    setRecords([]);
    setItems([]);
    setTotal(0);
    setError("");
    if (
      !version ||
      (mode === "findings" && (!analysisId || analysis?.status !== "succeeded"))
    ) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    (async () => {
      if (mode === "records") {
        const rows = await api.request<Page<DataRecord>>(
          `/versions/${version}/records?offset=${offset}&limit=100`,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setRecords(rows.items);
          setTotal(rows.total);
        }
        return;
      }
      const params = new URLSearchParams({
        analysis_id: analysisId,
        offset: String(offset),
        limit: "100",
        sort,
        descending: String(descending),
      });
      if (query) params.set("search", query);
      if (severity) params.set("severity", severity);
      if (fieldFilter) params.set("field_name", fieldFilter);
      if (reviewed) params.set("reviewed", reviewed);
      const page = await api.request<Page<FindingItem>>(
        "/findings?" + params,
        undefined,
        controller.signal,
      );
      const rows = page.items.length
        ? await api.request<Page<DataRecord>>(
            `/versions/${version}/records/query`,
            {
              schema_version: "1.0",
              record_ids: [
                ...new Set(page.items.map((x) => x.finding.record_id)),
              ],
            },
            controller.signal,
          )
        : { items: [] };
      if (!controller.signal.aborted) {
        setItems(page.items);
        setRecords(rows.items);
        setTotal(page.total);
      }
    })()
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [
    api,
    version,
    analysisId,
    analysis?.status,
    mode,
    offset,
    query,
    severity,
    fieldFilter,
    reviewed,
    sort,
    descending,
    revision,
  ]);
  const loadDetail = useCallback(
    async (id: string, signal?: AbortSignal) => {
      const d = await api.request<Detail>(`/findings/${id}`, undefined, signal);
      if (!signal?.aborted && selectionRef.current?.findingId === id) {
        setDetail(d);
        setDrafts((old) =>
          old[id]
            ? old
            : {
                ...old,
                [id]: {
                  assessment: "uncertain",
                  decision: d.suggestions.length
                    ? "deferred"
                    : "not_considered",
                  comment: "",
                  suggestionId: d.suggestions[0]?.suggestion.id ?? null,
                },
              },
        );
      }
      return d;
    },
    [api],
  );
  useEffect(() => {
    setDetail(null);
    setDetailError("");
    if (!selection?.findingId) {
      setDetailLoading(false);
      return;
    }
    const controller = new AbortController();
    setDetailLoading(true);
    loadDetail(selection.findingId, controller.signal)
      .catch((e) => {
        if (!controller.signal.aborted) setDetailError(message(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [selection, loadDetail]);
  useEffect(() => {
    if (
      !analysisId ||
      working ||
      !analysis ||
      !["queued", "running"].includes(analysis.status)
    )
      return;
    const controller = new AbortController();
    const timer = setInterval(() => {
      api
        .request<Job>(`/jobs/${analysisId}`, undefined, controller.signal)
        .then((j) => {
          setAnalyses((old) => old.map((x) => (x.id === j.id ? j : x)));
          setQueue((old) => [j, ...old.filter((x) => x.id !== j.id)]);
        })
        .catch((e) => {
          if (!controller.signal.aborted) setError(message(e));
        });
    }, 1500);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, [api, analysisId, analysis, working]);
  async function runJob(job: Job) {
    setQueue((old) => [job, ...old.filter((x) => x.id !== job.id)]);
    let current = job;
    while (alive.current && ["queued", "running"].includes(current.status)) {
      await new Promise((r) => setTimeout(r, 800));
      if (!alive.current) throw new DOMException("Disconnected", "AbortError");
      current = await api.request<Job>(`/jobs/${job.id}`);
      setQueue((old) => [current, ...old.filter((x) => x.id !== current.id)]);
      if (current.kind === "analysis")
        setAnalyses((old) => [
          current,
          ...old.filter((x) => x.id !== current.id),
        ]);
    }
    if (current.status !== "succeeded")
      throw new ApiError(422, current.error_code ?? current.status);
    return current;
  }
  async function action(work: () => Promise<void>) {
    setWorking(true);
    setError("");
    try {
      await work();
    } catch (e) {
      setError(message(e));
    } finally {
      if (alive.current) setWorking(false);
    }
  }
  async function startAnalysis() {
    await action(async () => {
      const job = await api.request<Job>("/analyses", {
        schema_version: "1.0",
        version_id: version,
        profile_id: profileId,
        use_models: useModels,
      });
      setAnalyses((old) => [job, ...old.filter((x) => x.id !== job.id)]);
      setAnalysisId(job.id);
      const done = await runJob(job);
      setAnalyses((old) => [done, ...old.filter((x) => x.id !== done.id)]);
      setMode("findings");
      setOffset(0);
      setRevision((x) => x + 1);
    });
  }
  async function saveReview() {
    if (!detail) throw new ApiError(422, "select_finding");
    const draft = drafts[detail.finding.id];
    const event = await api.request<Review>("/reviews", {
      schema_version: "1.0",
      finding_id: detail.finding.id,
      expected_version: detail.review_version,
      error_assessment: draft.assessment,
      suggestion_id: draft.suggestionId,
      suggestion_decision: draft.decision,
      comment: draft.comment || null,
    });
    if (analysisRef.current === analysisId)
      setBasket((old) => {
        const next = { ...old };
        if (event.review.suggestion_decision === "accepted")
          next[detail.finding.id] = event;
        else delete next[detail.finding.id];
        return next;
      });
    await loadDetail(detail.finding.id);
    setItems((old) =>
      old.map((x) =>
        x.finding.id === detail.finding.id
          ? { ...x, review_version: event.sequence, latest_review: event }
          : x,
      ),
    );
  }
  async function newVersion() {
    if (!dataset) return;
    await action(async () => {
      const current = await api.request<Dataset>(`/datasets/${dataset.id}`);
      const done = await runJob(
        await api.request<Job>(`/datasets/${dataset.id}/versions`, {
          schema_version: "1.0",
          base_version_id: version,
          analysis_id: analysisId,
          expected_version: current.etag,
          review_event_ids: Object.values(basket).map((r) => r.id),
        }),
      );
      if (typeof done.result?.version_id === "string") {
        await reload();
        setBasket({});
        setExportKind("corrected");
        setMode("records");
      }
    });
  }
  async function exportData() {
    await action(async () => {
      const done = await runJob(
        await api.request<Job>("/exports", {
          schema_version: "1.0",
          version_id: version,
          kind: exportKind,
          format,
          analysis_id: exportKind === "report" ? analysisId : null,
        }),
      );
      await api.download(done.id, format);
    });
  }
  function selectAnalysis(id: string) {
    const job = analyses.find((j) => j.id === id);
    setAnalysisId(id);
    if (job?.request?.version_id) setVersion(job.request.version_id);
    setBasket({});
    setOffset(0);
    setMode("findings");
  }
  const selectedRecord = records.find((r) => r.id === selection?.recordId);
  return (
    <div className="workspace">
      <nav className="rail" aria-label="Инструменты">
        <b>ЛК</b>
        <button
          title="Таблица"
          aria-label="Таблица"
          aria-pressed={panel === "table"}
          onClick={() => setPanel("table")}
        >
          ▦
        </button>
        <button
          title="Очередь заданий"
          aria-label="Очередь заданий"
          aria-pressed={panel === "jobs"}
          onClick={() => setPanel("jobs")}
        >
          ≡
        </button>
        <span />
        <button
          title="Отключиться"
          aria-label="Отключиться"
          onClick={disconnect}
        >
          ↗
        </button>
      </nav>
      <header className="topbar">
        <div>
          <strong>ЛесКонтроль</strong>
          <span> / Атлас проверки</span>
        </div>
        {session.data_mode === "synthetic" && (
          <b className="demo-label">Синтетические данные</b>
        )}
        <small>{session.operator} · локальный сервер</small>
        <button disabled={booting || working} onClick={() => setUpload(true)}>
          ＋ Загрузить набор
        </button>
      </header>
      <aside className="sidebar">
        <span className="eyebrow">ОБЪЕКТЫ / НАБОРЫ</span>
        <div className="dataset-list">
          {datasets.length ? (
            datasets.map((d) => (
              <button
                key={d.id}
                className={d.id === datasetId ? "current" : ""}
                title={d.name}
                onClick={() => {
                  setDatasetId(d.id);
                  setUpload(false);
                }}
              >
                <span>▤ {d.name}</span>
                <small>
                  {d.state === "ready"
                    ? "Импортирован"
                    : d.state === "uploaded"
                      ? "Mapping не подтверждён"
                      : "Импорт: см. очередь"}
                </small>
              </button>
            ))
          ) : (
            <p>{booting ? "Загрузка наборов…" : "Наборов пока нет"}</p>
          )}
        </div>
        <label>
          Версия
          <select
            value={version}
            onChange={(e) => {
              setVersion(e.target.value);
              setAnalysisId("");
              setBasket({});
              setOffset(0);
            }}
          >
            <option value="">Не выбрана</option>
            {versions.map((v) => (
              <option key={v.id} value={v.id}>
                Версия {v.number}
                {v.parent_id ? " · исправленная" : " · исходная"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Анализ
          <select
            aria-label="Анализ"
            value={analysisId}
            onChange={(e) => selectAnalysis(e.target.value)}
          >
            <option value="">Не выбран</option>
            {analyses
              .filter(
                (j) =>
                  versions.some((v) => v.id === j.request?.version_id) ||
                  j.id === analysisId,
              )
              .map((j) => (
                <option key={j.id} value={j.id}>
                  {jobs[j.status] ?? j.status} · {j.id.slice(0, 8)}
                </option>
              ))}
          </select>
        </label>
        <div className="sidebar-section">
          <span className="eyebrow">ПРОФИЛЬ ПРОВЕРКИ</span>
          <select
            aria-label="Профиль проверки"
            value={profileId}
            onChange={(e) => setProfileId(e.target.value)}
          >
            <option value="">Нет совместимого профиля</option>
            {compatible.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} · {p.version}
              </option>
            ))}
          </select>
          <label className="check">
            <input
              type="checkbox"
              checked={useModels}
              onChange={(e) => setUseModels(e.target.checked)}
            />
            Подключить доступные модели
          </label>
          <button
            className="primary"
            disabled={!version || !profileId || working}
            onClick={startAnalysis}
          >
            Запустить анализ
          </button>
        </div>
        <div className="sidebar-section">
          <span className="eyebrow">ФИЛЬТРЫ ЗАМЕЧАНИЙ</span>
          <label>
            Серьёзность
            <select
              value={severity}
              onChange={(e) => {
                setSeverity(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">Все уровни</option>
              <option value="info">Информация</option>
              <option value="warning">Замечание</option>
              <option value="error">Ошибка</option>
              <option value="critical">Критично</option>
            </select>
          </label>
          <label>
            Поле
            <select
              value={fieldFilter}
              onChange={(e) => {
                setFieldFilter(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">Все поля</option>
              {columns.map((c) => (
                <option key={c.field} value={c.field}>
                  {c.source_header}
                </option>
              ))}
            </select>
          </label>
          <label>
            Рассмотрение
            <select
              value={reviewed}
              onChange={(e) => {
                setReviewed(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">Все состояния</option>
              <option value="false">Без решений</option>
              <option value="true">Есть история решений</option>
            </select>
          </label>
        </div>
        <div className="scope-note">
          Проверка ограничена применимыми правилами и зарегистрированной
          когортой. Сигнал не равен доказанной ошибке.
        </div>
      </aside>
      <main className="canvas">
        <div className="canvas-title">
          <div>
            <span className="eyebrow">
              ВЕДОМОСТЬ /{" "}
              {versions.find((v) => v.id === version)?.number ?? "—"}
            </span>
            <h1>{dataset?.name ?? "Рабочее пространство"}</h1>
          </div>
          <button onClick={() => action(reload)} disabled={working}>
            Обновить
          </button>
        </div>
        {upload || dataset?.state === "uploaded" ? (
          <Upload
            key={dataset?.id ?? "new"}
            api={api}
            dataset={dataset ?? null}
            onDataset={(d) => {
              setDatasets((old) => [d, ...old.filter((x) => x.id !== d.id)]);
              setDatasetId(d.id);
            }}
            onJob={async (j) => {
              await runJob(j);
              await reload();
            }}
            onClose={() => setUpload(false)}
          />
        ) : (
          <>
            {panel === "jobs" ? (
              <section className="jobs-panel">
                <h2>Очередь и результаты заданий</h2>
                {[
                  ...queue,
                  ...analyses.filter((a) => !queue.some((q) => q.id === a.id)),
                ].map((j) => (
                  <div className="job" key={j.id}>
                    <b>
                      {j.kind} · {jobs[j.status] ?? j.status}
                    </b>
                    <small>{j.id}</small>
                    {j.error_code && <p role="alert">{j.error_code}</p>}
                    {["queued", "running"].includes(j.status) && (
                      <button
                        onClick={() =>
                          action(async () => {
                            const done = await api.request<Job>(
                              `/jobs/${j.id}/cancel`,
                              {},
                            );
                            setQueue((old) => [
                              done,
                              ...old.filter((x) => x.id !== done.id),
                            ]);
                          })
                        }
                      >
                        Отменить задание
                      </button>
                    )}
                    {j.kind === "export" && j.status === "succeeded" && (
                      <button
                        onClick={() =>
                          action(() =>
                            api.download(j.id, j.request.format ?? format),
                          )
                        }
                      >
                        Скачать экспорт
                      </button>
                    )}
                  </div>
                ))}
              </section>
            ) : (
              <>
                <div className="table-tools">
                  <div className="tabs">
                    <button
                      aria-pressed={mode === "findings"}
                      onClick={() => {
                        setMode("findings");
                        setOffset(0);
                      }}
                    >
                      Замечания
                    </button>
                    <button
                      aria-pressed={mode === "records"}
                      onClick={() => {
                        setMode("records");
                        setOffset(0);
                      }}
                    >
                      Значения версии
                    </button>
                  </div>
                  <input
                    aria-label="Поиск по ID или коду поля"
                    placeholder="ID записи / код поля"
                    value={search}
                    disabled={mode !== "findings"}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                  <select
                    aria-label="Сортировка"
                    value={sort}
                    disabled={mode !== "findings"}
                    onChange={(e) => {
                      setSort(e.target.value);
                      setOffset(0);
                    }}
                  >
                    <option value="priority">По приоритету</option>
                    <option value="field">По полю</option>
                    <option value="record">По ID строки</option>
                    <option value="signal">По сигналу</option>
                  </select>
                  <button
                    aria-label="Изменить направление сортировки"
                    disabled={mode !== "findings"}
                    onClick={() => {
                      setDescending((x) => !x);
                      setOffset(0);
                    }}
                  >
                    {descending ? "↓" : "↑"}
                  </button>
                </div>
                {loading ? (
                  <StatusNotice state="loading" />
                ) : analysis?.status === "failed" && mode === "findings" ? (
                  <StatusNotice
                    state="analysis_failed"
                    text={`Анализ завершился ошибкой: ${analysis.error_code ?? "причина не указана"}. Результаты не подменяются демонстрационными.`}
                  />
                ) : error ? (
                  <StatusNotice state="error" text={error} />
                ) : !version ? (
                  <StatusNotice
                    state="empty"
                    text="Загрузите CSV и подтвердите mapping, чтобы открыть таблицу."
                  />
                ) : mode === "findings" && !analysisId ? (
                  <StatusNotice
                    state="empty"
                    text="Выберите или запустите анализ. Исходные строки доступны отдельно."
                  />
                ) : mode === "findings" && analysis?.status === "cancelled" ? (
                  <StatusNotice
                    state="empty"
                    text="Анализ отменён. Результаты отсутствуют."
                  />
                ) : mode === "findings" && analysis?.status !== "succeeded" ? (
                  <StatusNotice
                    state="loading"
                    text={`${jobs[analysis?.status ?? "queued"]}. Дождитесь отдельного worker.`}
                  />
                ) : !total ? (
                  <StatusNotice state="empty" />
                ) : (
                  <Grid
                    records={records}
                    items={items}
                    columns={columns}
                    selection={selection}
                    onSelect={setSelection}
                    mode={mode}
                    total={total}
                    offset={offset}
                  />
                )}
                <footer className="pagination">
                  <span>
                    {total
                      ? `${offset + 1}–${Math.min(offset + 100, total)} из ${total}`
                      : "0 записей"}{" "}
                    · {mode === "findings" ? "сигналы" : "строки снимка"}
                  </span>
                  <button
                    disabled={!offset || loading}
                    onClick={() => setOffset((x) => Math.max(0, x - 100))}
                  >
                    ← Страница
                  </button>
                  <button
                    disabled={offset + 100 >= total || loading}
                    onClick={() => setOffset((x) => x + 100)}
                  >
                    Страница →
                  </button>
                </footer>
              </>
            )}
            <section className="export-bar" aria-label="Новая версия и экспорт">
              <button
                disabled={!Object.keys(basket).length || working}
                onClick={newVersion}
              >
                Создать версию из выбранных решений (
                {Object.keys(basket).length})
              </button>
              <select
                aria-label="Вид экспорта"
                value={exportKind}
                onChange={(e) => setExportKind(e.target.value)}
              >
                <option value="report">Отчёт анализа</option>
                <option value="original">Оригинал</option>
                <option value="corrected">Исправленная версия</option>
              </select>
              <select
                aria-label="Формат экспорта"
                value={format}
                onChange={(e) => setFormat(e.target.value)}
              >
                <option>json</option>
                <option>csv</option>
                <option>xlsx</option>
              </select>
              <button
                className="primary"
                disabled={
                  working ||
                  !version ||
                  (exportKind === "report" && !analysisId)
                }
                onClick={exportData}
              >
                Экспорт
              </button>
            </section>
            {error && panel === "jobs" && (
              <StatusNotice state="error" text={error} />
            )}
            <div className="job-status" aria-live="polite">
              {working
                ? `Выполняется: ${jobs[queue[0]?.status ?? "queued"]}. Процент не измеряется.`
                : queue[0]
                  ? `${queue[0].kind}: ${jobs[queue[0].status]}`
                  : "Локальный режим · исходники неизменны"}
            </div>
          </>
        )}
      </main>
      <aside
        ref={evidenceViewport}
        className="evidence-panel"
        aria-label="Панель выбранного поля"
      >
        {detailLoading ? (
          <StatusNotice state="loading" />
        ) : detailError ? (
          <StatusNotice state="error" text={detailError} />
        ) : selectedRecord && selection ? (
          <Evidence
            key={selection.findingId ?? selection.recordId + selection.field}
            record={selectedRecord}
            field={selection.field}
            detail={detail}
            columns={columns}
            draft={drafts[selection.findingId ?? ""] ?? blank}
            onDraft={(draft) =>
              setDrafts((old) => ({
                ...old,
                [selection.findingId ?? ""]: draft,
              }))
            }
            onSave={saveReview}
            onRefresh={async () => {
              if (selection.findingId) await loadDetail(selection.findingId);
            }}
          />
        ) : (
          <div className="evidence-empty">
            <span className="eyebrow">ДОКАЗАТЕЛЬСТВА</span>
            <h2>Выберите ячейку</h2>
            <p>
              Исходное значение, основание сигнала и возможная замена появятся
              здесь.
            </p>
            <p>
              ↑ ↓ — строки
              <br />← → — поля
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}
