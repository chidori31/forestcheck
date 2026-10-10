import { afterEach, expect, it, vi } from "vitest";
import { Api, ApiError } from "./api";
afterEach(() => vi.unstubAllGlobals());
it("запрещает внешний origin, URL с токеном и путь", () => {
  for (const url of [
    "https://example.com",
    "http://127.0.0.1/?token=x",
    "http://user:secret@127.0.0.1",
    "http://127.0.0.1/private",
  ])
    expect(() => new Api(url, "token")).toThrow(ApiError);
});
it("сетевой сбой не включает demo", async () => {
  const fetcher = vi.fn().mockRejectedValue(new TypeError("offline"));
  vi.stubGlobal("fetch", fetcher);
  await expect(
    new Api("http://127.0.0.1:8000", "token").request("/datasets"),
  ).rejects.toMatchObject({ code: "network_error" });
  expect(fetcher).toHaveBeenCalledTimes(1);
});
it("отвергает неизвестную версию контракта", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ schema_version: "99.0" }), {
        status: 200,
      }),
    ),
  );
  await expect(
    new Api("http://127.0.0.1:8000", "token").request("/datasets"),
  ).rejects.toMatchObject({ code: "unsupported_schema" });
});
it("повтор записи сохраняет ключ, токен только в заголовке", async () => {
  const fetcher = vi
    .fn()
    .mockImplementation(() =>
      Promise.resolve(new Response("{}", { status: 200 })),
    );
  vi.stubGlobal("fetch", fetcher);
  const api = new Api("http://127.0.0.1:8000", "private-token");
  await api.request("/reviews", { expected_version: 1 });
  await api.request("/reviews", { expected_version: 1 });
  const [url, options] = fetcher.mock.calls[0];
  expect(url).not.toContain("private-token");
  expect(options.headers.Authorization).toBe("Bearer private-token");
  expect(options.headers["Idempotency-Key"]).toBe(
    fetcher.mock.calls[1][1].headers["Idempotency-Key"],
  );
  expect(localStorage.length).toBe(0);
  expect(sessionStorage.length).toBe(0);
});
