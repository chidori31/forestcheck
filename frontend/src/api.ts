export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
  ) {
    super(code);
  }
}
export class Api {
  private keys = new Map<string, string>();
  readonly base: string;
  constructor(
    base: string,
    private token: string,
  ) {
    const url = new URL(base);
    if (
      url.protocol !== "http:" ||
      !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      url.pathname !== "/"
    )
      throw new ApiError(400, "loopback_required");
    this.base = url.origin + "/api/v1";
  }
  async request<T>(
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    let response: Response;
    const headers: Record<string, string> = {
      Authorization: "Bearer " + this.token,
    };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      const signature = path + JSON.stringify(body);
      if (!this.keys.has(signature))
        this.keys.set(signature, crypto.randomUUID());
      headers["Idempotency-Key"] = this.keys.get(signature)!;
    }
    try {
      response = await fetch(this.base + path, {
        method: body === undefined ? "GET" : "POST",
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal,
        cache: "no-store",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError")
        throw error;
      throw new ApiError(0, "network_error");
    }
    return this.read<T>(response);
  }
  private async read<T>(response: Response): Promise<T> {
    const data = await response.json().catch(() => null);
    if (!response.ok)
      throw new ApiError(response.status, data?.error?.code ?? "http_error");
    if (data?.schema_version !== undefined && data.schema_version !== "1.0")
      throw new ApiError(422, "unsupported_schema");
    return data as T;
  }
  async upload(file: File): Promise<import("./types").Dataset> {
    if (
      !file.name.toLowerCase().endsWith(".csv") ||
      file.size > 64 * 1024 ** 2 ||
      !file.size
    )
      throw new ApiError(415, "unsupported_upload");
    // Original filename may contain non-Latin headers; the server name is deliberately generic.
    return this.read(
      await fetch(this.base + "/datasets", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + this.token,
          "Content-Type": "text/csv",
          "X-Filename": "observations.csv",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: file,
      }),
    );
  }
  async download(id: string, format: string) {
    const response = await fetch(this.base + `/exports/${id}/download`, {
      headers: { Authorization: "Bearer " + this.token },
      cache: "no-store",
    });
    if (!response.ok) throw new ApiError(response.status, "export_not_ready");
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `forestcheck.${format}`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
export function message(error: unknown) {
  if (error instanceof ApiError) {
    if (error.status === 401)
      return "Не удалось войти. Проверьте токен оператора.";
    if (error.status === 409)
      return "Конфликт версии. Обновите сведения и сверяйте решение перед повтором.";
    if (error.status === 0)
      return "Сервер недоступен. Проверьте локальный backend и CORS. Данные demo не подставляются.";
    return `Операция не выполнена: ${error.code}`;
  }
  return "Операция не выполнена. Проверьте соединение и контракт файла.";
}
