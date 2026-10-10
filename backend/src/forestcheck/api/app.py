"""Authenticated local workflow. Heavy operations run in a separate worker."""

import hmac
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from forestcheck.api.contracts import (
    AnalysisRequest,
    DatasetView,
    ExportRequest,
    FindingDetail,
    FindingItem,
    JobView,
    MappingRequest,
    Page,
    QueuedJob,
    RecordQuery,
    ReviewEvent,
    ReviewRequest,
    SchemaRequest,
    SchemaView,
    VersionRequest,
)
from forestcheck.api.service import Service, page
from forestcheck.api.settings import ordinary_path
from forestcheck.domain.contracts import Record
from forestcheck.storage.application import Store, WorkflowError


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["ok"] = "ok"
    service: Literal["forestcheck"] = "forestcheck"
    schema_version: Literal["1.0"] = "1.0"


def error(code, status):
    messages = {
        401: "Требуется аутентификация оператора.",
        403: "Доступ запрещён.",
        404: "Объект не найден.",
        409: "Состояние или версия изменились; обновите объект и повторите действие.",
        413: "Превышен размер запроса.",
        415: "Формат не поддерживается.",
        422: "Проверьте контракт и поля запроса.",
        503: "Локальный backend не настроен.",
    }
    return JSONResponse(
        {
            "error": {
                "code": code,
                "message": messages.get(
                    status, "Операция не выполнена; содержимое и пути не раскрываются."
                ),
            }
        },
        status_code=status,
    )


class LocalBoundary:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        protected = scope["path"].startswith("/api/v1") or scope["path"] in {
            "/openapi.json",
            "/docs",
            "/redoc",
        }
        if not protected:
            return await self.app(scope, receive, send)
        settings = self.settings
        headers = {k.lower(): v for k, v in scope["headers"]}

        async def deny(code, status):
            await error(code, status)(scope, receive, send)

        if settings is None:
            return await deny("backend_not_configured", 503)
        try:
            allowed_host = urlsplit("//" + headers.get(b"host", b"").decode("latin1")).hostname in {
                "localhost",
                "127.0.0.1",
                "::1",
            }
        except ValueError:
            allowed_host = False
        if not allowed_host:
            return await deny("invalid_host", 400)
        origin = headers.get(b"origin", b"").decode("latin1")
        if origin and origin not in settings.origins:
            return await deny("origin_not_allowed", 403)
        preflight = (
            scope["method"] == "OPTIONS" and origin and b"access-control-request-method" in headers
        )
        if not preflight:
            token = headers.get(b"authorization", b"").decode("latin1")
            if not hmac.compare_digest(token.encode(), ("Bearer " + settings.token).encode()):
                return await deny("unauthorized", 401)
        upload = scope["path"] == "/api/v1/datasets" and scope["method"] == "POST"
        limit = settings.max_upload_bytes if upload else settings.max_json_bytes
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await deny("invalid_content_length", 400)
        if length < 0 or length > limit:
            return await deny("request_size_limit", 413)
        if scope["method"] in {"POST", "PUT", "PATCH"} and not upload:
            chunks, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > limit:
                    return await deny("request_size_limit", 413)
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            consumed = False
            original_receive = receive

            async def replay():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await original_receive()

            receive = replay
        started = False

        async def protected_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message["headers"] = [
                    *message.get("headers", []),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ]
            await send(message)

        try:
            await self.app(scope, receive, protected_send)
        except WorkflowError as exc:
            if not started:
                await error(exc.code, exc.status)(scope, receive, protected_send)
        except Exception:
            if not started:
                await error("internal_error", 500)(scope, receive, protected_send)


def create_app(settings=None):
    application = FastAPI(title="ForestCheck", version="0.2.0", docs_url=None, redoc_url=None)
    application.state.settings = settings
    store = Store(settings) if settings else None
    service = Service(store) if store else None
    bearer = HTTPBearer(auto_error=False)

    def configured(_=Depends(bearer)):
        if service is None or not store.database.is_file():
            raise WorkflowError("backend_not_initialized", 503)
        return service

    @application.exception_handler(WorkflowError)
    async def workflow_error(request, exc):
        return error(exc.code, exc.status)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error("invalid_request", 422)

    @application.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error("not_found" if exc.status_code == 404 else "http_error", exc.status_code)

    @application.get("/health", response_model=Health)
    def health():
        return Health()

    api = APIRouter(prefix="/api/v1", dependencies=[Depends(configured)])

    @api.get("/session")
    def session():
        return {
            "schema_version": "1.0",
            "operator": settings.operator,
            "data_mode": settings.data_mode,
        }

    @api.post(
        "/datasets",
        status_code=201,
        response_model=DatasetView,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def upload(
        request: Request,
        filename: str = Header(alias="X-Filename"),
        key: str = Header(alias="Idempotency-Key"),
    ):
        if request.headers.get("content-type", "").split(";")[0].strip() not in {
            "text/csv",
            "application/octet-stream",
        }:
            raise WorkflowError("csv_content_type_required", 415)
        path = store.root / "staging" / (str(uuid4()) + ".csv")
        ordinary_path(path, missing=True)
        size = 0
        try:
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise WorkflowError("upload_size_limit", 413)
                    if b"\x00" in chunk:
                        raise WorkflowError("unsupported_upload_content", 415)
                    stream.write(chunk)
            return service.register_upload(path, filename, settings.operator, key)
        finally:
            if path.exists():
                path.unlink()

    @api.get("/datasets", response_model=Page[DatasetView])
    def datasets(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return service.datasets(offset=offset, limit=limit)

    @api.get("/datasets/{dataset_id}", response_model=DatasetView)
    def dataset(dataset_id: UUID):
        return service.dataset(str(dataset_id))

    @api.post("/datasets/{dataset_id}/schema", response_model=SchemaView)
    def schema(dataset_id: UUID, body: SchemaRequest):
        return service.schema(str(dataset_id), encoding=body.encoding, delimiter=body.delimiter)

    @api.get("/datasets/{dataset_id}/schema", response_model=SchemaView)
    def confirmed_schema(dataset_id: UUID):
        return service.schema(str(dataset_id))

    @api.post("/datasets/{dataset_id}/mapping", status_code=202, response_model=QueuedJob)
    def confirm_mapping(
        dataset_id: UUID, body: MappingRequest, key: str = Header(alias="Idempotency-Key")
    ):
        return service.confirm_mapping(str(dataset_id), body, settings.operator, key)

    @api.get("/versions/{version_id}/records", response_model=Page[Record])
    def records(
        version_id: UUID, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        return service.records(str(version_id), offset=offset, limit=limit)

    @api.post("/datasets/{dataset_id}/versions", status_code=202, response_model=QueuedJob)
    def new_version(
        dataset_id: UUID, body: VersionRequest, key: str = Header(alias="Idempotency-Key")
    ):
        return service.create_version(str(dataset_id), body, settings.operator, key)

    @api.post("/versions/{version_id}/records/query", response_model=Page[Record])
    def record_query(version_id: UUID, body: RecordQuery):
        return service.record_query(str(version_id), body.record_ids)

    @api.get("/datasets/{dataset_id}/versions")
    def versions(
        dataset_id: UUID, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        return service.versions(str(dataset_id), offset=offset, limit=limit)

    @api.get("/analyses/profiles")
    def profiles(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        result = service.profiles()["items"]
        return page(result[offset : offset + limit], len(result), offset, limit)

    @api.get("/models")
    def models(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        result = [
            m | {"profile_id": p["id"]} for p in service.profiles()["items"] for m in p["models"]
        ]
        result.sort(key=lambda m: (m["profile_id"], m["id"]))
        return page(result[offset : offset + limit], len(result), offset, limit)

    @api.post("/analyses", status_code=202, response_model=QueuedJob)
    def analysis(body: AnalysisRequest, key: str = Header(alias="Idempotency-Key")):
        return service.start_analysis(body, settings.operator, key)

    @api.get("/analyses", response_model=Page[JobView])
    def analyses(
        status: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return service.jobs(kind="analysis", status=status, offset=offset, limit=limit)

    @api.get("/analyses/{analysis_id}", response_model=JobView)
    def get_analysis(analysis_id: UUID):
        result = service.job(str(analysis_id))
        if result["kind"] != "analysis":
            raise WorkflowError("not_found", 404)
        return result

    @api.get("/jobs", response_model=Page[JobView])
    def job_queue(
        status: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return service.jobs(status=status, offset=offset, limit=limit)

    @api.get("/jobs/{job_id}", response_model=JobView)
    def job(job_id: UUID):
        return service.job(str(job_id))

    @api.post("/jobs/{job_id}/cancel", response_model=JobView)
    def cancel(job_id: UUID):
        store.cancel(str(job_id))
        return service.job(str(job_id))

    @api.get("/findings", response_model=Page[FindingItem])
    def findings(
        analysis_id: UUID,
        severity: str | None = None,
        field_name: str | None = None,
        reviewed: bool | None = None,
        search: str | None = Query(None, max_length=200),
        sort: Literal["priority", "field", "record", "signal"] = "priority",
        descending: bool = False,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return service.findings(
            str(analysis_id),
            severity=severity,
            field_name=field_name,
            reviewed=reviewed,
            search=search,
            sort=sort,
            descending=descending,
            offset=offset,
            limit=limit,
        )

    @api.get("/findings/{finding_id}", response_model=FindingDetail)
    def finding(finding_id: UUID):
        return service.finding(str(finding_id))

    @api.post("/reviews", status_code=201, response_model=ReviewEvent)
    def review(body: ReviewRequest, key: str = Header(alias="Idempotency-Key")):
        return service.review(body, settings.operator, key)

    @api.get("/reviews", response_model=Page[ReviewEvent])
    def reviews(
        finding_id: UUID, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        return service.reviews(str(finding_id), offset=offset, limit=limit)

    @api.post("/exports", status_code=202, response_model=QueuedJob)
    def export(body: ExportRequest, key: str = Header(alias="Idempotency-Key")):
        return service.export(body, settings.operator, key)

    @api.get("/exports", response_model=Page[JobView])
    def exports(
        status: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return service.jobs(kind="export", status=status, offset=offset, limit=limit)

    @api.get("/exports/{export_id}", response_model=JobView)
    def get_export(export_id: UUID):
        result = service.job(str(export_id))
        if result["kind"] != "export":
            raise WorkflowError("not_found", 404)
        return result

    @api.get("/exports/{export_id}/download")
    def download(export_id: UUID):
        path, media, fmt = service.download(str(export_id))
        return FileResponse(path, media_type=media, filename="forestcheck-export." + fmt)

    application.include_router(api)
    if settings and settings.origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Filename"],
        )
    application.add_middleware(LocalBoundary, settings=settings)
    return application


def app_from_env():
    from forestcheck.api.settings import Settings

    return create_app(Settings.from_env())


app = create_app()
