import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from forestcheck.api.app import create_app
from forestcheck.api.execution import Worker
from forestcheck.api.service import Service
from test_backend_store import configured
from test_backend_workflow import fixture_mapping, fixture_profile


@pytest.fixture
def api(tmp_path):
    settings, store = configured(tmp_path)
    mapping = fixture_mapping()
    profile = fixture_profile(mapping)
    Service(store).register_profile(profile)
    with TestClient(
        create_app(settings), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        yield client, store, settings, mapping, profile


def auth(settings, key=None):
    return {
        "Authorization": "Bearer " + settings.token,
        **({"Idempotency-Key": key} if key else {}),
    }


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/datasets"),
        ("get", "/api/v1/findings"),
        ("get", "/api/v1/models"),
        ("get", "/openapi.json"),
        ("post", "/api/v1/reviews"),
        ("post", "/api/v1/analyses"),
    ],
)
def test_unauthorized_reads_and_writes_are_denied(api, method, path):
    client, _, _, _, _ = api
    response = getattr(client, method)(path)
    assert response.status_code == 401
    assert "traceback" not in response.text.lower()


def test_http_complete_cycle_and_authenticated_openapi(api):
    client, store, settings, mapping, profile = api
    uploaded = client.post(
        "/api/v1/datasets",
        headers=auth(settings, "upload") | {"X-Filename": "sample.csv", "Content-Type": "text/csv"},
        content=b"stand;per;area;total;label\n001;10;2;;=1+1\n",
    )
    assert uploaded.status_code == 201, uploaded.text
    dataset = uploaded.json()
    info = client.post(
        f"/api/v1/datasets/{dataset['id']}/schema",
        headers=auth(settings),
        json=dict(schema_version="1.0", encoding="utf-8", delimiter=";"),
    )
    assert info.json()["headers"] == ["stand", "per", "area", "total", "label"]
    result = client.post(
        f"/api/v1/datasets/{dataset['id']}/mapping",
        headers=auth(settings, "mapping"),
        json=dict(
            schema_version="1.0",
            expected_version=dataset["etag"],
            mapping=mapping.model_dump(mode="json"),
        ),
    )
    assert result.status_code == 202, result.text
    Worker(store).run_once()
    dataset = client.get(f"/api/v1/datasets/{dataset['id']}", headers=auth(settings)).json()
    req = dict(
        schema_version="1.0",
        version_id=dataset["current_version_id"],
        profile_id=str(profile.id),
        use_models=False,
    )
    run = client.post("/api/v1/analyses", headers=auth(settings, "analysis"), json=req)
    assert run.status_code == 202, run.text
    assert (
        client.post("/api/v1/analyses", headers=auth(settings, "analysis"), json=req).json()["id"]
        == run.json()["id"]
    )
    Worker(store).run_once()
    findings = client.get(
        "/api/v1/findings",
        params={"analysis_id": run.json()["id"], "limit": 1},
        headers=auth(settings),
    ).json()
    assert findings["total"] == 1 and findings["next_offset"] is None
    detail = client.get(
        "/api/v1/findings/" + findings["items"][0]["finding"]["id"], headers=auth(settings)
    ).json()
    suggestion = detail["suggestions"][0]["suggestion"]
    payload = dict(
        schema_version="1.0",
        finding_id=detail["finding"]["id"],
        expected_version=0,
        error_assessment="confirmed_error",
        suggestion_id=suggestion["id"],
        suggestion_decision="accepted",
        comment="Synthetic expert",
    )
    response = client.post("/api/v1/reviews", headers=auth(settings, "review"), json=payload)
    assert response.status_code == 201, response.text
    assert response.json()["operator"] == settings.operator
    assert (
        client.post(
            "/api/v1/reviews",
            headers=auth(settings, "forged"),
            json=payload | {"reviewer_id": "forged"},
        ).status_code
        == 422
    )
    versions = client.post(
        f"/api/v1/datasets/{dataset['id']}/versions",
        headers=auth(settings, "new-version"),
        json=dict(
            schema_version="1.0",
            base_version_id=dataset["current_version_id"],
            analysis_id=run.json()["id"],
            expected_version=dataset["etag"],
            review_event_ids=[response.json()["id"]],
        ),
    )
    assert versions.status_code == 202, versions.text
    Worker(store).run_once()
    job = client.get("/api/v1/jobs/" + versions.json()["id"], headers=auth(settings)).json()
    export = client.post(
        "/api/v1/exports",
        headers=auth(settings, "export"),
        json=dict(
            schema_version="1.0",
            version_id=job["result"]["version_id"],
            kind="corrected",
            format="csv",
        ),
    )
    assert export.status_code == 202, export.text
    Worker(store).run_once()
    file = client.get(
        "/api/v1/exports/" + export.json()["id"] + "/download", headers=auth(settings)
    )
    assert file.status_code == 200 and "'=1+1" in file.text and ";20;" in file.text
    schema = client.get("/openapi.json", headers=auth(settings)).json()
    assert "/api/v1/analyses" in schema["paths"] and schema["components"]["securitySchemes"]
    assert str(settings.root) not in json.dumps(schema)


@pytest.mark.parametrize(
    "name,body",
    [
        ("../escape.csv", b"a\n1"),
        ("model.pkl", b"abc"),
        ("book.xlsm", b"PK"),
        ("renamed.csv", b"MZ\x00binary"),
    ],
)
def test_disallowed_upload_paths_and_formats(api, name, body):
    client, _, settings, _, _ = api
    response = client.post(
        "/api/v1/datasets",
        headers=auth(settings, str(uuid4())) | {"X-Filename": name, "Content-Type": "text/csv"},
        content=body,
    )
    assert response.status_code in {415, 422}
    assert client.get("/api/v1/datasets", headers=auth(settings)).json()["total"] == 0


def test_bad_origin_invalid_filter_and_errors_do_not_echo_values(api):
    client, _, settings, _, _ = api
    assert (
        client.get(
            "/api/v1/models", headers=auth(settings) | {"Origin": "https://evil.invalid"}
        ).status_code
        == 403
    )
    response = client.post(
        "/api/v1/analyses",
        headers=auth(settings, "bad"),
        json={"schema_version": "SECRET_INPUT", "path": "PRIVATE_PATH"},
    )
    assert response.status_code == 422
    assert "SECRET_INPUT" not in response.text and "PRIVATE_PATH" not in response.text
    assert client.get("/api/v1/datasets?limit=10000", headers=auth(settings)).status_code == 422
    assert (
        client.get(
            "/api/v1/exports/" + str(uuid4()) + "/download", headers=auth(settings)
        ).status_code
        == 404
    )
