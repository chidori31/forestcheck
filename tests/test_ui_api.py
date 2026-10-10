from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from forestcheck.api.app import create_app
from forestcheck.storage.application import WorkflowError
from test_backend_api import auth
from test_backend_workflow import analysed


def test_findings_search_sort_and_record_batch(tmp_path):
    settings, _, service, _, version, _, run, detail, _ = analysed(tmp_path)
    page = service.findings(run["id"], search="total", sort="field", descending=True)
    assert page["total"] == 1
    assert service.findings(run["id"], search="not present")["total"] == 0
    with pytest.raises(WorkflowError):
        service.findings(run["id"], sort="DROP TABLE findings")
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        payload = {"schema_version": "1.0", "record_ids": [detail["finding"]["record_id"]]}
        response = client.post(
            f"/api/v1/versions/{version}/records/query", json=payload, headers=auth(settings)
        )
        assert response.status_code == 200
        assert response.json()["items"][0]["id"] == detail["finding"]["record_id"]
        assert (
            client.post(f"/api/v1/versions/{version}/records/query", json=payload).status_code
            == 401
        )
        payload["record_ids"] = [str(UUID(int=987))]
        assert (
            client.post(
                f"/api/v1/versions/{version}/records/query", json=payload, headers=auth(settings)
            ).status_code
            == 404
        )


def test_session_and_persistent_queue(tmp_path):
    settings, _, _, _, _, _, _, _, _ = analysed(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/v1/jobs").status_code == 401
        response = client.get("/api/v1/jobs", headers=auth(settings))
        assert response.status_code == 200
        assert response.json()["total"] >= 2
        session = client.get("/api/v1/session", headers=auth(settings)).json()
        assert session["operator"] == settings.operator
        assert session["data_mode"] == "real"
        assert settings.token not in str(session)
