"""FastAPI backend with a FakeLLM (no network)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from analyst.llm import FakeLLM
from api import main
from tests.test_agents import APPROVE, GOOD_CODE, PLAN, REPORT


def fake_factory(settings, tracker):
    replies = {"planner": [PLAN, PLAN], "coder": [GOOD_CODE], "reviewer": [APPROVE], "reporter": [REPORT],
               "solo": [GOOD_CODE, REPORT]}
    return FakeLLM(lambda agent, msgs: replies[agent].pop(0), tracker)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOADS_DIR", tmp_path / "uploads")
    main.app.state.llm_factory = fake_factory
    main.app.state.settings_overrides = {"outputs_dir": tmp_path / "out", "logs_dir": tmp_path / "logs"}
    main.JOBS.clear()
    yield TestClient(main.app)
    main.app.state.settings_overrides = {}


def wait_status(client, run_id, until, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] in until:
            return r
        time.sleep(0.2)
    raise AssertionError(f"run {run_id} stuck in {r['status']}")


def test_health_and_datasets(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["provider"] == "gemini"
    ids = [d["id"] for d in client.get("/api/datasets").json()]
    assert "sample:sales.csv" in ids and "sample:hr_employees.xlsx" in ids


def test_full_run_with_report_and_charts(client):
    r = client.post("/api/runs", json={"question": "Which region has most orders?", "datasets": ["sample:sales.csv"]})
    assert r.status_code == 201
    run_id = r.json()["run_id"]
    view = wait_status(client, run_id, {"completed", "failed", "error"})
    assert view["status"] == "completed", view
    res = view["result"]
    assert res["answer"] == "East has the most orders."
    assert res["charts"] == ["chart_1.png"]
    assert view["metrics"]["llm_calls"] == 4

    png = client.get(f"/api/runs/{run_id}/files/chart_1.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert "**Answer:**" in client.get(f"/api/runs/{run_id}/files/report.md").text
    assert "read_csv" in client.get(f"/api/runs/{run_id}/files/analysis.py").text

    ev = client.get(f"/api/runs/{run_id}/events").json()
    kinds = [e["kind"] for e in ev["events"]]
    assert kinds[0] == "start" and kinds[-1] == "done" and "review" in kinds
    assert client.get(f"/api/runs/{run_id}/events", params={"since": ev["next"]}).json()["events"] == []

    lines = [json.loads(l) for l in client.get(f"/api/runs/{run_id}/stream").text.splitlines() if l]
    assert lines[-1]["kind"] == "end" and lines[-1]["content"]["status"] == "completed"


def test_human_approval_flow(client):
    run_id = client.post("/api/runs", json={"question": "Which region has most orders?",
                                            "datasets": ["sample:sales.csv"], "require_approval": True}).json()["run_id"]
    view = wait_status(client, run_id, {"awaiting_approval"})
    assert view["pending_plan"] == json.loads(PLAN)["plan"]
    # feedback -> re-plan -> asks again
    assert client.post(f"/api/runs/{run_id}/approve", json={"approved": False, "feedback": "shorter"}).status_code == 200
    time.sleep(0.3)
    view = wait_status(client, run_id, {"awaiting_approval"})
    assert client.post(f"/api/runs/{run_id}/approve", json={"approved": True}).json() == {"ok": True}
    assert wait_status(client, run_id, {"completed", "failed", "error"})["status"] == "completed"
    # approving a finished run is a conflict
    assert client.post(f"/api/runs/{run_id}/approve", json={"approved": True}).status_code == 409


def test_single_variant(client):
    run_id = client.post("/api/runs", json={"question": "Which region has most orders?",
                                            "datasets": ["sample:sales.csv"], "variant": "single"}).json()["run_id"]
    assert wait_status(client, run_id, {"completed", "failed", "error"})["status"] == "completed"


def test_upload_and_validation(client):
    csv = b'city,sales\nParis,10\nLyon,"$1,200"\n'
    up = client.post("/api/datasets", files={"file": ("../../evil name.csv", csv, "text/csv")})
    assert up.status_code == 200, up.text
    body = up.json()
    assert body["id"].startswith("upload:") and "/" not in body["name"] and ".." not in body["name"]
    assert "city" in body["profile"]
    assert body["id"] in [d["id"] for d in client.get("/api/datasets").json()]

    assert client.post("/api/datasets", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.post("/api/datasets", files={"file": ("empty.csv", b"", "text/csv")}).status_code == 400
    bad = client.post("/api/datasets", files={"file": ("bad.csv", b"a,b\n1,2\n3,4,5\n", "text/csv")})
    assert bad.status_code == 400 and "Could not parse" in bad.json()["detail"]
    assert not any(p.name.endswith("bad.csv") for p in main.UPLOADS_DIR.iterdir())  # rejected file removed


def test_bad_requests(client):
    assert client.post("/api/runs", json={"question": "q?", "datasets": []}).status_code == 422
    assert client.post("/api/runs", json={"question": "Which?", "datasets": ["sample:../.env"]}).status_code in (400, 404)
    assert client.post("/api/runs", json={"question": "Which?", "datasets": ["sample:nope.csv"]}).status_code == 404
    assert client.post("/api/runs", json={"question": "Which?", "datasets": ["x"]}).status_code == 400
    assert client.post("/api/runs", json={"question": "Which?", "datasets": ["sample:sales.csv"],
                                          "variant": "bogus"}).status_code == 422
    assert client.get("/api/runs/nope").status_code == 404


def test_file_endpoint_rejects_traversal(client):
    run_id = client.post("/api/runs", json={"question": "Which region?", "datasets": ["sample:sales.csv"]}).json()["run_id"]
    wait_status(client, run_id, {"completed", "failed", "error"})
    assert client.get(f"/api/runs/{run_id}/files/..%2F..%2F.env").status_code in (400, 404)
    assert client.get(f"/api/runs/{run_id}/files/secret.txt").status_code == 400


def test_missing_key_gives_503(client):
    def no_key(settings, tracker):
        raise RuntimeError("GEMINI_API_KEY is not set")

    main.app.state.llm_factory = no_key
    r = client.post("/api/runs", json={"question": "Which region?", "datasets": ["sample:sales.csv"]})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
