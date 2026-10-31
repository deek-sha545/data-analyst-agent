"""
Streamlit UI driven headlessly with AppTest. Its HTTP calls are routed into the
FastAPI TestClient (with a FakeLLM), so no server or network is needed.
"""
import time
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from tests.test_api import client  # noqa: F401  (fixture: FastAPI TestClient with FakeLLM)

UI = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


@pytest.fixture
def ui(client, monkeypatch):  # noqa: F811
    def route(method, url, **kw):
        kw.pop("timeout", None)
        return client.request(method, url.replace("http://localhost:8000", ""), **kw)

    monkeypatch.setattr(requests, "request", route)
    monkeypatch.delenv("API_URL", raising=False)
    return AppTest.from_file(UI, default_timeout=30)


def run_until(at, cond, tries=60):
    for _ in range(tries):
        at.run()
        assert not at.exception, at.exception
        if cond(at):
            return at
        time.sleep(0.3)
    raise AssertionError("UI never reached the expected state")


def test_ui_run_with_plan_approval(ui):
    at = ui
    at.run()
    assert not at.exception, at.exception
    assert "Autonomous Data Analyst" in at.title[0].value
    assert "gemini" in at.sidebar.caption[0].value

    at.sidebar.checkbox[0].check()
    at.button[0].click()  # Run analysis
    run_until(at, lambda a: any("planner proposes" in i.value for i in a.info))

    next(b for b in at.button if "Approve" in b.label).click()
    run_until(at, lambda a: len(a.success) > 0)

    assert at.success[0].value == "Analysis complete"
    assert any("**Answer:** East has the most orders." in m.value for m in at.markdown)
    kinds = [type(e).__name__ for e in at.main]
    assert kinds.count("Image") == 1                          # the chart is rendered
    assert kinds.count("DownloadButton") == 2                 # report.md + analysis.py
    assert {m.label for m in at.metric} >= {"LLM calls", "Tokens", "Est. cost"}
