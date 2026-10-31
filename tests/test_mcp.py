"""MCP server, exercised through a real in-process MCP client."""
import json

import anyio
from mcp import Client

from analyst.config import get_settings
from analyst.runner import run_analysis
from analyst_mcp import server as mcp_server
from tests.test_api import fake_factory


def call(tool: str, args: dict | None = None):
    async def go():
        async with Client(mcp_server.server) as c:
            return await c.call_tool(tool, args or {})

    return anyio.run(go)


def payload(result):
    """Structured content if present, else the JSON in the first text block."""
    if result.structured_content is not None:
        sc = result.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


def test_tools_are_listed():
    async def go():
        async with Client(mcp_server.server) as c:
            return [t.name for t in (await c.list_tools()).tools]

    assert set(anyio.run(go)) >= {"list_datasets", "profile_dataset", "run_python", "ask_analyst"}


def test_list_and_profile():
    names = [d["name"] for d in payload(call("list_datasets"))]
    assert "sales.csv" in names
    prof = call("profile_dataset", {"dataset": "sales.csv"})
    assert not prof.is_error
    assert "numbers stored as text" in prof.content[0].text


def test_path_restrictions(monkeypatch):
    monkeypatch.delenv("ANALYST_MCP_ALLOW_ANY_PATH", raising=False)
    for bad in ["../.env", "/etc/passwd", ".env", "nope.csv"]:
        res = call("profile_dataset", {"dataset": bad})
        assert res.is_error, bad


def test_run_python_in_sandbox():
    ok = payload(call("run_python", {"code": "import pandas as pd\nprint(len(pd.read_csv('sales.csv')))",
                                     "datasets": ["sales.csv"]}))
    assert ok["ok"] and ok["stdout"].strip().isdigit()
    bad = payload(call("run_python", {"code": "import os\nos.system('id')"}))
    assert not bad["ok"] and "not allowed" in bad["error"]


def test_ask_analyst_with_fake_llm(monkeypatch, tmp_path):
    def fake_run(question, files, variant):
        s = get_settings(outputs_dir=tmp_path / "out", logs_dir=tmp_path / "logs")
        llm = fake_factory(s, None)
        return run_analysis(question, files, variant, settings=s, llm=llm, tracker=llm.tracker)

    monkeypatch.setattr(mcp_server, "run_analysis", fake_run)
    out = payload(call("ask_analyst", {"question": "Which region has most orders?", "datasets": ["sales.csv"]}))
    assert out["status"] == "completed"
    assert out["answer"] == "East has the most orders."
    assert len(out["charts"]) == 1 and out["metrics"]["llm_calls"] == 4
    assert call("ask_analyst", {"question": "q", "datasets": ["sales.csv"], "variant": "bogus"}).is_error
