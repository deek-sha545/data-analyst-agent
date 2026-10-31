"""LLM layer + run tracker, tested with a fake OpenAI client (no network)."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from analyst.config import PROVIDERS, get_settings
from analyst.llm import LLM, FakeLLM, LLMError, extract_code, parse_json
from analyst.tracker import RunTracker, estimate_cost


def _resp(text, p=100, c=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c),
    )


def _rate_limit():
    req = httpx.Request("POST", "https://x/chat/completions")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create with a queue of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make_llm(outcomes, **settings):
    s = get_settings("gemini", "gemini-3.5-flash", **settings)
    return LLM(s, RunTracker(), client=FakeClient(outcomes), sleep=lambda _: None)


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    s = get_settings()
    assert s.provider == "gemini"
    assert s.model == PROVIDERS["gemini"].default_model
    assert "generativelanguage.googleapis.com" in s.provider_info.base_url


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        get_settings("nope")


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLM(get_settings("groq"))


def test_chat_records_tokens_and_cost():
    llm = make_llm([_resp("hello", p=1_000_000, c=0)])
    assert llm.chat([{"role": "user", "content": "hi"}], agent="planner") == "hello"
    s = llm.tracker.summary()
    assert s["llm_calls"] == 1 and s["prompt_tokens"] == 1_000_000
    assert s["cost_usd"] == pytest.approx(0.30)
    assert s["by_agent"]["planner"]["calls"] == 1


def test_retries_rate_limit_then_succeeds():
    llm = make_llm([_rate_limit(), _rate_limit(), _resp("ok")])
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    s = llm.tracker.summary()
    assert s["retries"] == 2 and s["failed_calls"] == 0
    assert sum(e.kind == "retry" for e in llm.tracker.events) == 2


def test_empty_reply_is_retried():
    llm = make_llm([_resp(""), _resp("real answer")])
    assert llm.chat([{"role": "user", "content": "hi"}]) == "real answer"


def test_gives_up_after_max_retries():
    llm = make_llm([_rate_limit()] * 3, max_retries=2)
    with pytest.raises(LLMError, match="after 2 retries"):
        llm.chat([{"role": "user", "content": "hi"}])
    assert llm.tracker.summary()["failed_calls"] == 1


def test_auth_error_not_retried():
    req = httpx.Request("POST", "https://x")
    err = openai.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    llm = make_llm([err, _resp("never")])
    with pytest.raises(LLMError, match="401"):
        llm.chat([{"role": "user", "content": "hi"}])
    assert len(llm.client.requests) == 1


def test_json_mode_sets_response_format():
    llm = make_llm([_resp('{"a": 1}')])
    llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert llm.client.requests[0]["response_format"] == {"type": "json_object"}


def test_fake_llm_scripted_and_callable():
    f = FakeLLM(["one", "two"])
    assert f.chat([{"role": "user", "content": "x"}]) == "one"
    assert f.chat([{"role": "user", "content": "x"}], agent="coder") == "two"
    with pytest.raises(LLMError):
        f.chat([{"role": "user", "content": "x"}])
    g = FakeLLM(lambda agent, msgs: f"hi {agent}")
    assert g.chat([{"role": "user", "content": "x"}], agent="reviewer") == "hi reviewer"


def test_extract_code_and_parse_json():
    assert extract_code("Here:\n```python\nprint(1)\n```\nbye") == "print(1)"
    assert extract_code("print(2)") == "print(2)"
    assert parse_json('Sure!\n```json\n{"ok": true}\n```') == {"ok": True}
    assert parse_json('blah {"x": [1, 2]} blah') == {"x": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_cost_unknown_model_unpriced():
    assert estimate_cost("mystery-model", 100, 100) == (0.0, False)


def test_tracker_events_listeners_and_save(tmp_path):
    t = RunTracker()
    seen = []
    t.listeners.append(seen.append)
    t.listeners.append(lambda e: 1 / 0)  # a broken listener must not break the run
    t.event("planner", "plan", ["step 1"])
    assert seen[0].content == ["step 1"]
    path = t.save(tmp_path)
    assert path.exists() and t.run_id in path.name
    assert t.to_dict()["summary"]["llm_calls"] == 0
