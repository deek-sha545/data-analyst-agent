"""
Model-agnostic LLM layer.

`LLM.chat()` sends a chat-completions request to whichever OpenAI-compatible
provider is configured, retries transient failures with exponential backoff,
and records tokens / cost / latency / retries in the RunTracker.

`FakeLLM` has the same interface and returns scripted replies, so the whole
agent graph can be tested offline and deterministically.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import deque
from typing import Callable, Sequence

import openai

from analyst.config import Settings, get_settings
from analyst.tracker import LLMCall, RunTracker, estimate_cost

Message = dict[str, str]

# Errors worth retrying: rate limits, timeouts, dropped connections, 5xx.
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)


class LLMError(RuntimeError):
    """The LLM call failed permanently (bad key, unknown model, retries exhausted...)."""


class EmptyResponse(RuntimeError):
    """The model returned no text (e.g. reasoning used the whole token budget). Retried."""


class RateLimiter:
    """
    Process-wide sliding-window limiter: at most `rpm` request starts per 60s
    per key (provider URL + model). Shared by all threads, so parallel eval
    workers queue politely instead of all hitting HTTP 429 at once.
    """

    _windows: dict[str, deque] = {}
    _lock = threading.Lock()

    @classmethod
    def acquire(cls, key: str, rpm: int, sleep: Callable[[float], None] = time.sleep) -> float:
        """Block until a request may start; return seconds waited."""
        if rpm <= 0:
            return 0.0
        waited = 0.0
        while True:
            with cls._lock:
                win = cls._windows.setdefault(key, deque())
                now = time.monotonic()
                while win and now - win[0] >= 60.0:
                    win.popleft()
                if len(win) < rpm:
                    win.append(now)
                    return waited
                delay = 60.0 - (now - win[0]) + 0.05
            sleep(delay)
            waited += delay


_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


class LLM:
    def __init__(
        self,
        settings: Settings | None = None,
        tracker: RunTracker | None = None,
        client: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings or get_settings()
        self.tracker = tracker or RunTracker()
        self.model = self.settings.model
        self._sleep = sleep
        if client is None:
            if not self.settings.api_key:
                raise LLMError(
                    f"{self.settings.provider_info.key_env} is not set. Add it to .env "
                    f"(see .env.example) or choose another LLM_PROVIDER."
                )
            client = openai.OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.provider_info.base_url,
                timeout=self.settings.request_timeout,
                max_retries=0,  # we retry ourselves so retries are counted
            )
        self.client = client
        self._limit_key = f"{self.settings.provider_info.base_url}|{self.model}"

    def chat(self, messages: Sequence[Message], agent: str = "agent", json_mode: bool = False) -> str:
        kwargs = dict(
            model=self.model,
            messages=list(messages),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        retries = 0
        waited = 0.0
        start = time.time()
        while True:
            waited += RateLimiter.acquire(self._limit_key, self.settings.rpm, self._sleep)
            t_req = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
                usage = resp.usage
                p_tok = getattr(usage, "prompt_tokens", 0) or 0
                c_tok = getattr(usage, "completion_tokens", 0) or 0
                if not text:
                    # Count the tokens we were billed for, then retry.
                    self._record(agent, p_tok, c_tok, time.time() - t_req, 0, ok=False, error="empty response")
                    raise EmptyResponse("model returned an empty message")
                self._record(agent, p_tok, c_tok, time.time() - t_req, retries, wait=waited)
                return text
            except (*RETRYABLE, EmptyResponse) as e:
                if retries >= self.settings.max_retries:
                    self._record(agent, 0, 0, time.time() - t_req, retries, ok=False, error=str(e)[:300],
                                 wait=time.time() - start)
                    raise LLMError(f"{agent}: LLM call failed after {retries} retries: {e}") from e
                delay = self._backoff(retries, e)
                retries += 1
                self.tracker.event("system", "retry", f"{agent}: {type(e).__name__}, retry {retries} in {delay:.1f}s")
                self._sleep(delay)
                waited += delay + (time.time() - t_req if isinstance(e, openai.RateLimitError) else 0.0)
            except openai.APIStatusError as e:  # 4xx other than 429: bad key / model / request
                self._record(agent, 0, 0, time.time() - start, retries, ok=False, error=str(e)[:300])
                raise LLMError(f"{agent}: LLM request rejected ({e.status_code}): {e}") from e

    @staticmethod
    def _backoff(attempt: int, err: Exception) -> float:
        """
        Exponential backoff with jitter. Honours the server's hint when there is
        one: a Retry-After header, or Gemini's "Please retry in 57.06s" message.
        """
        hint = None
        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                hint = float(resp.headers.get("retry-after"))
            except (TypeError, ValueError):
                hint = None
        if hint is None:
            m = _RETRY_IN_RE.search(str(err))
            if m:
                hint = float(m.group(1))
        base = hint if hint is not None else 2 ** attempt * 2
        return min(base + random.uniform(0, 1), 65.0)

    def _record(self, agent, p_tok, c_tok, latency, retries, ok=True, error="", wait=0.0):
        cost, priced = estimate_cost(self.model, p_tok, c_tok)
        self.tracker.record_call(
            LLMCall(agent, self.model, p_tok, c_tok, round(latency, 3), retries, cost, priced, ok, error,
                    round(wait, 3))
        )


class FakeLLM:
    """
    Offline stand-in for LLM. `script` is either a list of replies (consumed in
    order) or a function (agent, messages) -> reply. Token counts are a rough
    len/4 estimate so the tracker still has numbers to aggregate.
    """

    def __init__(self, script: list[str] | Callable[[str, list[Message]], str], tracker: RunTracker | None = None):
        self.script = script
        self.tracker = tracker or RunTracker()
        self.model = "fake"
        self.calls: list[tuple[str, list[Message]]] = []

    def chat(self, messages: Sequence[Message], agent: str = "agent", json_mode: bool = False) -> str:
        messages = list(messages)
        self.calls.append((agent, messages))
        if callable(self.script):
            reply = self.script(agent, messages)
        else:
            if not self.script:
                raise LLMError(f"FakeLLM ran out of scripted replies (agent={agent})")
            reply = self.script.pop(0)
        p_tok = sum(len(m["content"]) for m in messages) // 4
        self.tracker.record_call(LLMCall(agent, "fake", p_tok, len(reply) // 4, 0.0, 0, 0.0, True))
        return reply


# --------------------------------------------------------------------- parsing
_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Return the longest fenced code block, or the whole text if there is none."""
    blocks = _CODE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()


def parse_json(text: str) -> dict:
    """Parse a JSON object from a reply that may wrap it in prose or ``` fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in reply: {text[:200]!r}")
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj
