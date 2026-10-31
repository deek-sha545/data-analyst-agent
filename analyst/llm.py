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
import time
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
        start = time.time()
        while True:
            try:
                resp = self.client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
                usage = resp.usage
                p_tok = getattr(usage, "prompt_tokens", 0) or 0
                c_tok = getattr(usage, "completion_tokens", 0) or 0
                if not text:
                    # Count the tokens we were billed for, then retry.
                    self._record(agent, p_tok, c_tok, 0.0, 0, ok=False, error="empty response")
                    raise EmptyResponse("model returned an empty message")
                self._record(agent, p_tok, c_tok, time.time() - start, retries)
                return text
            except (*RETRYABLE, EmptyResponse) as e:
                if retries >= self.settings.max_retries:
                    self._record(agent, 0, 0, time.time() - start, retries, ok=False, error=str(e)[:300])
                    raise LLMError(f"{agent}: LLM call failed after {retries} retries: {e}") from e
                delay = self._backoff(retries, e)
                retries += 1
                self.tracker.event("system", "retry", f"{agent}: {type(e).__name__}, retry {retries} in {delay:.1f}s")
                self._sleep(delay)
            except openai.APIStatusError as e:  # 4xx other than 429: bad key / model / request
                self._record(agent, 0, 0, time.time() - start, retries, ok=False, error=str(e)[:300])
                raise LLMError(f"{agent}: LLM request rejected ({e.status_code}): {e}") from e

    @staticmethod
    def _backoff(attempt: int, err: Exception) -> float:
        """Exponential backoff with jitter; honours a Retry-After header when present."""
        retry_after = None
        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                retry_after = float(resp.headers.get("retry-after"))
            except (TypeError, ValueError):
                retry_after = None
        base = retry_after if retry_after is not None else 2 ** attempt * 2
        return min(base + random.uniform(0, 1), 60.0)

    def _record(self, agent, p_tok, c_tok, latency, retries, ok=True, error=""):
        cost, priced = estimate_cost(self.model, p_tok, c_tok)
        self.tracker.record_call(
            LLMCall(agent, self.model, p_tok, c_tok, round(latency, 3), retries, cost, priced, ok, error)
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
