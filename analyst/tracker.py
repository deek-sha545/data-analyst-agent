"""
Run tracker: records every LLM call (tokens, cost, latency, retries) and every
agent step event for one analysis run.

The same object feeds three consumers:
  * the experiment harness (aggregate tokens / cost / latency per run),
  * the API (streams `events` live to the UI),
  * the log file written at the end of a run (logs/<run_id>.json).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from analyst.config import PRICES


@dataclass
class LLMCall:
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    retries: int
    cost_usd: float
    priced: bool
    ok: bool = True
    error: str = ""


@dataclass
class Event:
    t: float            # seconds since run start
    agent: str          # planner / coder / executor / reviewer / reporter / system
    kind: str           # e.g. "plan", "code", "execution", "review", "report", "error", "awaiting_approval"
    content: Any


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, bool]:
    """Return (usd, priced). Unknown models cost 0 and are reported as unpriced."""
    if model not in PRICES:
        return 0.0, False
    p_in, p_out = PRICES[model]
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000, True


@dataclass
class RunTracker:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    calls: list[LLMCall] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    listeners: list[Callable[[Event], None]] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)  # re-entrant: to_dict() calls summary()

    # ---------------------------------------------------------------- record
    def record_call(self, call: LLMCall) -> None:
        with self._lock:
            self.calls.append(call)

    def event(self, agent: str, kind: str, content: Any = None) -> Event:
        ev = Event(round(time.time() - self.started, 3), agent, kind, content)
        with self._lock:
            self.events.append(ev)
            listeners = list(self.listeners)
        for fn in listeners:
            try:
                fn(ev)
            except Exception:  # a broken listener must never break the run
                pass
        return ev

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        return {
            "run_id": self.run_id,
            "llm_calls": len(calls),
            "failed_calls": sum(not c.ok for c in calls),
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls),
            "cost_usd": round(sum(c.cost_usd for c in calls), 6),
            "all_priced": all(c.priced for c in calls),
            "llm_latency_s": round(sum(c.latency_s for c in calls), 3),
            "retries": sum(c.retries for c in calls),
            "wall_time_s": round(time.time() - self.started, 3),
            "by_agent": self._by_agent(calls),
        }

    @staticmethod
    def _by_agent(calls: list[LLMCall]) -> dict:
        out: dict[str, dict] = {}
        for c in calls:
            a = out.setdefault(c.agent, {"calls": 0, "tokens": 0, "cost_usd": 0.0, "latency_s": 0.0})
            a["calls"] += 1
            a["tokens"] += c.prompt_tokens + c.completion_tokens
            a["cost_usd"] = round(a["cost_usd"] + c.cost_usd, 6)
            a["latency_s"] = round(a["latency_s"] + c.latency_s, 3)
        return out

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "summary": self.summary(),
                "calls": [asdict(c) for c in self.calls],
                "events": [asdict(e) for e in self.events],
            }

    def save(self, logs_dir: Path) -> Path:
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path
