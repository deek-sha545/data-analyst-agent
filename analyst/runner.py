"""
One-call entry point used by the CLI, API, evaluation harness and MCP server.

    result = run_analysis("Which region grew fastest?", ["data/sales.csv"])

Human-in-the-loop: pass `require_approval=True` and an `approval_callback`.
The graph pauses with a LangGraph `interrupt` after planning; the callback
receives {"plan": [...], "question": ...} and returns
{"approved": bool, "feedback": str}. (The API implements the callback by
waiting for a POST /runs/{id}/approve.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from langgraph.types import Command

from analyst.agents.state import RunContext
from analyst.config import Settings, get_settings
from analyst.graph import VARIANTS, build_graph
from analyst.llm import LLM
from analyst.profiling import SUPPORTED
from analyst.tracker import RunTracker

ApprovalCallback = Callable[[dict], dict]


@dataclass
class RunResult:
    run_id: str
    question: str
    variant: str
    model: str
    status: str                       # completed | failed | rejected | error
    answer: str = ""
    report: str = ""
    report_path: str = ""
    charts: list[str] = field(default_factory=list)
    code: str = ""
    plan: list[str] = field(default_factory=list)
    review: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_analysis(
    question: str,
    data_files: list[str | Path],
    variant: str = "multi",
    *,
    settings: Settings | None = None,
    llm=None,
    tracker: RunTracker | None = None,
    require_approval: bool = False,
    approval_callback: ApprovalCallback | None = None,
) -> RunResult:
    settings = settings or get_settings()
    tracker = tracker or (llm.tracker if llm is not None else RunTracker())
    if llm is None:
        llm = LLM(settings, tracker)
    else:
        llm.tracker = tracker
    files = [Path(f).resolve() for f in data_files]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
        if f.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported file type: {f.name}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")

    run_dir = settings.outputs_dir / tracker.run_id
    ctx = RunContext(llm=llm, settings=settings, tracker=tracker, run_dir=run_dir, data_files=files)
    graph = build_graph(ctx, variant, require_approval=require_approval and variant != "single")
    config = {"configurable": {"thread_id": tracker.run_id}, "recursion_limit": 60}
    tracker.event("system", "start", {"question": question, "variant": variant, "model": llm.model,
                                      "files": [f.name for f in files]})

    result = RunResult(tracker.run_id, question, variant, llm.model, status="error")
    try:
        inp: object = {"question": question}
        while True:
            pending = None
            for chunk in graph.stream(inp, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    pending = chunk["__interrupt__"][0].value
            if pending is None:
                break
            decision = approval_callback(pending) if approval_callback else {"approved": True}
            inp = Command(resume=decision)
        state = graph.get_state(config).values
        result.status = state.get("status", "completed")
        result.answer = state.get("answer", "")
        result.report = state.get("report", "")
        result.plan = state.get("plan", [])
        result.review = state.get("review", {})
        result.attempts = state.get("attempts", [])
        good = state.get("last_good") or {}
        result.code = good.get("code") or state.get("code", "")
        if (run_dir / "report.md").exists():
            result.report_path = str(run_dir / "report.md")
            result.charts = sorted(str(p) for p in run_dir.glob("chart_*.png"))
    except Exception as e:  # LLM outage, bad key... report it instead of crashing the caller
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
        tracker.event("system", "error", result.error)
    result.metrics = tracker.summary()
    tracker.event("system", "done", {"status": result.status, "metrics": result.metrics})
    tracker.save(settings.logs_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ask the autonomous data analyst a question.")
    p.add_argument("question")
    p.add_argument("files", nargs="+", help="CSV / Excel / JSON / Parquet files")
    p.add_argument("--variant", choices=VARIANTS, default="multi")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--approve", action="store_true", help="review the plan before any code runs")
    args = p.parse_args(argv)

    def ask(pending: dict) -> dict:
        print("\nProposed plan:")
        for i, step in enumerate(pending["plan"], 1):
            print(f"  {i}. {step}")
        reply = input("\nApprove? [y = yes / n = cancel / anything else = feedback for the planner] ").strip()
        if reply.lower() in ("y", "yes", ""):
            return {"approved": True}
        if reply.lower() in ("n", "no"):
            return {"approved": False}
        return {"approved": False, "feedback": reply}

    settings = get_settings(args.provider, args.model)
    tracker = RunTracker()
    tracker.listeners.append(lambda ev: print(f"[{ev.t:6.1f}s] {ev.agent:<9} {ev.kind}", file=sys.stderr))
    res = run_analysis(args.question, args.files, args.variant, settings=settings, tracker=tracker,
                       require_approval=args.approve, approval_callback=ask)
    print("\n" + (res.report or res.error))
    print(f"\nStatus: {res.status} | report: {res.report_path or '-'}")
    print(json.dumps({k: res.metrics[k] for k in ("llm_calls", "total_tokens", "cost_usd", "retries", "wall_time_s")}))
    return 0 if res.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
