"""Executor: runs the current script in the sandbox (no LLM involved)."""

from __future__ import annotations

from analyst.agents.state import AnalysisState, RunContext
from analyst.sandbox import ExecResult, run_code


def exec_to_dict(r: ExecResult) -> dict:
    return {
        "ok": r.ok,
        "stdout": r.stdout,
        "error": r.error,
        "figures": [str(p) for p in r.figures],
        "duration_s": r.duration_s,
        "timed_out": r.timed_out,
        "violations": r.violations,
        "for_llm": r.for_llm(),
    }


def execute(code: str, version: int, ctx: RunContext) -> dict:
    r = run_code(
        code,
        ctx.data_files,
        timeout=ctx.settings.sandbox_timeout,
        figures_dir=ctx.run_dir / "attempts",
        figure_prefix=f"v{version}",
    )
    result = exec_to_dict(r)
    ctx.tracker.event(
        "executor",
        "execution",
        {k: result[k] for k in ("ok", "stdout", "error", "figures", "duration_s", "violations")} | {"version": version},
    )
    return result


def executor(state: AnalysisState, ctx: RunContext) -> dict:
    version = state.get("version", 1)
    result = execute(state["code"], version, ctx)
    attempts = list(state.get("attempts", [])) + [
        {"version": version, "ok": result["ok"], "error": result["error"]}
    ]
    update: dict = {"execution": result, "attempts": attempts}
    if result["ok"]:
        update["last_good"] = {"code": state["code"], "execution": result}
    else:
        update["fix_attempts"] = state.get("fix_attempts", 0) + 1
    return update
