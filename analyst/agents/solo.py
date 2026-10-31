"""
Single-agent baseline: one LLM in a ReAct-style loop. Each turn it either
writes code (which the executor runs and feeds back) or writes the final
report. Same sandbox, same dataset profile and the same execution budget as
the multi-agent graph, but no separate planning, review or reporting roles.
"""

from __future__ import annotations

from analyst import prompts
from analyst.agents.executor import execute
from analyst.agents.reporter import finalize
from analyst.agents.state import AnalysisState, RunContext
from analyst.llm import extract_code


def solo_budget(ctx: RunContext) -> int:
    """Executions allowed = the multi-agent budget: 1 draft + fixes + review rounds."""
    return 1 + ctx.settings.max_fix_attempts + ctx.settings.max_review_rounds


def solo_agent(state: AnalysisState, ctx: RunContext) -> dict:
    messages = list(state.get("solo_messages") or [])
    if not messages:
        messages = [
            {"role": "system", "content": prompts.SOLO},
            {
                "role": "user",
                "content": f"Question: {state['question']}\n\nData files (in the working directory): "
                f"{ctx.file_names}\n\nDataset profile:\n{state['profile']}",
            },
        ]
    turns = state.get("solo_turns", 0)
    out_of_budget = turns >= solo_budget(ctx)
    if out_of_budget:
        messages.append(
            {"role": "user", "content": "You have used all your code runs. Write the final report now (option B)."}
        )
    reply = ctx.llm.chat(messages, agent="solo")
    messages.append({"role": "assistant", "content": reply})

    wants_code = "```" in reply and "**Answer" not in reply
    if wants_code and not out_of_budget:
        code = extract_code(reply)
        version = turns + 1
        ctx.tracker.event("solo", "code", {"version": version, "mode": "solo", "code": code})
        result = execute(code, version, ctx)
        messages.append({"role": "user", "content": result["for_llm"]})
        update = {
            "solo_messages": messages,
            "solo_turns": version,
            "code": code,
            "version": version,
            "execution": result,
            "attempts": list(state.get("attempts", [])) + [{"version": version, "ok": result["ok"], "error": result["error"]}],
        }
        if result["ok"]:
            update["last_good"] = {"code": code, "execution": result}
        return update

    # Final report: attach the charts of the last successful run.
    last_good = state.get("last_good")
    figures = last_good["execution"]["figures"] if last_good else []
    code = last_good["code"] if last_good else state.get("code", "")
    status = "completed" if last_good else "failed"
    return finalize(reply, code, figures, ctx) | {"solo_messages": messages, "status": status, "solo_turns": turns}
