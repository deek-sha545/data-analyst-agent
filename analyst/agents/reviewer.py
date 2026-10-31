"""Reviewer: checks the script + output for correctness and either approves or asks for a revision."""

from __future__ import annotations

from analyst import prompts
from analyst.agents.state import AnalysisState, RunContext
from analyst.llm import parse_json


def reviewer(state: AnalysisState, ctx: RunContext) -> dict:
    plan = "\n".join(f"{i}. {s}" for i, s in enumerate(state.get("plan") or [], 1))
    user = (
        f"Question: {state['question']}\n\n"
        f"Dataset profile:\n{state['profile']}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Script:\n```python\n{state['code']}\n```\n\n"
        f"{state['execution']['for_llm']}"
    )
    reply = ctx.llm.chat(
        [{"role": "system", "content": prompts.REVIEWER}, {"role": "user", "content": user}],
        agent="reviewer",
        json_mode=True,
    )
    try:
        data = parse_json(reply)
    except ValueError:
        data = {}
    verdict = str(data.get("verdict", "approve")).lower().strip()
    issues = [str(i) for i in data.get("issues") or [] if str(i).strip()]
    if verdict not in ("approve", "revise") or (verdict == "revise" and not issues):
        verdict = "approve"  # unparseable or empty critique: don't loop on nothing
    rounds = state.get("review_rounds", 0) + (verdict == "revise")
    review = {"verdict": verdict, "issues": issues, "summary": str(data.get("summary", ""))}
    ctx.tracker.event("reviewer", "review", review | {"round": rounds})
    return {"review": review, "review_rounds": rounds}
