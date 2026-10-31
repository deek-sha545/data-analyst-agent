"""Planner: turns the question + dataset profile into a short step-by-step analysis plan."""

from __future__ import annotations

from analyst import prompts
from analyst.agents.state import AnalysisState, RunContext
from analyst.llm import parse_json


def planner(state: AnalysisState, ctx: RunContext) -> dict:
    user = (
        f"Question: {state['question']}\n\n"
        f"Data files: {ctx.file_names}\n\n"
        f"Dataset profile:\n{state['profile']}"
    )
    if state.get("plan_feedback"):
        user += (
            f"\n\nA human reviewed your previous plan:\n- " + "\n- ".join(state.get("plan", []))
            + f"\n\nTheir feedback: {state['plan_feedback']}\nWrite an improved plan."
        )
    reply = ctx.llm.chat(
        [{"role": "system", "content": prompts.PLANNER}, {"role": "user", "content": user}],
        agent="planner",
        json_mode=True,
    )
    try:
        plan = parse_json(reply).get("plan") or []
        plan = [str(s).strip() for s in plan if str(s).strip()]
    except ValueError:
        plan = []
    if not plan:  # model ignored the JSON format: fall back to non-empty lines
        plan = [ln.strip(" -*0123456789.").strip() for ln in reply.splitlines() if ln.strip()][:7]
    ctx.tracker.event("planner", "plan", plan)
    return {"plan": plan, "plan_feedback": ""}
