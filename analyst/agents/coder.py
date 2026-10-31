"""
Coder: writes the analysis script. Three modes, picked from the state:
  * first draft  - from the plan,
  * fix          - the last execution failed (self-correction loop),
  * revise       - the Reviewer asked for changes.
"""

from __future__ import annotations

from analyst import prompts
from analyst.agents.state import AnalysisState, RunContext
from analyst.llm import extract_code


def _task(state: AnalysisState, ctx: RunContext) -> str:
    plan = "\n".join(f"{i}. {s}" for i, s in enumerate(state.get("plan") or [], 1))
    return (
        f"Question: {state['question']}\n\n"
        f"Data files (in the working directory): {ctx.file_names}\n\n"
        f"Dataset profile:\n{state['profile']}\n\n"
        f"Analysis plan:\n{plan}"
    )


def coder(state: AnalysisState, ctx: RunContext) -> dict:
    messages = [
        {"role": "system", "content": prompts.CODER},
        {"role": "user", "content": _task(state, ctx)},
    ]
    execution = state.get("execution")
    review = state.get("review")
    mode = "draft"
    if execution and not execution["ok"]:
        mode = "fix"
        messages += [
            {"role": "assistant", "content": f"```python\n{state['code']}\n```"},
            {"role": "user", "content": f"{prompts.CODER_FIX}\n\n{execution['for_llm']}"},
        ]
    elif review and review.get("verdict") == "revise" and execution:
        mode = "revise"
        issues = "\n".join(f"- {i}" for i in review.get("issues", []))
        messages += [
            {"role": "assistant", "content": f"```python\n{state['code']}\n```"},
            {
                "role": "user",
                "content": f"{prompts.CODER_REVISE}\n\nOutput of your script:\n{execution['for_llm']}\n\n"
                f"Reviewer's issues:\n{issues}",
            },
        ]

    code = extract_code(ctx.llm.chat(messages, agent="coder"))
    version = state.get("version", 0) + 1
    ctx.tracker.event("coder", "code", {"version": version, "mode": mode, "code": code})
    update: dict = {"code": code, "version": version}
    if mode != "fix":
        update["fix_attempts"] = 0  # a new or revised script gets a fresh set of fix attempts
    if mode == "revise":
        update["review"] = {**review, "verdict": "addressed"}
    return update
