"""
LangGraph wiring for the three variants compared in the experiments:

  multi        profile -> planner -> [human approval] -> coder -> executor -> reviewer -> reporter
                                                           ^__ fix __|  ^____ revise ____|
  no_reviewer  same, without the reviewer (executor success goes straight to the reporter)
  single       profile -> solo agent (loop: write code / see output) -> report

The self-correction loop (executor failure -> coder) is capped by
settings.max_fix_attempts, the review loop by settings.max_review_rounds.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from analyst.agents.coder import coder
from analyst.agents.executor import executor
from analyst.agents.planner import planner
from analyst.agents.reporter import reporter
from analyst.agents.reviewer import reviewer
from analyst.agents.solo import solo_agent
from analyst.agents.state import AnalysisState, RunContext
from analyst.profiling import profile_files

VARIANTS = ("multi", "no_reviewer", "single")


def profile_node(state: AnalysisState, ctx: RunContext) -> dict:
    profile = profile_files(ctx.data_files)
    ctx.tracker.event("system", "profile", profile)
    return {"profile": profile}


def human_approval(state: AnalysisState, ctx: RunContext) -> dict:
    """Pause the graph until a human approves, rejects, or comments on the plan.

    Note: on resume LangGraph re-runs this node from the top, so side effects before
    interrupt() would happen twice - the "awaiting_approval" event is emitted by the runner.
    """
    decision = interrupt({"plan": state["plan"], "question": state["question"]})
    if not isinstance(decision, dict):
        decision = {"approved": bool(decision)}
    approved = bool(decision.get("approved"))
    feedback = str(decision.get("feedback") or "").strip()
    ctx.tracker.event("system", "approval", {"approved": approved, "feedback": feedback})
    if approved:
        return {"approved": True}
    if feedback:
        return {"approved": False, "plan_feedback": feedback}
    return {"approved": False, "status": "rejected", "answer": "", "report": "Analysis cancelled: the plan was rejected."}


def build_graph(ctx: RunContext, variant: str = "multi", require_approval: bool = False):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    s = ctx.settings
    g = StateGraph(AnalysisState)
    g.add_node("profile", partial(profile_node, ctx=ctx))
    g.add_edge(START, "profile")

    if variant == "single":
        g.add_node("solo", partial(solo_agent, ctx=ctx))
        g.add_edge("profile", "solo")
        g.add_conditional_edges("solo", lambda st: END if st.get("report") else "solo", ["solo", END])
        return g.compile(checkpointer=InMemorySaver())

    g.add_node("planner", partial(planner, ctx=ctx))
    g.add_node("coder", partial(coder, ctx=ctx))
    g.add_node("executor", partial(executor, ctx=ctx))
    g.add_node("reporter", partial(reporter, ctx=ctx))
    g.add_edge("profile", "planner")

    if require_approval:
        g.add_node("human_approval", partial(human_approval, ctx=ctx))
        g.add_edge("planner", "human_approval")

        def after_approval(st: AnalysisState) -> str:
            if st.get("approved"):
                return "coder"
            return "planner" if st.get("plan_feedback") else END

        g.add_conditional_edges("human_approval", after_approval, ["coder", "planner", END])
    else:
        g.add_edge("planner", "coder")

    g.add_edge("coder", "executor")
    success_target = "reviewer" if variant == "multi" else "reporter"

    def after_execution(st: AnalysisState) -> str:
        if st["execution"]["ok"]:
            return success_target
        if st.get("fix_attempts", 0) <= s.max_fix_attempts:
            return "coder"  # self-correction
        return "reporter"   # give up: report the last good result or the failure

    g.add_conditional_edges("executor", after_execution, ["coder", success_target, "reporter"])

    if variant == "multi":
        g.add_node("reviewer", partial(reviewer, ctx=ctx))

        def after_review(st: AnalysisState) -> str:
            review = st["review"]
            if review["verdict"] == "revise" and st.get("review_rounds", 0) <= s.max_review_rounds:
                return "coder"
            return "reporter"

        g.add_conditional_edges("reviewer", after_review, ["coder", "reporter"])

    g.add_edge("reporter", END)
    return g.compile(checkpointer=InMemorySaver())
