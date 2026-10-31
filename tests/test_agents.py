"""
The agent graph end-to-end with a scripted FakeLLM (offline, deterministic).
Code really runs in the sandbox, so these also test the executor wiring.
"""
import json
from pathlib import Path

import pytest

from analyst.agents.reporter import extract_answer
from analyst.config import get_settings
from analyst.llm import FakeLLM, LLMError
from analyst.runner import run_analysis
from eval.reference import DATA_DIR

SALES = DATA_DIR / "sales.csv"

GOOD_CODE = """```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("sales.csv").drop_duplicates()
df["region"] = df["region"].str.strip().str.title()
counts = df["region"].value_counts()
print(counts)
counts.plot.bar(); plt.title("Orders by region")
print("RESULT: East has the most orders")
```"""
BAD_CODE = "```python\nimport pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(df['no_such_column'])\n```"
UNSAFE_CODE = "```python\nimport os\nos.system('echo hi')\n```"
PLAN = json.dumps({"plan": ["Load sales.csv", "Normalise region casing", "Count orders per region"]})
APPROVE = json.dumps({"verdict": "approve", "issues": [], "summary": "Looks right."})
REVISE = json.dumps({"verdict": "revise", "issues": ["Also drop rows with missing region"], "summary": "Close."})
REPORT = "**Answer:** East has the most orders.\n\n## Key findings\n- East leads\n\n![x](bogus.png)"


def scripted(**replies):
    """FakeLLM whose replies are queued per agent."""
    queues = {agent: list(r) for agent, r in replies.items()}

    def reply(agent, messages):
        if not queues.get(agent):
            raise AssertionError(f"unexpected call to {agent}")
        return queues[agent].pop(0)

    return FakeLLM(reply), queues


@pytest.fixture
def settings(tmp_path):
    return get_settings("gemini", "fake", outputs_dir=tmp_path / "out", logs_dir=tmp_path / "logs", sandbox_timeout=60)


def test_happy_path(settings):
    llm, queues = scripted(planner=[PLAN], coder=[GOOD_CODE], reviewer=[APPROVE], reporter=[REPORT])
    res = run_analysis("Which region has most orders?", [SALES], settings=settings, llm=llm)
    assert res.status == "completed", res.error
    assert res.answer == "East has the most orders."
    assert res.plan[0] == "Load sales.csv"
    assert len(res.charts) == 1 and Path(res.charts[0]).name == "chart_1.png"
    report = Path(res.report_path).read_text()
    assert "![Chart 1](chart_1.png)" in report and "bogus.png" not in report
    assert (Path(res.report_path).parent / "analysis.py").exists()
    assert all(not q for q in queues.values())
    assert res.metrics["llm_calls"] == 4
    assert (settings.logs_dir / f"{res.run_id}.json").exists()


def test_self_correction_loop(settings):
    llm, _ = scripted(planner=[PLAN], coder=[BAD_CODE, UNSAFE_CODE, GOOD_CODE], reviewer=[APPROVE], reporter=[REPORT])
    res = run_analysis("q", [SALES], settings=settings, llm=llm)
    assert res.status == "completed"
    assert [a["ok"] for a in res.attempts] == [False, False, True]
    assert "KeyError" in res.attempts[0]["error"]
    # the fix prompt shows the coder its failing code and the error
    coder_calls = [m for a, m in llm.calls if a == "coder"]
    assert "no_such_column" in coder_calls[1][-1]["content"]
    assert "REJECTED" in coder_calls[2][-1]["content"]


def test_gives_up_after_max_fix_attempts(settings):
    failing = "**Answer:** Could not be determined - the column does not exist."
    llm, _ = scripted(planner=[PLAN], coder=[BAD_CODE] * 4, reporter=[failing])
    res = run_analysis("q", [SALES], settings=settings, llm=llm)
    assert res.status == "failed"
    assert len(res.attempts) == settings.max_fix_attempts + 1
    assert res.answer.startswith("Could not be determined")


def test_reviewer_revision_round(settings):
    llm, _ = scripted(planner=[PLAN], coder=[GOOD_CODE, GOOD_CODE], reviewer=[REVISE, APPROVE], reporter=[REPORT])
    res = run_analysis("q", [SALES], settings=settings, llm=llm)
    assert res.status == "completed" and res.review["verdict"] == "approve"
    revise_prompt = [m for a, m in llm.calls if a == "coder"][1][-1]["content"]
    assert "Also drop rows with missing region" in revise_prompt


def test_review_rounds_are_capped(settings):
    n = settings.max_review_rounds + 1
    llm, _ = scripted(planner=[PLAN], coder=[GOOD_CODE] * n, reviewer=[REVISE] * n, reporter=[REPORT])
    res = run_analysis("q", [SALES], settings=settings, llm=llm)
    assert res.status == "completed"
    # unresolved concerns are passed to the reporter as caveats
    assert "Unresolved reviewer concerns" in [m for a, m in llm.calls if a == "reporter"][0][-1]["content"]


def test_failed_revision_falls_back_to_last_good(settings):
    llm, _ = scripted(planner=[PLAN], coder=[GOOD_CODE] + [BAD_CODE] * 4, reviewer=[REVISE], reporter=[REPORT])
    res = run_analysis("q", [SALES], settings=settings, llm=llm)
    assert res.status == "completed"
    assert "last version that ran successfully" in res.report
    assert "value_counts" in res.code  # the good script is the one returned


def test_no_reviewer_variant(settings):
    llm, _ = scripted(planner=[PLAN], coder=[GOOD_CODE], reporter=[REPORT])
    res = run_analysis("q", [SALES], "no_reviewer", settings=settings, llm=llm)
    assert res.status == "completed"
    assert "reviewer" not in res.metrics["by_agent"]


def test_single_agent_variant(settings):
    llm, _ = scripted(solo=[BAD_CODE, GOOD_CODE, REPORT])
    res = run_analysis("q", [SALES], "single", settings=settings, llm=llm)
    assert res.status == "completed"
    assert res.answer == "East has the most orders."
    assert [a["ok"] for a in res.attempts] == [False, True]
    assert len(res.charts) == 1
    # the agent saw its own execution results
    assert "FAILED" in llm.calls[1][1][-1]["content"]


def test_single_agent_budget_forces_report(settings):
    budget = 1 + settings.max_fix_attempts + settings.max_review_rounds
    llm, _ = scripted(solo=[BAD_CODE] * budget + [REPORT])
    res = run_analysis("q", [SALES], "single", settings=settings, llm=llm)
    assert len(res.attempts) == budget
    assert res.status == "failed"  # nothing ever ran successfully
    assert "used all your code runs" in llm.calls[-1][1][-1]["content"]


def test_human_approval_feedback_then_approve(settings):
    plan2 = json.dumps({"plan": ["Better step"]})
    llm, _ = scripted(planner=[PLAN, plan2], coder=[GOOD_CODE], reviewer=[APPROVE], reporter=[REPORT])
    seen = []

    def human(pending):
        seen.append(pending["plan"])
        return {"approved": False, "feedback": "use fewer steps"} if len(seen) == 1 else {"approved": True}

    res = run_analysis("q", [SALES], settings=settings, llm=llm, require_approval=True, approval_callback=human)
    assert res.status == "completed"
    assert seen == [json.loads(PLAN)["plan"], ["Better step"]]
    assert "use fewer steps" in [m for a, m in llm.calls if a == "planner"][1][-1]["content"]


def test_human_rejection_stops_before_code(settings):
    llm, _ = scripted(planner=[PLAN])
    res = run_analysis("q", [SALES], settings=settings, llm=llm, require_approval=True,
                       approval_callback=lambda p: {"approved": False})
    assert res.status == "rejected"
    assert res.attempts == []


def test_llm_failure_reported_not_raised(settings):
    def boom(agent, messages):
        raise LLMError("quota exceeded")

    res = run_analysis("q", [SALES], settings=settings, llm=FakeLLM(boom))
    assert res.status == "error" and "quota exceeded" in res.error


def test_bad_inputs(settings, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_analysis("q", [tmp_path / "missing.csv"], settings=settings, llm=FakeLLM([]))
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(ValueError):
        run_analysis("q", [txt], settings=settings, llm=FakeLLM([]))
    with pytest.raises(ValueError):
        run_analysis("q", [SALES], "nonsense", settings=settings, llm=FakeLLM([]))


def test_events_stream_in_order(settings):
    llm, _ = scripted(planner=[PLAN], coder=[GOOD_CODE], reviewer=[APPROVE], reporter=[REPORT])
    from analyst.tracker import RunTracker

    tracker = RunTracker()
    kinds = []
    tracker.listeners.append(lambda e: kinds.append(e.kind))
    run_analysis("q", [SALES], settings=settings, llm=llm, tracker=tracker)
    assert kinds == ["start", "profile", "plan", "code", "execution", "review", "report", "done"]


def test_extract_answer():
    assert extract_answer("**Answer:** West fell 27%.\n\n## Key") == "West fell 27%."
    assert extract_answer("intro\n\n**Answer**: two\nlines\n## X") == "two lines"
    assert extract_answer("## Title\n\nFirst para here.\n\nSecond") == "First para here."
