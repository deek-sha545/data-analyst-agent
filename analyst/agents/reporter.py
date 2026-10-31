"""
Reporter: writes the final Markdown report from the *printed output* of the
best successful execution, then attaches the charts and saves everything to
outputs/<run_id>/ (report.md, chart_N.png, analysis.py).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from analyst import prompts
from analyst.agents.state import AnalysisState, RunContext

_ANSWER_RE = re.compile(r"\*\*Answer:?\*\*:?\s*(.+?)(?:\n\s*\n|\n#|\Z)", re.DOTALL | re.IGNORECASE)


def extract_answer(report: str) -> str:
    """The text after '**Answer:**' (up to the next blank line / heading); else the first paragraph."""
    m = _ANSWER_RE.search(report)
    if m:
        return " ".join(m.group(1).split())
    for para in report.split("\n\n"):
        para = para.strip()
        if para and not para.startswith("#"):
            return " ".join(para.split())
    return ""


def finalize(report: str, code: str, figures: list[str], ctx: RunContext, note: str = "") -> dict:
    """Copy charts next to the report, append them, save report + final code. Returns the state update."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    report = report.strip()
    # Drop any image links the model added anyway: charts are attached below, deterministically.
    report = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", report)
    if note:
        report += f"\n\n> **Note:** {note}"
    charts = []
    for i, fig in enumerate(figures, 1):
        dest = ctx.run_dir / f"chart_{i}.png"
        shutil.copy2(fig, dest)
        charts.append(dest.name)
    if charts:
        report += "\n\n## Charts\n\n" + "\n\n".join(f"![Chart {i}]({name})" for i, name in enumerate(charts, 1))
    if code:
        (ctx.run_dir / "analysis.py").write_text(code, encoding="utf-8")
    (ctx.run_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    answer = extract_answer(report)
    ctx.tracker.event("reporter", "report", {"answer": answer, "report": report, "charts": charts})
    return {"report": report, "answer": answer}


def reporter(state: AnalysisState, ctx: RunContext) -> dict:
    execution = state.get("execution") or {}
    last_good = state.get("last_good")
    note = ""
    if execution.get("ok"):
        code, used = state["code"], execution
    elif last_good:
        code, used = last_good["code"], last_good["execution"]
        note = "The latest revision of the analysis failed to run; this report is based on the last version that ran successfully."
    else:
        code, used = state.get("code", ""), None

    if used is None:
        attempts = "\n".join(f"- version {a['version']}: {a['error']}" for a in state.get("attempts", []))
        user = (
            f"Question: {state['question']}\n\nAttempts:\n{attempts}\n\n"
            f"Last execution:\n{execution.get('for_llm', 'no execution')}"
        )
        system, status, figures = prompts.REPORTER_FAILED, "failed", []
    else:
        review = state.get("review") or {}
        plan = "\n".join(f"{i}. {s}" for i, s in enumerate(state.get("plan") or [], 1))
        user = (
            f"Question: {state['question']}\n\n"
            + (f"Plan that was followed:\n{plan}\n\n" if plan else "")
            + f"Printed output of the analysis:\n{used['stdout']}\n\n"
            + f"Charts produced: {len(used['figures'])}\n"
            + (f"Reviewer's assessment: {review.get('summary', '')}\n" if review.get("summary") else "")
        )
        if review.get("verdict") == "revise":  # review rounds ran out with issues still open
            user += "Unresolved reviewer concerns (mention them as caveats):\n- " + "\n- ".join(review["issues"])
        system, status, figures = prompts.REPORTER, "completed", used["figures"]

    report = ctx.llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], agent="reporter"
    )
    return finalize(report, code, figures, ctx, note) | {"status": status}
