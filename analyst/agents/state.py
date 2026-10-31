"""Graph state shared by all agents, and the per-run context (LLM, settings, output dir)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from analyst.config import Settings
from analyst.tracker import RunTracker


class AnalysisState(TypedDict, total=False):
    question: str
    profile: str                    # deterministic dataset profile (markdown)
    plan: list[str]
    plan_feedback: str              # human feedback on a rejected plan
    approved: bool | None           # human decision on the plan (None = not asked)
    code: str                       # current script
    version: int                    # how many scripts have been written
    fix_attempts: int               # failed executions since the last new/revised script
    execution: dict                 # latest ExecResult as a dict
    last_good: dict                 # latest *successful* {"code", "execution"}
    attempts: list[dict]            # log of every execution (version, ok, error)
    review: dict                    # latest {"verdict", "issues", "summary"}
    review_rounds: int
    solo_messages: list[dict]       # conversation memory of the single-agent baseline
    solo_turns: int
    answer: str                     # one-line answer (graded by the eval)
    report: str                     # full markdown report
    status: str                     # "completed" | "failed" | "rejected"


@dataclass
class RunContext:
    llm: Any                        # analyst.llm.LLM or FakeLLM
    settings: Settings
    tracker: RunTracker
    run_dir: Path                   # outputs/<run_id>/ - report, charts, final code
    data_files: list[Path] = field(default_factory=list)

    @property
    def file_names(self) -> str:
        return ", ".join(f'"{p.name}"' for p in self.data_files)
