"""
Deterministic grader for the agent's one-line answer.

Deterministic on purpose: an LLM judge adds cost, variance and its own errors,
and every question here has a single label and/or number as its answer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from eval.questions import MONTHS, Question

_NUM_RE = re.compile(
    r"(?<![\w.])(-|−|minus\s)?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|percent|k\b|thousand|m\b|mn\b|million|bn\b|billion|x\b|times)?",
    re.IGNORECASE,
)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
_MONTH_ABBR = {m: [m, m[:3] + r"\b"] for m in MONTHS}


def extract_numbers(text: str) -> list[float]:
    nums = []
    for sign, whole, frac, unit in _NUM_RE.findall(text):
        value = float(whole.replace(",", "") + (frac or ""))
        if unit and unit.lower() in _SCALE:
            value *= _SCALE[unit.lower()]
        if sign:
            value = -value
        nums.append(value)
    return nums


def first_candidate(text: str, candidates: list[str]) -> str | None:
    """The candidate label mentioned earliest in the text (case-insensitive, whole word)."""
    best, best_pos = None, math.inf
    for c in candidates:
        patterns = _MONTH_ABBR.get(c, [re.escape(c) + r"\b"])
        for pat in patterns:
            m = re.search(r"(?<![\w-])" + pat, text, re.IGNORECASE)
            if m and m.start() < best_pos:
                best, best_pos = c, m.start()
    return best


def number_matches(q: Question, expected: float, found: list[float]) -> bool:
    tol = max(q.tol, abs(expected) * q.rel_tol)
    tol = max(tol, 1e-9)
    targets = [expected]
    if q.pct:  # 27.2 (%) can also be written as the fraction 0.272
        targets.append(expected / 100)
        tol_frac = tol / 100
    for n in found:
        for i, t in enumerate(targets):
            tt = tol if i == 0 else tol_frac
            if q.signed:
                if abs(n - t) <= tt + 1e-12:
                    return True
            elif abs(abs(n) - abs(t)) <= tt + 1e-12:
                return True
    return False


@dataclass
class Grade:
    correct: bool
    category_ok: bool | None
    number_ok: bool | None
    reason: str


def grade(q: Question, answer: str) -> Grade:
    expected = q.answer
    if not answer.strip():
        return Grade(False, None, None, "empty answer")
    cat_ok = num_ok = None
    reasons = []
    if "category" in expected:
        got = first_candidate(answer, q.candidates)
        cat_ok = got == expected["category"]
        if not cat_ok:
            reasons.append(f"expected {expected['category']!r}, first label mentioned was {got!r}")
    if "number" in expected:
        found = extract_numbers(answer)
        num_ok = number_matches(q, expected["number"], found)
        if not num_ok:
            reasons.append(f"expected ~{expected['number']:.4g}, numbers found {found[:6]}")
    correct = all(x for x in (cat_ok, num_ok) if x is not None)
    return Grade(correct, cat_ok, num_ok, "; ".join(reasons) or "ok")
