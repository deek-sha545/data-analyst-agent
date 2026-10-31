"""Evaluation set + deterministic grader."""
import pytest

from eval.grader import extract_numbers, first_candidate, grade
from eval.questions import BY_ID, QUESTIONS


def test_question_set_is_well_formed():
    assert len(QUESTIONS) >= 30
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)
    assert {q.file for q in QUESTIONS} == {"sales.csv", "web_traffic.csv", "hr_employees.xlsx", "support_tickets.csv"}
    for q in QUESTIONS:
        ans = q.answer
        assert ans, q.id
        if "category" in ans:
            assert ans["category"] in q.candidates, q.id
        if "number" in ans:
            assert q.tol > 0 or q.rel_tol > 0 or float(ans["number"]).is_integer(), q.id


def test_planted_answers():
    assert BY_ID["sales_q2q3_drop"].answer["category"] == "West"
    assert BY_ID["web_best_conversion"].answer["category"] == "Email"
    assert BY_ID["hr_top_attrition"].answer["category"] == "Sales"
    assert BY_ID["tickets_slowest_agent"].answer["category"] == "A07"
    assert BY_ID["web_outage_days"].answer["number"] == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Revenue was $6,616,681.98 in total", [6616681.98]),
        ("fell by 27.2%, from 1.2 million", [27.2, 1.2e6]),
        ("correlation of -0.71", [-0.71]),
        ("correlation of −0.71", [-0.71]),
        ("agent A07 took 39.95 hours in Q3", [39.95]),
        ("about 3.0x higher (450k sessions)", [3.0, 450000]),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_first_candidate():
    regions = ["North", "South", "East", "West"]
    assert first_candidate("West fell most, while East grew", regions) == "West"
    assert first_candidate("Northwestern?", regions) is None
    assert first_candidate("Most orders were in Dec 2024", BY_ID["sales_busiest_month"].candidates) == "December"


def test_grade_category_and_number():
    q = BY_ID["sales_q2q3_drop"]
    assert grade(q, "West had the largest decline, falling 27.24% from Q2 to Q3.").correct
    assert grade(q, "The West region dropped by -27.2%.").correct
    assert grade(q, "West declined by 0.272 (27%)").correct
    g = grade(q, "East had the largest decline, 27.2%.")
    assert not g.correct and g.category_ok is False and g.number_ok is True
    assert not grade(q, "West had the largest decline, 35%.").correct
    assert not grade(q, "").correct


def test_grade_signed_and_tolerances():
    corr = BY_ID["tickets_corr"]
    assert grade(corr, "The correlation is -0.71 (strong negative).").correct
    assert not grade(corr, "The correlation is 0.71.").correct
    total = BY_ID["sales_total_revenue"]
    assert grade(total, f"Total revenue was ${total.answer['number'] * 1.005:,.2f}.").correct
    assert grade(total, f"Total revenue was about ${total.answer['number'] / 1e6:.2f} million.").correct
    assert not grade(total, f"Total revenue was ${total.answer['number'] * 1.05:,.0f}.").correct
    count = BY_ID["tickets_open_count"]
    assert grade(count, "There are 141 open tickets.").correct
    assert not grade(count, "There are 142 open tickets.").correct


def test_unhinted_suite_removes_only_error_hints():
    from eval.questions import HINTS

    hinted = [q for q in QUESTIONS if q.has_hints]
    assert len(hinted) >= 12
    for q in hinted:
        text = q.text("unhinted")
        assert all(h.strip() not in text for h in HINTS), q.id
        assert text.endswith(("?", ".")) and "  " not in text
    sales = BY_ID["sales_total_revenue"].text("unhinted")
    assert "100 units" not in sales and "Keep orders with a missing region" in sales  # scope rule stays
    assert not BY_ID["web_outage_days"].has_hints
    assert BY_ID["web_outage_days"].text("unhinted") == BY_ID["web_outage_days"].question
