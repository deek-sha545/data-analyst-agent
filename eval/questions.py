"""
Evaluation set: 35 business questions over the 4 synthetic datasets.

Each expected answer is *computed* from the reference cleaners in
eval/reference.py (never typed in by hand), so answers stay correct if the
datasets are regenerated. Any cleaning rule that changes the answer is spelled
out in the question, so there is exactly one right answer.

Grading (eval/grader.py) looks only at the agent's one-line "**Answer:**":
  * category - the expected label must be the FIRST of the candidate labels mentioned,
  * number   - some number in the answer must be within `tol` of the expected value
               (percent/fraction forms are both accepted when `pct=True`).
A question can require both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import pandas as pd

from eval.reference import clean_hr, clean_sales, clean_tickets, clean_web

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Laptop Pro", "Laptop Air", "Phone X", "Phone Mini", "Headphones", "Smartwatch"]
DEPARTMENTS = ["Engineering", "Sales", "Support", "Marketing", "Finance", "HR"]

SALES_RULES = ("Remove exact duplicate rows and exclude orders with more than 100 units (data-entry errors); "
               "keep orders with a missing region unless the question is about regions.")
HR_RULES = "Salaries above 1,000,000 are typos with an extra zero: divide them by 10."
TICKET_RULES = "Treat resolution times below 0 or above 1000 hours as invalid (ignore them)."


@dataclass
class Question:
    id: str
    file: str
    question: str
    expected: Callable[[], dict]          # -> {"category"?: str, "number"?: float}
    candidates: list[str] = field(default_factory=list)
    tol: float = 0.0                      # absolute tolerance for "number"
    rel_tol: float = 0.0                  # relative tolerance (used if larger than tol)
    pct: bool = False                     # number is a percentage: accept 27.2 or 0.272
    signed: bool = False                  # sign matters (else "fell 27%" == -27%)

    @property
    def answer(self) -> dict:
        return _cached_answer(self.id)


# ---------------------------------------------------------------- helpers
@cache
def S() -> pd.DataFrame:
    return clean_sales()


@cache
def W() -> pd.DataFrame:
    return clean_web()


@cache
def H() -> pd.DataFrame:
    return clean_hr()


@cache
def T() -> pd.DataFrame:
    return clean_tickets()


def _q_change(df: pd.DataFrame) -> pd.Series:
    by = df.pivot_table(index="region", columns="quarter", values="revenue", aggfunc="sum")
    return (by[3] - by[2]) / by[2] * 100


def _daily_sessions() -> pd.Series:
    return W().groupby("date")["sessions"].sum()


def _weekend_drop() -> float:
    d = _daily_sessions()
    d = d[(d.index < "2024-06-10") | (d.index > "2024-06-12")]
    weekend = d[d.index.day_of_week >= 5].mean()
    weekday = d[d.index.day_of_week < 5].mean()
    return (weekday - weekend) / weekday * 100


def _paid_campaign_ratio() -> float:
    paid = W()[W()["channel"] == "Paid"].set_index("date")["sessions"]
    campaign = paid["2024-11-20":"2024-12-02"].mean()
    october = paid["2024-10-01":"2024-10-31"].mean()
    return campaign / october


def _organic_growth() -> float:
    org = W()[W()["channel"] == "Organic"]
    m = org.groupby(org["date"].dt.month)["sessions"].sum()
    return (m[12] - m[1]) / m[1] * 100


def _valid_tickets() -> pd.DataFrame:
    return T().dropna(subset=["resolution_hours"])


QUESTIONS: list[Question] = [
    # ------------------------------------------------------------- sales.csv
    Question("sales_q2q3_drop", "sales.csv",
             f"Which region had the largest percentage revenue decline from Q2 to Q3 2024, and how large was it (in %)? {SALES_RULES}",
             lambda: {"category": _q_change(S()).idxmin(), "number": _q_change(S()).min()},
             REGIONS, tol=1.0, pct=True),
    Question("sales_top_region", "sales.csv",
             f"Which region generated the highest total revenue in 2024? {SALES_RULES}",
             lambda: {"category": S().groupby("region")["revenue"].sum().idxmax()}, REGIONS),
    Question("sales_total_revenue", "sales.csv",
             f"What was the total revenue in 2024? {SALES_RULES}",
             lambda: {"number": S()["revenue"].sum()}, rel_tol=0.01),
    Question("sales_discount_channel", "sales.csv",
             "Which sales channel has the highest average discount (discount_pct), and what is that average discount in percent?",
             lambda: {"category": S().groupby("channel")["discount_pct"].mean().idxmax(),
                      "number": S().groupby("channel")["discount_pct"].mean().max() * 100},
             ["Online", "Retail", "Partner"], tol=0.5, pct=True),
    Question("sales_busiest_month", "sales.csv",
             f"In which month of 2024 were the most orders placed? Answer with the month name. {SALES_RULES}",
             lambda: {"category": MONTHS[S().groupby("month").size().idxmax() - 1]}, MONTHS),
    Question("sales_west_q3_product", "sales.csv",
             f"Which product generated the most revenue in the West region in Q3 2024? {SALES_RULES}",
             lambda: {"category": S()[(S()["region"] == "West") & (S()["quarter"] == 3)]
                      .groupby("product")["revenue"].sum().idxmax()}, PRODUCTS),
    Question("sales_order_count", "sales.csv",
             f"How many valid orders are in the dataset? {SALES_RULES}",
             lambda: {"number": len(S())}, tol=0),
    Question("sales_online_aov", "sales.csv",
             f"What is the average revenue per order for the Online channel? {SALES_RULES}",
             lambda: {"number": S()[S()["channel"] == "Online"]["revenue"].mean()}, rel_tol=0.01),
    Question("sales_top_category", "sales.csv",
             f"Which product category generated the most revenue? {SALES_RULES}",
             lambda: {"category": S().groupby("category")["revenue"].sum().idxmax()},
             ["Phones", "Computers", "Audio", "Wearables"]),

    # --------------------------------------------------------- web_traffic.csv
    Question("web_best_conversion", "web_traffic.csv",
             "Which marketing channel has the highest conversion rate (total conversions / total sessions), and what is it in %?",
             lambda: {"category": (W().groupby("channel")["conversions"].sum() / W().groupby("channel")["sessions"].sum()).idxmax(),
                      "number": (W().groupby("channel")["conversions"].sum() / W().groupby("channel")["sessions"].sum()).max() * 100},
             ["Organic", "Paid", "Social", "Email", "Referral", "Direct"], tol=0.3, pct=True),
    Question("web_worst_conversion", "web_traffic.csv",
             "Which marketing channel has the lowest conversion rate (total conversions / total sessions)?",
             lambda: {"category": (W().groupby("channel")["conversions"].sum() / W().groupby("channel")["sessions"].sum()).idxmin()},
             ["Organic", "Paid", "Social", "Email", "Referral", "Direct"]),
    Question("web_outage_days", "web_traffic.csv",
             "On how many days were total sessions across all channels exactly zero (a tracking outage)?",
             lambda: {"number": int((_daily_sessions() == 0).sum())}, tol=0),
    Question("web_weekend_drop", "web_traffic.csv",
             "Comparing average total daily sessions (all channels combined), by how many percent are weekend days lower "
             "than weekdays? Exclude the tracking-outage days 2024-06-10 to 2024-06-12.",
             lambda: {"number": _weekend_drop()}, tol=2.0, pct=True),
    Question("web_total_ad_spend", "web_traffic.csv",
             "What was the total ad spend across all channels in 2024?",
             lambda: {"number": W()["ad_spend"].sum()}, rel_tol=0.01),
    Question("web_paid_campaign", "web_traffic.csv",
             "During the campaign from 2024-11-20 to 2024-12-02 (inclusive), how many times higher were the average daily "
             "Paid-channel sessions than the average daily Paid sessions in October 2024?",
             lambda: {"number": _paid_campaign_ratio()}, tol=0.25),
    Question("web_top_channel", "web_traffic.csv",
             "Which channel brought the most total sessions in 2024?",
             lambda: {"category": W().groupby("channel")["sessions"].sum().idxmax()},
             ["Organic", "Paid", "Social", "Email", "Referral", "Direct"]),
    Question("web_organic_growth", "web_traffic.csv",
             "By how many percent did monthly Organic sessions grow from January 2024 to December 2024 (monthly totals)?",
             lambda: {"number": _organic_growth()}, tol=3.0, pct=True),
    Question("web_paid_cpa", "web_traffic.csv",
             "For the Paid channel, what is the cost per conversion (total ad_spend divided by total conversions)?",
             lambda: {"number": W()[W()["channel"] == "Paid"]["ad_spend"].sum() / W()[W()["channel"] == "Paid"]["conversions"].sum()},
             rel_tol=0.02),

    # ------------------------------------------------------ hr_employees.xlsx
    Question("hr_top_attrition", "hr_employees.xlsx",
             "Which department has the highest attrition rate (share of employees with attrition = Yes), and what is that rate in %?",
             lambda: {"category": H().groupby("department")["left"].mean().idxmax(),
                      "number": H().groupby("department")["left"].mean().max() * 100},
             DEPARTMENTS, tol=1.0, pct=True),
    Question("hr_low_attrition", "hr_employees.xlsx",
             "Which department has the lowest attrition rate?",
             lambda: {"category": H().groupby("department")["left"].mean().idxmin()}, DEPARTMENTS),
    Question("hr_overtime_ratio", "hr_employees.xlsx",
             "How many times higher is the attrition rate of employees who work overtime than of those who don't? "
             "(The overtime column mixes Yes/No/Y/N in different cases.)",
             lambda: {"number": H().groupby("overtime")["left"].mean()["Yes"] / H().groupby("overtime")["left"].mean()["No"]},
             tol=0.2),
    Question("hr_eng_median_salary", "hr_employees.xlsx",
             f"What is the median salary in the Engineering department? {HR_RULES}",
             lambda: {"number": H()[H()["department"] == "Engineering"]["salary"].median()}, rel_tol=0.01),
    Question("hr_top_median_salary", "hr_employees.xlsx",
             f"Which department has the highest median salary? {HR_RULES}",
             lambda: {"category": H().groupby("department")["salary"].median().idxmax()}, DEPARTMENTS),
    Question("hr_low_satisfaction_attrition", "hr_employees.xlsx",
             "What is the attrition rate (in %) among employees with a satisfaction score of 4 or lower?",
             lambda: {"number": H()[H()["satisfaction"] <= 4]["left"].mean() * 100}, tol=1.0, pct=True),
    Question("hr_level5_salary", "hr_employees.xlsx",
             f"What is the average salary of employees at job_level 5? {HR_RULES}",
             lambda: {"number": H()[H()["job_level"] == 5]["salary"].mean()}, rel_tol=0.01),
    Question("hr_remote_count", "hr_employees.xlsx",
             "How many employees work remotely?",
             lambda: {"number": int(H()["remote"].sum())}, tol=0),
    Question("hr_avg_rating", "hr_employees.xlsx",
             "What is the average performance rating, ignoring missing ratings?",
             lambda: {"number": H()["performance_rating"].mean()}, tol=0.02),

    # ---------------------------------------------------- support_tickets.csv
    Question("tickets_slowest_agent", "support_tickets.csv",
             f"Which support agent has the longest median resolution time, and what is it in hours? {TICKET_RULES}",
             lambda: {"category": _valid_tickets().groupby("agent_id")["resolution_hours"].median().idxmax(),
                      "number": _valid_tickets().groupby("agent_id")["resolution_hours"].median().max()},
             [f"A{i:02d}" for i in range(1, 13)], rel_tol=0.02),
    Question("tickets_best_channel", "support_tickets.csv",
             "Which support channel has the highest average satisfaction score?",
             lambda: {"category": T().groupby("channel")["satisfaction_score"].mean().idxmax()}, ["Email", "Chat", "Phone"]),
    Question("tickets_busiest_month", "support_tickets.csv",
             "In which month of 2024 were the most tickets created? Answer with the month name.",
             lambda: {"category": MONTHS[T().groupby(T()["created_at"].dt.month).size().idxmax() - 1]}, MONTHS),
    Question("tickets_september_product", "support_tickets.csv",
             "Which product had the most support tickets in September 2024?",
             lambda: {"category": T()[T()["created_at"].dt.month == 9].groupby("product").size().idxmax()}, PRODUCTS),
    Question("tickets_median_resolution", "support_tickets.csv",
             f"What is the overall median resolution time in hours? {TICKET_RULES}",
             lambda: {"number": _valid_tickets()["resolution_hours"].median()}, rel_tol=0.02),
    Question("tickets_open_count", "support_tickets.csv",
             "How many tickets are still open?",
             lambda: {"number": int((T()["status"] == "Open").sum())}, tol=0),
    Question("tickets_urgent_satisfaction", "support_tickets.csv",
             "What is the average satisfaction score for Urgent-priority tickets (ignoring missing scores)? "
             "Note the priority column has inconsistent casing.",
             lambda: {"number": T()[T()["priority"] == "Urgent"]["satisfaction_score"].mean()}, tol=0.05),
    Question("tickets_corr", "support_tickets.csv",
             f"What is the Pearson correlation between resolution time and satisfaction score? {TICKET_RULES} "
             "Use only tickets that have both values.",
             lambda: {"number": T()[["resolution_hours", "satisfaction_score"]].dropna().corr().iloc[0, 1]},
             tol=0.03, signed=True),
]

BY_ID = {q.id: q for q in QUESTIONS}


@cache
def _cached_answer(qid: str) -> dict:
    ans = BY_ID[qid].expected()
    if "number" in ans:
        ans["number"] = float(ans["number"])
    return ans
