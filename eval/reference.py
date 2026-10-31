"""
Reference ("ground truth") cleaning functions for the synthetic datasets.

The evaluation questions are answered by running hand-written pandas code on
data cleaned with these functions. Each function documents the cleaning rules,
and the questions in eval/questions.py spell out any rule that affects the
answer (e.g. "exclude orders with more than 100 units"), so the expected answer
is unambiguous for the agent too.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _to_number(series: pd.Series) -> pd.Series:
    """'$1,234.50' / '85,000' / '42.1%' -> 1234.5 / 85000 / 42.1"""
    cleaned = series.astype(str).str.replace(r"[$,%\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_sales() -> pd.DataFrame:
    """Dedupe exact rows, fix text casing, parse numbers/dates, drop 999-unit outliers."""
    df = pd.read_csv(DATA_DIR / "sales.csv").drop_duplicates()
    df["region"] = df["region"].str.strip().str.title()
    df["channel"] = df["channel"].str.strip().str.title()
    df["revenue"] = _to_number(df["revenue"])
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
    df = df[df["units"] <= 100]                      # remove data-entry outliers
    df["quarter"] = df["order_date"].dt.quarter
    df["month"] = df["order_date"].dt.month
    return df


def clean_web() -> pd.DataFrame:
    """Fix channel casing, parse bounce_rate '42.1%' -> 42.1, parse dates. Drop duplicates."""
    df = pd.read_csv(DATA_DIR / "web_traffic.csv")
    df["channel"] = df["channel"].str.strip().str.title()
    df["bounce_rate"] = _to_number(df["bounce_rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean_hr() -> pd.DataFrame:
    """Parse salary text, fix x10 salary typos (> 1,000,000 -> /10), normalise overtime/department."""
    df = pd.read_excel(DATA_DIR / "hr_employees.xlsx")
    df["department"] = df["department"].str.strip().str.title()
    df.loc[df["department"] == "Hr", "department"] = "HR"
    df["salary"] = _to_number(df["salary"])
    df.loc[df["salary"] > 1_000_000, "salary"] = df["salary"] / 10
    df["overtime"] = df["overtime"].str.strip().str.lower().str[0].map({"y": "Yes", "n": "No"})
    df["hire_date"] = pd.to_datetime(df["hire_date"], format="mixed")
    df["left"] = df["attrition"] == "Yes"
    return df


def clean_tickets() -> pd.DataFrame:
    """Fix priority casing, parse timestamps, mark invalid resolution times (<0 or >1000h) as NaN."""
    df = pd.read_csv(DATA_DIR / "support_tickets.csv")
    df["priority"] = df["priority"].str.strip().str.title()
    df["created_at"] = pd.to_datetime(df["created_at"])
    bad = (df["resolution_hours"] < 0) | (df["resolution_hours"] > 1000)
    df.loc[bad, "resolution_hours"] = np.nan
    return df


CLEANERS = {
    "sales.csv": clean_sales,
    "web_traffic.csv": clean_web,
    "hr_employees.xlsx": clean_hr,
    "support_tickets.csv": clean_tickets,
}
