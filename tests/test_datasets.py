"""Check the synthetic datasets really contain the planted patterns and messiness.

If these fail, the evaluation's "known correct answers" would be wrong.
"""
import pandas as pd

from eval.reference import DATA_DIR, clean_hr, clean_sales, clean_tickets, clean_web


def test_all_dataset_files_exist():
    for name in ["sales.csv", "web_traffic.csv", "hr_employees.xlsx", "support_tickets.csv"]:
        assert (DATA_DIR / name).exists(), f"run data/generate_datasets.py first ({name} missing)"


def test_sales_is_messy():
    raw = pd.read_csv(DATA_DIR / "sales.csv")
    assert not pd.api.types.is_numeric_dtype(raw["revenue"])  # '$1,234.50' strings mixed in
    assert raw["region"].isna().any()                     # missing values
    assert raw.duplicated().sum() > 0                     # duplicate rows
    assert (raw["units"] == 999).sum() == 6               # outliers


def test_sales_planted_patterns():
    df = clean_sales()
    by_q = df.pivot_table(index="region", columns="quarter", values="revenue", aggfunc="sum")
    change_q2_q3 = (by_q[3] - by_q[2]) / by_q[2]
    assert change_q2_q3.idxmin() == "West" and change_q2_q3["West"] < -0.2
    assert df.groupby("region")["revenue"].sum().idxmax() == "East"
    assert df.groupby("channel")["discount_pct"].mean().idxmax() == "Online"


def test_web_planted_patterns():
    df = clean_web()
    conv = df.groupby("channel")["conversions"].sum() / df.groupby("channel")["sessions"].sum()
    assert conv.idxmax() == "Email" and conv.idxmin() == "Social"
    outage = df[df["date"].between("2024-06-10", "2024-06-12")]
    assert (outage["sessions"] == 0).all()


def test_hr_planted_patterns():
    df = clean_hr()
    attrition = df.groupby("department")["left"].mean()
    assert attrition.idxmax() == "Sales" and attrition.idxmin() == "Engineering"
    by_ot = df.groupby("overtime")["left"].mean()
    assert by_ot["Yes"] > 2 * by_ot["No"]
    assert df.groupby("department")["salary"].median().idxmax() == "Engineering"


def test_tickets_planted_patterns():
    df = clean_tickets()
    assert df.groupby("agent_id")["resolution_hours"].median().idxmax() == "A07"
    assert df.groupby(df["created_at"].dt.month).size().idxmax() == 9
    assert df.groupby("channel")["satisfaction_score"].mean().idxmax() == "Chat"
