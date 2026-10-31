"""
Generate 4 realistic synthetic datasets with *planted patterns* and *messy data*.

Why synthetic?  Because we planted the patterns ourselves, we KNOW the correct
answers. That lets us measure the agent's accuracy objectively (see eval/).

Run:  .venv/bin/python data/generate_datasets.py

Datasets (all for the year 2024):
  1. sales.csv            - ~3,000 orders
  2. web_traffic.csv      - daily sessions per marketing channel
  3. hr_employees.xlsx    - 1,200 employees (Excel on purpose, to test Excel loading)
  4. support_tickets.csv  - ~2,500 customer-support tickets

Planted patterns (the "truth" the agent should discover) are listed in
PLANTED_PATTERNS at the bottom and written to data/README.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
SEED = 42  # fixed seed => the same files every time


# ---------------------------------------------------------------------------
# Small helpers for making data "messy"
# ---------------------------------------------------------------------------
def pick_rows(rng: np.random.Generator, n_rows: int, fraction: float) -> np.ndarray:
    """Return random row positions covering `fraction` of the rows."""
    return rng.choice(n_rows, size=int(n_rows * fraction), replace=False)


def mess_up_text(rng, series: pd.Series, fraction: float) -> pd.Series:
    """Randomly change casing / add whitespace, e.g. 'West' -> ' west' or 'WEST'."""
    s = series.copy().astype(object)
    for i in pick_rows(rng, len(s), fraction):
        value = s.iat[i]
        if isinstance(value, str):
            style = rng.integers(3)
            s.iat[i] = [value.lower(), value.upper(), f" {value} "][style]
    return s


# ---------------------------------------------------------------------------
# 1. SALES
# ---------------------------------------------------------------------------
PRODUCTS = {  # product: (category, list price)
    "Laptop Pro": ("Computers", 1800.0),
    "Laptop Air": ("Computers", 1100.0),
    "Phone X": ("Phones", 950.0),
    "Phone Mini": ("Phones", 600.0),
    "Headphones": ("Audio", 180.0),
    "Smartwatch": ("Wearables", 320.0),
}
REGIONS = ["North", "South", "East", "West"]


def make_sales(rng: np.random.Generator) -> pd.DataFrame:
    n = 3000
    # Order dates: more orders in Nov/Dec (holiday season).
    month_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.4, 1.5])
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    day_weights = month_weights[days.month - 1]
    dates = rng.choice(days, size=n, p=day_weights / day_weights.sum())

    # East sells the most, West second (before its Q3 problem).
    regions = rng.choice(REGIONS, size=n, p=[0.22, 0.20, 0.31, 0.27])
    products = rng.choice(list(PRODUCTS), size=n, p=[0.18, 0.17, 0.2, 0.15, 0.17, 0.13])
    channels = rng.choice(["Online", "Retail", "Partner"], size=n, p=[0.5, 0.35, 0.15])

    df = pd.DataFrame({"order_date": pd.to_datetime(dates), "region": regions,
                       "product": products, "channel": channels})

    # PLANTED: Laptop Pro was out of stock in the West during Q3 (Jul-Sep).
    # We replace ~90% of those orders with cheap Headphones orders -> West Q3 revenue falls.
    stockout = (df["region"] == "West") & (df["product"] == "Laptop Pro") & df["order_date"].dt.month.isin([7, 8, 9])
    stockout_idx = df.index[stockout]
    replace_idx = rng.choice(stockout_idx, size=int(len(stockout_idx) * 0.9), replace=False)
    df.loc[replace_idx, "product"] = "Headphones"
    # PLANTED: West also loses Laptop Air + Phone X demand in Q3 (sales partly redirected)
    west_q3_other = (df["region"] == "West") & df["product"].isin(["Laptop Air", "Phone X"]) & df["order_date"].dt.month.isin([7, 8, 9])
    drop_idx = df.index[west_q3_other]
    df = df.drop(rng.choice(drop_idx, size=int(len(drop_idx) * 0.4), replace=False)).reset_index(drop=True)

    df["category"] = df["product"].map(lambda p: PRODUCTS[p][0])
    df["units"] = rng.integers(1, 6, size=len(df))
    df["unit_price"] = df["product"].map(lambda p: PRODUCTS[p][1]) * rng.uniform(0.95, 1.05, size=len(df))
    # PLANTED: Online orders get bigger discounts (~15%) than Retail (~5%) / Partner (~10%).
    base_discount = df["channel"].map({"Online": 0.15, "Retail": 0.05, "Partner": 0.10})
    df["discount_pct"] = (base_discount + rng.normal(0, 0.02, size=len(df))).clip(0, 0.4).round(3)
    df["revenue"] = (df["units"] * df["unit_price"] * (1 - df["discount_pct"])).round(2)
    df["unit_price"] = df["unit_price"].round(2)
    df["customer_id"] = [f"C{x:05d}" for x in rng.integers(1, 1500, size=len(df))]

    df = df.sort_values("order_date").reset_index(drop=True)
    df.insert(0, "order_id", [f"ORD-{i:05d}" for i in range(1, len(df) + 1)])

    # ---------------- MESSY DATA ----------------
    # Outliers: 6 data-entry errors with 999 units (revenue explodes accordingly).
    out_idx = rng.choice(len(df), size=6, replace=False)
    df.loc[out_idx, "units"] = 999
    df.loc[out_idx, "revenue"] = (df.loc[out_idx, "units"] * df.loc[out_idx, "unit_price"]).round(2)

    # Missing values
    df.loc[pick_rows(rng, len(df), 0.015), "region"] = np.nan
    df.loc[pick_rows(rng, len(df), 0.02), "discount_pct"] = np.nan

    # Inconsistent text: ' west', 'WEST', ...
    df["region"] = mess_up_text(rng, df["region"], 0.05)
    df["channel"] = mess_up_text(rng, df["channel"], 0.03)

    # Wrong types: ~10% of revenue values stored as strings like "$1,234.50"
    revenue = df["revenue"].astype(object)
    for i in pick_rows(rng, len(df), 0.10):
        revenue.iat[i] = f"${revenue.iat[i]:,.2f}"
    df["revenue"] = revenue

    # Mixed date formats: ~5% as "15 Mar 2024"
    dates_txt = df["order_date"].dt.strftime("%Y-%m-%d").astype(object)
    for i in pick_rows(rng, len(df), 0.05):
        dates_txt.iat[i] = df["order_date"].iat[i].strftime("%d %b %Y")
    df["order_date"] = dates_txt

    # 25 exact duplicate rows
    dupes = df.iloc[rng.choice(len(df), size=25, replace=False)]
    df = pd.concat([df, dupes]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. WEB TRAFFIC
# ---------------------------------------------------------------------------
CHANNELS = {  # channel: (base daily sessions, conversion rate, bounce rate)
    "Organic": (2200, 0.025, 0.42),
    "Paid": (1500, 0.030, 0.55),
    "Social": (1300, 0.012, 0.63),
    "Email": (600, 0.065, 0.30),
    "Referral": (500, 0.022, 0.40),
}


def make_web_traffic(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows = []
    for day in days:
        for channel, (base, conv, bounce) in CHANNELS.items():
            sessions = base
            if day.day_of_week >= 5:            # PLANTED: weekends ~30% lower
                sessions *= 0.70
            if channel == "Organic":         # PLANTED: Organic grows ~60% over the year
                sessions *= 1 + 0.6 * (day.day_of_year / 366)
            if channel == "Paid" and pd.Timestamp("2024-11-20") <= day <= pd.Timestamp("2024-12-02"):
                sessions *= 3.0              # PLANTED: Black-Friday paid campaign (3x)
            sessions = max(0, int(sessions * rng.normal(1, 0.08)))
            if pd.Timestamp("2024-06-10") <= day <= pd.Timestamp("2024-06-12"):
                sessions = 0                 # PLANTED: tracking outage (3 days of zeros)
            conversions = int(rng.binomial(sessions, conv)) if sessions else 0
            ad_spend = round(sessions * rng.uniform(0.9, 1.1) * 0.8, 2) if channel == "Paid" else 0.0
            rows.append({
                "date": day.strftime("%Y-%m-%d"), "channel": channel, "sessions": sessions,
                "users": int(sessions * rng.uniform(0.75, 0.85)),
                "bounce_rate": round(float(np.clip(rng.normal(bounce, 0.03), 0, 1)) * 100, 1),
                "avg_session_sec": round(float(rng.normal(180 if channel != "Social" else 95, 15)), 1),
                "conversions": conversions, "ad_spend": ad_spend,
            })
    df = pd.DataFrame(rows)

    # ---------------- MESSY DATA ----------------
    # bounce_rate stored as text like "42.1%"; some missing
    df["bounce_rate"] = df["bounce_rate"].map(lambda x: f"{x}%").astype(object)
    df.loc[pick_rows(rng, len(df), 0.03), "bounce_rate"] = np.nan
    # A few missing session counts (not on outage days)
    not_outage = ~df["date"].between("2024-06-10", "2024-06-12")
    miss = rng.choice(df.index[not_outage], size=20, replace=False)
    df.loc[miss, "sessions"] = np.nan
    # Channel name variants
    df["channel"] = mess_up_text(rng, df["channel"], 0.04)
    # Spike outliers in avg_session_sec (bot traffic)
    df.loc[rng.choice(len(df), size=8, replace=False), "avg_session_sec"] = 9999.0
    return df


# ---------------------------------------------------------------------------
# 3. HR EMPLOYEES (Excel)
# ---------------------------------------------------------------------------
DEPARTMENTS = {  # dept: (share of staff, base salary, attrition probability)
    "Engineering": (0.30, 105000, 0.08),
    "Sales": (0.22, 70000, 0.30),
    "Marketing": (0.13, 75000, 0.15),
    "Support": (0.17, 52000, 0.18),
    "Finance": (0.10, 85000, 0.10),
    "HR": (0.08, 65000, 0.12),
}


def make_hr(rng: np.random.Generator) -> pd.DataFrame:
    n = 1200
    depts = rng.choice(list(DEPARTMENTS), size=n, p=[v[0] for v in DEPARTMENTS.values()])
    level = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.3, 0.3, 0.2, 0.13, 0.07])
    overtime = rng.random(n) < 0.28
    satisfaction = np.clip(rng.normal(6.5, 1.8, size=n) - overtime * 1.0, 1, 10).round().astype(int)
    hire = pd.Timestamp("2024-12-31") - pd.to_timedelta(rng.integers(30, 365 * 12, size=n), unit="D")

    base_salary = np.array([DEPARTMENTS[d][1] for d in depts])
    salary = base_salary * (1 + 0.22 * (level - 1)) * rng.normal(1, 0.08, size=n)

    # Attrition: department base rate, x2 with overtime, higher when satisfaction is low.
    p_leave = np.array([DEPARTMENTS[d][2] for d in depts])
    p_leave = p_leave * np.where(overtime, 2.0, 0.8) * np.where(satisfaction <= 4, 1.8, 0.85)
    attrition = rng.random(n) < np.clip(p_leave, 0, 0.95)

    df = pd.DataFrame({
        "employee_id": [f"E{i:04d}" for i in range(1, n + 1)],
        "department": depts,
        "job_level": level,
        "hire_date": hire.strftime("%Y-%m-%d"),
        "age": np.clip(rng.normal(36, 9, size=n), 21, 64).round().astype(int),
        "salary": salary.round(-2),
        "performance_rating": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.45, 0.25, 0.10]).astype(float),
        "overtime": np.where(overtime, "Yes", "No").astype(object),
        "satisfaction": satisfaction,
        "remote": rng.random(n) < 0.35,
        "attrition": np.where(attrition, "Yes", "No"),
    })

    # ---------------- MESSY DATA ----------------
    # Salary typos: 8 salaries accidentally x10
    df.loc[rng.choice(n, size=8, replace=False), "salary"] *= 10
    # Salary stored as text "85,000" for ~6% of rows
    salary_obj = df["salary"].astype(object)
    for i in pick_rows(rng, n, 0.06):
        salary_obj.iat[i] = f"{salary_obj.iat[i]:,.0f}"
    df["salary"] = salary_obj
    # Missing performance ratings (8%)
    df.loc[pick_rows(rng, n, 0.08), "performance_rating"] = np.nan
    # Overtime written inconsistently: 'Y' / 'N' / 'yes'
    for i in pick_rows(rng, n, 0.10):
        df.at[i, "overtime"] = {"Yes": rng.choice(["Y", "yes"]), "No": rng.choice(["N", "no"])}[df.at[i, "overtime"]]
    # Department casing / whitespace issues
    df["department"] = mess_up_text(rng, df["department"], 0.04)
    # Mixed hire_date formats (~7% as "03/15/2019")
    for i in pick_rows(rng, n, 0.07):
        df.at[i, "hire_date"] = pd.Timestamp(df.at[i, "hire_date"]).strftime("%m/%d/%Y")
    # Impossible ages
    df.loc[rng.choice(n, size=3, replace=False), "age"] = 0
    return df


# ---------------------------------------------------------------------------
# 4. SUPPORT TICKETS
# ---------------------------------------------------------------------------
def make_tickets(rng: np.random.Generator) -> pd.DataFrame:
    n = 2500
    created = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.uniform(0, 366 * 24, size=n), unit="h")
    products = rng.choice(["Laptop Pro", "Laptop Air", "Phone X", "Phone Mini", "Headphones", "Smartwatch"],
                          size=n, p=[0.17, 0.15, 0.2, 0.15, 0.15, 0.18])
    df = pd.DataFrame({"created_at": created, "product": products})

    # PLANTED: Smartwatch quality problem in September -> 120 extra Smartwatch tickets.
    extra = pd.DataFrame({
        "created_at": pd.to_datetime("2024-09-01") + pd.to_timedelta(rng.uniform(0, 30 * 24, size=120), unit="h"),
        "product": "Smartwatch",
    })
    df = pd.concat([df, extra], ignore_index=True)
    m = len(df)

    df["priority"] = rng.choice(["Low", "Medium", "High", "Urgent"], size=m, p=[0.35, 0.4, 0.18, 0.07])
    df["channel"] = rng.choice(["Email", "Chat", "Phone"], size=m, p=[0.45, 0.35, 0.2])
    df["agent_id"] = rng.choice([f"A{i:02d}" for i in range(1, 13)], size=m)

    # Resolution hours: depends on priority; agent A07 is ~2.2x slower (PLANTED).
    prio_hours = df["priority"].map({"Low": 30, "Medium": 20, "High": 10, "Urgent": 4})
    hours = prio_hours * rng.lognormal(0, 0.45, size=m)
    hours = np.where(df["agent_id"] == "A07", hours * 2.2, hours)
    # PLANTED: backlog in the week of 5-11 August 2024 -> resolution ~2.5x slower.
    backlog = df["created_at"].between("2024-08-05", "2024-08-11 23:59:59")
    hours = np.where(backlog, hours * 2.5, hours)
    df["resolution_hours"] = np.round(hours, 1)

    # Satisfaction (1-5): lower when resolution takes longer; Chat is best (PLANTED).
    sat = 4.6 - 0.035 * df["resolution_hours"] + df["channel"].map({"Chat": 0.35, "Email": -0.1, "Phone": 0.0})
    df["satisfaction_score"] = np.clip(np.round(sat + rng.normal(0, 0.5, size=m)), 1, 5)

    df["status"] = np.where(rng.random(m) < 0.06, "Open", "Closed")
    df = df.sort_values("created_at").reset_index(drop=True)
    df.insert(0, "ticket_id", [f"T{i:05d}" for i in range(1, m + 1)])
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")

    # ---------------- MESSY DATA ----------------
    df.loc[df["status"] == "Open", "resolution_hours"] = np.nan   # open tickets not resolved yet
    df.loc[pick_rows(rng, m, 0.20), "satisfaction_score"] = np.nan  # many customers skip the survey
    neg = rng.choice(df.index[df["status"] == "Closed"], size=6, replace=False)
    df.loc[neg, "resolution_hours"] = -df.loc[neg, "resolution_hours"]   # impossible negative times
    big = rng.choice(df.index[df["status"] == "Closed"], size=4, replace=False)
    df.loc[big, "resolution_hours"] = 2000.0                             # absurd outliers
    df["priority"] = mess_up_text(rng, df["priority"], 0.04)
    return df


# ---------------------------------------------------------------------------
PLANTED_PATTERNS = {
    "sales.csv": [
        "West revenue drops sharply in Q3 (Jul-Sep): 'Laptop Pro' was ~out of stock there and "
        "Laptop Air / Phone X demand fell ~40%.",
        "East has the highest total revenue.",
        "November and December have more orders (holiday season).",
        "Online orders have the highest average discount (~15%); Retail the lowest (~5%).",
        "MESSY: region/channel casing+whitespace, ~10% revenue as '$1,234.50' strings, mixed date "
        "formats, missing region/discount, 6 outlier orders with 999 units, 25 duplicate rows.",
    ],
    "web_traffic.csv": [
        "Weekend sessions are ~30% lower than weekdays.",
        "Organic sessions grow ~60% from January to December.",
        "Paid sessions triple during a campaign from 2024-11-20 to 2024-12-02.",
        "Tracking outage: all sessions are 0 on 2024-06-10..2024-06-12.",
        "Email has by far the highest conversion rate (~6.5%); Social the lowest.",
        "MESSY: bounce_rate as '42.1%' strings with gaps, 20 missing session values, channel casing, "
        "avg_session_sec outliers of 9999.",
    ],
    "hr_employees.xlsx": [
        "Sales has the highest attrition rate; Engineering the lowest.",
        "Employees who work overtime leave ~2.5x as often.",
        "Low satisfaction (<=4) is associated with higher attrition.",
        "Engineering has the highest median salary; salary rises with job_level.",
        "MESSY: 8 salaries x10 typos, ~6% salaries as '85,000' text, missing performance ratings, "
        "overtime as Yes/No/Y/N/yes/no, department casing, mixed hire_date formats, 3 ages of 0.",
    ],
    "support_tickets.csv": [
        "Agent A07 has a ~2x longer median resolution time than other agents.",
        "Week of 2024-08-05..11 has a resolution-time spike (backlog).",
        "Smartwatch tickets spike in September 2024 (quality issue).",
        "Chat has the highest satisfaction; satisfaction falls as resolution time rises.",
        "MESSY: open tickets have no resolution time, 20% missing satisfaction, 6 negative "
        "resolution times, 4 outliers of 2000 hours, priority casing.",
    ],
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    make_sales(rng).to_csv(DATA_DIR / "sales.csv", index=False)
    make_web_traffic(rng).to_csv(DATA_DIR / "web_traffic.csv", index=False)
    make_hr(rng).to_excel(DATA_DIR / "hr_employees.xlsx", index=False)
    make_tickets(rng).to_csv(DATA_DIR / "support_tickets.csv", index=False)

    # Document the planted patterns next to the data.
    lines = ["# Synthetic datasets\n", "Generated by `generate_datasets.py` (seed 42). "
             "Do not edit by hand - re-run the script instead.\n"]
    for name, patterns in PLANTED_PATTERNS.items():
        lines.append(f"\n## {name}\n")
        lines.extend(f"- {p}" for p in patterns)
    (DATA_DIR / "README.md").write_text("\n".join(lines) + "\n")
    print("Datasets written to", DATA_DIR)


if __name__ == "__main__":
    main()
