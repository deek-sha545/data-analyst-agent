"""
Deterministic dataset profile given to the agents before they plan or code.

LLMs write much better pandas when they can see the real column names, dtypes,
example values and - crucially - the *messiness* (numbers stored as text,
inconsistent casing, duplicates, missing values). This module computes that
cheaply without any LLM call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED = {".csv", ".xlsx", ".xls", ".tsv", ".json", ".parquet"}


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")


def _looks_numeric_text(s: pd.Series) -> float:
    """Share of non-null text values that parse as numbers after stripping $ , % and spaces."""
    vals = s.dropna().astype(str)
    if vals.empty:
        return 0.0
    cleaned = vals.str.replace(r"[$,%\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").notna().mean()


def _looks_date_text(s: pd.Series) -> bool:
    vals = s.dropna().astype(str).head(200)
    if vals.empty or vals.str.len().mean() < 6:
        return False
    parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def profile_dataframe(df: pd.DataFrame, name: str) -> str:
    lines = [f"### `{name}` - {len(df):,} rows x {df.shape[1]} columns"]
    dup = int(df.duplicated().sum())
    if dup:
        lines.append(f"- WARNING: {dup} fully duplicated rows")
    lines.append("")
    lines.append("| column | dtype | missing | unique | examples / stats | notes |")
    lines.append("|---|---|---|---|---|---|")
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        notes = []
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            stats = f"min {desc['min']:.4g}, median {s.median():.4g}, max {desc['max']:.4g}"
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out = s[(s < q1 - 5 * iqr) | (s > q3 + 5 * iqr)]
                if len(out):
                    vals = ", ".join(f"{v:.4g}" for v in out.value_counts().index[:4])
                    notes.append(
                        f"{len(out)} extreme outliers (beyond 5xIQR; values {vals}) - "
                        "likely data-entry errors, check whether they distort sums/means"
                    )
            if (s < 0).any() and (s >= 0).mean() > 0.95:
                notes.append(f"{int((s < 0).sum())} negative values (likely invalid)")
            if (s == 0).any() and (s > 0).mean() > 0.97 and s.min() == 0 and s[s > 0].min() >= 10:
                notes.append(f"{int((s == 0).sum())} zero values while all others are >= {s[s > 0].min():.4g} (likely missing/invalid)")
            examples = stats
        else:
            top = s.dropna().astype(str).value_counts().head(6)
            examples = ", ".join(f"{v!r}" for v in top.index[:6])
            if len(examples) > 120:
                examples = examples[:117] + "..."
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                num_share = _looks_numeric_text(s)
                if num_share > 0.9:
                    notes.append("numbers stored as text (strip $ , % before converting)")
                elif _looks_date_text(s):
                    notes.append("dates stored as text (mixed formats possible)")
                normalized = s.dropna().astype(str).str.strip().str.lower()
                if 0 < normalized.nunique() < nunique:
                    notes.append(
                        f"inconsistent casing/whitespace: {nunique} raw values -> "
                        f"{normalized.nunique()} after strip+lower"
                    )
        lines.append(
            f"| {col} | {s.dtype} | {missing} | {nunique} | {examples} | {'; '.join(notes)} |"
        )
    lines.append("")
    lines.append("First 5 rows:")
    lines.append("```")
    lines.append(df.head(5).to_string(max_colwidth=30))
    lines.append("```")
    return "\n".join(lines)


def profile_files(paths: list[str | Path]) -> str:
    parts = []
    for p in paths:
        p = Path(p)
        try:
            parts.append(profile_dataframe(load_table(p), p.name))
        except Exception as e:  # unreadable file: tell the agents rather than crash
            parts.append(f"### `{p.name}`\nCould not be read for profiling: {type(e).__name__}: {e}")
    return "\n\n".join(parts)
