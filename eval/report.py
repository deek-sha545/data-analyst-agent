"""
Summarise results/runs.jsonl into tables and charts:

    results/summary.md      human-readable tables (accuracy, cost, latency, per dataset, failures)
    results/summary.csv     one row per (model, variant)
    results/comparison_<suite>.png  accuracy / tokens / latency per configuration (small multiples)

    python -m eval.report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyst.config import ROOT

RESULTS = ROOT / "results"
VARIANT_ORDER = ["multi", "no_reviewer", "single"]
VARIANT_LABEL = {"multi": "Multi-agent + reviewer", "no_reviewer": "Multi-agent, no reviewer", "single": "Single agent"}
# Categorical slots 1-3 of the validated reference palette (light surface), fixed order per variant.
VARIANT_COLOR = {"multi": "#2a78d6", "no_reviewer": "#eb6834", "single": "#1baf7a"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"


def load(path: Path = RESULTS / "runs.jsonl") -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["suite"] = df["suite"].fillna("hinted") if "suite" in df else "hinted"
    # Latest attempt per combination; infra errors only count if nothing else exists.
    df["is_error"] = df["status"] == "error"
    df = df.sort_values(["is_error", "ts"], ascending=[False, True])
    df = df.drop_duplicates(["suite", "model", "variant", "qid"], keep="last").reset_index(drop=True)
    df["review_revisions"] = df["run_id"].map(_count_revisions)
    return df


def _count_revisions(run_id) -> int:
    """'revise' verdicts in the run's event log (eval/runs/logs/<run_id>.json); 0 if the log is missing."""
    log = ROOT / "eval" / "runs" / "logs" / f"{run_id}.json"
    if not isinstance(run_id, str) or not log.exists():
        return 0
    events = json.loads(log.read_text())["events"]
    return sum(e["kind"] == "review" and e["content"].get("verdict") == "revise" for e in events)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    # Runs that died on an API/quota error were never graded: they are reported as infra_errors
    # and excluded from every other column, so they cannot masquerade as wrong answers.
    graded = df[~df["is_error"]]
    g = graded.groupby(["suite", "model", "variant"])
    out = pd.DataFrame({
        "graded": g.size(),
        "accuracy_pct": g["correct"].mean() * 100,
        "completed_pct": g["status"].apply(lambda s: (s == "completed").mean() * 100),
        "avg_executions": g["executions"].mean(),
        "runs_with_self_correction": g["failed_executions"].apply(lambda s: (s > 0).sum()),
        "runs_revised_by_reviewer": g["review_revisions"].apply(lambda s: (s > 0).sum()),
        "avg_llm_calls": g["llm_calls"].mean(),
        "avg_tokens": g["total_tokens"].mean(),
        "avg_cost_usd": g["cost_usd"].mean(),
        "total_cost_usd": g["cost_usd"].sum(),
        "avg_llm_latency_s": g["llm_latency_s"].mean(),
        "avg_retries": g["retries"].mean(),
    })
    all_configs = df.groupby(["suite", "model", "variant"]).size().index
    out = out.reindex(all_configs)  # keep configurations whose runs all hit API errors
    out["graded"] = out["graded"].fillna(0).astype(int)
    out["infra_errors"] = df[df["is_error"]].groupby(["suite", "model", "variant"]).size()
    out["infra_errors"] = out["infra_errors"].fillna(0).astype(int)
    for col in ("runs_with_self_correction", "runs_revised_by_reviewer"):
        out[col] = out[col].astype("Int64")
    out = out.reset_index()
    out["variant"] = pd.Categorical(out["variant"], VARIANT_ORDER, ordered=True)
    return out.sort_values(["suite", "model", "variant"]).reset_index(drop=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart(summary: pd.DataFrame, path: Path, title: str) -> None:
    rows = summary[summary["graded"] > 0].copy()  # all-error configurations are listed in the table only
    rows["label"] = (rows["model"].str.replace("gemini-", "") + "\n" + rows["variant"].map(VARIANT_LABEL).astype(str)
                     + "  (n=" + rows["graded"].astype(str) + ")")
    rows = rows.iloc[::-1]  # first configuration on top
    panels = [("accuracy_pct", "Accuracy (%)", "{:.0f}%"),
              ("avg_tokens", "Tokens per question", "{:,.0f}"),
              ("avg_llm_latency_s", "Model latency per question (s)", "{:.1f}s")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.75 * len(rows) + 1.6), sharey=True, facecolor=SURFACE)
    for ax, (col, panel_title, fmt) in zip(axes, panels):
        _style(ax)
        colors = [VARIANT_COLOR[str(v)] for v in rows["variant"]]
        bars = ax.barh(rows["label"], rows[col], color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        vmax = rows[col].max()
        vmax = 1 if pd.isna(vmax) or vmax == 0 else vmax
        for b, v in zip(bars, rows[col]):
            text = "no graded runs" if pd.isna(v) else fmt.format(v)
            width = 0 if pd.isna(v) else b.get_width()
            ax.text(width + vmax * 0.02, b.get_y() + b.get_height() / 2, text,
                    va="center", ha="left", color=INK if not pd.isna(v) else INK_MUTED, fontsize=9)
        ax.set_xlim(0, (100 if col == "accuracy_pct" else vmax) * 1.18)
        ax.set_title(panel_title, loc="left", color=INK, fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="y", colors=INK, labelsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


SUITE_TEXT = {
    "hinted": "Questions as written: cleaning rules that change the answer are stated in the question.",
    "unhinted": "The same questions with the data-error hints removed (999-unit orders, x10 salary typos, invalid "
                "resolution times, casing notes): the agents must find the errors themselves. Only the 16 "
                "questions that carry such hints are in this suite.",
}


def main() -> None:
    df = load()
    summary = summarise(df)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    cols = ["model", "variant", "graded", "infra_errors", "accuracy_pct", "completed_pct", "avg_executions",
            "runs_with_self_correction", "runs_revised_by_reviewer", "avg_llm_calls", "avg_tokens",
            "avg_cost_usd", "avg_llm_latency_s"]
    md = [
        "# Experiment results",
        "",
        f"{int((~df['is_error']).sum())} graded runs over {df['qid'].nunique()} questions "
        f"({', '.join(sorted(df['model'].unique()))}). Generated by `python -m eval.report` from `results/runs.jsonl`.",
        "",
        "- **graded** - runs that finished and were graded; **infra_errors** - runs killed by API errors / the "
        "daily free-tier quota (not graded, excluded from all other columns)",
        "- **accuracy** - the one-line answer matched the reference answer (deterministic grader, `eval/grader.py`)",
        "- **runs_with_self_correction** - runs where at least one script failed and the agent fixed it",
        "- **runs_revised_by_reviewer** - runs where the reviewer asked for at least one revision",
        "- **avg_llm_latency_s** - summed model response time per question; time queued by the client-side "
        "rate limiter (free tier: 15 requests/min) is excluded",
        "- **avg_cost_usd** - estimate from the `analyst/config.py` price table (the free tier was actually used)",
        "",
    ]
    for suite in [s for s in ("hinted", "unhinted") if s in set(summary["suite"])]:
        part = summary[summary["suite"] == suite]
        sub = df[(df["suite"] == suite) & ~df["is_error"]]
        png = f"comparison_{suite}.png"
        n_q = df[df["suite"] == suite]["qid"].nunique()
        chart(part, RESULTS / png, f"{suite.capitalize()} suite - {n_q} questions (n = graded runs per bar)")
        by_dataset = (sub.groupby(["model", "variant", "dataset"])["correct"].mean().mul(100).round(0)
                      .unstack("dataset").reset_index())
        failures = sub[~sub["correct"]][["model", "variant", "qid", "status", "answer", "grade_reason"]].copy()
        failures["answer"] = failures["answer"].fillna("").str.slice(0, 140)
        md += [
            f"## {suite.capitalize()} suite",
            "",
            SUITE_TEXT[suite],
            "",
            _md_table(part[cols]),
            "",
            f"![{suite} comparison]({png})",
            "",
            "Accuracy by dataset (%):",
            "",
            _md_table(by_dataset, ".0f"),
            "",
            "Incorrect answers:",
            "",
            _md_table(failures) if len(failures) else "None.",
            "",
        ]
    (RESULTS / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
