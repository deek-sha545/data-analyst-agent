"""
MCP server exposing the analyst to any MCP client (Claude Desktop, Claude Code, IDEs...).

Tools
  list_datasets()                          datasets the server may read
  profile_dataset(dataset)                 columns, dtypes, missing values, messiness notes
  run_python(code, datasets)               run pandas/matplotlib code in the sandbox
  ask_analyst(question, datasets, variant) full multi-agent analysis -> answer, report, charts, metrics

Datasets are referenced by file name from data/ or uploads/. Set
ANALYST_MCP_ALLOW_ANY_PATH=1 to also allow absolute paths to local files.

Run over stdio (what desktop clients launch):
    python -m analyst_mcp.server
or over HTTP:
    python -m analyst_mcp.server --http --port 8001
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import anyio
from mcp.server.mcpserver import MCPServer

from analyst.config import ROOT, get_settings
from analyst.graph import VARIANTS
from analyst.profiling import SUPPORTED, profile_files
from analyst.runner import run_analysis
from analyst.sandbox import run_code

DATA_DIRS = [ROOT / "data", ROOT / "uploads"]

server = MCPServer(
    name="data-analyst",
    instructions=(
        "Autonomous data analyst. Call list_datasets first, profile_dataset to see columns and data-quality "
        "issues, then ask_analyst for a complete answer with a report and charts (it plans, writes and runs "
        "code, self-corrects and reviews its work). Use run_python for quick custom pandas checks."
    ),
)


def resolve(name: str) -> Path:
    """Map a dataset reference to a file, refusing anything outside the allowed folders."""
    p = Path(name)
    if p.is_absolute():
        if os.getenv("ANALYST_MCP_ALLOW_ANY_PATH") != "1":
            raise ValueError("Absolute paths are disabled; set ANALYST_MCP_ALLOW_ANY_PATH=1 to allow them.")
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            raise ValueError(f"{name}: not a readable {sorted(SUPPORTED)} file")
        return p
    if p.name != name or name.startswith("."):
        raise ValueError(f"Invalid dataset name {name!r} (use a bare file name from list_datasets)")
    for folder in DATA_DIRS:
        cand = folder / name
        if cand.is_file() and cand.suffix.lower() in SUPPORTED:
            return cand
    raise ValueError(f"Dataset {name!r} not found; call list_datasets to see what is available.")


@server.tool()
def list_datasets() -> list[dict]:
    """List the datasets available for analysis (file name, folder, size in bytes)."""
    out = []
    for folder in DATA_DIRS:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED:
                    out.append({"name": p.name, "folder": folder.name, "bytes": p.stat().st_size})
    return out


@server.tool()
def profile_dataset(dataset: str) -> str:
    """Profile a dataset: rows, columns, dtypes, missing values, example values and data-quality notes
    (duplicates, numbers stored as text, inconsistent casing, outliers). Returns Markdown."""
    return profile_files([resolve(dataset)])


@server.tool()
async def run_python(code: str, datasets: list[str] | None = None) -> dict:
    """Run a Python script in the sandbox. The given datasets are copied into the working directory,
    so open them by bare file name (pd.read_csv("sales.csv")). Allowed imports: pandas, numpy, matplotlib
    and a few stdlib modules; no file system or network access. Figures left open are saved as PNGs.
    Returns stdout, error and the chart file paths."""
    files = [resolve(d) for d in datasets or []]
    charts_dir = Path(tempfile.mkdtemp(prefix="analyst_mcp_charts_"))
    timeout = get_settings().sandbox_timeout
    r = await anyio.to_thread.run_sync(lambda: run_code(code, files, timeout=timeout, figures_dir=charts_dir))
    return {
        "ok": r.ok,
        "stdout": r.stdout,
        "error": (f"{r.error}: " + "; ".join(r.violations)) if r.violations else r.error,
        "stderr": r.stderr if not r.ok else "",
        "charts": [str(p) for p in r.figures],
        "duration_s": r.duration_s,
    }


@server.tool()
async def ask_analyst(question: str, datasets: list[str], variant: str = "multi") -> dict:
    """Answer a business question about the datasets with the autonomous multi-agent analyst.
    It plans, writes and runs pandas code, fixes its own errors, reviews the result and writes a
    Markdown report with charts. variant: "multi" (default, with reviewer), "no_reviewer" or "single".
    Takes 10-60 seconds. Returns the answer, full report, chart paths, final code and run metrics."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    files = [resolve(d) for d in datasets]
    res = await anyio.to_thread.run_sync(lambda: run_analysis(question, files, variant))
    if res.status == "error":
        raise RuntimeError(f"Analysis failed: {res.error}")
    return {
        "status": res.status,
        "answer": res.answer,
        "report": res.report,
        "report_path": res.report_path,
        "charts": res.charts,
        "code": res.code,
        "metrics": {k: res.metrics[k] for k in ("llm_calls", "total_tokens", "cost_usd", "retries", "wall_time_s")},
        "run_id": res.run_id,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MCP server for the autonomous data analyst")
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    if args.http:
        server.run("streamable-http", port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
