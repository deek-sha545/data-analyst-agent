"""
Experiment harness: run every (model x variant x question) combination and
append one JSON line per run to results/runs.jsonl.

Resumable: combinations already in the file are skipped (runs that ended in an
infrastructure error, e.g. an API outage, are retried). Then build the summary
tables and charts with `python -m eval.report`.

    python -m eval.harness --models gemini-3.5-flash-lite --variants multi single no_reviewer
    python -m eval.harness --questions sales_q2q3_drop hr_top_attrition --workers 1
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from analyst.config import ROOT, get_settings
from analyst.graph import VARIANTS
from analyst.runner import run_analysis
from eval.grader import grade
from eval.questions import BY_ID, QUESTIONS, Question
from eval.reference import DATA_DIR

RESULTS = ROOT / "results" / "runs.jsonl"
RUNS_DIR = ROOT / "eval" / "runs"   # per-run reports/logs (gitignored)


def load_done(path: Path) -> set[tuple[str, str, str]]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["status"] != "error":
                done.add((r["model"], r["variant"], r["qid"]))
    return done


def run_one(q: Question, model: str, variant: str, provider: str) -> dict:
    settings = get_settings(provider, model, outputs_dir=RUNS_DIR / "outputs", logs_dir=RUNS_DIR / "logs")
    t0 = time.time()
    res = run_analysis(q.question, [DATA_DIR / q.file], variant, settings=settings)
    g = grade(q, res.answer) if res.status != "error" else None
    m = res.metrics
    return {
        "qid": q.id,
        "dataset": q.file,
        "model": model,
        "variant": variant,
        "status": res.status,
        "correct": bool(g and g.correct),
        "category_ok": g.category_ok if g else None,
        "number_ok": g.number_ok if g else None,
        "grade_reason": g.reason if g else res.error,
        "answer": res.answer,
        "expected": q.answer,
        "executions": len(res.attempts),
        "failed_executions": sum(not a["ok"] for a in res.attempts),
        "review_verdict": (res.review or {}).get("verdict"),
        "review_rounds": res.review_rounds,
        "llm_calls": m["llm_calls"],
        "prompt_tokens": m["prompt_tokens"],
        "completion_tokens": m["completion_tokens"],
        "total_tokens": m["total_tokens"],
        "cost_usd": m["cost_usd"],
        "retries": m["retries"],
        "llm_latency_s": m["llm_latency_s"],
        "wall_time_s": round(time.time() - t0, 2),
        "run_id": res.run_id,
        "error": res.error,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default="gemini")
    p.add_argument("--models", nargs="+", default=[get_settings().model])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--questions", nargs="+", default=["all"], help="question ids, or 'all'")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args(argv)

    questions = QUESTIONS if args.questions == ["all"] else [BY_ID[q] for q in args.questions]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    jobs = [(q, m, v) for m in args.models for v in args.variants for q in questions if (m, v, q.id) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already in {args.out.name})", flush=True)

    lock = threading.Lock()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, q, m, v, args.provider): (q, m, v) for q, m, v in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            q, m, v = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # bug in the harness itself: record and move on
                row = {"qid": q.id, "model": m, "variant": v, "status": "error", "correct": False,
                       "error": f"{type(e).__name__}: {e}"}
            with lock:
                with args.out.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            n_ok += row["correct"]
            mark = "OK " if row["correct"] else ("ERR" if row["status"] == "error" else "XX ")
            print(f"[{i}/{len(jobs)}] {mark} {m:<24} {v:<12} {q.id:<30} "
                  f"{row.get('wall_time_s', 0):>5}s  {row.get('grade_reason', '')[:90]}", flush=True)
    print(f"done: {n_ok}/{len(jobs)} correct in this batch", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
