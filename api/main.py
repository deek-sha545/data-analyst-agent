"""
FastAPI backend.

    uvicorn api.main:app --reload

Endpoints
  GET  /api/health                      provider/model + whether an API key is configured
  GET  /api/datasets                    sample + uploaded datasets
  POST /api/datasets                    upload a CSV / Excel / JSON / Parquet file (returns its profile)
  POST /api/runs                        start an analysis -> {run_id}
  GET  /api/runs/{id}                   status, pending plan (if awaiting approval), result, metrics
  GET  /api/runs/{id}/events?since=N    agent step events as JSON (for polling clients)
  GET  /api/runs/{id}/stream            the same events live, as NDJSON (one JSON object per line)
  POST /api/runs/{id}/approve           {"approved": bool, "feedback": str} - human-in-the-loop decision
  GET  /api/runs/{id}/files/{name}      report.md, analysis.py or chart_N.png of a finished run

Runs execute in background threads; state lives in memory (single-process
deployment - see DECISIONS.md).
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from analyst.config import ROOT, Settings, get_settings
from analyst.graph import VARIANTS
from analyst.llm import LLM
from analyst.profiling import SUPPORTED, profile_files
from analyst.runner import RunResult, run_analysis
from analyst.tracker import Event, RunTracker

SAMPLES_DIR = ROOT / "data"
UPLOADS_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
APPROVAL_TIMEOUT_S = 30 * 60

app = FastAPI(title="Autonomous Data Analyst", version="1.0")

# Swappable in tests: (settings, tracker) -> LLM-like object, and extra Settings fields.
app.state.llm_factory = lambda settings, tracker: LLM(settings, tracker)
app.state.settings_overrides = {}


def _settings(model: str | None = None) -> Settings:
    return get_settings(model=model, **app.state.settings_overrides)


# ------------------------------------------------------------------ jobs
@dataclass
class Job:
    run_id: str
    question: str
    variant: str
    datasets: list[str]
    tracker: RunTracker
    status: str = "running"            # running | awaiting_approval | completed | failed | rejected | error
    pending_plan: list[str] | None = None
    result: RunResult | None = None
    created: float = field(default_factory=time.time)
    _decision: dict | None = None
    _approval: threading.Event = field(default_factory=threading.Event)
    _changed: threading.Condition = field(default_factory=threading.Condition)

    def notify(self, _event: Event | None = None) -> None:
        with self._changed:
            self._changed.notify_all()

    def wait_for_approval(self, pending: dict) -> dict:
        self.pending_plan = pending["plan"]
        self.status = "awaiting_approval"
        self._approval.clear()
        self.notify()
        if not self._approval.wait(APPROVAL_TIMEOUT_S):
            decision = {"approved": False, "feedback": ""}
            self.tracker.event("system", "approval_timeout", f"no decision within {APPROVAL_TIMEOUT_S}s")
        else:
            decision = self._decision or {"approved": True}
        self.pending_plan = None
        self.status = "running"
        return decision

    def decide(self, decision: dict) -> None:
        self._decision = decision
        self._approval.set()


JOBS: dict[str, Job] = {}


# --------------------------------------------------------------- datasets
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _list_datasets() -> list[dict]:
    out = []
    for kind, folder in (("sample", SAMPLES_DIR), ("upload", UPLOADS_DIR)):
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED:
                    out.append({"id": f"{kind}:{p.name}", "name": p.name, "kind": kind, "bytes": p.stat().st_size})
    return out


def resolve_dataset(dataset_id: str) -> Path:
    kind, _, name = dataset_id.partition(":")
    folder = {"sample": SAMPLES_DIR, "upload": UPLOADS_DIR}.get(kind)
    if folder is None or not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, f"Invalid dataset id {dataset_id!r}")
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        raise HTTPException(404, f"Dataset {dataset_id!r} not found")
    return path


# ---------------------------------------------------------------- schemas
class RunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    datasets: list[str] = Field(min_length=1, max_length=5)
    variant: Literal["multi", "no_reviewer", "single"] = "multi"
    require_approval: bool = False
    model: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool
    feedback: str = Field(default="", max_length=2000)


def _job(run_id: str) -> Job:
    job = JOBS.get(run_id)
    if job is None:
        raise HTTPException(404, f"Unknown run {run_id!r}")
    return job


def _job_view(job: Job) -> dict:
    view = {
        "run_id": job.run_id,
        "question": job.question,
        "variant": job.variant,
        "datasets": job.datasets,
        "status": job.status,
        "pending_plan": job.pending_plan,
        "n_events": len(job.tracker.events),
        "metrics": job.tracker.summary(),
        "result": None,
    }
    if job.result is not None:
        r = job.result.to_dict()
        r["charts"] = [Path(c).name for c in r["charts"]]
        r["report_path"] = Path(r["report_path"]).name if r["report_path"] else ""
        view["result"] = r
    return view


# ----------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    s = _settings()
    return {"status": "ok", "provider": s.provider, "model": s.model, "api_key_configured": bool(s.api_key),
            "variants": list(VARIANTS)}


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    return _list_datasets()


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    name = _SAFE_NAME.sub("_", Path(file.filename or "").name).strip("._")
    if not name or Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type; allowed: {sorted(SUPPORTED)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 50 MB)")
    if not data:
        raise HTTPException(400, "Empty file")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex[:8]}_{name}"
    path = UPLOADS_DIR / stored
    path.write_bytes(data)
    profile = profile_files([path])
    if "Could not be read" in profile:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse the file: {profile.splitlines()[-1]}")
    return {"id": f"upload:{stored}", "name": stored, "kind": "upload", "bytes": len(data), "profile": profile}


@app.post("/api/runs", status_code=201)
def start_run(req: RunRequest) -> dict:
    files = [resolve_dataset(d) for d in req.datasets]
    try:
        settings = _settings(req.model)
        tracker = RunTracker()
        llm = app.state.llm_factory(settings, tracker)
    except Exception as e:  # e.g. missing API key
        raise HTTPException(503, f"LLM not available: {e}") from e
    job = Job(tracker.run_id, req.question, req.variant, req.datasets, tracker)
    tracker.listeners.append(job.notify)
    JOBS[job.run_id] = job

    def work():
        try:
            job.result = run_analysis(
                req.question, files, req.variant, settings=settings, llm=llm, tracker=tracker,
                require_approval=req.require_approval, approval_callback=job.wait_for_approval,
            )
            job.status = job.result.status
        except Exception as e:  # defensive: run_analysis already catches LLM errors
            job.status = "error"
            tracker.event("system", "error", f"{type(e).__name__}: {e}")
        finally:
            job.notify()

    threading.Thread(target=work, name=f"run-{job.run_id}", daemon=True).start()
    return {"run_id": job.run_id}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)
    return [{"run_id": j.run_id, "question": j.question, "status": j.status, "variant": j.variant} for j in jobs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _job_view(_job(run_id))


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, since: int = 0) -> dict:
    job = _job(run_id)
    events = job.tracker.events[since:]
    return {"events": [asdict(e) for e in events], "next": since + len(events), "status": job.status}


@app.get("/api/runs/{run_id}/stream")
def stream_events(run_id: str) -> StreamingResponse:
    job = _job(run_id)

    def gen():
        sent = 0
        while True:
            with job._changed:
                if len(job.tracker.events) <= sent and job.status in ("running", "awaiting_approval"):
                    job._changed.wait(timeout=15)
            events = job.tracker.events[sent:]
            for e in events:
                yield json.dumps(asdict(e), default=str) + "\n"
            sent += len(events)
            if job.status not in ("running", "awaiting_approval") and sent >= len(job.tracker.events):
                yield json.dumps({"agent": "system", "kind": "end", "content": {"status": job.status}}) + "\n"
                return
            if not events:
                yield json.dumps({"agent": "system", "kind": "heartbeat", "content": job.status}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/runs/{run_id}/approve")
def approve(run_id: str, req: ApprovalRequest) -> dict:
    job = _job(run_id)
    if job.status != "awaiting_approval":
        raise HTTPException(409, f"Run is not awaiting approval (status: {job.status})")
    job.decide(req.model_dump())
    return {"ok": True}


@app.get("/api/runs/{run_id}/files/{name}")
def get_file(run_id: str, name: str) -> FileResponse:
    job = _job(run_id)
    if not re.fullmatch(r"(report\.md|analysis\.py|chart_\d+\.png)", name):
        raise HTTPException(400, "Unknown file")
    path = _settings().outputs_dir / job.run_id / name
    if job.result is not None and job.result.report_path:
        path = Path(job.result.report_path).parent / name
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path)
