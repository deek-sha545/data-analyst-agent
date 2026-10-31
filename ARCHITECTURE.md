# Architecture

## Overview

```
                ┌────────────── Streamlit UI (ui/app.py) ──────────────┐
                │ upload · ask · watch agent steps · approve plan · report │
                └───────────────┬──────────────────────────────────────┘
                                │ HTTP (JSON polling / NDJSON stream)
      MCP clients ──stdio/HTTP──┤
   (analyst_mcp/server.py)      ▼
                        FastAPI (api/main.py) ── background thread per run
                                │
                                ▼
                     analyst.runner.run_analysis()  ◄── CLI: python -m analyst.runner
                                │                   ◄── eval harness
                                ▼
          ┌──────────────── LangGraph (analyst/graph.py) ─────────────────┐
          │ profile → planner → [human approval] → coder → executor ──┐    │
          │                          ▲    ▲  fix (≤3)  │               │    │
          │                          │    └────────────┘               ▼    │
          │                          └──── revise (≤2) ──────────── reviewer │
          │                                                          │      │
          │                                              reporter ◄──┘      │
          └────────────────────────────────────────────────────────────────┘
                 │                     │                         │
          analyst/llm.py         analyst/sandbox.py        analyst/tracker.py
   (OpenAI-compatible client,  (AST allow-list + isolated   (tokens, cost, latency,
    retries, rate limiter)      subprocess, no network)      retries, step events)
```

## Components

| Module | Responsibility |
|---|---|
| `analyst/config.py` | Settings from env / `.env`; provider table (Gemini default, Groq, OpenAI, Ollama); price table |
| `analyst/llm.py` | `LLM.chat()` for any OpenAI-compatible endpoint: exponential backoff honouring server hints, process-wide sliding-window rate limiter, empty-reply retry; `FakeLLM` for tests; code/JSON extraction |
| `analyst/tracker.py` | `RunTracker`: every LLM call (tokens, cost, model latency, queue/backoff wait, retries) and every agent event; listeners power live streaming; saved to `logs/<run_id>.json` |
| `analyst/profiling.py` | Deterministic dataset profile: dtypes, missing values, examples, and *messiness notes* (numbers stored as text, inconsistent casing, outliers with their values, duplicates, suspicious zeros) |
| `analyst/sandbox.py` | Static safety check + execution in a separate `python -I` process (see Security) |
| `analyst/agents/` | One module per agent: `planner`, `coder`, `executor`, `reviewer`, `reporter`, `solo` (single-agent baseline) |
| `analyst/graph.py` | LangGraph wiring of the three variants, loop caps, approval interrupt |
| `analyst/runner.py` | `run_analysis()` – the single entry point; drives the graph, handles interrupts, returns a `RunResult`; also the CLI |
| `api/main.py` | FastAPI: datasets, runs, events, NDJSON stream, approval, files |
| `ui/app.py` | Streamlit front-end on top of the API |
| `analyst_mcp/server.py` | MCP server: `list_datasets`, `profile_dataset`, `run_python`, `ask_analyst` |
| `eval/` | Reference cleaners, 35 questions, deterministic grader, resumable harness, report |

## The agents

All agents share one typed state (`analyst/agents/state.py`).

1. **Profile** (no LLM) – computes the dataset profile once; every agent sees it.
2. **Planner** – question + profile → 3–7 step plan (JSON). Must address every data-quality note that can affect the answer.
3. **Human approval** (optional) – `langgraph.types.interrupt()` pauses the graph with the plan. Resume with `{"approved": true}`, `{"approved": false, "feedback": "..."}` (→ re-plan) or `{"approved": false}` (→ cancel).
4. **Coder** – writes one complete script. Three modes chosen from the state: *draft* (from the plan), *fix* (previous run failed: sees its code + the cleaned traceback), *revise* (reviewer issues + previous output).
5. **Executor** (no LLM) – runs the script in the sandbox; charts are captured as PNGs.
6. **Reviewer** – checks the question is answered, the cleaning, the calculation logic and plausibility; returns `approve` or `revise` + concrete issues.
7. **Reporter** – writes the Markdown report from the *printed output only* (no invented numbers), starting with a one-line `**Answer:**`. Charts are attached deterministically (model-written image links are stripped), and `report.md`, `chart_N.png`, `analysis.py` are saved in `outputs/<run_id>/`.

### Loops and their limits

- **Self-correction:** failed execution → Coder (fix) → Executor, up to `MAX_FIX_ATTEMPTS` (3) per script version. Safety-check rejections count as failures, so the coder learns what is not allowed.
- **Review:** `revise` → Coder (revise) → Executor → Reviewer, up to `MAX_REVIEW_ROUNDS` (2). If issues remain, they are passed to the Reporter as caveats.
- **Fallback:** if a revision can no longer be fixed, the Reporter uses the *last successful* execution and says so.
- **Total failure:** the Reporter writes an honest "could not be determined" report; status `failed`.

### Variants (for the experiments)

| Variant | Graph |
|---|---|
| `multi` | full graph above |
| `no_reviewer` | executor success → reporter directly |
| `single` | one ReAct-style agent: each turn it writes code (executed, output fed back) or the final report. Same profile, sandbox and execution budget (1 + fixes + review rounds = 6 runs). |

## Security model (sandbox)

Generated code is untrusted. Two layers:

1. **Static check** (`check_code`): parses the AST and rejects imports outside a data-science allow-list (pandas, numpy, matplotlib, a few stdlib modules), `eval/exec/open/__import__/getattr…`, any dunder access (blocks `().__class__.__subclasses__()` escapes), pickle/SQL/HTML/clipboard I/O, and string literals that are absolute paths, `..` traversals or URLs.
2. **Isolated runtime** (`run_code`): a separate `python -I` process in a fresh temp directory holding *copies* of the data files; wall-clock timeout (killed), CPU/memory/file-size rlimits, socket functions replaced so any network access raises, and a scrubbed environment (no API keys reach generated code). Library frames are stripped from tracebacks before they go back to the LLM.

This is defence in depth for a single-user tool; see DECISIONS.md for what a multi-tenant deployment would add.

## Observability

Every run writes `logs/<run_id>.json` with all LLM calls and events. The summary reports
calls, tokens, estimated cost, **model latency** and, separately, **rate-limit wait**
(queueing + backoff), so latency numbers are not inflated by free-tier throttling.
