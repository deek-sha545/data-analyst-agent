# PROGRESS

This file tracks the build so work can be resumed if interrupted.
**To resume:** read this file, then `ARCHITECTURE.md`, then run `.venv/bin/python -m pytest`.

## Plan (milestones)

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Scaffolding: venv (`uv venv --python 3.12 .venv`), `.gitignore`, requirements, this plan | ✅ done |
| M1 | Synthetic datasets with planted patterns + messy data (`data/generate_datasets.py`) + tests | ⏳ |
| M2 | Config + model-agnostic LLM layer + run tracker (tokens / cost / latency / retries) | ⏳ |
| M3 | Safe code sandbox (AST safety check + subprocess + timeout + temp dir + no network) + tests | ⏳ |
| M4 | LangGraph agents: Planner → Coder → Executor → Reviewer → Reporter, self-correction loop, human-in-the-loop approval, single-agent & no-reviewer variants, markdown report with charts + tests (fake LLM) | ⏳ |
| M5 | Evaluation: 30+ questions with reference answers, grader, harness + tests | ⏳ |
| M6 | Run experiments (single vs multi, ± reviewer, 2 LLMs) → `results/` tables + charts | ⏳ |
| M7 | FastAPI backend (streaming steps, approval endpoint) + Streamlit UI + tests | ⏳ |
| M8 | MCP server exposing the analysis tools + tests + docs | ⏳ |
| M9 | Dockerfile, README, ARCHITECTURE, DECISIONS final pass | ⏳ |

## Log

- 2026-10-31 — M0: created `.venv` with Python 3.12 via `uv` (system Python is 3.14; 3.12 has the widest wheel support).
  Dependencies installed into `.venv` only. Note: the `GROQ_API_KEY` in `.env` currently returns
  `401 Invalid API Key` from Groq — tests use a scripted fake LLM so they do not need it.
