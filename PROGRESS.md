# PROGRESS

This file tracks the build so work can be resumed if interrupted.
**To resume:** read this file, then `ARCHITECTURE.md`, then run `.venv/bin/python -m pytest`.

## Plan (milestones)

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Scaffolding: venv (`uv venv --python 3.12 .venv`), `.gitignore`, requirements, this plan | ✅ done |
| M1 | Synthetic datasets with planted patterns + messy data (`data/generate_datasets.py`) + tests | ✅ done |
| M2 | Config + model-agnostic LLM layer + run tracker (tokens / cost / latency / retries) | ✅ done |
| M3 | Safe code sandbox (AST safety check + subprocess + timeout + temp dir + no network) + tests | ✅ done |
| M4 | LangGraph agents: Planner → Coder → Executor → Reviewer → Reporter, self-correction loop, human-in-the-loop approval, single-agent & no-reviewer variants, markdown report with charts + tests (fake LLM) | ✅ done |
| M5 | Evaluation: 30+ questions with reference answers, grader, harness + tests | ⏳ |
| M6 | Run experiments (single vs multi, ± reviewer, 2 LLMs) → `results/` tables + charts | ⏳ |
| M7 | FastAPI backend (streaming steps, approval endpoint) + Streamlit UI + tests | ⏳ |
| M8 | MCP server exposing the analysis tools + tests + docs | ⏳ |
| M9 | Dockerfile, README, ARCHITECTURE, DECISIONS final pass | ⏳ |

## Log

- 2026-10-31 — M0: created `.venv` with Python 3.12 via `uv` (system Python is 3.14; 3.12 has the widest wheel support).
  Dependencies installed into `.venv` only. Note: the `GROQ_API_KEY` in `.env` currently returns
  `401 Invalid API Key` from Groq — tests use a scripted fake LLM so they do not need it.
- 2026-10-31 — Switched the default LLM provider from Groq to **Gemini** (via Gemini's
  OpenAI-compatible endpoint) because the Groq quota was exhausted. Groq/OpenAI/Ollama stay selectable via `LLM_PROVIDER`.
- 2026-10-31 — M2: `analyst/config.py`, `analyst/llm.py` (retries w/ backoff, FakeLLM), `analyst/tracker.py`.
  This key gets 404 for gemini-2.5-*; probed models: `gemini-3.5-flash-lite` ~1s/call (now the default),
  `gemini-3.5-flash` ~13s/call and sometimes times out (kept as the "stronger" model for experiments).
- 2026-10-31 — M3: `analyst/sandbox.py` (AST allow-list, `python -I` subprocess, rlimits, socket block, scrubbed env, figure capture) + 30 tests.
- 2026-10-31 — M4: `analyst/agents/*`, `analyst/graph.py`, `analyst/runner.py` (CLI: `python -m analyst.runner "question" data/sales.csv [--variant single|no_reviewer] [--approve]`),
  `analyst/profiling.py`. First live run gave a wrong answer (999-unit outliers kept; reviewer approved a +440% swing) ->
  profile now lists outlier values, planner must address every profile note, reviewer told to distrust implausible swings.
  After the fix: West -27.24% (matches reference). Live single-agent + approval-with-feedback runs verified.
