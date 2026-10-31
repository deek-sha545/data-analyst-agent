# Autonomous Data Analyst

A multi-agent system that answers business questions about messy spreadsheets. Upload a CSV or
Excel file and ask *"Why did West revenue drop in Q3?"*. A team of LangGraph agents then:

1. **profiles** the data (types, missing values, numbers stored as text, outliers…),
2. **plans** the analysis (optionally pausing for your approval),
3. **writes pandas code** and runs it in a **sandbox**,
4. **fixes its own errors** when the code fails,
5. has a **reviewer** check the logic and numbers, and
6. writes a **Markdown report with charts**, using only numbers the code actually printed.

It comes with a FastAPI backend, a Streamlit UI, an MCP server, a 35-question evaluation
with deterministic grading, and experiments comparing single- and multi-agent setups.

```
profile → planner → [you approve?] → coder → executor → reviewer → reporter
                                       ▲  fix ≤3 │         │ revise ≤2
                                       └─────────┴─────────┘
```

## Quick start

```bash
# Python 3.11+ (developed on 3.12)
uv venv --python 3.12 .venv && source .venv/bin/activate   # or: python -m venv .venv
pip install -r requirements.txt
cp .env.example .env          # then put your GEMINI_API_KEY in .env (free key: https://aistudio.google.com)
```

**Command line**

```bash
python -m analyst.runner "Which region had the biggest revenue drop from Q2 to Q3?" data/sales.csv
python -m analyst.runner "What drives attrition?" data/hr_employees.xlsx --approve          # review the plan first
python -m analyst.runner "Which channel converts best?" data/web_traffic.csv --variant single
```

**Web app**

```bash
uvicorn api.main:app --port 8000      # terminal 1 - API docs at http://localhost:8000/docs
streamlit run ui/app.py               # terminal 2 - http://localhost:8501
```

or with Docker: `docker compose up --build`.

**From Python**

```python
from analyst.runner import run_analysis
res = run_analysis("Which agent resolves tickets slowest?", ["data/support_tickets.csv"])
print(res.answer)          # one-line answer
print(res.report_path)     # outputs/<run_id>/report.md (+ chart_N.png, analysis.py)
print(res.metrics)         # llm_calls, tokens, cost_usd, latency, retries...
```

## MCP server

Exposes the analyst to MCP clients such as Claude Desktop, Claude Code or an IDE:

| Tool | What it does |
|---|---|
| `list_datasets` | datasets in `data/` and `uploads/` |
| `profile_dataset` | columns, dtypes, missing values, data-quality notes |
| `run_python` | run pandas/matplotlib code in the sandbox |
| `ask_analyst` | the full multi-agent analysis → answer, report, chart paths, metrics |

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "/path/to/data-analyst-agent/.venv/bin/python",
      "args": ["-m", "analyst_mcp.server"],
      "cwd": "/path/to/data-analyst-agent"
    }
  }
}
```

Claude Code: `claude mcp add data-analyst -- /path/to/.venv/bin/python -m analyst_mcp.server`
(run it from the project directory). For HTTP transport: `python -m analyst_mcp.server --http --port 8001`.

## Experiments

<!-- RESULTS -->

Reproduce with:

```bash
python -m eval.harness --models gemini-3.5-flash-lite --variants multi no_reviewer single
python -m eval.harness --models gemini-3.1-flash-lite --variants multi single
python -m eval.report          # -> results/summary.md, summary.csv, comparison.png
```

The harness can resume: re-running it skips finished combinations and retries runs that hit API errors.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | key for the default provider |
| `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, `openai` or `ollama` (all via the OpenAI-compatible API) |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | any model the provider serves |
| `LLM_RPM` | `14` for Gemini | client-side requests/minute cap (free tier allows 15); `0` = off |
| `MAX_FIX_ATTEMPTS` | `3` | self-correction rounds per script |
| `MAX_REVIEW_ROUNDS` | `2` | reviewer revision rounds |
| `SANDBOX_TIMEOUT` | `60` | seconds a generated script may run |
| `ANALYST_MCP_ALLOW_ANY_PATH` | unset | `1` lets MCP clients pass absolute file paths |

## Safety

Generated code never runs in the app's own process. It must pass a static allow-list check
(data-science imports only; no `eval`/`exec`/`open`, dunder access, file paths outside the
working directory, or URLs). It then runs in an isolated `python -I` subprocess in a temp
directory that holds copies of the data, with a timeout, CPU/memory limits, networking disabled
and **no API keys in its environment**. Details and limits are in [ARCHITECTURE.md](ARCHITECTURE.md#security-model-sandbox)
and [DECISIONS.md](DECISIONS.md).

## Tests

```bash
python -m pytest          # ~100 tests, offline: agents run against a scripted FakeLLM
```

The tests cover the LLM layer (retries, rate limiter, cost), sandbox escape attempts, every graph
branch (fix loop, review loop, fallback to the last good run, approve, re-plan and reject, budget
caps), the API (including upload and path-traversal checks), the MCP server through a real MCP client,
the grader and the Streamlit UI (headless).

## Project layout

```
analyst/            core library
  config.py         providers, models, prices, settings
  llm.py            OpenAI-compatible client, retries, rate limiter, FakeLLM
  tracker.py        per-run metrics + event stream
  profiling.py      deterministic dataset profile
  sandbox.py        safe code execution
  prompts.py        all agent prompts
  agents/           planner, coder, executor, reviewer, reporter, solo
  graph.py          LangGraph wiring for the three variants
  runner.py         run_analysis() + CLI
api/main.py         FastAPI backend
ui/app.py           Streamlit UI
analyst_mcp/        MCP server
eval/               reference cleaners, questions, grader, harness, report
data/               synthetic datasets with planted patterns + messiness (see data/README.md)
results/            experiment runs, summary tables, charts
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how it works and [DECISIONS.md](DECISIONS.md) for why.
