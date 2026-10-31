# Design decisions

Short records of the choices that shaped the project and what was traded off.

### 1. LangGraph for orchestration
The workflow has real loops (fix, revise), conditional routing and a human pause. LangGraph
gives typed state, conditional edges, `interrupt()` + checkpointer for human-in-the-loop, and
step streaming out of the box. Each agent is a plain function `(state, ctx) -> update`, so it
is testable without the graph.

### 2. One OpenAI-compatible client for every provider
Gemini, Groq, OpenAI and Ollama all expose the chat-completions protocol, so a provider is
just a base URL + key + default model. No LangChain wrappers: fewer dependencies, and
retries/metrics are under our control.

### 3. Gemini as the default provider
The Groq free quota was exhausted during development, so the default moved to Gemini.
This API key returns 404 for the `gemini-2.5-*` models; probing the available ones showed
`gemini-3.5-flash-lite` at ~1–3 s per call versus `gemini-3.5-flash` at ~13 s per call
with occasional timeouts. The fast model is the default; the stronger one is selectable
(`LLM_MODEL`). It could not be used in the experiments: its free tier allows only 20
requests per day (one full evaluation needs ~400), so the second model in the comparison
is the previous-generation `gemini-3.1-flash-lite`.

### 4. Client-side rate limiting + server-hinted backoff
The free tier allows 15 requests/minute per model. Parallel evaluation workers retrying
blindly produced bursts of 429s (5 retries, ~70 s per run in the pilot). A process-wide
sliding-window limiter (14 rpm default for Gemini, `LLM_RPM` to change) queues requests
instead, and backoff uses Gemini's "retry in N s" hint. Queueing time is tracked separately
from model latency so reported latencies stay honest. When the server's hint is long
(Gemini answers "retry in 18h41m" once a *daily* quota is used up) the client fails fast
with a clear `QuotaExhausted` error instead of retrying for minutes.

### 5. Deterministic dataset profile before any LLM call
The first live run produced a confidently wrong answer: six 999-unit orders (data-entry
errors) dominated regional totals, and the reviewer approved a +440% quarter-on-quarter
swing. The fix was not more agent calls but better context: the profile lists outlier
*values*, numbers stored as text, casing variants, duplicates and suspicious zeros; the
planner must address every note, and the reviewer is told that implausible swings usually
mean an ignored note. The answer then matched the reference.

### 6. One complete script per attempt (no persistent kernel)
Every script re-loads and re-cleans the data. That costs a second or two but makes every
attempt reproducible, makes the final `analysis.py` a self-contained deliverable, and
avoids hidden state between attempts. Notebook-style kernels would be faster for long
explorations, but harder to sandbox.

### 7. Sandbox = static allow-list + isolated subprocess (not a container)
Containers / gVisor / Firecracker are the right isolation for untrusted multi-tenant code,
but they need a daemon and privileges that a laptop project and its tests cannot assume.
The chosen layers (AST allow-list, `python -I` subprocess, temp dir with copied data,
timeouts, rlimits, socket block, scrubbed env) stop the realistic failure modes of
LLM-written analysis code and accidental exfiltration of API keys. **For a multi-user
deployment** the subprocess should additionally run in a container with no network
namespace, a read-only root and seccomp.

### 8. Reporter only sees printed output; charts attached deterministically
LLMs invent plausible numbers. The Reporter gets the script's stdout (not the data) and is
told to use only those numbers; the coder must print every number the answer depends on.
Image links written by the model are stripped and the real chart files are appended, so a
report can never reference a chart that does not exist.

### 9. Deterministic grader instead of an LLM judge
Every evaluation question has a single label and/or number as its answer, with cleaning
rules stated in the question so there is exactly one right answer. Expected answers are
computed from reference cleaners, never typed by hand, and every category answer leads the
runner-up by at least 8% so legitimate cleaning differences cannot flip it. Grading looks
only at the one-line `**Answer:**`: the first label mentioned must be right and some number
must be within tolerance. Cheaper, reproducible, and no judge bias.

### 10. Single-agent baseline gets the same budget
For a fair comparison the single agent has the same profile, sandbox and maximum number of
executions (1 + 3 fixes + 2 review rounds). It differs only in having no separate planning,
reviewing or reporting roles.

### 11. In-memory job store in the API
Runs are background threads with state in a dict; the UI polls `/events` (simple, survives
Streamlit's rerun model) and API clients can use the NDJSON `/stream`. This is fine for a
single process. Horizontal scaling would need a queue (e.g. Redis/RQ or Celery) and a
persistent LangGraph checkpointer (SQLite/Postgres) so paused approvals survive restarts.

### 12. MCP server package named `analyst_mcp`
A top-level package named `mcp` would shadow the MCP SDK. Dataset access is restricted to
`data/` and `uploads/` by default; absolute paths need `ANALYST_MCP_ALLOW_ANY_PATH=1`,
because the MCP client (an LLM) chooses the arguments.

### 13. A second, "unhinted" evaluation suite
With cleaning rules spelled out in the questions, all three variants scored 35/35, which
measures pandas fluency, not analyst judgement. The unhinted suite removes exactly the
error-detection hints (outlier orders, salary typos, invalid durations, casing notes) and
keeps definitional rules, so the expected answers are unchanged but the agents must find
the data errors themselves. That is where the variants start to differ.

### 14. Infrastructure errors are not wrong answers
Runs that die on API errors (503 overload, exhausted daily quota) are recorded with status
`error`, retried when the harness is re-run, and reported as `infra_errors`, excluded from
accuracy. Counting them as wrong would blame the agent for the provider's outage.
