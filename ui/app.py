"""
Streamlit front-end for the analyst API.

    uvicorn api.main:app --port 8000      # terminal 1
    streamlit run ui/app.py               # terminal 2   (API_URL defaults to http://localhost:8000)
"""

from __future__ import annotations

import os
import re
import time

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = {
    "sales.csv": "Which region had the largest percentage revenue decline from Q2 to Q3 2024, and why?",
    "web_traffic.csv": "Which marketing channel converts best, and how do weekends compare to weekdays?",
    "hr_employees.xlsx": "What drives employee attrition? Compare departments and overtime.",
    "support_tickets.csv": "Which agents or products are causing slow ticket resolution?",
}
VARIANT_HELP = {
    "multi": "Planner → Coder → Executor → Reviewer → Reporter (recommended)",
    "no_reviewer": "Same team without the Reviewer (faster, less checking)",
    "single": "One agent writes code, sees output and reports (baseline)",
}
ICONS = {"plan": "🗺️", "code": "💻", "execution": "⚙️", "review": "🔎", "report": "📝", "retry": "⏳",
         "awaiting_approval": "✋", "approval": "👍", "error": "❌", "profile": "📊", "start": "🚀", "done": "🏁"}

st.set_page_config(page_title="Autonomous Data Analyst", page_icon="📊", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kw)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API}. Start it with: `uvicorn api.main:app --port 8000`")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"API error {r.status_code}: {detail}")
        st.stop()
    return r


# ----------------------------------------------------------------- sidebar
health = api("GET", "/api/health").json()
with st.sidebar:
    st.header("Setup")
    st.caption(f"LLM: **{health['provider']} / {health['model']}**")
    if not health["api_key_configured"]:
        st.warning("No API key configured on the server - add it to `.env`.")

    uploaded = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "tsv", "json", "parquet"])
    if uploaded is not None and st.session_state.get("uploaded_name") != uploaded.name:
        info = api("POST", "/api/datasets", files={"file": (uploaded.name, uploaded.getvalue())}).json()
        st.session_state.uploaded_name = uploaded.name
        st.session_state.selected = [info["id"]]
        st.success(f"Uploaded {info['name']}")

    datasets = api("GET", "/api/datasets").json()
    ids = [d["id"] for d in datasets]
    default = [i for i in st.session_state.get("selected", ["sample:sales.csv"]) if i in ids]
    selected = st.multiselect("Datasets", ids, default=default,
                              format_func=lambda i: ("📁 " if i.startswith("upload:") else "🧪 ") + i.split(":", 1)[1])
    variant = st.radio("Agent setup", list(VARIANT_HELP), format_func=lambda v: v.replace("_", " "),
                       captions=list(VARIANT_HELP.values()))
    approval = st.checkbox("Let me review the plan before code runs", value=False, disabled=variant == "single")
    model = st.text_input("Model override (optional)", placeholder=health["model"])

# -------------------------------------------------------------------- main
st.title("📊 Autonomous Data Analyst")
st.caption("Ask a business question. A team of agents plans the analysis, writes and runs pandas code in a "
           "sandbox, fixes its own errors, reviews the result and writes a report with charts.")

example = EXAMPLES.get(selected[0].split(":", 1)[1], "") if selected else ""
question = st.text_area("Your question", value=example, height=80)
if st.button("Run analysis", type="primary", disabled=not (selected and question.strip())):
    body = {"question": question.strip(), "datasets": selected, "variant": variant,
            "require_approval": approval and variant != "single"}
    if model.strip():
        body["model"] = model.strip()
    st.session_state.run_id = api("POST", "/api/runs", json=body).json()["run_id"]
    st.rerun()

run_id = st.session_state.get("run_id")
if not run_id:
    st.stop()

run = api("GET", f"/api/runs/{run_id}").json()
events = api("GET", f"/api/runs/{run_id}/events").json()["events"]
running = run["status"] in ("running", "awaiting_approval")

st.divider()
left, right = st.columns([3, 2], gap="large")

# --------------------------------------------------------- agent timeline
with right:
    st.subheader("Agent steps")
    for ev in events:
        kind, agent, c = ev["kind"], ev["agent"], ev["content"]
        icon = ICONS.get(kind, "•")
        label = f"{icon} `{ev['t']:.1f}s` **{agent}** · {kind.replace('_', ' ')}"
        if kind == "plan":
            with st.expander(label, expanded=False):
                st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(c, 1)))
        elif kind == "code":
            with st.expander(f"{label} (v{c['version']}, {c['mode']})"):
                st.code(c["code"], language="python")
        elif kind == "execution":
            ok = "✅ ran" if c["ok"] else f"❌ {c['error'][:120]}"
            with st.expander(f"{label} - {ok}"):
                if c.get("violations"):
                    st.error("\n".join(c["violations"]))
                st.code(c["stdout"] or "(no output)", language="text")
        elif kind == "review":
            verdict = "✅ approved" if c["verdict"] == "approve" else "🔁 revision requested"
            with st.expander(f"{label} - {verdict}"):
                st.write(c.get("summary", ""))
                for issue in c.get("issues", []):
                    st.markdown(f"- {issue}")
        elif kind == "profile":
            with st.expander(label):
                st.markdown(c)
        elif kind in ("retry", "error"):
            st.warning(f"{label}: {c}")
        elif kind in ("report", "done", "approval", "awaiting_approval", "start"):
            st.markdown(label)

# ------------------------------------------------------ approval / report
with left:
    st.subheader(run["question"])
    if run["status"] == "awaiting_approval":
        st.info("✋ The planner proposes this plan. Approve it, reject it, or send feedback to re-plan.")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(run["pending_plan"], 1)))
        feedback = st.text_input("Feedback for the planner (optional)")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Approve", type="primary"):
            api("POST", f"/api/runs/{run_id}/approve", json={"approved": True})
            st.rerun()
        if c2.button("💬 Re-plan with feedback", disabled=not feedback.strip()):
            api("POST", f"/api/runs/{run_id}/approve", json={"approved": False, "feedback": feedback})
            st.rerun()
        if c3.button("🛑 Cancel"):
            api("POST", f"/api/runs/{run_id}/approve", json={"approved": False})
            st.rerun()
    elif running:
        st.info("⏳ The agents are working... this page updates automatically.")
    else:
        res = run.get("result") or {}
        status = run["status"]
        if status == "completed":
            st.success("Analysis complete")
        elif status == "rejected":
            st.warning("Cancelled: the plan was rejected.")
        else:
            st.error(f"Analysis {status}. {res.get('error', '')}")
        report = res.get("report") or ""
        if report:
            # charts are shown from the API below, so drop the markdown image section
            st.markdown(re.split(r"\n## Charts\n", report)[0])
            for name in res.get("charts", []):
                st.image(api("GET", f"/api/runs/{run_id}/files/{name}").content, width="stretch")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ report.md", report, file_name=f"report_{run_id}.md")
            if res.get("code"):
                d2.download_button("⬇️ analysis.py", res["code"], file_name=f"analysis_{run_id}.py")

    m = run["metrics"]
    cols = st.columns(5)
    cols[0].metric("LLM calls", m["llm_calls"])
    cols[1].metric("Tokens", f"{m['total_tokens']:,}")
    cols[2].metric("Est. cost", f"${m['cost_usd']:.4f}")
    cols[3].metric("Retries", m["retries"])
    cols[4].metric("Elapsed", f"{m['wall_time_s']:.0f}s")

if run["status"] == "running":  # not while awaiting approval: a rerun could wipe half-typed feedback
    time.sleep(1.0)
    st.rerun()
