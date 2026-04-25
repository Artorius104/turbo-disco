import json
from typing import Any, Dict

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Scientist", page_icon="🧪", layout="wide")

DEFAULT_BACKEND = "http://localhost:8000"

SAMPLES = {
    "Diagnostics — CRP biosensor": (
        "A paper-based electrochemical biosensor functionalized with anti-CRP antibodies will "
        "detect C-reactive protein in whole blood at concentrations below 0.5 mg/L within 10 minutes, "
        "matching laboratory ELISA sensitivity without requiring sample preprocessing."
    ),
    "Gut Health — L. rhamnosus GG": (
        "Supplementing C57BL/6 mice with Lactobacillus rhamnosus GG for 4 weeks will reduce "
        "intestinal permeability by at least 30% compared to controls, measured by FITC-dextran "
        "assay, due to upregulation of tight junction proteins claudin-1 and occludin."
    ),
    "Cell Biology — Trehalose cryoprotectant": (
        "Replacing sucrose with trehalose as a cryoprotectant in the freezing medium will increase "
        "post-thaw viability of HeLa cells by at least 15 percentage points compared to the standard "
        "DMSO protocol, due to trehalose's superior membrane stabilization at low temperatures."
    ),
    "Climate — Sporomusa CO2 fixation": (
        "Introducing Sporomusa ovata into a bioelectrochemical system at a cathode potential of "
        "−400mV vs SHE will fix CO₂ into acetate at a rate of at least 150 mmol/L/day, "
        "outperforming current biocatalytic carbon capture benchmarks by at least 20%."
    ),
}

NOVELTY_COLOR = {
    "not found": "#1f9d55",
    "similar work exists": "#d97706",
    "exact match found": "#dc2626",
}

SOURCE_LABEL = {
    "api:arxiv": "Live API · arXiv",
    "api:crossref": "Live API · CrossRef",
    "local_fallback": "Fallback · Local corpus",
}


def post(backend: str, path: str, payload: Dict[str, Any], timeout: float = 90.0) -> Dict:
    url = backend.rstrip("/") + path
    r = httpx.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def render_qc(qc: Dict):
    novelty = qc.get("novelty", "not found")
    color = NOVELTY_COLOR.get(novelty, "#6b7280")
    source = SOURCE_LABEL.get(qc.get("source", ""), qc.get("source", "?"))
    cols = st.columns([2, 3])
    with cols[0]:
        st.markdown(
            f"<div style='padding:14px;border-radius:10px;background:{color};color:white;"
            f"font-weight:600;text-align:center;font-size:18px;'>"
            f"{novelty.upper()}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<div style='padding:14px;border-radius:10px;background:#1f2937;color:#e5e7eb;"
            f"font-weight:500;'>Source: {source}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Top references")
    papers = qc.get("papers") or []
    if not papers:
        st.info("No related papers retrieved.")
        return
    for p in papers:
        score = p.get("similarity_score", 0.0)
        with st.container(border=True):
            st.markdown(f"**{p.get('title', 'Untitled')}**")
            meta = f"{p.get('authors', '')} · {p.get('year', '')}"
            link = p.get("link") or ""
            if link:
                meta += f" · [link]({link})"
            st.markdown(meta)
            st.progress(min(max(score, 0.0), 1.0), text=f"similarity {score:.2f}")
            if p.get("abstract"):
                with st.expander("abstract"):
                    st.write(p["abstract"])


def render_plan(plan: Dict):
    tabs = st.tabs(["Protocol", "Materials", "Budget", "Timeline", "Validation", "Raw JSON"])

    with tabs[0]:
        for step in plan.get("protocol", []):
            with st.expander(f"Step {step.get('step', '?')} · {step.get('title', '')}  ({step.get('duration', '')})"):
                st.write(step.get("description", ""))
                refs = step.get("references") or []
                if refs:
                    st.caption("references: " + ", ".join(refs))

    with tabs[1]:
        materials = plan.get("materials", [])
        if materials:
            df = pd.DataFrame(materials)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No materials listed.")

    with tabs[2]:
        budget = plan.get("budget") or {}
        total = budget.get("total_usd", 0)
        line_items = budget.get("line_items", [])
        st.metric("Estimated Total", f"${total:,.0f}")
        if line_items:
            df = pd.DataFrame(line_items)
            fig = px.pie(df, names="category", values="amount_usd", hole=0.4)
            fig.update_layout(margin=dict(t=20, b=0, l=0, r=0), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[3]:
        timeline = plan.get("timeline", [])
        if timeline:
            rows = []
            offset = {}
            for ph in timeline:
                deps = ph.get("depends_on") or []
                start = max((offset.get(d, 0) for d in deps), default=0)
                duration = float(ph.get("duration_weeks", 0) or 0)
                end = start + duration
                offset[ph["phase"]] = end
                rows.append({
                    "Phase": ph["phase"],
                    "Start": start,
                    "Finish": end,
                    "Duration (weeks)": duration,
                })
            df = pd.DataFrame(rows)
            fig = px.bar(
                df, x="Duration (weeks)", y="Phase", base="Start", orientation="h",
                hover_data=["Start", "Finish"],
            )
            fig.update_layout(
                xaxis_title="Weeks from start",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10, b=0, l=0, r=0),
                height=max(220, 60 * len(rows)),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No timeline provided.")

    with tabs[4]:
        v = plan.get("validation") or {}
        st.markdown(f"**Primary metric:** {v.get('primary_metric', '—')}")
        st.markdown(f"**Success criteria:** {v.get('success_criteria', '—')}")
        controls = v.get("controls") or []
        if controls:
            st.markdown("**Controls:**")
            for c in controls:
                st.markdown(f"- {c}")

    with tabs[5]:
        st.code(json.dumps(plan, indent=2), language="json")


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("🧪 The AI Scientist")
st.caption("Hypothesis → Literature QC → Operationally complete experiment plan")

with st.sidebar:
    st.subheader("Settings")
    backend = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    show_parsed = st.checkbox("Show parsed hypothesis", value=True)

st.markdown("##### 1. Choose or write a hypothesis")
sample_label = st.radio(
    "Sample hypotheses",
    options=list(SAMPLES.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
hypothesis = st.text_area(
    "Hypothesis",
    value=SAMPLES[sample_label],
    height=140,
    label_visibility="collapsed",
)

run = st.button("Run analysis", type="primary", use_container_width=True)

if run and hypothesis.strip():
    parsed_payload = None
    parse_status = st.status("Parsing hypothesis…", expanded=False)
    try:
        parsed = post(backend, "/parse", {"hypothesis": hypothesis}, timeout=60)
        parsed_payload = parsed
        parse_status.update(label="Hypothesis parsed", state="complete")
    except Exception as e:
        parse_status.update(label=f"Parse failed: {e}", state="error")
        st.stop()

    if show_parsed:
        with st.expander("Parsed hypothesis", expanded=False):
            st.json(parsed_payload)

    qc_status = st.status("Running literature QC…", expanded=False)
    try:
        qc = post(backend, "/literature_qc",
                  {"hypothesis": hypothesis, "parsed": parsed_payload}, timeout=60)
        qc_status.update(label="Literature QC complete", state="complete")
    except Exception as e:
        qc_status.update(label=f"Literature QC failed: {e}", state="error")
        st.stop()

    st.markdown("---")
    st.markdown("##### 2. Literature QC")
    render_qc(qc)

    plan_status = st.status("Generating experiment plan…", expanded=False)
    try:
        plan = post(
            backend, "/generate_plan",
            {"hypothesis": hypothesis, "parsed": parsed_payload, "papers": qc.get("papers", [])},
            timeout=180,
        )
        plan_status.update(label="Plan ready", state="complete")
    except Exception as e:
        plan_status.update(label=f"Plan generation failed: {e}", state="error")
        st.stop()

    st.markdown("---")
    st.markdown("##### 3. Experiment Plan")
    render_plan(plan)
elif run:
    st.warning("Hypothesis cannot be empty.")
