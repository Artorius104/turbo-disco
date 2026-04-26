import json
from typing import Any, Dict, Optional

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
    "api:europepmc": "Live API · Europe PMC",
    "api:openalex": "Live API · OpenAlex",
    "api:crossref": "Live API · CrossRef",
    "local_fallback": "Fallback · Local corpus",
}


def post(backend: str, path: str, payload: Dict[str, Any], timeout: float = 90.0) -> Dict:
    url = backend.rstrip("/") + path
    r = httpx.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_bytes(backend: str, path: str, payload: Dict[str, Any], timeout: float = 60.0) -> bytes:
    url = backend.rstrip("/") + path
    r = httpx.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.content


def render_validation(v: Dict, on_use_suggested):
    score = float(v.get("score", 0.0))
    status = v.get("status", "ok")
    if status == "ok":
        st.success(f"Hypothesis quality: {score:.0%} — ready to proceed.")
        return
    color = "#d97706"
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{color};color:white;"
        f"font-weight:600;font-size:15px;'>"
        f"Hypothesis quality: {score:.0%} — needs revision</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Issues**")
        for i in v.get("issues", []):
            st.markdown(f"- {i}")
    with cols[1]:
        st.markdown("**Suggestions**")
        for s in v.get("suggestions", []):
            st.markdown(f"- {s}")
    improved = v.get("improved_hypothesis") or ""
    if improved:
        st.markdown("**Suggested rewrite**")
        st.info(improved)
        st.button(
            "Use suggested rewrite",
            on_click=on_use_suggested,
            args=(improved,),
            type="primary",
            key="use_suggested_btn",
        )


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


def render_protocols(protocols: list[Dict]):
    if not protocols:
        return
    st.markdown("#### Related protocols")
    for i, pr in enumerate(protocols, 1):
        with st.container(border=True):
            link = pr.get("link") or ""
            title = pr.get("title", "Untitled")
            label = f"**PR{i}.** {title}"
            if link:
                label += f"  ·  [open]({link})"
            st.markdown(label)
            src = pr.get("source") or ""
            if src:
                st.caption(src)
            if pr.get("summary"):
                st.write(pr["summary"])


def render_plan(plan: Dict):
    tabs = st.tabs(["Protocol", "Materials", "Budget", "Timeline", "Validation", "Raw JSON"])

    with tabs[0]:
        for step in plan.get("protocol", []):
            with st.expander(
                f"Step {step.get('step', '?')} · {step.get('title', '')}  ({step.get('duration', '')})"
            ):
                st.write(step.get("description", ""))
                refs = step.get("references") or []
                if refs:
                    st.caption("references: " + ", ".join(refs))

    with tabs[1]:
        materials = plan.get("materials", [])
        if materials:
            df = pd.DataFrame(materials)
            preferred = ["name", "supplier", "catalog", "quantity", "unit_cost_usd", "url"]
            cols = [c for c in preferred if c in df.columns] + [
                c for c in df.columns if c not in preferred
            ]
            st.dataframe(
                df[cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("url"),
                    "unit_cost_usd": st.column_config.NumberColumn("unit_cost_usd", format="$%.0f"),
                },
            )
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


def render_review_panel(backend: str, hypothesis: str, parsed: Optional[Dict],
                        plan: Dict):
    st.markdown("---")
    with st.expander("📝 Review this plan (helps the model improve next time)", expanded=False):
        sections = ["protocol", "materials", "budget", "timeline", "validation", "overall"]
        items: list[dict] = []
        for sec in sections:
            cols = st.columns([1, 2, 3])
            with cols[0]:
                st.markdown(f"**{sec}**")
            with cols[1]:
                rating = st.slider(
                    f"rating_{sec}", 1, 5, 3,
                    label_visibility="collapsed", key=f"rating_{sec}",
                )
            with cols[2]:
                correction = st.text_input(
                    f"correction_{sec}",
                    placeholder="What was wrong / what should it have been?",
                    label_visibility="collapsed", key=f"corr_{sec}",
                )
            if correction.strip() or rating != 3:
                items.append({
                    "section": sec,
                    "rating": rating,
                    "correction": correction,
                    "comment": "",
                })
        comment = st.text_area("Overall comment (optional)",
                                key="overall_comment", height=80)
        if comment.strip():
            items.append({"section": "overall", "rating": None,
                          "correction": "", "comment": comment})

        if st.button("Submit feedback", type="primary", key="submit_feedback_btn"):
            if not items:
                st.warning("No changes captured — adjust a rating or write a correction first.")
            else:
                try:
                    resp = post(backend, "/feedback", {
                        "hypothesis": hypothesis,
                        "parsed": parsed,
                        "plan": plan,
                        "items": items,
                    })
                    st.toast(f"Feedback stored ({resp.get('stored', 0)} items)", icon="✅")
                except Exception as e:
                    st.error(f"Feedback submission failed: {e}")


def render_pdf_button(backend: str, hypothesis: str, parsed: Optional[Dict],
                      qc: Optional[Dict], plan: Dict):
    try:
        pdf_bytes = post_bytes(backend, "/export_pdf", {
            "hypothesis": hypothesis,
            "parsed": parsed,
            "qc": qc,
            "plan": plan,
        }, timeout=60)
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return
    st.download_button(
        label="📄 Download experiment plan (PDF)",
        data=pdf_bytes,
        file_name="experiment-plan.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ─── Session state ────────────────────────────────────────────────────────────

if "hypothesis" not in st.session_state:
    st.session_state.hypothesis = list(SAMPLES.values())[0]
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # holds {parsed, qc, plan}


def _on_sample_change():
    st.session_state.hypothesis = SAMPLES[st.session_state.sample_label]
    st.session_state.last_run = None


def _use_suggested(text: str):
    st.session_state.hypothesis = text
    st.session_state.last_run = None


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("🧪 The AI Scientist")
st.caption("Hypothesis → Quality check → Literature QC → Operationally complete experiment plan")

with st.sidebar:
    st.subheader("Settings")
    backend = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    show_parsed = st.checkbox("Show parsed hypothesis", value=True)
    skip_validation = st.checkbox("Skip hypothesis quality gate", value=False)

st.markdown("##### 1. Choose or write a hypothesis")
st.radio(
    "Sample hypotheses",
    options=list(SAMPLES.keys()),
    horizontal=True,
    label_visibility="collapsed",
    key="sample_label",
    on_change=_on_sample_change,
)
hypothesis = st.text_area(
    "Hypothesis",
    height=140,
    label_visibility="collapsed",
    key="hypothesis",
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

    proceed = True
    if not skip_validation:
        val_status = st.status("Validating hypothesis quality…", expanded=False)
        try:
            validation = post(backend, "/validate_hypothesis",
                              {"hypothesis": hypothesis, "parsed": parsed_payload}, timeout=60)
            val_status.update(label=f"Quality {validation.get('score', 0):.0%}",
                              state="complete")
        except Exception as e:
            val_status.update(label=f"Validation failed: {e}", state="error")
            validation = None
        if validation:
            st.markdown("##### Hypothesis quality")
            render_validation(validation, _use_suggested)
            if validation.get("status") != "ok":
                st.warning(
                    "Click **Use suggested rewrite** to retry, edit the text above, "
                    "or tick *Skip hypothesis quality gate* in the sidebar to proceed anyway."
                )
                proceed = False

    if proceed:
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
                timeout=240,
            )
            plan_status.update(label="Plan ready", state="complete")
        except Exception as e:
            plan_status.update(label=f"Plan generation failed: {e}", state="error")
            st.stop()

        st.session_state.last_run = {
            "parsed": parsed_payload, "qc": qc, "plan": plan,
        }

        st.markdown("---")
        st.markdown("##### 3. Experiment Plan")
        render_protocols(plan.get("protocols_used") or [])
        render_plan(plan)
        render_pdf_button(backend, hypothesis, parsed_payload, qc, plan)
        render_review_panel(backend, hypothesis, parsed_payload, plan)
elif run:
    st.warning("Hypothesis cannot be empty.")
elif st.session_state.last_run:
    last = st.session_state.last_run
    st.markdown("---")
    st.markdown("##### Last result")
    render_qc(last["qc"])
    render_protocols(last["plan"].get("protocols_used") or [])
    render_plan(last["plan"])
    render_pdf_button(backend, st.session_state.hypothesis,
                      last["parsed"], last["qc"], last["plan"])
    render_review_panel(backend, st.session_state.hypothesis,
                        last["parsed"], last["plan"])
