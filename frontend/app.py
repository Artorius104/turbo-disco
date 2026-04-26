import json
from typing import Any, Dict, List, Optional

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
    "not found":            "#10B981",
    "similar work exists":  "#F59E0B",
    "exact match found":    "#EF4444",
}

SOURCE_LABEL = {
    "api:arxiv":      "arXiv",
    "api:europepmc":  "Europe PMC",
    "api:openalex":   "OpenAlex",
    "api:crossref":   "CrossRef",
    "local_fallback": "Local corpus",
}

COST_SOURCE_BADGE = {
    "catalog":   ("#10B981", "catalog"),
    "web":       ("#3B82F6", "web"),
    "estimated": ("#F59E0B", "estimated"),
    "unknown":   ("#6B7280", "unknown"),
}


# ─── styling ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
:root {
    --indigo: #6366F1;
    --indigo-2: #818CF8;
    --bg-card: #0F172A;
    --border: #1F2937;
}
.block-container { padding-top: 1.5rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.app-hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(56,189,248,0.10));
    border: 1px solid var(--border);
    border-radius: 16px; padding: 22px 26px; margin-bottom: 12px;
}
.app-hero h1 { margin: 0; font-size: 28px; }
.app-hero p  { margin: 4px 0 0; color: #9CA3AF; font-size: 14px; }

.stepper { display: flex; gap: 8px; margin: 14px 0 22px; }
.step {
    flex: 1; padding: 10px 14px; border-radius: 12px;
    background: #0F172A; border: 1px solid var(--border);
    color: #94A3B8; font-size: 13px; display: flex; align-items: center; gap: 8px;
}
.step .dot {
    width: 22px; height: 22px; border-radius: 50%;
    background: #1E293B; color: #94A3B8;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600;
}
.step.done { background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.35); color: #D1FAE5; }
.step.done .dot { background: #10B981; color: white; }
.step.active { background: rgba(99,102,241,0.14); border-color: rgba(99,102,241,0.45); color: #E0E7FF; }
.step.active .dot { background: var(--indigo); color: white; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 18px; }
.kpi {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 16px;
}
.kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; }
.kpi .value { font-size: 24px; font-weight: 700; color: #F1F5F9; margin-top: 4px; }
.kpi .sub   { font-size: 12px; color: #64748B; margin-top: 2px; }

.badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
    color: white;
}
.badge-soft {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
    border: 1px solid var(--border); color: #CBD5E1; background: #0F172A;
}

.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 10px 14px; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(99,102,241,0.18), transparent);
    border-bottom: 2px solid var(--indigo) !important;
}

.regen-banner {
    background: linear-gradient(90deg, rgba(99,102,241,0.20), rgba(56,189,248,0.10));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 12px; padding: 10px 14px; margin: 8px 0 16px;
    color: #E0E7FF; font-weight: 500;
}
</style>
"""


def stepper(active: int, completed: int):
    labels = ["Hypothesis", "Quality", "Literature", "Plan"]
    parts = ['<div class="stepper">']
    for i, label in enumerate(labels, 1):
        cls = "step"
        if i <= completed:
            cls += " done"
            mark = "✓"
        elif i == active:
            cls += " active"
            mark = str(i)
        else:
            mark = str(i)
        parts.append(f'<div class="{cls}"><span class="dot">{mark}</span>{label}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def kpi_row(items: List[Dict[str, str]]):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(
            f'<div class="kpi"><div class="label">{it["label"]}</div>'
            f'<div class="value">{it["value"]}</div>'
            f'<div class="sub">{it.get("sub", "")}</div></div>',
            unsafe_allow_html=True,
        )


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

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


# ─── renderers ────────────────────────────────────────────────────────────────

def render_validation(v: Dict, on_use_suggested):
    score = float(v.get("score", 0.0))
    status = v.get("status", "ok")
    if status == "ok":
        st.success(f"Hypothesis quality: {score:.0%} — ready to proceed.")
        return
    st.markdown(
        f"<div class='card' style='background:rgba(245,158,11,0.10);border-color:rgba(245,158,11,0.35);"
        f"color:#FEF3C7;font-weight:600;'>Hypothesis quality: {score:.0%} — needs revision</div>",
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
            f"<div class='card' style='background:{color};color:white;font-weight:700;"
            f"text-align:center;font-size:18px;border-color:{color};'>"
            f"{novelty.upper()}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<div class='card'>Source: <span class='badge-soft'>{source}</span></div>",
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


def render_protocols(protocols: List[Dict]):
    if not protocols:
        return
    st.markdown("#### Related protocols")
    for i, pr in enumerate(protocols, 1):
        with st.container(border=True):
            link = pr.get("link") or ""
            status = pr.get("link_status", "unchecked")
            title = pr.get("title", "Untitled")
            if status == "ok" and link:
                badge = "<span class='badge' style='background:#10B981;'>✓ link verified</span>"
                label = f"**PR{i}.** {title} · [open]({link}) {badge}"
            elif status == "unavailable":
                badge = "<span class='badge' style='background:#9CA3AF;'>⚠ link unavailable</span>"
                label = f"**PR{i}.** {title} {badge}"
            else:
                badge = "<span class='badge-soft'>unchecked</span>"
                label = f"**PR{i}.** {title} {badge}"
            st.markdown(label, unsafe_allow_html=True)
            src = pr.get("source") or ""
            if src:
                st.caption(src)
            if pr.get("summary"):
                st.write(pr["summary"])


def render_kpis(plan: Dict):
    budget = plan.get("budget") or {}
    total = budget.get("total_usd", 0)
    materials = plan.get("materials") or []
    timeline = plan.get("timeline") or []
    protocols_used = plan.get("protocols_used") or []
    weeks_total = sum(float(t.get("duration_weeks", 0) or 0) for t in timeline)
    kpi_row([
        {"label": "Estimated budget", "value": f"${total:,.0f}",
         "sub": budget.get("notes", "")},
        {"label": "Materials",        "value": str(len(materials)),
         "sub": f"{sum(1 for m in materials if m.get('cost_source')=='catalog')} priced from catalog"},
        {"label": "Timeline",         "value": f"{weeks_total:g} wk",
         "sub": f"{len(timeline)} phases"},
        {"label": "Protocols cited",  "value": str(len(protocols_used)),
         "sub": f"{sum(1 for p in protocols_used if p.get('link_status')=='ok')} verified"},
    ])


def render_plan(plan: Dict):
    render_kpis(plan)
    tabs = st.tabs(["Protocol", "Materials", "Budget", "Timeline", "Validation", "Raw JSON"])

    with tabs[0]:
        for step in plan.get("protocol", []):
            with st.expander(
                f"Step {step.get('step', '?')} · {step.get('title', '')}  ({step.get('duration', '')})"
            ):
                st.write(step.get("description", ""))
                refs = step.get("references") or []
                if refs:
                    pretty = []
                    for r in refs:
                        if r == "link unavailable":
                            pretty.append("⚠ link unavailable")
                        elif r.startswith("http"):
                            pretty.append(f"[{r[:48]}…]({r})")
                        else:
                            pretty.append(f"`{r}`")
                    st.caption("references: " + " · ".join(pretty))

    with tabs[1]:
        materials = plan.get("materials", [])
        if materials:
            rows = []
            for m in materials:
                color, label = COST_SOURCE_BADGE.get(m.get("cost_source", "unknown"), COST_SOURCE_BADGE["unknown"])
                rows.append({
                    "name": m.get("name", ""),
                    "supplier": m.get("supplier", ""),
                    "catalog": m.get("catalog", ""),
                    "quantity": m.get("quantity", ""),
                    "cost": m.get("cost_display", "unknown"),
                    "source": label,
                    "category": m.get("category", "Other"),
                    "url": m.get("url", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("url", display_text="open"),
                    "cost": st.column_config.TextColumn("unit cost"),
                    "source": st.column_config.TextColumn("price source"),
                },
            )
            unknown = sum(1 for m in materials if m.get("cost_source") == "unknown")
            estimated = sum(1 for m in materials if m.get("cost_source") == "estimated")
            if unknown or estimated:
                st.caption(
                    f"💡 {estimated} estimated by category heuristic, "
                    f"{unknown} unknown (excluded from total). "
                    "Catalog prices are real; estimates use $50–500 reagents/consumables, "
                    "$100–1000 kits, $1000–10000 equipment."
                )
        else:
            st.info("No materials listed.")

    with tabs[2]:
        budget = plan.get("budget") or {}
        total = budget.get("total_usd", 0)
        line_items = budget.get("line_items", [])
        cols = st.columns([1, 2])
        with cols[0]:
            st.metric("Estimated total", f"${total:,.0f}")
            if budget.get("notes"):
                st.caption(budget["notes"])
        with cols[1]:
            if line_items:
                df = pd.DataFrame(line_items)
                fig = px.pie(df, names="category", values="amount_usd", hole=0.55,
                             color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=260,
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(font=dict(color="#E5E7EB")))
                st.plotly_chart(fig, use_container_width=True)
        if line_items:
            df = pd.DataFrame(line_items)
            df["amount_usd"] = df["amount_usd"].map(lambda v: f"${v:,.0f}")
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
                    "Phase": ph["phase"], "Start": start,
                    "Finish": end, "Duration (weeks)": duration,
                })
            df = pd.DataFrame(rows)
            fig = px.bar(
                df, x="Duration (weeks)", y="Phase", base="Start", orientation="h",
                hover_data=["Start", "Finish"],
                color_discrete_sequence=["#6366F1"],
            )
            fig.update_layout(
                xaxis_title="Weeks from start",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10, b=0, l=0, r=0),
                height=max(220, 60 * len(rows)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E5E7EB"),
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


def _collect_feedback_items() -> List[Dict[str, Any]]:
    sections = ["protocol", "materials", "budget", "timeline", "validation", "overall"]
    items: List[Dict[str, Any]] = []
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
                "section": sec, "rating": rating,
                "correction": correction, "comment": "",
            })
    comment = st.text_area("Overall comment (optional)",
                            key="overall_comment", height=80)
    if comment.strip():
        items.append({"section": "overall", "rating": None,
                      "correction": "", "comment": comment})
    return items


def render_review_and_regenerate(backend: str, hypothesis: str,
                                  parsed: Optional[Dict], qc: Optional[Dict],
                                  plan: Dict):
    st.markdown("---")
    with st.expander("📝 Review this plan & regenerate with your feedback", expanded=False):
        items = _collect_feedback_items()

        c1, c2 = st.columns([3, 1])
        with c1:
            regen = st.button(
                "🔄 Submit feedback & regenerate plan",
                type="primary", key="regen_btn", use_container_width=True,
            )
        with c2:
            store_only = st.button(
                "Save only", key="save_only_btn", use_container_width=True,
            )

        if (regen or store_only) and not items:
            st.warning("No changes captured — adjust a rating or write a correction first.")
        elif store_only:
            try:
                resp = post(backend, "/feedback", {
                    "hypothesis": hypothesis, "parsed": parsed,
                    "plan": plan, "items": items,
                })
                st.toast(f"Feedback stored ({resp.get('stored', 0)} items)", icon="✅")
            except Exception as e:
                st.error(f"Feedback submission failed: {e}")
        elif regen:
            try:
                with st.spinner("Regenerating plan with your feedback…"):
                    post(backend, "/feedback", {
                        "hypothesis": hypothesis, "parsed": parsed,
                        "plan": plan, "items": items,
                    })
                    new_plan = post(
                        backend, "/generate_plan",
                        {
                            "hypothesis": hypothesis,
                            "parsed": parsed,
                            "papers": (qc or {}).get("papers", []),
                            "inline_feedback": items,
                            "previous_plan": plan,
                        },
                        timeout=240,
                    )
                st.session_state.last_run = {
                    "parsed": parsed, "qc": qc, "plan": new_plan,
                    "version": st.session_state.last_run.get("version", 1) + 1
                              if st.session_state.last_run else 2,
                    "feedback_count": len(items),
                }
                st.rerun()
            except Exception as e:
                st.error(f"Regeneration failed: {e}")


def render_pdf_button(backend: str, hypothesis: str, parsed: Optional[Dict],
                      qc: Optional[Dict], plan: Dict):
    try:
        pdf_bytes = post_bytes(backend, "/export_pdf", {
            "hypothesis": hypothesis, "parsed": parsed,
            "qc": qc, "plan": plan,
        }, timeout=60)
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return
    st.download_button(
        label="📄 Download experiment plan (PDF)",
        data=pdf_bytes, file_name="experiment-plan.pdf",
        mime="application/pdf", use_container_width=True,
    )


# ─── Session state ────────────────────────────────────────────────────────────

if "hypothesis" not in st.session_state:
    st.session_state.hypothesis = list(SAMPLES.values())[0]
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # {parsed, qc, plan, version?, feedback_count?}


def _on_sample_change():
    st.session_state.hypothesis = SAMPLES[st.session_state.sample_label]
    st.session_state.last_run = None


def _use_suggested(text: str):
    st.session_state.hypothesis = text
    st.session_state.last_run = None


# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='app-hero'><h1>🧪 The AI Scientist</h1>"
    "<p>Hypothesis → Quality check → Literature QC → Operationally complete experiment plan</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Settings")
    backend = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    show_parsed = st.checkbox("Show parsed hypothesis", value=False)
    skip_validation = st.checkbox("Skip hypothesis quality gate", value=False)
    st.markdown("---")
    st.caption("Plans now: ✓ link-checked  ✓ priced (catalog/web/heuristic)  ✓ feedback-aware regenerate")

last_run = st.session_state.last_run
completed = 4 if last_run else 0
stepper(active=1 if not last_run else 4, completed=completed)

st.markdown("##### 1. Choose or write a hypothesis")
try:
    st.segmented_control(
        "Sample hypotheses",
        options=list(SAMPLES.keys()),
        key="sample_label",
        on_change=_on_sample_change,
        label_visibility="collapsed",
    )
except (AttributeError, TypeError):
    st.radio(
        "Sample hypotheses", options=list(SAMPLES.keys()), horizontal=True,
        label_visibility="collapsed", key="sample_label", on_change=_on_sample_change,
    )

hypothesis = st.text_area(
    "Hypothesis", height=120, label_visibility="collapsed", key="hypothesis",
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
            "version": 1, "feedback_count": 0,
        }

        st.markdown("---")
        st.markdown("##### 3. Experiment Plan")
        render_protocols(plan.get("protocols_used") or [])
        render_plan(plan)
        render_pdf_button(backend, hypothesis, parsed_payload, qc, plan)
        render_review_and_regenerate(backend, hypothesis, parsed_payload, qc, plan)
elif run:
    st.warning("Hypothesis cannot be empty.")
elif st.session_state.last_run:
    last = st.session_state.last_run
    st.markdown("---")
    if (last.get("version") or 1) > 1:
        st.markdown(
            f"<div class='regen-banner'>🔄 Plan v{last['version']} — incorporates your "
            f"feedback ({last.get('feedback_count', 0)} items)</div>",
            unsafe_allow_html=True,
        )
    st.markdown("##### Last result")
    render_qc(last["qc"])
    render_protocols(last["plan"].get("protocols_used") or [])
    render_plan(last["plan"])
    render_pdf_button(backend, st.session_state.hypothesis,
                      last["parsed"], last["qc"], last["plan"])
    render_review_and_regenerate(backend, st.session_state.hypothesis,
                                  last["parsed"], last["qc"], last["plan"])
