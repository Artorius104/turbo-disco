import datetime as dt
import io
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

from .schemas import ExperimentPlan, LiteratureQCResponse, ParsedHypothesis


_COLORS = {
    "header": colors.HexColor("#1f2937"),
    "table_header_bg": colors.HexColor("#374151"),
    "table_row_alt": colors.HexColor("#f3f4f6"),
    "novelty_ok": colors.HexColor("#1f9d55"),
    "novelty_warn": colors.HexColor("#d97706"),
    "novelty_bad": colors.HexColor("#dc2626"),
    "muted": colors.HexColor("#6b7280"),
}

_NOVELTY_COLOR = {
    "not found": _COLORS["novelty_ok"],
    "similar work exists": _COLORS["novelty_warn"],
    "exact match found": _COLORS["novelty_bad"],
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"],
                                 fontSize=22, leading=26, spaceAfter=12,
                                 textColor=_COLORS["header"]),
        "h2": ParagraphStyle("H2", parent=base["Heading2"],
                              fontSize=14, leading=18, spaceBefore=12, spaceAfter=6,
                              textColor=_COLORS["header"]),
        "h3": ParagraphStyle("H3", parent=base["Heading3"],
                              fontSize=11, leading=14, spaceBefore=8, spaceAfter=2,
                              textColor=_COLORS["header"]),
        "body": ParagraphStyle("Body", parent=base["BodyText"],
                                fontSize=9.5, leading=13, spaceAfter=4),
        "small": ParagraphStyle("Small", parent=base["BodyText"],
                                 fontSize=8, leading=11,
                                 textColor=_COLORS["muted"]),
        "novelty": ParagraphStyle("Novelty", parent=base["BodyText"],
                                   fontSize=11, leading=14, alignment=1,
                                   textColor=colors.white),
    }


def _header_cell(text: str, st) -> Paragraph:
    return Paragraph(f'<font color="white"><b>{text}</b></font>', st["body"])


def _link(text: str, url: str, st) -> Paragraph:
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;")
    if url:
        return Paragraph(f'<link href="{url}"><font color="#2563eb">{safe}</font></link>', st["body"])
    return Paragraph(safe, st["body"])


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wrap_doc(elements, output: io.BytesIO):
    doc = SimpleDocTemplate(
        output, pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Experiment Plan",
    )
    doc.build(elements)


def _table(rows: list[list], col_widths: list[float], header: bool = True):
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
         [colors.white, _COLORS["table_row_alt"]]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
    ]
    if header:
        style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), _COLORS["table_header_bg"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style))
    return t


def render_pdf(hypothesis: str, parsed: Optional[ParsedHypothesis],
               qc: Optional[LiteratureQCResponse],
               plan: ExperimentPlan) -> bytes:
    st = _styles()
    elements: List = []

    elements.append(Paragraph("Experiment Plan", st["title"]))
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append(Paragraph(f"Generated {ts}", st["small"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Hypothesis</b>", st["h3"]))
    elements.append(Paragraph(_esc(hypothesis), st["body"]))

    if parsed:
        elements.append(Paragraph("Parsed hypothesis", st["h2"]))
        rows = [
            [_header_cell("Field", st), _header_cell("Value", st)],
            [Paragraph("Intervention", st["body"]), Paragraph(_esc(parsed.intervention), st["body"])],
            [Paragraph("Outcome", st["body"]), Paragraph(_esc(parsed.outcome), st["body"])],
            [Paragraph("Mechanism", st["body"]), Paragraph(_esc(parsed.mechanism), st["body"])],
            [Paragraph("System", st["body"]), Paragraph(_esc(parsed.system), st["body"])],
            [Paragraph("Measurement", st["body"]), Paragraph(_esc(parsed.measurement), st["body"])],
            [Paragraph("Domain", st["body"]), Paragraph(_esc(parsed.domain), st["body"])],
            [Paragraph("Keywords", st["body"]), Paragraph(_esc(", ".join(parsed.keywords)), st["body"])],
        ]
        elements.append(_table(rows, [1.4 * inch, 5.6 * inch]))

    if qc:
        elements.append(Paragraph("Literature QC", st["h2"]))
        novelty = qc.novelty
        bg = _NOVELTY_COLOR.get(novelty, _COLORS["muted"])
        novelty_table = Table([[Paragraph(novelty.upper(), st["novelty"])]],
                              colWidths=[7.0 * inch], rowHeights=[0.32 * inch])
        novelty_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(novelty_table)
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"Source: {qc.source}", st["small"]))
        elements.append(Spacer(1, 6))
        for i, p in enumerate(qc.papers, 1):
            elements.append(Paragraph(
                f"<b>P{i}.</b> {_esc(p.title)} "
                f'<link href="{p.link}"><font color="#2563eb">[link]</font></link>',
                st["body"],
            ))
            elements.append(Paragraph(
                f"{_esc(p.authors)} · {_esc(p.year)} · similarity {p.similarity_score:.2f}",
                st["small"],
            ))

    if plan.protocols_used:
        elements.append(Paragraph("Related protocols", st["h2"]))
        for i, pr in enumerate(plan.protocols_used, 1):
            elements.append(Paragraph(
                f"<b>PR{i}.</b> {_esc(pr.title)} <font color='#6b7280'>({_esc(pr.source)})</font> "
                f'<link href="{pr.link}"><font color="#2563eb">[link]</font></link>',
                st["body"],
            ))
            if pr.summary:
                elements.append(Paragraph(_esc(pr.summary), st["small"]))

    elements.append(PageBreak())

    elements.append(Paragraph("Protocol", st["h2"]))
    for step in plan.protocol:
        elements.append(Paragraph(
            f"<b>Step {step.step}. {_esc(step.title)}</b> "
            f"<font color='#6b7280'>({_esc(step.duration)})</font>",
            st["h3"],
        ))
        elements.append(Paragraph(_esc(step.description), st["body"]))
        if step.references:
            elements.append(Paragraph(
                "References: " + _esc(", ".join(step.references)), st["small"]
            ))

    elements.append(Paragraph("Materials", st["h2"]))
    rows = [[
        _header_cell("Name", st), _header_cell("Supplier", st),
        _header_cell("Catalog", st), _header_cell("Qty", st),
        _header_cell("Unit $", st), _header_cell("URL", st),
    ]]
    for m in plan.materials:
        rows.append([
            Paragraph(_esc(m.name), st["body"]),
            Paragraph(_esc(m.supplier), st["body"]),
            Paragraph(_esc(m.catalog), st["body"]),
            Paragraph(_esc(m.quantity), st["body"]),
            Paragraph(f"${m.unit_cost_usd:,.0f}", st["body"]),
            _link("link", m.url, st) if m.url else Paragraph("—", st["body"]),
        ])
    elements.append(_table(rows, [2.0 * inch, 1.0 * inch, 0.9 * inch,
                                   0.9 * inch, 0.6 * inch, 0.6 * inch]))

    elements.append(Paragraph("Budget", st["h2"]))
    elements.append(Paragraph(f"<b>Total: ${plan.budget.total_usd:,.0f}</b>", st["body"]))
    rows = [[_header_cell("Category", st), _header_cell("Amount (USD)", st)]]
    for li in plan.budget.line_items:
        rows.append([
            Paragraph(_esc(li.category), st["body"]),
            Paragraph(f"${li.amount_usd:,.0f}", st["body"]),
        ])
    elements.append(_table(rows, [4.5 * inch, 2.5 * inch]))

    elements.append(Paragraph("Timeline", st["h2"]))
    rows = [[_header_cell("Phase", st), _header_cell("Duration (weeks)", st),
             _header_cell("Depends on", st)]]
    for ph in plan.timeline:
        rows.append([
            Paragraph(_esc(ph.phase), st["body"]),
            Paragraph(f"{ph.duration_weeks:g}", st["body"]),
            Paragraph(_esc(", ".join(ph.depends_on)) or "—", st["body"]),
        ])
    elements.append(_table(rows, [3.0 * inch, 1.5 * inch, 2.5 * inch]))

    elements.append(Paragraph("Validation", st["h2"]))
    elements.append(Paragraph(f"<b>Primary metric:</b> {_esc(plan.validation.primary_metric)}", st["body"]))
    elements.append(Paragraph(f"<b>Success criteria:</b> {_esc(plan.validation.success_criteria)}", st["body"]))
    if plan.validation.controls:
        elements.append(Paragraph("<b>Controls:</b>", st["body"]))
        for c in plan.validation.controls:
            elements.append(Paragraph(f"• {_esc(c)}", st["body"]))

    if plan.references_used or plan.protocols_used:
        elements.append(Paragraph("References", st["h2"]))
        for r in plan.references_used:
            elements.append(Paragraph(f"• {_esc(r)}", st["body"]))
        for i, pr in enumerate(plan.protocols_used, 1):
            elements.append(Paragraph(
                f"• PR{i}: {_esc(pr.title)} ({_esc(pr.source)}) — {pr.link}",
                st["body"],
            ))

    output = io.BytesIO()
    _wrap_doc(elements, output)
    return output.getvalue()
