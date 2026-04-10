"""
LexGuard — PDF Audit Report Generator
=======================================
Generates professional PDF risk audit reports from extracted clause data.
Uses fpdf2 (pure Python, no system dependencies).

Usage:
    from report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(filename, clauses, brief)
"""

from __future__ import annotations

import io
import datetime
from fpdf import FPDF


class LexGuardReport(FPDF):
    """Custom PDF with header/footer branding."""

    def __init__(self, doc_name: str = ""):
        super().__init__()
        self.doc_name = doc_name

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "LexGuard - Contract Risk Audit Report", align="L")
        self.ln(4)
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _safe_text(text: str) -> str:
    """Strip characters that fpdf2 can't encode in latin-1."""
    if not text:
        return ""
    # Replace common problematic unicode with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u00a0": " ",
        "\u2022": "*", "\u2023": ">", "\u25cf": "*", "\u2192": "->",
        "\u2714": "[Y]", "\u2716": "[X]", "\u26a0": "[!]",
        "\u2705": "[OK]", "\u274c": "[X]", "\u2139": "[i]",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Final fallback: encode to latin-1, replacing unknowns
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(
    filename: str,
    clauses: dict | None = None,
    brief: dict | None = None,
) -> bytes:
    """
    Generate a professional PDF audit report.

    Args:
        filename: Name of the analyzed document.
        clauses: Dict of clause_name -> info dict (from cross-validation).
        brief: Dict of metadata fields (from extract_contract_brief).

    Returns:
        PDF file content as bytes.
    """
    pdf = LexGuardReport(doc_name=filename)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title ──
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 12, "Contract Risk Audit Report", ln=True, align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, _safe_text(f"Document: {filename}"), ln=True, align="C")
    pdf.cell(
        0, 8,
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        ln=True, align="C",
    )
    pdf.ln(6)

    # ── Contract Brief ──
    if brief:
        _section_header(pdf, "Contract Summary")
        for field, value in brief.items():
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 0, 0)
            label = _safe_text(f"{field}:")
            pdf.cell(0, 7, label, ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 6, _safe_text(f"  {str(value)}"))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        pdf.ln(4)

    # ── Risk Summary ──
    if clauses:
        detected = {k: v for k, v in clauses.items()
                    if isinstance(v, dict) and (
                        v.get("llm_detected") or v.get("bert_detected") or v.get("detected")
                    )}
        not_found = len(clauses) - len(detected)

        _section_header(pdf, "Risk Summary")

        # Stats bar
        high = sum(1 for v in detected.values() if v.get("risk_level") == "High")
        medium = sum(1 for v in detected.values() if v.get("risk_level") == "Medium")
        low = sum(1 for v in detected.values() if v.get("risk_level") == "Low")
        verified = sum(1 for v in detected.values() if v.get("agreement") == "agreed")

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _safe_text(
            f"Total Detected: {len(detected)} | "
            f"High: {high} | Medium: {medium} | Low: {low} | "
            f"Verified by Both Models: {verified} | "
            f"Not Found: {not_found}"
        ), ln=True)
        pdf.ln(4)

        # ── Risk Table ──
        _section_header(pdf, "Detected Risk Clauses")

        # Table header
        col_widths = [8, 52, 22, 30, 22, 50]
        headers = ["#", "Clause", "Risk", "Verification", "BERT", "Section"]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(0, 51, 102)
        pdf.set_text_color(255, 255, 255)
        for i, (header, w) in enumerate(zip(headers, col_widths)):
            pdf.cell(w, 7, header, border=1, fill=True, align="C")
        pdf.ln()

        # Table rows
        pdf.set_text_color(0, 0, 0)
        row_num = 0
        for clause_name, info in detected.items():
            row_num += 1
            risk_level = info.get("risk_level", "Unknown")
            section = info.get("section", "-")
            agreement = info.get("agreement", "")
            bert_conf = info.get("bert_confidence", 0)

            # Verification text
            verify_map = {
                "agreed": "Verified",
                "disagreement": "REVIEW",
                "llm_only": "LLM-Only",
                "bert_only": "BERT-Only",
            }
            verify_text = verify_map.get(agreement, "-")
            conf_str = f"{bert_conf:.0%}" if bert_conf > 0 else "-"

            # Row colors based on risk
            if risk_level == "High":
                pdf.set_fill_color(255, 230, 230)
            elif risk_level == "Medium":
                pdf.set_fill_color(255, 248, 220)
            else:
                pdf.set_fill_color(230, 255, 230)

            pdf.set_font("Helvetica", "", 8)
            pdf.cell(col_widths[0], 6, str(row_num), border=1, fill=True, align="C")
            pdf.cell(col_widths[1], 6, _safe_text(clause_name[:30]), border=1, fill=True)
            pdf.cell(col_widths[2], 6, risk_level, border=1, fill=True, align="C")
            pdf.cell(col_widths[3], 6, verify_text, border=1, fill=True, align="C")
            pdf.cell(col_widths[4], 6, conf_str, border=1, fill=True, align="C")
            pdf.cell(col_widths[5], 6, _safe_text(str(section)[:28]), border=1, fill=True)
            pdf.ln()

        pdf.ln(6)

        # ── Detailed Findings ──
        _section_header(pdf, "Detailed Findings")

        for clause_name, info in detected.items():
            risk_level = info.get("risk_level", "Unknown")
            excerpt = info.get("excerpt", "")
            section = info.get("section", "-")
            agreement = info.get("agreement", "")
            bert_excerpt = info.get("bert_excerpt", "")
            bert_conf = info.get("bert_confidence", 0)

            # Clause header
            pdf.set_font("Helvetica", "B", 10)
            if risk_level == "High":
                pdf.set_text_color(180, 0, 0)
            elif risk_level == "Medium":
                pdf.set_text_color(180, 130, 0)
            else:
                pdf.set_text_color(0, 130, 0)

            pdf.cell(0, 7, _safe_text(f"{clause_name} [{risk_level} Risk]"), ln=True)
            pdf.set_text_color(0, 0, 0)

            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5, _safe_text(f"Section: {section}"), ln=True)

            verify_map = {
                "agreed": "Both LLM and BERT agree",
                "disagreement": "BERT detected, LLM missed - NEEDS REVIEW",
                "llm_only": "LLM detected only (BERT context limit)",
            }
            verify_text = f"Verification: {verify_map.get(agreement, '-')}"
            if bert_conf > 0:
                verify_text += f" | BERT Confidence: {bert_conf:.2%}"
            pdf.cell(0, 5, _safe_text(verify_text), ln=True)

            if excerpt:
                pdf.ln(2)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(60, 60, 60)
                pdf.set_x(10)  # Reset to left margin
                excerpt_text = _safe_text(f'LLM: "{str(excerpt)[:400]}"')
                pdf.multi_cell(0, 5, excerpt_text)
                pdf.set_text_color(0, 0, 0)

            if bert_excerpt and agreement == "agreed":
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(60, 60, 120)
                pdf.set_x(10)  # Reset to left margin
                bert_text = _safe_text(f'BERT: "{str(bert_excerpt)[:250]}"')
                pdf.multi_cell(0, 5, bert_text)
                pdf.set_text_color(0, 0, 0)

            pdf.ln(4)

    # ── Disclaimer ──
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, _safe_text(
        "Disclaimer: This report was generated by LexGuard, an AI-powered contract "
        "risk auditing tool. It is intended for informational purposes only and does "
        "not constitute legal advice. All findings should be reviewed by a qualified "
        "legal professional before making any decisions."
    ))

    return pdf.output()


def _section_header(pdf: FPDF, title: str):
    """Draw a styled section header."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_draw_color(0, 102, 204)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)
