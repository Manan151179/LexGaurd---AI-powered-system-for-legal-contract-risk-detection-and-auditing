"""
Generate Fixtures — Synthetic Dummy Contract PDF
==================================================
Creates a small 2-page PDF containing synthetic legal text with known
keywords for smoke-testing the LexGuard pipeline.

Usage:
    python tests/generate_fixtures.py
"""

import os
from pathlib import Path

from fpdf import FPDF


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Synthetic legal clauses with known risk keywords
PAGE_1_TEXT = (
    "RESIDENTIAL LEASE AGREEMENT\n\n"
    "This Residential Lease Agreement (\"Agreement\") is entered into as of "
    "January 1, 2025, by and between Landlord Corp (\"Landlord\") and "
    "Jane Doe (\"Tenant\").\n\n"
    "1. PREMISES: The Landlord agrees to rent to the Tenant the property "
    "located at 123 Main Street, Suite 4B, Springfield, IL 62701.\n\n"
    "2. TERM: The lease term shall commence on February 1, 2025 and "
    "terminate on January 31, 2026.\n\n"
    "3. RENT: Tenant shall pay a monthly rent of $1,500.00, due on the "
    "first day of each calendar month.\n\n"
    "4. SECURITY DEPOSIT: Tenant shall deposit $3,000.00 as a security "
    "deposit. This deposit shall be returned within 30 days of lease "
    "termination, less any deductions for damages."
)

PAGE_2_TEXT = (
    "5. INDEMNIFICATION: Tenant shall indemnify and hold harmless the "
    "Landlord from and against any and all claims, damages, losses, and "
    "expenses arising from Tenant's use of the premises.\n\n"
    "6. TERMINATION: Either party may pursue immediate termination of this "
    "Agreement upon material breach by the other party. Written notice of "
    "at least 30 days is required.\n\n"
    "7. PENALTIES: Late payment shall incur a penalty of 5% of the monthly "
    "rent for each day past the due date.\n\n"
    "8. GOVERNING LAW: This Agreement shall be governed by the laws of "
    "the State of Illinois.\n\n"
    "IN WITNESS WHEREOF, the parties have executed this Agreement as of "
    "the date first written above.\n\n"
    "Landlord: _________________________\n"
    "Tenant:  _________________________"
)


def generate_dummy_pdf() -> Path:
    """Generate a 2-page dummy contract PDF and return its path."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, PAGE_1_TEXT)

    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, PAGE_2_TEXT)

    out_path = FIXTURES_DIR / "dummy_contract.pdf"
    pdf.output(str(out_path))
    print(f"✅ Dummy contract PDF generated: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_dummy_pdf()
