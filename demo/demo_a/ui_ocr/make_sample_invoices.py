#!/usr/bin/env python3
"""Write three synthetic, clearly fictional sample invoices into the demo inbox (dev/demo helper; needs reportlab).
Deterministic bytes (reportlab invariant mode). Does not touch anything outside the target directory."""
import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "runtime" / "inbox"


def pdf(name, lines=(), shapes=False):
    c = canvas.Canvas(str(OUT / name), pagesize=A4, invariant=1)
    y = 800
    for ln in lines:
        c.setFont("Helvetica-Bold" if ln.startswith("#") else "Helvetica", 14 if ln.startswith("#") else 11)
        c.drawString(60, y, ln.lstrip("# ")); y -= 22
    if shapes:  # image-like page with no embedded text
        for i in range(6):
            c.rect(60, 700 - i * 60, 470, 40, fill=(i % 2 == 0))
    c.showPage(); c.save()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdf("sample_complete_invoice.pdf", ["# INVOICE", "Supplier: Fictional Office Supplies Ltd", "Invoice No: FOS-2026-0417",
        "Invoice Date: 2026-09-02", "Due Date: 2026-10-02", "", "Description                      Qty     Amount",
        "Printer paper (A4)                10      120.00", "Toner cartridge                    2      310.50", "",
        "Total Due: EUR 430.50", "", "Beneficiary: Fictional Office Supplies Ltd", "IBAN: DE89 3704 0044 0532 0130 00"])
    pdf("sample_missing_fields.pdf", ["# INVOICE", "Supplier: Imaginary Freight Co", "Invoice No: IFC-88213", "Date: 12 September 2026",
        "", "Total: 1,250.00", "", "Thank you for your business."])
    pdf("sample_scanned_no_text.pdf", shapes=True)
    print("wrote 3 sample PDFs to", OUT)


if __name__ == "__main__":
    main()
