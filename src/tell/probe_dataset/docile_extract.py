"""Extracts the deterministic, purpose-built `DocumentFacts` this corpus
needs from an official DocILE train-split annotation file, directly from
`data/docile/annotations/<docid>.json` -- the same raw field-extraction
records `scripts/build_pilot_cohort.py` reads, using the identical
fieldtype-name mapping it established (vendor_name, document_id ->
invoice_number, date_issue -> invoice_date, amount_due /
amount_total_gross -> payable amount, line_item_description -> line
items).

This is a deliberately narrower representation than
`tell.evaluation.scenario.DocileInvoiceRecord` (no bbox/page geometry,
no rendered-image bookkeeping, no Pydantic tool-view projection): the
probe corpus never executes `tell.agent.tools.read_invoice` or any other
existing tool function, and never needs bbox data, so building the full
scenario schema here would be unused ceremony. Every text value is still
copied verbatim from the official annotation -- nothing here invents or
overwrites a DocILE-derived business fact (see `TEXT_INVENTED = False`
sentinel on every DocumentFacts instance's provenance, for tests to
check).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

DOCILE_ANNOTATIONS_DIR = Path("/home/hp5/tell/data/docile/annotations")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_whitespace(text: str) -> str:
    """Collapses embedded newlines/repeated whitespace for use inside
    synthesized prose (a few DocILE vendor names span multiple lines in
    the source annotation). The raw, unmodified text is always preserved
    separately on `DocumentFacts.vendor_name_raw`."""
    return re.sub(r"\s+", " ", text or "").strip()


def _occurrences(field_extractions: list[dict], fieldtype: str) -> list[dict]:
    return [f for f in field_extractions if f["fieldtype"] == fieldtype]


def _first_text(field_extractions: list[dict], fieldtype: str) -> str | None:
    hits = _occurrences(field_extractions, fieldtype)
    return hits[0]["text"] if hits else None


def _parse_amount_minor_units(amount_text: str) -> int:
    cleaned = re.sub(r"[^0-9.\-]", "", amount_text or "")
    try:
        value = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        raise ValueError(f"Cannot parse payable amount {amount_text!r}")
    minor = int((value * 100).to_integral_value())
    if minor <= 0:
        raise ValueError(f"Payable amount must be positive, got {amount_text!r} -> {minor}")
    return minor


@dataclass(frozen=True)
class LineItemSummary:
    line_item_id: int
    description: str | None


@dataclass(frozen=True)
class DocumentFacts:
    """Deterministic facts for one selected DocILE document. Every
    string field traces verbatim to the official annotation file at
    `annotation_path` (never a DocILE validation-split document -- the
    caller is responsible for only ever extracting train-split docids,
    exactly as `select_probe_corpus_documents.py` selected them)."""

    docid: str
    split: str
    annotation_path: str
    annotation_sha256: str
    vendor_name_raw: str
    vendor_name_display: str
    invoice_number: str
    invoice_date: str
    due_date: str | None
    currency: str
    payable_amount_text: str
    payable_amount_source_field: str
    payable_amount_minor_units: int
    line_items: tuple[LineItemSummary, ...]
    line_item_count: int
    vendor_address: str | None
    vendor_email: str | None
    document_type: str | None
    cluster_id: int
    page_count: int
    ocr_word_count: int


def extract_document_facts(docid: str, split: str) -> DocumentFacts:
    annotation_path = DOCILE_ANNOTATIONS_DIR / f"{docid}.json"
    ann = json.loads(annotation_path.read_text())
    fields = ann["field_extractions"]
    meta = ann["metadata"]

    vendor_name_raw = _first_text(fields, "vendor_name")
    invoice_number = _first_text(fields, "document_id")
    invoice_date = _first_text(fields, "date_issue")
    due_date = _first_text(fields, "date_due")
    vendor_address = _first_text(fields, "vendor_address")
    vendor_email = _first_text(fields, "vendor_email")

    if not vendor_name_raw or not invoice_number or not invoice_date:
        raise ValueError(f"{docid}: missing an unambiguous vendor name, invoice number, or invoice date")

    amount_due = _first_text(fields, "amount_due")
    amount_total_gross = _first_text(fields, "amount_total_gross")
    if amount_due:
        payable_amount_text, payable_amount_source_field = amount_due, "amount_due"
    elif amount_total_gross:
        payable_amount_text, payable_amount_source_field = amount_total_gross, "amount_total_gross"
    else:
        raise ValueError(f"{docid}: no amount_due or amount_total_gross field")

    currency = (meta.get("currency") or "").lower()
    if not currency:
        raise ValueError(f"{docid}: missing currency metadata")

    line_item_ids = sorted({li["line_item_id"] for li in ann["line_item_extractions"]})
    line_items: list[LineItemSummary] = []
    for lid in line_item_ids:
        desc_hits = [
            li["text"] for li in ann["line_item_extractions"] if li["line_item_id"] == lid and li["fieldtype"] == "line_item_description"
        ]
        line_items.append(LineItemSummary(line_item_id=lid, description=desc_hits[0] if desc_hits else None))

    return DocumentFacts(
        docid=docid,
        split=split,
        annotation_path=str(annotation_path),
        annotation_sha256=_sha256_file(annotation_path),
        vendor_name_raw=vendor_name_raw,
        vendor_name_display=normalize_whitespace(vendor_name_raw),
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        currency=currency,
        payable_amount_text=payable_amount_text,
        payable_amount_source_field=payable_amount_source_field,
        payable_amount_minor_units=_parse_amount_minor_units(payable_amount_text),
        line_items=tuple(line_items),
        line_item_count=len(line_items),
        vendor_address=vendor_address,
        vendor_email=vendor_email,
        document_type=meta.get("document_type"),
        cluster_id=meta.get("cluster_id"),
        page_count=int(meta.get("page_count", 1)),
        ocr_word_count=0,  # not needed downstream; recorded on the selection manifest instead
    )


__all__ = ["DocumentFacts", "LineItemSummary", "extract_document_facts", "normalize_whitespace"]
