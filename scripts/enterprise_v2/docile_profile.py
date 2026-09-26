"""Per-document DocILE profiling for enterprise corpus v2.

Reads ONLY `data/docile/train.json` as a split list (never val.json or
trainval.json) plus the per-document annotation/OCR JSON for train
docids. Annotation interpretation is reused, not re-implemented:

  - OCR-under-bbox agreement: `select_clean_candidate.similarity`,
    `ocr_words_for_bbox`
    (the validated helpers behind results/dataset_inspection/
    clean_candidate_scores.json).
  - Vendor-name whitespace normalization:
    `tell.probe_dataset.docile_extract.normalize_whitespace`.
  - Fieldtype-to-business-field mapping (vendor_name, document_id ->
    invoice number, date_issue, amount_due / amount_total_gross -> payable
    amount, line_item_description) follows
    `tell.probe_dataset.docile_extract.extract_document_facts`.

`select_clean_candidate` imports Pillow at module import time for its
page-rendering helpers only; Pillow is not installed in the project venv.
`_import_select_clean_candidate` installs an inert placeholder `PIL`
module *only if* Pillow is missing, so the text/OCR helpers can be reused
without duplicating them. No rendering helper is ever called here.

Everything returned is a plain dict of observed facts; no value is
invented.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
DOCILE_ROOT = REPO_ROOT / "data" / "docile"
TRAIN_LIST_PATH = DOCILE_ROOT / "train.json"  # the only split list this module opens

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tell.payment.ledger import SUPPORTED_CURRENCIES  # noqa: E402
from tell.probe_dataset.docile_extract import normalize_whitespace  # noqa: E402


def _import_select_clean_candidate():
    try:
        import select_clean_candidate as scc  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "PIL":
            raise
        pil = types.ModuleType("PIL")
        for sub in ("Image", "ImageDraw", "ImageFont"):
            mod = types.ModuleType(f"PIL.{sub}")
            setattr(pil, sub, mod)
            sys.modules[f"PIL.{sub}"] = mod
        sys.modules["PIL"] = pil
        try:
            import select_clean_candidate as scc  # noqa: F811
        finally:
            # The placeholder is only needed while that module binds its
            # names; leaving it registered would make other libraries
            # (e.g. transformers' find_spec probe) believe Pillow exists.
            for name in ("PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont"):
                sys.modules.pop(name, None)
    return scc


scc = _import_select_clean_candidate()

# Document types an AP agent would legitimately be asked to pay. Excluded:
# purchase_order (buyer-issued, not a payable), receipt (already paid),
# credit_note (a credit, not a payable).
PAYABLE_DOCUMENT_TYPES = ("tax_invoice", "utility_bill", "proforma", "debit_note", "order", "sales_order")
EXCLUDED_DOCUMENT_TYPES = {
    "purchase_order": "buyer-issued purchase order, not a supplier payable",
    "receipt": "receipt of an already-completed payment",
    "credit_note": "credit note (negative payable)",
}

PAYMENT_DESTINATION_FIELDTYPES = ("account_num", "bank_num", "iban", "bic")
SUPPORTED_LEDGER_CURRENCIES = tuple(sorted(SUPPORTED_CURRENCIES))

_VENDOR_SUFFIXES = {
    "INC", "LLC", "LTD", "CO", "CORP", "CORPORATION", "COMPANY", "LIMITED", "THE", "GMBH", "PLC", "LLP", "SA", "AG",
    "INCORPORATED", "LP", "PTY", "BV", "NV", "SRL", "SPA",
}

_AMOUNT_OK_RE = re.compile(r"^(\d{1,3}(,\d{3})+|\d+)(\.\d{1,2})?$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train_docids() -> list[str]:
    return sorted(json.loads(TRAIN_LIST_PATH.read_text()))


def vendor_key(name: str | None) -> str:
    """Conservative vendor identity key: upper-case, punctuation removed,
    common corporate suffixes dropped -- so "ACME, Inc." and "Acme Inc"
    collide (stricter isolation), never the reverse."""
    if not name:
        return ""
    t = re.sub(r"[^A-Z0-9 ]", " ", normalize_whitespace(name).upper())
    return " ".join(w for w in t.split() if w not in _VENDOR_SUFFIXES)


def parse_payable_minor_units(text: str | None) -> int | None:
    """Strict, deterministic payable-amount parser for OBSERVED text.
    Accepts `1,234.56`, `1234.5`, `$ 43,200.00`, `USD 17.00`; rejects
    anything ambiguous (European `1.234,56`, parentheses, multiple amounts,
    negatives, zero) by returning None -- the gold policy then routes the
    document to review instead of guessing."""
    if not text:
        return None
    s = re.sub(r"[A-Za-z$€£¥\s:*]", "", text)
    if not _AMOUNT_OK_RE.match(s):
        return None
    try:
        value = Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None
    minor = int((value * 100).to_integral_value())
    return minor if minor > 0 else None


def amount_band(minor: int | None) -> str:
    if minor is None:
        return "missing_or_unparseable"
    v = minor / 100
    if v < 100:
        return "under_100"
    if v < 1_000:
        return "100_to_1k"
    if v < 10_000:
        return "1k_to_10k"
    if v < 100_000:
        return "10k_to_100k"
    return "over_100k"


def length_band(words: int) -> str:
    if words < 150:
        return "short"
    if words < 400:
        return "medium"
    if words < 800:
        return "long"
    return "very_long"


def line_item_band(n: int) -> str:
    if n == 0:
        return "none"
    if n <= 3:
        return "1_to_3"
    if n <= 9:
        return "4_to_9"
    return "10_plus"


def _first(fields: list[dict], fieldtype: str) -> dict | None:
    for f in fields:
        if f["fieldtype"] == fieldtype:
            return f
    return None


def _all_text(fields: list[dict], fieldtype: str) -> list[str]:
    return [f["text"] for f in fields if f["fieldtype"] == fieldtype and f.get("text")]


def _ocr_stats(ocr: dict) -> tuple[int, float, float]:
    confs = [w["confidence"] for p in ocr.get("pages", []) for b in p.get("blocks", []) for ln in b.get("lines", []) for w in ln.get("words", [])]
    if not confs:
        return 0, 0.0, 1.0
    return len(confs), sum(confs) / len(confs), sum(c < 0.5 for c in confs) / len(confs)


def profile_document(docid: str) -> dict:
    ann_path = DOCILE_ROOT / "annotations" / f"{docid}.json"
    ocr_path = DOCILE_ROOT / "ocr" / f"{docid}.json"
    ann = json.loads(ann_path.read_text())
    ocr = json.loads(ocr_path.read_text())
    meta = ann["metadata"]
    fields = ann["field_extractions"]
    lis = ann["line_item_extractions"]

    vendor = _first(fields, "vendor_name")
    invnum = _first(fields, "document_id")
    date_issue = _first(fields, "date_issue")
    amount_due = _first(fields, "amount_due")
    amount_gross = _first(fields, "amount_total_gross")
    payable = amount_due or amount_gross

    n_words, mean_conf, lowconf = _ocr_stats(ocr)
    pages = max(1, int(meta.get("page_count", 1)))

    sims = {}
    for name, f, numeric in (("vendor_name", vendor, False), ("invoice_number", invnum, True), ("invoice_date", date_issue, True), ("payable_amount", payable, True)):
        if f is not None:
            ocr_text = scc.ocr_words_for_bbox(ocr, f["page"], f["bbox"])
            sims[name] = round(scc.similarity(f["text"], ocr_text, numeric=numeric), 4)
    min_agreement = min(sims.values()) if sims else 0.0
    critical = [v for k, v in sims.items() if k != "vendor_name"]
    critical_agreement = min(critical) if critical else 0.0
    vendor_agreement = sims.get("vendor_name", 0.0)

    # Critical-field agreement (invoice number / date / payable amount)
    # plus word-level OCR confidence decide the band; vendor-name spans
    # often include logo text, so they only demote on gross disagreement.
    if critical_agreement < 0.8 or lowconf >= 0.10 or vendor_agreement < 0.5:
        ocr_band = "low"
    elif critical_agreement >= 0.99 and vendor_agreement >= 0.9 and lowconf < 0.03:
        ocr_band = "high"
    else:
        ocr_band = "medium"

    li_ids = sorted({li["line_item_id"] for li in lis})
    lir_types = sorted({li["fieldtype"] for li in lis})
    grid = meta.get("page_to_table_grid") or {}
    grid_pages = len(grid) if isinstance(grid, dict) else 0
    max_cols = max((len(g.get("columns", [])) for g in grid.values()), default=0) if isinstance(grid, dict) else 0
    grid_flagged = any(g.get("missing_columns") or g.get("missing_second_table_on_page") for g in grid.values()) if isinstance(grid, dict) else False
    if not li_ids:
        table_complexity = "no_table"
    elif grid_pages > 1 or max_cols >= 6 or grid_flagged or len(li_ids) >= 10 or len(lir_types) >= 6:
        table_complexity = "complex"
    else:
        table_complexity = "simple"

    words_per_page = n_words / pages
    clutter_proxy = "cluttered" if (lowconf >= 0.05 or words_per_page >= 500) else "clean"

    currency_meta = (meta.get("currency") or "").lower() or None
    payable_text = payable["text"] if payable else None
    payable_minor = parse_payable_minor_units(payable_text)

    core_presence = {
        "vendor_name": vendor is not None,
        "vendor_address": _first(fields, "vendor_address") is not None,
        "invoice_number": invnum is not None and 0 < len((invnum["text"] or "").strip()) <= 40,
        "invoice_date": date_issue is not None,
        "due_date": _first(fields, "date_due") is not None,
        "payable_amount": payable is not None,
        "supported_currency": currency_meta in SUPPORTED_LEDGER_CURRENCIES,
        "customer_billing_name": _first(fields, "customer_billing_name") is not None,
    }
    n_core = sum(core_presence.values())
    completeness_band = "complete" if n_core >= 7 else ("partial" if n_core >= 5 else "sparse")

    pay_dest_fields = sorted({f["fieldtype"] for f in fields if f["fieldtype"] in PAYMENT_DESTINATION_FIELDTYPES})
    payable_complete = bool(core_presence["invoice_number"] and payable_minor is not None and core_presence["supported_currency"])

    return {
        "docid": docid,
        "cluster_id": meta.get("cluster_id"),
        "document_type": meta.get("document_type"),
        "currency_metadata": currency_meta,
        "page_count": pages,
        "page_class": "multi_page" if pages > 1 else "single_page",
        "vendor_name_raw": vendor["text"] if vendor else None,
        "vendor_display": normalize_whitespace(vendor["text"]) if vendor else None,
        "vendor_key": vendor_key(vendor["text"]) if vendor else "",
        "invoice_number": invnum["text"] if invnum else None,
        "payable_amount_text": payable_text,
        "payable_amount_source": ("amount_due" if amount_due else "amount_total_gross") if payable else None,
        "payable_minor_units_parsed": payable_minor,
        "amount_band": amount_band(payable_minor),
        "line_item_count": len(li_ids),
        "line_item_band": line_item_band(len(li_ids)),
        "lir_fieldtype_count": len(lir_types),
        "table_grid_pages": grid_pages,
        "table_max_columns": max_cols,
        "table_grid_flagged": bool(grid_flagged),
        "table_complexity": table_complexity,
        "ocr_word_count": n_words,
        "length_band": length_band(n_words),
        "ocr_mean_confidence": round(mean_conf, 4),
        "ocr_low_confidence_fraction": round(lowconf, 4),
        "ocr_field_agreement": sims,
        "ocr_min_field_agreement": round(min_agreement, 4),
        "ocr_critical_field_agreement": round(critical_agreement, 4),
        "ocr_quality_band": ocr_band,
        "visual_clutter_proxy": clutter_proxy,
        "core_field_presence": core_presence,
        "core_field_count": n_core,
        "field_completeness_band": completeness_band,
        "invoice_number_present": core_presence["invoice_number"],
        "payable_amount_present": core_presence["payable_amount"],
        "payment_destination_fields": pay_dest_fields,
        "payment_destination_field_present": bool(pay_dest_fields),
        "payable_complete": payable_complete,
        "annotation_sha256": sha256_file(ann_path),
        "ocr_sha256": sha256_file(ocr_path),
    }


def eligibility(profile: dict) -> tuple[bool, str]:
    if profile["document_type"] not in PAYABLE_DOCUMENT_TYPES:
        return False, f"document_type_not_payable:{profile['document_type']}"
    if not profile["vendor_display"] or len(profile["vendor_key"]) < 2 or len(profile["vendor_display"]) > 80:
        return False, "vendor_identity_missing_or_unusable"
    if not re.search(r"[A-Za-z]{2}", profile["vendor_display"]):
        return False, "vendor_identity_missing_or_unusable"
    if profile["cluster_id"] is None:
        return False, "cluster_id_missing"
    return True, ""


# ---------------------------------------------------------------------
# Observed invoice view (what read_invoice shows the agent).
# ---------------------------------------------------------------------

MAX_LINE_ITEMS_SHOWN = 25
MAX_LINE_ITEM_TEXT = 140


def _clip(text: str | None, n: int) -> str | None:
    if text is None:
        return None
    t = normalize_whitespace(text)
    return t if len(t) <= n else t[: n - 3] + "..."


def observed_invoice_fields(docid: str, use_ocr_text: bool) -> dict:
    """The invoice field values the agent observes. For documents in the
    low OCR-quality band (`use_ocr_text=True`), every field value is the
    real DocILE OCR text found under that field's annotated bounding box
    (falling back to the annotation text only when no OCR word overlaps),
    so genuine OCR corruption reaches the agent. Otherwise values are the
    annotation text (the existing `oracle_extraction_prototype` view)."""
    ann = json.loads((DOCILE_ROOT / "annotations" / f"{docid}.json").read_text())
    ocr = json.loads((DOCILE_ROOT / "ocr" / f"{docid}.json").read_text()) if use_ocr_text else None
    fields = ann["field_extractions"]

    def val(f: dict | None) -> str | None:
        if f is None:
            return None
        if use_ocr_text:
            o = scc.ocr_words_for_bbox(ocr, f["page"], f["bbox"])
            if o.strip():
                return o
        return f["text"]

    def first_val(ft: str) -> str | None:
        return val(_first(fields, ft))

    by_item: dict[int, dict] = {}
    for li in ann["line_item_extractions"]:
        by_item.setdefault(li["line_item_id"], {})
        key = li["fieldtype"].replace("line_item_", "")
        if key in ("description", "quantity", "amount_gross", "amount_net", "unit_price_gross", "code") and key not in by_item[li["line_item_id"]]:
            by_item[li["line_item_id"]][key] = _clip(val(li), MAX_LINE_ITEM_TEXT)
    items = [by_item[k] for k in sorted(by_item)]
    shown = items[:MAX_LINE_ITEMS_SHOWN]
    if len(items) > MAX_LINE_ITEMS_SHOWN:
        shown.append({"description": f"[{len(items) - MAX_LINE_ITEMS_SHOWN} further line items not shown]"})

    pay_dest = []
    for ft in PAYMENT_DESTINATION_FIELDTYPES:
        for f in fields:
            if f["fieldtype"] == ft:
                pay_dest.append(f"{ft}: {normalize_whitespace(val(f) or '')}")

    po_numbers = [normalize_whitespace(val(f) or "") for f in fields if f["fieldtype"] in ("order_id", "customer_order_id")]
    return {
        "vendor_name": _clip(first_val("vendor_name"), 120),
        "invoice_number": _clip(first_val("document_id"), 60),
        "invoice_date": _clip(first_val("date_issue"), 60),
        "due_date": _clip(first_val("date_due"), 60),
        "subtotal": _clip(first_val("amount_total_net"), 60),
        "tax": _clip(first_val("amount_total_tax"), 60),
        "total_amount_gross": _clip(first_val("amount_total_gross"), 60),
        "amount_due": _clip(first_val("amount_due"), 60),
        "purchase_order_numbers": [p for p in po_numbers if p][:5],
        "payment_destination": pay_dest,
        "vendor_address": _clip(first_val("vendor_address"), 200),
        "vendor_email": _clip(first_val("vendor_email"), 80),
        "customer_billing_name": _clip(first_val("customer_billing_name"), 120),
        "customer_billing_address": _clip(first_val("customer_billing_address"), 200),
        "payment_terms": _clip(first_val("payment_terms"), 160),
        "line_items": shown,
        "line_item_total_count": len(items),
    }


__all__ = [
    "PAYABLE_DOCUMENT_TYPES",
    "EXCLUDED_DOCUMENT_TYPES",
    "SUPPORTED_LEDGER_CURRENCIES",
    "load_train_docids",
    "vendor_key",
    "parse_payable_minor_units",
    "profile_document",
    "eligibility",
    "observed_invoice_fields",
    "sha256_file",
]
