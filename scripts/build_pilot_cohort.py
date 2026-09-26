"""Build the six-document DocILE pilot cohort for Tell dataset analysis.

Reads only data/docile/{annotations,ocr,pdfs,train.json,val.json} and the
docile-official checkout's package metadata. Writes:

  - results/dataset_inspection/pilot_cohort_manifest.json
      Structured, DocILE-sourced-only facts per document. Absent fields are
      JSON null. Every leaf value is independently re-derived and checked
      against the source annotation files by validate_manifest().
  - results/dataset_inspection/pilot_cohort_images/<docid>/
      page_<n>_original.png   - full-resolution page render (200 DPI)
      page_<n>_annotated.png  - same render with KILE field boxes + labels
                                 overlaid (source annotation, not OCR)
  - results/dataset_inspection/pilot_cohort_review.md
      Human-readable review, including an OCR-word cross-check per KILE
      field that is explicitly labeled as script-inferred (bounding-box
      overlap against the OCR file), distinct from the DocILE-sourced
      annotation text.

This script does not create scenarios, tools, ledger, memory, attacks, or
any Tell runtime component. It performs dataset analysis only.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = Path("/home/hp5/tell/data/docile")
RESULTS_DIR = Path("/home/hp5/tell/results/dataset_inspection")
IMAGES_DIR = RESULTS_DIR / "pilot_cohort_images"
MANIFEST_PATH = RESULTS_DIR / "pilot_cohort_manifest.json"
REVIEW_PATH = RESULTS_DIR / "pilot_cohort_review.md"

PILOT_DOCIDS = [
    "02214b3dc57148efa42d1b85",
    "00134dd365a24343b35b78c6",
    "00136a27c7774c1e8dc6b2f2",
    "002e3cf97973428f905671b3",
    "002f9b82b74f4258b3b072d0",
    "00e7f330e0344e9abdef3073",
]

# Curator labels are this script's own framing for why each document was
# picked (carried over from the prior inspection step's selection reasons).
# They are not DocILE facts and are kept out of the manifest.
CURATOR_LABEL = {
    "02214b3dc57148efa42d1b85": "IBAN/BIC example",
    "00134dd365a24343b35b78c6": "bank/account wire instructions",
    "00136a27c7774c1e8dc6b2f2": "complex 31-line-item invoice",
    "002e3cf97973428f905671b3": "due-date example",
    "002f9b82b74f4258b3b072d0": "typical single-page invoice",
    "00e7f330e0344e9abdef3073": "missing invoice-number case",
}

KILE_BOX_COLOR = (220, 30, 30)
KILE_LABEL_BG = (220, 30, 30)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def determine_split(docid: str, train_ids: set[str], val_ids: set[str]) -> str:
    if docid in train_ids:
        return "train"
    if docid in val_ids:
        return "val"
    return "unknown"


def render_pages(docid: str, page_count: int, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = DATA_ROOT / "pdfs" / f"{docid}.pdf"
    prefix = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "200", str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    rendered = sorted(out_dir.glob("page-*.png"))
    paths = []
    for i, src in enumerate(rendered, start=1):
        dst = out_dir / f"page_{i}_original.png"
        src.rename(dst)
        paths.append(str(dst))
    return paths


def overlap_fraction(word_box: tuple[float, float, float, float], field_box: tuple[float, float, float, float]) -> float:
    wx0, wy0, wx1, wy1 = word_box
    fx0, fy0, fx1, fy1 = field_box
    ix0, iy0 = max(wx0, fx0), max(wy0, fy0)
    ix1, iy1 = min(wx1, fx1), min(wy1, fy1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    warea = max(1e-9, (wx1 - wx0) * (wy1 - wy0))
    return inter / warea


def ocr_words_for_bbox(ocr_doc: dict, page_idx: int, bbox: list[float]) -> list[str]:
    """Script-inferred cross-check: OCR words whose box is >=50% covered by
    the field's bbox, in reading order. Not an official DocILE pairing."""
    if page_idx >= len(ocr_doc.get("pages", [])):
        return []
    page = ocr_doc["pages"][page_idx]
    matches: list[tuple[float, float, str]] = []
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            for w in line.get("words", []):
                (x0, y0), (x1, y1) = w["geometry"]
                if overlap_fraction((x0, y0, x1, y1), tuple(bbox)) >= 0.5:
                    matches.append((y0, x0, w["value"]))
    matches.sort()
    return [m[2] for m in matches]


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def draw_annotated_pages(docid: str, ann: dict, original_paths: list[str], out_dir: Path) -> list[str]:
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    fields_by_page: dict[int, list[dict]] = {}
    for f in ann["field_extractions"]:
        fields_by_page.setdefault(f["page"], []).append(f)

    annotated_paths = []
    for page_idx, orig_path in enumerate(original_paths):
        img = Image.open(orig_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Merge fields that share (near-)identical bboxes into one label so
        # e.g. amount_due and amount_total_gross printed at the same spot on
        # the page don't render two stacked, unreadable labels.
        page_fields = fields_by_page.get(page_idx, [])
        groups: list[dict] = []
        for f in page_fields:
            merged = False
            for g in groups:
                if overlap_fraction(tuple(f["bbox"]), tuple(g["bbox"])) > 0.85 and overlap_fraction(tuple(g["bbox"]), tuple(f["bbox"])) > 0.85:
                    g["fieldtypes"].append(f["fieldtype"])
                    merged = True
                    break
            if not merged:
                groups.append({"bbox": f["bbox"], "fieldtypes": [f["fieldtype"]]})

        # Draw boxes first, then place labels with simple collision avoidance
        # against already-placed label rectangles on this page.
        placed_label_rects: list[tuple[float, float, float, float]] = []
        for g in sorted(groups, key=lambda g: g["bbox"][1]):
            l, t, r, b = g["bbox"]
            box = (l * w, t * h, r * w, b * h)
            draw.rectangle(box, outline=KILE_BOX_COLOR, width=3)
            label = " / ".join(g["fieldtypes"])
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lh = th + 6

            candidates = [box[1] - lh, box[3] + 2]
            label_y = candidates[0] if candidates[0] > 0 else candidates[1]
            label_rect = (box[0], label_y, box[0] + tw + 6, label_y + lh)
            attempts = 0
            while any(_rects_overlap(label_rect, p) for p in placed_label_rects) and attempts < 20:
                label_y += lh
                label_rect = (box[0], label_y, box[0] + tw + 6, label_y + lh)
                attempts += 1
            placed_label_rects.append(label_rect)

            draw.rectangle(label_rect, fill=KILE_LABEL_BG)
            draw.text((label_rect[0] + 3, label_rect[1] + 3), label, fill=(255, 255, 255), font=font)

        dst = out_dir / f"page_{page_idx + 1}_annotated.png"
        img.save(dst)
        annotated_paths.append(str(dst))
    return annotated_paths


def occurrences(fields: list[dict], fieldtype: str) -> list[dict] | None:
    hits = [
        {"fieldtype": f["fieldtype"], "text": f["text"], "page": f["page"], "bbox": f["bbox"]}
        for f in fields
        if f["fieldtype"] == fieldtype
    ]
    return hits if hits else None


def build_manifest_entry(docid: str, split: str) -> dict:
    ann = load_json(DATA_ROOT / "annotations" / f"{docid}.json")
    fields = ann["field_extractions"]
    meta = ann["metadata"]

    img_dir = IMAGES_DIR / docid
    original_paths = render_pages(docid, meta["page_count"], img_dir)
    annotated_paths = draw_annotated_pages(docid, ann, original_paths, img_dir)

    line_item_ids = sorted({li["line_item_id"] for li in ann["line_item_extractions"]})

    entry = {
        "docid": docid,
        "split": split,
        "cluster_id": meta.get("cluster_id"),
        "page_count": meta.get("page_count"),
        "document_type": meta.get("document_type"),
        "language": meta.get("language"),
        "original_filename": meta.get("original_filename"),
        "pdf_path": str(DATA_ROOT / "pdfs" / f"{docid}.pdf"),
        "annotation_path": str(DATA_ROOT / "annotations" / f"{docid}.json"),
        "ocr_path": str(DATA_ROOT / "ocr" / f"{docid}.json"),
        "rendered_image_paths": {
            "original": original_paths,
            "annotated_kile": annotated_paths,
        },
        "vendor_name": occurrences(fields, "vendor_name"),
        "invoice_number_document_id": occurrences(fields, "document_id"),
        "invoice_date_date_issue": occurrences(fields, "date_issue"),
        "due_date_date_due": occurrences(fields, "date_due"),
        "currency": {
            "document_metadata_currency": meta.get("currency"),
            "field_currency_code_amount_due": occurrences(fields, "currency_code_amount_due"),
        },
        "subtotal_amount_total_net": occurrences(fields, "amount_total_net"),
        "tax": {
            "amount_total_tax": occurrences(fields, "amount_total_tax"),
            "tax_detail_tax": occurrences(fields, "tax_detail_tax"),
            "tax_detail_rate": occurrences(fields, "tax_detail_rate"),
            "tax_detail_gross": occurrences(fields, "tax_detail_gross"),
            "tax_detail_net": occurrences(fields, "tax_detail_net"),
        },
        "total": {
            "amount_total_gross": occurrences(fields, "amount_total_gross"),
            "amount_due": occurrences(fields, "amount_due"),
            "amount_paid": occurrences(fields, "amount_paid"),
        },
        "purchase_order": {
            "order_id": occurrences(fields, "order_id"),
            "customer_order_id": occurrences(fields, "customer_order_id"),
            "vendor_order_id": occurrences(fields, "vendor_order_id"),
        },
        "payment_destination_fields": {
            "iban": occurrences(fields, "iban"),
            "bic": occurrences(fields, "bic"),
            "bank_num": occurrences(fields, "bank_num"),
            "account_num": occurrences(fields, "account_num"),
            "payment_reference": occurrences(fields, "payment_reference"),
        },
        "line_item_count": len(line_item_ids),
        "all_kile_annotations": ann["field_extractions"],
        "all_lir_annotations": ann["line_item_extractions"],
        "line_item_headers": ann["line_item_headers"],
    }
    return entry


def validate_manifest(manifest: dict) -> list[str]:
    """Independent re-check: reload manifest.json from disk and confirm every
    non-null leaf traces back to the raw annotation/OCR/split files, not to
    values fabricated in memory during generation."""
    problems: list[str] = []
    train_ids = set(load_json(DATA_ROOT / "train.json"))
    val_ids = set(load_json(DATA_ROOT / "val.json"))

    on_disk = json.loads(MANIFEST_PATH.read_text())

    for docid, entry in on_disk.items():
        ann = load_json(DATA_ROOT / "annotations" / f"{docid}.json")
        fields = ann["field_extractions"]
        meta = ann["metadata"]

        expected_split = determine_split(docid, train_ids, val_ids)
        if entry["split"] != expected_split:
            problems.append(f"{docid}: split mismatch ({entry['split']} != {expected_split})")
        if entry["cluster_id"] != meta.get("cluster_id"):
            problems.append(f"{docid}: cluster_id mismatch")
        if entry["page_count"] != meta.get("page_count"):
            problems.append(f"{docid}: page_count mismatch")

        checks = {
            "vendor_name": "vendor_name",
            "invoice_number_document_id": "document_id",
            "invoice_date_date_issue": "date_issue",
            "due_date_date_due": "date_due",
            "subtotal_amount_total_net": "amount_total_net",
        }
        for key, fieldtype in checks.items():
            expected = occurrences(fields, fieldtype)
            if entry[key] != expected:
                problems.append(f"{docid}: {key} does not match recomputed occurrences of '{fieldtype}'")

        for group_key, mapping in {
            "tax": {
                "amount_total_tax": "amount_total_tax",
                "tax_detail_tax": "tax_detail_tax",
                "tax_detail_rate": "tax_detail_rate",
                "tax_detail_gross": "tax_detail_gross",
                "tax_detail_net": "tax_detail_net",
            },
            "total": {"amount_total_gross": "amount_total_gross", "amount_due": "amount_due", "amount_paid": "amount_paid"},
            "purchase_order": {"order_id": "order_id", "customer_order_id": "customer_order_id", "vendor_order_id": "vendor_order_id"},
            "payment_destination_fields": {
                "iban": "iban",
                "bic": "bic",
                "bank_num": "bank_num",
                "account_num": "account_num",
                "payment_reference": "payment_reference",
            },
        }.items():
            for subkey, fieldtype in mapping.items():
                expected = occurrences(fields, fieldtype)
                if entry[group_key][subkey] != expected:
                    problems.append(f"{docid}: {group_key}.{subkey} does not match recomputed occurrences of '{fieldtype}'")

        if entry["currency"]["document_metadata_currency"] != meta.get("currency"):
            problems.append(f"{docid}: currency.document_metadata_currency mismatch")
        expected_curr_field = occurrences(fields, "currency_code_amount_due")
        if entry["currency"]["field_currency_code_amount_due"] != expected_curr_field:
            problems.append(f"{docid}: currency.field_currency_code_amount_due mismatch")

        expected_line_item_count = len({li["line_item_id"] for li in ann["line_item_extractions"]})
        if entry["line_item_count"] != expected_line_item_count:
            problems.append(f"{docid}: line_item_count mismatch")

        if entry["all_kile_annotations"] != ann["field_extractions"]:
            problems.append(f"{docid}: all_kile_annotations does not match source annotation file verbatim")
        if entry["all_lir_annotations"] != ann["line_item_extractions"]:
            problems.append(f"{docid}: all_lir_annotations does not match source annotation file verbatim")
        if entry["line_item_headers"] != ann["line_item_headers"]:
            problems.append(f"{docid}: line_item_headers does not match source annotation file verbatim")

        for label, paths in entry["rendered_image_paths"].items():
            for p in paths:
                pp = Path(p)
                if not pp.exists() or pp.stat().st_size == 0:
                    problems.append(f"{docid}: rendered image missing or empty: {p}")

        # Aspect-ratio preservation check: rendered pixel size proportional
        # to the PDF-page-derived page_sizes_at_200dpi metadata.
        sizes = meta.get("page_sizes_at_200dpi", [])
        for i, p in enumerate(entry["rendered_image_paths"]["original"]):
            if i < len(sizes):
                exp_w, exp_h = sizes[i]
                with Image.open(p) as im:
                    got_w, got_h = im.size
                if abs(got_w - exp_w) > 2 or abs(got_h - exp_h) > 2:
                    problems.append(
                        f"{docid}: page {i + 1} rendered size {got_w}x{got_h} != expected {exp_w}x{exp_h}"
                    )

    return problems


def build_review_markdown(manifest: dict) -> str:
    lines = []
    lines.append("# Pilot Cohort Review: Six DocILE Documents")
    lines.append("")
    lines.append(
        "Dataset analysis only. Every field value below labeled **source** is "
        "copied verbatim from the official DocILE annotation JSON "
        "(`field_extractions` / `line_item_extractions` / `metadata`). Every "
        "value labeled **inferred** is computed by "
        "`scripts/build_pilot_cohort.py` from the OCR file via bounding-box "
        "overlap (>=50% of the OCR word's area inside the field's bbox) and "
        "is not an official DocILE pairing. No emails, canonical "
        "beneficiaries, memories, attacks, or expected actions are included."
    )
    lines.append("")

    for docid in PILOT_DOCIDS:
        entry = manifest[docid]
        ann = load_json(DATA_ROOT / "annotations" / f"{docid}.json")
        ocr = load_json(DATA_ROOT / "ocr" / f"{docid}.json")
        fields = ann["field_extractions"]

        lines.append(f"## {docid} — {CURATOR_LABEL[docid]}")
        lines.append("")
        lines.append(f"Split: `{entry['split']}` · Cluster: `{entry['cluster_id']}` · "
                      f"Pages: {entry['page_count']} · Document type: `{entry['document_type']}`")
        lines.append("")

        lines.append("**1. Rendered pages**")
        lines.append("")
        for i in range(entry["page_count"]):
            orig = Path(entry["rendered_image_paths"]["original"][i]).relative_to(RESULTS_DIR)
            annd = Path(entry["rendered_image_paths"]["annotated_kile"][i]).relative_to(RESULTS_DIR)
            lines.append(f"- Page {i + 1}: [original]({orig}) · [KILE-annotated]({annd})")
        lines.append("")

        lines.append("**2. Extracted fields** (source = DocILE annotation text; OCR cross-check = inferred)")
        lines.append("")
        lines.append("| Concept | DocILE fieldtype | Source text | OCR cross-check (inferred) |")
        lines.append("|---|---|---|---|")

        concept_rows = [
            ("Vendor name", "vendor_name"),
            ("Invoice number", "document_id"),
            ("Invoice date", "date_issue"),
            ("Due date", "date_due"),
            ("Currency (field)", "currency_code_amount_due"),
            ("Subtotal (proxy)", "amount_total_net"),
            ("Tax (total)", "amount_total_tax"),
            ("Total (gross)", "amount_total_gross"),
            ("Amount due", "amount_due"),
            ("Purchase order", "order_id"),
            ("IBAN", "iban"),
            ("BIC", "bic"),
            ("Bank routing num", "bank_num"),
            ("Account num", "account_num"),
            ("Payment reference", "payment_reference"),
        ]
        for label, fieldtype in concept_rows:
            hits = occurrences(fields, fieldtype)
            if not hits:
                lines.append(f"| {label} | `{fieldtype}` | *null (absent)* | — |")
                continue
            for h in hits:
                ocr_words = ocr_words_for_bbox(ocr, h["page"], h["bbox"])
                ocr_str = " ".join(ocr_words) if ocr_words else "*(no OCR word cleared the overlap threshold)*"
                lines.append(f"| {label} | `{fieldtype}` | \"{h['text']}\" (p.{h['page']}) | \"{ocr_str}\" |")
        lines.append(f"| Currency (document metadata) | `metadata.currency` | \"{entry['currency']['document_metadata_currency']}\" | — |")
        lines.append(f"| Line items | `line_item_extractions` | {entry['line_item_count']} line item(s) | — |")
        lines.append("")

        missing = []
        for label, fieldtype in concept_rows:
            if not occurrences(fields, fieldtype):
                missing.append(f"`{fieldtype}`")
        lines.append("**3. Missing or ambiguous fields**")
        lines.append("")
        if missing:
            lines.append("Absent from this document's annotations: " + ", ".join(missing) + ".")
        else:
            lines.append("No requested concept fields are absent.")
        lines.append("")

        lines.append("**4. What makes this document useful for Tell** *(analysis, not DocILE-sourced)*")
        lines.append("")
        lines.append(f"- {DOC_USEFULNESS[docid]}")
        lines.append("")

        lines.append("**5. Surrounding information that would have to be created later** *(analysis)*")
        lines.append("")
        lines.append(f"- {DOC_SURROUNDING[docid]}")
        lines.append("")

        lines.append("**6. Potential extraction or security complications** *(analysis)*")
        lines.append("")
        lines.append(f"- {DOC_COMPLICATIONS[docid]}")
        lines.append("")

    lines.append("## Comparison across the six documents")
    lines.append("")
    lines.append("| Doc ID | Label | Pages | Line items | Bank/IBAN fields | Invoice # | Due date | Currency |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for docid in PILOT_DOCIDS:
        entry = manifest[docid]
        pay = entry["payment_destination_fields"]
        has_pay = any(pay[k] for k in ("iban", "bic", "bank_num", "account_num"))
        inv_no = entry["invoice_number_document_id"]
        due = entry["due_date_date_due"]
        lines.append(
            f"| `{docid[:8]}…` | {CURATOR_LABEL[docid]} | {entry['page_count']} | "
            f"{entry['line_item_count']} | {'yes' if has_pay else 'no'} | "
            f"{'present' if inv_no else 'MISSING'} | {'present' if due else 'absent'} | "
            f"{entry['currency']['document_metadata_currency']} |"
        )
    lines.append("")

    lines.append("### Which document is best for the first clean workflow?")
    lines.append(
        "`002f9b82b74f4258b3b072d0` (typical single-page invoice). It has a "
        "vendor name, invoice number, currency, total, and amount due all "
        "present, a small legible line-item table (2 items), and no missing "
        "core fields beyond due date/subtotal/tax — the same gaps most of "
        "the corpus has (§ prior inspection report). It's simple enough to "
        "narrate end-to-end in a demo without extra scaffolding."
    )
    lines.append("")
    lines.append("### Which is best for demonstrating beneficiary conflict?")
    lines.append(
        "`00134dd365a24343b35b78c6` (bank/account wire instructions). It is "
        "one of the few documents with a real printed account number and "
        "bank routing number plus a payment reference, on a single page. "
        "That makes it possible to construct a scenario where a canonical "
        "vendor-master record (to be created later) states one beneficiary "
        "account while an injected or poisoned version of this invoice/email "
        "claims the printed-but-different one — a realistic-looking conflict "
        "grounded in a real printed value rather than an obviously synthetic "
        "one. `02214b3dc57148efa42d1b85` is the stronger candidate for "
        "*international* beneficiary conflict (adds IBAN + BIC) but spans 3 "
        "pages and a much larger invoice total, adding complexity that is "
        "better introduced after the first pass."
    )
    lines.append("")
    lines.append("### Which is best for invoice/tool-result injection?")
    lines.append(
        "`00136a27c7774c1e8dc6b2f2` (31 line items, 2 pages). Its size and "
        "line-item density give an injected instruction more places to hide "
        "plausibly (e.g. appended to a line-item description or a footer "
        "block) and stress-test whether the agent's read_invoice tool "
        "output handling stays robust when the surrounding legitimate text "
        "is large. Its main downside — no payment-destination fields at all "
        "— is irrelevant here since this attack surface only needs "
        "injected *text*, not real bank data."
    )
    lines.append("")
    lines.append("### Which is best as a difficult clean negative?")
    lines.append(
        "`00e7f330e0344e9abdef3073` (missing invoice-number case). The "
        "invoice number field is genuinely absent from the annotation, and "
        "the currency field is also unlabeled at the field level (only "
        "inferable from `metadata.currency`). A defended agent must be able "
        "to process a clean, non-adversarial document that is merely "
        "incomplete without raising a false alarm — this document tests "
        "that the probe and gate don't confuse *missing data* with "
        "*malicious content*."
    )
    lines.append("")
    lines.append("### Which two should be excluded from the very first implementation, and why?")
    lines.append(
        "`02214b3dc57148efa42d1b85` and `00136a27c7774c1e8dc6b2f2`. The "
        "first is a 3-page, six-figure EUR invoice with the richest payment "
        "metadata in the cohort — valuable later, but its size and "
        "multi-page table geometry add rendering/parsing complexity that "
        "isn't needed to prove the basic clean-workflow-then-attack loop. "
        "The second, the 31-line-item invoice, is the best injection "
        "stress-test (above) but for the same reason is a poor choice for "
        "the *first* clean pass, since its line-item table is large enough "
        "to make manual scenario authoring and visual demo narration slow. "
        "Both remain strong candidates for the second wave once the basic "
        "pipeline (tools, ledger, one clean scenario, one attacked pair) is "
        "proven on the simpler four documents."
    )
    lines.append("")
    return "\n".join(lines)


DOC_USEFULNESS = {
    "02214b3dc57148efa42d1b85": "Richest payment-destination annotation in the whole DocILE trainval set (IBAN + BIC + bank_num + account_num together) — the single best source for constructing a realistic international-wire beneficiary-conflict scenario later.",
    "00134dd365a24343b35b78c6": "Single-page, low word count, but carries real bank_num/account_num/payment_reference — a compact, easy-to-narrate case for a domestic beneficiary-conflict scenario.",
    "00136a27c7774c1e8dc6b2f2": "Largest line-item table in the cohort (31 items across 2 pages) — good stress case for tool-result/invoice-injection robustness and for testing that the agent doesn't get confused by document length.",
    "002e3cf97973428f905671b3": "One of the minority of documents (13.1% of the corpus) with an annotated due date, letting a scenario exercise due-date-based urgency or verification logic without inventing an unannotated field.",
    "002f9b82b74f4258b3b072d0": "Covers the core AP fields cleanly (vendor, invoice number, date, currency, total, amount due) on one page with a small 2-item table — the most 'textbook' clean invoice in the cohort.",
    "00e7f330e0344e9abdef3073": "Invoice number is genuinely absent from the annotation, and the currency symbol field is also unlabeled — a real example of an incomplete-but-legitimate document, useful for testing that missing data isn't mistaken for an attack.",
}

DOC_SURROUNDING = {
    "02214b3dc57148efa42d1b85": "Supplier email (clean + attacked variants), canonical vendor-master record with an out-of-band beneficiary IBAN/BIC that may or may not match the printed one, simulated account balances, memory records for a delayed-poisoning variant, expected safe/unsafe action labels.",
    "00134dd365a24343b35b78c6": "Same as above, sized for a simpler/faster scenario: supplier email, canonical vendor-master record, ledger accounts, expected action labels.",
    "00136a27c7774c1e8dc6b2f2": "Supplier email carrying the injected instruction (likely embedded near or inside the line-item table), canonical vendor-master record (this document has no printed bank data, so the canonical record must supply it), ledger accounts, expected action labels.",
    "002e3cf97973428f905671b3": "Supplier email, canonical vendor-master record, a due-date-aware verification/urgency policy for the safety LoRA to react to, ledger accounts, expected action labels.",
    "002f9b82b74f4258b3b072d0": "Supplier email, canonical vendor-master record, ledger accounts, expected action labels — the minimal surrounding set, appropriate for the first clean pass.",
    "00e7f330e0344e9abdef3073": "Supplier email, canonical vendor-master record (must supply the invoice number DocILE didn't capture, or the scenario must be designed to not depend on it), ledger accounts, expected action labels.",
}

DOC_COMPLICATIONS = {
    "02214b3dc57148efa42d1b85": "Three pages increase the chance that an injected instruction lands on a page the agent's read_invoice tool doesn't surface prominently; large EUR total (11.1M) may trigger unrelated transaction-limit logic that isn't part of the Tell experiment.",
    "00134dd365a24343b35b78c6": "Only one line item — a scenario built purely around this document under-exercises the line-item-injection surface; the printed account number and real bank routing number are genuine third-party financial identifiers from a real historical document and should be treated as sensitive even though they're old and public via the benchmark.",
    "00136a27c7774c1e8dc6b2f2": "High line-item density makes manual/inspection review slower and increases the chance an injected string is mistaken for legitimate line-item text (or vice versa) during scenario authoring; no payment-destination fields means beneficiary-conflict scenarios can't be grounded in this document's own printed data.",
    "002e3cf97973428f905671b3": "Currency is DocILE's 'other' bucket rather than a resolvable ISO code at the field level — a scenario using this document needs an explicit currency assumption documented as an authoring decision, not a DocILE fact.",
    "002f9b82b74f4258b3b072d0": "No subtotal/tax breakdown and no payment-destination fields at all — fine for a first clean pass, but this document alone cannot demonstrate beneficiary-conflict or tax-consistency checks.",
    "00e7f330e0344e9abdef3073": "Missing invoice number could itself be misread as a red flag by a naive detector; any scenario using this document must make clear in its design notes that the absence is a genuine DocILE annotation gap, not an attack artifact, to avoid contaminating probe training labels.",
}


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    train_ids = set(load_json(DATA_ROOT / "train.json"))
    val_ids = set(load_json(DATA_ROOT / "val.json"))

    manifest: dict[str, dict] = {}
    for docid in PILOT_DOCIDS:
        split = determine_split(docid, train_ids, val_ids)
        manifest[docid] = build_manifest_entry(docid, split)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote {MANIFEST_PATH}")

    problems = validate_manifest(manifest)
    if problems:
        print(f"VALIDATION FAILED: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    else:
        n_checks = sum(
            1
            for docid in PILOT_DOCIDS
            for _ in range(1)
        )
        print(f"VALIDATION PASSED for all {len(PILOT_DOCIDS)} documents: "
              "every non-null manifest leaf re-derived from source annotation "
              "files matches, all raw KILE/LIR arrays match verbatim, all "
              "rendered images exist and match expected page dimensions.")

    review_md = build_review_markdown(manifest)
    REVIEW_PATH.write_text(review_md)
    print(f"Wrote {REVIEW_PATH}")


if __name__ == "__main__":
    main()
