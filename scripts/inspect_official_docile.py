"""Read-only inspection of the official DocILE dataset checkout for Tell.

Scans data/docile (annotations/ocr/pdfs + split files), cross-checks the
schema against docile.dataset.KILE_FIELDTYPES / LIR_FIELDTYPES from the
official docile-official checkout, computes Tell-relevant field coverage,
and selects ten representative documents (rendering their first page as a
PNG for visual review).

This script only reads data/docile and docile-official and writes into
results/dataset_inspection/. It does not touch src/tell, does not train or
evaluate anything, and makes no network calls.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("/home/hp5/tell/data/docile")
RESULTS_DIR = Path("/home/hp5/tell/results/dataset_inspection")
IMAGES_DIR = RESULTS_DIR / "sample_images"

# DocILE does not define distinct "beneficiary_name" / "remittance_address"
# field types. These map to the closest existing field as a proxy only.
TELL_FIELD_MAP: dict[str, list[str]] = {
    "vendor_supplier_name": ["vendor_name"],
    "invoice_number": ["document_id"],
    "invoice_date": ["date_issue"],
    "due_date": ["date_due"],
    "currency": ["currency_code_amount_due"],  # metadata.currency also checked separately
    "subtotal": ["amount_total_net"],  # proxy: no explicit "subtotal" fieldtype
    "tax": ["amount_total_tax", "tax_detail_tax", "tax_detail_rate", "tax_detail_gross", "tax_detail_net"],
    "total_amount": ["amount_total_gross", "amount_due"],
    "purchase_order": ["order_id", "customer_order_id", "vendor_order_id"],
    "beneficiary_name": ["vendor_name"],  # proxy only; DocILE has no distinct beneficiary field
    "bank_account": ["account_num"],
    "iban": ["iban"],
    "routing_details": ["bank_num", "bic"],
    "remittance_address": ["vendor_address"],  # proxy only; DocILE has no distinct remittance field
}
PROXY_ONLY_CONCEPTS = {"beneficiary_name", "remittance_address"}
LINE_ITEMS_CONCEPT = "line_items"


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}PB"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def ocr_word_count(ocr_doc: dict) -> int:
    total = 0
    for page in ocr_doc.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                total += len(line.get("words", []))
    return total


def inspect_structure() -> dict:
    report: dict = {"root": str(DATA_ROOT)}
    report["total_size_bytes"] = dir_size_bytes(DATA_ROOT)
    subdirs = ["annotations", "ocr", "pdfs"]
    report["subdirs"] = {}
    for sub in subdirs:
        p = DATA_ROOT / sub
        files = sorted(p.iterdir())
        exts = Counter(f.suffix.lstrip(".") for f in files)
        report["subdirs"][sub] = {
            "path": str(p),
            "file_count": len(files),
            "size_bytes": dir_size_bytes(p),
            "extensions": dict(exts),
        }
    report["split_files"] = {}
    for split in ["train.json", "val.json", "trainval.json"]:
        p = DATA_ROOT / split
        ids = load_json(p)
        report["split_files"][split] = {"count": len(ids), "size_bytes": p.stat().st_size}
    return report


def check_integrity(doc_ids: list[str]) -> dict:
    missing_ann, missing_ocr, missing_pdf = [], [], []
    empty_ann, empty_ocr, empty_pdf = [], [], []
    for docid in doc_ids:
        ann_p = DATA_ROOT / "annotations" / f"{docid}.json"
        ocr_p = DATA_ROOT / "ocr" / f"{docid}.json"
        pdf_p = DATA_ROOT / "pdfs" / f"{docid}.pdf"
        if not ann_p.exists():
            missing_ann.append(docid)
        elif ann_p.stat().st_size == 0:
            empty_ann.append(docid)
        if not ocr_p.exists():
            missing_ocr.append(docid)
        elif ocr_p.stat().st_size == 0:
            empty_ocr.append(docid)
        if not pdf_p.exists():
            missing_pdf.append(docid)
        elif pdf_p.stat().st_size == 0:
            empty_pdf.append(docid)
    return {
        "checked": len(doc_ids),
        "missing_annotations": missing_ann,
        "missing_ocr": missing_ocr,
        "missing_pdfs": missing_pdf,
        "empty_annotations": empty_ann,
        "empty_ocr": empty_ocr,
        "empty_pdfs": empty_pdf,
    }


def scan_documents(doc_ids: list[str]) -> dict:
    """Single pass over every annotated document; no docile package needed."""
    observed_kile_types: Counter = Counter()
    observed_lir_types: Counter = Counter()
    cluster_ids: Counter = Counter()
    page_counts: Counter = Counter()
    currencies: Counter = Counter()
    document_types: Counter = Counter()
    languages: Counter = Counter()

    docs_with_kile = 0
    docs_with_lir = 0
    docs_with_cluster = 0
    docs_with_table_grid = 0

    tell_field_hits: dict[str, int] = {k: 0 for k in TELL_FIELD_MAP}
    tell_field_hits["currency_metadata"] = 0
    line_item_doc_count = 0
    line_item_counts: list[int] = []

    per_doc_records: dict[str, dict] = {}

    for docid in doc_ids:
        ann = load_json(DATA_ROOT / "annotations" / f"{docid}.json")
        kile_types_here = {f["fieldtype"] for f in ann["field_extractions"]}
        lir_types_here = {f["fieldtype"] for f in ann["line_item_extractions"]}
        observed_kile_types.update(kile_types_here)
        observed_lir_types.update(lir_types_here)

        if kile_types_here:
            docs_with_kile += 1
        if lir_types_here:
            docs_with_lir += 1

        meta = ann.get("metadata", {})
        if "cluster_id" in meta and meta["cluster_id"] is not None:
            docs_with_cluster += 1
            cluster_ids[meta["cluster_id"]] += 1
        page_counts[meta.get("page_count")] += 1
        currencies[meta.get("currency")] += 1
        document_types[meta.get("document_type")] += 1
        languages[meta.get("language")] += 1
        if meta.get("page_to_table_grid"):
            docs_with_table_grid += 1
        if meta.get("currency"):
            tell_field_hits["currency_metadata"] += 1

        for concept, fieldtypes in TELL_FIELD_MAP.items():
            if kile_types_here.intersection(fieldtypes):
                tell_field_hits[concept] += 1

        n_line_items = len({li["line_item_id"] for li in ann["line_item_extractions"]})
        line_item_counts.append(n_line_items)
        if n_line_items > 0:
            line_item_doc_count += 1

        per_doc_records[docid] = {
            "cluster_id": meta.get("cluster_id"),
            "page_count": meta.get("page_count"),
            "currency": meta.get("currency"),
            "document_type": meta.get("document_type"),
            "n_kile_fields": len(ann["field_extractions"]),
            "n_kile_types": len(kile_types_here),
            "n_line_items": n_line_items,
            "kile_types": sorted(kile_types_here),
        }

    n = len(doc_ids)
    coverage_pct = {k: round(100 * v / n, 1) for k, v in tell_field_hits.items()}
    coverage_pct[LINE_ITEMS_CONCEPT] = round(100 * line_item_doc_count / n, 1)

    return {
        "n_docs": n,
        "docs_with_kile_fields": docs_with_kile,
        "docs_with_lir_fields": docs_with_lir,
        "docs_with_cluster_id": docs_with_cluster,
        "docs_with_table_grid": docs_with_table_grid,
        "distinct_clusters": len(cluster_ids),
        "top_clusters": cluster_ids.most_common(10),
        "page_count_distribution": dict(page_counts),
        "currency_distribution": dict(currencies),
        "document_type_distribution": dict(document_types),
        "language_distribution": dict(languages),
        "observed_kile_types": sorted(observed_kile_types),
        "observed_lir_types": sorted(observed_lir_types),
        "kile_type_frequency": observed_kile_types.most_common(),
        "lir_type_frequency": observed_lir_types.most_common(),
        "line_item_count_stats": {
            "mean": round(statistics.mean(line_item_counts), 2),
            "median": statistics.median(line_item_counts),
            "max": max(line_item_counts),
            "docs_with_zero_line_items": sum(1 for c in line_item_counts if c == 0),
        },
        "tell_field_coverage_pct": coverage_pct,
        "tell_field_hit_counts": tell_field_hits,
        "per_doc": per_doc_records,
    }


def select_representative_documents(per_doc: dict, doc_ids: list[str]) -> list[str]:
    """Pick 10 docs by deliberate criteria rather than random diversity alone.

    DocILE's Tell-relevant payment-destination fields (iban, bank_num,
    account_num) are rare (<2% of documents), so a purely diversity-based
    sample is likely to miss them entirely. Instead this explicitly slots in
    one example of each notable case, then fills remaining slots with
    "typical" single-page invoices from distinct clusters.
    """
    selected: list[str] = []
    seen_clusters: set = set()

    def has_field(docid: str, fieldtype: str) -> bool:
        ann = load_json(DATA_ROOT / "annotations" / f"{docid}.json")
        return any(f["fieldtype"] == fieldtype for f in ann["field_extractions"])

    def add(docid: str) -> None:
        if docid not in selected:
            selected.append(docid)
            seen_clusters.add(per_doc[docid]["cluster_id"])

    # 1. Rarest Tell-relevant field: IBAN present (only 3/5680 documents).
    for d in doc_ids:
        if has_field(d, "iban"):
            add(d)
            break

    # 2. A document with bank routing + account number annotated together
    #    (demonstrates DocILE *can* carry payment-destination data, rarely).
    for d in doc_ids:
        if d not in selected and has_field(d, "bank_num") and has_field(d, "account_num"):
            add(d)
            break

    # 3. A genuinely multi-page document.
    for d in doc_ids:
        if d not in selected and (per_doc[d]["page_count"] or 1) > 1:
            add(d)
            break

    # 4. A document with zero line items (non-tabular, e.g. a simple order/receipt).
    for d in doc_ids:
        if d not in selected and per_doc[d]["n_line_items"] == 0:
            add(d)
            break

    # 5. A high-complexity document with many line items.
    for d in sorted(doc_ids, key=lambda d: -per_doc[d]["n_line_items"]):
        if d not in selected:
            add(d)
            break

    # 6. A document with date_due annotated (only ~13% of documents).
    for d in doc_ids:
        if d not in selected and has_field(d, "date_due"):
            add(d)
            break

    # 7. A document with a purchase-order-family field annotated.
    for d in doc_ids:
        if d not in selected and any(
            has_field(d, ft) for ft in ("order_id", "customer_order_id", "vendor_order_id")
        ):
            add(d)
            break

    # 8-10. Fill remaining slots with typical single-page tax invoices of
    # moderate complexity (2-8 line items), one per distinct cluster.
    typical = [
        d
        for d in doc_ids
        if (per_doc[d]["page_count"] or 1) == 1
        and 2 <= per_doc[d]["n_line_items"] <= 8
        and per_doc[d]["document_type"] == "tax_invoice"
    ]
    for d in typical:
        if len(selected) >= 10:
            break
        if per_doc[d]["cluster_id"] in seen_clusters:
            continue
        add(d)

    # Fallback fill if still short.
    for d in doc_ids:
        if len(selected) >= 10:
            break
        if d not in selected:
            add(d)

    return selected[:10]


def field_first_value(fields: list[dict], fieldtype: str) -> str | None:
    for f in fields:
        if f["fieldtype"] == fieldtype:
            return f["text"]
    return None


def render_first_page(docid: str, out_dir: Path) -> str | None:
    pdf_path = DATA_ROOT / "pdfs" / f"{docid}.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / docid
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-f", "1", "-l", "1", "-r", "150", str(pdf_path), str(out_prefix)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"RENDER_FAILED: {exc}"
    candidates = sorted(out_dir.glob(f"{docid}*.png"))
    return str(candidates[0]) if candidates else None


def describe_document(docid: str, split_of: dict[str, str]) -> dict:
    ann = load_json(DATA_ROOT / "annotations" / f"{docid}.json")
    ocr = load_json(DATA_ROOT / "ocr" / f"{docid}.json")
    fields = ann["field_extractions"]
    meta = ann["metadata"]
    n_line_items = len({li["line_item_id"] for li in ann["line_item_extractions"]})

    account_num = field_first_value(fields, "account_num")
    iban = field_first_value(fields, "iban")
    bank_num = field_first_value(fields, "bank_num")
    bic = field_first_value(fields, "bic")
    payment_reference = field_first_value(fields, "payment_reference")

    missing = []
    for concept, fieldtypes in TELL_FIELD_MAP.items():
        if concept in PROXY_ONLY_CONCEPTS:
            continue
        if not any(field_first_value(fields, ft) for ft in fieldtypes):
            missing.append(concept)
    if n_line_items == 0:
        missing.append("line_items")

    image_path = render_first_page(docid, IMAGES_DIR)

    return {
        "docid": docid,
        "split": split_of.get(docid, "unknown"),
        "cluster_id": meta.get("cluster_id"),
        "page_count": meta.get("page_count"),
        "pdf_path": str(DATA_ROOT / "pdfs" / f"{docid}.pdf"),
        "rendered_image_path": image_path,
        "ocr_word_count": ocr_word_count(ocr),
        "vendor_name": field_first_value(fields, "vendor_name"),
        "invoice_number_document_id": field_first_value(fields, "document_id"),
        "date_issue": field_first_value(fields, "date_issue"),
        "date_due": field_first_value(fields, "date_due"),
        "currency_metadata": meta.get("currency"),
        "currency_code_amount_due_field": field_first_value(fields, "currency_code_amount_due"),
        "amount_total_gross": field_first_value(fields, "amount_total_gross"),
        "amount_due": field_first_value(fields, "amount_due"),
        "line_item_count": n_line_items,
        "payment_destination_fields": {
            "account_num": account_num,
            "iban": iban,
            "bank_num": bank_num,
            "bic": bic,
            "payment_reference": payment_reference,
        },
        "missing_or_ambiguous": missing,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    structure = inspect_structure()

    trainval_ids = load_json(DATA_ROOT / "trainval.json")
    train_ids = load_json(DATA_ROOT / "train.json")
    val_ids = load_json(DATA_ROOT / "val.json")
    split_of = {d: "train" for d in train_ids}
    split_of.update({d: "val" for d in val_ids})

    integrity = check_integrity(trainval_ids)
    scan = scan_documents(trainval_ids)

    representative_ids = select_representative_documents(scan["per_doc"], trainval_ids)
    representative_docs = [describe_document(d, split_of) for d in representative_ids]

    output = {
        "structure": structure,
        "integrity": integrity,
        "scan": {k: v for k, v in scan.items() if k != "per_doc"},
        "representative_documents": representative_docs,
    }

    out_path = RESULTS_DIR / "official_docile_inspection.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Wrote {out_path}")
    print(f"Wrote rendered page images to {IMAGES_DIR}")
    print(json.dumps({"structure": structure, "integrity_summary": {
        "checked": integrity["checked"],
        "missing_annotations": len(integrity["missing_annotations"]),
        "missing_ocr": len(integrity["missing_ocr"]),
        "missing_pdfs": len(integrity["missing_pdfs"]),
    }}, indent=2, default=str))


if __name__ == "__main__":
    main()
