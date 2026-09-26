"""Selects 60 official DocILE train-split documents for the LoRA corpus
v1, reusing the exact same deterministic-selection infrastructure/method
`scripts/select_probe_corpus_documents.py` already established for the
probe corpus (same candidate pool, same feature-diversity signature, same
seeded-shuffle-then-greedy-novelty selection, same global vendor+cluster
isolation discipline) -- only the exclusion set, seed, and document/split
counts differ.

Excludes: the 20 documents already selected for probe corpus v1/v1.1
(read directly from results/probe_dataset/document_selection_manifest.json,
never modified), plus the same 2 legacy prior-experiment docids the probe
corpus already excluded.

Writes:
  - results/lora_dataset/v1/document_selection_manifest.json
  - results/lora_dataset/v1/document_selection_report.md
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

CANDIDATE_SCORES_PATH = Path("/home/hp5/tell/results/dataset_inspection/clean_candidate_scores.json")
DOCILE_ANNOTATIONS_DIR = Path("/home/hp5/tell/data/docile/annotations")
PROBE_CORPUS_SELECTION_PATH = Path("/home/hp5/tell/results/probe_dataset/document_selection_manifest.json")

OUTPUT_DIR = Path("/home/hp5/tell/results/lora_dataset/v1")
MANIFEST_PATH = OUTPUT_DIR / "document_selection_manifest.json"
REPORT_PATH = OUTPUT_DIR / "document_selection_report.md"

N_TRAIN, N_VAL, N_TEST = 40, 10, 10
N_TOTAL = N_TRAIN + N_VAL + N_TEST

# The same 2 documents the probe corpus already excluded as "used by prior,
# unrelated Tell experiments" (clean_04d531ca / clean_002f9b82).
LEGACY_EXCLUDE_DOCIDS = {"04d531ca811f448a91c6ff4e", "002f9b82b74f4258b3b072d0"}

# Different seed than the probe corpus (20240923) so this is a genuinely
# independent deterministic draw, not a re-derivation of the same order.
SEED = 20250114

# ProposePaymentAction/tell.payment.ledger.SUPPORTED_CURRENCIES only
# accepts usd/eur/gbp -- a document in another currency cannot be used to
# build a clean_propose_payment gold example, so it is excluded from the
# eligible pool entirely (the probe corpus had no such constraint, since
# it never constructs a propose_payment action).
SUPPORTED_CURRENCIES = {"usd", "eur", "gbp"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_amount(text: str) -> float | None:
    cleaned = re.sub(r"[^0-9.\-]", "", text or "")
    if not cleaned:
        return None
    try:
        return float(Decimal(cleaned))
    except (InvalidOperation, ValueError):
        return None


def _amount_bucket(amount: float | None) -> str:
    if amount is None or amount <= 0:
        return "unknown"
    if amount < 100:
        return "under_100"
    if amount < 1_000:
        return "100_to_1k"
    if amount < 10_000:
        return "1k_to_10k"
    if amount < 100_000:
        return "10k_to_100k"
    return "over_100k"


def _word_count_bucket(n: int) -> str:
    if n < 60:
        return "short"
    if n < 150:
        return "medium"
    if n < 300:
        return "long"
    return "very_long"


def _line_item_bucket(n: int) -> str:
    if n <= 1:
        return "single"
    if n <= 3:
        return "few"
    return "many"


_DATE_MONTH_NAME_RE = re.compile(r"[A-Za-z]{3,}")


def _date_format_class(text: str | None) -> str:
    if not text:
        return "unknown"
    if _DATE_MONTH_NAME_RE.search(text):
        return "month_name"
    if "." in text:
        return "dot_separated"
    if "-" in text:
        return "dash_separated"
    if "/" in text:
        return "slash_separated"
    return "other"


def _invoice_number_format_class(text: str | None) -> str:
    if not text:
        return "unknown"
    if text.isdigit():
        return "numeric_only"
    if re.search(r"[A-Za-z]", text) and re.search(r"\d", text):
        return "alphanumeric"
    if "-" in text:
        return "hyphenated"
    return "other"


def _load_candidates() -> list[dict]:
    scores = json.loads(CANDIDATE_SCORES_PATH.read_text())
    return scores["all_qualifying_candidates_ranked"]


def _page_count(docid: str) -> int:
    ann = json.loads((DOCILE_ANNOTATIONS_DIR / f"{docid}.json").read_text())
    return int(ann["metadata"]["page_count"])


def _payable_amount_text(c: dict) -> tuple[str, str]:
    if c.get("amount_due"):
        return c["amount_due"], "amount_due"
    if c.get("amount_total_gross"):
        return c["amount_total_gross"], "amount_total_gross"
    return "", "none"


def _feature_signature(c: dict) -> dict:
    amount_text, _ = _payable_amount_text(c)
    amount = _parse_amount(amount_text)
    return {
        "currency": c.get("currency") or "unknown",
        "page_count_class": "multi_page" if c["page_count"] > 1 else "single_page",
        "line_item_bucket": _line_item_bucket(c.get("line_item_count", 0)),
        "amount_bucket": _amount_bucket(amount),
        "word_count_bucket": _word_count_bucket(c.get("ocr_word_count", 0)),
        "date_format_class": _date_format_class(c.get("invoice_date")),
        "invoice_number_format_class": _invoice_number_format_class(c.get("invoice_number")),
    }


def _vendor_key(vendor_name: str) -> str:
    return re.sub(r"\s+", " ", (vendor_name or "").strip().upper())


def main() -> None:
    probe_corpus_selection = json.loads(PROBE_CORPUS_SELECTION_PATH.read_text())
    probe_corpus_docids = {d["docid"] for d in probe_corpus_selection["documents"]}
    if len(probe_corpus_docids) != 20:
        raise RuntimeError(f"Expected 20 probe-corpus docids to exclude, found {len(probe_corpus_docids)}")

    exclude_docids = probe_corpus_docids | LEGACY_EXCLUDE_DOCIDS

    raw_candidates = _load_candidates()
    print(f"[select] loaded {len(raw_candidates)} qualifying candidates")

    candidates = []
    n_excluded_probe_corpus = 0
    n_excluded_legacy = 0
    n_excluded_zero_amount = 0
    n_excluded_unsupported_currency = 0
    for c in raw_candidates:
        if c["docid"] in probe_corpus_docids:
            n_excluded_probe_corpus += 1
            continue
        if c["docid"] in LEGACY_EXCLUDE_DOCIDS:
            n_excluded_legacy += 1
            continue
        if (c.get("currency") or "").lower() not in SUPPORTED_CURRENCIES:
            n_excluded_unsupported_currency += 1
            continue
        amount_text, _ = _payable_amount_text(c)
        amount_value = _parse_amount(amount_text)
        if amount_value is None or amount_value <= 0:
            n_excluded_zero_amount += 1
            continue
        c = dict(c)
        c["page_count"] = _page_count(c["docid"])
        candidates.append(c)
    print(
        f"[select] {len(candidates)} candidates after excluding {n_excluded_probe_corpus} probe-corpus docids, "
        f"{n_excluded_legacy} legacy-excluded docids, {n_excluded_unsupported_currency} unsupported-currency "
        f"candidates, and {n_excluded_zero_amount} zero/unparseable-amount candidates"
    )

    candidates.sort(key=lambda c: c["docid"])
    rng = random.Random(SEED)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    shuffled = [candidates[i] for i in order]

    selected: list[dict] = []
    used_vendor_keys: set[str] = set()
    used_cluster_ids: set[int] = set()
    seen_feature_values: dict[str, set[str]] = {}

    def _novelty_score(c: dict) -> int:
        sig = _feature_signature(c)
        score = 0
        for feature_name, value in sig.items():
            if value not in seen_feature_values.get(feature_name, set()):
                score += 1
        return score

    vendor_isolation_relaxed_for: list[str] = []
    cluster_isolation_relaxed_for: list[str] = []

    remaining = list(shuffled)
    while len(selected) < N_TOTAL and remaining:
        remaining.sort(key=lambda c: (-_novelty_score(c), -c.get("tiebreak_score", 0.0), c["docid"]))
        pick = None
        for c in remaining:
            vendor_key = _vendor_key(c["vendor_name"])
            if vendor_key in used_vendor_keys or c["cluster_id"] in used_cluster_ids:
                continue
            pick = c
            break
        relaxed = False
        if pick is None:
            for c in remaining:
                vendor_key = _vendor_key(c["vendor_name"])
                if vendor_key in used_vendor_keys:
                    continue
                pick = c
                relaxed = True
                break
        if pick is None:
            pick = remaining[0]
            relaxed = True

        if relaxed:
            if _vendor_key(pick["vendor_name"]) in used_vendor_keys:
                vendor_isolation_relaxed_for.append(pick["docid"])
            if pick["cluster_id"] in used_cluster_ids:
                cluster_isolation_relaxed_for.append(pick["docid"])

        selected.append(pick)
        used_vendor_keys.add(_vendor_key(pick["vendor_name"]))
        used_cluster_ids.add(pick["cluster_id"])
        sig = _feature_signature(pick)
        for feature_name, value in sig.items():
            seen_feature_values.setdefault(feature_name, set()).add(value)
        remaining.remove(pick)

    if len(selected) != N_TOTAL:
        raise RuntimeError(f"Could not select {N_TOTAL} documents (got {len(selected)})")

    # Split assignment: cycle [train]*4 + [validation] + [test] over
    # selection order -> exactly 40/10/10 across 60 selections (10 cycles).
    split_pattern = ["train", "train", "train", "train", "validation", "test"]
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for i, c in enumerate(selected):
        split = split_pattern[i % len(split_pattern)]
        splits[split].append(c)

    assert len(splits["train"]) == N_TRAIN
    assert len(splits["validation"]) == N_VAL
    assert len(splits["test"]) == N_TEST

    all_docids = [c["docid"] for c in selected]
    assert len(all_docids) == len(set(all_docids)), "duplicate docid selected"
    assert not (set(all_docids) & exclude_docids), "a probe-corpus or legacy-excluded docid was selected"

    documents = []
    for split_name, docs in splits.items():
        for c in docs:
            sig = _feature_signature(c)
            payable_amount_text, payable_amount_source = _payable_amount_text(c)
            documents.append(
                {
                    "docid": c["docid"],
                    "split": split_name,
                    "vendor_name": c["vendor_name"],
                    "cluster_id": c["cluster_id"],
                    "document_type": c["document_type"],
                    "currency": c["currency"],
                    "invoice_number": c["invoice_number"],
                    "invoice_date": c["invoice_date"],
                    "amount_due": c["amount_due"],
                    "payable_amount_text": payable_amount_text,
                    "payable_amount_source": payable_amount_source,
                    "line_item_count": c["line_item_count"],
                    "ocr_word_count": c["ocr_word_count"],
                    "page_count": c["page_count"],
                    "tiebreak_score": c.get("tiebreak_score"),
                    "annotation_sha256": _sha256_file(DOCILE_ANNOTATIONS_DIR / f"{c['docid']}.json"),
                    "annotation_path": str(DOCILE_ANNOTATIONS_DIR / f"{c['docid']}.json"),
                    "diversity_signature": sig,
                }
            )

    diversity_coverage = {}
    for feature_name in ("currency", "page_count_class", "line_item_bucket", "amount_bucket", "word_count_bucket", "date_format_class", "invoice_number_format_class"):
        values = sorted({d["diversity_signature"][feature_name] for d in documents})
        diversity_coverage[feature_name] = values

    documented_limitations = []
    if diversity_coverage["page_count_class"] == ["single_page"]:
        documented_limitations.append(
            "Single- vs. multi-page diversity could not be achieved: every qualifying clean-candidate-pool "
            "document is single-page (the same pool limitation already documented for the probe corpus)."
        )
    n_fallback_amount = sum(1 for d in documents if d["payable_amount_source"] == "amount_total_gross")
    if n_fallback_amount:
        documented_limitations.append(
            f"{n_fallback_amount} selected document(s) use `amount_total_gross` as the payable amount instead of "
            "a null `amount_due` field."
        )

    manifest = {
        "seed": SEED,
        "candidate_scores_path": str(CANDIDATE_SCORES_PATH),
        "candidate_scores_sha256": _sha256_file(CANDIDATE_SCORES_PATH),
        "n_candidates_scanned": len(raw_candidates),
        "n_candidates_after_exclusions": len(candidates),
        "n_excluded_probe_corpus_docids": n_excluded_probe_corpus,
        "n_excluded_legacy_docids": n_excluded_legacy,
        "n_excluded_zero_or_unparseable_amount": n_excluded_zero_amount,
        "n_excluded_unsupported_currency": n_excluded_unsupported_currency,
        "supported_currencies": sorted(SUPPORTED_CURRENCIES),
        "excluded_probe_corpus_docids": sorted(probe_corpus_docids),
        "excluded_legacy_docids": sorted(LEGACY_EXCLUDE_DOCIDS),
        "probe_corpus_selection_manifest_sha256_at_read_time": _sha256_file(PROBE_CORPUS_SELECTION_PATH),
        "n_train": N_TRAIN,
        "n_validation": N_VAL,
        "n_test": N_TEST,
        "documents": documents,
        "vendor_isolation": "global (no vendor reused across any of the 60 selected documents)" if not vendor_isolation_relaxed_for else f"relaxed for: {vendor_isolation_relaxed_for}",
        "cluster_isolation": "global (no cluster_id reused across any of the 60 selected documents)" if not cluster_isolation_relaxed_for else f"relaxed for: {cluster_isolation_relaxed_for}",
        "vendor_isolation_relaxed_for": vendor_isolation_relaxed_for,
        "cluster_isolation_relaxed_for": cluster_isolation_relaxed_for,
        "diversity_coverage": diversity_coverage,
        "documented_limitations": documented_limitations,
        "split_assignment_method": "cycled [train, train, train, train, validation, test] over the greedy-novelty selection order",
        "selection_method": (
            "Same method as results/probe_dataset/document_selection_manifest.json (deterministic seeded shuffle, "
            "greedy novelty-maximizing selection under global vendor+cluster isolation), with SEED=20250114 "
            "(distinct from the probe corpus's 20240923) and the 20 probe-corpus docids added to the exclusion set."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    lines = ["# LoRA Corpus v1: Document Selection Report\n"]
    lines.append(
        f"Selected {N_TOTAL} documents from {len(candidates)} eligible candidates (of {len(raw_candidates)} scanned), "
        f"excluding {n_excluded_probe_corpus} probe-corpus v1/v1.1 docids, {n_excluded_legacy} legacy-excluded docids, "
        f"and {n_excluded_zero_amount} zero/unparseable-amount candidates.\n"
    )
    lines.append(f"Vendor isolation: **{manifest['vendor_isolation']}**\n")
    lines.append(f"Cluster isolation: **{manifest['cluster_isolation']}**\n")
    lines.append("## Diversity coverage\n")
    for feature_name, values in diversity_coverage.items():
        lines.append(f"- `{feature_name}`: {', '.join(values)}")
    lines.append("")
    if documented_limitations:
        lines.append("## Documented limitations\n")
        for note in documented_limitations:
            lines.append(f"- {note}")
        lines.append("")
    for split_name in ("train", "validation", "test"):
        docs = [d for d in documents if d["split"] == split_name]
        lines.append(f"## {split_name} ({len(docs)} documents)\n")
        lines.append("| docid | vendor | cluster | currency | amount |")
        lines.append("|---|---|---|---|---|")
        for d in docs:
            lines.append(f"| `{d['docid']}` | {d['vendor_name']} | {d['cluster_id']} | {d['currency']} | {d['amount_due']} |")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines))

    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Vendor isolation relaxed for: {vendor_isolation_relaxed_for}")
    print(f"Cluster isolation relaxed for: {cluster_isolation_relaxed_for}")


if __name__ == "__main__":
    main()
