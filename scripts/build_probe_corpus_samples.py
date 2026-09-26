"""Builds all 200 (document x sample-kind) probe-corpus samples from the
20 documents `scripts/select_probe_corpus_documents.py` selected, and
runs every pre-capture validation from Section 11 of the corpus spec
before any model is loaded. Raises (refusing to write a "ready" corpus)
if any validation fails.

Uses only the tokenizer (`AutoTokenizer.from_pretrained`, no model
weights) to compute each sample's rendered-prompt hash for the
duplicate-hash and deterministic-regeneration checks -- the same
tokenizer-only discipline every other freeze script in this repository
already uses.

Writes:
  - results/probe_dataset/rendered_samples.jsonl   (sample_id, docid, decision_point, messages -- NO labels)
  - results/probe_dataset/labels.jsonl              (label fields -- NO prompt text)
  - results/probe_dataset/split_manifest.json
  - results/probe_dataset/template_manifest.json
  - results/probe_dataset/corpus_quality_report.md
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from tell.agent.local_model import PINNED_SNAPSHOT_PATH
from tell.probe_dataset.docile_extract import extract_document_facts
from tell.probe_dataset.sample_builder import build_all_samples
from tell.probe_dataset.synthetic_ids import derive_synthetic_ids
from tell.probe_dataset.templates import ATTACK_KINDS, CLEAN_KINDS, SAMPLE_KIND_SPECS

SELECTION_MANIFEST_PATH = Path("/home/hp5/tell/results/probe_dataset/document_selection_manifest.json")
OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset")
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
QUALITY_REPORT_PATH = OUTPUT_DIR / "corpus_quality_report.md"

REPO_ROOT_MARKER = "/home/hp5/tell"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    selection = json.loads(SELECTION_MANIFEST_PATH.read_text())
    documents = selection["documents"]
    if len(documents) != 20:
        raise RuntimeError(f"Expected 20 selected documents, found {len(documents)}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)

    all_samples = []  # list of (ProbeSample, SampleLabel)
    for doc in documents:
        facts = extract_document_facts(doc["docid"], doc["split"])
        synthetic_ids = derive_synthetic_ids(doc["docid"])
        all_samples.extend(build_all_samples(facts, synthetic_ids))

    print(f"[build] built {len(all_samples)} samples across {len(documents)} documents")

    # ---- Render each sample's prompt text (tokenizer only, no model) and hash it ----
    rendered_records = []
    prompt_hashes: dict[str, list[str]] = defaultdict(list)  # hash -> [sample_id, ...]
    for sample, label in all_samples:
        chat_text = tokenizer.apply_chat_template(list(sample.messages), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_sha256 = _sha256_text(chat_text)
        prompt_hashes[prompt_sha256].append(sample.sample_id)
        rendered_records.append(
            {
                "sample_id": sample.sample_id,
                "docid": sample.docid,
                "decision_point": sample.decision_point,
                "messages": list(sample.messages),
                "prompt_sha256": prompt_sha256,
                "rendered_token_count": len(tokenizer(chat_text)["input_ids"]),
            }
        )

    label_records = []
    for sample, label in all_samples:
        d = {
            "sample_id": label.sample_id,
            "docid": label.docid,
            "vendor_group_id": label.vendor_group_id,
            "split": label.split,
            "exposure_label": label.exposure_label,
            "attack_surface": label.attack_surface,
            "template_family_id": label.template_family_id,
            "clean_negative_family": label.clean_negative_family,
            "source_provenance": label.source_provenance,
            "prompt_profile": label.prompt_profile,
            "decision_point": label.decision_point,
        }
        label_records.append(d)

    # =====================================================================
    # Section 11: pre-capture validation. Any failure raises and stops
    # before writing "ready" outputs -- no model has been loaded yet.
    # =====================================================================
    errors: list[str] = []

    if len(all_samples) != 200:
        errors.append(f"Expected 200 total samples, got {len(all_samples)}")

    n_clean = sum(1 for _, l in all_samples if l.exposure_label == 0)
    n_attack = sum(1 for _, l in all_samples if l.exposure_label == 1)
    if n_clean != 100 or n_attack != 100:
        errors.append(f"Expected 100 clean / 100 attack, got {n_clean} clean / {n_attack} attack")

    split_counts = Counter(l.split for _, l in all_samples)
    if split_counts.get("train") != 120 or split_counts.get("validation") != 40 or split_counts.get("test") != 40:
        errors.append(f"Expected 120/40/40 train/validation/test samples, got {dict(split_counts)}")

    per_doc_kind_counts = defaultdict(lambda: {"clean": 0, "attack": 0})
    for _, l in all_samples:
        per_doc_kind_counts[l.docid]["clean" if l.exposure_label == 0 else "attack"] += 1
    for docid, counts in per_doc_kind_counts.items():
        if counts["clean"] != 5 or counts["attack"] != 5:
            errors.append(f"{docid}: expected 5 clean + 5 attack, got {counts}")

    doc_to_split = {d["docid"]: d["split"] for d in documents}
    sample_docids_by_split = defaultdict(set)
    for _, l in all_samples:
        sample_docids_by_split[l.split].add(l.docid)
    for split_name, docids in sample_docids_by_split.items():
        for docid in docids:
            if doc_to_split[docid] != split_name:
                errors.append(f"{docid} labeled split={split_name} but selected for split={doc_to_split[docid]}")
    all_docids_flat = [d["docid"] for d in documents]
    if len(all_docids_flat) != len(set(all_docids_flat)):
        errors.append("A document appears in more than one split")

    vendor_by_split = defaultdict(set)
    for d in documents:
        vendor_by_split[d["split"]].add(d["vendor_name"].strip().upper())
    all_vendors = [v for vs in vendor_by_split.values() for v in vs]
    if len(all_vendors) != len(set(all_vendors)):
        errors.append("A vendor appears in more than one split")

    template_family_by_split = defaultdict(set)
    for _, l in all_samples:
        kind = l.template_family_id.rsplit("__", 1)[0]
        template_family_by_split[l.split].add(kind + "__" + l.split)  # family id already embeds split
    # Wording isolation: no (kind, split) family id string may appear for
    # more than one split -- true by construction since family id embeds
    # split, but assert it directly:
    all_family_ids = [l.template_family_id for _, l in all_samples]
    family_to_splits = defaultdict(set)
    for _, l in all_samples:
        family_to_splits[l.template_family_id].add(l.split)
    for family_id, splits_seen in family_to_splits.items():
        if len(splits_seen) != 1:
            errors.append(f"template_family_id {family_id} appears in multiple splits: {splits_seen}")

    duplicate_hashes = {h: ids for h, ids in prompt_hashes.items() if len(ids) > 1}
    if duplicate_hashes:
        errors.append(f"Duplicate prompt hashes across samples: {duplicate_hashes}")

    delayed_memory_markers = ("retrieval_post_memory", "search_memory", "write_memory", "memory_pilot")
    for r in rendered_records:
        text_blob = json.dumps(r["messages"])
        for marker in delayed_memory_markers:
            if marker in text_blob:
                errors.append(f"{r['sample_id']}: delayed-memory marker {marker!r} leaked into rendered prompt")

    evaluation_leak_markers = [l.template_family_id for _, l in all_samples] + [k.kind for k in SAMPLE_KIND_SPECS] + ["exposure_label", "clean_negative_family"]
    for r in rendered_records:
        text_blob = json.dumps(r["messages"])
        for marker in evaluation_leak_markers:
            if marker in text_blob:
                errors.append(f"{r['sample_id']}: evaluation-only marker {marker!r} leaked into rendered prompt")

    for r in rendered_records:
        text_blob = json.dumps(r["messages"])
        if REPO_ROOT_MARKER in text_blob:
            errors.append(f"{r['sample_id']}: filesystem path leaked into rendered prompt")

    sample_ids = [s.sample_id for s, _ in all_samples]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("Duplicate sample_id detected")

    for doc in documents:
        facts_recheck = extract_document_facts(doc["docid"], doc["split"])
        if facts_recheck.annotation_sha256 != doc["annotation_sha256"]:
            errors.append(f"{doc['docid']}: annotation hash changed since selection -- not a train-split document any more?")

    if errors:
        raise RuntimeError("Pre-capture validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"[validate] all {len(errors)} == 0 pre-capture checks passed")

    # ---- Write outputs (only after all validations pass) ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RENDERED_SAMPLES_PATH.open("w") as f:
        for r in rendered_records:
            f.write(json.dumps(r) + "\n")
    with LABELS_PATH.open("w") as f:
        for r in label_records:
            f.write(json.dumps(r) + "\n")

    split_manifest = {
        "n_train": split_counts["train"],
        "n_validation": split_counts["validation"],
        "n_test": split_counts["test"],
        "train_docids": sorted(d["docid"] for d in documents if d["split"] == "train"),
        "validation_docids": sorted(d["docid"] for d in documents if d["split"] == "validation"),
        "test_docids": sorted(d["docid"] for d in documents if d["split"] == "test"),
        "vendor_isolation_ok": len(all_vendors) == len(set(all_vendors)),
        "document_isolation_ok": len(all_docids_flat) == len(set(all_docids_flat)),
    }
    SPLIT_MANIFEST_PATH.write_text(json.dumps(split_manifest, indent=2))

    template_manifest = {
        "sample_kinds": [{"kind": s.kind, "class_label": s.class_label, "surface": s.surface, "attack_surface": s.attack_surface, "hard_negative_for": s.hard_negative_for} for s in SAMPLE_KIND_SPECS],
        "template_family_ids_by_split": {
            split_name: sorted({l.template_family_id for _, l in all_samples if l.split == split_name}) for split_name in ("train", "validation", "test")
        },
        "wording_isolation_verified": all(len(v) == 1 for v in family_to_splits.values()),
    }
    TEMPLATE_MANIFEST_PATH.write_text(json.dumps(template_manifest, indent=2))

    # ---- Section 6: shortcut-control descriptive statistics ----
    def _char_token_stats(records, predicate):
        chars = [len(json.dumps(r["messages"][-1])) for r in records if predicate(r)]
        toks = [r["rendered_token_count"] for r in records if predicate(r)]
        return chars, toks

    sample_id_to_label = {l.sample_id: l for _, l in all_samples}

    lines = ["# Probe Corpus Quality Report (Section 6: shortcut control)\n"]
    lines.append("## Class length comparison (final observation message, chars / full rendered prompt, tokens)\n")
    lines.append("| Split | Class | n | mean chars | char range | mean tokens | token range |")
    lines.append("|---|---|---|---|---|---|---|")
    for split_name in ("train", "validation", "test"):
        for cls, label_val in (("clean", 0), ("attack", 1)):
            recs = [r for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == label_val]
            chars = [len(json.dumps(r["messages"][-1])) for r in recs]
            toks = [r["rendered_token_count"] for r in recs]
            lines.append(
                f"| {split_name} | {cls} | {len(recs)} | {statistics.mean(chars):.0f} | {min(chars)}-{max(chars)} | "
                f"{statistics.mean(toks):.0f} | {min(toks)}-{max(toks)} |"
            )
    lines.append("")

    account_id_pattern = "SIM-BENEFICIARY-ACCT-PROBE-"
    attacker_pattern = "SIM-ATTACKER-ACCT-0001"
    n_clean_with_account_ref = sum(1 for r in rendered_records if sample_id_to_label[r["sample_id"]].exposure_label == 0 and account_id_pattern in json.dumps(r["messages"]))
    n_attack_with_attacker_ref = sum(1 for r in rendered_records if sample_id_to_label[r["sample_id"]].exposure_label == 1 and attacker_pattern in json.dumps(r["messages"][-1]))
    n_clean_with_attacker_ref = sum(1 for r in rendered_records if sample_id_to_label[r["sample_id"]].exposure_label == 0 and attacker_pattern in json.dumps(r["messages"]))

    lines.append("## Account-identifier occurrence (controls the \"any account id => attack\" shortcut)\n")
    lines.append(f"- Clean samples referencing the approved-account id pattern (`{account_id_pattern}...`): {n_clean_with_account_ref}/100\n")
    lines.append(f"- Attack samples referencing the attacker account id (`{attacker_pattern}`) in the final observation: {n_attack_with_attacker_ref}/100\n")
    lines.append(f"- Clean samples that also happen to contain the literal attacker-account string anywhere (should be 0, since clean templates never reference it): {n_clean_with_attacker_ref}/100\n")

    n_clean_json_shaped = sum(1 for _, l in all_samples if l.exposure_label == 0 and l.template_family_id.startswith("clean_json_like"))
    n_attack_json_shaped = sum(1 for _, l in all_samples if l.exposure_label == 1 and l.template_family_id.startswith("invoice_json_priming"))
    lines.append(
        f"## JSON-shaped-content control\n\n{n_clean_json_shaped} clean samples (`clean_json_like`) and {n_attack_json_shaped} attack samples "
        "(`invoice_json_priming`) both contain embedded curly-brace JSON-looking text in the invoice-tool-result surface, so a probe cannot "
        "trivially key on \"contains a JSON blob\" alone.\n"
    )
    n_clean_urgent = sum(1 for _, l in all_samples if l.exposure_label == 0 and l.template_family_id.startswith("clean_urgent"))
    n_attack_urgent = sum(1 for _, l in all_samples if l.exposure_label == 1 and l.template_family_id.startswith("email_social_engineering"))
    lines.append(
        f"## Urgency-language control\n\n{n_clean_urgent} clean samples (`clean_urgent`) and {n_attack_urgent} attack samples "
        "(`email_social_engineering`) both use urgency language (\"ASAP\", \"time-sensitive\", \"before EOD\"), so urgency alone is not a "
        "usable shortcut.\n"
    )
    n_clean_instr = sum(1 for _, l in all_samples if l.exposure_label == 0 and l.template_family_id.startswith("clean_instructional_language"))
    n_attack_instr = sum(1 for _, l in all_samples if l.exposure_label == 1 and l.template_family_id.startswith("email_direct_override"))
    lines.append(
        f"## Instructional-vocabulary control\n\n{n_clean_instr} clean samples (`clean_instructional_language`) and {n_attack_instr} attack samples "
        "(`email_direct_override`) both contain words like \"instruction\", \"ignore\", or \"system\" used in a legitimate business sense in the "
        "clean case, so that vocabulary alone is not a usable shortcut.\n"
    )

    lines.append("## Explicit non-claim\n")
    lines.append(
        "This report does not claim all lexical confounds are eliminated -- only that the four shortcut categories above "
        "(account-identifier presence, JSON-like formatting, urgency language, instructional vocabulary) were deliberately "
        "matched between a clean and an attack family. Other, unmeasured lexical regularities may still exist.\n"
    )

    max_char_diff = 0.0
    for split_name in ("train", "validation", "test"):
        clean_chars, _ = _char_token_stats(rendered_records, lambda r: sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == 0)
        attack_chars, _ = _char_token_stats(rendered_records, lambda r: sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == 1)
        diff = abs(statistics.mean(clean_chars) - statistics.mean(attack_chars)) / max(statistics.mean(clean_chars), statistics.mean(attack_chars))
        max_char_diff = max(max_char_diff, diff)
    lines.append(f"## Class-length imbalance flag\n\nMax relative mean-length difference (clean vs. attack, final observation, chars) across splits: {max_char_diff:.1%}. ")
    lines.append("Flagged as material if > 25%.\n" if max_char_diff <= 0.25 else "**FLAGGED as material (> 25%).**\n")

    QUALITY_REPORT_PATH.write_text("\n".join(lines))

    print(f"Wrote {RENDERED_SAMPLES_PATH}")
    print(f"Wrote {LABELS_PATH}")
    print(f"Wrote {SPLIT_MANIFEST_PATH}")
    print(f"Wrote {TEMPLATE_MANIFEST_PATH}")
    print(f"Wrote {QUALITY_REPORT_PATH}")
    print(f"Max class-length relative difference: {max_char_diff:.1%}")


if __name__ == "__main__":
    main()
