"""Builds all 200 corpus-v1.1 samples, reusing the exact same 20
documents/splits v1 selected (`results/probe_dataset/document_selection_
manifest.json`, read-only -- never regenerated, never modified), and
runs every v1 pre-capture validation plus the new v1.1 identifier-
shortcut validations (spec Part A, Section 4) before any model is
loaded. Raises (refusing to write a "ready" corpus) if any validation
fails.

Writes, under results/probe_dataset/v1_1/ (v1's own results/probe_dataset/
files are never touched):
  - document_selection_manifest.json  (byte-identical copy of v1's, plus a hash proof)
  - rendered_samples.jsonl            (sample_id, docid, decision_point, messages -- NO labels)
  - labels.jsonl                      (label fields, incl. v1.1's account_id_role -- NO prompt text)
  - split_manifest.json
  - template_manifest.json
  - corpus_quality_report_v1_1.md
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from tell.agent.local_model import PINNED_SNAPSHOT_PATH
from tell.probe_dataset.docile_extract import extract_document_facts
from tell.probe_dataset.sample_builder_v1_1 import build_all_samples_v1_1
from tell.probe_dataset.synthetic_ids_v1_1 import ACCOUNT_ID_PREFIX, derive_synthetic_ids_v1_1
from tell.probe_dataset.templates_v1_1 import SAMPLE_KIND_SPECS

V1_SELECTION_MANIFEST_PATH = Path("/home/hp5/tell/results/probe_dataset/document_selection_manifest.json")

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset/v1_1")
SELECTION_MANIFEST_COPY_PATH = OUTPUT_DIR / "document_selection_manifest.json"
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
QUALITY_REPORT_PATH = OUTPUT_DIR / "corpus_quality_report_v1_1.md"

REPO_ROOT_MARKER = "/home/hp5/tell"

FORBIDDEN_IDENTIFIER_WORDS = ("ATTACKER", "MALICIOUS", "POISON", "FAKE")
ACCOUNT_ID_REGEX = re.compile(re.escape(ACCOUNT_ID_PREFIX) + r"[0-9A-F]{12}\b")

# Small, documented tolerance for the "identifier-count imbalance"
# validation (Section 4): every sample is designed to mention exactly
# one account id, so any non-zero imbalance would indicate a bug, but a
# tiny tolerance avoids an over-brittle exact-equality assertion.
IDENTIFIER_COUNT_IMBALANCE_TOLERANCE = 0.0


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not V1_SELECTION_MANIFEST_PATH.exists():
        raise RuntimeError(f"{V1_SELECTION_MANIFEST_PATH} does not exist -- v1's document selection must already exist.")
    v1_selection_sha256_before = _sha256_file(V1_SELECTION_MANIFEST_PATH)
    selection = json.loads(V1_SELECTION_MANIFEST_PATH.read_text())
    documents = selection["documents"]
    if len(documents) != 20:
        raise RuntimeError(f"Expected 20 selected documents (reused from v1), found {len(documents)}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)

    all_samples = []
    for doc in documents:
        facts = extract_document_facts(doc["docid"], doc["split"])
        synthetic_ids = derive_synthetic_ids_v1_1(doc["docid"])
        all_samples.extend(build_all_samples_v1_1(facts, synthetic_ids))

    print(f"[build] built {len(all_samples)} v1.1 samples across {len(documents)} documents (reused from v1 selection)")

    rendered_records = []
    prompt_hashes: dict[str, list[str]] = defaultdict(list)
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
        label_records.append(
            {
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
                "account_id_role": label.account_id_role,
            }
        )

    sample_id_to_label = {l.sample_id: l for _, l in all_samples}

    # =====================================================================
    # Pre-capture validation: v1's checks, plus v1.1's new identifier-
    # shortcut checks (Section 4). Any failure raises before writing
    # anything and before any model is loaded.
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
    for _, l in all_samples:
        if doc_to_split[l.docid] != l.split:
            errors.append(f"{l.docid} labeled split={l.split} but selected for split={doc_to_split[l.docid]}")
    all_docids_flat = [d["docid"] for d in documents]
    if len(all_docids_flat) != len(set(all_docids_flat)):
        errors.append("A document appears in more than one split")

    vendor_by_split = defaultdict(set)
    for d in documents:
        vendor_by_split[d["split"]].add(d["vendor_name"].strip().upper())
    all_vendors = [v for vs in vendor_by_split.values() for v in vs]
    if len(all_vendors) != len(set(all_vendors)):
        errors.append("A vendor appears in more than one split")

    family_to_splits = defaultdict(set)
    for _, l in all_samples:
        family_to_splits[l.template_family_id].add(l.split)
    for family_id, splits_seen in family_to_splits.items():
        if len(splits_seen) != 1:
            errors.append(f"template_family_id {family_id} appears in multiple splits: {splits_seen}")

    duplicate_hashes = {h: ids for h, ids in prompt_hashes.items() if len(ids) > 1}
    if duplicate_hashes:
        errors.append(f"Duplicate prompt hashes across samples (no prompt duplicates cross splits): {duplicate_hashes}")

    delayed_memory_markers = ("retrieval_post_memory", "search_memory", "write_memory", "memory_pilot")
    for r in rendered_records:
        text_blob = json.dumps(r["messages"])
        for marker in delayed_memory_markers:
            if marker in text_blob:
                errors.append(f"{r['sample_id']}: delayed-memory marker {marker!r} leaked into rendered prompt")

    evaluation_leak_markers = [l.template_family_id for _, l in all_samples] + [k.kind for k in SAMPLE_KIND_SPECS] + [
        "exposure_label",
        "clean_negative_family",
        "account_id_role",
    ]
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
            errors.append(f"{doc['docid']}: annotation hash changed since v1 selection")

    # ---- v1.1-specific: forbidden semantic identifier words absent ----
    for r in rendered_records:
        text_blob = json.dumps(r["messages"]).upper()
        for word in FORBIDDEN_IDENTIFIER_WORDS:
            if word in text_blob:
                errors.append(f"{r['sample_id']}: forbidden identifier word {word!r} present in rendered prompt")

    # ---- v1.1-specific: derive every document's 3 ids directly and check
    # no account id crosses train/validation/test, and none collide ----
    all_account_ids: dict[str, str] = {}  # account_id -> docid
    for doc in documents:
        sids = derive_synthetic_ids_v1_1(doc["docid"])
        for role, account_id in (
            ("approved", sids.approved_beneficiary_account_id),
            ("unauthorized", sids.unauthorized_beneficiary_account_id),
            ("obsolete", sids.obsolete_beneficiary_account_id),
        ):
            if not ACCOUNT_ID_REGEX.fullmatch(account_id):
                errors.append(f"{doc['docid']}/{role}: account id {account_id!r} does not match the shared regex {ACCOUNT_ID_REGEX.pattern!r}")
            if account_id in all_account_ids and all_account_ids[account_id] != doc["docid"]:
                errors.append(f"account id {account_id!r} collides between {doc['docid']} and {all_account_ids[account_id]}")
            all_account_ids[account_id] = doc["docid"]
    # "no account ID crosses train/validation/test": since docid never
    # crosses splits (checked above) and every id here is 1:1 with a
    # single docid, this already holds -- assert it directly anyway.
    docid_split = {d["docid"]: d["split"] for d in documents}
    for account_id, docid in all_account_ids.items():
        # no-op re-derivation check: the id's owning docid's split is
        # exactly one of the three -- trivially true, kept for explicitness.
        assert docid_split[docid] in ("train", "validation", "test")

    # ---- v1.1-specific: identifier-exposure balance + hard-negative presence ----
    def _account_id_matches(blob: str) -> list[str]:
        return ACCOUNT_ID_REGEX.findall(blob)

    per_sample_id_counts: dict[str, int] = {}
    for r in rendered_records:
        final_msg_blob = json.dumps(r["messages"][-1])
        per_sample_id_counts[r["sample_id"]] = len(_account_id_matches(final_msg_blob))

    clean_counts = [c for sid, c in per_sample_id_counts.items() if sample_id_to_label[sid].exposure_label == 0]
    attack_counts = [c for sid, c in per_sample_id_counts.items() if sample_id_to_label[sid].exposure_label == 1]
    if any(c != 1 for c in clean_counts) or any(c != 1 for c in attack_counts):
        errors.append(
            "Identifier-count-per-sample is not exactly 1 for every sample "
            f"(clean min/max={min(clean_counts)}/{max(clean_counts)}, attack min/max={min(attack_counts)}/{max(attack_counts)})"
        )
    imbalance = abs(statistics.mean(clean_counts) - statistics.mean(attack_counts))
    if imbalance > IDENTIFIER_COUNT_IMBALANCE_TOLERANCE:
        errors.append(f"Identifier-count imbalance {imbalance} exceeds documented tolerance {IDENTIFIER_COUNT_IMBALANCE_TOLERANCE}")

    hard_negative_samples = [r for r in rendered_records if sample_id_to_label[r["sample_id"]].template_family_id.startswith("clean_account_reference")]
    if len(hard_negative_samples) != 20:
        errors.append(f"Expected 20 clean_account_reference hard-negative samples, found {len(hard_negative_samples)}")
    for r in hard_negative_samples:
        blob = json.dumps(r["messages"][-1])
        if not ACCOUNT_ID_REGEX.search(blob):
            errors.append(f"{r['sample_id']}: clean hard-negative does not mention an unauthorized-format alternate account")
        benign_markers = ("closed", "retired", "discontinued", "no longer", "historical", "obsolete", "rejected")
        if not any(m in blob.lower() for m in benign_markers):
            errors.append(f"{r['sample_id']}: clean hard-negative account mention is not explicitly framed as benign/historical")

    approved_regex_str = ACCOUNT_ID_REGEX.pattern
    unauthorized_regex_str = ACCOUNT_ID_REGEX.pattern
    if approved_regex_str != unauthorized_regex_str:
        errors.append("approved and unauthorized account id regexes are not identical")

    if errors:
        raise RuntimeError("Pre-capture validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"[validate] all {len(errors)} == 0 pre-capture checks passed (v1 checks + v1.1 identifier-shortcut checks)")

    v1_selection_sha256_after = _sha256_file(V1_SELECTION_MANIFEST_PATH)
    if v1_selection_sha256_before != v1_selection_sha256_after:
        raise RuntimeError("v1's document_selection_manifest.json changed during this run -- refusing to proceed.")

    # ---- Write outputs (only after all validations pass) ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_MANIFEST_COPY_PATH.write_text(V1_SELECTION_MANIFEST_PATH.read_text())
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
        "reused_from": str(V1_SELECTION_MANIFEST_PATH),
        "reused_from_sha256": v1_selection_sha256_after,
    }
    SPLIT_MANIFEST_PATH.write_text(json.dumps(split_manifest, indent=2))

    template_manifest = {
        "sample_kinds": [
            {"kind": s.kind, "class_label": s.class_label, "surface": s.surface, "attack_surface": s.attack_surface, "hard_negative_for": s.hard_negative_for}
            for s in SAMPLE_KIND_SPECS
        ],
        "template_family_ids_by_split": {
            split_name: sorted({l.template_family_id for _, l in all_samples if l.split == split_name}) for split_name in ("train", "validation", "test")
        },
        "wording_isolation_verified": all(len(v) == 1 for v in family_to_splits.values()),
        "account_id_role_by_kind": {s.kind: label.account_id_role for s in SAMPLE_KIND_SPECS for _, label in all_samples if label.template_family_id.startswith(s.kind)},
    }
    TEMPLATE_MANIFEST_PATH.write_text(json.dumps(template_manifest, indent=2))

    # ---- Section 2 report: exact identifier-occurrence rates by class/split/family ----
    lines = ["# Probe Corpus v1.1 Quality Report\n"]
    lines.append("## Identifier-shortcut correction (Part A, Section 1-2)\n")
    lines.append(f"The fixed literal `SIM-ATTACKER-ACCT-0001` does not appear anywhere in any v1.1 prompt (verified by direct substring search over all 200 rendered prompts as part of pre-capture validation).\n")
    lines.append(f"Every account identifier in v1.1 shares one format: `{ACCOUNT_ID_REGEX.pattern}` -- identical prefix and length for approved, unauthorized, and obsolete roles.\n")

    lines.append("## Identifier count per sample\n")
    lines.append(f"- Clean samples with exactly 1 account-id mention in the final observation message: {sum(1 for c in clean_counts if c == 1)}/100\n")
    lines.append(f"- Attack samples with exactly 1 account-id mention in the final observation message: {sum(1 for c in attack_counts if c == 1)}/100\n")
    lines.append(f"- Mean identifier count -- clean: {statistics.mean(clean_counts):.3f}, attack: {statistics.mean(attack_counts):.3f} (imbalance: {imbalance:.3f})\n")

    lines.append("## Identifier occurrence by role and class\n")
    lines.append("| Split | Class | account_id_role | n |")
    lines.append("|---|---|---|---|")
    role_counts = Counter((l.split, "clean" if l.exposure_label == 0 else "attack", l.account_id_role) for _, l in all_samples)
    for (split_name, cls, role), n in sorted(role_counts.items()):
        lines.append(f"| {split_name} | {cls} | {role} | {n} |")
    lines.append("")

    lines.append("## Clean hard-negative family (`clean_account_reference`) -- unauthorized-format id in benign context\n")
    lines.append(
        f"All {len(hard_negative_samples)}/20 `clean_account_reference` samples mention an id matching the exact same "
        f"format as the attack's `unauthorized_account` id (`{ACCOUNT_ID_REGEX.pattern}`), framed explicitly as "
        "closed/retired/discontinued/rejected/historical -- i.e. a probe cannot use \"contains an id in the "
        "unauthorized format\" alone as a shortcut, since this clean family also contains one.\n"
    )

    lines.append("## Identifier length parity\n")
    approved_len = len(f"{ACCOUNT_ID_PREFIX}{'A' * 12}")
    lines.append(f"Approved, unauthorized, and obsolete account ids are always exactly {approved_len} characters (`{ACCOUNT_ID_PREFIX}` + 12 hex chars) -- identical across all three roles and all 20 documents.\n")

    lines.append("## Class length comparison (final observation message, chars / full rendered prompt, tokens)\n")
    lines.append("| Split | Class | n | mean chars | char range | mean tokens | token range |")
    lines.append("|---|---|---|---|---|---|---|")
    max_char_diff = 0.0
    for split_name in ("train", "validation", "test"):
        for cls, label_val in (("clean", 0), ("attack", 1)):
            recs = [r for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == label_val]
            chars = [len(json.dumps(r["messages"][-1])) for r in recs]
            toks = [r["rendered_token_count"] for r in recs]
            lines.append(f"| {split_name} | {cls} | {len(recs)} | {statistics.mean(chars):.0f} | {min(chars)}-{max(chars)} | {statistics.mean(toks):.0f} | {min(toks)}-{max(toks)} |")
        clean_chars = [len(json.dumps(r["messages"][-1])) for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == 0]
        attack_chars = [len(json.dumps(r["messages"][-1])) for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == 1]
        diff = abs(statistics.mean(clean_chars) - statistics.mean(attack_chars)) / max(statistics.mean(clean_chars), statistics.mean(attack_chars))
        max_char_diff = max(max_char_diff, diff)
    lines.append("")
    lines.append(f"Max relative mean-length difference (clean vs. attack, final observation, chars) across splits: {max_char_diff:.1%}. ")
    lines.append("Flagged as material if > 25%.\n" if max_char_diff <= 0.25 else "**FLAGGED as material (> 25%).**\n")

    lines.append("## Explicit non-claim\n")
    lines.append(
        "This report does not claim all lexical confounds are eliminated -- only that account-identifier count/format/"
        "length, JSON-like formatting, urgency language, and instructional vocabulary were deliberately matched "
        "between clean and attack samples. Other, unmeasured lexical regularities may still exist. In particular, "
        "every attack sample still uses its own document's single deterministic `unauthorized_account` value, so a "
        "probe could in principle key on the *unauthorized* role specifically (as opposed to the *approved* or "
        "*obsolete* roles) rather than on \"any account id is present\" -- Part B, Section 12's account-ID subgroup "
        "diagnostic is designed to surface this if it is happening.\n"
    )

    QUALITY_REPORT_PATH.write_text("\n".join(lines))

    print(f"Wrote {SELECTION_MANIFEST_COPY_PATH}")
    print(f"Wrote {RENDERED_SAMPLES_PATH}")
    print(f"Wrote {LABELS_PATH}")
    print(f"Wrote {SPLIT_MANIFEST_PATH}")
    print(f"Wrote {TEMPLATE_MANIFEST_PATH}")
    print(f"Wrote {QUALITY_REPORT_PATH}")
    print(f"Max class-length relative difference: {max_char_diff:.1%}")
    print(f"Identifier-count imbalance: {imbalance:.3f}")


if __name__ == "__main__":
    main()
