"""Builds all 600 LoRA-corpus-v1 samples from the 60 documents
`scripts/select_lora_corpus_documents.py` selected, tokenizes and masks
each one (Section "SFT representation" of the spec), and runs every
leakage/shortcut/masking validation before writing a "ready" corpus.
Raises (refusing to write) if any validation fails -- no GPU is touched.

Writes, under results/lora_dataset/v1/ (never overwrites the probe
corpus or any other existing artifact):
  - rendered_samples.jsonl   (sample_id, docid, decision_point, messages,
                               gold_action_dict, gold_action_type,
                               prompt_sha256, target_sha256,
                               prompt_token_count, completion_token_count,
                               total_token_count -- NO evaluation labels)
  - labels.jsonl              (exposure_label/kind/split/etc -- NO prompt text)
  - split_manifest.json
  - template_manifest.json
  - lora_corpus_v1_manifest.json
  - lora_corpus_v1_report.md
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from tell.agent.actions import AgentAction, action_json_schema
from tell.agent.local_model import PINNED_SNAPSHOT_PATH
from tell.lora_dataset.masking import build_masked_example, render_gold_completion_text
from tell.lora_dataset.synthetic_ids import ACCOUNT_ID_PREFIX, derive_lora_synthetic_ids
from tell.lora_dataset.templates import LORA_ATTACK_KINDS, LORA_CLEAN_KINDS, LORA_SAMPLE_KIND_SPECS
from tell.lora_dataset.sample_builder import build_all_samples
from tell.probe_dataset.docile_extract import extract_document_facts
from pydantic import TypeAdapter
import re

SELECTION_MANIFEST_PATH = Path("/home/hp5/tell/results/lora_dataset/v1/document_selection_manifest.json")
OUTPUT_DIR = Path("/home/hp5/tell/results/lora_dataset/v1")
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
QUALITY_REPORT_PATH = OUTPUT_DIR / "lora_corpus_v1_report.md"

REPO_ROOT_MARKER = "/home/hp5/tell"
MAX_SEQ_LEN = 6144
ACTION_CLASS_CAP = 0.40
FORBIDDEN_IDENTIFIER_WORDS = ("ATTACKER", "MALICIOUS", "POISON", "FAKE")
ACCOUNT_ID_REGEX = re.compile(re.escape(ACCOUNT_ID_PREFIX) + r"[0-9A-F]{12}\b")
_ACTION_ADAPTER: TypeAdapter = TypeAdapter(AgentAction)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    selection = json.loads(SELECTION_MANIFEST_PATH.read_text())
    documents = selection["documents"]
    if len(documents) != 60:
        raise RuntimeError(f"Expected 60 selected documents, found {len(documents)}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)

    facts_by_docid = {d["docid"]: extract_document_facts(d["docid"], d["split"]) for d in documents}
    vendor_name_by_docid = {docid: f.vendor_name_display for docid, f in facts_by_docid.items()}
    split_docids_sorted = {
        split_name: sorted(d["docid"] for d in documents if d["split"] == split_name) for split_name in ("train", "validation", "test")
    }

    all_samples = []
    for doc in documents:
        facts = facts_by_docid[doc["docid"]]
        synthetic_ids = derive_lora_synthetic_ids(doc["docid"])
        all_samples.extend(
            build_all_samples(
                facts,
                synthetic_ids,
                split_docids_sorted=split_docids_sorted[doc["split"]],
                vendor_name_by_docid=vendor_name_by_docid,
            )
        )
    print(f"[build] built {len(all_samples)} samples across {len(documents)} documents")

    # ---- Tokenize/mask + hash each sample ----
    rendered_records = []
    prompt_hashes: dict[str, list[str]] = defaultdict(list)
    masking_rejections = []
    for sample, label in all_samples:
        masked = build_masked_example(
            tokenizer,
            sample_id=sample.sample_id,
            messages=list(sample.messages),
            gold_action_dict=sample.gold_action_dict,
            max_seq_len=MAX_SEQ_LEN,
        )
        if masked.rejected:
            masking_rejections.append({"sample_id": sample.sample_id, "reason": masked.reject_reason})
            continue

        prompt_text = tokenizer.apply_chat_template(list(sample.messages), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_sha256 = _sha256_text(prompt_text)
        target_text = render_gold_completion_text(sample.gold_action_dict)
        target_sha256 = _sha256_text(target_text)
        prompt_hashes[prompt_sha256].append(sample.sample_id)

        rendered_records.append(
            {
                "sample_id": sample.sample_id,
                "docid": sample.docid,
                "decision_point": sample.decision_point,
                "messages": list(sample.messages),
                "gold_action_dict": sample.gold_action_dict,
                "gold_action_type": sample.gold_action_type,
                "prompt_sha256": prompt_sha256,
                "target_sha256": target_sha256,
                "prompt_token_count": masked.prompt_token_count,
                "completion_token_count": masked.completion_token_count,
                "total_token_count": masked.total_token_count,
                "prefix_verified": masked.prefix_verified,
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
                "decision_point": label.decision_point,
                "gold_action_type": label.gold_action_type,
                "template_family_id": label.template_family_id,
                "hard_negative_for": label.hard_negative_for,
                "source_provenance": label.source_provenance,
                "prompt_profile": label.prompt_profile,
                "account_id_role": label.account_id_role,
            }
        )

    sample_id_to_label = {l.sample_id: l for _, l in all_samples}

    # =====================================================================
    # Pre-training validation. Any failure raises before writing anything.
    # =====================================================================
    errors: list[str] = []

    if masking_rejections:
        errors.append(f"{len(masking_rejections)} sample(s) rejected for masking/length reasons (never truncated): {masking_rejections}")

    if len(all_samples) != 600:
        errors.append(f"Expected 600 total samples built, got {len(all_samples)}")
    if len(rendered_records) != 600:
        errors.append(f"Expected 600 rendered (non-rejected) samples, got {len(rendered_records)}")

    n_clean = sum(1 for _, l in all_samples if l.exposure_label == 0)
    n_attack = sum(1 for _, l in all_samples if l.exposure_label == 1)
    if n_clean != 300 or n_attack != 300:
        errors.append(f"Expected 300 clean / 300 attack, got {n_clean} clean / {n_attack} attack")

    split_counts = Counter(l.split for _, l in all_samples)
    if split_counts.get("train") != 400 or split_counts.get("validation") != 100 or split_counts.get("test") != 100:
        errors.append(f"Expected 400/100/100 train/validation/test samples, got {dict(split_counts)}")

    for split_name in ("train", "validation", "test"):
        split_labels = [l for _, l in all_samples if l.split == split_name]
        n_c = sum(1 for l in split_labels if l.exposure_label == 0)
        n_a = sum(1 for l in split_labels if l.exposure_label == 1)
        if n_c != n_a:
            errors.append(f"{split_name}: class imbalance ({n_c} clean vs {n_a} attack, expected 50/50)")

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
        errors.append(f"Duplicate prompt hashes across samples: {duplicate_hashes}")

    evaluation_leak_markers = (
        [l.template_family_id for _, l in all_samples]
        + [k.kind for k in LORA_SAMPLE_KIND_SPECS]
        + ["exposure_label", "hard_negative_for", "account_id_role", "clean_negative_family"]
    )
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
            errors.append(f"{doc['docid']}: annotation hash changed since selection")

    # ---- Account-ID discipline (corpus-v1.1 style) ----
    for r in rendered_records:
        text_blob = json.dumps(r["messages"]).upper()
        for word in FORBIDDEN_IDENTIFIER_WORDS:
            if word in text_blob:
                errors.append(f"{r['sample_id']}: forbidden identifier word {word!r} present in rendered prompt")

    all_account_ids: dict[str, str] = {}
    for doc in documents:
        sids = derive_lora_synthetic_ids(doc["docid"])
        for role, account_id in (
            ("approved", sids.approved_beneficiary_account_id),
            ("unauthorized", sids.unauthorized_beneficiary_account_id),
            ("obsolete", sids.obsolete_beneficiary_account_id),
        ):
            if not ACCOUNT_ID_REGEX.fullmatch(account_id):
                errors.append(f"{doc['docid']}/{role}: account id {account_id!r} does not match the shared regex")
            if account_id in all_account_ids and all_account_ids[account_id] != doc["docid"]:
                errors.append(f"account id {account_id!r} collides between {doc['docid']} and {all_account_ids[account_id]}")
            all_account_ids[account_id] = doc["docid"]

    # ---- Action-schema validation: every gold action re-parses via the
    # real AgentAction TypeAdapter (extra sanity check beyond construction) ----
    for r in rendered_records:
        try:
            action = _ACTION_ADAPTER.validate_python(r["gold_action_dict"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{r['sample_id']}: gold_action_dict does not validate against AgentAction: {exc}")
            continue
        if action.action.value != r["gold_action_type"]:
            errors.append(f"{r['sample_id']}: gold_action_type {r['gold_action_type']!r} != re-parsed action {action.action.value!r}")

    # ---- Target-evidence consistency: propose_payment actions must cite
    # exactly what the rendered invoice/vendor-record observations show ----
    for r in rendered_records:
        if r["gold_action_type"] != "propose_payment":
            continue
        gold = r["gold_action_dict"]
        docid = r["docid"]
        if gold["invoice_document_id"] != docid:
            errors.append(f"{r['sample_id']}: propose_payment invoice_document_id != docid")
        if gold["invoice_document_id"] == gold["invoice_number"]:
            errors.append(f"{r['sample_id']}: invoice_document_id must differ from invoice_number (opaque id vs. human number)")
        if gold["evidence"]["invoice_document_id"] != gold["invoice_document_id"]:
            errors.append(f"{r['sample_id']}: evidence.invoice_document_id mismatch")
        # cross-check against the actually-rendered read_invoice/get_vendor_record observation content
        invoice_content = None
        vendor_content = None
        for m in r["messages"]:
            if m["role"] != "user" or "Tool result:" not in m["content"]:
                continue
            payload = json.loads(m["content"].split("Tool result:\n\n", 1)[1].split("\n\n[trust_boundary", 1)[0])
            if payload["tool_name"] == "read_invoice":
                invoice_content = payload["content"]
            elif payload["tool_name"] == "get_vendor_record":
                vendor_content = payload["content"]
        if invoice_content is None or vendor_content is None:
            errors.append(f"{r['sample_id']}: propose_payment example missing an observed read_invoice or get_vendor_record turn")
        else:
            if gold["beneficiary_account_id"] != vendor_content["beneficiary_account_id"]:
                errors.append(f"{r['sample_id']}: beneficiary_account_id not sourced from the observed vendor record")
            if gold["evidence"]["vendor_record_id"] != vendor_content["vendor_id"]:
                errors.append(f"{r['sample_id']}: evidence.vendor_record_id not sourced from the observed vendor record")
            if invoice_content["amount_due"] != facts_by_docid[docid].payable_amount_text:
                errors.append(f"{r['sample_id']}: observed invoice amount_due does not match the source document facts")

    # ---- Action-class distribution cap ----
    action_class_counts = Counter(r["gold_action_type"] for r in rendered_records)
    max_action_class, max_action_count = action_class_counts.most_common(1)[0]
    max_action_fraction = max_action_count / len(rendered_records)
    action_cap_exceeded = max_action_fraction > ACTION_CLASS_CAP
    if action_cap_exceeded:
        errors.append(
            f"Action class {max_action_class!r} is {max_action_fraction:.1%} of the corpus, exceeding the {ACTION_CLASS_CAP:.0%} cap "
            "-- per spec, stopping before training rather than proceeding silently."
        )

    if errors:
        raise RuntimeError("Pre-training validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"[validate] all {len(errors)} == 0 pre-training checks passed")

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
        "sample_kinds": [
            {"kind": s.kind, "class_label": s.class_label, "decision_point": s.decision_point, "gold_action_type": s.gold_action_type, "hard_negative_for": s.hard_negative_for}
            for s in LORA_SAMPLE_KIND_SPECS
        ],
        "template_family_ids_by_split": {
            split_name: sorted({l.template_family_id for _, l in all_samples if l.split == split_name}) for split_name in ("train", "validation", "test")
        },
        "wording_isolation_verified": all(len(v) == 1 for v in family_to_splits.values()),
        "action_class_distribution": {k: v / len(rendered_records) for k, v in action_class_counts.items()},
        "action_class_cap": ACTION_CLASS_CAP,
        "action_class_cap_exceeded": action_cap_exceeded,
        "decision_point_counts": dict(Counter(l.decision_point for _, l in all_samples)),
    }
    TEMPLATE_MANIFEST_PATH.write_text(json.dumps(template_manifest, indent=2))

    # ---- Report ----
    lines = ["# LoRA Corpus v1 Quality Report\n"]
    lines.append(f"Built {len(rendered_records)}/600 samples, 0 masking rejections, 0 pre-training validation errors.\n")

    lines.append("## Decision-point counts\n")
    for dp, n in sorted(Counter(l.decision_point for _, l in all_samples).items()):
        lines.append(f"- `{dp}`: {n}")
    lines.append("")

    lines.append("## Action-class distribution\n")
    lines.append("| Action | n | % |")
    lines.append("|---|---|---|")
    for action, n in action_class_counts.most_common():
        lines.append(f"| {action} | {n} | {n/len(rendered_records):.1%} |")
    lines.append(f"\nMax class: {max_action_class} at {max_action_fraction:.1%} (cap {ACTION_CLASS_CAP:.0%}).\n")

    lines.append("## Kind (clean/attack family) distribution\n")
    kind_counts = Counter(l.template_family_id.rsplit("__", 1)[0] for _, l in all_samples)
    for kind, n in sorted(kind_counts.items()):
        lines.append(f"- `{kind}`: {n}")
    lines.append("")

    lines.append("## Identifier occurrence by role and class\n")
    role_counts = Counter((l.split, "clean" if l.exposure_label == 0 else "attack", l.account_id_role) for _, l in all_samples)
    lines.append("| Split | Class | account_id_role | n |")
    lines.append("|---|---|---|---|")
    for (split_name, cls, role), n in sorted(role_counts.items()):
        lines.append(f"| {split_name} | {cls} | {role} | {n} |")
    lines.append("")

    lines.append("## Token-length statistics\n")
    prompt_lens = [r["prompt_token_count"] for r in rendered_records]
    completion_lens = [r["completion_token_count"] for r in rendered_records]
    lines.append(f"- Prompt tokens: mean {statistics.mean(prompt_lens):.0f}, range {min(prompt_lens)}-{max(prompt_lens)}")
    lines.append(f"- Completion tokens (incl. EOS): mean {statistics.mean(completion_lens):.0f}, range {min(completion_lens)}-{max(completion_lens)}")
    lines.append(f"- MAX_SEQ_LEN: {MAX_SEQ_LEN} (0 examples rejected for length)\n")

    lines.append("## Class length comparison (final observation message, chars)\n")
    lines.append("| Split | Class | n | mean chars | char range |")
    lines.append("|---|---|---|---|---|")
    max_char_diff = 0.0
    for split_name in ("train", "validation", "test"):
        for cls, label_val in (("clean", 0), ("attack", 1)):
            recs = [r for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == label_val]
            chars = [len(json.dumps(r["messages"][-1])) for r in recs]
            lines.append(f"| {split_name} | {cls} | {len(recs)} | {statistics.mean(chars):.0f} | {min(chars)}-{max(chars)} |")
        clean_chars = [len(json.dumps(r["messages"][-1])) for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == 0]
        attack_chars = [len(json.dumps(r["messages"][-1])) for r in rendered_records if sample_id_to_label[r["sample_id"]].split == split_name and sample_id_to_label[r["sample_id"]].exposure_label == 1]
        diff = abs(statistics.mean(clean_chars) - statistics.mean(attack_chars)) / max(statistics.mean(clean_chars), statistics.mean(attack_chars))
        max_char_diff = max(max_char_diff, diff)
    lines.append(f"\nMax relative mean-length difference (clean vs. attack) across splits: {max_char_diff:.1%}.\n")

    lines.append("## Explicit non-claim\n")
    lines.append(
        "This does not claim every lexical confound is eliminated -- only that account-identifier format/length, "
        "and (for the two attack surfaces that vary lexically) urgency/JSON/instruction-injection register, were "
        "deliberately matched against a corresponding clean hard negative sharing the same decision point and "
        "(where applicable) the same gold action type. Other, unmeasured lexical regularities may still exist.\n"
    )

    QUALITY_REPORT_PATH.write_text("\n".join(lines))

    print(f"Wrote {RENDERED_SAMPLES_PATH}")
    print(f"Wrote {LABELS_PATH}")
    print(f"Wrote {SPLIT_MANIFEST_PATH}")
    print(f"Wrote {TEMPLATE_MANIFEST_PATH}")
    print(f"Wrote {QUALITY_REPORT_PATH}")
    print(f"Max action class fraction: {max_action_fraction:.1%}")
    print(f"Prompt token range: {min(prompt_lens)}-{max(prompt_lens)}")


if __name__ == "__main__":
    main()
