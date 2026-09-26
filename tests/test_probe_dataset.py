"""Tests for the probe activation corpus (`tell.probe_dataset` and its
generator scripts). No GPU, no model load: all checks read the
already-generated, deterministic corpus artifacts on disk, or exercise
the pure generation functions directly.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from tell.probe_dataset.docile_extract import extract_document_facts
from tell.probe_dataset.sample_builder import build_all_samples
from tell.probe_dataset.synthetic_ids import derive_synthetic_ids
from tell.probe_dataset.templates import SAMPLE_KIND_BY_NAME, SAMPLE_KIND_SPECS
from tell.probe_dataset.text_block import MAX_BLOCK_CHARS, build_docile_ocr_block, build_synthetic_overlay_block

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset")
SELECTION_MANIFEST_PATH = OUTPUT_DIR / "document_selection_manifest.json"
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
SAMPLE_INDEX_MAPPING_PATH = ACTIVATIONS_DIR / "sample_index_mapping.json"

REPO_ROOT_MARKER = "/home/hp5/tell"

# Historical artifacts this experiment reads from but must never write to.
# Hashes captured before this corpus's document-selection script ever ran.
HISTORICAL_HASHES = {
    Path("/home/hp5/tell/results/dataset_inspection/clean_candidate_scores.json"): "b32ecb88d0bc61985c670eead2e7a9b66e071f122109c2ca78595b412f2cdac9",
    Path("/home/hp5/tell/data/scenarios/clean/clean_04d531ca_v1.json"): "0f55d150024414bf1f8f31626c295b3c03f1ff7ff3747f40ab6fa2fe7efa64d0",
    Path("/home/hp5/tell/data/scenarios/clean/clean_002f9b82_v1.json"): "4fb48576ac8daf2806448a09cda4e2d9bcef5c0d86177f598ab25521f24b4129",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not generated in this checkout")


# ---------------------------------------------------------------------
# Deterministic document selection / fact extraction
# ---------------------------------------------------------------------


def test_document_selection_manifest_is_internally_consistent():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    docs = manifest["documents"]
    assert len(docs) == 20
    assert len({d["docid"] for d in docs}) == 20
    assert sum(1 for d in docs if d["split"] == "train") == 12
    assert sum(1 for d in docs if d["split"] == "validation") == 4
    assert sum(1 for d in docs if d["split"] == "test") == 4


def test_extract_document_facts_is_deterministic():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    doc = manifest["documents"][0]
    a = extract_document_facts(doc["docid"], doc["split"])
    b = extract_document_facts(doc["docid"], doc["split"])
    assert a == b  # frozen dataclass equality -- byte-for-byte reproducible


def test_document_facts_never_invents_business_fields():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    for doc in manifest["documents"][:3]:
        facts = extract_document_facts(doc["docid"], doc["split"])
        assert facts.vendor_name_raw == doc["vendor_name"]
        assert facts.invoice_number == doc["invoice_number"]


# ---------------------------------------------------------------------
# Class balance / split isolation / template isolation (read generated corpus)
# ---------------------------------------------------------------------


def test_exact_class_balance():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    assert len(labels) == 200
    assert sum(1 for l in labels if l["exposure_label"] == 0) == 100
    assert sum(1 for l in labels if l["exposure_label"] == 1) == 100


def test_split_counts_120_40_40():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    counts = defaultdict(int)
    for l in labels:
        counts[l["split"]] += 1
    assert counts["train"] == 120
    assert counts["validation"] == 40
    assert counts["test"] == 40


def test_five_clean_five_attack_per_document():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    per_doc = defaultdict(lambda: {"clean": 0, "attack": 0})
    for l in labels:
        per_doc[l["docid"]]["clean" if l["exposure_label"] == 0 else "attack"] += 1
    assert len(per_doc) == 20
    for docid, counts in per_doc.items():
        assert counts == {"clean": 5, "attack": 5}, docid


def test_document_isolation_across_splits():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    docids = [d["docid"] for d in manifest["documents"]]
    assert len(docids) == len(set(docids))


def test_vendor_isolation_across_splits():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    vendors = [d["vendor_name"].strip().upper() for d in manifest["documents"]]
    assert len(vendors) == len(set(vendors)), "vendor reused across selected documents"


def test_attack_template_wording_isolation_across_splits():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    family_to_splits = defaultdict(set)
    for l in labels:
        family_to_splits[l["template_family_id"]].add(l["split"])
    for family_id, splits_seen in family_to_splits.items():
        assert len(splits_seen) == 1, f"{family_id} appears in multiple splits: {splits_seen}"


def test_no_prompt_hash_duplicates():
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    hashes = [r["prompt_sha256"] for r in records]
    assert len(hashes) == len(set(hashes))


# ---------------------------------------------------------------------
# Clean hard-negative construction
# ---------------------------------------------------------------------


def test_clean_hard_negative_pairing_is_declared():
    pairing = {s.kind: s.hard_negative_for for s in SAMPLE_KIND_SPECS if s.class_label == 0}
    assert pairing == {
        "clean_standard": "email_fake_tool_result",
        "clean_urgent": "email_social_engineering",
        "clean_instructional_language": "email_direct_override",
        "clean_json_like": "invoice_json_priming",
        "clean_account_reference": "invoice_instruction_injection",
    }
    attack_kinds = {s.kind for s in SAMPLE_KIND_SPECS if s.class_label == 1}
    assert set(pairing.values()) == attack_kinds


def test_clean_and_paired_attack_share_surface():
    by_name = SAMPLE_KIND_BY_NAME
    for clean_kind, attack_kind in (
        ("clean_standard", "email_fake_tool_result"),
        ("clean_urgent", "email_social_engineering"),
        ("clean_instructional_language", "email_direct_override"),
        ("clean_json_like", "invoice_json_priming"),
        ("clean_account_reference", "invoice_instruction_injection"),
    ):
        assert by_name[clean_kind].surface == by_name[attack_kind].surface


# ---------------------------------------------------------------------
# DocILE vs. synthetic provenance; invoice-text observation validation
# ---------------------------------------------------------------------


def test_docile_ocr_block_provenance_and_bound():
    block = build_docile_ocr_block(block_id="x-ocr-0", vendor_address="123 Main St", line_item_descriptions=["Widget A", "Widget B"])
    assert block.source_type == "docile_ocr"
    assert block.provenance == "docile_annotation"
    assert block.trust_boundary == "untrusted"
    assert len(block.text) <= MAX_BLOCK_CHARS


def test_synthetic_overlay_block_provenance_and_bound():
    block = build_synthetic_overlay_block(block_id="x-overlay-0", text="benign note")
    assert block.source_type == "synthetic_overlay"
    assert block.provenance == "synthetic_controlled"
    assert block.trust_boundary == "untrusted"
    with pytest.raises(ValueError):
        build_synthetic_overlay_block(block_id="x-overlay-1", text="a" * (MAX_BLOCK_CHARS + 1))


def test_sample_email_and_invoice_provenance_distinguished():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    doc = manifest["documents"][0]
    facts = extract_document_facts(doc["docid"], doc["split"])
    sids = derive_synthetic_ids(doc["docid"])
    samples = {label.template_family_id.rsplit("__", 1)[0]: (sample, label) for sample, label in build_all_samples(facts, sids)}

    sample, _ = samples["clean_standard"]
    email_payload = json.loads(sample.messages[-1]["content"].split("Tool result:\n\n", 1)[1].rsplit("\n\n[trust_boundary", 1)[0])
    assert email_payload["provenance"]["provenance"] == "synthetic_controlled"

    sample, _ = samples["invoice_instruction_injection"]
    invoice_payload = json.loads(sample.messages[-1]["content"].split("Tool result:\n\n", 1)[1].rsplit("\n\n[trust_boundary", 1)[0])
    text_blocks = invoice_payload["content"]["text_blocks"]
    assert {b["source_type"] for b in text_blocks} == {"docile_ocr", "synthetic_overlay"}
    assert {b["provenance"] for b in text_blocks} == {"docile_annotation", "synthetic_controlled"}
    assert all(b["trust_boundary"] == "untrusted" for b in text_blocks)


# ---------------------------------------------------------------------
# No leakage: labels, delayed-memory markers, filesystem paths
# ---------------------------------------------------------------------


def test_no_evaluation_label_leakage_in_rendered_prompts():
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    forbidden = [s.kind for s in SAMPLE_KIND_SPECS] + ["exposure_label", "clean_negative_family", "attack_surface"]
    for r in records:
        blob = json.dumps(r["messages"])
        for marker in forbidden:
            assert marker not in blob, f"{r['sample_id']}: leaked {marker!r}"


def test_no_delayed_memory_markers_in_rendered_prompts():
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    markers = ("retrieval_post_memory", "search_memory", "write_memory", "memory_pilot")
    for r in records:
        blob = json.dumps(r["messages"])
        for marker in markers:
            assert marker not in blob


def test_no_filesystem_path_leakage_in_rendered_prompts():
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    for r in records:
        blob = json.dumps(r["messages"])
        assert REPO_ROOT_MARKER not in blob, f"{r['sample_id']}: filesystem path leaked"


def test_activation_metadata_has_no_label_fields():
    metadata_path = ACTIVATIONS_DIR / "capture_metadata.jsonl"
    _require(metadata_path)
    records = _load_jsonl(metadata_path)
    forbidden = ["exposure_label", "attack_surface", "template_family_id", "clean_negative_family", "expected_outcome", "evaluation_only"]
    for r in records:
        for key in forbidden:
            assert key not in r, f"{r['sample_id']}: activation metadata contains {key!r}"


# ---------------------------------------------------------------------
# SafeTensors row mapping
# ---------------------------------------------------------------------


def test_safetensors_row_mapping_consistent():
    _require(SAMPLE_INDEX_MAPPING_PATH)
    mapping = json.loads(SAMPLE_INDEX_MAPPING_PATH.read_text())
    from safetensors import safe_open

    for split, sample_ids in mapping.items():
        with safe_open(str(ACTIVATIONS_DIR / f"{split}_layer9.safetensors"), framework="pt") as f:
            tensor = f.get_tensor("activations")
        assert tensor.shape[0] == len(sample_ids)
        assert tensor.shape[1] == 4096
    all_ids = [sid for ids in mapping.values() for sid in ids]
    assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------
# Resumable shard behavior (capture script's own resumability logic)
# ---------------------------------------------------------------------


def test_capture_script_resume_detection(tmp_path, monkeypatch):
    import capture_probe_corpus_activations as capture_script  # scripts/ on pythonpath

    monkeypatch.setattr(capture_script, "TMP_PER_SAMPLE_DIR", tmp_path)
    assert capture_script._sample_done("sample-a") is False

    (tmp_path / "sample-a.safetensors").write_bytes(b"fake")
    assert capture_script._sample_done("sample-a") is False  # metadata still missing

    (tmp_path / "sample-a_metadata.json").write_text("{}")
    assert capture_script._sample_done("sample-a") is True

    assert capture_script._sample_done("sample-b") is False


# ---------------------------------------------------------------------
# Historical artifacts unchanged
# ---------------------------------------------------------------------


def test_historical_artifacts_unchanged():
    for path, expected_hash in HISTORICAL_HASHES.items():
        assert _sha256_file(path) == expected_hash, f"{path} changed"


# ---------------------------------------------------------------------
# No unit test in this module loads Qwen
# ---------------------------------------------------------------------


def test_no_test_in_this_module_loads_qwen():
    import ast

    tree = ast.parse(Path(__file__).read_text())
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "QwenLocalRuntime" not in calls
