"""Tests for the LoRA corpus v1 (`tell.lora_dataset`) and its generator/
training scripts. No GPU, no model load for corpus-generation checks; the
few checks that need a tokenizer use `AutoTokenizer` only (never
`Qwen3ForCausalLM`/`QwenLocalRuntime`).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from tell.agent.actions import AgentAction
from tell.lora_dataset.masking import IGNORE_INDEX, build_masked_example, decode_non_masked_labels, render_gold_completion_text
from tell.lora_dataset.sample_builder import build_all_samples
from tell.lora_dataset.synthetic_ids import ACCOUNT_ID_PREFIX, derive_lora_synthetic_ids
from tell.lora_dataset.templates import LORA_SAMPLE_KIND_SPECS
from tell.probe_dataset.docile_extract import extract_document_facts
from pydantic import TypeAdapter

CORPUS_DIR = Path("/home/hp5/tell/results/lora_dataset/v1")
SELECTION_MANIFEST_PATH = CORPUS_DIR / "document_selection_manifest.json"
RENDERED_SAMPLES_PATH = CORPUS_DIR / "rendered_samples.jsonl"
LABELS_PATH = CORPUS_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = CORPUS_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = CORPUS_DIR / "template_manifest.json"

SMOKE_SUBSET_PATH = Path("/home/hp5/tell/results/lora_training/smoke_v1/smoke_subset_manifest.json")
DIAGNOSTIC_SUBSET_PATH = Path("/home/hp5/tell/results/lora_training/smoke_v1/diagnostic_subset_manifest.json")
PRETRAIN_CHECKS_PATH = Path("/home/hp5/tell/results/lora_training/smoke_v1/pretrain_checks.json")

REPO_ROOT_MARKER = "/home/hp5/tell"
ACCOUNT_ID_REGEX = re.compile(re.escape(ACCOUNT_ID_PREFIX) + r"[0-9A-F]{12}\b")
_ACTION_ADAPTER: TypeAdapter = TypeAdapter(AgentAction)


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not generated in this checkout")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# ---------------------------------------------------------------------
# Deterministic corpus generation
# ---------------------------------------------------------------------


def test_synthetic_ids_deterministic_and_distinct():
    a1 = derive_lora_synthetic_ids("docid-x")
    a2 = derive_lora_synthetic_ids("docid-x")
    assert a1 == a2
    ids = {a1.approved_beneficiary_account_id, a1.unauthorized_beneficiary_account_id, a1.obsolete_beneficiary_account_id}
    assert len(ids) == 3
    b1 = derive_lora_synthetic_ids("docid-y")
    assert a1.approved_beneficiary_account_id != b1.approved_beneficiary_account_id


def test_sample_builder_deterministic_regeneration():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    doc = manifest["documents"][0]
    train_docids = sorted(d["docid"] for d in manifest["documents"] if d["split"] == doc["split"])
    vendor_by_docid = {d["docid"]: extract_document_facts(d["docid"], d["split"]).vendor_name_display for d in manifest["documents"] if d["split"] == doc["split"]}
    facts = extract_document_facts(doc["docid"], doc["split"])
    sids = derive_lora_synthetic_ids(doc["docid"])
    a = build_all_samples(facts, sids, split_docids_sorted=train_docids, vendor_name_by_docid=vendor_by_docid)
    b = build_all_samples(facts, sids, split_docids_sorted=train_docids, vendor_name_by_docid=vendor_by_docid)
    assert [s.gold_action_dict for s, _ in a] == [s.gold_action_dict for s, _ in b]
    assert [s.messages for s, _ in a] == [s.messages for s, _ in b]


# ---------------------------------------------------------------------
# Group isolation / class balance
# ---------------------------------------------------------------------


def test_document_selection_group_isolated_60_docs_40_10_10():
    _require(SELECTION_MANIFEST_PATH)
    manifest = json.loads(SELECTION_MANIFEST_PATH.read_text())
    docs = manifest["documents"]
    assert len(docs) == 60
    assert len({d["docid"] for d in docs}) == 60
    assert sum(1 for d in docs if d["split"] == "train") == 40
    assert sum(1 for d in docs if d["split"] == "validation") == 10
    assert sum(1 for d in docs if d["split"] == "test") == 10
    vendors = [d["vendor_name"].strip().upper() for d in docs]
    assert len(vendors) == len(set(vendors))
    clusters = [d["cluster_id"] for d in docs]
    assert len(clusters) == len(set(clusters))


def test_no_overlap_with_probe_corpus_documents():
    _require(SELECTION_MANIFEST_PATH)
    probe_manifest_path = Path("/home/hp5/tell/results/probe_dataset/document_selection_manifest.json")
    _require(probe_manifest_path)
    lora_docids = {d["docid"] for d in json.loads(SELECTION_MANIFEST_PATH.read_text())["documents"]}
    probe_docids = {d["docid"] for d in json.loads(probe_manifest_path.read_text())["documents"]}
    assert lora_docids.isdisjoint(probe_docids)


def test_exact_class_balance_and_split_counts():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    assert len(labels) == 600
    assert sum(1 for l in labels if l["exposure_label"] == 0) == 300
    assert sum(1 for l in labels if l["exposure_label"] == 1) == 300
    counts = defaultdict(int)
    for l in labels:
        counts[l["split"]] += 1
    assert counts["train"] == 400 and counts["validation"] == 100 and counts["test"] == 100


def test_class_balance_per_split():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    for split_name in ("train", "validation", "test"):
        split_labels = [l for l in labels if l["split"] == split_name]
        n_c = sum(1 for l in split_labels if l["exposure_label"] == 0)
        n_a = sum(1 for l in split_labels if l["exposure_label"] == 1)
        assert n_c == n_a, split_name


def test_five_clean_five_attack_per_document():
    _require(LABELS_PATH)
    labels = _load_jsonl(LABELS_PATH)
    per_doc = defaultdict(lambda: {"clean": 0, "attack": 0})
    for l in labels:
        per_doc[l["docid"]]["clean" if l["exposure_label"] == 0 else "attack"] += 1
    assert len(per_doc) == 60
    for docid, counts in per_doc.items():
        assert counts == {"clean": 5, "attack": 5}, docid


def test_no_template_wording_reused_across_splits():
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
# Account-ID balance / neutrality
# ---------------------------------------------------------------------


def test_account_ids_neutral_format_no_forbidden_words():
    ids = derive_lora_synthetic_ids("some-docid")
    forbidden = ("ATTACKER", "MALICIOUS", "POISON", "FAKE")
    for account_id in (ids.approved_beneficiary_account_id, ids.unauthorized_beneficiary_account_id, ids.obsolete_beneficiary_account_id):
        assert ACCOUNT_ID_REGEX.fullmatch(account_id)
        for word in forbidden:
            assert word not in account_id.upper()


def test_no_account_id_crosses_splits():
    _require(SPLIT_MANIFEST_PATH)
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text())
    ids_by_split: dict[str, set[str]] = {}
    for split_name, key in (("train", "train_docids"), ("validation", "validation_docids"), ("test", "test_docids")):
        s = set()
        for docid in split_manifest[key]:
            sids = derive_lora_synthetic_ids(docid)
            s |= {sids.approved_beneficiary_account_id, sids.unauthorized_beneficiary_account_id, sids.obsolete_beneficiary_account_id}
        ids_by_split[split_name] = s
    assert ids_by_split["train"].isdisjoint(ids_by_split["validation"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["validation"].isdisjoint(ids_by_split["test"])


def test_no_lora_id_collides_with_probe_corpus_id():
    """The LoRA corpus uses a different prefix (SIM-ACCT-LORA- vs.
    SIM-ACCT-PROBE-) so no id can accidentally collide across corpora."""
    lora_ids = derive_lora_synthetic_ids("shared-docid-for-test")
    assert "LORA" in lora_ids.approved_beneficiary_account_id
    assert "PROBE" not in lora_ids.approved_beneficiary_account_id


# ---------------------------------------------------------------------
# Evaluation-leakage absence
# ---------------------------------------------------------------------


def test_no_evaluation_label_leakage_in_rendered_prompts():
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    forbidden = [s.kind for s in LORA_SAMPLE_KIND_SPECS] + ["exposure_label", "hard_negative_for", "account_id_role"]
    for r in records:
        blob = json.dumps(r["messages"])
        for marker in forbidden:
            assert marker not in blob, f"{r['sample_id']}: leaked {marker!r}"


def test_no_filesystem_path_leakage_in_rendered_prompts():
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    for r in records:
        blob = json.dumps(r["messages"])
        assert REPO_ROOT_MARKER not in blob


def test_split_name_never_appears_as_a_labeled_field():
    """The split name string can legitimately appear as ordinary English
    text inside a real DocILE vendor name/address/line-item (e.g. "Test
    Corp") -- that is not a leak. What must never happen is the split
    name appearing as this corpus's own JSON key/value (the way it does
    in labels.jsonl), which would mean the build script accidentally
    wrote evaluation bookkeeping into the model-visible prompt."""
    _require(RENDERED_SAMPLES_PATH)
    for r in _load_jsonl(RENDERED_SAMPLES_PATH):
        blob = json.dumps(r["messages"])
        for marker in ('"split":', "'split':", '"split_name"'):
            assert marker not in blob


# ---------------------------------------------------------------------
# Action-schema validation / document-id vs invoice-number / evidence
# ---------------------------------------------------------------------


def test_every_gold_action_validates_against_agent_action_schema():
    _require(RENDERED_SAMPLES_PATH)
    for r in _load_jsonl(RENDERED_SAMPLES_PATH):
        action = _ACTION_ADAPTER.validate_python(r["gold_action_dict"])
        assert action.action.value == r["gold_action_type"]


def test_propose_payment_document_id_differs_from_invoice_number():
    _require(RENDERED_SAMPLES_PATH)
    for r in _load_jsonl(RENDERED_SAMPLES_PATH):
        if r["gold_action_type"] != "propose_payment":
            continue
        gold = r["gold_action_dict"]
        assert gold["invoice_document_id"] == r["docid"]
        assert gold["invoice_document_id"] != gold["invoice_number"]


def test_propose_payment_evidence_matches_action_fields():
    _require(RENDERED_SAMPLES_PATH)
    for r in _load_jsonl(RENDERED_SAMPLES_PATH):
        if r["gold_action_type"] != "propose_payment":
            continue
        gold = r["gold_action_dict"]
        assert gold["evidence"]["invoice_document_id"] == gold["invoice_document_id"]


# ---------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------


def test_masking_boundary_prompt_tokens_all_ignored():
    from transformers import AutoTokenizer

    from tell.agent.local_model import PINNED_SNAPSHOT_PATH

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    gold = {"action": "read_email", "message_id": "SIM-MSG-TEST-0001"}
    masked = build_masked_example(tokenizer, sample_id="test-1", messages=messages, gold_action_dict=gold, max_seq_len=6144)
    assert not masked.rejected
    assert masked.labels[: masked.prompt_token_count] == [IGNORE_INDEX] * masked.prompt_token_count
    assert IGNORE_INDEX not in masked.labels[masked.prompt_token_count :]


def test_masking_decodes_to_exact_gold_completion():
    from transformers import AutoTokenizer

    from tell.agent.local_model import PINNED_SNAPSHOT_PATH

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    gold = {"action": "get_vendor_record", "vendor_id": "SIM-VENDOR-LORA-TEST"}
    masked = build_masked_example(tokenizer, sample_id="test-2", messages=messages, gold_action_dict=gold, max_seq_len=6144)
    decoded = decode_non_masked_labels(tokenizer, masked.labels)
    assert decoded == render_gold_completion_text(gold) + tokenizer.eos_token


def test_no_truncation_overlength_example_is_rejected_not_cut():
    from transformers import AutoTokenizer

    from tell.agent.local_model import PINNED_SNAPSHOT_PATH

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi " * 5000}]
    gold = {"action": "read_email", "message_id": "SIM-MSG-TEST-0001"}
    masked = build_masked_example(tokenizer, sample_id="test-overlength", messages=messages, gold_action_dict=gold, max_seq_len=100)
    assert masked.rejected is True
    assert masked.input_ids == []  # never a silently truncated partial sequence
    assert "max_seq_len" in masked.reject_reason


def test_corpus_reports_zero_masking_rejections():
    _require(TEMPLATE_MANIFEST_PATH)
    _require(RENDERED_SAMPLES_PATH)
    records = _load_jsonl(RENDERED_SAMPLES_PATH)
    assert len(records) == 600
    assert all(r["prefix_verified"] for r in records)


# ---------------------------------------------------------------------
# Action-class distribution cap
# ---------------------------------------------------------------------


def test_action_class_distribution_under_cap():
    _require(TEMPLATE_MANIFEST_PATH)
    template_manifest = json.loads(TEMPLATE_MANIFEST_PATH.read_text())
    assert template_manifest["action_class_cap_exceeded"] is False
    assert max(template_manifest["action_class_distribution"].values()) <= template_manifest["action_class_cap"]


# ---------------------------------------------------------------------
# Smoke-subset / diagnostic-subset selection
# ---------------------------------------------------------------------


def test_smoke_subset_32_balanced_two_plus_surfaces_train_only():
    _require(SMOKE_SUBSET_PATH)
    _require(LABELS_PATH)
    smoke = json.loads(SMOKE_SUBSET_PATH.read_text())
    assert smoke["n_examples"] == 32
    assert smoke["n_clean"] == 16 and smoke["n_attack"] == 16
    assert len(smoke["attack_surfaces_present"]) >= 2
    labels = {l["sample_id"]: l for l in _load_jsonl(LABELS_PATH)}
    assert all(labels[sid]["split"] == "train" for sid in smoke["sample_ids"])
    assert len(set(smoke["sample_ids"])) == 32


def test_diagnostic_subset_8_balanced_validation_only_disjoint_from_smoke():
    _require(DIAGNOSTIC_SUBSET_PATH)
    _require(SMOKE_SUBSET_PATH)
    _require(LABELS_PATH)
    diagnostic = json.loads(DIAGNOSTIC_SUBSET_PATH.read_text())
    smoke = json.loads(SMOKE_SUBSET_PATH.read_text())
    assert diagnostic["n_examples"] == 8
    assert diagnostic["n_clean"] == 4 and diagnostic["n_attack"] == 4
    labels = {l["sample_id"]: l for l in _load_jsonl(LABELS_PATH)}
    assert all(labels[sid]["split"] == "validation" for sid in diagnostic["sample_ids"])
    assert set(diagnostic["sample_ids"]).isdisjoint(set(smoke["sample_ids"]))


def test_diagnostic_subset_never_touches_test_split():
    _require(DIAGNOSTIC_SUBSET_PATH)
    _require(LABELS_PATH)
    diagnostic = json.loads(DIAGNOSTIC_SUBSET_PATH.read_text())
    labels = {l["sample_id"]: l for l in _load_jsonl(LABELS_PATH)}
    assert all(labels[sid]["split"] != "test" for sid in diagnostic["sample_ids"])


# ---------------------------------------------------------------------
# Adapter configuration validation
# ---------------------------------------------------------------------


def test_adapter_config_validation():
    config_path = Path("/home/hp5/tell/configs/lora/smoke_v1_config.json")
    _require(config_path)
    config = json.loads(config_path.read_text())
    lora_cfg = config["lora_config"]
    assert lora_cfg["r"] == 8
    assert lora_cfg["lora_alpha"] == 16
    assert set(lora_cfg["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    train_cfg = config["training_config"]
    assert train_cfg["batch_size"] == 1
    assert train_cfg["gradient_accumulation_steps"] == 4
    assert train_cfg["n_optimizer_steps"] == 20


def test_pretrain_checks_every_trainable_param_is_adapter():
    _require(PRETRAIN_CHECKS_PATH)
    checks = json.loads(PRETRAIN_CHECKS_PATH.read_text())
    assert checks["every_trainable_param_is_adapter"] is True
    assert checks["base_model_frozen"] is True
    assert checks["non_adapter_trainable_param_names"] == []
    assert 0 < checks["trainable_percentage"] < 1.0  # LoRA over attention projections is a tiny fraction of 8B params
    assert checks["sample_batch_check"]["non_masked_labels_decode_to_expected_gold_action"] is True


# ---------------------------------------------------------------------
# Evaluation metric calculation
# ---------------------------------------------------------------------


def test_evaluation_metric_calculation_pure_function():
    import sys

    sys.path.insert(0, "/home/hp5/tell/scripts")
    from run_lora_diagnostic_eval import _evaluate_predictions

    predictions = [
        {"exposure_label": 0, "gold_action_type": "read_invoice", "parsed_outcome": "valid", "predicted_action_type": "read_invoice", "is_safe_response": True},
        {"exposure_label": 1, "gold_action_type": "request_review", "parsed_outcome": "valid", "predicted_action_type": "request_review", "is_safe_response": True},
        {"exposure_label": 1, "gold_action_type": "get_vendor_record", "parsed_outcome": "malformed_json", "predicted_action_type": None, "is_safe_response": False},
    ]
    metrics = _evaluate_predictions(predictions, {})
    assert metrics["n_examples"] == 3
    assert abs(metrics["valid_structured_action_rate"] - 2 / 3) < 1e-9
    assert abs(metrics["exact_action_type_match_rate"] - 2 / 3) < 1e-9
    assert abs(metrics["attacked_safe_action_rate"] - 1 / 2) < 1e-9


def test_diagnostic_results_no_full_collapse():
    results_path = Path("/home/hp5/tell/results/lora_training/smoke_v1/diagnostic_eval_results.json")
    _require(results_path)
    results = json.loads(results_path.read_text())
    assert results["adapter_reload_ok"] is True
    for side in ("base", "adapter"):
        m = results[side]["metrics"]
        assert not (m["always_review_collapse_suspected"] and m["n_examples"] > 1)


# ---------------------------------------------------------------------
# No test in this module loads the model
# ---------------------------------------------------------------------


def test_no_test_in_this_module_loads_qwen_model():
    tree = ast.parse(Path(__file__).read_text())
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "QwenLocalRuntime" not in calls
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    assert "Qwen3ForCausalLM" not in imported_names
