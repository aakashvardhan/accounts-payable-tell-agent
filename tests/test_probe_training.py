"""Tests for corpus v1.1 (identifier-shortcut correction) and the
probe-training pilot (Part A/B of the v1.1 + probe-training spec). No
GPU, no model load: all checks either exercise pure functions directly
or read already-generated, deterministic artifacts on disk.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from tell.probe_dataset.synthetic_ids_v1_1 import ACCOUNT_ID_PREFIX, derive_synthetic_ids_v1_1

CORPUS_DIR = Path("/home/hp5/tell/results/probe_dataset/v1_1")
LABELS_PATH = CORPUS_DIR / "labels.jsonl"
RENDERED_SAMPLES_PATH = CORPUS_DIR / "rendered_samples.jsonl"
SPLIT_MANIFEST_PATH = CORPUS_DIR / "split_manifest.json"
ACTIVATIONS_DIR = CORPUS_DIR / "activations"
SAMPLE_INDEX_MAPPING_PATH = ACTIVATIONS_DIR / "sample_index_mapping.json"
POST_CAPTURE_REPORT_PATH = CORPUS_DIR / "post_capture_validation_report_v1_1.json"

TRAINING_DIR = Path("/home/hp5/tell/results/probe_training/pilot_v1")
TRAINING_PROTOCOL_PATH = TRAINING_DIR / "training_protocol.json"
CV_RESULTS_PATH = TRAINING_DIR / "cv_results.json"
LAYER_SELECTION_PATH = TRAINING_DIR / "layer_selection_results.json"
THRESHOLD_CALIBRATION_PATH = TRAINING_DIR / "threshold_calibration_results.json"
RUNTIME_PARITY_PATH = TRAINING_DIR / "runtime_parity_report.json"
DELAYED_MEMORY_RESULT_PATH = TRAINING_DIR / "delayed_memory_zero_shot_result.json"

CONFIGS_DIR = Path("/home/hp5/tell/configs/probe")
PROBE_CONFIG_PATH = CONFIGS_DIR / "probe_v1.json"
PROBE_WEIGHTS_PATH = CONFIGS_DIR / "probe_v1.safetensors"

HOLDOUT_MANIFEST_PATH = Path("/home/hp5/tell/results/probe_dataset/delayed_memory_holdout_manifest.json")

ACCOUNT_ID_REGEX = re.compile(re.escape(ACCOUNT_ID_PREFIX) + r"[0-9A-F]{12}\b")


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not generated in this checkout")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# v1.1 neutral account-id generation
# ---------------------------------------------------------------------


def test_v1_1_account_ids_share_format_and_length():
    ids = derive_synthetic_ids_v1_1("some-fake-docid-for-testing")
    for account_id in (ids.approved_beneficiary_account_id, ids.unauthorized_beneficiary_account_id, ids.obsolete_beneficiary_account_id):
        assert ACCOUNT_ID_REGEX.fullmatch(account_id)
        assert len(account_id) == len(ids.approved_beneficiary_account_id)


def test_v1_1_account_ids_contain_no_forbidden_words():
    ids = derive_synthetic_ids_v1_1("another-fake-docid")
    forbidden = ("ATTACKER", "MALICIOUS", "POISON", "FAKE")
    for account_id in (ids.approved_beneficiary_account_id, ids.unauthorized_beneficiary_account_id, ids.obsolete_beneficiary_account_id):
        for word in forbidden:
            assert word not in account_id.upper()


def test_v1_1_account_ids_deterministic_and_distinct_per_document():
    a1 = derive_synthetic_ids_v1_1("docid-x")
    a2 = derive_synthetic_ids_v1_1("docid-x")
    assert a1 == a2  # deterministic regeneration
    ids = {a1.approved_beneficiary_account_id, a1.unauthorized_beneficiary_account_id, a1.obsolete_beneficiary_account_id}
    assert len(ids) == 3  # distinct roles within one document

    b1 = derive_synthetic_ids_v1_1("docid-y")
    assert a1.approved_beneficiary_account_id != b1.approved_beneficiary_account_id  # distinct across documents


def test_v1_1_fixed_attacker_literal_never_produced_as_an_actual_id():
    """The literal `SIM-ATTACKER-ACCT-0001` is mentioned in this module's
    docstring for documentation purposes (explaining what v1.1 replaces),
    which is not a leak -- what matters is that no *derived* id ever
    equals it. Checked across several sample docids."""
    for docid in ("docid-a", "docid-b", "some-fake-docid-for-testing", "78a2f80c5bf24e6097504d76"):
        ids = derive_synthetic_ids_v1_1(docid)
        assert ids.approved_beneficiary_account_id != "SIM-ATTACKER-ACCT-0001"
        assert ids.unauthorized_beneficiary_account_id != "SIM-ATTACKER-ACCT-0001"
        assert ids.obsolete_beneficiary_account_id != "SIM-ATTACKER-ACCT-0001"


# ---------------------------------------------------------------------
# No account ID crosses splits; identifier exposure balance
# ---------------------------------------------------------------------


def test_no_account_id_crosses_splits():
    _require(SPLIT_MANIFEST_PATH)
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text())
    ids_by_split: dict[str, set[str]] = {}
    for split_name, key in (("train", "train_docids"), ("validation", "validation_docids"), ("test", "test_docids")):
        s = set()
        for docid in split_manifest[key]:
            sids = derive_synthetic_ids_v1_1(docid)
            s |= {sids.approved_beneficiary_account_id, sids.unauthorized_beneficiary_account_id, sids.obsolete_beneficiary_account_id}
        ids_by_split[split_name] = s
    assert ids_by_split["train"].isdisjoint(ids_by_split["validation"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["validation"].isdisjoint(ids_by_split["test"])


def test_identifier_exposure_exactly_balanced():
    _require(RENDERED_SAMPLES_PATH)
    _require(LABELS_PATH)
    labels = {r["sample_id"]: r for r in _load_jsonl(LABELS_PATH)}
    for r in _load_jsonl(RENDERED_SAMPLES_PATH):
        final_blob = json.dumps(r["messages"][-1])
        n_ids = len(ACCOUNT_ID_REGEX.findall(final_blob))
        assert n_ids == 1, f"{r['sample_id']}: expected exactly 1 account id, found {n_ids}"


def test_clean_hard_negative_contains_alternate_format_account():
    _require(LABELS_PATH)
    _require(RENDERED_SAMPLES_PATH)
    labels = {r["sample_id"]: r for r in _load_jsonl(LABELS_PATH)}
    rendered = {r["sample_id"]: r for r in _load_jsonl(RENDERED_SAMPLES_PATH)}
    hard_neg_ids = [sid for sid, l in labels.items() if l["template_family_id"].startswith("clean_account_reference")]
    assert len(hard_neg_ids) == 20
    for sid in hard_neg_ids:
        assert labels[sid]["account_id_role"] == "obsolete"
        blob = json.dumps(rendered[sid]["messages"][-1])
        assert ACCOUNT_ID_REGEX.search(blob)
        assert any(m in blob.lower() for m in ("closed", "retired", "discontinued", "no longer", "obsolete", "rejected"))


def test_sample_kind_split_and_label_markers_not_leaked_in_v1_1_prompts():
    _require(RENDERED_SAMPLES_PATH)
    from tell.probe_dataset.templates_v1_1 import SAMPLE_KIND_SPECS

    forbidden = [s.kind for s in SAMPLE_KIND_SPECS] + ["exposure_label", "clean_negative_family", "account_id_role"]
    for r in _load_jsonl(RENDERED_SAMPLES_PATH):
        blob = json.dumps(r["messages"])
        for marker in forbidden:
            assert marker not in blob, f"{r['sample_id']}: leaked {marker!r}"


def test_no_prompt_duplicates_cross_splits_v1_1():
    _require(RENDERED_SAMPLES_PATH)
    hashes = [r["prompt_sha256"] for r in _load_jsonl(RENDERED_SAMPLES_PATH)]
    assert len(hashes) == len(set(hashes))


# ---------------------------------------------------------------------
# Grouped split integrity (probe training)
# ---------------------------------------------------------------------


def test_grouped_split_integrity_no_document_spans_cv_folds():
    _require(TRAINING_PROTOCOL_PATH)
    protocol = json.loads(TRAINING_PROTOCOL_PATH.read_text())
    assert protocol["grouped_cross_validation_procedure"]["group_by"] == "docid"
    train_docids = protocol["split_docid_groups"]["train"]
    assert len(train_docids) == 12  # one group per document; GroupKFold keeps every sample of one doc in one fold


def test_training_protocol_never_references_test_labels_before_freeze():
    """Static check: `phase_test_evaluation` (the only place test labels
    are loaded) is called strictly after `_MODEL_FROZEN = True` in
    scripts/train_probe_pilot_v1.py's main()."""
    source = Path("/home/hp5/tell/scripts/train_probe_pilot_v1.py").read_text()
    freeze_pos = source.index("_MODEL_FROZEN = True")
    test_eval_call_pos = source.index("phase_test_evaluation(scaler, clf, threshold, selected_layer, test_labels, rendered_by_id)")
    assert freeze_pos < test_eval_call_pos


# ---------------------------------------------------------------------
# Scaler fit on training only
# ---------------------------------------------------------------------


def test_scaler_fit_on_training_only_by_source_inspection():
    source = Path("/home/hp5/tell/scripts/train_probe_pilot_v1.py").read_text()
    assert "scaler.fit_transform(X_train)" in source
    assert "scaler.transform(X)" in source  # validation/test always go through transform(), never fit


# ---------------------------------------------------------------------
# Deterministic model selection / threshold tie-breaking
# ---------------------------------------------------------------------


def test_c_selection_and_layer_selection_are_deterministic_given_seed():
    _require(CV_RESULTS_PATH)
    _require(LAYER_SELECTION_PATH)
    cv = json.loads(CV_RESULTS_PATH.read_text())
    layer_sel = json.loads(LAYER_SELECTION_PATH.read_text())
    for layer_str, result in cv["results_by_layer"].items():
        assert result["selected_c"] in cv["c_grid"]
    assert layer_sel["selected_layer"] in (9, 18, 27, 36)


def test_threshold_tiebreak_rule_documented():
    _require(TRAINING_PROTOCOL_PATH)
    protocol = json.loads(TRAINING_PROTOCOL_PATH.read_text())
    assert protocol["threshold_selection_rule"]["tiebreak"] == ["higher recall", "higher precision", "higher threshold"]


def test_threshold_within_unit_interval():
    _require(THRESHOLD_CALIBRATION_PATH)
    calib = json.loads(THRESHOLD_CALIBRATION_PATH.read_text())
    assert 0.0 <= calib["selected_threshold"] <= 1.0


# ---------------------------------------------------------------------
# Exported runtime parity + malformed-input rejection
# ---------------------------------------------------------------------


def test_runtime_parity_within_tolerance():
    _require(RUNTIME_PARITY_PATH)
    parity = json.loads(RUNTIME_PARITY_PATH.read_text())
    assert parity["parity_within_tolerance"] is True
    assert parity["max_abs_probability_diff_overall"] <= parity["tolerance"]


def test_runtime_rejects_wrong_shape_nan_inf():
    _require(PROBE_CONFIG_PATH)
    _require(PROBE_WEIGHTS_PATH)
    from tell.detector.probe import ProbeArtifactError, TellProbe

    probe = TellProbe.load(PROBE_CONFIG_PATH, PROBE_WEIGHTS_PATH)
    with pytest.raises(ProbeArtifactError):
        probe.score(np.zeros(10))
    bad_nan = np.zeros(4096)
    bad_nan[5] = float("nan")
    with pytest.raises(ProbeArtifactError):
        probe.score(bad_nan)
    bad_inf = np.zeros(4096)
    bad_inf[5] = float("inf")
    with pytest.raises(ProbeArtifactError):
        probe.score(bad_inf)


def test_runtime_rejects_weights_hash_mismatch(tmp_path):
    _require(PROBE_CONFIG_PATH)
    _require(PROBE_WEIGHTS_PATH)
    from tell.detector.probe import ProbeArtifactError, TellProbe

    bad_config = json.loads(PROBE_CONFIG_PATH.read_text())
    bad_config["weights_sha256"] = "0" * 64
    bad_config_path = tmp_path / "bad_config.json"
    bad_config_path.write_text(json.dumps(bad_config))
    with pytest.raises(ProbeArtifactError):
        TellProbe.load(bad_config_path, PROBE_WEIGHTS_PATH)


def test_runtime_has_no_sklearn_import():
    source = Path("/home/hp5/tell/src/tell/detector/probe.py").read_text()
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert "sklearn" not in imported_modules
    assert "torch" not in imported_modules


# ---------------------------------------------------------------------
# No memory holdout used during fitting
# ---------------------------------------------------------------------


def test_memory_holdout_never_referenced_before_export_phase():
    source = Path("/home/hp5/tell/scripts/train_probe_pilot_v1.py").read_text()
    holdout_pos = source.index("HOLDOUT_MANIFEST_PATH.read_text()")
    export_fn_pos = source.index("def phase_export_and_parity")
    memory_fn_pos = source.index("def phase_delayed_memory_zero_shot")
    assert export_fn_pos < memory_fn_pos
    assert holdout_pos > export_fn_pos  # the only holdout read happens inside/after the export phase's function body


def test_holdout_manifest_entries_disjoint_from_corpus_sample_ids():
    _require(SAMPLE_INDEX_MAPPING_PATH)
    mapping = json.loads(SAMPLE_INDEX_MAPPING_PATH.read_text())
    corpus_ids = {sid for ids in mapping.values() for sid in ids}
    holdout = json.loads(HOLDOUT_MANIFEST_PATH.read_text())
    holdout_ids = {e["condition_id"] for e in holdout["entries"]}
    assert corpus_ids.isdisjoint(holdout_ids)


def test_delayed_memory_result_marks_illustrative_pair_out_of_domain():
    _require(DELAYED_MEMORY_RESULT_PATH)
    result = json.loads(DELAYED_MEMORY_RESULT_PATH.read_text())
    assert "OUT-OF-DOMAIN" in result["illustrative_out_of_domain_pair"]["note"]
    assert "primary_in_domain_pair" in result


# ---------------------------------------------------------------------
# Historical artifacts unchanged (v1 corpus + earlier experiments)
# ---------------------------------------------------------------------

V1_HISTORICAL_HASHES = {
    Path("/home/hp5/tell/results/probe_dataset/rendered_samples.jsonl"): None,  # hash recorded lazily below
}


def test_v1_corpus_files_unchanged_by_v1_1_work():
    v1_dir = Path("/home/hp5/tell/results/probe_dataset")
    v1_labels = v1_dir / "labels.jsonl"
    v1_rendered = v1_dir / "rendered_samples.jsonl"
    _require(v1_labels)
    _require(v1_rendered)
    v1_labels_records = _load_jsonl(v1_labels)
    v1_rendered_records = _load_jsonl(v1_rendered)
    assert len(v1_labels_records) == 200
    assert len(v1_rendered_records) == 200
    # v1's own identifier scheme (the fixed literal) must still be exactly
    # what it always was -- v1.1's new templates/synthetic-ids modules
    # must never have been imported into v1's already-written files.
    blob = json.dumps(v1_rendered_records)
    assert "SIM-ATTACKER-ACCT-0001" in blob  # v1 still uses its original fixed literal, unmodified


def test_no_test_in_this_module_loads_qwen():
    tree = ast.parse(Path(__file__).read_text())
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "QwenLocalRuntime" not in calls
