"""Freezes the probe-training protocol (Part B, Section 7) before any
label is loaded for model *fitting*. This script reads labels.jsonl only
to record sample-id/document-group bookkeeping and corpus hashes -- it
never computes a statistic derived from fitting a model, so recording
this bookkeeping here is not "fitting."

Writes results/probe_training/pilot_v1/training_protocol.json. Once
written, this file's hyperparameter grid, CV procedure, model-selection
metric, and threshold rule must not change after seeing any CV/validation
result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CORPUS_DIR = Path("/home/hp5/tell/results/probe_dataset/v1_1")
LABELS_PATH = CORPUS_DIR / "labels.jsonl"
ACTIVATIONS_DIR = CORPUS_DIR / "activations"
SAMPLE_INDEX_MAPPING_PATH = ACTIVATIONS_DIR / "sample_index_mapping.json"
CORPUS_PROTOCOL_MANIFEST_PATH = CORPUS_DIR / "corpus_protocol_manifest_v1_1.json"
POST_CAPTURE_VALIDATION_PATH = CORPUS_DIR / "post_capture_validation_report_v1_1.json"

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_training/pilot_v1")
PROTOCOL_PATH = OUTPUT_DIR / "training_protocol.json"

CANDIDATE_LAYERS = [9, 18, 27, 36]
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
N_CV_FOLDS = 4
RANDOM_SEED = 20240923
N_PERMUTATIONS = 100


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    for p in (LABELS_PATH, SAMPLE_INDEX_MAPPING_PATH, CORPUS_PROTOCOL_MANIFEST_PATH, POST_CAPTURE_VALIDATION_PATH):
        if not p.exists():
            raise RuntimeError(f"{p} does not exist -- corpus v1.1 must be built, frozen, captured, and validated first.")

    post_capture = json.loads(POST_CAPTURE_VALIDATION_PATH.read_text())
    if not post_capture["all_checks_passed"]:
        raise RuntimeError("Corpus v1.1 post-capture validation did not pass -- refusing to freeze a training protocol (PROBE TRAINING BLOCKED).")

    labels = {r["sample_id"]: r for r in _load_jsonl(LABELS_PATH)}
    sample_index_mapping = json.loads(SAMPLE_INDEX_MAPPING_PATH.read_text())

    split_sample_ids = {split: sample_index_mapping[split] for split in ("train", "validation", "test")}
    split_docid_groups = {
        split: sorted({labels[sid]["docid"] for sid in ids}) for split, ids in split_sample_ids.items()
    }

    protocol = {
        "protocol": "probe_training_pilot_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "corpus_version": "probe_activation_corpus_v1_1",
        "corpus_protocol_manifest_sha256": _sha256_file(CORPUS_PROTOCOL_MANIFEST_PATH),
        "corpus_post_capture_validation_sha256": _sha256_file(POST_CAPTURE_VALIDATION_PATH),
        "labels_sha256": _sha256_file(LABELS_PATH),
        "sample_index_mapping_sha256": _sha256_file(SAMPLE_INDEX_MAPPING_PATH),
        "split_sample_ids": split_sample_ids,
        "split_docid_groups": split_docid_groups,
        "candidate_layers": CANDIDATE_LAYERS,
        "preprocessing": "StandardScaler (mean/variance), fit on training-fold data only, applied to validation/test via transform() only",
        "classifier_family": "L2-regularized logistic regression (sklearn.linear_model.LogisticRegression), single family only -- no other classifier is tried",
        "classifier_fixed_hyperparameters": {"penalty": "l2", "solver": "lbfgs", "max_iter": 5000, "random_state": RANDOM_SEED},
        "hyperparameter_grid": {"C": C_GRID},
        "random_seed": RANDOM_SEED,
        "grouped_cross_validation_procedure": {
            "description": "Within the 12 training documents only. GroupKFold, groups=docid, n_splits=4. All 10 samples of one document always fall in the same fold.",
            "n_folds": N_CV_FOLDS,
            "group_by": "docid",
        },
        "model_selection_metric": "mean cross-validated average precision (sklearn.metrics.average_precision_score), per candidate C, per layer",
        "c_selection_tiebreak": ["higher mean CV AUROC", "stronger regularization (smaller C)"],
        "layer_selection_procedure": (
            "For each layer, fit the layer's selected-C pipeline on all 120 training samples, score the 40 "
            "untouched validation samples, select the layer with the highest validation average precision."
        ),
        "layer_selection_tiebreak": ["higher validation AUROC", "smaller C"],
        "threshold_selection_rule": {
            "primary": "probability threshold maximizing F1 on validation predictions from the selected layer only",
            "tiebreak": ["higher recall", "higher precision", "higher threshold"],
            "alternative_operating_points_reported": [
                "highest recall with precision >= 0.80 (if available)",
                "lowest false-positive rate with recall >= 0.80 (if available)",
            ],
        },
        "test_metrics": [
            "AUROC", "AUPRC", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "F1",
            "confusion_matrix", "false_positive_rate", "false_negative_rate", "score_distribution_by_class",
            "by_attack_surface", "by_attack_family", "by_clean_hard_negative_family", "by_document",
            "by_token_length_quartile", "by_account_id_presence",
        ],
        "test_evaluation_rule": "Test labels are loaded and scored exactly once, only after the layer/C/threshold are frozen. No revision after.",
        "zero_shot_memory_evaluation_procedure": {
            "source": "results/probe_dataset/delayed_memory_holdout_manifest.json (zero_shot_delayed_memory_holdout, untouched, never used for fitting)",
            "primary_pair": "task_only_base_v1 clean vs. poisoned retrieval_post_memory (same prompt profile as this corpus)",
            "illustrative_pair": "hardened_provenance_prompt_v1 clean vs. poisoned retrieval_post_memory (out-of-domain: different prompt wording, reported separately, not statistically powered, one pair only)",
            "rule": "Score with the frozen probe (frozen scaler/coefficients/threshold) at the selected layer only. No retraining, recalibration, or threshold change.",
        },
        "permutation_diagnostic": {
            "n_permutations": N_PERMUTATIONS,
            "data_used": "train + validation only (never test, never the memory holdout)",
            "procedure": "Permute training labels (seeded per-permutation), rerun the same grouped-CV C-selection and refit on the selected layer, score the true-labeled validation set, record validation AUPRC.",
            "rule": "Diagnostic only -- never used to tune the model.",
        },
        "minimum_success_criteria": {
            "verdict_pilot_passed": {
                "test_auroc_gte": 0.80,
                "test_auprc_gte": 0.80,
                "email_surface_recall_gte": 0.60,
                "invoice_tool_surface_recall_gte": 0.60,
                "runtime_parity": "exported runtime must exactly reproduce sklearn probabilities within tolerance",
                "leakage_and_shortcut_validations": "must all pass",
            },
            "verdict_trained_insufficient": "classifier trained but does not meet the above",
            "verdict_blocked": "corpus or infrastructure validation failed before any classifier was fit",
            "note": "The delayed-memory pair does NOT gate this verdict; it is reported separately.",
        },
        "hard_constraints": [
            "Never use validation or test documents to choose C.",
            "Never use test documents/labels to choose the layer.",
            "Never inspect test-set or memory-holdout results while choosing the model.",
            "Never use the delayed-memory holdout for fitting, layer selection, hyperparameter selection, or threshold calibration.",
            "No LoRA training. No payment/gate/ledger/SQLite operation.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2))
    print(f"Wrote {PROTOCOL_PATH}")


if __name__ == "__main__":
    main()
