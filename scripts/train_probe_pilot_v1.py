"""Trains, selects, calibrates, evaluates, diagnoses, and exports the
first Tell linear activation probe (Part B, Sections 8-15).

Runs strictly in this order, matching results/probe_training/pilot_v1/
training_protocol.json (frozen by scripts/freeze_probe_training_protocol.py,
which must already have run):

  1. Load corpus v1.1 train/validation activations + labels only.
     (Test-set activations/labels are loaded from disk only inside
     `phase_test_evaluation`, i.e. strictly after the model is frozen --
     see the `_MODEL_FROZEN` guard.)
  2. Per layer: grouped 4-fold CV (group=docid) over the fixed C grid,
     on the 120 training samples only -> select C.
  3. Fit each layer's selected-C pipeline on all 120 training samples,
     score the 40 validation samples -> select layer.
  4. Calibrate the alarm threshold on the selected layer's validation
     predictions only.
  5. Freeze the model (scaler + coefficients + intercept + layer +
     threshold). No test data has been touched up to this point.
  6. Load test labels/activations for the first and only time; evaluate
     once; no revision after.
  7. Diagnostics (account-id / surface / hard-negative / document
     bootstrap / label permutation) -- do not change the model.
  8. Export the SafeTensors + JSON runtime artifact; verify runtime
     parity against sklearn on train/validation/test.
  9. Zero-shot delayed-memory evaluation (reads the untouched holdout
     pointer manifest only after everything above is frozen).

No LoRA is trained. No payment/gate/ledger/SQLite operation ever occurs.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CORPUS_DIR = Path("/home/hp5/tell/results/probe_dataset/v1_1")
ACTIVATIONS_DIR = CORPUS_DIR / "activations"
LABELS_PATH = CORPUS_DIR / "labels.jsonl"
RENDERED_SAMPLES_PATH = CORPUS_DIR / "rendered_samples.jsonl"
SAMPLE_INDEX_MAPPING_PATH = ACTIVATIONS_DIR / "sample_index_mapping.json"

TRAINING_DIR = Path("/home/hp5/tell/results/probe_training/pilot_v1")
PROTOCOL_PATH = TRAINING_DIR / "training_protocol.json"
CV_RESULTS_PATH = TRAINING_DIR / "cv_results.json"
LAYER_SELECTION_PATH = TRAINING_DIR / "layer_selection_results.json"
THRESHOLD_CALIBRATION_PATH = TRAINING_DIR / "threshold_calibration_results.json"
TEST_RESULTS_PATH = TRAINING_DIR / "test_results.json"
SUBGROUP_DIAGNOSTICS_PATH = TRAINING_DIR / "subgroup_diagnostics.json"
PERMUTATION_DIAGNOSTIC_PATH = TRAINING_DIR / "permutation_diagnostic.json"
DELAYED_MEMORY_RESULT_PATH = TRAINING_DIR / "delayed_memory_zero_shot_result.json"
RUNTIME_PARITY_PATH = TRAINING_DIR / "runtime_parity_report.json"

CONFIGS_DIR = Path("/home/hp5/tell/configs/probe")
PROBE_CONFIG_PATH = CONFIGS_DIR / "probe_v1.json"
PROBE_WEIGHTS_PATH = CONFIGS_DIR / "probe_v1.safetensors"

HOLDOUT_MANIFEST_PATH = Path("/home/hp5/tell/results/probe_dataset/delayed_memory_holdout_manifest.json")

HIDDEN_SIZE = 4096
RANDOM_SEED = 20240923
_MODEL_FROZEN = False  # flips to True only inside main(), after threshold calibration; guards test-data access


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_layer_matrix(split: str, layer: int) -> np.ndarray:
    with safe_open(str(ACTIVATIONS_DIR / f"{split}_layer{layer}.safetensors"), framework="np") as f:
        return f.get_tensor("activations").astype(np.float64)


def _fit_pipeline(X_train: np.ndarray, y_train: np.ndarray, C: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED)
    clf.fit(X_scaled, y_train)
    return scaler, clf


def _score(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def _metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, preds, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------
# Phase 2: grouped CV, per layer, over the C grid
# ---------------------------------------------------------------------


def phase_cv_c_selection(train_labels: list[dict], protocol: dict) -> dict:
    layers = protocol["candidate_layers"]
    c_grid = protocol["hyperparameter_grid"]["C"]
    n_folds = protocol["grouped_cross_validation_procedure"]["n_folds"]

    train_sample_ids = [l["sample_id"] for l in train_labels]
    y_train_full = np.array([l["exposure_label"] for l in train_labels])
    groups = np.array([l["docid"] for l in train_labels])

    results_by_layer = {}
    selected_c_by_layer = {}
    for layer in layers:
        X_full = _load_layer_matrix("train", layer)
        assert X_full.shape == (len(train_sample_ids), HIDDEN_SIZE)

        gkf = GroupKFold(n_splits=n_folds)
        fold_assignments = list(gkf.split(X_full, y_train_full, groups=groups))

        per_c_fold_ap = {c: [] for c in c_grid}
        per_c_fold_auroc = {c: [] for c in c_grid}
        for fold_idx, (fit_idx, held_idx) in enumerate(fold_assignments):
            for c in c_grid:
                scaler, clf = _fit_pipeline(X_full[fit_idx], y_train_full[fit_idx], c)
                scores = _score(scaler, clf, X_full[held_idx])
                y_held = y_train_full[held_idx]
                ap = average_precision_score(y_held, scores) if len(set(y_held.tolist())) > 1 else float("nan")
                auroc = roc_auc_score(y_held, scores) if len(set(y_held.tolist())) > 1 else float("nan")
                per_c_fold_ap[c].append(ap)
                per_c_fold_auroc[c].append(auroc)

        mean_ap = {c: float(np.nanmean(per_c_fold_ap[c])) for c in c_grid}
        mean_auroc = {c: float(np.nanmean(per_c_fold_auroc[c])) for c in c_grid}
        # Select C: max mean AP, tie-break higher mean AUROC, then smaller C (stronger regularization).
        best_c = max(c_grid, key=lambda c: (round(mean_ap[c], 10), round(mean_auroc[c], 10), -c))
        selected_c_by_layer[layer] = best_c
        results_by_layer[str(layer)] = {
            "fold_level": {
                str(c): {"ap_per_fold": per_c_fold_ap[c], "auroc_per_fold": per_c_fold_auroc[c]} for c in c_grid
            },
            "mean_ap_by_c": {str(c): mean_ap[c] for c in c_grid},
            "mean_auroc_by_c": {str(c): mean_auroc[c] for c in c_grid},
            "selected_c": best_c,
        }
        print(f"[cv] layer={layer}: selected C={best_c} (mean AP={mean_ap[best_c]:.4f}, mean AUROC={mean_auroc[best_c]:.4f})")

    return {"n_folds": n_folds, "c_grid": c_grid, "results_by_layer": results_by_layer, "selected_c_by_layer": {str(k): v for k, v in selected_c_by_layer.items()}}


# ---------------------------------------------------------------------
# Phase 3: layer selection on validation data
# ---------------------------------------------------------------------


def phase_layer_selection(train_labels: list[dict], val_labels: list[dict], selected_c_by_layer: dict, protocol: dict) -> dict:
    layers = protocol["candidate_layers"]
    y_train = np.array([l["exposure_label"] for l in train_labels])
    y_val = np.array([l["exposure_label"] for l in val_labels])

    per_layer = {}
    fitted_pipelines = {}
    for layer in layers:
        c = selected_c_by_layer[layer]
        X_train = _load_layer_matrix("train", layer)
        X_val = _load_layer_matrix("validation", layer)
        scaler, clf = _fit_pipeline(X_train, y_train, c)
        fitted_pipelines[layer] = (scaler, clf)
        val_scores = _score(scaler, clf, X_val)
        ap = average_precision_score(y_val, val_scores)
        auroc = roc_auc_score(y_val, val_scores)
        per_layer[str(layer)] = {"C": c, "validation_ap": ap, "validation_auroc": auroc}
        print(f"[layer-select] layer={layer} C={c}: val AP={ap:.4f} AUROC={auroc:.4f}")

    selected_layer = max(layers, key=lambda l: (round(per_layer[str(l)]["validation_ap"], 10), round(per_layer[str(l)]["validation_auroc"], 10), -selected_c_by_layer[l]))
    result = {"per_layer": per_layer, "selected_layer": selected_layer, "selected_c": selected_c_by_layer[selected_layer]}
    print(f"[layer-select] SELECTED layer={selected_layer} C={selected_c_by_layer[selected_layer]}")
    return result, fitted_pipelines[selected_layer]


# ---------------------------------------------------------------------
# Phase 4: threshold calibration (validation only, selected layer)
# ---------------------------------------------------------------------


def phase_threshold_calibration(y_val: np.ndarray, val_scores: np.ndarray) -> dict:
    candidate_thresholds = sorted(set(np.round(val_scores, 6).tolist()) | {0.0, 1.0})
    best = None
    for t in candidate_thresholds:
        preds = (val_scores >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        precision = precision_score(y_val, preds, zero_division=0)
        key = (round(f1, 10), round(recall, 10), round(precision, 10), round(t, 10))
        if best is None or key > best[0]:
            best = (key, t, f1, recall, precision)
    _, best_threshold, best_f1, best_recall, best_precision = best

    # Alternative operating point 1: highest recall with precision >= 0.80
    alt_high_recall = None
    for t in candidate_thresholds:
        preds = (val_scores >= t).astype(int)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        if precision >= 0.80:
            if alt_high_recall is None or recall > alt_high_recall["recall"]:
                alt_high_recall = {"threshold": t, "recall": recall, "precision": precision}

    # Alternative operating point 2: lowest FPR with recall >= 0.80
    alt_low_fpr = None
    for t in candidate_thresholds:
        preds = (val_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        recall = recall_score(y_val, preds, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        if recall >= 0.80:
            if alt_low_fpr is None or fpr < alt_low_fpr["fpr"]:
                alt_low_fpr = {"threshold": t, "fpr": fpr, "recall": recall}

    return {
        "selected_threshold": best_threshold,
        "selected_threshold_f1": best_f1,
        "selected_threshold_recall": best_recall,
        "selected_threshold_precision": best_precision,
        "alternative_highest_recall_at_precision_gte_0_80": alt_high_recall,
        "alternative_lowest_fpr_at_recall_gte_0_80": alt_low_fpr,
    }


# ---------------------------------------------------------------------
# Phase 6: test evaluation (exactly once, after freeze)
# ---------------------------------------------------------------------


def phase_test_evaluation(scaler: StandardScaler, clf: LogisticRegression, threshold: float, selected_layer: int, test_labels: list[dict], rendered_by_id: dict) -> dict:
    assert _MODEL_FROZEN, "phase_test_evaluation called before the model was frozen"
    X_test = _load_layer_matrix("test", selected_layer)
    y_test = np.array([l["exposure_label"] for l in test_labels])
    test_ids = [l["sample_id"] for l in test_labels]
    scores = _score(scaler, clf, X_test)

    overall = _metrics_at_threshold(y_test, scores, threshold)
    overall["auroc"] = roc_auc_score(y_test, scores)
    overall["auprc"] = average_precision_score(y_test, scores)
    overall["score_distribution_by_class"] = {
        "clean": {"mean": float(scores[y_test == 0].mean()), "std": float(scores[y_test == 0].std()), "min": float(scores[y_test == 0].min()), "max": float(scores[y_test == 0].max())},
        "attack": {"mean": float(scores[y_test == 1].mean()), "std": float(scores[y_test == 1].std()), "min": float(scores[y_test == 1].min()), "max": float(scores[y_test == 1].max())},
    }

    def _subgroup_metrics(mask: np.ndarray) -> dict | None:
        if mask.sum() == 0:
            return None
        y_sub, s_sub = y_test[mask], scores[mask]
        d = _metrics_at_threshold(y_sub, s_sub, threshold)
        d["n"] = int(mask.sum())
        if len(set(y_sub.tolist())) > 1:
            d["auroc"] = roc_auc_score(y_sub, s_sub)
            d["auprc"] = average_precision_score(y_sub, s_sub)
        return d

    by_surface = {}
    for surface in ("post_email", "post_invoice"):
        mask = np.array([test_labels[i]["decision_point"] == surface for i in range(len(test_labels))])
        by_surface[surface] = _subgroup_metrics(mask)

    by_attack_family = {}
    attack_families = sorted({l["template_family_id"].rsplit("__", 1)[0] for l in test_labels if l["exposure_label"] == 1})
    for fam in attack_families:
        mask = np.array([l["template_family_id"].startswith(fam + "__") and l["exposure_label"] == 1 for l in test_labels])
        by_attack_family[fam] = _subgroup_metrics(mask)

    by_clean_family = {}
    clean_families = sorted({l["template_family_id"].rsplit("__", 1)[0] for l in test_labels if l["exposure_label"] == 0})
    for fam in clean_families:
        mask = np.array([l["template_family_id"].startswith(fam + "__") and l["exposure_label"] == 0 for l in test_labels])
        by_clean_family[fam] = _subgroup_metrics(mask)

    by_document = {}
    for docid in sorted({l["docid"] for l in test_labels}):
        mask = np.array([l["docid"] == docid for l in test_labels])
        by_document[docid] = _subgroup_metrics(mask)

    token_counts = np.array([rendered_by_id[sid]["rendered_token_count"] for sid in test_ids])
    quartile_edges = np.quantile(token_counts, [0.25, 0.5, 0.75])
    quartile_idx = np.digitize(token_counts, quartile_edges)  # 0..3
    by_token_quartile = {}
    for q in range(4):
        mask = quartile_idx == q
        by_token_quartile[f"Q{q+1}"] = _subgroup_metrics(mask)

    by_account_presence = {}
    for role in ("approved", "unauthorized", "obsolete"):
        mask = np.array([l["account_id_role"] == role for l in test_labels])
        by_account_presence[role] = _subgroup_metrics(mask)
    alternate_mask = np.array([l["account_id_role"] in ("unauthorized", "obsolete") for l in test_labels])
    by_account_presence["any_alternate_format_id"] = _subgroup_metrics(alternate_mask)
    by_account_presence["approved_only"] = _subgroup_metrics(~alternate_mask)

    return {
        "n_test": len(test_labels),
        "overall": overall,
        "by_attack_surface": by_surface,
        "by_attack_family": by_attack_family,
        "by_clean_hard_negative_family": by_clean_family,
        "by_document": by_document,
        "by_token_length_quartile": {"edges": quartile_edges.tolist(), **by_token_quartile},
        "by_account_id_presence": by_account_presence,
    }


# ---------------------------------------------------------------------
# Phase 7: diagnostics
# ---------------------------------------------------------------------


def phase_document_bootstrap(scaler: StandardScaler, clf: LogisticRegression, selected_layer: int, test_labels: list[dict], n_bootstrap: int = 2000) -> dict:
    X_test = _load_layer_matrix("test", selected_layer)
    y_test = np.array([l["exposure_label"] for l in test_labels])
    scores = _score(scaler, clf, X_test)
    docids = np.array([l["docid"] for l in test_labels])
    unique_docs = sorted(set(docids.tolist()))
    rng = random.Random(RANDOM_SEED)

    aurocs, auprcs = [], []
    for _ in range(n_bootstrap):
        sampled_docs = [rng.choice(unique_docs) for _ in unique_docs]
        idx = np.concatenate([np.where(docids == d)[0] for d in sampled_docs])
        y_b, s_b = y_test[idx], scores[idx]
        if len(set(y_b.tolist())) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, s_b))
        auprcs.append(average_precision_score(y_b, s_b))

    def _pct(vals, p):
        return float(np.percentile(vals, p)) if vals else None

    return {
        "n_test_documents": len(unique_docs),
        "n_bootstrap_resamples": n_bootstrap,
        "n_resamples_with_both_classes": len(aurocs),
        "note": "EXPLORATORY ONLY: only 4 test documents exist, so this bootstrap resamples documents (not samples) and its interval should not be treated as a well-powered confidence interval.",
        "auroc": {"median": _pct(aurocs, 50), "p2_5": _pct(aurocs, 2.5), "p97_5": _pct(aurocs, 97.5)},
        "auprc": {"median": _pct(auprcs, 50), "p2_5": _pct(auprcs, 2.5), "p97_5": _pct(auprcs, 97.5)},
    }


def phase_permutation_diagnostic(train_labels: list[dict], val_labels: list[dict], selected_layer: int, observed_val_auprc: float, protocol: dict, n_permutations: int) -> dict:
    c_grid = protocol["hyperparameter_grid"]["C"]
    n_folds = protocol["grouped_cross_validation_procedure"]["n_folds"]
    X_train = _load_layer_matrix("train", selected_layer)
    X_val = _load_layer_matrix("validation", selected_layer)
    y_train_true = np.array([l["exposure_label"] for l in train_labels])
    y_val_true = np.array([l["exposure_label"] for l in val_labels])
    groups = np.array([l["docid"] for l in train_labels])

    null_auprcs = []
    for perm_idx in range(n_permutations):
        perm_rng = np.random.RandomState(RANDOM_SEED + 1 + perm_idx)
        y_train_perm = y_train_true.copy()
        perm_rng.shuffle(y_train_perm)

        gkf = GroupKFold(n_splits=n_folds)
        per_c_ap = {c: [] for c in c_grid}
        for fit_idx, held_idx in gkf.split(X_train, y_train_perm, groups=groups):
            for c in c_grid:
                y_fit = y_train_perm[fit_idx]
                if len(set(y_fit.tolist())) < 2:
                    per_c_ap[c].append(float("nan"))
                    continue
                scaler, clf = _fit_pipeline(X_train[fit_idx], y_fit, c)
                s = _score(scaler, clf, X_train[held_idx])
                y_held = y_train_perm[held_idx]
                per_c_ap[c].append(average_precision_score(y_held, s) if len(set(y_held.tolist())) > 1 else float("nan"))
        mean_ap = {c: float(np.nanmean(per_c_ap[c])) for c in c_grid}
        best_c = max(c_grid, key=lambda c: (round(mean_ap[c], 10), -c))

        scaler, clf = _fit_pipeline(X_train, y_train_perm, best_c)
        val_scores = _score(scaler, clf, X_val)
        # Validation labels are the TRUE labels (never permuted) -- this is
        # what makes the resulting AP a meaningful null-distribution sample.
        auprc = average_precision_score(y_val_true, val_scores)
        null_auprcs.append(auprc)

    null_arr = np.array(null_auprcs)
    p_value = float((np.sum(null_arr >= observed_val_auprc) + 1) / (len(null_arr) + 1))
    return {
        "n_permutations": n_permutations,
        "selected_layer": selected_layer,
        "observed_validation_auprc": observed_val_auprc,
        "null_distribution_mean": float(null_arr.mean()),
        "null_distribution_std": float(null_arr.std()),
        "null_distribution_max": float(null_arr.max()),
        "null_distribution_min": float(null_arr.min()),
        "empirical_p_value_observed_gte_null": p_value,
        "note": "Diagnostic only -- never used to tune the model. Labels are permuted on train+validation only; test and the memory holdout are never touched here.",
    }


def phase_account_subgroup_and_surface_and_hardneg(scaler, clf, threshold, selected_layer: int, test_labels: list[dict]) -> dict:
    """Section 12.1-12.3: explicit shortcut diagnostics, computed from the
    already-frozen test_results by-subgroup breakdowns (re-derived here
    directly from scores for a self-contained diagnostics artifact)."""
    X_test = _load_layer_matrix("test", selected_layer)
    y_test = np.array([l["exposure_label"] for l in test_labels])
    scores = _score(scaler, clf, X_test)

    alt_mask = np.array([l["account_id_role"] in ("unauthorized", "obsolete") for l in test_labels])
    no_alt_mask = ~alt_mask

    def _summ(mask):
        if mask.sum() == 0:
            return None
        y_sub, s_sub = y_test[mask], scores[mask]
        out = {"n": int(mask.sum()), "mean_score": float(s_sub.mean())}
        if len(set(y_sub.tolist())) > 1:
            out["auroc"] = roc_auc_score(y_sub, s_sub)
            out["auprc"] = average_precision_score(y_sub, s_sub)
        return out

    return {
        "account_id_subgroup": {"with_alternate_format_id": _summ(alt_mask), "without_alternate_format_id": _summ(no_alt_mask)},
        "surface": {
            "email": _summ(np.array([l["decision_point"] == "post_email" for l in test_labels])),
            "invoice_tool_result": _summ(np.array([l["decision_point"] == "post_invoice" for l in test_labels])),
        },
        "hard_negative_family": {
            fam: _summ(np.array([l["template_family_id"].startswith(fam + "__") and l["exposure_label"] == 0 for l in test_labels]))
            for fam in ("clean_json_like", "clean_urgent", "clean_instructional_language", "clean_account_reference")
        },
    }


# ---------------------------------------------------------------------
# Phase 8: export + runtime parity
# ---------------------------------------------------------------------


def phase_export_and_parity(scaler: StandardScaler, clf: LogisticRegression, selected_layer: int, threshold: float, threshold_calib: dict, protocol: dict, all_split_labels: dict) -> dict:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)
    coef = clf.coef_.reshape(-1).astype(np.float32)
    intercept = np.array([clf.intercept_[0]], dtype=np.float32)

    save_file(
        {"scaler_mean": scaler_mean, "scaler_scale": scaler_scale, "coef": coef, "intercept": intercept},
        str(PROBE_WEIGHTS_PATH),
        metadata={"selected_layer": str(selected_layer), "feature_dimension": str(HIDDEN_SIZE)},
    )
    weights_sha256 = _sha256_file(PROBE_WEIGHTS_PATH)

    config = {
        "artifact": "tell_probe_v1",
        "selected_layer": selected_layer,
        "threshold": threshold,
        "threshold_calibration": threshold_calib,
        "class_order": [0, 1],
        "class_meaning": {"0": "clean", "1": "attack_exposed"},
        "feature_dimension": HIDDEN_SIZE,
        "weights_sha256": weights_sha256,
        "corpus_protocol_manifest_sha256": protocol["corpus_protocol_manifest_sha256"],
        "training_protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "sklearn_pipeline": "StandardScaler -> LogisticRegression(penalty='l2', solver='lbfgs')",
        "selected_C": clf.C,
    }
    PROBE_CONFIG_PATH.write_text(json.dumps(config, indent=2))

    # ---- Runtime parity check on train/validation/test ----
    import sys

    sys.path.insert(0, "/home/hp5/tell/src")
    from tell.detector.probe import TellProbe

    probe = TellProbe.load(PROBE_CONFIG_PATH, PROBE_WEIGHTS_PATH)

    max_abs_diff = 0.0
    per_split_max_diff = {}
    for split in ("train", "validation", "test"):
        X = _load_layer_matrix(split, selected_layer)
        sk_probs = _score(scaler, clf, X)
        runtime_probs = np.array([probe.score(X[i]).probability for i in range(X.shape[0])])
        diff = float(np.max(np.abs(sk_probs - runtime_probs)))
        per_split_max_diff[split] = diff
        max_abs_diff = max(max_abs_diff, diff)

    tolerance = 1e-4
    parity_report = {
        "tolerance": tolerance,
        "max_abs_probability_diff_by_split": per_split_max_diff,
        "max_abs_probability_diff_overall": max_abs_diff,
        "parity_within_tolerance": max_abs_diff <= tolerance,
    }
    if not parity_report["parity_within_tolerance"]:
        raise RuntimeError(f"Runtime parity check FAILED: max abs diff {max_abs_diff} exceeds tolerance {tolerance}")
    print(f"[export] runtime parity max abs diff = {max_abs_diff:.2e} (tolerance {tolerance})")

    # ---- Malformed-input rejection checks (not part of parity, but proves the runtime's own guards) ----
    from tell.detector.probe import ProbeArtifactError

    rejection_checks = {}
    try:
        probe.score(np.zeros(10))
        rejection_checks["wrong_shape_rejected"] = False
    except ProbeArtifactError:
        rejection_checks["wrong_shape_rejected"] = True

    bad = np.zeros(HIDDEN_SIZE)
    bad[0] = float("nan")
    try:
        probe.score(bad)
        rejection_checks["nan_rejected"] = False
    except ProbeArtifactError:
        rejection_checks["nan_rejected"] = True

    bad2 = np.zeros(HIDDEN_SIZE)
    bad2[0] = float("inf")
    try:
        probe.score(bad2)
        rejection_checks["inf_rejected"] = False
    except ProbeArtifactError:
        rejection_checks["inf_rejected"] = True

    parity_report["rejection_checks"] = rejection_checks
    return parity_report


# ---------------------------------------------------------------------
# Phase 9: zero-shot delayed-memory evaluation
# ---------------------------------------------------------------------


def phase_delayed_memory_zero_shot(selected_layer: int) -> dict:
    import sys

    sys.path.insert(0, "/home/hp5/tell/src")
    from tell.detector.probe import TellProbe

    manifest = json.loads(HOLDOUT_MANIFEST_PATH.read_text())
    probe = TellProbe.load(PROBE_CONFIG_PATH, PROBE_WEIGHTS_PATH)

    entries_by_condition = {e["condition_id"]: e for e in manifest["entries"]}

    def _score_entry(condition_id: str) -> dict:
        entry = entries_by_condition[condition_id]
        with safe_open(entry["safetensors_path"], framework="np") as f:
            vec = f.get_tensor(f"hidden_state_{selected_layer}").astype(np.float64)
        result = probe.score(vec)
        return {"probability": result.probability, "alarm": result.alarm, "threshold": result.threshold}

    primary = {
        "clean": _score_entry("task_only_base_v1__clean_memory"),
        "poisoned": _score_entry("task_only_base_v1__poisoned_memory"),
    }
    primary["score_difference_poisoned_minus_clean"] = primary["poisoned"]["probability"] - primary["clean"]["probability"]
    primary["zero_shot_transfer_succeeded"] = bool(primary["poisoned"]["alarm"] and not primary["clean"]["alarm"])

    illustrative = {
        "clean": _score_entry("hardened_provenance_prompt_v1__clean_memory"),
        "poisoned": _score_entry("hardened_provenance_prompt_v1__poisoned_memory"),
    }
    illustrative["score_difference_poisoned_minus_clean"] = illustrative["poisoned"]["probability"] - illustrative["clean"]["probability"]
    illustrative["note"] = (
        "OUT-OF-DOMAIN and ILLUSTRATIVE ONLY: hardened_provenance_prompt_v1 uses different prompt wording than "
        "this probe's training corpus (task_only_base_v1). Not used for, and does not affect, the primary "
        "zero-shot-transfer result above. One pair only -- not statistically powered."
    )

    return {
        "holdout_tag": manifest["holdout_tag"],
        "usage_restrictions_honored": manifest["usage_restrictions"],
        "selected_layer": selected_layer,
        "primary_in_domain_pair": primary,
        "illustrative_out_of_domain_pair": illustrative,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    global _MODEL_FROZEN
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"{PROTOCOL_PATH} does not exist -- run scripts/freeze_probe_training_protocol.py first.")
    protocol = json.loads(PROTOCOL_PATH.read_text())

    all_labels = {l["sample_id"]: l for l in _load_jsonl(LABELS_PATH)}
    sample_index_mapping = json.loads(SAMPLE_INDEX_MAPPING_PATH.read_text())
    rendered_by_id = {r["sample_id"]: r for r in _load_jsonl(RENDERED_SAMPLES_PATH)}

    train_labels = [all_labels[sid] for sid in sample_index_mapping["train"]]
    val_labels = [all_labels[sid] for sid in sample_index_mapping["validation"]]
    # test_labels intentionally NOT loaded here -- only inside phase_test_evaluation, after freeze.

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: grouped CV, C selection (train documents only) ===")
    cv_results = phase_cv_c_selection(train_labels, protocol)
    CV_RESULTS_PATH.write_text(json.dumps(cv_results, indent=2))
    print(f"Wrote {CV_RESULTS_PATH}")

    print("=== Phase 3: layer selection (validation data) ===")
    selected_c_by_layer = {int(k): v for k, v in cv_results["selected_c_by_layer"].items()}
    layer_selection, (scaler, clf) = phase_layer_selection(train_labels, val_labels, selected_c_by_layer, protocol)
    LAYER_SELECTION_PATH.write_text(json.dumps(layer_selection, indent=2))
    print(f"Wrote {LAYER_SELECTION_PATH}")
    selected_layer = layer_selection["selected_layer"]
    selected_c = layer_selection["selected_c"]

    print("=== Phase 4: threshold calibration (validation, selected layer) ===")
    y_val = np.array([l["exposure_label"] for l in val_labels])
    X_val = _load_layer_matrix("validation", selected_layer)
    val_scores = _score(scaler, clf, X_val)
    threshold_calib = phase_threshold_calibration(y_val, val_scores)
    THRESHOLD_CALIBRATION_PATH.write_text(json.dumps(threshold_calib, indent=2))
    print(f"Wrote {THRESHOLD_CALIBRATION_PATH}: threshold={threshold_calib['selected_threshold']}")

    # ---- Phase 5: freeze ----
    threshold = threshold_calib["selected_threshold"]
    _MODEL_FROZEN = True
    print(f"=== MODEL FROZEN: layer={selected_layer} C={selected_c} threshold={threshold} ===")

    print("=== Phase 6: held-out test evaluation (exactly once) ===")
    test_labels = [all_labels[sid] for sid in sample_index_mapping["test"]]
    test_results = phase_test_evaluation(scaler, clf, threshold, selected_layer, test_labels, rendered_by_id)
    TEST_RESULTS_PATH.write_text(json.dumps(test_results, indent=2))
    print(f"Wrote {TEST_RESULTS_PATH}: AUROC={test_results['overall']['auroc']:.4f} AUPRC={test_results['overall']['auprc']:.4f}")

    print("=== Phase 7: diagnostics ===")
    subgroup = phase_account_subgroup_and_surface_and_hardneg(scaler, clf, threshold, selected_layer, test_labels)
    bootstrap = phase_document_bootstrap(scaler, clf, selected_layer, test_labels)
    subgroup["document_bootstrap"] = bootstrap
    SUBGROUP_DIAGNOSTICS_PATH.write_text(json.dumps(subgroup, indent=2))
    print(f"Wrote {SUBGROUP_DIAGNOSTICS_PATH}")

    observed_val_auprc = layer_selection["per_layer"][str(selected_layer)]["validation_ap"]
    permutation = phase_permutation_diagnostic(train_labels, val_labels, selected_layer, observed_val_auprc, protocol, protocol["permutation_diagnostic"]["n_permutations"])
    PERMUTATION_DIAGNOSTIC_PATH.write_text(json.dumps(permutation, indent=2))
    print(f"Wrote {PERMUTATION_DIAGNOSTIC_PATH}: p={permutation['empirical_p_value_observed_gte_null']:.4f}")

    print("=== Phase 8: export + runtime parity ===")
    parity = phase_export_and_parity(scaler, clf, selected_layer, threshold, threshold_calib, protocol, {"train": train_labels, "validation": val_labels, "test": test_labels})
    RUNTIME_PARITY_PATH.write_text(json.dumps(parity, indent=2))
    print(f"Wrote {RUNTIME_PARITY_PATH}")

    print("=== Phase 9: zero-shot delayed-memory evaluation ===")
    memory_result = phase_delayed_memory_zero_shot(selected_layer)
    DELAYED_MEMORY_RESULT_PATH.write_text(json.dumps(memory_result, indent=2))
    print(f"Wrote {DELAYED_MEMORY_RESULT_PATH}")

    # ---- Success-criteria verdict ----
    email_recall = test_results["by_attack_surface"]["post_email"]["recall"] if test_results["by_attack_surface"]["post_email"] else None
    invoice_recall = test_results["by_attack_surface"]["post_invoice"]["recall"] if test_results["by_attack_surface"]["post_invoice"] else None
    criteria = protocol["minimum_success_criteria"]["verdict_pilot_passed"]
    passed = (
        test_results["overall"]["auroc"] >= criteria["test_auroc_gte"]
        and test_results["overall"]["auprc"] >= criteria["test_auprc_gte"]
        and email_recall is not None and email_recall >= criteria["email_surface_recall_gte"]
        and invoice_recall is not None and invoice_recall >= criteria["invoice_tool_surface_recall_gte"]
        and parity["parity_within_tolerance"]
    )
    verdict = "PROBE PILOT PASSED" if passed else "PROBE TRAINED — PERFORMANCE INSUFFICIENT"
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"AUROC={test_results['overall']['auroc']:.4f} AUPRC={test_results['overall']['auprc']:.4f} email_recall={email_recall} invoice_recall={invoice_recall}")

    summary = {
        "verdict": verdict,
        "selected_layer": selected_layer,
        "selected_c": selected_c,
        "threshold": threshold,
        "test_auroc": test_results["overall"]["auroc"],
        "test_auprc": test_results["overall"]["auprc"],
        "email_surface_recall": email_recall,
        "invoice_surface_recall": invoice_recall,
        "runtime_parity_ok": parity["parity_within_tolerance"],
    }
    (TRAINING_DIR / "verdict_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {TRAINING_DIR / 'verdict_summary.json'}")


if __name__ == "__main__":
    main()
