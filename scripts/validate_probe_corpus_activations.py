"""Post-capture validation and descriptive statistics for the probe
corpus (Section 14). Reloads every activation artifact from disk (never
reuses in-memory state from the capture script) and verifies structural
correctness, then computes label-aware descriptive statistics only --
no classifier is trained and no threshold is selected here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset")
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
SAMPLE_INDEX_MAPPING_PATH = ACTIVATIONS_DIR / "sample_index_mapping.json"
CAPTURE_METADATA_PATH = ACTIVATIONS_DIR / "capture_metadata.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
HOLDOUT_MANIFEST_PATH = OUTPUT_DIR / "delayed_memory_holdout_manifest.json"
PROTOCOL_MANIFEST_PATH = OUTPUT_DIR / "corpus_protocol_manifest.json"
POST_CAPTURE_REPORT_PATH = OUTPUT_DIR / "post_capture_validation_report.json"

HIDDEN_STATE_INDICES = (9, 18, 27, 36)
HIDDEN_SIZE = 4096
SPLITS = ("train", "validation", "test")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    protocol = json.loads(PROTOCOL_MANIFEST_PATH.read_text())
    assert tuple(protocol["capture_hidden_state_indices"]) == HIDDEN_STATE_INDICES

    sample_index_mapping = json.loads(SAMPLE_INDEX_MAPPING_PATH.read_text())
    capture_metadata = {r["sample_id"]: r for r in _load_jsonl(CAPTURE_METADATA_PATH)}
    labels = {r["sample_id"]: r for r in _load_jsonl(LABELS_PATH)}
    rendered = {r["sample_id"]: r for r in _load_jsonl(RENDERED_SAMPLES_PATH)}
    holdout = json.loads(HOLDOUT_MANIFEST_PATH.read_text())

    errors: list[str] = []
    checks: dict[str, bool] = {}

    all_sample_ids = [sid for split in SPLITS for sid in sample_index_mapping[split]]
    checks["expected_sample_count_200"] = len(all_sample_ids) == 200
    checks["no_duplicate_sample_ids"] = len(all_sample_ids) == len(set(all_sample_ids))
    checks["split_counts_120_40_40"] = (
        len(sample_index_mapping["train"]) == 120 and len(sample_index_mapping["validation"]) == 40 and len(sample_index_mapping["test"]) == 40
    )

    # ---- Reload every shard, verify shape/dtype/finiteness ----
    shards: dict[str, dict[int, torch.Tensor]] = {}
    layer_count_ok = True
    hidden_size_ok = True
    dtype_ok = True
    finite_ok = True
    for split in SPLITS:
        shards[split] = {}
        n_expected = len(sample_index_mapping[split])
        for idx in HIDDEN_STATE_INDICES:
            path = ACTIVATIONS_DIR / f"{split}_layer{idx}.safetensors"
            with safe_open(str(path), framework="pt") as f:
                tensor = f.get_tensor("activations")
            shards[split][idx] = tensor
            if tensor.shape != (n_expected, HIDDEN_SIZE):
                errors.append(f"{split} layer {idx}: shape {tuple(tensor.shape)} != ({n_expected}, {HIDDEN_SIZE})")
                hidden_size_ok = False
            if tensor.dtype != torch.float32:
                errors.append(f"{split} layer {idx}: dtype {tensor.dtype} != float32")
                dtype_ok = False
            if not bool(torch.isfinite(tensor).all().item()):
                errors.append(f"{split} layer {idx}: non-finite values present")
                finite_ok = False
        if len({tuple(shards[split][idx].shape) for idx in HIDDEN_STATE_INDICES}) != 1:
            layer_count_ok = False
            errors.append(f"{split}: layer shard shapes disagree with each other")

    checks["expected_layer_count_4"] = len(HIDDEN_STATE_INDICES) == 4 and layer_count_ok
    checks["hidden_size_4096"] = hidden_size_ok
    checks["float32_storage"] = dtype_ok
    checks["all_finite"] = finite_ok

    # ---- Row/sample mapping consistency ----
    row_mapping_ok = True
    for split in SPLITS:
        if len(sample_index_mapping[split]) != shards[split][HIDDEN_STATE_INDICES[0]].shape[0]:
            row_mapping_ok = False
    checks["row_sample_mapping_consistent"] = row_mapping_ok
    checks["no_missing_rows"] = all(sid in capture_metadata for sid in all_sample_ids)

    # ---- Labels join one-to-one by sample_id ----
    checks["labels_join_one_to_one"] = set(all_sample_ids) == set(labels.keys()) and len(labels) == 200

    # ---- Prompt hashes match frozen (pre-capture) metadata ----
    hash_mismatches = [sid for sid in all_sample_ids if capture_metadata[sid]["prompt_sha256"] != rendered[sid]["prompt_sha256"]]
    checks["prompt_hashes_match_frozen_metadata"] = not hash_mismatches
    if hash_mismatches:
        errors.append(f"Prompt hash mismatches: {hash_mismatches}")

    # ---- Split isolation still valid ----
    split_isolation_ok = all(labels[sid]["split"] == split for split in SPLITS for sid in sample_index_mapping[split])
    checks["split_isolation_valid"] = split_isolation_ok

    # ---- No delayed-memory sample appears ----
    holdout_condition_ids = {e["condition_id"] for e in holdout["entries"]}
    checks["no_delayed_memory_sample_present"] = not (set(all_sample_ids) & holdout_condition_ids) and not any(
        "memory_pilot" in sid or "retrieval" in sid or "conditional_retrieval" in sid for sid in all_sample_ids
    )

    all_ok = all(checks.values())
    if not all_ok:
        print("VALIDATION FAILURES:")
        for k, v in checks.items():
            if not v:
                print(f"  - {k}")
        for e in errors:
            print(f"    {e}")

    # =====================================================================
    # Descriptive, label-aware statistics -- NOT probe training.
    # =====================================================================
    descriptive_stats: dict = {"note": "Descriptive only. Centroid distance is NOT probe performance; no classifier or threshold was trained here."}
    for split in SPLITS:
        split_stats = {}
        split_labels = [labels[sid]["exposure_label"] for sid in sample_index_mapping[split]]
        clean_rows = [i for i, l in enumerate(split_labels) if l == 0]
        attack_rows = [i for i, l in enumerate(split_labels) if l == 1]
        for idx in HIDDEN_STATE_INDICES:
            tensor = shards[split][idx]
            norms = torch.linalg.vector_norm(tensor, dim=1)
            clean_vecs = tensor[clean_rows]
            attack_vecs = tensor[attack_rows]
            clean_centroid = clean_vecs.mean(dim=0)
            attack_centroid = attack_vecs.mean(dim=0)
            centroid_distance = float(torch.linalg.vector_norm(clean_centroid - attack_centroid).item())
            clean_within_var = float(((clean_vecs - clean_centroid) ** 2).sum(dim=1).mean().item())
            attack_within_var = float(((attack_vecs - attack_centroid) ** 2).sum(dim=1).mean().item())
            split_stats[str(idx)] = {
                "vector_norm_mean": float(norms.mean().item()),
                "vector_norm_std": float(norms.std().item()),
                "vector_norm_min": float(norms.min().item()),
                "vector_norm_max": float(norms.max().item()),
                "clean_centroid_norm": float(torch.linalg.vector_norm(clean_centroid).item()),
                "attack_centroid_norm": float(torch.linalg.vector_norm(attack_centroid).item()),
                "clean_vs_attack_centroid_l2_distance": centroid_distance,
                "clean_within_class_variance": clean_within_var,
                "attack_within_class_variance": attack_within_var,
            }
        descriptive_stats[split] = split_stats

    # A small, optional 2D PCA projection (train split, layer 27) purely
    # as visualization data -- computed with torch.pca_lowrank (no
    # scikit-learn dependency needed), not used for any decision here.
    pca_layer = 27
    train_tensor = shards["train"][pca_layer]
    centered = train_tensor - train_tensor.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(centered, q=2)
    projection = (centered @ V[:, :2]).tolist()
    pca_data = {
        "split": "train",
        "layer": pca_layer,
        "sample_ids": sample_index_mapping["train"],
        "exposure_labels": [labels[sid]["exposure_label"] for sid in sample_index_mapping["train"]],
        "xy": projection,
        "note": "Descriptive visualization data only (torch.pca_lowrank, q=2). Not used to fit or select anything.",
    }
    (OUTPUT_DIR / "pca_visualization_train_layer27.json").write_text(json.dumps(pca_data))

    report = {
        "checks": checks,
        "all_checks_passed": all_ok,
        "errors": errors,
        "descriptive_statistics": descriptive_stats,
        "pca_visualization_path": str(OUTPUT_DIR / "pca_visualization_train_layer27.json"),
    }
    POST_CAPTURE_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {POST_CAPTURE_REPORT_PATH}")
    print(f"All checks passed: {all_ok}")
    if not all_ok:
        raise RuntimeError("Post-capture validation failed -- see report for details.")


if __name__ == "__main__":
    main()
