"""Build the 3 real training-epoch manifests for the frozen v1.1 sampler
contract, reading ONLY already-frozen, hash-verified files -- never calling
`lora_pretraining_v1.pipeline.build()` (which regenerates v2.1/v2.2 from raw
DocILE via `select_global` and currently fails in this environment: "could
only draw 509 vendor/cluster-isolated documents (need 600)"). That failure
is a raw-DocILE-pool availability issue in reconstructing the corpus from
scratch; it has nothing to do with the already-built, already-frozen files
this script reads, so it does not block using them as-is.

Reuses, unmodified: `lora_pretraining_v1.sampler.PoolItem` / `.pipeline`'s
`v22_pool`/`supplement_pool`/`load_frozen_v22_lora` (frozen-file readers
only, no DocILE), and `lora_pretraining_v1_1.sampler`'s coverage-first
epoch sampler -- exactly the tested, contract-verified code path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from lora_pretraining_v1.pipeline import supplement_pool, v22_pool  # noqa: E402
from lora_pretraining_v1_1 import sampler as S11  # noqa: E402

V22_DIR = REPO / "results/enterprise_corpus/v2_2"
SUP_DIR = REPO / "results/enterprise_corpus/v2_2_training_supplement"
CFG_PATH = REPO / "configs/lora/enterprise_v2_2/sampler_v1_1.json"
OUT = REPO / "results/lora_training/agent_s_v1/epoch_manifests"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_verified(path: Path, expected_sha256: str) -> list[dict]:
    raw = path.read_bytes()
    got = sha(raw)
    if got != expected_sha256:
        raise RuntimeError(f"PROTECTED-ARTIFACT MISMATCH: {path} expected {expected_sha256} got {got}")
    return [json.loads(l) for l in raw.decode().splitlines()]


def load_pool() -> tuple[list, dict, dict]:
    v22_man = json.loads((V22_DIR / "lora_v2_2_manifest.json").read_text())
    v22_labels = load_verified(V22_DIR / "lora_v2_2_labels.jsonl", v22_man["files"]["labels"]["sha256"])
    v22_inputs = {r["sample_id"]: r for r in load_verified(V22_DIR / "lora_v2_2_inputs.jsonl", v22_man["files"]["inputs"]["sha256"])}
    v22_targets = {r["sample_id"]: r for r in load_verified(V22_DIR / "lora_v2_2_targets.jsonl", v22_man["files"]["targets"]["sha256"])}

    sup_man = json.loads((SUP_DIR / "memory_supplement_v1_manifest.json").read_text())
    sup_files = sup_man["files"]
    sup_labels = load_verified(SUP_DIR / "memory_supplement_v1_labels.jsonl", sup_files["memory_supplement_v1_labels.jsonl"]["sha256"])
    sup_inputs = {r["sample_id"]: r for r in load_verified(SUP_DIR / "memory_supplement_v1_inputs.jsonl", sup_files["memory_supplement_v1_inputs.jsonl"]["sha256"])}
    sup_targets = {r["sample_id"]: r for r in load_verified(SUP_DIR / "memory_supplement_v1_targets.jsonl", sup_files["memory_supplement_v1_targets.jsonl"]["sha256"])}

    pool = v22_pool(v22_labels) + supplement_pool(sup_labels)
    inputs = {**v22_inputs, **sup_inputs}
    targets = {**v22_targets, **sup_targets}
    return pool, inputs, targets


def main() -> dict:
    pool, inputs, targets = load_pool()
    cfg = json.loads(CFG_PATH.read_text())
    S11.validate_pool_v1_1(pool, cfg)

    # Leakage guard: no training-pool sample_id may collide with a
    # validation/test split id from the same two label files.
    v22_all = [json.loads(l) for l in (V22_DIR / "lora_v2_2_labels.jsonl").read_text().splitlines()]
    sup_all = [json.loads(l) for l in (SUP_DIR / "memory_supplement_v1_labels.jsonl").read_text().splitlines()]
    non_train_ids = {r["sample_id"] for r in v22_all if r["split"] != "train"} | {r["sample_id"] for r in sup_all if r["split"] != "train"}
    pool_ids = {p.sample_id for p in pool}
    overlap = pool_ids & non_train_ids
    if overlap:
        raise RuntimeError(f"LEAKAGE: {len(overlap)} training-pool sample_ids also appear in a non-train split")

    gvr_pool = [p for p in pool if p.action == S11.GVR]
    state = S11.GvrCoverageState.empty(gvr_pool)
    epochs, coverage_table = [], []
    for e in range(cfg["planned_epochs"]):
        items = S11.sample_epoch_v1_1(pool, cfg, e, state)
        assert len(items) == cfg["epoch_size"]
        epochs.append(items)
        coverage_table.append(S11.epoch_coverage_row(pool, items, state, e))

    # Reproduce the frozen contract's own verified numbers exactly.
    expect_cov = [0.4631, 0.9262, 1.0]
    got_cov = [round(r["cumulative_coverage_fraction"], 4) for r in coverage_table]
    assert all(abs(a - b) < 1e-3 for a, b in zip(got_cov, expect_cov)), (got_cov, expect_cov)

    replay = S11.GvrCoverageState.empty(gvr_pool)
    replay0 = S11.sample_epoch_v1_1(pool, cfg, 0, replay)
    assert [p.sample_id for p in replay0] == [p.sample_id for p in epochs[0]], "sampler is not deterministic"

    missing = [p.sample_id for e in epochs for p in e if p.sample_id not in inputs or p.sample_id not in targets]
    if missing:
        raise RuntimeError(f"{len(missing)} sampled sample_ids missing from frozen inputs/targets, e.g. {missing[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    for e, items in enumerate(epochs):
        rows = [{"epoch": e, "position": i, "sample_id": p.sample_id, "source": p.source, "action": p.action,
                "document_group_id": p.document_group_id, "step_role": p.step_role} for i, p in enumerate(items)]
        (OUT / f"epoch_{e}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (OUT / "coverage_table.json").write_text(json.dumps(coverage_table, indent=1))
    (OUT / "manifest.json").write_text(json.dumps({
        "sampler_config_path": str(CFG_PATH), "sampler_config_sha256": sha(CFG_PATH.read_bytes()),
        "pool_size": len(pool), "gvr_pool_size": len(gvr_pool), "epoch_size": cfg["epoch_size"],
        "n_epochs": cfg["planned_epochs"], "total_examples": sum(len(e) for e in epochs),
        "coverage_by_epoch": got_cov, "deterministic_replay_verified": True, "train_test_leakage_overlap": 0,
        "source_files_sha256": {
            "lora_v2_2_labels": sha((V22_DIR / "lora_v2_2_labels.jsonl").read_bytes()),
            "lora_v2_2_inputs": sha((V22_DIR / "lora_v2_2_inputs.jsonl").read_bytes()),
            "lora_v2_2_targets": sha((V22_DIR / "lora_v2_2_targets.jsonl").read_bytes()),
            "memory_supplement_v1_labels": sha((SUP_DIR / "memory_supplement_v1_labels.jsonl").read_bytes()),
            "memory_supplement_v1_inputs": sha((SUP_DIR / "memory_supplement_v1_inputs.jsonl").read_bytes()),
            "memory_supplement_v1_targets": sha((SUP_DIR / "memory_supplement_v1_targets.jsonl").read_bytes()),
        },
    }, indent=1))
    print(json.dumps({"pool_size": len(pool), "gvr_pool_size": len(gvr_pool), "epoch_sizes": [len(e) for e in epochs],
                      "coverage_by_epoch": got_cov, "leakage_overlap": len(overlap)}, indent=1))
    return {"pool": pool, "inputs": inputs, "targets": targets, "epochs": epochs}


if __name__ == "__main__":
    main()
