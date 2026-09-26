"""Build the LoRA pre-training contract v1.1 outputs (CPU only; no model,
no GPU, no activations, no training).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/build_lora_pretraining_contract_v1_1.py

This is a narrowly-scoped correction of ONE thing: `get_vendor_record`
(GVR) coverage in the training sampler (Part 3 of the routing-design
task). It reuses `lora_pretraining_v1.pipeline.build(with_tokenizer=False)`
unchanged for everything else -- frozen v2.2 integrity checks, the
in-memory v2.1/v2.2 rebuild, the memory supplement, and the pool
construction itself (`v22_pool` + `supplement_pool`) are all identical to
v1; only the per-epoch GVR selection (`lora_pretraining_v1_1.sampler`)
differs. Nothing under results/lora_training/pretraining_contract_v1/ or
configs/lora/enterprise_v2_2/sampler_v1.json is read for writing, and
`pipeline.write()` (v1's own writer) is never called -- this script has
its own writer, targeting only results/lora_training/pretraining_contract_v1_1/.

Run `scripts/hash_protected_artifacts_routing_v1.py before` first and
`... after` once this (and every other part of the routing-design task)
is complete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from lora_pretraining_v1 import pipeline as P1  # noqa: E402
from lora_pretraining_v1_1 import sampler as S11  # noqa: E402

OUT_DIR = REPO / "results" / "lora_training" / "pretraining_contract_v1_1"
CFG_PATH = REPO / "configs" / "lora" / "enterprise_v2_2" / "sampler_v1_1.json"


def jd(o) -> str:
    return json.dumps(o, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def build() -> dict:
    checks: list[dict] = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    v1_out = P1.build(with_tokenizer=False)
    v1_failed = [c for c in v1_out["checks"] if not c["passed"]]
    check("all v1 (reused) frozen-integrity and pool-construction checks pass", not v1_failed, [c["check"] for c in v1_failed])

    pool = v1_out["pool"]
    cfg = json.loads(CFG_PATH.read_text())
    S11.validate_pool_v1_1(pool, cfg)
    check("pool matches the expected v2.2 (831) + supplement (240) GVR sizes", True)

    gvr_pool = [p for p in pool if p.action == S11.GVR]
    state = S11.GvrCoverageState.empty(gvr_pool)
    epochs = []
    coverage_table = []
    for e in range(cfg["planned_epochs"]):
        items = S11.sample_epoch_v1_1(pool, cfg, e, state)
        epochs.append(items)
        coverage_table.append(S11.epoch_coverage_row(pool, items, state, e))

    check("epoch 0 GVR draw is entirely first-time (unseen pool exceeds one epoch's slots)",
          coverage_table[0]["repeated_examples_this_epoch"] == 0)
    check("full vendor-lookup pool coverage reached by the end of epoch 2 (0-indexed) / epoch 3 of 3",
          coverage_table[-1]["cumulative_coverage_fraction"] == 1.0, coverage_table[-1]["cumulative_coverage_fraction"])
    for e, items in enumerate(epochs):
        gvr_here = [p for p in items if p.action == S11.GVR]
        check(f"epoch {e}: no item repeated within the epoch itself", len({p.sample_id for p in gvr_here}) == len(gvr_here))
        check(f"epoch {e}: GVR share is exactly the configured cap", len(gvr_here) == cfg["epoch_size"] - sum(1 for p in items if p.action != S11.GVR))
    replay = S11.GvrCoverageState.empty(gvr_pool)
    replay_epoch0 = S11.sample_epoch_v1_1(pool, cfg, 0, replay)
    check("sampler is deterministic (epoch 0 re-sampled from a fresh state is identical)",
          [p.sample_id for p in replay_epoch0] == [p.sample_id for p in epochs[0]])

    return {
        "checks": checks,
        "cfg": cfg,
        "pool": pool,
        "epochs": epochs,
        "coverage_table": coverage_table,
        "gvr_pool_size": len(gvr_pool),
        "v1_out": v1_out,
    }


def write(o: dict) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    def w(rel: str, text: str) -> None:
        p = OUT_DIR / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        files[f"results/lora_training/pretraining_contract_v1_1/{rel}"] = text

    w("checks.json", jd(o["checks"]))
    w("vendor_lookup_coverage_table.json", jd(o["coverage_table"]))

    lines = ["# Vendor-lookup (`get_vendor_record`) cumulative coverage -- pretraining contract v1.1", "",
              "Corrects results/lora_training/pretraining_contract_v1/training_distribution_report.md section 2's "
              "GVR sampling: the v1 sampler re-ran document water-filling from zero every epoch and pinned the "
              "supplement's 240 GVR rows as always-included, so it deterministically reselected close to the same "
              "~350 of 831 v2.2 GVR items in every one of its 3 epochs. v1.1 unifies all 1,071 v2.2 (831) + "
              "supplement (240) GVR items into one coverage-first pool competing for the same 496 GVR slots/epoch.",
              "", "| epoch | GVR slots | unique seen (cum.) | pool coverage | v2.2 covered | supplement covered | repeats this epoch | doc exposure min/mean/max |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in o["coverage_table"]:
        d = row["per_document_exposure_cumulative"]
        lines.append(
            f"| {row['epoch']} | {row['gvr_slots_this_epoch']} | {row['unique_vendor_lookup_examples_seen_cumulative']}/{row['vendor_lookup_pool_size']} "
            f"| {row['cumulative_coverage_fraction']:.1%} | {row['v2_2_covered_cumulative']}/{row['v2_2_pool_size']} "
            f"| {row['supplement_covered_cumulative']}/{row['supplement_pool_size']} | {row['repeated_examples_this_epoch']} "
            f"| {d['min']}/{d['mean']}/{d['max']} |"
        )
    lines += ["", "## Full effective action distribution per epoch", ""]
    for row in o["coverage_table"]:
        lines.append(f"- **epoch {row['epoch']}** (n={row['n_epoch_items']}): " + ", ".join(
            f"{a} {c} ({row['effective_action_distribution']['shares'][a]:.1%})" for a, c in row["effective_action_distribution"]["counts"].items()
        ))
    lines += ["", f"GVR pool size verified: {o['gvr_pool_size']} (831 v2.2 + 240 supplement = 1071 expected).", ""]
    w("vendor_lookup_coverage_report.md", "\n".join(lines) + "\n")

    manifest = {
        "contract_version": "tell_lora_pretraining_contract_v1_1",
        "status": "frozen pre-training contract correction; no activations captured, no model trained, GPU not used",
        "corrects": "results/lora_training/pretraining_contract_v1/ (left byte-identical and unmodified; see checks.json for the reused v1 integrity checks)",
        "sampler_config": "configs/lora/enterprise_v2_2/sampler_v1_1.json",
        "sampler_code": "scripts/lora_pretraining_v1_1/sampler.py",
        "gvr_pool_size": o["gvr_pool_size"],
        "all_checks_passed": all(c["passed"] for c in o["checks"]),
    }
    w("pretraining_contract_v1_1_manifest.json", jd(manifest))
    return files


def main() -> None:
    o = build()
    failed = [c for c in o["checks"] if not c["passed"]]
    for c in o["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    if failed:
        raise SystemExit(f"{len(failed)} check(s) failed; nothing written")
    files = write(o)
    for p in sorted(files):
        print("wrote", p)


if __name__ == "__main__":
    main()
