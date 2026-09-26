"""Selects the 32-example smoke-training subset (train split only) and
the 8-example bounded-diagnostic subset (validation split only, disjoint
from the smoke subset by construction since it is a different split),
deterministically, from the frozen LoRA corpus v1.

Writes:
  - results/lora_training/smoke_v1/smoke_subset_manifest.json
  - results/lora_training/smoke_v1/diagnostic_subset_manifest.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CORPUS_DIR = Path("/home/hp5/tell/results/lora_dataset/v1")
LABELS_PATH = CORPUS_DIR / "labels.jsonl"

OUTPUT_DIR = Path("/home/hp5/tell/results/lora_training/smoke_v1")
SMOKE_MANIFEST_PATH = OUTPUT_DIR / "smoke_subset_manifest.json"
DIAGNOSTIC_MANIFEST_PATH = OUTPUT_DIR / "diagnostic_subset_manifest.json"

# 16 clean + 16 attack = 32, spread as evenly as possible across each
# class's 5 kinds (40 available per kind in train): [4, 3, 3, 3, 3] in
# alphabetical kind order, so the first kind (alphabetically) gets the
# one extra slot -- fully deterministic, no randomness.
SMOKE_CLEAN_KINDS = sorted(
    ["clean_read_email_first", "clean_request_invoice", "clean_request_vendor_record", "clean_verify_before_pay", "clean_propose_payment"]
)
SMOKE_ATTACK_KINDS = sorted(
    ["attack_email_beneficiary_change", "attack_email_forged_tool_result", "attack_invoice_instruction_injection", "attack_vendor_unverified", "attack_vendor_identity_conflict"]
)
SMOKE_COUNTS_PER_KIND = [4, 3, 3, 3, 3]

# 4 clean + 4 attack diagnostic kinds, chosen to skip the trivial
# clean_read_email_first (no tool result observed yet, least informative
# for a base-vs-adapter behavioral check) and to span all three attack
# surfaces (email, invoice, vendor_record).
DIAGNOSTIC_CLEAN_KINDS = ["clean_request_invoice", "clean_request_vendor_record", "clean_verify_before_pay", "clean_propose_payment"]
DIAGNOSTIC_ATTACK_KINDS = ["attack_email_beneficiary_change", "attack_invoice_instruction_injection", "attack_vendor_unverified", "attack_vendor_identity_conflict"]


def _load_labels() -> list[dict]:
    return [json.loads(line) for line in LABELS_PATH.open()]


def main() -> None:
    labels = _load_labels()
    by_split_kind: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for l in labels:
        kind = l["template_family_id"].rsplit("__", 1)[0]
        by_split_kind[(l["split"], kind)].append(l)
    for key in by_split_kind:
        by_split_kind[key].sort(key=lambda l: l["docid"])

    # ---- Smoke subset (train only) ----
    smoke_sample_ids: list[str] = []
    smoke_breakdown = []
    for kinds, cls in ((SMOKE_CLEAN_KINDS, "clean"), (SMOKE_ATTACK_KINDS, "attack")):
        for kind, n in zip(kinds, SMOKE_COUNTS_PER_KIND):
            candidates = by_split_kind[("train", kind)]
            if len(candidates) < n:
                raise RuntimeError(f"Not enough train examples for kind {kind}: need {n}, have {len(candidates)}")
            chosen = candidates[:n]
            smoke_sample_ids.extend(l["sample_id"] for l in chosen)
            smoke_breakdown.append({"kind": kind, "class": cls, "n": n, "sample_ids": [l["sample_id"] for l in chosen]})

    if len(smoke_sample_ids) != 32:
        raise RuntimeError(f"Expected 32 smoke examples, got {len(smoke_sample_ids)}")
    n_clean = sum(1 for sid in smoke_sample_ids if any(sid in b["sample_ids"] and b["class"] == "clean" for b in smoke_breakdown))
    n_attack = 32 - n_clean
    if n_clean != 16 or n_attack != 16:
        raise RuntimeError(f"Smoke subset not balanced: {n_clean} clean / {n_attack} attack")

    attack_surfaces_present = set()
    for kind in SMOKE_ATTACK_KINDS:
        if kind.startswith("attack_email"):
            attack_surfaces_present.add("email")
        elif kind.startswith("attack_invoice"):
            attack_surfaces_present.add("invoice")
        elif kind.startswith("attack_vendor"):
            attack_surfaces_present.add("vendor_record")
    if len(attack_surfaces_present) < 2:
        raise RuntimeError(f"Smoke subset covers only {attack_surfaces_present}, need >= 2 attack surfaces")

    smoke_manifest = {
        "n_examples": 32,
        "n_clean": n_clean,
        "n_attack": n_attack,
        "source_split": "train",
        "attack_surfaces_present": sorted(attack_surfaces_present),
        "sample_ids": smoke_sample_ids,
        "breakdown": smoke_breakdown,
        "selection_method": "deterministic: first N train docids (sorted) per kind, counts [4,3,3,3,3] in alphabetical kind order per class",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_MANIFEST_PATH.write_text(json.dumps(smoke_manifest, indent=2))
    print(f"Wrote {SMOKE_MANIFEST_PATH}: {len(smoke_sample_ids)} examples, surfaces={sorted(attack_surfaces_present)}")

    # ---- Diagnostic subset (validation only, disjoint by split) ----
    diagnostic_sample_ids: list[str] = []
    diagnostic_breakdown = []
    for kinds, cls in ((DIAGNOSTIC_CLEAN_KINDS, "clean"), (DIAGNOSTIC_ATTACK_KINDS, "attack")):
        for kind in kinds:
            candidates = by_split_kind[("validation", kind)]
            if not candidates:
                raise RuntimeError(f"No validation examples for kind {kind}")
            chosen = candidates[0]
            diagnostic_sample_ids.append(chosen["sample_id"])
            diagnostic_breakdown.append({"kind": kind, "class": cls, "sample_id": chosen["sample_id"], "docid": chosen["docid"]})

    if len(diagnostic_sample_ids) != 8:
        raise RuntimeError(f"Expected 8 diagnostic examples, got {len(diagnostic_sample_ids)}")
    if len(set(diagnostic_sample_ids) & set(smoke_sample_ids)) != 0:
        raise RuntimeError("Diagnostic subset overlaps with smoke subset")

    diagnostic_manifest = {
        "n_examples": 8,
        "n_clean": 4,
        "n_attack": 4,
        "source_split": "validation",
        "sample_ids": diagnostic_sample_ids,
        "breakdown": diagnostic_breakdown,
        "disjoint_from_smoke_subset": True,
        "disjoint_from_lora_test_split": True,
        "note": "Drawn only from the validation split -- the 100-example test split is never inspected or scored by this smoke experiment.",
    }
    DIAGNOSTIC_MANIFEST_PATH.write_text(json.dumps(diagnostic_manifest, indent=2))
    print(f"Wrote {DIAGNOSTIC_MANIFEST_PATH}: {len(diagnostic_sample_ids)} examples")


if __name__ == "__main__":
    main()
