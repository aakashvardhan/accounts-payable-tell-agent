"""Deterministic, coverage-first action-stratified LoRA sampler (v1.1).

Corrects the one defect in `scripts.lora_pretraining_v1.sampler` (v1):
that sampler calls `sample_epoch(pool, cfg, epoch)` independently per
epoch, starting its water-filling document counters at zero every time.
Because the water-filling rule always prefers the least-exposed document
and (for a fixed pool and a fixed set of already-included items) the
"least exposed" ranking is nearly identical epoch to epoch, this
deterministically reselects close to the SAME `get_vendor_record` (GVR)
items in every epoch instead of cycling through the pool -- confirmed by
results/lora_training/pretraining_contract_v1/training_distribution_report.md,
where epoch 0/1/2's GVR item sets are, in effect, the same repeated draw
rather than three different slices of the 831 v2.2 + 240 supplement GVR
pool. Only ~350 of the 831 v2.2 GVR examples are ever seen across all
three planned epochs.

Fix: this module keeps CROSS-EPOCH state (`GvrCoverageState`) -- how many
times each GVR sample has already been drawn in prior epochs -- and fills
each epoch's fixed 496-slot GVR quota by:

  1. unseen items first (cumulative exposure == 0);
  2. within either the unseen or (once exhausted) the previously-seen
     pool, the item whose DOCUMENT has the lowest cumulative exposure so
     far wins ties, so no one document is drained before others are
     touched;
  3. remaining ties broken by sha256(seed:epoch:"gvr":sample_id) --
     `seed` is the same fixed, published sampler seed, but every epoch
     mixes in its own epoch index, so each epoch's tie-break order is
     distinct and reproducible;
  4. an item is repeated across epochs only after the entire GVR pool has
     been seen at least once (verified empirically below: full coverage
     completes partway through epoch 2 of 3 for the frozen v2.2 + supplement
     pool -- see build_pretraining_contract_v1_1.py's coverage table).

This module is purely additive: `scripts.lora_pretraining_v1.sampler` is
never imported for its `sample_epoch`/`validate_pool` functions (those
stay exactly as they were, still used by the historical, already-frozen
pretraining_contract_v1 report) -- only its `PoolItem` dataclass and `GVR`
/ `derived_epoch_size` constants are reused, since the pool definition
itself (which rows exist, which action/document/split they carry) is
unchanged by this fix; only how GVR items are chosen per epoch changes.

Everything about non-GVR items is untouched: every non-GVR pool item
(including the supplement's search_memory and terminal rows) is still
included in EVERY epoch exactly once, exactly as v1 did -- there is no
coverage problem for those, since 1159 < 1655 with room to spare.

Unlike v1, this module does NOT special-case supplement GVR items as
always-included "priority" rows outside the coverage cycle: pinning 240
of the 1,071-item combined GVR pool to appear in literally every epoch
would itself re-create a coverage gap (leaving only 256 of the remaining
496 GVR slots per epoch for the other 831 v2.2 items, requiring >3 epochs
to cover them once). Instead all 1,071 GVR items -- v2.2 and supplement
alike -- compete equally for the 496 GVR slots each epoch, which is what
lets 3 epochs x 496 slots = 1,488 draws cover the ~1,071-item pool at
least once with slots to spare -- exactly the arithmetic Part 3 of the
routing-design task calls for.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path("/home/hp5/tell/scripts")))

from lora_pretraining_v1.sampler import GVR, PoolItem, derived_epoch_size  # noqa: E402

__all__ = ["GVR", "PoolItem", "derived_epoch_size", "GvrCoverageState", "sample_epoch_v1_1", "epoch_coverage_row", "validate_pool_v1_1"]


def _key(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def validate_pool_v1_1(pool: list[PoolItem], cfg: dict) -> None:
    """Same structural checks as v1's `validate_pool`, plus the explicit
    "fail loudly if the pool changes" requirement (Part 3, requirement 8)
    applied to the exact v2.2/supplement GVR split, not just the total."""
    if any(p.split != "train" for p in pool):
        raise ValueError("sampler pool must contain LoRA training-split items only")
    if len({p.sample_id for p in pool}) != len(pool):
        raise ValueError("duplicate sample_id in pool")
    n_non = sum(1 for p in pool if p.action != GVR)
    want = derived_epoch_size(n_non, cfg["gvr_cap"])
    if cfg["epoch_size"] != want:
        raise ValueError(f"configured epoch_size {cfg['epoch_size']} != derived {want}; refusing to change the epoch definition silently")
    n_gvr = sum(1 for p in pool if p.action == GVR)
    if "expected_gvr_pool_size" in cfg and n_gvr != cfg["expected_gvr_pool_size"]:
        raise ValueError(f"GVR pool size changed: expected {cfg['expected_gvr_pool_size']}, found {n_gvr}; refusing to silently resize the coverage target")
    n_gvr_v22 = sum(1 for p in pool if p.action == GVR and p.source == "lora_v2_2")
    n_gvr_sup = sum(1 for p in pool if p.action == GVR and p.source != "lora_v2_2")
    if "expected_gvr_v2_2" in cfg and n_gvr_v22 != cfg["expected_gvr_v2_2"]:
        raise ValueError(f"v2.2 GVR pool size changed: expected {cfg['expected_gvr_v2_2']}, found {n_gvr_v22}")
    if "expected_gvr_supplement" in cfg and n_gvr_sup != cfg["expected_gvr_supplement"]:
        raise ValueError(f"supplement GVR pool size changed: expected {cfg['expected_gvr_supplement']}, found {n_gvr_sup}")


@dataclass
class GvrCoverageState:
    """Cross-epoch exposure bookkeeping, passed epoch to epoch by the
    caller (this module has no global/module-level mutable state)."""

    sample_exposure: dict[str, int]
    document_exposure: dict[str, int]

    @classmethod
    def empty(cls, gvr_pool: list[PoolItem]) -> "GvrCoverageState":
        return cls(
            sample_exposure={p.sample_id: 0 for p in gvr_pool},
            document_exposure={d: 0 for d in {p.document_group_id for p in gvr_pool}},
        )

    def unseen_count(self) -> int:
        return sum(1 for v in self.sample_exposure.values() if v == 0)

    def coverage_fraction(self) -> float:
        n = len(self.sample_exposure)
        return 0.0 if n == 0 else 1 - self.unseen_count() / n


def _fill_gvr_slots(gvr_pool: list[PoolItem], n_slots: int, seed, epoch: int, state: GvrCoverageState) -> list[PoolItem]:
    by_id = {p.sample_id: p for p in gvr_pool}
    if len(by_id) != len(gvr_pool):
        raise ValueError("duplicate sample_id within the GVR pool")

    remaining_unseen = [p for p in gvr_pool if state.sample_exposure[p.sample_id] == 0]
    remaining_seen = [p for p in gvr_pool if state.sample_exposure[p.sample_id] > 0]
    chosen: list[PoolItem] = []
    chosen_ids: set[str] = set()

    def draw_from(candidates: list[PoolItem]) -> None:
        pool_left = list(candidates)
        while pool_left and len(chosen) < n_slots:
            pool_left.sort(key=lambda p: (state.document_exposure[p.document_group_id], _key(seed, epoch, "gvr", p.sample_id)))
            p = pool_left.pop(0)
            if p.sample_id in chosen_ids:
                continue
            chosen.append(p)
            chosen_ids.add(p.sample_id)
            state.document_exposure[p.document_group_id] += 1

    draw_from(remaining_unseen)  # 1. unseen first, lowest document exposure first, seeded tie-break
    if len(chosen) < n_slots:
        draw_from(remaining_seen)  # 4. only repeat once the unseen pool for this epoch is exhausted

    if len(chosen) != n_slots:
        raise AssertionError(f"could not fill {n_slots} GVR slots without an in-epoch repeat (GVR pool has only {len(gvr_pool)} items)")
    if len({p.sample_id for p in chosen}) != n_slots:
        raise AssertionError("an item was selected twice within the same epoch")

    for p in chosen:
        state.sample_exposure[p.sample_id] += 1
    chosen.sort(key=lambda p: _key(seed, epoch, "order", p.sample_id))
    return chosen


def sample_epoch_v1_1(pool: list[PoolItem], cfg: dict, epoch: int, state: GvrCoverageState) -> list[PoolItem]:
    """Like v1's `sample_epoch`, but GVR slots are filled by coverage-first
    selection against the persistent `state` (mutated in place -- pass the
    same `state` object across epochs 0, 1, 2, ... in order)."""
    validate_pool_v1_1(pool, cfg)
    seed = cfg["seed"]
    N = cfg["epoch_size"]

    non_gvr = [p for p in pool if p.action != GVR]
    gvr_pool = [p for p in pool if p.action == GVR]
    n_gvr_slots = N - len(non_gvr)
    cap_limit = math.floor(cfg["gvr_cap"] * N)
    if n_gvr_slots > cap_limit:
        raise ValueError(f"epoch would need {n_gvr_slots} GVR slots, exceeding the {cfg['gvr_cap']:.0%} cap ({cap_limit})")

    gvr_chosen = _fill_gvr_slots(gvr_pool, n_gvr_slots, seed, epoch, state)
    chosen = non_gvr + gvr_chosen
    if len(chosen) != N:
        raise AssertionError(f"epoch has {len(chosen)} items, expected {N}")
    chosen.sort(key=lambda p: _key(seed, epoch, "order", p.sample_id))
    return chosen


def epoch_coverage_row(pool: list[PoolItem], epoch_items: list[PoolItem], state: GvrCoverageState, epoch: int) -> dict:
    """One row of the Part-3 cumulative coverage table, reported after
    sampling `epoch` (so `state` already reflects this epoch's draws)."""
    gvr_pool = [p for p in pool if p.action == GVR]
    n_pool = len(gvr_pool)
    v22_ids = {p.sample_id for p in gvr_pool if p.source == "lora_v2_2"}
    sup_ids = {p.sample_id for p in gvr_pool if p.source != "lora_v2_2"}
    seen_ids = {sid for sid, n in state.sample_exposure.items() if n > 0}
    epoch_gvr = [p for p in epoch_items if p.action == GVR]
    repeats_this_epoch = sum(1 for p in epoch_gvr if state.sample_exposure[p.sample_id] > 1)
    doc_vals = list(state.document_exposure.values())
    act = Counter(p.action for p in epoch_items)
    return {
        "epoch": epoch,
        "n_epoch_items": len(epoch_items),
        "gvr_slots_this_epoch": len(epoch_gvr),
        "unique_vendor_lookup_examples_seen_cumulative": len(seen_ids),
        "vendor_lookup_pool_size": n_pool,
        "cumulative_coverage_fraction": round(len(seen_ids) / n_pool, 4) if n_pool else None,
        "v2_2_covered_cumulative": len(seen_ids & v22_ids),
        "v2_2_pool_size": len(v22_ids),
        "supplement_covered_cumulative": len(seen_ids & sup_ids),
        "supplement_pool_size": len(sup_ids),
        "repeated_examples_this_epoch": repeats_this_epoch,
        "first_time_examples_this_epoch": len(epoch_gvr) - repeats_this_epoch,
        "per_document_exposure_cumulative": {
            "n_documents": len(doc_vals),
            "min": min(doc_vals) if doc_vals else 0,
            "mean": round(sum(doc_vals) / len(doc_vals), 3) if doc_vals else 0.0,
            "max": max(doc_vals) if doc_vals else 0,
            "max_over_mean": round(max(doc_vals) / (sum(doc_vals) / len(doc_vals)), 3) if doc_vals and sum(doc_vals) else None,
        },
        "effective_action_distribution": {
            "counts": dict(sorted(act.items())),
            "shares": {a: round(c / len(epoch_items), 4) for a, c in sorted(act.items())},
        },
    }
