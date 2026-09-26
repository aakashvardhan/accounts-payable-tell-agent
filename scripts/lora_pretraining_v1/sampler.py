"""Deterministic action-stratified LoRA sampler (v1).

Pool: ONLY the LoRA training split -- frozen v2.2 `lora_v2_2` train rows
plus the memory-supplement train rows. Validation and test are never read
into the pool (the sampler refuses any non-train row), so their natural
distributions are untouched.

Exact rule, per epoch e (fixed seed, fixed epoch size N):

  1. Every non-`get_vendor_record` item (all other actions, including the
     supplement's `search_memory` and terminal items) is included exactly
     once.
  2. `get_vendor_record` (GVR) fills the remaining N - n_nonGVR slots,
     which must be <= floor(cap * N) with cap = 0.30.
  3. Priority GVR items are always included: the supplement's
     post-retrieval trusted-verification steps and its matched negatives
     (they carry the "verify after memory" and "do not search without need"
     signals and must not be diluted relative to `search_memory`).
  4. The remaining N - n_nonGVR - |priority| GVR slots come from the v2.2 GVR items,
     allocated across their step roles (trusted_resolution_step /
     workflow_read_step) in proportion to pool counts (largest remainder),
     and within a step role by document water-filling: repeatedly take one
     unused item from the document with the fewest effective samples so far
     in this epoch (ties and the item choice broken by
     sha256(seed:epoch:...)).
  5. No item appears twice in an epoch. The epoch order is
     sorted by sha256(seed:epoch:order:sample_id).

N is a fixed configured constant. It was set so that rule 1 and the cap
hold exactly with no repetition: N = n_nonGVR + floor(cap * n_nonGVR /
(1 - cap)). If the pool ever changes so that N no longer satisfies that
identity, the sampler raises instead of silently changing N.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

GVR = "get_vendor_record"


@dataclass(frozen=True)
class PoolItem:
    sample_id: str
    source: str
    split: str
    contract: str
    action: str
    document_group_id: str
    step_role: str
    priority: bool


def _key(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def derived_epoch_size(n_non_gvr: int, cap: float) -> int:
    return n_non_gvr + math.floor(cap * n_non_gvr / (1 - cap))


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def validate_pool(pool: list[PoolItem], cfg: dict) -> None:
    if any(p.split != "train" for p in pool):
        raise ValueError("sampler pool must contain LoRA training-split items only")
    if len({p.sample_id for p in pool}) != len(pool):
        raise ValueError("duplicate sample_id in pool")
    n_non = sum(1 for p in pool if p.action != GVR)
    want = derived_epoch_size(n_non, cfg["gvr_cap"])
    if cfg["epoch_size"] != want:
        raise ValueError(f"configured epoch_size {cfg['epoch_size']} != derived {want}; refusing to change the epoch definition silently")
    g = math.floor(cfg["gvr_cap"] * cfg["epoch_size"])
    if cfg["epoch_size"] - n_non > g:
        raise ValueError("epoch would exceed the GVR cap")
    if sum(1 for p in pool if p.action == GVR and p.priority) > cfg["epoch_size"] - n_non:
        raise ValueError("priority GVR items exceed the GVR slots")


def sample_epoch(pool: list[PoolItem], cfg: dict, epoch: int) -> list[PoolItem]:
    validate_pool(pool, cfg)
    seed = cfg["seed"]
    N = cfg["epoch_size"]
    chosen = [p for p in pool if p.action != GVR or p.priority]
    count = Counter(p.document_group_id for p in chosen)
    for p in pool:
        count.setdefault(p.document_group_id, 0)

    rest = [p for p in pool if p.action == GVR and not p.priority]
    by_role: dict[str, dict[str, list[PoolItem]]] = defaultdict(lambda: defaultdict(list))
    for p in rest:
        by_role[p.step_role][p.document_group_id].append(p)
    for role in by_role:
        for d in by_role[role]:
            by_role[role][d].sort(key=lambda p: _key(seed, epoch, "item", p.sample_id))
    quota = _largest_remainder(N - len(chosen), {r: sum(len(v) for v in docs.values()) for r, docs in by_role.items()})
    taken = Counter()
    while sum(taken.values()) < sum(quota.values()):
        open_roles = [r for r in quota if taken[r] < quota[r]]
        role = min(open_roles, key=lambda r: ((taken[r] + 1) / quota[r], r))
        docs = [d for d, items in by_role[role].items() if items]
        d = min(docs, key=lambda d: (count[d], _key(seed, epoch, "doc", d)))
        chosen.append(by_role[role][d].pop(0))
        count[d] += 1
        taken[role] += 1
    if len(chosen) != N:
        raise AssertionError(f"epoch has {len(chosen)} items, expected {N}")
    chosen.sort(key=lambda p: _key(seed, epoch, "order", p.sample_id))
    return chosen


def epoch_statistics(pool: list[PoolItem], epoch_items: list[PoolItem], cfg: dict) -> dict:
    n = len(epoch_items)
    act = Counter(p.action for p in epoch_items)
    per_doc = Counter(p.document_group_id for p in epoch_items)
    docs = sorted({p.document_group_id for p in pool})
    vals = [per_doc.get(d, 0) for d in docs]
    mean = sum(vals) / len(vals)
    return {
        "n": n,
        "action_counts": dict(sorted(act.items())),
        "action_shares": {a: round(c / n, 4) for a, c in sorted(act.items())},
        "gvr_share": round(act[GVR] / n, 4),
        "search_memory_share": round(act["search_memory"] / n, 4),
        "max_item_repeats": max(Counter(p.sample_id for p in epoch_items).values()),
        "per_document": {"n_documents": len(docs), "min": min(vals), "max": max(vals), "mean": round(mean, 3),
                         "max_over_mean": round(max(vals) / mean, 3), "histogram": dict(sorted(Counter(vals).items()))},
        "by_source": dict(sorted(Counter(p.source for p in epoch_items).items())),
        "gvr_by_step_role": dict(sorted(Counter(p.step_role for p in epoch_items if p.action == GVR).items())),
    }


def check_epoch(pool: list[PoolItem], epoch_items: list[PoolItem], cfg: dict) -> list[str]:
    s = epoch_statistics(pool, epoch_items, cfg)
    errs = []
    if s["action_counts"].get(GVR, 0) > math.floor(cfg["gvr_cap"] * s["n"]):
        errs.append(f"GVR share {s['gvr_share']} > cap")
    if set(s["action_counts"]) != {p.action for p in pool}:
        errs.append("a target action represented in the pool is missing from the epoch")
    if s["max_item_repeats"] > cfg["max_item_repeats_per_epoch"]:
        errs.append("an item is repeated within an epoch")
    if s["per_document"]["max_over_mean"] > cfg["max_document_count_over_mean"]:
        errs.append(f"document oversampled: max/mean {s['per_document']['max_over_mean']}")
    if s["per_document"]["min"] < 1:
        errs.append("a training document is absent from the epoch")
    lo, hi = cfg["search_memory_share_target"]
    if not lo <= s["search_memory_share"] <= hi:
        errs.append(f"search_memory share {s['search_memory_share']} outside {lo}-{hi}")
    return errs
