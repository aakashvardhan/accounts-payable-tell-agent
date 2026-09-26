"""Document selection and population/split allocation for enterprise
corpus v2.

Selection is driven only by representativeness and security coverage
(document-structure strata), never by any economic quantity and never by
any model outcome. The whole draw is a deterministic function of
(SEED, train-split profiles, prior-document exclusion set).

Steps
-----
1. `discover_prior_docids`: every train docid referenced (by content or
   by file/directory name) anywhere in the frozen pilot history --
   results/* (except this task's own output roots), data/scenarios,
   configs, demo. `results/dataset_inspection/clean_candidate_scores.json`
   is the one file NOT treated as "used": it is a 744-row ranked
   *inspection* list of candidates, not a corpus, pilot, or scenario. The
   documents that inspection actually rendered/reviewed are still excluded
   through their image directories and review reports.
2. Eligibility (payable document type, usable vendor identity, cluster
   id), then exclusion of every prior docid.
3. Global draw of exactly 600 documents with ONE document per cluster and
   ONE document per normalized vendor key across all three populations:
   quota phases in seeded-hash order -- rare payment-destination fields
   (capped), 10+ line items, multi-page -- then a representative fill in
   the same seeded order. Benchmark-eligible documents (cluster and
   vendor also disjoint from all 92 prior documents) are tracked so the
   benchmark can be drawn from them.
4. Allocation: benchmark population first by systematic stratified
   sampling over benchmark-eligible documents, then probe/LoRA splits by
   stratified largest-remainder interleaving over the rest. Allocation
   uses only document-structure strata and the seeded hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")

SEED = 20260923

PRIOR_SCAN_ROOTS = ("results", "data/scenarios", "configs", "demo")
TASK_OUTPUT_ROOTS = ("results/enterprise_corpus", "results/enterprise_benchmark", "results/economics", "configs/enterprise_corpus")
INSPECTION_ONLY_FILES = ("results/dataset_inspection/clean_candidate_scores.json",)
_TEXT_SUFFIXES = (".json", ".jsonl", ".md", ".txt", ".py", ".csv")
_DOCID_RE = re.compile(r"[0-9a-f]{24}")

N_TOTAL = 600
POPULATION_SPLITS = {
    "probe_v2": {"train": 150, "validation": 50, "test": 50},
    "lora_v2": {"train": 175, "validation": 40, "test": 35},
    "enterprise_benchmark_v1": {"benchmark": 100},
}

TARGET_PAYMENT_DESTINATION_CAP = 30
TARGET_LINE_ITEMS_10_PLUS = 132  # 22% of 600; required floor is 20%
TARGET_MULTI_PAGE = 156  # 26% of 600; required floor is 20%


def rank_key(docid: str, salt: str = "") -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{docid}".encode()).hexdigest()


def _is_task_output(rel: str) -> bool:
    return any(rel == r or rel.startswith(r + "/") for r in TASK_OUTPUT_ROOTS)


def discover_prior_docids(train_docids: set[str]) -> dict[str, list[str]]:
    """docid -> sorted list of repository paths that reference it."""
    refs: dict[str, set[str]] = {}
    for root in PRIOR_SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            rel = str(p.relative_to(REPO_ROOT))
            if _is_task_output(rel) or rel in INSPECTION_ONLY_FILES:
                continue
            found = set(_DOCID_RE.findall(p.name)) & train_docids
            if p.is_file() and p.suffix in _TEXT_SUFFIXES:
                found |= set(_DOCID_RE.findall(p.read_text(errors="ignore"))) & train_docids
            for d in found:
                refs.setdefault(d, set()).add(rel)
    return {d: sorted(v) for d, v in sorted(refs.items())}


def _accept(p: dict, used_clusters: set, used_vendors: set) -> bool:
    return p["cluster_id"] not in used_clusters and p["vendor_key"] not in used_vendors


def select_global(pool: list[dict]) -> tuple[list[dict], dict]:
    """pool: eligible, prior-excluded profiles. Returns (selected, trace)."""
    ordered = sorted(pool, key=lambda p: rank_key(p["docid"], "global"))
    selected: list[dict] = []
    used_clusters: set = set()
    used_vendors: set = set()
    trace = {"phases": []}

    def take(pred, limit_fn, name):
        n0 = len(selected)
        for p in ordered:
            if limit_fn():
                break
            if pred(p) and _accept(p, used_clusters, used_vendors) and p not in selected:
                selected.append(p)
                used_clusters.add(p["cluster_id"])
                used_vendors.add(p["vendor_key"])
        trace["phases"].append({"phase": name, "added": len(selected) - n0, "total_after": len(selected)})

    take(lambda p: p["payment_destination_field_present"],
         lambda: sum(q["payment_destination_field_present"] for q in selected) >= TARGET_PAYMENT_DESTINATION_CAP,
         "rare_payment_destination_fields")
    take(lambda p: p["line_item_count"] >= 10,
         lambda: sum(q["line_item_count"] >= 10 for q in selected) >= TARGET_LINE_ITEMS_10_PLUS,
         "line_items_10_plus")
    take(lambda p: p["page_count"] > 1,
         lambda: sum(q["page_count"] > 1 for q in selected) >= TARGET_MULTI_PAGE,
         "multi_page")
    # Representative fill: keep the payable-complete share at the eligible
    # pool's natural rate (no quota phase above may skew it).
    natural_complete = sum(p["payable_complete"] for p in pool) / len(pool)
    trace["natural_payable_complete_rate"] = round(natural_complete, 4)
    n0 = len(selected)
    queues = {True: [p for p in ordered if p["payable_complete"]], False: [p for p in ordered if not p["payable_complete"]]}
    while len(selected) < N_TOTAL and (queues[True] or queues[False]):
        want = (sum(q["payable_complete"] for q in selected) / max(1, len(selected))) < natural_complete
        q = queues[want] or queues[not want]
        p = q.pop(0)
        if p not in selected and _accept(p, used_clusters, used_vendors):
            selected.append(p)
            used_clusters.add(p["cluster_id"])
            used_vendors.add(p["vendor_key"])
    trace["phases"].append({"phase": "representative_fill", "added": len(selected) - n0, "total_after": len(selected)})
    if len(selected) != N_TOTAL:
        raise RuntimeError(f"STOP: could only draw {len(selected)} vendor/cluster-isolated documents (need {N_TOTAL}).")
    return selected, trace


def stratum(p: dict) -> tuple:
    return (p["page_class"], p["line_item_band"], p["payable_complete"], p["ocr_quality_band"], p["amount_band"], p["field_completeness_band"])


def allocate(selected: list[dict], benchmark_ok: set[str]) -> dict[str, dict[str, list[dict]]]:
    """Returns population -> split -> documents."""
    # Benchmark: systematic stratified sample over benchmark-eligible docs.
    bench_pool = sorted((p for p in selected if p["docid"] in benchmark_ok), key=lambda p: (stratum(p), rank_key(p["docid"], "alloc")))
    n_b = POPULATION_SPLITS["enterprise_benchmark_v1"]["benchmark"]
    if len(bench_pool) < n_b:
        raise RuntimeError(f"STOP: only {len(bench_pool)} benchmark-eligible documents (need {n_b}).")
    step = len(bench_pool) / n_b
    bench = [bench_pool[int((i + 0.5) * step)] for i in range(n_b)]
    bench_ids = {p["docid"] for p in bench}

    rest = sorted((p for p in selected if p["docid"] not in bench_ids), key=lambda p: (stratum(p), rank_key(p["docid"], "alloc")))
    buckets = [(pop, split, n) for pop in ("probe_v2", "lora_v2") for split, n in POPULATION_SPLITS[pop].items()]
    total = sum(n for _, _, n in buckets)
    assert total == len(rest), (total, len(rest))
    assigned = {(pop, split): [] for pop, split, _ in buckets}
    for j, p in enumerate(rest):
        # largest-remainder interleave: bucket furthest behind its quota
        best = max(buckets, key=lambda b: (b[2] * (j + 1) / total - len(assigned[(b[0], b[1])]), b[2]))
        assigned[(best[0], best[1])].append(p)
    out = {"enterprise_benchmark_v1": {"benchmark": sorted(bench, key=lambda p: p["docid"])}}
    for pop, split, n in buckets:
        docs = assigned[(pop, split)]
        assert len(docs) == n, (pop, split, len(docs), n)
        out.setdefault(pop, {})[split] = sorted(docs, key=lambda p: p["docid"])
    return out


def selection_fingerprint(allocation: dict) -> str:
    flat = sorted((pop, split, p["docid"]) for pop, s in allocation.items() for split, docs in s.items() for p in docs)
    return hashlib.sha256(json.dumps(flat).encode()).hexdigest()


__all__ = ["SEED", "N_TOTAL", "POPULATION_SPLITS", "discover_prior_docids", "select_global", "allocate", "stratum", "rank_key", "selection_fingerprint"]
