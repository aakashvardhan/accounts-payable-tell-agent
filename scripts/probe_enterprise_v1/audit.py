"""Dataset audit + grouped-leakage check for the frozen probe_v2_2 corpora (read-only)."""
import collections as C
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/enterprise_corpus/v2_2"
OUT = ROOT / "results/probe_training/enterprise_v1"
SETS = ["probe_v2_2", "probe_v2_2_lexical_challenge", "probe_v2_2_delayed_memory_ood", "probe_v2_2_initial_calibration"]
GROUP_KEYS = ["docid", "vendor_group_key", "cluster_id", "pair_id", "template_family_id", "v2_1_sample_id"]


def partition(rec):
    pop = rec["population"]
    if pop == "probe_v2_2":
        return rec["split"]
    return {"probe_v2_2_lexical_challenge": "lexical_challenge",
            "probe_v2_2_delayed_memory_ood": "delayed_memory_ood",
            "probe_v2_2_initial_calibration": f"calibration_{rec['split']}"}[pop]


def load():
    rows = []
    for s in SETS:
        labels = [json.loads(l) for l in open(BASE / f"{s}_labels.jsonl")]
        inputs = [json.loads(l) for l in open(BASE / f"{s}_inputs.jsonl")]
        assert len(labels) == len(inputs), s
        for lab in labels:
            lab["partition"] = partition(lab)
            rows.append(lab)
    return rows


def main():
    rows = load()
    ids = [r["sample_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate sample_id"
    parts = C.defaultdict(list)
    for r in rows:
        parts[r["partition"]].append(r)

    audit = {"corpus_version": "probe_v2_2", "audit_version": "enterprise_probe_audit_v1", "partitions": {}}
    for p, rs in sorted(parts.items()):
        cls = C.Counter(r["class"] for r in rs)
        docs_by_class = {c: len({r["docid"] for r in rs if r["class"] == c}) for c in cls}
        audit["partitions"][p] = {
            "items": len(rs),
            "by_class": dict(cls),
            "unique_documents": len({r["docid"] for r in rs}),
            "unique_documents_by_class": docs_by_class,
            "unique_vendors": len({r.get("vendor_group_key") for r in rs}),
            "unique_clusters": len({r.get("cluster_id") for r in rs}),
            "unique_pairs": len({r.get("pair_id") for r in rs if r.get("pair_id")}),
            "unique_template_families": len({r.get("template_family_id") for r in rs if r.get("template_family_id")}),
            "by_attack_surface": dict(C.Counter(str(r.get("attack_surface")) for r in rs)),
            "by_attack_family": dict(C.Counter(str(r.get("attack_family")) for r in rs)),
            "by_decision_point": dict(C.Counter(str(r.get("decision_point")) for r in rs)),
            "by_carrier": dict(C.Counter(str(r.get("carrier")) for r in rs)),
            "by_clean_control_role": dict(C.Counter(str(r.get("clean_control_role")) for r in rs)),
            "items_per_document": dict(C.Counter(C.Counter(r["docid"] for r in rs).values())),
        }

    # Grouped leakage: every group key value must map to exactly one partition
    # (calibration_* partitions share nothing with probe sets by construction; checked too).
    leakage = {}
    for k in GROUP_KEYS:
        where = C.defaultdict(set)
        for r in rows:
            v = r.get(k)
            if v is not None:
                where[v].add(r["partition"])
        crossing = {str(v): sorted(ps) for v, ps in where.items() if len(ps) > 1}
        # template families are per-partition wording pools; a shared name would be leakage too
        leakage[k] = {"n_values": len(where), "n_crossing_partitions": len(crossing), "examples": dict(list(crossing.items())[:10])}
    # counterpart links must stay within a partition
    by_id = {r["sample_id"]: r for r in rows}
    cp_cross = [r["sample_id"] for r in rows if r.get("counterpart_sample_id") and r["counterpart_sample_id"] in by_id
                and by_id[r["counterpart_sample_id"]]["partition"] != r["partition"]]
    cp_missing = [r["sample_id"] for r in rows if r.get("counterpart_sample_id") and r["counterpart_sample_id"] not in by_id]
    leakage["counterpart_sample_id"] = {"n_crossing_partitions": len(cp_cross), "n_dangling": len(cp_missing)}
    delayed_in_fit = [r["sample_id"] for r in rows if r["partition"] in ("train", "validation")
                      and (r.get("attack_surface") == "delayed_memory_poisoning")]
    leakage["delayed_memory_in_train_or_validation"] = len(delayed_in_fit)
    audit["leakage"] = leakage

    tr = parts["train"]
    audit["training_groups_by_class"] = {c: len({r["docid"] for r in tr if r["class"] == c}) for c in ("clean", "attacked")}
    audit["totals"] = {
        "records_all_sets": len(rows),
        "unique_documents_all_sets": len({r["docid"] for r in rows}),
        "unique_vendors_all_sets": len({r.get("vendor_group_key") for r in rows}),
        "capture_points": len(rows),
        "exposure_positive": sum(r["exposure_label"] == 1 for r in rows),
        "exposure_negative": sum(r["exposure_label"] == 0 for r in rows),
    }
    audit["memory_form_categories_present"] = sorted({str(r.get("attack_family")) for r in rows if "memory" in str(r.get("attack_surface")) or r["partition"] == "delayed_memory_ood"})
    audit["unit_note"] = ("One record = one decision-point context = one capture point = one activation vector per layer. "
                          "Records are NOT independent: each document contributes several records (both classes, several decision points). "
                          "The independent unit is the document (docid == cluster_id == vendor within probe_v2_2).")

    # Held-out partitions intentionally reuse the 50 test documents (and calibration reuses
    # val/test docs); leakage is a group value spanning different sides (fit/select/evaluate).
    side = lambda p: "fit" if p == "train" else "select" if p in ("validation", "calibration_validation") else "evaluate"
    for k in GROUP_KEYS:
        where = C.defaultdict(set)
        for r in rows:
            if r.get(k) is not None:
                where[r[k]].add(side(r["partition"]))
        leakage[k]["n_crossing_sides"] = sum(len(s) > 1 for s in where.values())
    doc_parts = C.defaultdict(set)
    for r in rows:
        doc_parts[r["docid"]].add(r["partition"])
    shared = C.Counter(tuple(sorted(ps)) for ps in doc_parts.values() if len(ps) > 1)
    leakage["docid_partition_sharing_pattern"] = {"|".join(k): v for k, v in shared.items()}
    hard_fail = [k for k in GROUP_KEYS if leakage[k]["n_crossing_sides"]] + (["counterpart"] if cp_cross else []) + (["delayed_in_fit"] if delayed_in_fit else [])
    min_groups = min(audit["training_groups_by_class"].values())
    audit["verdict"] = {"leakage_failures": hard_fail, "min_training_groups_per_class": min_groups,
                        "sufficient_groups_(>=100)": min_groups >= 100, "proceed": not hard_fail and min_groups >= 100}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_audit_v1.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"verdict": audit["verdict"], "leakage": {k: v if isinstance(v, int) else {kk: vv for kk, vv in v.items() if kk != 'examples'} for k, v in leakage.items()},
                      "training_groups_by_class": audit["training_groups_by_class"], "totals": audit["totals"],
                      "partitions": {p: {k: v for k, v in d.items() if k in ("items", "by_class", "unique_documents", "unique_vendors", "unique_template_families", "items_per_document")} for p, d in audit["partitions"].items()}}, indent=1))
    sys.exit(0 if audit["verdict"]["proceed"] else 2)


if __name__ == "__main__":
    main()
