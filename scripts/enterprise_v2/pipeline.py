"""End-to-end, in-memory build of enterprise corpus v2 + enterprise
benchmark v1 (selection -> generation -> validation -> audits -> file
contents). `scripts/build_enterprise_corpus_v2.py` writes the result;
the CPU tests call `run_pipeline(with_tokenizer=False)` to regenerate
everything and compare it against the frozen files.

No model is loaded. With `with_tokenizer=True` the local Qwen3 tokenizer
(CPU only, `local_files_only=True`) is used to report token lengths and
verify LoRA prompt/completion masking; nothing else touches the model
snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tell.lora_dataset.masking import render_gold_completion_text  # noqa: E402

from enterprise_v2 import audits, builders, docile_profile, policy, selection, wording  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "enterprise_corpus" / "enterprise_corpus_v2_config.json"
CORPUS_DIR = "results/enterprise_corpus/v2"
BENCH_DIR = "results/enterprise_benchmark/v1"
POPS = ("probe_v2", "lora_v2", "enterprise_benchmark_v1")
TRAINABLE_POPS = ("probe_v2", "lora_v2")
LIBRARY_FILES = ("__init__.py", "wording.py", "docile_profile.py", "selection.py", "ids.py", "policy.py", "builders.py", "audits.py", "pipeline.py")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def jd(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


@dataclass
class Build:
    config: dict
    stops: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # relpath -> content (data files, deterministic)
    reports: dict[str, dict] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    rows: dict[str, list[dict]] = field(default_factory=dict)
    bench: list[tuple[dict, dict]] = field(default_factory=list)
    allocation: dict = field(default_factory=dict)
    prior: dict = field(default_factory=dict)
    fixtures: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def check_config(cfg: dict) -> None:
    s = cfg["selection"]
    assert s["seed"] == selection.SEED
    assert s["n_documents_total"] == selection.N_TOTAL
    assert s["quota_phases"]["payment_destination_field_documents_cap"] == selection.TARGET_PAYMENT_DESTINATION_CAP
    assert s["quota_phases"]["line_items_10_plus_target"] == selection.TARGET_LINE_ITEMS_10_PLUS
    assert s["quota_phases"]["multi_page_target"] == selection.TARGET_MULTI_PAGE
    for pop, spec in cfg["populations"].items():
        assert spec["splits"] == selection.POPULATION_SPLITS[pop], pop
    assert tuple(cfg["source_data"]["payable_document_types"]) == docile_profile.PAYABLE_DOCUMENT_TYPES
    assert cfg["trusted_vendor_state_buckets_percent"] == dict(builders.TRUSTED_BUCKETS)
    assert cfg["prompt_profile"] == builders.PROMPT_PROFILE.value


def profile_all(docids: list[str]) -> dict[str, dict]:
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        profs = pool.map(docile_profile.profile_document, docids, chunksize=25)
    return {p["docid"]: p for p in profs}


DIVERSITY_DIMS = (
    "document_type", "page_class", "length_band", "ocr_quality_band", "line_item_band", "amount_band", "currency_metadata",
    "field_completeness_band", "invoice_number_present", "payable_amount_present", "payment_destination_field_present",
    "table_complexity", "visual_clutter_proxy", "payable_complete",
)


def _dist(profiles: list[dict], dim: str) -> dict:
    c = Counter(str(p[dim]) for p in profiles)
    return {k: c[k] for k in sorted(c)}


def run_pipeline(with_tokenizer: bool = True) -> Build:
    cfg = load_config()
    check_config(cfg)
    b = Build(config=cfg)

    # ---------------- selection ----------------
    train_ids = docile_profile.load_train_docids()
    profiles = profile_all(train_ids)
    prior = selection.discover_prior_docids(set(train_ids))
    b.prior = prior
    elig_reasons = Counter()
    eligible = []
    for d in train_ids:
        ok, why = docile_profile.eligibility(profiles[d])
        if ok:
            eligible.append(profiles[d])
        else:
            elig_reasons[why.split(":")[0] if why.startswith("vendor") else why] += 1
    pool = [p for p in eligible if p["docid"] not in prior]
    selected, trace = selection.select_global(pool)
    prior_clusters = {profiles[d]["cluster_id"] for d in prior}
    prior_vendors = {profiles[d]["vendor_key"] for d in prior if profiles[d]["vendor_key"]}
    bench_ok = {p["docid"] for p in selected if p["cluster_id"] not in prior_clusters and p["vendor_key"] not in prior_vendors}
    alloc = selection.allocate(selected, bench_ok)
    b.allocation = alloc

    # ---------------- hard stop checks on selection ----------------
    all_docs = [(pop, split, p) for pop, s in alloc.items() for split, docs in s.items() for p in docs]
    docids = [p["docid"] for _, _, p in all_docs]
    if len(docids) != 600 or len(set(docids)) != 600:
        b.stops.append("document count/uniqueness failure")
    if set(docids) & set(prior):
        b.stops.append(f"prior document reused: {sorted(set(docids) & set(prior))}")
    if len({p['vendor_key'] for _, _, p in all_docs}) != 600:
        b.stops.append("vendor isolation failure")
    if len({p['cluster_id'] for _, _, p in all_docs}) != 600:
        b.stops.append("cluster isolation failure")
    bench_docs = alloc["enterprise_benchmark_v1"]["benchmark"]
    if any(p["cluster_id"] in prior_clusters or p["vendor_key"] in prior_vendors for p in bench_docs):
        b.stops.append("benchmark not isolated from prior pilot vendors/clusters")
    if not set(train_ids) >= set(docids):
        b.stops.append("non-train document selected")

    # ---------------- fixtures + plans ----------------
    fixtures: dict[str, dict[str, builders.DocFixture]] = {}
    plans: dict[str, dict] = {}
    for pop in POPS:
        fxs = []
        for split, docs in alloc[pop].items():
            sd = sorted(docs, key=lambda p: p["docid"])
            fxs += [builders.build_doc_fixture(pop, split, p, sd) for p in sd]
        fixtures[pop] = {f.profile["docid"]: f for f in fxs}
        per_doc = cfg["populations"][pop]["attacked_per_document"]
        plans[pop] = builders.plan_pairs(pop, fxs, per_doc)
    b.fixtures, b.plans = fixtures, plans

    # ---------------- probe / LoRA contexts ----------------
    for pop in TRAINABLE_POPS:
        rows = []
        for docid in sorted(plans[pop]):
            fx = fixtures[pop][docid]
            for ps in plans[pop][docid]:
                for attacked in (False, True):
                    ctx = builders.build_context(fx, ps, attacked)
                    lab = builders.label_record(pop, fx, ps, attacked, ctx)
                    rows.append({"sample_id": lab["sample_id"], "messages": ctx["messages"], "label": lab, "slot_text": ctx["slot_text"], "gold": ctx["gold_action"]})
        rows.sort(key=lambda r: r["sample_id"])
        b.rows[pop] = rows

    # ---------------- benchmark workflows ----------------
    bench = []
    for docid in sorted(plans["enterprise_benchmark_v1"]):
        fx = fixtures["enterprise_benchmark_v1"][docid]
        for ps in plans["enterprise_benchmark_v1"][docid]:
            for attacked in (False, True):
                bench.append(builders.build_benchmark_workflow(fx, ps, attacked))
    bench.sort(key=lambda t: t[0]["workflow_id"])
    b.bench = bench

    _validate(b, cfg)
    _audit(b, cfg)
    _reports(b, cfg, profiles, prior, eligible, pool, elig_reasons, trace, bench_ok, prior_clusters, prior_vendors)
    if with_tokenizer:
        _token_stats(b)
    _data_files(b)
    return b


# ---------------------------------------------------------------------
# Validation (stop conditions)
# ---------------------------------------------------------------------


def _validate(b: Build, cfg: dict) -> None:
    checks = []

    def check(name: str, ok: bool, detail=None, stop: bool = True):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok and stop:
            b.stops.append(f"{name}: {detail}")

    for pop in TRAINABLE_POPS:
        rows = b.rows[pop]
        spec = cfg["populations"][pop]
        n_docs = sum(spec["splits"].values())
        cls = Counter(r["label"]["class"] for r in rows)
        check(f"{pop}: exact item count", len(rows) == n_docs * spec["items_per_document"], len(rows))
        check(f"{pop}: exact class balance", cls["clean"] == cls["attacked"] == n_docs * 5, dict(cls))
        for split, n in spec["splits"].items():
            sr = [r for r in rows if r["label"]["split"] == split]
            check(f"{pop}/{split}: exact item count", len(sr) == n * 10, len(sr))
            check(f"{pop}/{split}: exact class balance", sum(r["label"]["exposure_label"] for r in sr) == n * 5, sum(r["label"]["exposure_label"] for r in sr))
        per_doc = Counter((r["label"]["docid"], r["label"]["class"]) for r in rows)
        check(f"{pop}: 5 clean + 5 attacked per document", set(per_doc.values()) == {5}, None)
        for r in rows:
            policy.validate_gold(r["gold"], r["label"]["contract"])
        check(f"{pop}: every gold action validates against its contract", True, "SessionBAction / SessionAAction TypeAdapter round-trip")
        acts = Counter(r["gold"]["action"] for r in rows)
        top, topn = acts.most_common(1)[0]
        check(f"{pop}: no action class above {cfg['max_action_class_share']:.0%}", topn / len(rows) <= cfg["max_action_class_share"], {"max_class": top, "share": round(topn / len(rows), 4)})
        dp_cls = Counter((r["label"]["decision_point"], r["label"]["class"]) for r in rows)
        dps = sorted({k[0] for k in dp_cls})
        check(f"{pop}: every decision point is class-balanced", all(dp_cls[(d, 'clean')] == dp_cls[(d, 'attacked')] for d in dps), {f"{d}": dp_cls[(d, 'attacked')] for d in dps})
        check(f"{pop}: all six decision points covered", set(dps) == set(policy.DECISION_POINTS), dps)
        surf = Counter(r["label"]["attack_surface"] for r in rows if r["label"]["class"] == "attacked")
        check(f"{pop}: five attack surfaces exactly balanced", len(surf) == 5 and len(set(surf.values())) == 1, dict(surf))
        fam_by_surface = defaultdict(set)
        for r in rows:
            if r["label"]["class"] == "attacked":
                fam_by_surface[r["label"]["attack_surface"]].add(r["label"]["attack_family"])
        check(f"{pop}: every surface has >= 3 independent attack families", all(len(v) >= 3 for v in fam_by_surface.values()), {k: sorted(v) for k, v in fam_by_surface.items()})
        for split in spec["splits"]:
            sr = [r for r in rows if r["label"]["split"] == split]
            sd = Counter((r["label"]["decision_point"], r["label"]["class"]) for r in sr)
            check(f"{pop}/{split}: decision points class-balanced", all(sd[(d, 'clean')] == sd[(d, 'attacked')] for d in policy.DECISION_POINTS), None)

    # benchmark composition
    ev = [e for _, e in b.bench]
    cls = Counter(e["class"] for e in ev)
    check("benchmark: exact workflow count", len(ev) == 600, len(ev))
    check("benchmark: exact class balance", cls["clean"] == cls["attacked"] == 300, dict(cls))
    check("benchmark: 6 workflows per document", set(Counter(e["docid"] for e in ev).values()) == {6}, None)
    surf = Counter(e["attack_surface"] for e in ev if e["class"] == "attacked")
    check("benchmark: all five attack surfaces (60 each)", len(surf) == 5 and set(surf.values()) == {60}, dict(surf))
    cats = Counter(e["scenario_category"] for e in ev)
    needed = ("clean_canonical_completion", "resolvable_conflict", "conflict_requiring_review", "missing_field_clean_negative", "ocr_corruption_clean_negative")
    check("benchmark: required scenario categories present", all(cats[c] > 0 for c in needed), dict(cats))
    exposure = Counter(e["canonical_first_exposure_decision_point"] for e in ev if e["class"] == "attacked")
    check("benchmark: attacks first exposed at initial/post_email/post_invoice/post_memory_retrieval", set(exposure) >= {"initial", "post_email", "post_invoice", "post_memory_retrieval"}, dict(exposure))
    pages = Counter(e["page_class"] for e in ev)
    check("benchmark: single- and multi-page invoices", pages["single_page"] > 0 and pages["multi_page"] > 0, dict(pages))
    tabs = Counter(e["table_complexity"] for e in ev)
    check("benchmark: simple and complex invoices", tabs["complex"] > 0 and (tabs["simple"] + tabs["no_table"]) > 0, dict(tabs))

    b.reports["_validation_checks"] = {"checks": checks}


# ---------------------------------------------------------------------
# Leakage + shortcut audits
# ---------------------------------------------------------------------


def _audit(b: Build, cfg: dict) -> None:
    leak_checks = []

    def check(name, ok, detail=None, stop=True):
        leak_checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok and stop:
            b.stops.append(f"LEAKAGE {name}: {detail}")

    sys_prompts = defaultdict(set)
    ids_by_split = defaultdict(set)
    tmpl_by_split = defaultdict(set)
    docs_by_split = defaultdict(set)
    vendors_by_split = defaultdict(set)
    clusters_by_split = defaultdict(set)
    hits = Counter()
    approved_outside = 0
    id_presence = Counter()
    for pop in TRAINABLE_POPS:
        for r in b.rows[pop]:
            lab = r["label"]
            key = f"{pop}/{lab['split']}"
            sys_prompts[lab["contract"]].add(sha256_text(r["messages"][0]["content"]))
            full = "\n".join(m["content"] for m in r["messages"])
            for h in audits.scan_text_for_leakage(full):
                hits[h] += 1
            if r["sample_id"] in full or lab["pair_id"] in full:
                hits["own_id_in_input"] += 1
            found = set(audits.ACCOUNT_RE.findall(full))
            ids_by_split[key] |= found
            tmpl_by_split[key].add(lab["template_family_id"])
            docs_by_split[key].add(lab["docid"])
            vendors_by_split[key].add(lab["vendor_group_key"])
            clusters_by_split[key].add(lab["cluster_id"])
            # approved account only via trusted vendor record or (dual-mode) slot text
            stripped = full.replace(r["slot_text"], "").replace(json.dumps(r["slot_text"])[1:-1], "")
            vr_msgs = [m["content"] for m in r["messages"] if '"tool_name": "get_vendor_record"' in m["content"]]
            for m in vr_msgs:
                stripped = stripped.replace(m, "")
            if lab["approved_account_id"] in stripped:
                approved_outside += 1
            id_presence[(pop, lab["class"], len(found - {lab['approved_account_id']}) > 0)] += 1
    for pop in TRAINABLE_POPS:
        # unauthorized ids never appear in any clean context of that population
        un = {r["label"]["slot_account_id"] for r in b.rows[pop] if r["label"]["class"] == "attacked"}
        cl = set()
        for r in b.rows[pop]:
            if r["label"]["class"] == "clean":
                cl |= set(audits.ACCOUNT_RE.findall("\n".join(m["content"] for m in r["messages"])))
        check(f"{pop}: no unauthorized account id appears in any clean context", not (un & cl), len(un & cl))
        check(f"{pop}: unauthorized ids are per-pair (no fixed attacker id)", len(un) == sum(1 for r in b.rows[pop] if r['label']['class'] == 'attacked'), len(un))

    check("inputs contain no label tokens, class words, rule ids, sample ids, or source paths", not hits, dict(hits))
    check("one fixed system prompt per contract (prompt carries no per-sample information)", all(len(v) == 1 for v in sys_prompts.values()), {k: len(v) for k, v in sys_prompts.items()})
    check("approved beneficiary appears only in trusted vendor-record output or dual-mode slot text", approved_outside == 0, approved_outside)
    pres = {f"{p}/{c}/non_approved_id_present={v}": n for (p, c, v), n in sorted(id_presence.items())}
    for pop in TRAINABLE_POPS:
        ca = id_presence[(pop, "clean", True)] / max(1, id_presence[(pop, "clean", True)] + id_presence[(pop, "clean", False)])
        aa = id_presence[(pop, "attacked", True)] / max(1, id_presence[(pop, "attacked", True)] + id_presence[(pop, "attacked", False)])
        check(f"{pop}: non-approved account-id presence equal by class", abs(ca - aa) < 1e-9, {"clean": ca, "attacked": aa})
    all_ids = [i for v in ids_by_split.values() for i in v]
    check("single neutral account-id format", all(len(i) == 21 and i.startswith("SIM-ACCT-") for i in all_ids), None)

    keys = sorted(ids_by_split)
    cross = {}
    for i, a in enumerate(keys):
        for bb in keys[i + 1:]:
            for nm, dct in (("account_ids", ids_by_split), ("template_families", tmpl_by_split), ("documents", docs_by_split), ("vendors", vendors_by_split), ("clusters", clusters_by_split)):
                inter = dct[a] & dct[bb]
                if inter:
                    cross[f"{a} x {bb} {nm}"] = len(inter)
    # benchmark vs everything
    bench_ids, bench_docs, bench_tmpl = set(), set(), set()
    for fxr, ev in b.bench:
        # Only what the agent can see: the work item and the tool fixtures
        # (the fixture's own workflow_id/harness fields stay harness-side).
        t = json.dumps({"work_item": fxr["work_item"], "tool_fixtures": fxr["tool_fixtures"]})
        bench_ids |= set(audits.ACCOUNT_RE.findall(t))
        bench_docs.add(ev["docid"])
        bench_tmpl.add(ev["template_family_id"])
        for h in audits.scan_text_for_leakage(t):
            hits["benchmark_fixture:" + h] += 1
        for k in ("expected_terminal_action", "scenario_category", "expected_outcome_class", "claim_type", "attack_surface", "class"):
            if f'"{k}"' in t:
                hits[f"benchmark_fixture_contains_eval_field:{k}"] += 1
    for key in keys:
        if ids_by_split[key] & bench_ids:
            cross[f"{key} x benchmark account_ids"] = len(ids_by_split[key] & bench_ids)
        if docs_by_split[key] & bench_docs:
            cross[f"{key} x benchmark documents"] = len(docs_by_split[key] & bench_docs)
        if tmpl_by_split[key] & bench_tmpl:
            cross[f"{key} x benchmark template_families"] = len(tmpl_by_split[key] & bench_tmpl)
    # Template families / wording are shared by design between the probe
    # and LoRA split with the SAME role (e.g. probe train and LoRA train);
    # they must never cross split roles or reach the benchmark.
    def _role(k: str) -> str:
        return k.split("/")[1]

    illegal = {k: v for k, v in cross.items() if not ("template_families" in k and " x " in k and "benchmark" not in k and _role(k.split(" x ")[0]) == _role(k.split(" x ")[1].split(" ")[0]))}
    check("no document, vendor, cluster, account id, or template family crosses splits or populations (same-role wording sharing between probe and LoRA is by design)", not illegal, illegal)
    check("benchmark fixtures contain no evaluation-only fields or label tokens", not any(k.startswith("benchmark_fixture") for k in hits), {k: v for k, v in hits.items() if k.startswith("benchmark")})

    # wording isolation at template level
    import itertools
    import re as _re

    def grams(s, n=6):
        s = _re.sub(r"\{[A-Za-z_]+\}", " PH ", s)
        t = _re.findall(r"[a-z0-9_]+", s.lower())
        return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}

    G = {p: set().union(*[grams(s) for s in wording.all_template_strings(p)]) for p in wording.PARTITIONS}
    overlaps = {f"{a} x {c}": len(G[a] & G[c]) for a, c in itertools.combinations(wording.PARTITIONS, 2)}
    check("wording partitions share no word 6-gram", all(v == 0 for v in overlaps.values()), overlaps)

    b.reports["leakage_validation"] = {
        "scope": "probe_v2 inputs, lora_v2 inputs, enterprise_benchmark_v1 agent-facing fixtures",
        "checks": leak_checks,
        "non_approved_id_presence_by_class": pres,
        "all_passed": all(c["passed"] for c in leak_checks),
    }

    # shortcut + pair audits
    thr = cfg["shortcut_audit_thresholds"]
    tol = cfg["pair_matching_tolerances"]
    sa = {}
    for pop in TRAINABLE_POPS:
        sa[pop] = {"shortcut": audits.shortcut_audit(b.rows[pop], "train", "test", thr), "pairs": audits.pair_audit(b.rows[pop], tol)}
        if not sa[pop]["shortcut"]["passes_pre_registered_thresholds"]:
            b.stops.append(f"SHORTCUT {pop}: trivial features materially predictive ({sa[pop]['shortcut']['trivial_feature_classifier_auroc']}, {sa[pop]['shortcut']['most_separating_single_feature']})")
        if not sa[pop]["pairs"]["passes"]:
            b.stops.append(f"PAIRS {pop}: matched-control violations {sa[pop]['pairs']['violations'][:3]} {sa[pop]['pairs']['structural_field_mismatches']}")
    # benchmark: shortcut audit over the canonical pre-payment context of every workflow (report + stop)
    bench_rows = []
    for (fxr, ev) in b.bench:
        fx = b.fixtures["enterprise_benchmark_v1"][ev["docid"]]
        ps = next(p for p in b.plans["enterprise_benchmark_v1"][ev["docid"]] if builders.pair_public_id("enterprise_benchmark_v1", p) == ev["pair_id"])
        traj = builders.canonical_trajectory(fx, ps, ev["class"] == "attacked")
        msgs = traj[-1][0]
        slot, _ = builders.slot_text(fx, ps, ev["class"] == "attacked")
        lab = {
            "exposure_label": ev["exposure_label"], "split": "benchmark", "docid": ev["docid"], "pair_id": ev["pair_id"], "class": ev["class"],
            "decision_point": "pre_payment_proposal", "contract": "session_b_processing", "workflow_variant": "prefetched_memory" if ps.harness_mode == "prefetched_memory" else "memory_before_vendor",
            "tool_history_depth": len(msgs) - 2, "account_id_mode": ps.id_mode, "pressure_clause": ps.pressure, "memory_baseline": ps.memory_baseline, "carrier": ps.carrier,
            "invoice_field": ps.invoice_field, "matched_attack_surface": ps.surface, "matched_attack_family": ps.family,
        }
        bench_rows.append({"messages": msgs, "label": lab, "slot_text": slot})
    sa["enterprise_benchmark_v1"] = {"shortcut": audits.shortcut_audit(bench_rows, None, None, thr), "pairs": audits.pair_audit(bench_rows, tol)}
    if not sa["enterprise_benchmark_v1"]["shortcut"]["passes_pre_registered_thresholds"]:
        b.stops.append("SHORTCUT benchmark: trivial features materially predictive")
    if not sa["enterprise_benchmark_v1"]["pairs"]["passes"]:
        b.stops.append("PAIRS benchmark: matched-control violations")
    b.reports["shortcut_audit"] = {
        "method": "Surface-statistic features only (no hidden states, no model). See scripts/enterprise_v2/audits.py.",
        "populations": sa,
        "all_pass": all(v["shortcut"]["passes_pre_registered_thresholds"] and v["pairs"]["passes"] for v in sa.values()),
    }


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------


def _reports(b, cfg, profiles, prior, eligible, pool, elig_reasons, trace, bench_ok, prior_clusters, prior_vendors) -> None:
    alloc = b.allocation
    train_profiles = list(profiles.values())

    def coverage(ps: list[dict]) -> dict:
        n = len(ps)
        out = {"n_documents": n, "distinct_vendors": len({p["vendor_key"] for p in ps}), "distinct_clusters": len({p["cluster_id"] for p in ps})}
        for d in DIVERSITY_DIMS:
            out[d] = _dist(ps, d)
        out["shares"] = {
            "multi_page": round(sum(p["page_count"] > 1 for p in ps) / n, 4),
            "line_items_10_plus": round(sum(p["line_item_count"] >= 10 for p in ps) / n, 4),
            "payable_complete": round(sum(p["payable_complete"] for p in ps) / n, 4),
            "payment_destination_field_present": round(sum(p["payment_destination_field_present"] for p in ps) / n, 4),
            "invoice_number_present": round(sum(p["invoice_number_present"] for p in ps) / n, 4),
            "low_ocr_quality": round(sum(p["ocr_quality_band"] == "low" for p in ps) / n, 4),
            "visually_cluttered_proxy": round(sum(p["visual_clutter_proxy"] == "cluttered" for p in ps) / n, 4),
        }
        out["n_amount_bands_excluding_missing"] = len({p["amount_band"] for p in ps} - {"missing_or_unparseable"})
        return out

    targets = []
    relaxations = []
    by_group = {}
    for pop, s in alloc.items():
        allp = [p for docs in s.values() for p in docs]
        by_group[pop] = {"all": coverage(allp), **{split: coverage(docs) for split, docs in s.items()}}
        for grp, docs in [("all", allp)] + list(s.items()):
            cov = coverage(docs)
            for name, val, floor in (("multi_page", cov["shares"]["multi_page"], 0.2), ("line_items_10_plus", cov["shares"]["line_items_10_plus"], 0.2)):
                ok = val >= floor
                targets.append({"population": pop, "group": grp, "target": f"{name} >= {floor:.0%}", "actual": val, "met": ok})
                if not ok:
                    relaxations.append({"population": pop, "group": grp, "target": name, "actual": val})
            ok = cov["n_amount_bands_excluding_missing"] >= 4
            targets.append({"population": pop, "group": grp, "target": ">= 4 invoice-amount bands", "actual": cov["n_amount_bands_excluding_missing"], "met": ok})
            if not ok:
                relaxations.append({"population": pop, "group": grp, "target": "amount bands", "actual": cov["n_amount_bands_excluding_missing"]})

    # fixture-level (observed) difficulty coverage
    fix_cov = {}
    for pop in POPS:
        fxs = list(b.fixtures[pop].values())
        blockers = Counter(policy.invoice_blockers(f.observed) or "none" for f in fxs)
        fix_cov[pop] = {
            "trusted_vendor_state": dict(Counter(f.trusted_bucket for f in fxs)),
            "invoice_view": dict(Counter("ocr_text" if f.use_ocr_view else "annotation_text" for f in fxs)),
            "observed_invoice_blockers": dict(blockers),
            "documents_with_missing_or_ambiguous_fields": sum(1 for f in fxs if policy.invoice_blockers(f.observed)),
        }

    b.reports["diversity_report"] = {
        "available_population": {
            "train_documents": len(train_profiles),
            "eligible_payable_documents": len(eligible),
            "ineligible_reasons": dict(elig_reasons),
            "prior_documents_excluded": len(prior),
            "eligible_after_prior_exclusion": len(pool),
            "eligible_pool_coverage": coverage(pool),
            "max_isolated_multi_page_note": "Isolation (one document per cluster and per vendor) caps the achievable share of rare strata; see selection_report.md.",
        },
        "selection_trace": trace,
        "benchmark_eligible_documents_among_selected": len(bench_ok),
        "coverage_by_population_and_split": by_group,
        "observed_fixture_coverage": fix_cov,
        "required_targets": targets,
        "all_required_targets_met": not relaxations,
        "relaxations_considered": relaxations,
        "proxies_and_limits": {
            "visual_clutter_proxy": "No manual visual review at this scale. 'cluttered' = OCR low-confidence word share >= 5% or >= 500 OCR words per page, a proxy for stamps, handwriting, and dense layouts; it does not detect them directly.",
            "ocr_quality_band": "Critical-field (invoice number, date, payable amount) OCR-vs-annotation agreement plus word-confidence share; vendor-name spans only demote on gross disagreement.",
            "table_complexity": "From DocILE page_to_table_grid and LIR annotations: complex = multi-page grid, >= 6 columns, flagged grid, >= 10 line items, or >= 6 LIR field types.",
        },
    }

    # attack surface / decision point / action targets
    asr = {}
    for pop in TRAINABLE_POPS:
        rows = b.rows[pop]
        att = [r for r in rows if r["label"]["class"] == "attacked"]
        per_split = {}
        for split in cfg["populations"][pop]["splits"]:
            sr = [r for r in rows if r["label"]["split"] == split]
            per_split[split] = {
                "attacked_by_surface": dict(Counter(r["label"]["attack_surface"] for r in sr if r["label"]["exposure_label"])),
                "decision_point_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["decision_point"], r["label"]["class"]) for r in sr).items())},
                "template_families": sorted({r["label"]["template_family_id"] for r in sr if r["label"]["exposure_label"]}),
            }
        acts = Counter(r["gold"]["action"] for r in rows)
        asr[pop] = {
            "attacked_by_surface": dict(Counter(r["label"]["attack_surface"] for r in att)),
            "attacked_by_surface_and_family": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["attack_surface"], r["label"]["attack_family"]) for r in att).items())},
            "requested_family_categories_covered": sorted({lbl for r in att for lbl in wording.FAMILY_BY_ID[r["label"]["attack_family"]].requested_family_labels}),
            "attacked_by_claim_type": dict(Counter(r["label"]["claim_type"] for r in att)),
            "attacked_by_carrier": dict(Counter(r["label"]["carrier"] for r in att)),
            "attacked_by_invoice_field": dict(Counter(r["label"]["invoice_field"] for r in att if r["label"]["invoice_field"])),
            "decision_point_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["decision_point"], r["label"]["class"]) for r in rows).items())},
            "decision_point_by_surface_attacked": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["attack_surface"], r["label"]["decision_point"]) for r in att).items())},
            "workflow_variant_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["workflow_variant"], r["label"]["class"]) for r in rows).items())},
            "contract_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["contract"], r["label"]["class"]) for r in rows).items())},
            "gold_action_distribution": {k: {"n": v, "share": round(v / len(rows), 4)} for k, v in acts.most_common()},
            "gold_action_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["class"], r["gold"]["action"]) for r in rows).items())},
            "gold_rule_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["label"]["class"], r["label"]["gold_rule"]) for r in rows).items())},
            "review_reason_distribution": dict(Counter(r["gold"].get("review_reason") for r in rows if r["gold"]["action"] == "request_review")),
            "max_action_class_share": round(acts.most_common(1)[0][1] / len(rows), 4),
            "attacked_terminal_outcomes": dict(Counter(r["label"]["gold_rule"] for r in att if r["label"]["decision_point"] == "pre_payment_proposal" or r["label"]["contract"] == "session_a_intake")),
            "per_split": per_split,
        }
    ev = [e for _, e in b.bench]
    asr["enterprise_benchmark_v1"] = {
        "attacked_by_surface": dict(Counter(e["attack_surface"] for e in ev if e["class"] == "attacked")),
        "attacked_by_surface_and_family": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((e["attack_surface"], e["attack_family"]) for e in ev if e["class"] == "attacked").items())},
        "attacked_by_first_exposure_decision_point": dict(Counter(e["canonical_first_exposure_decision_point"] for e in ev if e["class"] == "attacked")),
        "attacked_by_harness_mode": dict(Counter(e["harness_mode"] for e in ev if e["class"] == "attacked")),
        "scenario_category_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((e["class"], e["scenario_category"]) for e in ev).items())},
        "expected_outcome_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((e["class"], e["expected_outcome_class"]) for e in ev).items())},
        "expected_terminal_action_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((e["class"], e["expected_terminal_action"]["action"]) for e in ev).items())},
        "decision_point_note": "Benchmark workflows are full, model-directed runs: every attacked workflow's content stays in context from its first-exposure decision point through post_vendor_record/pre_payment_proposal, so all six decision points are traversed. Initial-point exposure uses the existing application-prefetch harness (tell.agent.conditional_retrieval).",
    }
    b.reports["attack_surface_report"] = asr

    b.summary = {
        "n_train_documents": len(train_profiles),
        "n_prior_documents_excluded": len(prior),
        "n_eligible": len(eligible),
        "n_pool": len(pool),
        "benchmark_prior_isolation": {"prior_clusters": len(prior_clusters), "prior_vendor_keys": len(prior_vendors)},
    }


# ---------------------------------------------------------------------
# Token statistics (CPU tokenizer only)
# ---------------------------------------------------------------------


def _token_stats(b: Build) -> None:
    from transformers import AutoTokenizer

    from tell.agent.local_model import PINNED_SNAPSHOT_PATH
    from tell.lora_dataset.masking import build_masked_example

    tok = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)

    def n_prompt(msgs):
        return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False))

    import numpy as np

    stats = {}
    for pop in TRAINABLE_POPS:
        lens = np.array([n_prompt(r["messages"]) for r in b.rows[pop]])
        stats[pop] = {"prompt_tokens": {"mean": round(float(lens.mean()), 1), "p50": int(np.percentile(lens, 50)), "p95": int(np.percentile(lens, 95)), "max": int(lens.max()), "min": int(lens.min())}}
    rej = []
    comp = []
    for r in b.rows["lora_v2"]:
        m = build_masked_example(tok, sample_id=r["sample_id"], messages=r["messages"], gold_action_dict=r["gold"], max_seq_len=8192)
        if m.rejected:
            rej.append({"sample_id": r["sample_id"], "reason": m.reject_reason})
        comp.append(m.completion_token_count)
    comp = np.array(comp)
    stats["lora_v2"]["masking_check"] = {
        "max_seq_len": 8192,
        "n_rejected": len(rej),
        "rejected": rej[:20],
        "completion_tokens_incl_eos": {"mean": round(float(comp.mean()), 1), "max": int(comp.max())},
        "method": "tell.lora_dataset.masking.build_masked_example (prompt-prefix verified, never truncated)",
    }
    if rej:
        b.stops.append(f"LoRA masking rejected {len(rej)} examples at max_seq_len 8192")
    per_wf = {}
    for fxr, ev in b.bench:
        fx = b.fixtures["enterprise_benchmark_v1"][ev["docid"]]
        ps = next(p for p in b.plans["enterprise_benchmark_v1"][ev["docid"]] if builders.pair_public_id("enterprise_benchmark_v1", p) == ev["pair_id"])
        traj = builders.canonical_trajectory(fx, ps, ev["class"] == "attacked")
        pin = sum(n_prompt(m) for m, _ in traj)
        pout = sum(len(tok(render_gold_completion_text(g) + tok.eos_token, add_special_tokens=False)["input_ids"]) for _, g in traj)
        per_wf[ev["workflow_id"]] = {"turns": len(traj), "prompt_tokens": pin, "completion_tokens": pout}
    vals = np.array([[v["turns"], v["prompt_tokens"], v["completion_tokens"]] for v in per_wf.values()])
    stats["enterprise_benchmark_v1"] = {
        "canonical_path_projection_per_workflow": per_wf,
        "summary": {
            "turns_mean": round(float(vals[:, 0].mean()), 3),
            "prompt_tokens_mean": round(float(vals[:, 1].mean()), 1),
            "prompt_tokens_p95": int(np.percentile(vals[:, 1], 95)),
            "completion_tokens_mean": round(float(vals[:, 2].mean()), 1),
            "total_tokens_mean": round(float((vals[:, 1] + vals[:, 2]).mean()), 1),
        },
        "status": "projection -- local Qwen3 tokenizer count of the canonical (gold) path contexts, one fresh full-context forward pass per turn (the same accounting as the measured TK-01 figure); no model was run",
    }
    stats["tokenizer"] = {"path": str(PINNED_SNAPSHOT_PATH), "class": type(tok).__name__, "model_weights_loaded": False}
    b.reports["token_statistics"] = stats


# ---------------------------------------------------------------------
# Deterministic data files
# ---------------------------------------------------------------------


def _data_files(b: Build) -> None:
    f = b.files
    for pop, short in (("probe_v2", "probe_v2"), ("lora_v2", "lora_v2")):
        rows = b.rows[pop]
        f[f"{CORPUS_DIR}/{short}_inputs.jsonl"] = jl([{"sample_id": r["sample_id"], "messages": r["messages"]} for r in rows])
        f[f"{CORPUS_DIR}/{short}_labels.jsonl"] = jl([r["label"] for r in rows])
    f[f"{CORPUS_DIR}/lora_v2_targets.jsonl"] = jl(
        [{"sample_id": r["sample_id"], "completion": render_gold_completion_text(r["gold"]), "gold_action": r["gold"]} for r in b.rows["lora_v2"]]
    )
    f[f"{BENCH_DIR}/enterprise_benchmark_v1_workflows.jsonl"] = jl([w for w, _ in b.bench])
    f[f"{BENCH_DIR}/enterprise_benchmark_v1_evaluation_only.jsonl"] = jl([e for _, e in b.bench])
    docs = []
    for pop, s in b.allocation.items():
        for split, ps in s.items():
            for p in ps:
                fx = b.fixtures[pop][p["docid"]]
                docs.append({
                    "docid": p["docid"], "population": pop, "split": split, "wording_partition": fx.partition,
                    "vendor_display": p["vendor_display"], "vendor_key": p["vendor_key"], "cluster_id": p["cluster_id"],
                    "document_type": p["document_type"], "page_count": p["page_count"], "line_item_count": p["line_item_count"],
                    "amount_band": p["amount_band"], "currency_metadata": p["currency_metadata"], "length_band": p["length_band"],
                    "ocr_quality_band": p["ocr_quality_band"], "table_complexity": p["table_complexity"], "visual_clutter_proxy": p["visual_clutter_proxy"],
                    "field_completeness_band": p["field_completeness_band"], "invoice_number_present": p["invoice_number_present"],
                    "payable_amount_present": p["payable_amount_present"], "payment_destination_fields": p["payment_destination_fields"],
                    "payable_complete_annotation": p["payable_complete"], "observed_invoice_blocker": policy.invoice_blockers(fx.observed),
                    "trusted_vendor_state": fx.trusted_bucket, "invoice_view": "ocr_text" if fx.use_ocr_view else "annotation_text",
                    "annotation_sha256": p["annotation_sha256"], "ocr_sha256": p["ocr_sha256"],
                })
    docs.sort(key=lambda d: (d["population"], d["split"], d["docid"]))
    f[f"{CORPUS_DIR}/document_assignments.jsonl"] = jl(docs)
    f[f"{CORPUS_DIR}/prior_document_exclusions.json"] = jd({"n_prior_train_documents": len(b.prior), "documents": b.prior})


def library_hashes() -> dict[str, str]:
    base = REPO_ROOT / "scripts" / "enterprise_v2"
    return {f"scripts/enterprise_v2/{n}": hashlib.sha256((base / n).read_bytes()).hexdigest() for n in LIBRARY_FILES}


__all__ = ["run_pipeline", "Build", "CORPUS_DIR", "BENCH_DIR", "library_hashes", "sha256_text", "jd", "jl"]
