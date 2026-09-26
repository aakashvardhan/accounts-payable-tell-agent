"""CPU-only tests for enterprise corpus v2 (probe v2 + safety LoRA v2),
enterprise benchmark v1, the enterprise replay manifest, and the
enterprise economics v1 model.

No model, tokenizer, or GPU is used. One in-memory regeneration of the
whole corpus (`run_pipeline(with_tokenizer=False)`, ~15 s) is shared by
the session and compared byte-for-byte against the frozen files.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from enterprise_v2 import audits, docile_profile, economics, pipeline, policy, selection, wording  # noqa: E402

from tell.agent.actions import ActionParseOutcome, parse_session_a_action, parse_session_b_action  # noqa: E402

CORPUS = REPO / "results/enterprise_corpus/v2"
BENCH = REPO / "results/enterprise_benchmark/v1"
ECON = REPO / "results/economics/enterprise_v1"
ACCOUNT_RE = re.compile(r"SIM-ACCT-[0-9A-F]{12}")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def build():
    return pipeline.run_pipeline(with_tokenizer=False)


@pytest.fixture(scope="session")
def docs():
    return _jsonl(CORPUS / "document_assignments.jsonl")


@pytest.fixture(scope="session")
def labels():
    return {"probe_v2": _jsonl(CORPUS / "probe_v2_labels.jsonl"), "lora_v2": _jsonl(CORPUS / "lora_v2_labels.jsonl")}


@pytest.fixture(scope="session")
def inputs():
    return {"probe_v2": _jsonl(CORPUS / "probe_v2_inputs.jsonl"), "lora_v2": _jsonl(CORPUS / "lora_v2_inputs.jsonl")}


@pytest.fixture(scope="session")
def bench():
    return _jsonl(BENCH / "enterprise_benchmark_v1_workflows.jsonl"), _jsonl(BENCH / "enterprise_benchmark_v1_evaluation_only.jsonl")


# ---------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------


def test_deterministic_selection_and_generation(build):
    top = json.loads((CORPUS / "enterprise_corpus_v2_manifest.json").read_text())
    assert selection.selection_fingerprint(build.allocation) == top["selection_fingerprint"]
    for rel, entry in top["data_files"].items():
        assert hashlib.sha256(build.files[rel].encode("utf-8")).hexdigest() == entry["sha256"], rel
    assert build.stops == []


def test_selection_is_a_pure_function_of_seed_and_pool(build):
    pool = [p for d in build.allocation.values() for s in d.values() for p in s]
    a, _ = selection.select_global(sorted(pool, key=lambda p: p["docid"]))
    b, _ = selection.select_global(sorted(pool, key=lambda p: p["docid"], reverse=True))
    assert [p["docid"] for p in a] == [p["docid"] for p in b]


def test_benchmark_immutability():
    man = json.loads((BENCH / "enterprise_benchmark_v1_manifest.json").read_text())
    assert man["frozen"] is True
    for key in ("workflows", "evaluation_only"):
        f = man["files"][key]
        assert _sha(REPO / f["path"]) == f["sha256"]
    assert _sha(REPO / man["files"]["evaluation_protocol"]["path"]) == man["files"]["evaluation_protocol"]["sha256"]
    # a second build must refuse to overwrite the frozen benchmark
    r = subprocess.run([sys.executable, str(REPO / "scripts/build_enterprise_corpus_v2.py")], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "frozen" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------
# Source data, prior exclusion, isolation
# ---------------------------------------------------------------------


def test_only_train_split_documents(docs):
    train = set(json.loads((REPO / "data/docile/train.json").read_text()))
    assert {d["docid"] for d in docs} <= train
    assert docile_profile.TRAIN_LIST_PATH.name == "train.json"
    for mod in ("docile_profile", "selection", "pipeline", "builders"):
        src = (REPO / f"scripts/enterprise_v2/{mod}.py").read_text()
        assert not re.search(r"""["'/](train)?val\.json["']""", src), mod


def test_every_prior_document_excluded(docs):
    train = set(json.loads((REPO / "data/docile/train.json").read_text()))
    prior = selection.discover_prior_docids(train)
    frozen_prior = json.loads((CORPUS / "prior_document_exclusions.json").read_text())["documents"]
    assert set(prior) == set(frozen_prior) and len(prior) == 92
    chosen = {d["docid"] for d in docs}
    assert not chosen & set(prior)
    explicit = set()
    for path in ("results/probe_dataset/document_selection_manifest.json", "results/probe_dataset/v1_1/document_selection_manifest.json", "results/lora_dataset/v1/document_selection_manifest.json"):
        explicit |= {d["docid"] for d in json.loads((REPO / path).read_text())["documents"]}
    explicit |= {"04d531ca811f448a91c6ff4e", "002f9b82b74f4258b3b072d0"}
    assert explicit <= set(prior)
    assert not chosen & explicit


def test_global_vendor_and_cluster_isolation(docs):
    assert len(docs) == 600
    assert len({d["docid"] for d in docs}) == 600
    assert len({d["vendor_key"] for d in docs}) == 600
    assert len({d["cluster_id"] for d in docs}) == 600
    assert len({docile_profile.vendor_key(d["vendor_display"]) for d in docs}) == 600


def test_benchmark_isolated_from_prior_vendors_and_clusters(docs, build):
    prior = json.loads((CORPUS / "prior_document_exclusions.json").read_text())["documents"]
    prof = {p["docid"]: p for s in build.allocation.values() for d in s.values() for p in d}
    prior_profiles = [docile_profile.profile_document(d) for d in prior]
    pc = {p["cluster_id"] for p in prior_profiles}
    pv = {p["vendor_key"] for p in prior_profiles if p["vendor_key"]}
    for d in docs:
        if d["population"] == "enterprise_benchmark_v1":
            assert d["cluster_id"] not in pc and prof[d["docid"]]["vendor_key"] not in pv


def test_split_isolation(labels, inputs):
    for pop in ("probe_v2", "lora_v2"):
        by = defaultdict(lambda: defaultdict(set))
        ids = {r["sample_id"]: r for r in inputs[pop]}
        for lab in labels[pop]:
            s = lab["split"]
            by["doc"][s].add(lab["docid"])
            by["vendor"][s].add(lab["vendor_group_key"])
            by["cluster"][s].add(lab["cluster_id"])
            by["template"][s].add(lab["template_family_id"])
            by["partition"][s].add(lab["wording_partition"])
            by["acct"][s] |= set(ACCOUNT_RE.findall(json.dumps(ids[lab["sample_id"]]["messages"])))
        for kind, sets in by.items():
            for a, b in itertools.combinations(sets, 2):
                assert not sets[a] & sets[b], (pop, kind, a, b)


def test_probe_lora_benchmark_isolation(docs, labels, inputs, bench):
    pops = defaultdict(set)
    for d in docs:
        pops[d["population"]].add((d["docid"], d["vendor_key"], d["cluster_id"]))
    for a, b in itertools.combinations(pops, 2):
        for i in range(3):
            assert not {x[i] for x in pops[a]} & {x[i] for x in pops[b]}
    accts = {pop: set(ACCOUNT_RE.findall("".join(json.dumps(r["messages"]) for r in inputs[pop]))) for pop in inputs}
    accts["bench"] = set(ACCOUNT_RE.findall(json.dumps(bench[0])))
    for a, b in itertools.combinations(accts, 2):
        assert not accts[a] & accts[b]
    bench_tmpl = {e["template_family_id"] for e in bench[1]}
    for pop in labels:
        assert not bench_tmpl & {lab["template_family_id"] for lab in labels[pop]}


def test_wording_partitions_share_no_6gram():
    def grams(s, n=6):
        s = re.sub(r"\{[A-Za-z_]+\}", " PH ", s)
        t = re.findall(r"[a-z0-9_]+", s.lower())
        return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}

    g = {p: set().union(*[grams(s) for s in wording.all_template_strings(p)]) for p in wording.PARTITIONS}
    for a, b in itertools.combinations(wording.PARTITIONS, 2):
        assert not g[a] & g[b], (a, b)


def test_attack_and_mirror_templates_are_keyword_balanced():
    def feats(s):
        low = s.lower()
        f = Counter({k: len(re.findall(rf"\b{k}\b", low)) for k in audits.KEYWORDS})
        for ch in ";:!?-{[":
            f[ch] = s.count(ch)
        return f

    for (fam, part), variants in wording.CORES.items():
        for a, c in variants:
            assert feats(a) == feats(c), (fam, part, a)
    for part in wording.PARTITIONS:
        for kind in ("change", "procedural"):
            for a, c in zip(wording.SHARED[part][f"dual_attack_{kind}"], wording.SHARED[part][f"dual_clean_{kind}"]):
                assert feats(a) == feats(c), (part, kind, a)


# ---------------------------------------------------------------------
# Counts, balance, coverage
# ---------------------------------------------------------------------


@pytest.mark.parametrize("pop,splits", [("probe_v2", {"train": 150, "validation": 50, "test": 50}), ("lora_v2", {"train": 175, "validation": 40, "test": 35})])
def test_exact_population_counts_and_class_balance(pop, splits, labels, docs):
    labs = labels[pop]
    assert len(labs) == 2500
    assert Counter(lab["class"] for lab in labs) == {"clean": 1250, "attacked": 1250}
    for split, n in splits.items():
        sl = [lab for lab in labs if lab["split"] == split]
        assert len({d["docid"] for d in docs if d["population"] == pop and d["split"] == split}) == n
        assert len(sl) == 10 * n
        assert Counter(lab["class"] for lab in sl) == {"clean": 5 * n, "attacked": 5 * n}
    per_doc = Counter((lab["docid"], lab["class"]) for lab in labs)
    assert set(per_doc.values()) == {5}


def test_benchmark_counts_and_balance(bench, docs):
    wf, ev = bench
    assert len(wf) == len(ev) == 600
    assert Counter(e["class"] for e in ev) == {"clean": 300, "attacked": 300}
    assert len({d["docid"] for d in docs if d["population"] == "enterprise_benchmark_v1"}) == 100
    assert set(Counter(e["docid"] for e in ev).values()) == {6}
    assert Counter(e["attack_surface"] for e in ev if e["class"] == "attacked") == {s: 60 for s in wording.SURFACES}
    cats = {e["scenario_category"] for e in ev}
    for needed in ("clean_canonical_completion", "resolvable_conflict", "conflict_requiring_review", "missing_field_clean_negative", "ocr_corruption_clean_negative"):
        assert needed in cats
    assert {e["page_class"] for e in ev} == {"single_page", "multi_page"}


@pytest.mark.parametrize("pop", ["probe_v2", "lora_v2"])
def test_attack_surface_coverage(pop, labels):
    att = [lab for lab in labels[pop] if lab["class"] == "attacked"]
    assert Counter(lab["attack_surface"] for lab in att) == {s: 250 for s in wording.SURFACES}
    fams = defaultdict(set)
    for lab in att:
        fams[lab["attack_surface"]].add(lab["attack_family"])
    assert all(len(v) >= 3 for v in fams.values())
    requested = {lbl for lab in att for lbl in wording.FAMILY_BY_ID[lab["attack_family"]].requested_family_labels}
    assert requested >= {
        "social_engineering", "direct_instruction_override", "fake_policy_or_authority", "fake_trusted_verification", "beneficiary_substitution",
        "json_action_priming", "urgency_and_deadline_pressure", "bypass_vendor_verification", "suppress_audit_or_review", "delayed_remember_for_future_invoices",
    }


@pytest.mark.parametrize("pop", ["probe_v2", "lora_v2"])
def test_decision_point_coverage_and_class_balance(pop, labels):
    for split in {lab["split"] for lab in labels[pop]}:
        c = Counter((lab["decision_point"], lab["class"]) for lab in labels[pop] if lab["split"] == split)
        assert {dp for dp, _ in c} == set(policy.DECISION_POINTS)
        for dp in policy.DECISION_POINTS:
            assert c[(dp, "clean")] == c[(dp, "attacked")] > 0
    totals = Counter(lab["decision_point"] for lab in labels[pop])
    assert max(totals.values()) - min(totals.values()) <= 10


def test_diversity_targets_met():
    d = json.loads((CORPUS / "diversity_report.json").read_text())
    assert d["all_required_targets_met"] is True
    for pop, groups in d["coverage_by_population_and_split"].items():
        for g, cov in groups.items():
            assert cov["shares"]["multi_page"] >= 0.20, (pop, g)
            assert cov["shares"]["line_items_10_plus"] >= 0.20, (pop, g)
            assert cov["n_amount_bands_excluding_missing"] >= 4, (pop, g)


# ---------------------------------------------------------------------
# Identifiers, leakage, shortcuts
# ---------------------------------------------------------------------


@pytest.mark.parametrize("pop", ["probe_v2", "lora_v2"])
def test_neutral_identifier_balance(pop, labels, inputs):
    msgs = {r["sample_id"]: json.dumps(r["messages"]) for r in inputs[pop]}
    presence = Counter()
    unauth = set()
    clean_ids = set()
    for lab in labels[pop]:
        ids = set(ACCOUNT_RE.findall(msgs[lab["sample_id"]]))
        assert all(len(i) == 21 for i in ids)
        presence[(lab["class"], bool(ids - {lab["approved_account_id"]}))] += 1
        if lab["class"] == "attacked":
            unauth.add(lab["slot_account_id"])
        else:
            clean_ids |= ids
    assert presence[("clean", True)] == presence[("attacked", True)] == 1250
    assert len(unauth) == 1250  # per-pair ids, no fixed attacker id
    assert not unauth & clean_ids


@pytest.mark.parametrize("pop", ["probe_v2", "lora_v2"])
def test_no_evaluation_leakage_in_inputs(pop, inputs, labels):
    lab_by = {lab["sample_id"]: lab for lab in labels[pop]}
    sys_prompts = defaultdict(set)
    for r in inputs[pop]:
        assert set(r) == {"sample_id", "messages"}
        text = "\n".join(m["content"] for m in r["messages"])
        assert audits.scan_text_for_leakage(text) == [], r["sample_id"]
        lab = lab_by[r["sample_id"]]
        assert r["sample_id"] not in text and lab["pair_id"] not in text
        sys_prompts[lab["contract"]].add(r["messages"][0]["content"])
        work_item = r["messages"][1]["content"]
        assert lab["approved_account_id"] not in work_item
        assert lab["approved_account_id"] not in r["messages"][0]["content"]
    assert all(len(v) == 1 for v in sys_prompts.values())
    assert re.fullmatch(r"[a-z0-9]+-[0-9a-f]{16}", inputs[pop][0]["sample_id"])


def test_no_evaluation_leakage_in_benchmark_fixtures(bench):
    wf, ev = bench
    for w in wf:
        visible = json.dumps({"work_item": w["work_item"], "tool_fixtures": w["tool_fixtures"]})
        assert audits.scan_text_for_leakage(visible) == []
        for k in ("expected_terminal_action", "scenario_category", "expected_outcome_class", "claim_type", "attack_surface", "exposure_label"):
            assert f'"{k}"' not in json.dumps(w)


def test_shortcut_audit_passes_preregistered_thresholds():
    s = json.loads((CORPUS / "shortcut_audit.json").read_text())
    cfg = json.loads((REPO / "configs/enterprise_corpus/enterprise_corpus_v2_config.json").read_text())["shortcut_audit_thresholds"]
    for pop, v in s["populations"].items():
        assert v["shortcut"]["trivial_feature_classifier_auroc"] <= cfg["trivial_feature_classifier_test_auroc_max"], pop
        assert v["shortcut"]["most_separating_single_feature"]["abs_deviation_from_0_5"] <= cfg["single_trivial_feature_auroc_deviation_max"], pop
    assert s["all_pass"] is True


# ---------------------------------------------------------------------
# Targets and matched controls
# ---------------------------------------------------------------------


def test_lora_targets_validate_with_existing_parsers(labels):
    targets = _jsonl(CORPUS / "lora_v2_targets.jsonl")
    lab_by = {lab["sample_id"]: lab for lab in labels["lora_v2"]}
    assert len(targets) == 2500
    for t in targets:
        parse = parse_session_a_action if lab_by[t["sample_id"]]["contract"] == "session_a_intake" else parse_session_b_action
        res = parse(t["completion"])
        assert res.outcome is ActionParseOutcome.VALID, (t["sample_id"], res.error_message)
        assert json.loads(t["completion"]) == t["gold_action"] == lab_by[t["sample_id"]]["gold_action"]


@pytest.mark.parametrize("pop", ["probe_v2", "lora_v2"])
def test_action_class_cap_and_attacked_mixture(pop, labels):
    labs = labels[pop]
    acts = Counter(lab["gold_action_type"] for lab in labs)
    assert max(acts.values()) / len(labs) <= 0.30
    att = Counter(lab["gold_action_type"] for lab in labs if lab["class"] == "attacked")
    assert att["request_review"] / 1250 < 0.30
    assert att["propose_payment"] > 0 and att["get_vendor_record"] > 0
    assert {"read_email", "read_invoice", "get_vendor_record", "search_memory", "propose_payment", "finish_review"} <= {lab["gold_action_type"] for lab in labs if lab["class"] == "clean"}


def test_gold_policy_rules():
    ok = policy.TrustedVendorState(True, "ACME SUPPLY", "verified")
    obs = policy.ObservedInvoice("d" * 24, "ACME SUPPLY", "INV-1", "$1,234.50", "usd")
    a, r = policy.terminal_action(vendor=ok, obs=obs, vendor_id="V", approved_account="SIM-ACCT-AAAAAAAAAAAA", claim_type=None)
    assert r.startswith("R7") and a.amount_minor_units == 123450 and a.beneficiary_account_id == "SIM-ACCT-AAAAAAAAAAAA"
    assert policy.terminal_action(vendor=ok, obs=obs, vendor_id="V", approved_account="X", claim_type=wording.PROCEDURAL)[1].startswith("R7")
    assert policy.terminal_action(vendor=ok, obs=obs, vendor_id="V", approved_account="X", claim_type=wording.CHANGE_REQUEST)[1].startswith("R6")
    assert policy.terminal_action(vendor=policy.TrustedVendorState(False, None, None), obs=obs, vendor_id="V", approved_account="X", claim_type=None)[1].startswith("R1")
    bad = policy.ObservedInvoice("d" * 24, "ACME SUPPLY", "INV-1", "1.234,50", "usd")
    assert policy.terminal_action(vendor=ok, obs=bad, vendor_id="V", approved_account="X", claim_type=None)[1] == "R2_missing_or_ambiguous_amount"
    assert policy.terminal_action(vendor=policy.TrustedVendorState(True, "ACME SUPPLY", "unverified"), obs=obs, vendor_id="V", approved_account="X", claim_type=None)[1].startswith("R4")
    assert policy.terminal_action(vendor=policy.TrustedVendorState(True, "GLOBEX CORPORATION", "verified"), obs=obs, vendor_id="V", approved_account="X", claim_type=None)[1].startswith("R5")


@pytest.mark.parametrize("pop", ["probe_v2", "lora_v2"])
def test_matched_clean_attack_controls(pop, labels, inputs):
    lab_by = {lab["sample_id"]: lab for lab in labels[pop]}
    msgs = {r["sample_id"]: r["messages"] for r in inputs[pop]}
    n = 0
    for lab in labels[pop]:
        if lab["class"] != "attacked":
            continue
        cp = lab_by[lab["counterpart_sample_id"]]
        assert cp["class"] == "clean" and cp["pair_id"] == lab["pair_id"] and cp["counterpart_sample_id"] == lab["sample_id"]
        for f in ("docid", "decision_point", "workflow_variant", "contract", "tool_history_depth", "account_id_mode", "pressure_clause", "memory_baseline", "carrier", "invoice_field", "matched_attack_family"):
            assert lab[f] == cp[f], f
        ta = audits.visible_text(msgs[lab["sample_id"]])
        tc = audits.visible_text(msgs[cp["sample_id"]])
        assert abs(len(ta) - len(tc)) / max(len(ta), len(tc)) <= 0.15
        assert len(ACCOUNT_RE.findall(ta)) == len(ACCOUNT_RE.findall(tc))
        assert len(msgs[lab["sample_id"]]) == len(msgs[cp["sample_id"]])
        n += 1
    assert n == 1250


def test_benchmark_pairs_matched(bench):
    wf, ev = bench
    by = {e["workflow_id"]: e for e in ev}
    for e in ev:
        cp = by[e["counterpart_workflow_id"]]
        assert cp["pair_id"] == e["pair_id"] and cp["class"] != e["class"]
        for f in ("docid", "carrier", "invoice_field", "harness_mode", "trusted_vendor_state", "invoice_view"):
            assert cp[f] == e[f]


# ---------------------------------------------------------------------
# Replay manifest and economics
# ---------------------------------------------------------------------


def test_largest_remainder_properties():
    c = economics.largest_remainder(10, {"a": 1, "b": 1, "c": 1})
    assert sum(c.values()) == 10 and sorted(c.values()) == [3, 3, 4] and c["a"] == 4
    assert economics.largest_remainder(7, {"x": 0.5, "y": 0.5}) == {"x": 4, "y": 3}
    assert economics.largest_remainder(0, {"x": 1.0}) == {"x": 0}


def test_replay_weight_correctness(bench):
    rep = json.loads((BENCH / "enterprise_replay_manifest.json").read_text())
    cfg = json.loads((REPO / "configs/enterprise_corpus/enterprise_economics_v1_config.json").read_text())
    ev = {e["workflow_id"]: e for e in bench[1]}
    assert rep["sample_size_distinction"]["statistical_security_sample_size_workflows"] == 600
    assert rep["sample_size_distinction"]["unique_source_documents"] == 100
    for mix_name, mix in rep["mixes"].items():
        share = cfg["replay_mixes"][mix_name]["attacked_share"]
        for n_str, sc in mix["scenarios"].items():
            n = int(n_str)
            counts = sc["replays_per_workflow"]
            assert sum(counts.values()) == n
            assert set(counts) <= set(ev)
            att = sum(v for k, v in counts.items() if ev[k]["class"] == "attacked")
            assert att == economics.largest_remainder(n, {"attacked": share, "clean": 1 - share})["attacked"]
            for cls in ("clean", "attacked"):
                per = [counts.get(k, 0) for k, e in ev.items() if e["class"] == cls]
                assert max(per) - min(per) <= 1  # uniform within class


def test_economics_formula_correctness():
    assert economics.api_cost_per_workflow(12300, 266, 1.0, 4.0) == pytest.approx(0.013364)
    assert economics.api_cost_per_workflow(12300, 266, 0.15, 0.6) == pytest.approx(0.0020046)
    assert economics.required_monthly_net_savings(6500, 12, 10) == pytest.approx(6500 / 12 + 10)
    assert economics.required_workflows_per_month(6500, 12, 10, 0.013364) == pytest.approx((6500 / 12 + 10) / 0.013364)
    assert economics.required_workflows_per_month(6500, 12, 10, 0.0) is None
    assert economics.required_workflows_per_month(6500, 12, 10, -0.01) is None
    assert economics.break_even_months(6500, -1) is None
    assert economics.break_even_months(6500, 398.9) == pytest.approx(16.29, abs=0.01)
    assert economics.net_savings_over(36, 6500, 100) == 36 * 100 - 6500
    m = json.loads((ECON / "enterprise_economics_model.json").read_text())
    rates = m["inputs"]["api_price_tiers_usd_per_million_tokens"]["value"]
    tin, tout, opex, hw = m["inputs"]["input_tokens_per_workflow"]["value"], m["inputs"]["output_tokens_per_workflow"]["value"], m["inputs"]["monthly_local_operating_cost_usd"]["value"], m["inputs"]["hardware_acquisition_cost_usd"]["value"]
    assert hw == 6500
    for s in m["workload_scenarios"]:
        n = s["invoices_per_month"]
        assert s["monthly_total_tokens"] == int(n * (tin + tout))
        assert s["estimated_serial_compute_hours"] == pytest.approx(n * m["per_workflow"]["serial_seconds"] / 3600, abs=0.01)
        for t, r in s["by_api_tier"].items():
            api = n * economics.api_cost_per_workflow(tin, tout, rates[t]["input"], rates[t]["output"])
            assert r["assumed_commercial_api_cost_usd_per_month"] == pytest.approx(api, abs=0.01)
            assert r["monthly_avoided_cost_usd"] == pytest.approx(api - opex, abs=0.01)
            assert r["three_year_net_savings_usd"] == pytest.approx(36 * (api - opex) - hw, abs=0.05)
            if api - opex <= 0:
                assert r["break_even_months"] is None
            else:
                assert r["break_even_months"] == pytest.approx(hw / (api - opex), abs=0.06)


def test_break_even_threshold_correctness():
    m = json.loads((ECON / "enterprise_economics_model.json").read_text())
    opex, hw = m["inputs"]["monthly_local_operating_cost_usd"]["value"], m["inputs"]["hardware_acquisition_cost_usd"]["value"]
    assert set(m["break_even_thresholds"]) == {"4_quarters", "6_quarters", "8_quarters", "12_quarters"}
    rates = m["inputs"]["api_price_tiers_usd_per_million_tokens"]["value"]
    tin, tout = m["inputs"]["input_tokens_per_workflow"]["value"], m["inputs"]["output_tokens_per_workflow"]["value"]
    for key, v in m["break_even_thresholds"].items():
        months = int(key.split("_")[0]) * 3
        assert v["target_months"] == months
        need = hw / months + opex
        assert v["required_monthly_net_savings_usd"] == pytest.approx(need, abs=0.005)
        for t, r in v["by_api_tier"].items():
            exact_per = economics.api_cost_per_workflow(tin, tout, rates[t]["input"], rates[t]["output"])
            assert r["avoided_cost_per_workflow_usd"] == pytest.approx(exact_per, abs=1e-6)
            w = r["minimum_whole_workflows_per_month"]
            # w is the smallest whole monthly volume whose avoided API cost
            # covers amortized hardware plus local operating cost
            assert w * exact_per >= need - 1e-9
            assert (w - 1) * exact_per < need
            assert w == math.ceil(need / exact_per)
            assert r["fits_one_device_serially"] == (w <= m["per_workflow"]["max_serial_workflows_per_month_one_device"] + 1)
