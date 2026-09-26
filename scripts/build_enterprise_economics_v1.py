"""Enterprise workload and $6,500 break-even model (economics v1), plus the
deterministic enterprise replay manifest for enterprise benchmark v1.

Runs strictly AFTER the corpus/benchmark are frozen and never feeds back
into them: it refuses to run unless the benchmark manifest is frozen and
its data-file hashes still match, and corpus selection code never imports
this script or `enterprise_v2.economics`.

Every input is copied from an existing reporting artifact (named per
value) or from configs/enterprise_corpus/enterprise_economics_v1_config.json,
where each assumption is labeled.

    .venv/bin/python scripts/build_enterprise_economics_v1.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from enterprise_v2 import economics as E  # noqa: E402

CFG_PATH = REPO_ROOT / "configs/enterprise_corpus/enterprise_economics_v1_config.json"
ECON_V1 = REPO_ROOT / "results/reporting/tell_economics_model_v1.json"
LEDGER_V1 = REPO_ROOT / "results/reporting/tell_metrics_ledger_v1.json"
BENCH_DIR = REPO_ROOT / "results/enterprise_benchmark/v1"
OUT_DIR = REPO_ROOT / "results/economics/enterprise_v1"

TIERS = ("low_cost_tier", "mid_tier", "premium_tier")


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def claim(ledger: dict, cid: str) -> dict:
    for c in ledger["claims"]:
        if c["claim_id"] == cid:
            return c
    raise KeyError(cid)


def load_inputs() -> dict:
    cfg = json.loads(CFG_PATH.read_text())
    econ = json.loads(ECON_V1.read_text())
    ledger = json.loads(LEDGER_V1.read_text())
    bench_manifest = json.loads((BENCH_DIR / "enterprise_benchmark_v1_manifest.json").read_text())
    if not bench_manifest.get("frozen"):
        raise SystemExit("benchmark is not frozen; economics must run after freezing")
    for key in ("workflows", "evaluation_only"):
        f = bench_manifest["files"][key]
        if sha_file(REPO_ROOT / f["path"]) != f["sha256"]:
            raise SystemExit(f"frozen benchmark file changed: {f['path']}")
    wb = econ["workload_basis"]
    lc = econ["local_cost"]
    lt03 = claim(ledger, "LT-03")
    lt04 = claim(ledger, "LT-04")
    lt06 = claim(ledger, "LT-06")
    tk01 = claim(ledger, "TK-01")
    rt08 = claim(ledger, "RT-08")
    proj = json.loads((BENCH_DIR / "enterprise_benchmark_v1_token_projection.json").read_text())
    ev = [json.loads(line) for line in (BENCH_DIR / "enterprise_benchmark_v1_evaluation_only.jsonl").read_text().splitlines()]
    return {
        "cfg": cfg,
        "econ": econ,
        "hardware_cost": float(lc["hardware_acquisition_cost_usd"]["value"]),
        "opex_base": float(lc["monthly_local_operating_cost_usd_used_below"]),
        "power_w": lc["power_draw_watts_during_active_inference"],
        "elec_rate": lc["electricity_rate_usd_per_kwh"],
        "tiers": econ["equivalent_paid_model_api_cost"]["pricing_tiers_usd_per_million_tokens"],
        "input_tokens": float(wb["rounded_workload_figures_used_below"]["input_tokens_per_workflow"]),
        "output_tokens": float(wb["rounded_workload_figures_used_below"]["output_tokens_per_workflow"]),
        "measured_input_tokens_tk01": wb["input_tokens_per_completed_workflow"],
        "measured_total_tokens_tk01": tk01["value"],
        "gen_seconds": float(lt03["value"]),
        "capture_seconds": float(lt04["value"]),
        "review_path_seconds": lt06["value"],
        "cold_load_seconds_mean": float(rt08["value"]),
        "projection": proj,
        "evaluation": ev,
        "bench_manifest_sha256": sha_file(BENCH_DIR / "enterprise_benchmark_v1_manifest.json"),
    }


def build(inp: dict) -> tuple[dict, dict, str]:
    cfg = inp["cfg"]
    hw = inp["hardware_cost"]
    opex = inp["opex_base"]
    sec_wf = inp["gen_seconds"] + inp["capture_seconds"]
    hours_month = cfg["hours_per_month"]
    cap_per_month = hours_month * 3600 / sec_wf
    in_t, out_t = inp["input_tokens"], inp["output_tokens"]
    proj_sum = inp["projection"]["summary"]
    proj_in, proj_out = proj_sum["prompt_tokens_mean"], proj_sum["completion_tokens_mean"]
    proj_scale = (proj_in + proj_out) / (in_t + out_t)

    per_wf = {t: E.api_cost_per_workflow(in_t, out_t, inp["tiers"][t]["input"], inp["tiers"][t]["output"]) for t in TIERS}
    per_wf_proj = {t: E.api_cost_per_workflow(proj_in, proj_out, inp["tiers"][t]["input"], inp["tiers"][t]["output"]) for t in TIERS}

    scenarios = []
    for n in cfg["workload_scenarios_invoices_per_month"]:
        hours = E.serial_compute_hours(n, sec_wf)
        row = {
            "invoices_per_month": n,
            "monthly_workflows": n,
            "monthly_input_tokens": int(n * in_t),
            "monthly_output_tokens": int(n * out_t),
            "monthly_total_tokens": int(n * (in_t + out_t)),
            "estimated_serial_compute_hours": round(hours, 2),
            "implied_average_device_utilization": round(hours / hours_month, 4),
            "implied_utilization_if_business_hours_only": round(hours / cfg["business_hours_per_month_reference"], 4),
            "local_operating_cost_usd_per_month": opex,
            "by_api_tier": {},
            "sensitivity_projected_enterprise_tokens": {
                "monthly_total_tokens": int(n * (proj_in + proj_out)),
                "estimated_serial_compute_hours": round(hours * proj_scale, 2),
                "implied_average_device_utilization": round(hours * proj_scale / hours_month, 4),
            },
            "sensitivity_usage_scaled_electricity_usd_per_month": {
                lvl: round(hours * inp["power_w"][lvl] / 1000 * inp["elec_rate"][lvl], 2) for lvl in ("low", "base", "high")
            },
        }
        for t in TIERS:
            api = n * per_wf[t]
            avoided = api - opex
            bem = E.break_even_months(hw, avoided)
            api_p = n * per_wf_proj[t]
            bem_p = E.break_even_months(hw, api_p - opex)
            row["by_api_tier"][t] = {
                "assumed_commercial_api_cost_usd_per_month": round(api, 2),
                "local_operating_cost_usd_per_month": opex,
                "monthly_avoided_cost_usd": round(avoided, 2),
                "break_even_months": None if bem is None else round(bem, 1),
                "break_even_quarters": None if bem is None else round(bem / 3, 1),
                "three_year_net_savings_usd": round(E.net_savings_over(cfg["three_year_horizon_months"], hw, avoided), 2),
                "sensitivity_projected_enterprise_tokens": {
                    "assumed_commercial_api_cost_usd_per_month": round(api_p, 2),
                    "break_even_months": None if bem_p is None else round(bem_p, 1),
                    "three_year_net_savings_usd": round(E.net_savings_over(cfg["three_year_horizon_months"], hw, api_p - opex), 2),
                },
            }
        scenarios.append(row)

    thresholds = {}
    for q in cfg["break_even_targets_quarters"]:
        months = q * 3
        need = E.required_monthly_net_savings(hw, months, opex)
        thresholds[f"{q}_quarters"] = {"target_months": months, "required_monthly_net_savings_usd": round(need, 2), "by_api_tier": {}}
        for t in TIERS:
            req = E.required_workflows_per_month(hw, months, opex, per_wf[t])
            req_p = E.required_workflows_per_month(hw, months, opex, per_wf_proj[t])
            wf = E.min_whole_workflows(req)
            thresholds[f"{q}_quarters"]["by_api_tier"][t] = {
                "avoided_cost_per_workflow_usd": round(per_wf[t], 6),
                "required_workflows_per_month_exact": None if req is None else round(req, 2),
                "minimum_whole_workflows_per_month": wf,
                "serial_compute_hours_at_that_volume": None if wf is None else round(E.serial_compute_hours(wf, sec_wf), 1),
                "implied_average_device_utilization": None if wf is None else round(E.serial_compute_hours(wf, sec_wf) / hours_month, 3),
                "fits_one_device_serially": None if wf is None else wf <= cap_per_month,
                "sensitivity_projected_enterprise_tokens_min_workflows": E.min_whole_workflows(req_p),
            }

    model = {
        "model_version": "enterprise_economics_v1",
        "generated_after_freeze": True,
        "benchmark_manifest_sha256_at_read": inp["bench_manifest_sha256"],
        "independence_statement": "Computed after the corpus and benchmark were frozen. No selection, split, template, or sample was chosen or changed based on any quantity here; the selection code never imports this module.",
        "sample_size_distinction": {
            "statistical_security_sample_size_workflows": 600,
            "unique_source_documents": 100,
            "projected_monthly_business_workload_max": max(cfg["workload_scenarios_invoices_per_month"]),
            "note": "Monthly workloads are replay projections of the 600 benchmark workflows. They are not additional independent security examples.",
        },
        "inputs": {
            "hardware_acquisition_cost_usd": {"value": hw, "source": "results/reporting/tell_economics_model_v1.json local_cost.hardware_acquisition_cost_usd (quoted, Project-Expenditure.txt)", "status": "assumption (quote, not an invoice)"},
            "input_tokens_per_workflow": {"value": in_t, "source": "tell_economics_model_v1.json workload_basis.rounded_workload_figures_used_below (midpoint of measured 12,207-12,539; TK-01 measured 12,207 input / 12,473 total)", "status": "measured proxy (Qwen tokenizer count of the real 4-turn autonomous_v2 workflow)"},
            "output_tokens_per_workflow": {"value": out_t, "source": "same (propose_payment path; the request_review path is 124)", "status": "measured"},
            "warm_generation_seconds_per_workflow": {"value": inp["gen_seconds"], "source": "tell_metrics_ledger_v1.json LT-03", "status": "measured (one deterministic run, hardened prompt, 4 turns)"},
            "capture_seconds_per_workflow": {"value": inp["capture_seconds"], "source": "tell_metrics_ledger_v1.json LT-04", "status": "measured (capture at every turn; production captures a subset)"},
            "serial_seconds_per_workflow_used": {"value": round(sec_wf, 2), "status": "reconstructed = LT-03 + LT-04; excludes unmeasured probe scoring, gate, LoRA switching, ledger write"},
            "request_review_path_generation_seconds": {"value": inp["review_path_seconds"], "source": "LT-06", "status": "measured; not used (conservative: all workflows costed at the longer propose path)"},
            "model_cold_load_seconds_mean": {"value": inp["cold_load_seconds_mean"], "source": "RT-08", "status": "one-time per process start; excluded from per-workflow time"},
            "monthly_local_operating_cost_usd": {"value": opex, "source": "tell_economics_model_v1.json local_cost.monthly_local_operating_cost_usd_used_below", "status": "assumption"},
            "api_price_tiers_usd_per_million_tokens": {"value": inp["tiers"], "source": "tell_economics_model_v1.json equivalent_paid_model_api_cost", "status": "assumption (no dated, sourced price in the repository)"},
            "projected_enterprise_tokens_per_workflow": {"value": {"input": proj_in, "output": proj_out}, "source": "results/enterprise_benchmark/v1/enterprise_benchmark_v1_token_projection.json", "status": "projection (tokenizer count of canonical benchmark paths; no model run); sensitivity only"},
        },
        "per_workflow": {
            "api_cost_usd_measured_basis": {t: round(v, 6) for t, v in per_wf.items()},
            "api_cost_usd_projected_enterprise_basis": {t: round(v, 6) for t, v in per_wf_proj.items()},
            "serial_seconds": round(sec_wf, 2),
            "max_serial_workflows_per_month_one_device": int(cap_per_month),
        },
        "formulas": {
            "api_cost_per_workflow": "input_tokens/1e6*input_rate + output_tokens/1e6*output_rate",
            "monthly_avoided_cost": "monthly_workflows*api_cost_per_workflow - monthly_local_operating_cost",
            "break_even_months": "hardware_cost / monthly_avoided_cost (none if <= 0)",
            "required_monthly_net_savings": "hardware_cost / target_break_even_months + monthly_local_operating_cost",
            "required_workflows_per_month": "required_monthly_net_savings / avoided_cost_per_workflow (none if avoided_cost_per_workflow <= 0)",
            "three_year_net_savings": "36*monthly_avoided_cost - hardware_cost",
            "serial_compute_hours": "workflows * serial_seconds_per_workflow / 3600",
            "implied_utilization": "serial_compute_hours / 730.5",
        },
        "workload_scenarios": scenarios,
        "break_even_thresholds": thresholds,
        "value_categories": {
            "token_cost_break_even": "modeled above (the only monetized category)",
            "privacy_and_security_value": "not monetized: financial documents, beneficiary data and agent memory stay on-device; hidden-state access for Tell is only possible locally",
            "labor_savings": "unmeasured; not monetized (no measured AP handling time or review-rate data)",
            "fraud_loss_avoidance": "unmeasured; not monetized (no evidence-based incidence or loss-per-incident data; simulated invoice values are not avoided losses)",
            "device_sharing_across_other_local_workloads": "not monetized; at the modeled volumes the device is mostly idle, so capacity for other local workloads exists but has no measured value",
        },
        "missing_measurements": [
            "probe scoring latency (TK-04: expected sub-ms, never timed)",
            "gate latency (LT-09)",
            "LoRA adapter-switch / replan overhead (TK-05)",
            "true end-to-end integrated latency (LT-11)",
            "device power draw (active and idle)",
            "enterprise-v2 workflow timing (longer Session-B prompt, 4-5 turns)",
            "batched / concurrent throughput (all compute figures here are strictly serial)",
        ],
    }
    replay = build_replay(inp, sec_wf)
    report = report_md(model, replay, inp)
    return model, replay, report


def build_replay(inp: dict, sec_wf: float) -> dict:
    cfg = inp["cfg"]
    ev = sorted(inp["evaluation"], key=lambda e: e["workflow_id"])
    per_wf_proj = inp["projection"]["canonical_path_projection_per_workflow"]
    mixes = {}
    for mix_name, mix in cfg["replay_mixes"].items():
        share = {"attacked": mix["attacked_share"], "clean": 1 - mix["attacked_share"]}
        per_scenario = {}
        for n in cfg["workload_scenarios_invoices_per_month"]:
            # Stage 1: split the monthly volume between classes by share;
            # stage 2: spread each class's count uniformly over its workflows.
            class_counts = E.largest_remainder(n, share)
            counts = {}
            for cls in ("clean", "attacked"):
                ids = [e["workflow_id"] for e in ev if e["class"] == cls]
                counts.update(E.largest_remainder(class_counts[cls], {i: 1.0 for i in ids}))
            assert sum(counts.values()) == n
            cats = Counter()
            proj_tokens = 0
            for e in ev:
                cats[f"{e['class']}|{e['scenario_category']}"] += counts[e["workflow_id"]]
                p = per_wf_proj[e["workflow_id"]]
                proj_tokens += counts[e["workflow_id"]] * (p["prompt_tokens"] + p["completion_tokens"])
            per_scenario[str(n)] = {
                "monthly_workflows": n,
                "replays_per_workflow": {k: v for k, v in counts.items() if v},
                "replays_by_class": dict(Counter({"clean": sum(counts[e["workflow_id"]] for e in ev if e["class"] == "clean"), "attacked": sum(counts[e["workflow_id"]] for e in ev if e["class"] == "attacked")})),
                "replays_by_category": dict(sorted(cats.items())),
                "distinct_benchmark_workflows_replayed": sum(1 for v in counts.values() if v),
                "projected_monthly_tokens_from_replayed_fixtures": proj_tokens,
                "measured_basis_monthly_tokens": int(n * (inp["input_tokens"] + inp["output_tokens"])),
                "measured_basis_serial_compute_hours": round(n * sec_wf / 3600, 2),
            }
        mixes[mix_name] = {"definition": mix, "scenarios": per_scenario}
    return {
        "manifest_version": "enterprise_replay_manifest_1",
        "benchmark_manifest_sha256": inp["bench_manifest_sha256"],
        "sample_size_distinction": {
            "statistical_security_sample_size_workflows": 600,
            "unique_source_documents": 100,
            "projected_monthly_business_workload_workflows": cfg["workload_scenarios_invoices_per_month"],
            "statement": "Replays repeat the 600 frozen benchmark workflows with deterministic weights to model monthly business volume (tokens, compute hours). A replayed workflow is the same security example, not a new one; security claims rest on the 600 workflows / 100 documents only.",
        },
        "weighting_method": "two-stage largest-remainder apportionment (ties broken by key): the monthly volume is first split between clean and attacked by the mix's class share, then each class's count is spread uniformly over that class's 300 workflows. Counts sum exactly to the monthly volume.",
        "mixes": mixes,
    }


def _fmt_money(x):
    return "n/a" if x is None else f"${x:,.2f}"


def report_md(m: dict, replay: dict, inp: dict) -> str:
    L = ["# Enterprise Break-Even Report (economics v1)\n"]
    L.append("> **Three different numbers, never interchangeable.** Statistical security sample: **600 benchmark workflows**. Unique source documents: **100**. "
             "Projected monthly business workload: **1,000-25,000 workflows**, produced by replaying those 600 workflows (see `results/enterprise_benchmark/v1/enterprise_replay_manifest.json`). "
             "Replays are not additional security examples.\n")
    L.append("Computed after the corpus and benchmark were frozen. No sample was chosen or changed to improve these figures.\n")
    pw = m["per_workflow"]
    L.append("## 1. Inputs\n")
    L.append("| Input | Value | Status / source |\n|---|---|---|")
    for k, v in m["inputs"].items():
        L.append(f"| {k} | {v['value']} | {v['status']} ({v.get('source', '')}) |")
    L.append(f"\nPer-workflow commercial-API cost (measured token basis): low {_fmt_money(pw['api_cost_usd_measured_basis']['low_cost_tier'])}, mid {_fmt_money(pw['api_cost_usd_measured_basis']['mid_tier'])}, premium {_fmt_money(pw['api_cost_usd_measured_basis']['premium_tier'])}. Values below $0.01 round to $0.00 at two decimals; exact values are in the JSON model.")
    L.append(f"Serial compute per workflow: {pw['serial_seconds']} s (so one device can serve at most ~{pw['max_serial_workflows_per_month_one_device']:,} workflows/month serially).\n")
    L.append("## 2. Workload scenarios (measured token basis, $10/month local operating cost)\n")
    L.append("| Invoices/month | Input tokens | Output tokens | Total tokens | Serial compute h | Utilization (730.5 h) | Utilization (176 business h) |\n|---|---|---|---|---|---|---|")
    for s in m["workload_scenarios"]:
        L.append(f"| {s['invoices_per_month']:,} | {s['monthly_input_tokens']:,} | {s['monthly_output_tokens']:,} | {s['monthly_total_tokens']:,} | {s['estimated_serial_compute_hours']} | {s['implied_average_device_utilization']:.1%} | {s['implied_utilization_if_business_hours_only']:.1%} |")
    L.append("")
    L.append("| Invoices/month | API tier | API cost/month | Local cost/month | Avoided/month | Break-even months | Quarters | 3-year net |\n|---|---|---|---|---|---|---|---|")
    for s in m["workload_scenarios"]:
        for t, r in s["by_api_tier"].items():
            be = "no break-even" if r["break_even_months"] is None else f"{r['break_even_months']:,}"
            bq = "no break-even" if r["break_even_quarters"] is None else f"{r['break_even_quarters']:,}"
            L.append(f"| {s['invoices_per_month']:,} | {t} | {_fmt_money(r['assumed_commercial_api_cost_usd_per_month'])} | {_fmt_money(r['local_operating_cost_usd_per_month'])} | {_fmt_money(r['monthly_avoided_cost_usd'])} | {be} | {bq} | {_fmt_money(r['three_year_net_savings_usd'])} |")
    L.append("")
    L.append("## 3. Minimum monthly workflows for the $6,500 device to break even\n")
    L.append("`required workflows/month = (6500 / target_months + 10) / avoided_cost_per_workflow`\n")
    L.append("| Target | Required net savings/month | Low tier | Mid tier | Premium tier |\n|---|---|---|---|---|")
    for k, v in m["break_even_thresholds"].items():
        cells = []
        for t in TIERS:
            r = v["by_api_tier"][t]
            fit = "" if r["fits_one_device_serially"] else " (exceeds one device's serial capacity)"
            cells.append(f"{r['minimum_whole_workflows_per_month']:,} ({r['implied_average_device_utilization']:.0%} util.){fit}")
        L.append(f"| {k.replace('_', ' ')} ({v['target_months']} mo) | {_fmt_money(v['required_monthly_net_savings_usd'])} | " + " | ".join(cells) + " |")
    L.append("\nSensitivity with the projected enterprise-v2 token volume (tokenizer count of the canonical benchmark paths, ~33% more tokens per workflow than the measured pilot): "
             + "; ".join(f"{k.replace('_', ' ')}: " + ", ".join(f"{t.split('_')[0]} {v['by_api_tier'][t]['sensitivity_projected_enterprise_tokens_min_workflows']:,}" for t in TIERS) for k, v in m["break_even_thresholds"].items()) + ".\n")
    s10 = next(s for s in m["workload_scenarios"] if s["invoices_per_month"] == 10000)
    L.append("## 4. Is 10,000 invoices/month computationally feasible?\n")
    L.append(f"Using only measured component timing (LT-03 generation 28.57 s + LT-04 capture 2.90 s = {pw['serial_seconds']} s per workflow, warm, strictly serial): 10,000 workflows need "
             f"**{s10['estimated_serial_compute_hours']} device-hours/month**, i.e. **{s10['implied_average_device_utilization']:.1%}** average utilization of a 730.5-hour month, "
             f"or {s10['implied_utilization_if_business_hours_only']:.1%} if work is confined to 176 business hours. "
             f"With the projected longer enterprise-v2 context (~33% more tokens, timing scaled proportionally, an assumption) it is {s10['sensitivity_projected_enterprise_tokens']['estimated_serial_compute_hours']} h ({s10['sensitivity_projected_enterprise_tokens']['implied_average_device_utilization']:.1%}). "
             "**Yes, it is feasible on one device with large headroom.** Caveats: probe scoring, gate, LoRA switching, and ledger time were never measured. The timing comes from a single deterministic 4-turn run. No concurrent or batched throughput was measured. The ~103 s model cold load is one-time.\n")
    L.append("## 5. What is and is not monetized\n")
    for k, v in m["value_categories"].items():
        L.append(f"- **{k.replace('_', ' ')}**: {v}")
    L.append("\n## 6. Inputs that remain assumptions\n")
    L.append("- $6,500 hardware price (quote), $10/month local operating cost, and API price tiers (no dated source). Also power draw and electricity rate, the 1% attacked share in the business replay mix, and the compute-time scaling used for the projected-token sensitivity.")
    L.append("- Missing measurements: " + "; ".join(m["missing_measurements"]) + ".\n")
    L.append("## 7. Honest reading\n")
    t4 = m["break_even_thresholds"]["4_quarters"]["by_api_tier"]
    fast = [f"{sc['invoices_per_month']:,}/month at {t.split('_')[0]}" for sc in m["workload_scenarios"] for t, r in sc["by_api_tier"].items() if r["break_even_months"] is not None and r["break_even_months"] <= 12]
    within4 = ("Of the modeled scenarios, only " + ", ".join(fast) + " break" + ("s" if len(fast) == 1 else "") + " even within 4 quarters.") if fast else "None of the modeled scenarios breaks even within 4 quarters."
    s10p = s10["by_api_tier"]["premium_tier"]
    s10m = s10["by_api_tier"]["mid_tier"]
    t12 = m["break_even_thresholds"]["12_quarters"]["by_api_tier"]
    L.append(f"On token cost alone, the device pays back within 4 quarters only at about {t4['premium_tier']['minimum_whole_workflows_per_month']:,} workflows/month at premium API pricing, "
             f"or {t4['mid_tier']['minimum_whole_workflows_per_month']:,} at mid-tier pricing. "
             f"At low-cost API pricing even a 12-quarter payback needs {t12['low_cost_tier']['minimum_whole_workflows_per_month']:,}/month. "
             + within4 + " "
             + f"10,000/month breaks even in {s10p['break_even_quarters']} quarters at premium pricing and {s10m['break_even_months'] / 12:.1f} years at mid-tier pricing. "
             "The case for the device therefore rests mostly on unmonetized factors: data residency for sensitive AP data, the hidden-state access Tell requires (not available through any hosted API), and sharing the largely idle device with other local workloads. It does not rest on token savings. "
             "Labor and fraud-loss effects are real candidates but are unmeasured, so they are not counted.\n")
    return "\n".join(L) + "\n"


def main() -> None:
    inp = load_inputs()
    model, replay, report = build(inp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "enterprise_economics_model.json").write_text(json.dumps(model, indent=2) + "\n")
    (OUT_DIR / "enterprise_break_even_report.md").write_text(report)
    (BENCH_DIR / "enterprise_replay_manifest.json").write_text(json.dumps(replay, indent=2) + "\n")
    print("wrote economics model, break-even report, replay manifest")


if __name__ == "__main__":
    main()
