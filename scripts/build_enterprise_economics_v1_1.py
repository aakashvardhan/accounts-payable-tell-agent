"""Enterprise economics v1.1: the frozen economics v1 formulas and inputs
(imported unchanged from scripts/build_enterprise_economics_v1.py), with
the workload replay derived from the v1.1 representative operations
benchmark. results/economics/enterprise_v1 and the v1 replay manifest are
never written.

    .venv/bin/python scripts/build_enterprise_economics_v1_1.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_enterprise_economics_v1 as v1  # noqa: E402

CFG = REPO_ROOT / "configs/enterprise_corpus/v2_1/enterprise_economics_v1_1_config.json"
BENCH = REPO_ROOT / "results/enterprise_benchmark/v1_1"
OUT = REPO_ROOT / "results/economics/enterprise_v1_1"
REP = "representative_operations_benchmark"


def load_inputs() -> dict:
    cfg = json.loads(CFG.read_text())
    man_path = BENCH / f"{REP}_manifest.json"
    man = json.loads(man_path.read_text())
    if not man.get("frozen"):
        raise SystemExit("representative benchmark is not frozen")
    for key in ("workflows", "evaluation_only"):
        f = man["files"][key]
        if hashlib.sha256((REPO_ROOT / f["path"]).read_bytes()).hexdigest() != f["sha256"]:
            raise SystemExit(f"frozen file changed: {f['path']}")
    econ = json.loads(v1.ECON_V1.read_text())
    ledger = json.loads(v1.LEDGER_V1.read_text())
    wb, lc = econ["workload_basis"], econ["local_cost"]
    return {
        "cfg": cfg, "econ": econ,
        "hardware_cost": float(lc["hardware_acquisition_cost_usd"]["value"]),
        "opex_base": float(lc["monthly_local_operating_cost_usd_used_below"]),
        "power_w": lc["power_draw_watts_during_active_inference"],
        "elec_rate": lc["electricity_rate_usd_per_kwh"],
        "tiers": econ["equivalent_paid_model_api_cost"]["pricing_tiers_usd_per_million_tokens"],
        "input_tokens": float(wb["rounded_workload_figures_used_below"]["input_tokens_per_workflow"]),
        "output_tokens": float(wb["rounded_workload_figures_used_below"]["output_tokens_per_workflow"]),
        "measured_input_tokens_tk01": wb["input_tokens_per_completed_workflow"],
        "measured_total_tokens_tk01": v1.claim(ledger, "TK-01")["value"],
        "gen_seconds": float(v1.claim(ledger, "LT-03")["value"]),
        "capture_seconds": float(v1.claim(ledger, "LT-04")["value"]),
        "review_path_seconds": v1.claim(ledger, "LT-06")["value"],
        "cold_load_seconds_mean": float(v1.claim(ledger, "RT-08")["value"]),
        "projection": json.loads((BENCH / f"{REP}_token_projection.json").read_text()),
        "evaluation": [json.loads(x) for x in (BENCH / f"{REP}_evaluation_only.jsonl").read_text().splitlines()],
        "bench_manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    inp = load_inputs()
    model, replay, report = v1.build(inp)
    model["model_version"] = "enterprise_economics_v1_1"
    model["relation_to_v1"] = "Formulas, inputs, and assumptions identical to economics v1; only the replay source changed to the v1.1 representative operations benchmark (invoice-only). Token-cost results are therefore unchanged except the projected-token sensitivity, which now uses the v1.1 representative benchmark's canonical paths."
    model["replay_source"] = f"results/enterprise_benchmark/v1_1/{REP} (claim 1 workload). The attack_eligible_security_challenge is never replayed."
    replay["manifest_version"] = "enterprise_replay_manifest_1_1"
    replay["replay_source_benchmark"] = REP
    replay["sample_size_distinction"]["statement"] += " The attack-eligible security challenge (300 pairs, 46 documents) is a separate causal-security sample and is not part of any workload replay."
    report = report.replace("# Enterprise Break-Even Report (economics v1)", "# Enterprise Break-Even Report (economics v1.1)").replace(
        "results/enterprise_benchmark/v1/enterprise_replay_manifest.json", "results/enterprise_benchmark/v1_1/enterprise_replay_manifest_v1_1.json")
    report = report.replace("Statistical security sample: **600 benchmark workflows**. Unique source documents: **100**.",
                            "Statistical samples: **600 representative-operations workflows** (100 invoice documents; claim 1) and, separately, **300 attack-eligible matched pairs** (46 documents; claim 2).")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "enterprise_economics_model.json").write_text(json.dumps(model, indent=2) + "\n")
    (OUT / "enterprise_break_even_report.md").write_text(report)
    (BENCH / "enterprise_replay_manifest_v1_1.json").write_text(json.dumps(replay, indent=2) + "\n")
    print("wrote economics v1.1 model, report, replay manifest")


if __name__ == "__main__":
    main()
