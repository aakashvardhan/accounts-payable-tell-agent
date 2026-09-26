"""TRAINING-SPLIT EPOCH-0 PREVIEW data capture. Runs the real local flow
(Agent 1 -> live hidden activation -> frozen Tell probe -> operational
threshold -> Agent 1 or epoch-0 Agent S -> trusted lookup tools ->
validator -> deterministic gate -> simulated terminal outcome) over 24
frozen TRAINING-split records (12 clean / 12 attacked) using the
EPOCH_0_PREVIEW adapter snapshot.

This is explicitly NOT checkpoint selection (needs all 3 epoch-boundary
checkpoints scored together per checkpoint_selection_contract_v2.json) and
NOT the frozen three-arm evaluation (needs the final selected+frozen
adapter and the held-out test/lexical/delayed-memory-OOD partitions,
untouched here). No interface/UI is built by this script -- it only
produces immutable JSONL/JSON/CSV/Markdown data artifacts.

Labeling discipline enforced throughout:
- NOT_EVALUABLE_FROM_SOURCE_RECORD: amount/currency/invoice-number ground
  truth does not exist in this corpus; never fabricated, never silently
  omitted -- always this explicit marker.
- SYNTHETIC_TRUSTED_DEMO_CONTEXT: the ONE deterministic enrichment this
  script performs is choosing a `source_account_id` string to satisfy
  `route_and_validate_payment`'s required parameter (it is never compared
  against anything and never affects any outcome) -- labeled here, not
  presented as real evidence. The proposal's own amount/currency/invoice-
  number, when present, are the MODEL's own generated claim, not fabricated
  by this script, and are used as-is (never invented, never overridden).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CORPUS = REPO / "results/enterprise_corpus/v2_2"
SUP_DIR = REPO / "results/enterprise_corpus/v2_2_training_supplement"
FROZEN_PROBE_DIR = REPO / "results/probe_training/enterprise_v1/frozen_probe"
CANDIDATE_DIR = REPO / "results/lora_training/agent_s_v1/candidates/epoch0_candidate_preview"
OUT = REPO / "results/lora_training/agent_s_v1/interface_demo_v1"
MAX_NEW_TOKENS = 256
SEED = 20260924
N_SAMPLES = 24
LABELS = ("EPOCH_0_PREVIEW", "NOT_FINAL_SELECTED_ADAPTER", "NOT_HELD_OUT_PERFORMANCE", "TRAINING-SPLIT EPOCH-0 PREVIEW")
NOT_EVALUABLE = "NOT_EVALUABLE_FROM_SOURCE_RECORD"
SYNTHETIC_DEMO_SOURCE_ACCOUNT = "SYNTHETIC_TRUSTED_DEMO_CONTEXT:SIM-SOURCE-DEMO"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha(p.read_bytes())


def load_train_pool() -> tuple[dict, dict]:
    v22_labels = [json.loads(l) for l in open(CORPUS / "lora_v2_2_labels.jsonl")]
    v22_inputs = {r["sample_id"]: r for r in (json.loads(l) for l in open(CORPUS / "lora_v2_2_inputs.jsonl"))}
    sup_labels = [json.loads(l) for l in open(SUP_DIR / "memory_supplement_v1_labels.jsonl")]
    sup_inputs = {r["sample_id"]: r for r in (json.loads(l) for l in open(SUP_DIR / "memory_supplement_v1_inputs.jsonl"))}
    labels = {r["sample_id"]: r for r in v22_labels if r["split"] == "train"}
    labels.update({r["sample_id"]: r for r in sup_labels if r["split"] == "train"})
    inputs = {**v22_inputs, **sup_inputs}
    return labels, inputs


def freeze_demo_sample() -> list[str]:
    demo_path = OUT / "demo_sample_ids.json"
    if demo_path.exists():
        return json.loads(demo_path.read_text())["sample_ids"]
    labels, _ = load_train_pool()
    rng = np.random.default_rng(SEED)
    by_stratum: dict[str, list[str]] = {}
    for sid, r in labels.items():
        key = f"{r['class']}|{r.get('gold_action_type')}"
        by_stratum.setdefault(key, []).append(sid)
    picked = []
    for key in sorted(by_stratum):
        group = sorted(by_stratum[key])
        n = max(1, round(len(group) * N_SAMPLES / len(labels)))
        idx = rng.choice(len(group), size=min(n, len(group)), replace=False)
        picked.extend(group[i] for i in idx)
    clean_ids = sorted(sid for sid in picked if labels[sid]["class"] == "clean")[:12]
    attacked_ids = sorted(sid for sid in picked if labels[sid]["class"] == "attacked")[:12]
    picked = sorted(clean_ids + attacked_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(json.dumps({"seed": SEED, "n": len(picked), "n_clean": len(clean_ids), "n_attacked": len(attacked_ids),
                                     "sample_ids": picked, "source": "lora_v2_2 + memory_supplement_v1, split=='train' only"}, indent=1))
    return picked


def run_trusted_lookups(row: dict) -> dict:
    from tell.agent.agent_s_tools import ScenarioMetadataProvider
    from tell.agent.trusted_lookups import DisputeCaseQuery, InvoicePaymentHistoryQuery, check_trusted_dispute_case_status, check_trusted_invoice_payment_history

    provider = ScenarioMetadataProvider()
    hist = check_trusted_invoice_payment_history(provider, InvoicePaymentHistoryQuery(invoice_document_id=row["docid"], vendor_id=row["docid"]))
    disp = check_trusted_dispute_case_status(provider, DisputeCaseQuery(invoice_document_id=row["docid"], vendor_id=row["docid"]))
    return {"payment_history": {"tool_name": hist.tool_name.value, "status": hist.status.value, "record": hist.record.model_dump(mode="json") if hist.record else None},
           "dispute_case": {"tool_name": disp.tool_name.value, "status": disp.status.value, "record": disp.record.model_dump(mode="json") if disp.record else None}}


def build_trusted_vendor_and_evidence(row: dict):
    from tell.safety.payment_validation import ObservedEvidence, TrustedVendorRecord, VendorStatus, VendorVerificationStatus

    docid = row["docid"]
    vendor_state = row.get("trusted_vendor_state", "verified_match")
    obs = set(row.get("observations_in_context", []))
    if vendor_state == "vendor_lookup_unavailable":
        trusted_vendor = None
    else:
        trusted_vendor = TrustedVendorRecord(
            vendor_id=docid, vendor_name=row.get("vendor_group_key", "vendor"), beneficiary_account_id=row.get("approved_account_id", "UNKNOWN"),
            verification_status=VendorVerificationStatus.UNVERIFIED if vendor_state == "vendor_unverified" else VendorVerificationStatus.VERIFIED,
            vendor_status=VendorStatus.ACTIVE)
    observed = ObservedEvidence(observed_invoice_document_id=docid if "invoice" in obs else None,
                                observed_vendor_record_id=docid if "vendor" in obs else None)
    return trusted_vendor, observed


def run_through_orchestrator(row: dict, action: dict | None, action_type: str | None, routed_agent):
    """Real, unmodified route_and_validate_payment. `trusted_invoice` is
    honestly None -- no amount/currency ground truth exists in this
    corpus, so validate_payment_proposal cannot check the model's claimed
    amount/currency against anything and correctly reports
    INVOICE_INCOMPLETE for any propose_payment, rather than a fabricated
    VALID. The ProposedPayment itself uses ONLY the model's own generated
    fields (already schema-validated by parse_action_v22 if parse_outcome
    was 'valid') -- nothing here invents an amount, currency, or invoice
    number."""
    from tell.agent.routing_orchestrator import NonPaymentAction, RoutingDecision, route_and_validate_payment
    from tell.safety.alarm import RoutingAlarmState
    from tell.safety.payment_validation import ProposedPayment

    class _SpyExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, a):
            self.calls.append(a)

    trusted_vendor, observed = build_trusted_vendor_and_evidence(row)
    proposal = None
    non_payment_action = None
    coverage_note = {"amount_correctness": NOT_EVALUABLE, "currency_correctness": NOT_EVALUABLE, "invoice_number_correctness": NOT_EVALUABLE,
                     "trusted_invoice_available": False}
    beneficiary_check = None
    if action_type == "propose_payment" and action:
        proposal = ProposedPayment(invoice_document_id=action["invoice_document_id"], invoice_number=action["invoice_number"],
                                   beneficiary_account_id=action["beneficiary_account_id"], amount_minor_units=action["amount_minor_units"],
                                   currency=action["currency"], evidence_invoice_document_id=action["evidence"]["invoice_document_id"],
                                   evidence_vendor_record_id=action["evidence"]["vendor_record_id"])
        beneficiary_check = {"matches_approved_beneficiary": action.get("beneficiary_account_id") == row.get("approved_account_id"),
                             "note": "independent signal computed here, not the validator's own outcome"}
    elif action_type in ("request_vendor_clarification", "submit_evidence_report", "fail_closed"):
        non_payment_action = {"request_vendor_clarification": NonPaymentAction.REQUEST_VENDOR_CLARIFICATION,
                              "submit_evidence_report": NonPaymentAction.SUBMIT_EVIDENCE_REPORT,
                              "fail_closed": NonPaymentAction.FAIL_CLOSED}[action_type]

    executor = _SpyExecutor()
    record = route_and_validate_payment(
        workflow_id=row["sample_id"], routing_decision=RoutingDecision(routed_agent=routed_agent, probe_score=row["tell_score"], reason="epoch0-preview-demo"),
        starting_alarm_state=RoutingAlarmState.CLEAR, proposal=proposal, non_payment_action=non_payment_action,
        trusted_vendor=trusted_vendor, trusted_invoice=None, observed=observed, pending_case=None, resolution=None,
        evidence_sources=["trusted_vendor_record"], source_account_id=SYNTHETIC_DEMO_SOURCE_ACCOUNT, reason="epoch0-preview-demo-replay", executor=executor,
    )
    return record, beneficiary_check, coverage_note, len(executor.calls)


def main(dry_run: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    demo_ids = freeze_demo_sample()
    labels, inputs = load_train_pool()
    rows = [{**labels[sid], "messages": inputs[sid]["messages"]} for sid in demo_ids]
    print(f"demo sample: {len(rows)} training-split records ({sum(1 for r in rows if r['class']=='clean')} clean / "
         f"{sum(1 for r in rows if r['class']=='attacked')} attacked), ids frozen at {OUT / 'demo_sample_ids.json'}")

    if dry_run:
        print(json.dumps({"planned_records": len(rows), "labels": LABELS}, indent=1))
        return

    import torch  # noqa: F401
    from enterprise_v2_2.contract import parse_action_v22
    from tell.agent.operational_router import OperationalTellRouter
    from tell.detector.capture import CaptureRequest, capture_predecision_activations
    from tell.routing.records import RoutedAgent
    from tell.safety.adapter_runtime import AdapterAwareRuntime
    from safetensors.numpy import load_file

    probe_cfg = json.loads((FROZEN_PROBE_DIR / "probe_config.json").read_text())
    probe_w = load_file(str(FROZEN_PROBE_DIR / "probe_weights.safetensors"))
    layer = probe_cfg["selected_layer"]
    candidate_provenance = json.loads((CANDIDATE_DIR / "provenance.json").read_text())

    runtime = AdapterAwareRuntime()
    runtime.load()
    info = runtime.attach_agent_s_adapter(CANDIDATE_DIR)
    print("attached EPOCH_0_PREVIEW adapter:", info)
    router = OperationalTellRouter()

    run_hashes = {"probe_config_sha256": sha_file(FROZEN_PROBE_DIR / "probe_config.json"), "probe_weights_sha256": sha_file(FROZEN_PROBE_DIR / "probe_weights.safetensors"),
                 "epoch0_preview_adapter_weights_sha256": info.adapter_weights_sha256, "epoch0_preview_provenance": candidate_provenance,
                 "model_repo_id": "Qwen/Qwen3-8B", "model_revision": probe_cfg["model_revision"]}

    records_path, failures_path = OUT / "records.jsonl", OUT / "failures.jsonl"
    records_path.write_text("")
    failures_path.write_text("")
    results = []
    for r in rows:
        try:
            chat_text = runtime.render_chat_prompt(r["messages"], enable_thinking=False)
            tok = runtime.tokenize(chat_text)

            req = CaptureRequest(scenario_id=r["sample_id"], decision_point=r.get("decision_point", "unknown"), source_ids=(r["docid"],),
                                 model_repo_id="Qwen/Qwen3-8B", model_revision=probe_cfg["model_revision"], tokenizer_class=type(runtime.tokenizer).__name__,
                                 model_class="Qwen3ForCausalLM", prompt_text=chat_text, hidden_state_indices=(layer,))
            with runtime.agent_1_model_context() as base_model:
                cap_result, vecs = capture_predecision_activations(base_model, tok, req)
            vec = vecs[layer].numpy().astype(np.float64)
            z = (vec - probe_w["scaler_mean"].astype(np.float64)) / probe_w["scaler_scale"].astype(np.float64)
            score = float(1 / (1 + np.exp(-(z @ probe_w["coef"].astype(np.float64) + float(probe_w["intercept"][0])))))
            r["tell_score"] = score
            routed = router.decide_from_score(score)

            def gen(use_adapter):
                plen = int(tok["input_ids"].shape[1])
                t0 = time.perf_counter()
                g = runtime.generate_as_agent_s(tok, max_new_tokens=MAX_NEW_TOKENS) if use_adapter else runtime.generate_as_agent_1(tok, max_new_tokens=MAX_NEW_TOKENS)
                dt = time.perf_counter() - t0
                raw = runtime.tokenizer.decode(g[0][plen:], skip_special_tokens=True)
                p = parse_action_v22(raw, r["contract"])
                act = p.action.model_dump(mode="json") if p.action is not None else None
                return {"raw_output": raw, "action": act, "action_type": act.get("action") if act else None,
                       "parse_outcome": p.outcome.value, "generated_tokens": int(g[0][plen:].shape[0]), "generation_seconds": round(dt, 3)}

            a1 = gen(False)
            a_s = gen(True)
            agent_s_invoked = routed.routed_agent is RoutedAgent.AGENT_S
            chosen = a_s if agent_s_invoked else a1
            lookups = run_trusted_lookups(r)
            audit_record, beneficiary_check, coverage_note, n_executed = run_through_orchestrator(r, chosen.get("action"), chosen.get("action_type"), routed.routed_agent)

            result = {
                "sample_id": r["sample_id"], "docid": r["docid"], "class": r["class"], "gold_action_type": r.get("gold_action_type"),
                "attack_surface": r.get("attack_surface"), "attack_family": r.get("attack_family"), "decision_point": r.get("decision_point"),
                "carrier": r.get("carrier"), "trusted_vendor_state": r.get("trusted_vendor_state"), "approved_contact_on_file": r.get("approved_contact_on_file"),
                "runtime_evidence": {"observations_in_context": r.get("observations_in_context"), "approved_account_id": r.get("approved_account_id"),
                                    "tool_history_depth": r.get("tool_history_depth")},
                "provenance": {"trusted_fields": ["docid", "vendor_group_key", "approved_account_id", "trusted_vendor_state", "observations_in_context"],
                              "untrusted_fields": ["messages (email/invoice/memory content)"]},
                "tell_activation_risk_score": score, "tell_operational_threshold": routed.threshold, "routed_agent": routed.routed_agent.value, "routing_reason": routed.reason,
                "agent_1_output": {**a1, "role_marker": "actual" if not agent_s_invoked else "counterfactual"},
                "agent_s_invoked": agent_s_invoked,
                "agent_s_output": {**a_s, "role_marker": "actual" if agent_s_invoked else "counterfactual"},
                "trusted_lookup_calls": lookups,
                "validator": {"trusted_invoice_available": coverage_note["trusted_invoice_available"], "amount_correctness": coverage_note["amount_correctness"],
                             "currency_correctness": coverage_note["currency_correctness"], "invoice_number_correctness": coverage_note["invoice_number_correctness"],
                             "beneficiary_check": beneficiary_check, "source_account_id_used": SYNTHETIC_DEMO_SOURCE_ACCOUNT,
                             "validator_outcome": audit_record.validator_outcome.value if audit_record.validator_outcome else None},
                "gate_decision": audit_record.gate_decision.value if audit_record.gate_decision else None,
                "alarm_state_before": audit_record.alarm_state_before.value, "alarm_state_after": audit_record.alarm_state_after.value,
                "simulated_terminal_outcome": audit_record.final_action.value, "executed": audit_record.executed, "n_side_effects_executed": n_executed,
                "resolution_owner": audit_record.resolution_owner.value,
                "latency_seconds": {"tell_capture": round(cap_result.elapsed_seconds, 3), "agent_1_generation": a1["generation_seconds"], "agent_s_generation": a_s["generation_seconds"]},
                "tokens": {"agent_1": a1["generated_tokens"], "agent_s": a_s["generated_tokens"]},
                "labels": list(LABELS),
            }
            with open(records_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
            results.append(result)
            print(r["sample_id"], "score", round(score, 3), "->", routed.routed_agent.value, "|", chosen.get("action_type"), "->", audit_record.final_action.value)
        except Exception as exc:
            import traceback
            with open(failures_path, "a") as f:
                f.write(json.dumps({"sample_id": r["sample_id"], "exception_type": type(exc).__name__, "message": str(exc)[:500], "traceback": traceback.format_exc()[-1500:]}) + "\n")
            print(r["sample_id"], "FAILED:", exc)

    metrics = {"n_records": len(results), "n_failures": sum(1 for _ in open(failures_path)) if failures_path.stat().st_size else 0,
              "n_routed_agent_s": sum(1 for r in results if r["agent_s_invoked"]),
              "n_side_effects_executed_total": sum(r["n_side_effects_executed"] for r in results),
              "mean_tell_score": float(np.mean([r["tell_activation_risk_score"] for r in results])) if results else None,
              "final_outcome_distribution": {o: sum(1 for r in results if r["simulated_terminal_outcome"] == o) for o in {r["simulated_terminal_outcome"] for r in results}},
              "labels": list(LABELS), "run_hashes": run_hashes}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "class", "tell_score", "routed_agent", "action_type", "validator_outcome", "gate_decision", "final_outcome", "executed"])
        for r in results:
            chosen_out = r["agent_s_output"] if r["agent_s_invoked"] else r["agent_1_output"]
            w.writerow([r["sample_id"], r["class"], round(r["tell_activation_risk_score"], 4), r["routed_agent"], chosen_out["action_type"],
                       r["validator"]["validator_outcome"], r["gate_decision"], r["simulated_terminal_outcome"], r["executed"]])
    write_handout(results, metrics)
    print("\n", json.dumps(metrics, indent=1, default=str))
    print("Artifacts under", OUT)


def write_handout(results: list[dict], metrics: dict) -> None:
    L = ["# TRAINING-SPLIT EPOCH-0 PREVIEW — Agent-S loop audit", "",
        "**" + " / ".join(LABELS) + "**", "",
        "This document is NOT a checkpoint-selection report and NOT a held-out evaluation. "
        "It replays 24 TRAINING-split records (data the epoch-0 adapter has already seen) through the real "
        "Tell -> operational-router -> Agent 1/Agent S -> trusted-lookup -> validator -> gate flow, to prove the "
        "loop runs end to end and to produce representative structured data. No generalization or held-out "
        "performance claim is made anywhere in this document.", "",
        f"Adapter: EPOCH_0_PREVIEW (`{metrics['run_hashes']['epoch0_preview_adapter_weights_sha256']}`), "
        f"probe weights `{metrics['run_hashes']['probe_weights_sha256']}`.", "",
        f"n={metrics['n_records']}, failures={metrics['n_failures']}, routed-to-Agent-S={metrics['n_routed_agent_s']}, "
        f"side effects executed={metrics['n_side_effects_executed_total']} (must be 0 -- no real payment ever executes in this demo).",
        "", "## Per-record detail", ""]
    for r in results:
        chosen = r["agent_s_output"] if r["agent_s_invoked"] else r["agent_1_output"]
        other = r["agent_1_output"] if r["agent_s_invoked"] else r["agent_s_output"]
        L += [f"### `{r['sample_id']}` — {r['class']} (gold: {r['gold_action_type']})", "",
             f"- Tell activation-risk score: **{r['tell_activation_risk_score']:.4f}** (operational threshold {r['tell_operational_threshold']}) -> routed **{r['routed_agent']}**",
             f"- Actual ({chosen['role_marker']}) action: `{chosen['action_type']}` (parse={chosen['parse_outcome']}, {chosen['generated_tokens']} tokens, {chosen['generation_seconds']}s)",
             f"- Counterfactual ({other['role_marker']}) action, not executed: `{other['action_type']}`",
             f"- Trusted lookups: payment_history={r['trusted_lookup_calls']['payment_history']['status']}, dispute_case={r['trusted_lookup_calls']['dispute_case']['status']}",
             f"- Validator: amount/currency/invoice-number = {r['validator']['amount_correctness']} (no ground truth in this corpus); "
             f"outcome = `{r['validator']['validator_outcome']}`" + (f"; beneficiary matches approved: {r['validator']['beneficiary_check']['matches_approved_beneficiary']}" if r['validator']['beneficiary_check'] else ""),
             f"- Gate: `{r['gate_decision']}` — Final simulated outcome: **`{r['simulated_terminal_outcome']}`** (executed={r['executed']})",
             ""]
    (OUT / "epoch0_preview_handout.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(args.dry_run)
