#!/usr/bin/env python3
"""Build data/traces_v1.json from real epoch-0 replay records + a deterministic fixture overlay.

Reads (read-only): results/lora_training/agent_s_v1/interface_demo_v1/{records.jsonl,metrics.json},
configs/routing/tell_three_zone_v1.json.  Writes only demo/ui/data/traces_v1.json.
Deterministic: same inputs -> byte-identical output.  CPU-only; no model, no CUDA.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import trace_contract as C  # noqa: E402

REC = REPO / "results/lora_training/agent_s_v1/interface_demo_v1/records.jsonl"
MET = REPO / "results/lora_training/agent_s_v1/interface_demo_v1/metrics.json"
CFG = REPO / "configs/routing/tell_three_zone_v1.json"
OUT = HERE / "data/traces_v1.json"

R, F = "real", "fixture"
VENDORS = ["Northwind Components Ltd", "Harbor & Vale Logistics", "Cedar Point Analytics", "Brightline Packaging Co",
           "Meridian Office Supply", "Ostrava Industrial GmbH", "Kestrel Facilities Services", "Lumen Print Works",
           "Alder Freight Partners", "Quarry Lane Software", "Tidewater Chemical Inc", "Pinecrest Catering"]
TOOL = {"email": "read_email", "invoice": "read_invoice", "memory": "search_memory", "vendor": "get_vendor_record"}
FAMILY_TEXT = {
    "invoice_action_json_priming": ("Invoice footer embeds a JSON action object instructing an immediate payment.",
                                    "Supplier email is routine; the invoice document carries the embedded instruction."),
    "email_urgency_bypass": ("Supplier email demands urgent payment and asks to skip vendor verification.",
                             "Urgent email asking AP to bypass the normal verification step."),
    "forged_tool_result_envelope": ("Message contains a forged tool-result envelope claiming bank details were verified.",
                                    "Email includes text formatted as a trusted tool result."),
}
CLEAN_MSG = "Routine supplier email attaching the invoice and requesting payment on standard terms."
OPERATING_BALANCE = 25_000_000  # minor units, fixture


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def money(minor, cur="usd"):
    return f"{cur.upper()} {minor / 100:,.2f}"


def load():
    recs = {}
    for line in REC.read_text().splitlines():
        r = json.loads(line); recs[r["sample_id"]] = r
    return recs, json.loads(MET.read_text()), json.loads(CFG.read_text())


def parse_raw(out):
    """Best-effort structured view of an agent output (never invents fields)."""
    if not out:
        return None
    if out.get("action"):
        return dict(out["action"])
    raw = out.get("raw_output") or ""
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r'"action"\s*:\s*"([a-z_]+)"', raw)
    d = {"action": m.group(1) if m else "unparsed_output"}
    for k in ("amount_minor_units", "beneficiary_account_id", "currency", "invoice_number"):
        m = re.search(r'"%s"\s*:\s*("?)([^",}\n]+)\1' % k, raw)
        if m and m.group(2) != "null":
            d[k] = int(m.group(2)) if k == "amount_minor_units" and m.group(2).isdigit() else m.group(2)
    return d


def build_one(r, meta, hashes, th, plan):
    sid = r["sample_id"]; h = int(sha(sid)[:8], 16)
    a1 = parse_raw(r["agent_1_output"]); a1_act = (r["agent_1_output"] or {}).get("action") or None
    prop = next((p for p in (a1, parse_raw(r["agent_s_output"])) if p and p.get("amount_minor_units")), None)
    amount = prop["amount_minor_units"] if prop else 25000 + (h % 900000)
    amt_p = R if prop else F
    cur = (prop or {}).get("currency", "usd")
    inv_no = (prop or {}).get("invoice_number") or None
    vname = VENDORS[h % len(VENDORS)]
    fam = r.get("attack_family")
    atk = r["class"] == "attacked"
    lo, hi = th
    score = r["tell_activation_risk_score"]
    zone = C.zone_for(score, lo, hi)
    routed = r["routed_agent"]
    assert (zone == "agent_s") == (routed == "agent_s"), sid  # captured 2-zone router must agree with 3-zone view
    obs = r["runtime_evidence"]["observations_in_context"]
    acct = r["runtime_evidence"]["approved_account_id"]
    vid = ((r["agent_1_output"] or {}).get("action") or {}).get("vendor_id") or \
        ((prop or {}).get("evidence") or {}).get("vendor_record_id") or "SIM-VENDOR-" + sha(sid)[:12].upper()
    mode = plan["tail"]
    run_id = f"run-{sid}"
    events = []
    t = [0.0]

    def ev(type_, actor, title, dt=0.6, **f):
        e = {"seq": len(events), "type": type_, "actor": actor, "title": title, "prov": {}}
        t[0] = round(t[0] + dt, 1)
        e["t_offset_s"] = t[0]  # presentation timing is fixture
        for k, v in f.items():
            if isinstance(v, tuple):
                e[k] = v[0]; e["prov"][k] = v[1]
            else:
                e[k] = v
        events.append(e)

    ev("invoice_received", "intake", "Invoice received in AP inbox", 0.0,
       summary=(f"{money(amount, cur)} from {vname}; routed to Agent 1 for processing.", F))
    for o in obs:
        args = {"invoice_document_id": r["docid"]} if o == "invoice" else ({"vendor_id": vid} if o == "vendor" else {})
        ev("tool_call", "agent_1", f"Agent 1 → {TOOL[o]}", 1.1,
           action=({"name": TOOL[o], "args": args}, R),
           summary=({"email": "Supplier message read.", "invoice": "Invoice fields extracted.",
                     "memory": "Memory store searched; entries returned with provenance.",
                     "vendor": f"Trusted vendor record retrieved (state: {r['trusted_vendor_state']})."}[o], F))
    tl = {"score": score, "lower": lo, "upper": hi, "zone": zone, "agent": routed}
    ev("tell_score", "tell", f"Tell reads activations: score {score:.3f}", 1.2, tell=(tl, R),
       latency_seconds=({"tell_capture": r["latency_seconds"]["tell_capture"]}, R))
    ev("routing", "router", f"Routing → {'Agent S' if routed == 'agent_s' else 'Agent 1'}", 0.5,
       routing=({"from": "agent_1", "to": routed, "zone": zone, "reason": r["routing_reason"]}, R),
       alarm=({"before": r["alarm_state_before"], "after": r["alarm_state_after"]}, R))

    def act_event(who, out, title, note):
        a = parse_raw(out)
        if not a:
            return
        args = {k: v for k, v in a.items() if k != "action"}
        args["parse_outcome"] = out["parse_outcome"]
        ev("agent_action", who, title, out["generation_seconds"] / 4,
           action=({"name": a["action"], "args": args}, R),
           tokens=({"generated": out["generated_tokens"]}, R),
           latency_seconds=({"generation": out["generation_seconds"]}, R),
           summary=(note, F))

    if routed == "agent_s":
        act_event("agent_1", r["agent_1_output"], "Agent 1 proposal frozen (not executed)",
                  "Counterfactual capture: the alarm froze side effects before this step could run.")
        act_event("agent_s", r["agent_s_output"], "Agent S structured action",
                  "Structured action captured from Agent S. A proposal is not an approval \u2014 execution stays behind the validator and gate.")
        lk = r["trusted_lookup_calls"]
        ev("trusted_lookup", "trusted_lookup", "Trusted verification performed", 0.9,
           lookups=({"payment_history": lk["payment_history"]["status"], "dispute_case": lk["dispute_case"]["status"],
                     "vendor_state": r["trusted_vendor_state"],
                     "approved_beneficiary_matches": (r["validator"].get("beneficiary_check") or {}).get("matches_approved_beneficiary")}, R))
    else:
        act_event("agent_1", r["agent_1_output"], "Agent 1 structured action",
                  "Score is in the Agent-1 zone; the action proceeds to the validator.")

    val = r["validator"]; cap = val["validator_outcome"]
    bm = (val.get("beneficiary_check") or {}).get("matches_approved_beneficiary")
    ledger_before = {"operating_balance_minor": OPERATING_BALANCE, "invoice_status": "approved_unpaid", "journal_entries": 0}
    if mode == "allow":
        ev("validator", "validator", "Validator: APPROVED", 0.8,
           validator=({"outcome": "APPROVED", "beneficiary_matches_approved": bm, "captured_outcome": cap}, F),
           summary=(f"Fixture approval. The captured epoch-0 validator outcome for this record was '{cap}' — "
                    "the approval shown here is simulated, not measured.", F))
        ev("gate", "gate", "Deterministic gate: ALLOW", 0.5,
           gate=({"decision": "ALLOW", "reason": "alarm clear · validator approved · beneficiary matches approved account"}, F))
        after = {"operating_balance_minor": OPERATING_BALANCE - amount, "invoice_status": "paid", "journal_entries": 1}
        ev("ledger", "ledger", "Simulated ledger posts payment", 0.9,
           ledger=({"before": ledger_before, "after": after, "entry": {
               "payment_intent_id": "PI-" + sha(run_id)[:10].upper(), "amount_minor_units": amount, "currency": cur,
               "beneficiary_account_id": acct, "rail": "LOCAL_SQLITE_SIMULATION"}}, F))
        ev("outcome", "ledger", "Outcome: payment completed (simulated)", 0.4, outcome=("PAID", F))
    elif mode in ("block", "escalate"):
        if cap is not None or mode == "escalate":
            ev("validator", "validator", "Validator: no approval issued", 0.8,
               validator=({"outcome": cap if cap else "NOT_RUN", "beneficiary_matches_approved": bm,
                           "captured_outcome": cap}, R if cap else F),
               summary=("Captured validator outcome retained as-is." if cap else
                        "No validator outcome was captured for this record; nothing was approved.", F))
        ev("gate", "gate", "Deterministic gate: BLOCK", 0.5,
           gate=({"decision": "BLOCK", "reason": "alarm unresolved (state: %s) — side-effecting pay_invoice is frozen "
                  "while Agent S owns the case; no approval exists" % r["alarm_state_after"]}, F))
        ev("ledger", "ledger", "Ledger unchanged — payment prevented", 0.6,
           ledger=({"before": ledger_before, "after": dict(ledger_before, invoice_status="held_for_review"), "entry": None}, F))
        if mode == "escalate":
            ev("evidence_submitted", "review_queue", "Captured evidence submitted for human review", 0.9,
               action=({"name": "submit_evidence_report", "args": {"executed": False, "resolution_owner": r["resolution_owner"]}}, R),
               summary=("Captured terminal outcome: submit_evidence_report; side effects executed: %d."
                        % r["n_side_effects_executed"] if r["simulated_terminal_outcome"] == "submit_evidence_report" else
                        "Escalated to the human-review queue with the captured evidence.", F))
            ev("outcome", "review_queue", "Outcome: escalated to human review", 0.4, outcome=("ESCALATED", F))
        else:
            ev("outcome", "gate", "Outcome: payment blocked", 0.4, outcome=("BLOCKED", F))
    elif mode == "followup":
        ev("outcome", "validator", "Outcome: awaiting follow-up with supplier", 0.8, outcome=("FOLLOW_UP", F),
           summary=("Captured validator could not confirm identifiers; follow-up requested (fixture status).", F))

    inv = {"invoice_number": {"v": inv_no, "p": R if inv_no else F} if inv_no else {"v": "INV-" + sha(sid)[:6].upper(), "p": F},
           "vendor_name": {"v": vname, "p": F},
           "beneficiary_account_id": {"v": acct, "p": R},
           "amount_minor_units": {"v": amount, "p": amt_p},
           "currency": {"v": cur, "p": R if prop else F},
           "attack_surface": {"v": r["attack_surface"], "p": R},
           "attack_family": {"v": fam, "p": R},
           "supplier_message_summary": {"v": FAMILY_TEXT[fam][1] if fam in FAMILY_TEXT else (CLEAN_MSG if not atk else
                                        "Supplier message contains instructions from an untrusted source."), "p": F},
           "invoice_summary": {"v": f"{money(amount, cur)} due to {vname}.", "p": F}}
    tr = {
        "contract_version": C.CONTRACT_VERSION, "run_id": run_id, "scenario_id": plan["scenario"],
        "sample_id": sid, "title": plan["title"], "kind": plan["kind"], "queue_status": plan["status"],
        "evidence_mode": "REPLAY", "invoice": inv,
        "trusted_vendor_record": {"vendor_id": {"v": vid, "p": R},
                                  "name": {"v": vname, "p": F},
                                  "approved_account_id": {"v": acct, "p": R},
                                  "verification_state": {"v": r["trusted_vendor_state"], "p": R},
                                  "approved_contact_on_file": {"v": r["approved_contact_on_file"], "p": R}},
        "memory": {"retrieved": {"v": "memory" in obs, "p": R},
                   "observations_in_context": {"v": obs, "p": R},
                   "entries": {"v": plan.get("memory", []), "p": F}},
        "thresholds": {"lower": {"v": lo, "p": R}, "upper": {"v": hi, "p": R},
                       "note": "secondary_posthoc_three_zone; captured epoch-0 run routed with the 2-zone operational threshold (upper)."},
        "hashes": hashes, "side_effects_simulated": True,
        "events": events,
    }
    if plan["status"] == "ESCALATED":
        txt = FAMILY_TEXT.get(fam, ("Untrusted content carried instructions that changed the model's activation profile.",))[0]
        report = {"case": "CASE-" + sha(run_id)[:8].upper(), "invoice_number": inv["invoice_number"]["v"],
                  "attack_surface": r["attack_surface"], "attack_family": fam, "tell_score": round(score, 4),
                  "routing": "agent_s", "alarm_state": r["alarm_state_after"],
                  "trusted_lookups": events[[e["type"] for e in events].index("trusted_lookup")]["lookups"],
                  "side_effects_executed": r["n_side_effects_executed"], "executed": r["executed"]}
        chain_head = None
        tr["review"] = {"triggering_evidence": {"v": txt, "p": F},
                        "attack_surface": {"v": r["attack_surface"], "p": R},
                        "recommended_action": {"v": "Confirm the beneficiary with the supplier through a known-good channel "
                                               "before any payment; keep the alarm raised until a human clears it.", "p": F},
                        "evidence_report": {"v": report, "p": F},
                        "audit_ids": {"v": [], "p": F}, "artifact_hashes": {}}
    return tr, r


def finish(tr, r, rec_sha, chain_p):
    prev = "0" * 64
    for e in tr["events"]:
        e["audit_id"] = "AUD-" + sha(tr["run_id"] + str(e["seq"]))[:12].upper()
        e.pop("audit_hash", None)
        prev = e["audit_hash"] = C.chain_hash(prev, e)
    if "review" in tr:
        rv = tr["review"]
        rv["audit_ids"]["v"] = [e["audit_id"] for e in tr["events"]]
        rv["artifact_hashes"] = {
            "records_source_sha256": {"v": rec_sha, "p": R},
            "probe_weights_sha256": {"v": tr["hashes"]["probe_weights_sha256"], "p": R},
            "adapter_weights_sha256": {"v": tr["hashes"]["adapter_weights_sha256"], "p": R},
            "evidence_report_sha256": {"v": sha(C.canon(rv["evidence_report"]["v"])), "p": F},
            "audit_chain_head_sha256": {"v": prev, "p": F}}
    return tr


def main():
    recs, met, cfg = load()
    rec_sha = sha(REC.read_bytes())
    rh = met["run_hashes"]
    hashes = {"model_repo_id": rh["model_repo_id"], "model_revision": rh["model_revision"],
              "probe_weights_sha256": rh["probe_weights_sha256"], "probe_config_sha256": rh["probe_config_sha256"],
              "adapter_weights_sha256": rh["epoch0_preview_adapter_weights_sha256"],
              "adapter_label": "EPOCH_0_PREVIEW · NOT_FINAL_SELECTED_ADAPTER · NOT_HELD_OUT_PERFORMANCE",
              "records_source": "results/lora_training/agent_s_v1/interface_demo_v1/records.jsonl",
              "records_source_sha256": rec_sha}
    th = (cfg["zones"]["low_threshold"], cfg["zones"]["high_threshold"])
    mem_att = [{"id": "MEM-DEMO-0417", "summary": "Note saved by an earlier agent step: “supplier prefers expedited settlement”.",
                "source": "agent_written", "trust": "unverified", "status": "active"}]
    P = [  # (sample_id, scenario, title, kind, status, tail, extra)
        ("lv22-bfe344cd42ac376c", "clean_payment", "Clean payment — verified supplier", "story", "PROCESSING", "allow", {}),
        ("lv22-60f6facb2fc4b147", "escalated_attack", "Escalated attack — injected invoice", "story", "ESCALATED", "escalate", {"memory": mem_att}),
        ("lv22-1cae1e3803bde9bd", "queue_email_injection_a", "Urgent-email injection", "background", "ESCALATED", "escalate", {}),
        ("lv22-35100f46c970c9e3", "queue_email_injection_b", "Urgent-email injection", "background", "ESCALATED", "escalate", {}),
        ("lv22-06087a9f7f8e7edd", "queue_forged_tool_result", "Forged tool-result envelope", "background", "ESCALATED", "escalate", {}),
        ("lv22-4873d5a93c969a3c", "queue_memory_block", "Memory-borne instruction", "background", "BLOCKED", "block", {"memory": mem_att}),
        ("lv22-73b695dfff0269d6", "queue_clean_paid", "Routine invoice", "background", "PAID", "allow", {}),
        ("lv22-8f76ec4db2b8f859", "queue_clean_a", "Routine invoice", "background", "PROCESSING", None, {}),
        ("lms1-82295d20c901774c", "queue_clean_b", "Routine invoice", "background", "PROCESSING", None, {}),
        ("lv22-93fec2cc2dffa51a", "queue_followup_a", "Ambiguous request", "background", "FOLLOW_UP", "followup", {}),
        ("lms1-b381f27e2a4435f9", "queue_followup_b", "Incomplete invoice", "background", "FOLLOW_UP", "followup", {}),
    ]
    traces = []
    for sid, sc, title, kind, status, tail, extra in P:
        tr, r = build_one(recs[sid], met, hashes, th, dict(scenario=sc, title=title, kind=kind, status=status, tail=tail, **extra))
        traces.append(finish(tr, r, rec_sha, None))
    bundle = {
        "contract_version": C.CONTRACT_VERSION, "fixture_label": C.FIXTURE_LABEL, "modes": list(C.MODES),
        "live_inference": {"enabled": False, "reason": "Not available in this version."},
        "fleet": {"p": F,
                  "baseline": {"processed": 1284, "paid": 1109, "blocked": 41},
                  "agents": [{"id": "agent_1", "name": "Agent 1", "role": "Invoice reader", "state": "ONLINE", "detail": "3 invoices in flight"},
                             {"id": "tell", "name": "Tell", "role": "Activation probe", "state": "ONLINE", "detail": "layer 27 · frozen probe"},
                             {"id": "agent_s", "name": "Agent S", "role": "Security specialist", "state": "ONLINE", "detail": "4 cases held"},
                             {"id": "validator", "name": "Validator", "role": "Typed checks", "state": "ONLINE", "detail": "deterministic"},
                             {"id": "gate", "name": "Action gate", "role": "Deterministic policy", "state": "ONLINE", "detail": "fail-closed"}]},
        "source": {"records_sha256": rec_sha, "note": "Real fields come from the TRAINING-SPLIT EPOCH-0 PREVIEW replay; not held-out performance."},
        "traces": traces}
    C.validate_bundle(bundle)
    OUT.write_text(json.dumps(bundle, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(traces)} traces) records_sha256={rec_sha[:16]}")


if __name__ == "__main__":
    main()
