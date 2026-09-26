"""Deterministic clean payment demonstration for clean_04d531ca_v1.

Sequence: read_email -> read_invoice -> get_vendor_record -> construct a
PayInvoiceCandidate -> create_payment_intent -> evaluate_gate(CLEAR) ->
record_gate_decision -> execute_intent. Displays balances before/after,
the balanced journal entries, and final invoice/intent status. Writes a
structured JSONL trace.

This script is the only thing in this slice that performs the executor
step, and it does so strictly through the ledger's status machine
(create -> permitted -> executed); there is no shortcut that moves money
without an intervening gate decision.

Generated artifacts (this script touches nothing else):
  - results/runtime/clean_04d531ca.sqlite
  - results/traces/clean_04d531ca_payment_phase.jsonl
On each run, only these two paths (plus SQLite's own -wal/-shm/-journal
sidecar files for the same stem) are removed and recreated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tell.agent.tools import (
    GetVendorRecordArgs,
    PayInvoiceCandidate,
    ReadEmailArgs,
    ReadInvoiceArgs,
    get_vendor_record,
    read_email,
    read_invoice,
)
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, ScenarioBundle, load_scenario
from tell.payment.ledger import Ledger
from tell.safety.gate import AlarmState, evaluate_gate

DB_PATH = Path("/home/hp5/tell/results/runtime/clean_04d531ca.sqlite")
TRACE_PATH = Path("/home/hp5/tell/results/traces/clean_04d531ca_payment_phase.jsonl")

INTENT_ID = "INTENT-CLEAN-04D531CA-0001"


def _reset_sqlite_artifact(db_path: Path) -> None:
    """Removes exactly db_path and its SQLite sidecar files, nothing else."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()


def run(bundle: ScenarioBundle, db_path: Path) -> list[dict]:
    """Runs the full candidate -> intent -> gate -> execute sequence
    against a fresh ledger at db_path. Returns the ordered trace records.
    Side-effecting by necessity (writes db_path); every record is still
    built from deterministic scenario/ledger data, no randomness."""
    records: list[dict] = []
    records.append(
        {
            "record_type": "run_header",
            "scenario_id": bundle.scenario_id,
            "created_from_docid": bundle.created_from_docid,
        }
    )

    # --- read phase (steps 4-6) ---
    message_id = bundle.untrusted_inputs.supplier_email.message_id
    email_result = read_email(bundle, ReadEmailArgs(message_id=message_id))
    records.append({"record_type": "tool_call", "step": 1, "tool_name": "read_email", "result": email_result.model_dump(mode="json")})

    document_id = bundle.untrusted_inputs.invoice_document.docid
    invoice_result = read_invoice(bundle, ReadInvoiceArgs(document_id=document_id))
    records.append({"record_type": "tool_call", "step": 2, "tool_name": "read_invoice", "result": invoice_result.model_dump(mode="json")})

    vendor_id = bundle.trusted_state.canonical_vendor_id
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=vendor_id))
    records.append({"record_type": "tool_call", "step": 3, "tool_name": "get_vendor_record", "result": vendor_result.model_dump(mode="json")})

    # --- ledger setup ---
    seed = bundle.trusted_state.ledger_seed
    ledger = Ledger(db_path)
    ledger.init_schema()
    ledger.seed_accounts_and_invoice(
        company_account_id=bundle.trusted_state.company_account_id,
        supplier_account_id=bundle.trusted_state.supplier_account_id,
        currency=seed.currency,
        company_opening_balance_minor_units=seed.company_opening_balance_minor_units,
        supplier_opening_balance_minor_units=seed.supplier_opening_balance_minor_units,
        invoice_id=document_id,
        invoice_amount_minor_units=seed.invoice_amount_minor_units,
        invoice_status=seed.invoice_payment_status,
    )
    records.append(
        {
            "record_type": "ledger_seeded",
            "company_account_id": bundle.trusted_state.company_account_id,
            "supplier_account_id": bundle.trusted_state.supplier_account_id,
            "currency": seed.currency,
            "company_opening_balance_minor_units": seed.company_opening_balance_minor_units,
            "supplier_opening_balance_minor_units": seed.supplier_opening_balance_minor_units,
            "invoice_amount_minor_units": seed.invoice_amount_minor_units,
        }
    )

    balances_before = {
        bundle.trusted_state.company_account_id: ledger.get_balance(bundle.trusted_state.company_account_id),
        bundle.trusted_state.supplier_account_id: ledger.get_balance(bundle.trusted_state.supplier_account_id),
    }
    records.append({"record_type": "balances_before", "balances_minor_units": balances_before})

    # --- step 7: deterministically construct the candidate. Amount comes
    # from the invoice observation (converted to minor units via the
    # scenario's own ledger_seed, not re-parsed from money text here);
    # beneficiary comes from the trusted vendor record, never from the
    # invoice or email. ---
    candidate = PayInvoiceCandidate(
        invoice_id=invoice_result.content.docid,
        source_account_id=bundle.trusted_state.company_account_id,
        beneficiary_account_id=vendor_result.content.beneficiary_account_id,
        amount_minor_units=seed.invoice_amount_minor_units,
        currency=seed.currency,
        reason=f"Clean invoice payment for {invoice_result.content.invoice_number} per verified vendor record",
    )
    records.append({"record_type": "candidate_constructed", "candidate": candidate.model_dump(mode="json")})

    # --- step 8: create payment intent (no money moves) ---
    intent = ledger.create_payment_intent(
        intent_id=INTENT_ID,
        invoice_id=candidate.invoice_id,
        source_account_id=candidate.source_account_id,
        beneficiary_account_id=candidate.beneficiary_account_id,
        amount_minor_units=candidate.amount_minor_units,
        currency=candidate.currency,
        reason=candidate.reason,
    )
    records.append({"record_type": "intent_created", "intent": intent.model_dump(mode="json")})
    records.append(
        {
            "record_type": "balances_after_intent_created",
            "balances_minor_units": {
                bundle.trusted_state.company_account_id: ledger.get_balance(bundle.trusted_state.company_account_id),
                bundle.trusted_state.supplier_account_id: ledger.get_balance(bundle.trusted_state.supplier_account_id),
            },
            "note": "unchanged -- creating an intent never moves money",
        }
    )

    # --- step 9: evaluate the gate with AlarmState.clear ---
    decision = evaluate_gate(candidate, AlarmState.CLEAR)
    records.append({"record_type": "gate_evaluated", "decision": decision.model_dump(mode="json")})

    ledger.record_gate_decision(
        intent_id=INTENT_ID,
        alarm_state=decision.alarm_state.value,
        decision=decision.decision.value,
        reason=decision.reason_message,
    )
    records.append({"record_type": "gate_decision_recorded", "intent_id": INTENT_ID, "decision": decision.decision.value})

    # --- step 10: execute the permitted simulated payment ---
    executed_intent = ledger.execute_intent(INTENT_ID)
    records.append({"record_type": "intent_executed", "intent": executed_intent.model_dump(mode="json")})

    # --- steps 11-13: balances, journal, statuses ---
    balances_after = {
        bundle.trusted_state.company_account_id: ledger.get_balance(bundle.trusted_state.company_account_id),
        bundle.trusted_state.supplier_account_id: ledger.get_balance(bundle.trusted_state.supplier_account_id),
    }
    records.append({"record_type": "balances_after", "balances_minor_units": balances_after})

    journal_entries = [je.model_dump(mode="json") for je in ledger.list_journal_entries(INTENT_ID)]
    records.append({"record_type": "journal_entries", "entries": journal_entries})

    final_invoice = ledger.get_invoice(candidate.invoice_id)
    records.append(
        {
            "record_type": "final_status",
            "invoice_status": final_invoice.status.value,
            "intent_status": executed_intent.status.value,
        }
    )

    audit_events = [ae.model_dump(mode="json") for ae in ledger.list_audit_events()]
    records.append({"record_type": "audit_events", "events": audit_events})

    records.append(
        {
            "record_type": "run_footer",
            "payment_tool_called": False,
            "note": "No pay_invoice tool exists; the executor step ran directly through Ledger.execute_intent after a recorded gate permit.",
        }
    )

    ledger.close()
    return records


def _write_trace(records: list[dict]) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _print_summary(records: list[dict]) -> None:
    by_type = {r["record_type"]: r for r in records if r["record_type"] not in ("tool_call",)}
    tool_calls = [r for r in records if r["record_type"] == "tool_call"]

    print("=== Tell clean payment-phase demonstration ===")
    print(f"Scenario: {records[0]['scenario_id']} (DocILE document {records[0]['created_from_docid']})")
    print()

    for tc in tool_calls:
        print(f"[{tc['step']}] {tc['tool_name']}: {tc['result']['status']}")
    print()

    print("Balances before:")
    for acct, bal in by_type["balances_before"]["balances_minor_units"].items():
        print(f"  {acct}: {bal:,} minor units")
    print()

    candidate = by_type["candidate_constructed"]["candidate"]
    print(f"Candidate pay_invoice: {candidate['amount_minor_units']:,} {candidate['currency']} "
          f"from {candidate['source_account_id']} to {candidate['beneficiary_account_id']}")
    print(f"Intent created: {by_type['intent_created']['intent']['status']}")

    decision = by_type["gate_evaluated"]["decision"]
    print(f"Gate decision: {decision['decision']} ({decision['reason_code']}) -- alarm_state={decision['alarm_state']}")
    print(f"Intent executed: {by_type['intent_executed']['intent']['status']}")
    print()

    print("Balances after:")
    for acct, bal in by_type["balances_after"]["balances_minor_units"].items():
        print(f"  {acct}: {bal:,} minor units")
    print()

    print("Journal entries:")
    for je in by_type["journal_entries"]["entries"]:
        print(f"  {je['direction']:>6}  {je['account_id']}  {je['amount_minor_units']:,} {je['currency']}")
    print()

    final = by_type["final_status"]
    print(f"Invoice status: {final['invoice_status']}  |  Intent status: {final['intent_status']}")
    print()
    print(f"Trace written to {TRACE_PATH}")
    print(f"Ledger written to {DB_PATH}")


def main() -> None:
    bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _reset_sqlite_artifact(DB_PATH)

    records = run(bundle, DB_PATH)
    records[0]["run_started_at"] = datetime.now(timezone.utc).isoformat()

    _write_trace(records)
    _print_summary(records)


if __name__ == "__main__":
    main()
