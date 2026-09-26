"""Integration tests for the clean payment-phase demonstration
(scripts/run_clean_payment_phase.py): candidate -> intent -> gate ->
execute, end to end, against a real (isolated, tmp_path) ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from run_clean_payment_phase import TRACE_PATH, run

from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, ScenarioBundle, load_scenario
from tell.payment.ledger import Ledger

FORBIDDEN_EVALUATION_MARKERS = [
    "is_attacked", "attack_surface", "expected_safe_action",
    "expected_beneficiary_account_id", "expected_amount",
    "expected_currency", "expected_outcome", "evaluation_labels", "probe_label",
]


@pytest.fixture(scope="module")
def bundle() -> ScenarioBundle:
    return load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)


def test_full_sequence_produces_executed_payment(bundle: ScenarioBundle, tmp_path: Path):
    db_path = tmp_path / "phase.sqlite"
    records = run(bundle, db_path)

    by_type = {r["record_type"]: r for r in records}
    assert by_type["intent_created"]["intent"]["status"] == "proposed"
    assert by_type["gate_evaluated"]["decision"]["decision"] == "permit"
    assert by_type["intent_executed"]["intent"]["status"] == "executed"
    assert by_type["final_status"]["invoice_status"] == "paid"
    assert by_type["final_status"]["intent_status"] == "executed"

    before = by_type["balances_before"]["balances_minor_units"]
    after = by_type["balances_after"]["balances_minor_units"]
    company_id = bundle.trusted_state.company_account_id
    supplier_id = bundle.trusted_state.supplier_account_id
    seed = bundle.trusted_state.ledger_seed

    assert after[company_id] == before[company_id] - seed.invoice_amount_minor_units
    assert after[supplier_id] == before[supplier_id] + seed.invoice_amount_minor_units


def test_candidate_beneficiary_comes_from_vendor_record_not_invoice(bundle: ScenarioBundle, tmp_path: Path):
    records = run(bundle, tmp_path / "phase2.sqlite")
    by_type = {r["record_type"]: r for r in records}
    candidate = by_type["candidate_constructed"]["candidate"]
    assert candidate["beneficiary_account_id"] == bundle.trusted_state.approved_beneficiary_account_id


def test_journal_entries_in_trace_balance(bundle: ScenarioBundle, tmp_path: Path):
    records = run(bundle, tmp_path / "phase3.sqlite")
    by_type = {r["record_type"]: r for r in records}
    entries = by_type["journal_entries"]["entries"]
    assert len(entries) == 2
    debit = next(e for e in entries if e["direction"] == "debit")
    credit = next(e for e in entries if e["direction"] == "credit")
    assert debit["amount_minor_units"] == credit["amount_minor_units"]
    assert debit["currency"] == credit["currency"]


def test_no_evaluation_only_fields_enter_the_trace(bundle: ScenarioBundle, tmp_path: Path):
    records = run(bundle, tmp_path / "phase4.sqlite")
    dumped = json.dumps(records)
    for marker in FORBIDDEN_EVALUATION_MARKERS:
        assert marker not in dumped, f"payment-phase trace leaked {marker}"


def test_every_step_recorded_in_audit_events_within_trace(bundle: ScenarioBundle, tmp_path: Path):
    records = run(bundle, tmp_path / "phase5.sqlite")
    by_type = {r["record_type"]: r for r in records}
    audit_event_types = [e["event_type"] for e in by_type["audit_events"]["events"]]
    assert "payment_intent_created" in audit_event_types
    assert "gate_decision_recorded" in audit_event_types
    assert "payment_executed" in audit_event_types


def test_ledger_db_independently_shows_correct_final_state(bundle: ScenarioBundle, tmp_path: Path):
    db_path = tmp_path / "phase6.sqlite"
    run(bundle, db_path)
    # Re-open the database fresh (new connection) to confirm the on-disk
    # state, not just the in-memory trace, is correct.
    ledger = Ledger(db_path)
    invoice = ledger.get_invoice(bundle.created_from_docid)
    assert invoice.status.value == "paid"
    company_balance = ledger.get_balance(bundle.trusted_state.company_account_id)
    supplier_balance = ledger.get_balance(bundle.trusted_state.supplier_account_id)
    seed = bundle.trusted_state.ledger_seed
    assert company_balance == seed.company_opening_balance_minor_units - seed.invoice_amount_minor_units
    assert supplier_balance == seed.supplier_opening_balance_minor_units + seed.invoice_amount_minor_units
    ledger.close()


def test_script_generates_the_real_artifacts_at_the_fixed_paths():
    """The actual demonstration run (not a tmp_path test double) must have
    produced these two exact artifacts; run scripts/run_clean_payment_phase.py
    before running this test."""
    from run_clean_payment_phase import DB_PATH

    assert DB_PATH.exists()
    assert TRACE_PATH.exists()
    with TRACE_PATH.open() as f:
        lines = [json.loads(line) for line in f]
    assert lines[0]["record_type"] == "run_header"
    assert lines[-1]["record_type"] == "run_footer"
    dumped = json.dumps(lines)
    for marker in FORBIDDEN_EVALUATION_MARKERS:
        assert marker not in dumped
