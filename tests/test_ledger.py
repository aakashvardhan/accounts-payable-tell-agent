"""Tests for the simulated SQLite ledger (tell.payment.ledger)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tell.payment.ledger import (
    CurrencyMismatchError,
    DuplicateExecutionError,
    InsufficientFundsError,
    IntentNotPermittedError,
    InvalidAmountError,
    Ledger,
    UnknownAccountError,
    UnsupportedCurrencyError,
)
from tell.payment.models import IntentStatus, InvoiceStatus, JournalDirection

COMPANY = "SIM-COMPANY-ACCT-0001"
SUPPLIER = "SIM-BENEFICIARY-ACCT-MDSPHARMA-0001"
INVOICE_ID = "04d531ca811f448a91c6ff4e"
AMOUNT = 3_000_000
COMPANY_BALANCE = 10_000_000


@pytest.fixture()
def ledger(tmp_path: Path) -> Ledger:
    led = Ledger(tmp_path / "test_ledger.sqlite")
    led.init_schema()
    led.seed_accounts_and_invoice(
        company_account_id=COMPANY,
        supplier_account_id=SUPPLIER,
        currency="usd",
        company_opening_balance_minor_units=COMPANY_BALANCE,
        supplier_opening_balance_minor_units=0,
        invoice_id=INVOICE_ID,
        invoice_amount_minor_units=AMOUNT,
    )
    yield led
    led.close()


def _create_intent(ledger: Ledger, intent_id: str = "INTENT-1", amount: int = AMOUNT) -> None:
    ledger.create_payment_intent(
        intent_id=intent_id,
        invoice_id=INVOICE_ID,
        source_account_id=COMPANY,
        beneficiary_account_id=SUPPLIER,
        amount_minor_units=amount,
        currency="usd",
        reason="test payment",
    )


def test_schema_initialization_is_idempotent(tmp_path: Path):
    led = Ledger(tmp_path / "idempotent.sqlite")
    led.init_schema()
    led.init_schema()  # must not raise
    led.close()


def test_seeding_creates_accounts_and_invoice(ledger: Ledger):
    assert ledger.get_balance(COMPANY) == COMPANY_BALANCE
    assert ledger.get_balance(SUPPLIER) == 0
    invoice = ledger.get_invoice(INVOICE_ID)
    assert invoice.amount_minor_units == AMOUNT
    assert invoice.status == InvoiceStatus.UNPAID


# 1. Creating an intent does not move money.
def test_create_intent_does_not_move_money(ledger: Ledger):
    before_company = ledger.get_balance(COMPANY)
    before_supplier = ledger.get_balance(SUPPLIER)
    _create_intent(ledger)
    assert ledger.get_balance(COMPANY) == before_company
    assert ledger.get_balance(SUPPLIER) == before_supplier
    intent = ledger.get_intent("INTENT-1")
    assert intent.status == IntentStatus.PROPOSED


# 2. A clear-state permitted payment executes correctly.
def test_permitted_execution_debits_and_credits_correct_accounts(ledger: Ledger):
    _create_intent(ledger)
    ledger.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    result = ledger.execute_intent("INTENT-1")
    assert result.status == IntentStatus.EXECUTED
    assert ledger.get_balance(COMPANY) == COMPANY_BALANCE - AMOUNT
    assert ledger.get_balance(SUPPLIER) == AMOUNT


# 3. Blocked payment leaves balances and journal unchanged.
def test_blocked_intent_leaves_balances_and_journal_unchanged(ledger: Ledger):
    _create_intent(ledger)
    ledger.record_gate_decision(intent_id="INTENT-1", alarm_state="unresolved", decision="block", reason="alarm unresolved")
    with pytest.raises(IntentNotPermittedError):
        ledger.execute_intent("INTENT-1")
    assert ledger.get_balance(COMPANY) == COMPANY_BALANCE
    assert ledger.get_balance(SUPPLIER) == 0
    assert ledger.list_journal_entries("INTENT-1") == []
    assert ledger.get_invoice(INVOICE_ID).status == InvoiceStatus.UNPAID


# 4. Journal entries balance exactly.
def test_journal_entries_balance_exactly(ledger: Ledger):
    _create_intent(ledger)
    ledger.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    ledger.execute_intent("INTENT-1")
    entries = ledger.list_journal_entries("INTENT-1")
    assert len(entries) == 2
    debit = [e for e in entries if e.direction == JournalDirection.DEBIT]
    credit = [e for e in entries if e.direction == JournalDirection.CREDIT]
    assert len(debit) == 1 and len(credit) == 1
    assert debit[0].amount_minor_units == credit[0].amount_minor_units == AMOUNT
    assert debit[0].account_id == COMPANY
    assert credit[0].account_id == SUPPLIER
    assert debit[0].currency == credit[0].currency == "usd"


# 5. Invoice becomes paid, intent becomes executed.
def test_invoice_becomes_paid_and_intent_becomes_executed(ledger: Ledger):
    _create_intent(ledger)
    ledger.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    result = ledger.execute_intent("INTENT-1")
    assert result.status == IntentStatus.EXECUTED
    assert ledger.get_invoice(INVOICE_ID).status == InvoiceStatus.PAID


# 6. Insufficient funds rolls back fully.
def test_insufficient_funds_rolls_back_fully(tmp_path: Path):
    led = Ledger(tmp_path / "poor.sqlite")
    led.init_schema()
    led.seed_accounts_and_invoice(
        company_account_id=COMPANY,
        supplier_account_id=SUPPLIER,
        currency="usd",
        company_opening_balance_minor_units=1_000,  # far less than AMOUNT
        supplier_opening_balance_minor_units=0,
        invoice_id=INVOICE_ID,
        invoice_amount_minor_units=AMOUNT,
    )
    _create_intent(led)
    led.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    with pytest.raises(InsufficientFundsError):
        led.execute_intent("INTENT-1")
    assert led.get_balance(COMPANY) == 1_000
    assert led.get_balance(SUPPLIER) == 0
    assert led.list_journal_entries("INTENT-1") == []
    assert led.get_invoice(INVOICE_ID).status == InvoiceStatus.UNPAID
    assert led.get_intent("INTENT-1").status == IntentStatus.PERMITTED  # not silently marked executed
    led.close()


# 7. Duplicate execution is rejected.
def test_duplicate_execution_is_rejected(ledger: Ledger):
    _create_intent(ledger)
    ledger.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    ledger.execute_intent("INTENT-1")
    balance_after_first = ledger.get_balance(COMPANY)
    with pytest.raises(DuplicateExecutionError):
        ledger.execute_intent("INTENT-1")
    assert ledger.get_balance(COMPANY) == balance_after_first  # unchanged by the rejected duplicate
    assert len(ledger.list_journal_entries("INTENT-1")) == 2  # still exactly one pair


# 8. Currency mismatch is rejected.
def test_currency_mismatch_rejected_at_intent_creation(ledger: Ledger):
    with pytest.raises(CurrencyMismatchError):
        ledger.create_payment_intent(
            intent_id="INTENT-BAD-CURRENCY",
            invoice_id=INVOICE_ID,
            source_account_id=COMPANY,
            beneficiary_account_id=SUPPLIER,
            amount_minor_units=AMOUNT,
            currency="eur",  # invoice is usd
            reason="mismatched currency",
        )


def test_unsupported_currency_rejected(ledger: Ledger):
    with pytest.raises(UnsupportedCurrencyError):
        ledger.create_payment_intent(
            intent_id="INTENT-UNSUPPORTED",
            invoice_id=INVOICE_ID,
            source_account_id=COMPANY,
            beneficiary_account_id=SUPPLIER,
            amount_minor_units=AMOUNT,
            currency="xyz",
            reason="unsupported currency",
        )


# Reject zero/negative amounts.
def test_zero_and_negative_amounts_rejected(ledger: Ledger):
    with pytest.raises(InvalidAmountError):
        ledger.create_payment_intent(
            intent_id="INTENT-ZERO",
            invoice_id=INVOICE_ID,
            source_account_id=COMPANY,
            beneficiary_account_id=SUPPLIER,
            amount_minor_units=0,
            currency="usd",
            reason="zero amount",
        )
    with pytest.raises(InvalidAmountError):
        ledger.create_payment_intent(
            intent_id="INTENT-NEGATIVE",
            invoice_id=INVOICE_ID,
            source_account_id=COMPANY,
            beneficiary_account_id=SUPPLIER,
            amount_minor_units=-100,
            currency="usd",
            reason="negative amount",
        )


# Reject unknown accounts.
def test_unknown_accounts_rejected(ledger: Ledger):
    with pytest.raises(UnknownAccountError):
        ledger.create_payment_intent(
            intent_id="INTENT-UNKNOWN-SRC",
            invoice_id=INVOICE_ID,
            source_account_id="SIM-DOES-NOT-EXIST",
            beneficiary_account_id=SUPPLIER,
            amount_minor_units=AMOUNT,
            currency="usd",
            reason="unknown source",
        )
    with pytest.raises(UnknownAccountError):
        ledger.create_payment_intent(
            intent_id="INTENT-UNKNOWN-BEN",
            invoice_id=INVOICE_ID,
            source_account_id=COMPANY,
            beneficiary_account_id="SIM-DOES-NOT-EXIST",
            amount_minor_units=AMOUNT,
            currency="usd",
            reason="unknown beneficiary",
        )


# Every proposal, gate decision, and execution is audited.
def test_every_step_is_audited(ledger: Ledger):
    _create_intent(ledger)
    ledger.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    ledger.execute_intent("INTENT-1")
    event_types = [e.event_type for e in ledger.list_audit_events()]
    assert "ledger_seeded" in event_types
    assert "payment_intent_created" in event_types
    assert "gate_decision_recorded" in event_types
    assert "payment_executed" in event_types


def test_execution_failure_is_audited(tmp_path: Path):
    led = Ledger(tmp_path / "audited_failure.sqlite")
    led.init_schema()
    led.seed_accounts_and_invoice(
        company_account_id=COMPANY,
        supplier_account_id=SUPPLIER,
        currency="usd",
        company_opening_balance_minor_units=0,
        supplier_opening_balance_minor_units=0,
        invoice_id=INVOICE_ID,
        invoice_amount_minor_units=AMOUNT,
    )
    _create_intent(led)
    led.record_gate_decision(intent_id="INTENT-1", alarm_state="clear", decision="permit", reason="clean")
    with pytest.raises(InsufficientFundsError):
        led.execute_intent("INTENT-1")
    event_types = [e.event_type for e in led.list_audit_events()]
    assert "payment_execution_failed" in event_types
    led.close()


# No evaluation-only fields enter the ledger at all -- structural guarantee:
# every Ledger method signature takes only primitive ids/amounts/strings,
# never a ScenarioBundle, so there is no path for evaluation_only to reach it.
def test_ledger_api_never_accepts_a_scenario_bundle():
    import inspect

    for name, method in inspect.getmembers(Ledger, predicate=inspect.isfunction):
        sig = inspect.signature(method)
        for param in sig.parameters.values():
            assert param.annotation != "ScenarioBundle", f"Ledger.{name} must not accept a ScenarioBundle"
