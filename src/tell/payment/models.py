"""Typed row/result models for the simulated SQLite ledger.

These mirror the SQLite schema in tell.payment.ledger and are used for
reads and for structuring trace/audit output. They carry no business
logic and no side effects; Ledger in tell.payment.ledger owns all
SQL and transaction handling.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class AccountType(str, Enum):
    COMPANY = "company"
    SUPPLIER = "supplier"
    ATTACKER = "attacker"


class InvoiceStatus(str, Enum):
    UNPAID = "unpaid"
    PAID = "paid"


class IntentStatus(str, Enum):
    PROPOSED = "proposed"
    PERMITTED = "permitted"
    BLOCKED = "blocked"
    EXECUTED = "executed"
    REJECTED = "rejected"


class JournalDirection(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class Account(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    account_type: AccountType
    currency: str
    balance_minor_units: int


class Invoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    invoice_id: str
    currency: str
    amount_minor_units: int
    status: InvoiceStatus


class PaymentIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent_id: str
    invoice_id: str
    source_account_id: str
    beneficiary_account_id: str
    amount_minor_units: int
    currency: str
    reason: str
    status: IntentStatus
    alarm_state: str | None
    gate_decision: str | None
    gate_reason: str | None
    created_at: str


class JournalEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: int
    intent_id: str
    account_id: str
    direction: JournalDirection
    amount_minor_units: int
    currency: str
    created_at: str


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: int
    event_type: str
    payload: dict
    created_at: str
