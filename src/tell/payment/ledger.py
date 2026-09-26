"""Simulated SQLite payment ledger.

Built-in `sqlite3` only, parameterized SQL throughout, integer minor units
only (no floats), explicit transactions with rollback on failure. This is
a local simulation: no external financial service, no network call, no
real money.

Money can only ever move through this exact sequence, enforced by the
`payment_intents.status` state machine, not merely by caller discipline:

    create_payment_intent   -> status = 'proposed'   (no money moves)
    record_gate_decision    -> status = 'permitted' | 'blocked'
    execute_intent          -> requires status == 'permitted';
                                'blocked'/'proposed'/'executed' all raise.

There is no function here that accepts a candidate action and moves money
without first passing through record_gate_decision setting the intent to
'permitted'. Every proposal, gate decision, and execution (successful or
rejected) is written to audit_events.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tell.payment.models import (
    Account,
    AccountType,
    AuditEvent,
    Invoice,
    InvoiceStatus,
    IntentStatus,
    JournalDirection,
    JournalEntry,
    PaymentIntent,
)

SUPPORTED_CURRENCIES = {"usd", "eur", "gbp"}


class LedgerError(Exception):
    """Base class for every ledger-rejected operation."""


class InvalidAmountError(LedgerError):
    pass


class UnsupportedCurrencyError(LedgerError):
    pass


class CurrencyMismatchError(LedgerError):
    pass


class UnknownAccountError(LedgerError):
    pass


class UnknownInvoiceError(LedgerError):
    pass


class UnknownIntentError(LedgerError):
    pass


class InsufficientFundsError(LedgerError):
    pass


class DuplicateExecutionError(LedgerError):
    pass


class IntentNotPermittedError(LedgerError):
    pass


class GateDecisionAlreadyRecordedError(LedgerError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    account_type TEXT NOT NULL CHECK(account_type IN ('company','supplier','attacker')),
    currency TEXT NOT NULL,
    balance_minor_units INTEGER NOT NULL CHECK(balance_minor_units >= 0)
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    currency TEXT NOT NULL,
    amount_minor_units INTEGER NOT NULL CHECK(amount_minor_units > 0),
    status TEXT NOT NULL CHECK(status IN ('unpaid','paid')) DEFAULT 'unpaid'
);

CREATE TABLE IF NOT EXISTS payment_intents (
    intent_id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id),
    source_account_id TEXT NOT NULL REFERENCES accounts(account_id),
    beneficiary_account_id TEXT NOT NULL REFERENCES accounts(account_id),
    amount_minor_units INTEGER NOT NULL CHECK(amount_minor_units > 0),
    currency TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('proposed','permitted','blocked','executed','rejected')) DEFAULT 'proposed',
    alarm_state TEXT,
    gate_decision TEXT,
    gate_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id TEXT NOT NULL REFERENCES payment_intents(intent_id),
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    direction TEXT NOT NULL CHECK(direction IN ('debit','credit')),
    amount_minor_units INTEGER NOT NULL CHECK(amount_minor_units > 0),
    currency TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Ledger:
    """One SQLite database connection, opened in manual-transaction mode
    (isolation_level=None) so every write path issues its own explicit
    BEGIN IMMEDIATE / COMMIT / ROLLBACK."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # 1. Schema initialization
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Autocommits immediately and independently of any surrounding
        transaction, so a failure audit record survives even when the
        transaction that triggered it is rolled back."""
        self.conn.execute(
            "INSERT INTO audit_events (event_type, payload_json, created_at) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), _now()),
        )

    def _get_account_row(self, account_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        return cur.fetchone()

    def _get_invoice_row(self, invoice_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM invoices WHERE invoice_id = ?", (invoice_id,))
        return cur.fetchone()

    def _get_intent_row(self, intent_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM payment_intents WHERE intent_id = ?", (intent_id,))
        return cur.fetchone()

    # ------------------------------------------------------------------
    # 2. Seeding accounts and invoice state from the clean scenario
    # ------------------------------------------------------------------

    def seed_accounts_and_invoice(
        self,
        *,
        company_account_id: str,
        supplier_account_id: str,
        currency: str,
        company_opening_balance_minor_units: int,
        supplier_opening_balance_minor_units: int,
        invoice_id: str,
        invoice_amount_minor_units: int,
        invoice_status: str = "unpaid",
    ) -> None:
        if currency not in SUPPORTED_CURRENCIES:
            raise UnsupportedCurrencyError(f"Unsupported currency: {currency!r}")
        if invoice_amount_minor_units <= 0:
            raise InvalidAmountError(f"Invoice amount must be positive, got {invoice_amount_minor_units}")
        if company_opening_balance_minor_units < 0 or supplier_opening_balance_minor_units < 0:
            raise InvalidAmountError("Opening balances must be non-negative")

        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "INSERT INTO accounts (account_id, account_type, currency, balance_minor_units) VALUES (?,?,?,?)",
                (company_account_id, AccountType.COMPANY.value, currency, company_opening_balance_minor_units),
            )
            self.conn.execute(
                "INSERT INTO accounts (account_id, account_type, currency, balance_minor_units) VALUES (?,?,?,?)",
                (supplier_account_id, AccountType.SUPPLIER.value, currency, supplier_opening_balance_minor_units),
            )
            self.conn.execute(
                "INSERT INTO invoices (invoice_id, currency, amount_minor_units, status) VALUES (?,?,?,?)",
                (invoice_id, currency, invoice_amount_minor_units, invoice_status),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._audit(
            "ledger_seeded",
            {
                "company_account_id": company_account_id,
                "supplier_account_id": supplier_account_id,
                "currency": currency,
                "company_opening_balance_minor_units": company_opening_balance_minor_units,
                "supplier_opening_balance_minor_units": supplier_opening_balance_minor_units,
                "invoice_id": invoice_id,
                "invoice_amount_minor_units": invoice_amount_minor_units,
                "invoice_status": invoice_status,
            },
        )

    # ------------------------------------------------------------------
    # 3. Creating a payment intent without moving money
    # ------------------------------------------------------------------

    def create_payment_intent(
        self,
        *,
        intent_id: str,
        invoice_id: str,
        source_account_id: str,
        beneficiary_account_id: str,
        amount_minor_units: int,
        currency: str,
        reason: str,
    ) -> PaymentIntent:
        if amount_minor_units <= 0:
            raise InvalidAmountError(f"Amount must be positive, got {amount_minor_units}")
        if currency not in SUPPORTED_CURRENCIES:
            raise UnsupportedCurrencyError(f"Unsupported currency: {currency!r}")

        invoice_row = self._get_invoice_row(invoice_id)
        if invoice_row is None:
            raise UnknownInvoiceError(f"Unknown invoice_id: {invoice_id!r}")
        if invoice_row["currency"] != currency:
            raise CurrencyMismatchError(
                f"Intent currency {currency!r} does not match invoice currency {invoice_row['currency']!r}"
            )

        for account_id in (source_account_id, beneficiary_account_id):
            if self._get_account_row(account_id) is None:
                raise UnknownAccountError(f"Unknown account_id: {account_id!r}")

        created_at = _now()
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                """INSERT INTO payment_intents
                   (intent_id, invoice_id, source_account_id, beneficiary_account_id,
                    amount_minor_units, currency, reason, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    intent_id,
                    invoice_id,
                    source_account_id,
                    beneficiary_account_id,
                    amount_minor_units,
                    currency,
                    reason,
                    IntentStatus.PROPOSED.value,
                    created_at,
                ),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._audit(
            "payment_intent_created",
            {
                "intent_id": intent_id,
                "invoice_id": invoice_id,
                "source_account_id": source_account_id,
                "beneficiary_account_id": beneficiary_account_id,
                "amount_minor_units": amount_minor_units,
                "currency": currency,
                "reason": reason,
            },
        )
        return self.get_intent(intent_id)

    # ------------------------------------------------------------------
    # 4. Recording a gate decision
    # ------------------------------------------------------------------

    def record_gate_decision(
        self,
        *,
        intent_id: str,
        alarm_state: str,
        decision: str,
        reason: str,
    ) -> PaymentIntent:
        if decision not in ("permit", "block"):
            raise LedgerError(f"decision must be 'permit' or 'block', got {decision!r}")

        row = self._get_intent_row(intent_id)
        if row is None:
            raise UnknownIntentError(f"Unknown intent_id: {intent_id!r}")
        if row["status"] != IntentStatus.PROPOSED.value:
            raise GateDecisionAlreadyRecordedError(
                f"Intent {intent_id!r} already has a gate decision recorded (status={row['status']!r})"
            )

        new_status = IntentStatus.PERMITTED.value if decision == "permit" else IntentStatus.BLOCKED.value
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "UPDATE payment_intents SET status = ?, alarm_state = ?, gate_decision = ?, gate_reason = ? WHERE intent_id = ?",
                (new_status, alarm_state, decision, reason, intent_id),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._audit(
            "gate_decision_recorded",
            {
                "intent_id": intent_id,
                "alarm_state": alarm_state,
                "decision": decision,
                "reason": reason,
                "resulting_status": new_status,
            },
        )
        return self.get_intent(intent_id)

    # ------------------------------------------------------------------
    # 5-7. Executing a permitted intent atomically: balanced journal
    # entries, updated balances, invoice status, intent status.
    # ------------------------------------------------------------------

    def execute_intent(self, intent_id: str) -> PaymentIntent:
        row = self._get_intent_row(intent_id)
        if row is None:
            raise UnknownIntentError(f"Unknown intent_id: {intent_id!r}")

        if row["status"] == IntentStatus.EXECUTED.value:
            self._audit("duplicate_execution_rejected", {"intent_id": intent_id})
            raise DuplicateExecutionError(f"Intent {intent_id!r} was already executed")

        if row["status"] != IntentStatus.PERMITTED.value:
            self._audit(
                "execution_rejected_not_permitted",
                {"intent_id": intent_id, "status": row["status"]},
            )
            raise IntentNotPermittedError(
                f"Intent {intent_id!r} is not permitted (status={row['status']!r}); "
                "it must be created, then pass through record_gate_decision(decision='permit') first"
            )

        invoice_id = row["invoice_id"]
        source_account_id = row["source_account_id"]
        beneficiary_account_id = row["beneficiary_account_id"]
        amount = row["amount_minor_units"]
        currency = row["currency"]

        try:
            self.conn.execute("BEGIN IMMEDIATE")

            invoice_row = self._get_invoice_row(invoice_id)
            if invoice_row is None:
                raise UnknownInvoiceError(f"Unknown invoice_id: {invoice_id!r}")
            if invoice_row["currency"] != currency:
                raise CurrencyMismatchError(
                    f"Intent currency {currency!r} does not match invoice currency {invoice_row['currency']!r}"
                )

            source_row = self._get_account_row(source_account_id)
            beneficiary_row = self._get_account_row(beneficiary_account_id)
            if source_row is None:
                raise UnknownAccountError(f"Unknown source account: {source_account_id!r}")
            if beneficiary_row is None:
                raise UnknownAccountError(f"Unknown beneficiary account: {beneficiary_account_id!r}")
            if source_row["currency"] != currency or beneficiary_row["currency"] != currency:
                raise CurrencyMismatchError("Account currency does not match intent currency")

            if source_row["balance_minor_units"] < amount:
                raise InsufficientFundsError(
                    f"Source account {source_account_id!r} has {source_row['balance_minor_units']} "
                    f"minor units, needs {amount}"
                )

            created_at = _now()

            self.conn.execute(
                "UPDATE accounts SET balance_minor_units = balance_minor_units - ? WHERE account_id = ?",
                (amount, source_account_id),
            )
            self.conn.execute(
                "UPDATE accounts SET balance_minor_units = balance_minor_units + ? WHERE account_id = ?",
                (amount, beneficiary_account_id),
            )
            self.conn.execute(
                "INSERT INTO journal_entries (intent_id, account_id, direction, amount_minor_units, currency, created_at) VALUES (?,?,?,?,?,?)",
                (intent_id, source_account_id, JournalDirection.DEBIT.value, amount, currency, created_at),
            )
            self.conn.execute(
                "INSERT INTO journal_entries (intent_id, account_id, direction, amount_minor_units, currency, created_at) VALUES (?,?,?,?,?,?)",
                (intent_id, beneficiary_account_id, JournalDirection.CREDIT.value, amount, currency, created_at),
            )
            self.conn.execute(
                "UPDATE invoices SET status = 'paid' WHERE invoice_id = ?",
                (invoice_id,),
            )
            self.conn.execute(
                "UPDATE payment_intents SET status = ? WHERE intent_id = ?",
                (IntentStatus.EXECUTED.value, intent_id),
            )
            self.conn.execute("COMMIT")
        except Exception as exc:
            self.conn.execute("ROLLBACK")
            self._audit(
                "payment_execution_failed",
                {"intent_id": intent_id, "error_type": type(exc).__name__, "error_message": str(exc)},
            )
            raise

        self._audit(
            "payment_executed",
            {
                "intent_id": intent_id,
                "invoice_id": invoice_id,
                "source_account_id": source_account_id,
                "beneficiary_account_id": beneficiary_account_id,
                "amount_minor_units": amount,
                "currency": currency,
            },
        )
        return self.get_intent(intent_id)

    # ------------------------------------------------------------------
    # 8. Reads
    # ------------------------------------------------------------------

    def get_balance(self, account_id: str) -> int:
        row = self._get_account_row(account_id)
        if row is None:
            raise UnknownAccountError(f"Unknown account_id: {account_id!r}")
        return int(row["balance_minor_units"])

    def get_account(self, account_id: str) -> Account:
        row = self._get_account_row(account_id)
        if row is None:
            raise UnknownAccountError(f"Unknown account_id: {account_id!r}")
        return Account(
            account_id=row["account_id"],
            account_type=AccountType(row["account_type"]),
            currency=row["currency"],
            balance_minor_units=row["balance_minor_units"],
        )

    def get_invoice(self, invoice_id: str) -> Invoice:
        row = self._get_invoice_row(invoice_id)
        if row is None:
            raise UnknownInvoiceError(f"Unknown invoice_id: {invoice_id!r}")
        return Invoice(
            invoice_id=row["invoice_id"],
            currency=row["currency"],
            amount_minor_units=row["amount_minor_units"],
            status=InvoiceStatus(row["status"]),
        )

    def get_intent(self, intent_id: str) -> PaymentIntent:
        row = self._get_intent_row(intent_id)
        if row is None:
            raise UnknownIntentError(f"Unknown intent_id: {intent_id!r}")
        return PaymentIntent(
            intent_id=row["intent_id"],
            invoice_id=row["invoice_id"],
            source_account_id=row["source_account_id"],
            beneficiary_account_id=row["beneficiary_account_id"],
            amount_minor_units=row["amount_minor_units"],
            currency=row["currency"],
            reason=row["reason"],
            status=IntentStatus(row["status"]),
            alarm_state=row["alarm_state"],
            gate_decision=row["gate_decision"],
            gate_reason=row["gate_reason"],
            created_at=row["created_at"],
        )

    def list_journal_entries(self, intent_id: str | None = None) -> list[JournalEntry]:
        if intent_id is None:
            cur = self.conn.execute("SELECT * FROM journal_entries ORDER BY entry_id")
        else:
            cur = self.conn.execute(
                "SELECT * FROM journal_entries WHERE intent_id = ? ORDER BY entry_id", (intent_id,)
            )
        return [
            JournalEntry(
                entry_id=r["entry_id"],
                intent_id=r["intent_id"],
                account_id=r["account_id"],
                direction=JournalDirection(r["direction"]),
                amount_minor_units=r["amount_minor_units"],
                currency=r["currency"],
                created_at=r["created_at"],
            )
            for r in cur.fetchall()
        ]

    def list_audit_events(self) -> list[AuditEvent]:
        cur = self.conn.execute("SELECT * FROM audit_events ORDER BY event_id")
        return [
            AuditEvent(
                event_id=r["event_id"],
                event_type=r["event_type"],
                payload=json.loads(r["payload_json"]),
                created_at=r["created_at"],
            )
            for r in cur.fetchall()
        ]
