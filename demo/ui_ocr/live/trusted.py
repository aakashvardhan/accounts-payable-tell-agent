"""Trusted demo context for the live runtime (worker side: imports the project's `tell.*` packages).

Everything the *trusted side* of the system needs, backed by an isolated runtime directory:
  * vendor master (demo fixture `data/vendor_master_demo.json`)  -> canonical vendor records + verified AR contacts
  * trusted ERP (demo fixture `data/trusted_demo_erp.json`)      -> what an approved invoice actually is (amount/currency/number), disputes
  * the project's real simulated ledger (`tell.payment.ledger.Ledger`, SQLite) seeded from that ERP
  * the project's real `ResolutionCoordinator` (persisted simulated outbox + review queue)
The fixtures are labelled SYNTHETIC: they stand in for an ERP that does not exist in this demo. Outcomes still come from the
model, the validator, the gate and the ledger -- the fixtures only say what the trusted records contain.
"""
import json
import re
import shutil
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from enterprise_v2_2.runtime import ResolutionCoordinator, TrustedVendorMasterRecord  # noqa: E402
from tell.agent.trusted_lookups import (  # noqa: E402
    AmbiguousIdentifierError, DisputeCaseLookupOutcome, DisputeCaseQuery, DisputeCaseRecord, DisputeCaseStatus,
    InvoicePaymentHistoryLookupOutcome, InvoicePaymentHistoryQuery, InvoicePaymentHistoryRecord,
    PermittedResolutionClassification, PriorPaymentStatus)
from tell.evaluation.scenario import OperationalProvenance, ProvenanceSource, SourceType, TrustBoundary  # noqa: E402
from tell.memory.models import MemoryKind, MemoryStatus  # noqa: E402
from tell.memory.store import MemoryStore  # noqa: E402
from tell.payment.ledger import DuplicateExecutionError, Ledger, LedgerError  # noqa: E402
from tell.safety.gate import PermittedExecutorAction  # noqa: E402
from tell.safety.payment_validation import (  # noqa: E402
    TrustedInvoiceRecord, TrustedVendorRecord, VendorStatus, VendorVerificationStatus)

DATA = Path(__file__).resolve().parent.parent / "data"
VENDOR_MASTER = DATA / "vendor_master_demo.json"
TRUSTED_ERP = DATA / "trusted_demo_erp.json"
MEMORY_SEED = DATA / "demo_memory_seed.json"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_name(name):
    s = unicodedata.normalize("NFKC", str(name)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def norm_number(n):
    return re.sub(r"[^a-z0-9]", "", str(n or "").lower())


def trusted_provenance(source_type, source_id):
    """Trusted synthetic demo stores use the project's `synthetic_controlled` provenance."""
    return OperationalProvenance(source_type=source_type, source_id=source_id, provenance=ProvenanceSource.SYNTHETIC_CONTROLLED,
                                 recorded_at=_now(), trust_boundary=TrustBoundary.TRUSTED)


class TrustedDemoData:
    """Read-only view of the two fixtures."""

    def __init__(self, vendor_master=VENDOR_MASTER, erp=TRUSTED_ERP):
        vm = json.loads(Path(vendor_master).read_text())
        self.vendors = {v["vendor_id"]: v for v in vm["vendors"]}
        self.vendor_label = vm.get("label")
        e = json.loads(Path(erp).read_text())
        self.erp_label = e.get("label")
        self.company_accounts = e["company_accounts"]
        self.invoices = e["invoices"]
        self.disputes = e["disputes"]

    def vendor_by_name(self, name):
        """Unique canonical vendor for an extracted supplier name (normalised equality with name or alias), else None."""
        if not name:
            return None
        key = norm_name(name)
        hits = [v for v in self.vendors.values() if key in {norm_name(v["vendor_name"])} | {norm_name(a) for a in v.get("aliases", [])}]
        return hits[0] if len(hits) == 1 else None

    def erp_invoice(self, vendor_id, invoice_number):
        n = norm_number(invoice_number)
        hits = [i for i in self.invoices if i["vendor_id"] == vendor_id and norm_number(i["invoice_number"]) == n and n]
        return hits[0] if len(hits) == 1 else None

    def dispute(self, vendor_id, invoice_number):
        n = norm_number(invoice_number)
        for d in self.disputes:
            if d["vendor_id"] == vendor_id and norm_number(d["invoice_number"]) == n and n:
                return d
        return None

    def master_records(self):
        return {vid: TrustedVendorMasterRecord(vendor_id=vid, vendor_name=v["vendor_name"], beneficiary_account_id=v["beneficiary_account_id"],
                                               verification_status=v["verification_status"], vendor_status=v["vendor_status"],
                                               approved_contact_email=v.get("approved_contact_email"),
                                               approved_contact_verified=bool(v.get("approved_contact_verified")))
                for vid, v in self.vendors.items()}

    def trusted_vendor_record(self, vendor_id):
        v = self.vendors.get(vendor_id)
        if v is None:
            return None
        return TrustedVendorRecord(vendor_id=vendor_id, vendor_name=v["vendor_name"], beneficiary_account_id=v["beneficiary_account_id"],
                                   verification_status=VendorVerificationStatus(v["verification_status"]), vendor_status=VendorStatus(v["vendor_status"]),
                                   approved_contact_email=v.get("approved_contact_email"), approved_contact_verified=bool(v.get("approved_contact_verified")))


class LiveContext:
    """Per-runtime-directory trusted state: ledger, coordinator, fixtures. `reset()` rebuilds both stores from the fixtures."""

    def __init__(self, runtime_dir, data=None):
        self.root = Path(runtime_dir) / "live"
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = data or TrustedDemoData()
        self.ledger_path = self.root / "ledger.sqlite"
        self.baseline_ledger_path = self.root / "ledger_tellsecured_off.sqlite"   # separate books for the undefended (TellSecured OFF) comparison
        self.coord_dir = self.root / "coordinator"
        self.memory_path = self.root / "memory.sqlite"
        self._open()

    def _open(self):
        fresh = not self.ledger_path.exists()
        self.baseline_ledger = self._open_ledger(self.baseline_ledger_path)
        self.ledger = self._open_ledger(self.ledger_path)
        self.coordinator = ResolutionCoordinator(self.coord_dir, self.data.master_records(), clock=_now)
        mem_fresh = not self.memory_path.exists()
        self.memory = MemoryStore(self.memory_path)          # the project's memory store (provenance + status, incl. quarantine)
        self.memory.init_schema()
        if mem_fresh:
            self._seed_memory()
        return fresh

    def _seed_memory(self):
        """Demo memory seed (synthetic fixture): what the agent 'learned' earlier, each record keeping its untrusted origin."""
        if not MEMORY_SEED.exists():
            return
        for m in json.loads(MEMORY_SEED.read_text())["memories"]:
            self.memory.append_memory(vendor_id=m["vendor_id"], memory_kind=MemoryKind(m["memory_kind"]), content=m["content"],
                                      origin_source_type=SourceType(m["origin_source_type"]), origin_source_id=m["origin_source_id"],
                                      origin_provenance=ProvenanceSource.SYNTHETIC_CONTROLLED, origin_trust_boundary=TrustBoundary(m["origin_trust_boundary"]))

    def _open_ledger(self, path):
        led = Ledger(path)
        led.init_schema()
        led.conn.execute("CREATE TABLE IF NOT EXISTS document_payments(document_id TEXT PRIMARY KEY, erp_invoice_id TEXT NOT NULL, intent_id TEXT NOT NULL)")
        self.ledger = led
        self._seed()
        return led

    def _seed(self):
        c = self.ledger.conn
        for a in self.data.company_accounts:
            c.execute("INSERT OR IGNORE INTO accounts(account_id, account_type, currency, balance_minor_units) VALUES(?,?,?,?)",
                      (a["account_id"], "company", a["currency"], a["balance_minor_units"]))
        for v in self.data.vendors.values():
            c.execute("INSERT OR IGNORE INTO accounts(account_id, account_type, currency, balance_minor_units) VALUES(?,?,?,?)",
                      (v["beneficiary_account_id"], "supplier", v["payment_currency"], 0))
        for inv in self.data.invoices:
            c.execute("INSERT OR IGNORE INTO invoices(invoice_id, currency, amount_minor_units, status) VALUES(?,?,?,?)",
                      (inv["erp_invoice_id"], inv["currency"], inv["amount_minor_units"], "unpaid"))
        for inv in self.data.invoices:
            if inv.get("seed_prior_payment") and self.ledger.get_invoice(inv["erp_invoice_id"]).status != "paid":
                v = self.data.vendors[inv["vendor_id"]]
                iid = f"INT-SEED-{inv['erp_invoice_id']}"
                self.ledger.create_payment_intent(intent_id=iid, invoice_id=inv["erp_invoice_id"], source_account_id=self.company_account(inv["currency"]),
                                                  beneficiary_account_id=v["beneficiary_account_id"], amount_minor_units=inv["amount_minor_units"],
                                                  currency=inv["currency"], reason="demo seed: prior payment")
                self.ledger.record_gate_decision(intent_id=iid, alarm_state="clear", decision="permit", reason="demo seed: prior payment")
                self.ledger.execute_intent(iid)

    def baseline_pay(self, document_id, action, vendor_id):
        """TellSecured OFF: execute a proposal as-is on the simulated ledger (the undefended baseline). The ERP invoice is used when one
        matches; otherwise an ad-hoc invoice row is created for the document. An unknown beneficiary becomes an external account."""
        led = self.baseline_ledger
        c = led.conn
        erp = self.data.erp_invoice(vendor_id, action.invoice_number) if vendor_id else None
        invoice_id = erp["erp_invoice_id"] if erp else f"DOC-{document_id}"
        cur = action.currency.value if hasattr(action.currency, "value") else str(action.currency)
        c.execute("INSERT OR IGNORE INTO invoices(invoice_id, currency, amount_minor_units, status) VALUES(?,?,?,?)", (invoice_id, cur, action.amount_minor_units, "unpaid"))
        c.execute("INSERT OR IGNORE INTO accounts(account_id, account_type, currency, balance_minor_units) VALUES(?,?,?,?)", (action.beneficiary_account_id, "supplier", cur, 0))
        intent_id = f"INT-BASE-{document_id}"
        existing = c.execute("SELECT status FROM payment_intents WHERE intent_id=?", (intent_id,)).fetchone()
        created = existing is None
        source = self.company_account(cur)
        if created:
            led.create_payment_intent(intent_id=intent_id, invoice_id=invoice_id, source_account_id=source, beneficiary_account_id=action.beneficiary_account_id,
                                      amount_minor_units=action.amount_minor_units, currency=cur, reason=f"TellSecured OFF baseline for document {document_id}")
            led.record_gate_decision(intent_id=intent_id, alarm_state="clear", decision="permit", reason="TellSecured OFF: no gate applied (undefended baseline)")
        posted = False
        if led.get_intent(intent_id).status != "executed":
            try:
                led.execute_intent(intent_id); posted = True
            except DuplicateExecutionError:
                pass
        c.execute("INSERT OR IGNORE INTO document_payments(document_id, erp_invoice_id, intent_id) VALUES(?,?,?)", (document_id, invoice_id, intent_id))
        i = led.get_intent(intent_id)
        return {"intent_id": intent_id, "erp_invoice_id": invoice_id, "intent_created": created, "ledger_posted": posted, "intent_status": i.status,
                "amount_minor_units": i.amount_minor_units, "currency": i.currency, "beneficiary_account_id": i.beneficiary_account_id,
                "company_balance_minor_units": led.get_balance(source), "already_executed": not posted and not created}

    def company_account(self, currency):
        for a in self.data.company_accounts:
            if a["currency"] == currency:
                return a["account_id"]
        return None

    def reset(self):
        self.ledger.close()
        self.baseline_ledger.close()
        self.memory.close()
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open()

    def close(self):
        self.ledger.close()
        self.baseline_ledger.close()
        self.memory.close()

    # -- lookups -----------------------------------------------------------
    def history_provider(self, invoice_number, vendor_id):
        return HistoryProvider(self, invoice_number, vendor_id)

    def dispute_provider(self, invoice_number, vendor_id):
        return DisputeProvider(self, invoice_number, vendor_id)

    def paid_for_document(self, document_id):
        r = self.ledger.conn.execute("SELECT erp_invoice_id, intent_id FROM document_payments WHERE document_id=?", (document_id,)).fetchone()
        return dict(r) if r else None

    def trusted_invoice(self, document_id, vendor_id, observed_number):
        """What the trusted side says this invoice is. Amount/currency/number stay None when no approved ERP record exists."""
        erp = self.data.erp_invoice(vendor_id, observed_number) if vendor_id else None
        if erp is None:
            return TrustedInvoiceRecord(invoice_document_id=document_id, vendor_id=vendor_id or "UNRESOLVED", invoice_number=None, amount_minor_units=None, currency=None), None
        return TrustedInvoiceRecord(invoice_document_id=document_id, vendor_id=vendor_id, invoice_number=erp["invoice_number"],
                                    amount_minor_units=erp["amount_minor_units"], currency=erp["currency"]), erp


class HistoryProvider:
    """`tell.agent.trusted_lookups.InvoicePaymentHistoryProvider` backed by the real ledger."""

    def __init__(self, ctx, invoice_number, vendor_id):
        self.ctx, self.invoice_number, self.vendor_id = ctx, invoice_number, vendor_id

    def lookup(self, query: InvoicePaymentHistoryQuery) -> InvoicePaymentHistoryLookupOutcome:
        prior = self.ctx.paid_for_document(query.invoice_document_id)
        erp = self.ctx.data.erp_invoice(query.vendor_id, self.invoice_number)
        erp_id = prior["erp_invoice_id"] if prior else (erp["erp_invoice_id"] if erp else None)
        if erp_id is None:
            return InvoicePaymentHistoryLookupOutcome(record=None)
        inv = self.ctx.ledger.get_invoice(erp_id)
        if inv.status != "paid":
            return InvoicePaymentHistoryLookupOutcome(record=None)
        row = self.ctx.ledger.conn.execute("SELECT intent_id, amount_minor_units, currency, created_at FROM payment_intents WHERE invoice_id=? AND status='executed' ORDER BY created_at LIMIT 1", (erp_id,)).fetchone()
        return InvoicePaymentHistoryLookupOutcome(record=InvoicePaymentHistoryRecord(
            invoice_document_id=query.invoice_document_id, vendor_id=query.vendor_id, prior_payment_status=PriorPaymentStatus.PAID,
            amount_minor_units=row["amount_minor_units"] if row else None, currency=row["currency"] if row else None,
            payment_reference=row["intent_id"] if row else None, payment_timestamp=row["created_at"] if row else None,
            record_id=f"HIST-{erp_id}", provenance=trusted_provenance(SourceType.INVOICE_PAYMENT_HISTORY, f"HIST-{erp_id}")))


class DisputeProvider:
    def __init__(self, ctx, invoice_number, vendor_id):
        self.ctx, self.invoice_number, self.vendor_id = ctx, invoice_number, vendor_id

    def lookup(self, query: DisputeCaseQuery) -> DisputeCaseLookupOutcome:
        d = self.ctx.data.dispute(query.vendor_id, self.invoice_number)
        if d is None:
            return DisputeCaseLookupOutcome(record=None)
        return DisputeCaseLookupOutcome(record=DisputeCaseRecord(
            case_id=d["case_id"], invoice_document_id=query.invoice_document_id, vendor_id=query.vendor_id, status=DisputeCaseStatus(d["status"]),
            permitted_resolution_classification=PermittedResolutionClassification(d["permitted_resolution_classification"]),
            provenance=trusted_provenance(SourceType.DISPUTE_CASE, d["case_id"])))


class LedgerExecutor:
    """`tell.agent.routing_orchestrator.PaymentExecutor` backed by the project's simulated ledger. Idempotent: a document/ERP invoice can
    be paid at most once; a repeat call reports the existing execution instead of posting again."""

    def __init__(self, ctx, doc_to_erp):
        self.ctx, self.doc_to_erp, self.result = ctx, dict(doc_to_erp), None

    def execute(self, action: PermittedExecutorAction) -> None:
        erp_id = self.doc_to_erp.get(action.invoice_id)
        if erp_id is None:
            raise LedgerError(f"no trusted ERP invoice is linked to document {action.invoice_id!r}")
        led = self.ctx.ledger
        intent_id = f"INT-{erp_id}"
        existing = led.conn.execute("SELECT status FROM payment_intents WHERE intent_id=?", (intent_id,)).fetchone()
        created = False
        if existing is None:
            led.create_payment_intent(intent_id=intent_id, invoice_id=erp_id, source_account_id=action.source_account_id,
                                      beneficiary_account_id=action.beneficiary_account_id, amount_minor_units=action.amount_minor_units,
                                      currency=action.currency, reason=f"live run for document {action.invoice_id}")
            created = True
            led.record_gate_decision(intent_id=intent_id, alarm_state="clear", decision="permit", reason="deterministic gate permitted")
        status = led.get_intent(intent_id).status
        posted = False
        if status != "executed":
            try:
                led.execute_intent(intent_id)
                posted = True
            except DuplicateExecutionError:
                posted = False
        self.ctx.ledger.conn.execute("INSERT OR IGNORE INTO document_payments(document_id, erp_invoice_id, intent_id) VALUES(?,?,?)", (action.invoice_id, erp_id, intent_id))
        intent = led.get_intent(intent_id)
        self.result = {"intent_id": intent_id, "erp_invoice_id": erp_id, "intent_created": created, "ledger_posted": posted, "intent_status": intent.status,
                       "amount_minor_units": intent.amount_minor_units, "currency": intent.currency, "beneficiary_account_id": intent.beneficiary_account_id,
                       "company_balance_minor_units": led.get_balance(action.source_account_id), "already_executed": (not posted and not created)}
