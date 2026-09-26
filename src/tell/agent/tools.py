"""Read-only Tell agent tools: read_email, read_invoice, get_vendor_record.

Each tool takes typed Pydantic arguments, reads only the one section of a
loaded ScenarioBundle it is responsible for, and returns a typed result
carrying operational provenance (source_type, source_id, provenance,
recorded_at, trust_boundary) alongside structured content -- never
evaluation_only data. Tools are pure functions: given a ScenarioBundle and
typed arguments, they read and return a new result object. They never
write to the bundle (every scenario model is frozen), the filesystem, or
the network, and they make no side-effecting calls.

This module implements exactly three read-only tools, plus a typed
candidate-action schema for a still-unimplemented `pay_invoice`. It does
not implement the pay_invoice function itself, search_memory,
write_memory, the ledger, the agent loop, the model, the probe, the LoRA,
the gate, the API, or the UI.

--------------------------------------------------------------------------
Prototype extraction disclosure
--------------------------------------------------------------------------
read_invoice's structured content is read directly from the DocILE
annotation (the ground-truth field_extractions/line_item_extractions),
not from OCR or a model. That is labeled explicitly via
`extraction_method = "oracle_extraction_prototype"` on every
ReadInvoiceContent.

--------------------------------------------------------------------------
No ground-truth leakage
--------------------------------------------------------------------------
ReadInvoiceContent never includes PDF/OCR/annotation filesystem paths --
those would expose exactly the gold-label files an evaluation harness
uses to grade the agent. The only document reference exposed is
`DocumentReference`, an opaque document_id plus a generic source_system
label. The real filesystem paths remain available only on the internal
DocileInvoiceRecord model (`tell.evaluation.scenario`), which the
evaluation harness may read directly; they never flow through a tool
result or a runtime trace.

--------------------------------------------------------------------------
Candidate pay_invoice schema
--------------------------------------------------------------------------
PayInvoiceCandidate is a pure data structure. Constructing one does not
move money, touch the ledger, or have any side effect -- it exists so an
agent (or, in the demonstration script, a deterministic stand-in for one)
can propose a payment. The only way a candidate can ever cause money to
move is: create_payment_intent -> evaluate_gate -> record_gate_decision ->
execute_intent (see tell.safety.gate and tell.payment.ledger). No function
in this module or in tell.payment.ledger accepts a PayInvoiceCandidate and
moves money directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tell.evaluation.scenario import OperationalProvenance, ScenarioBundle


class ToolName(str, Enum):
    READ_EMAIL = "read_email"
    READ_INVOICE = "read_invoice"
    GET_VENDOR_RECORD = "get_vendor_record"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ToolError(BaseModel):
    """Concise, typed error information for a failed lookup. Never a raw
    exception or traceback -- just enough for the agent/gate to act on."""

    model_config = ConfigDict(frozen=True)

    error_code: Literal["message_not_found", "document_not_found", "vendor_not_found"]
    message: str


# --------------------------------------------------------------------------
# read_email
# --------------------------------------------------------------------------


class ReadEmailArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    message_id: str


class ReadEmailContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    sender_display_name: str
    sender_address: str
    subject: str
    body: str
    references_docid: str
    references_invoice_number: str | None


class ReadEmailResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.READ_EMAIL] = ToolName.READ_EMAIL
    status: ToolStatus
    provenance: OperationalProvenance | None
    content: ReadEmailContent | None
    error: ToolError | None


def read_email(bundle: ScenarioBundle, args: ReadEmailArgs) -> ReadEmailResult:
    """Reads only bundle.untrusted_inputs.supplier_email. Never touches
    invoice_document, trusted_state, or evaluation_only."""
    email = bundle.untrusted_inputs.supplier_email
    if email.message_id is None or email.message_id != args.message_id:
        return ReadEmailResult(
            status=ToolStatus.FAILURE,
            provenance=None,
            content=None,
            error=ToolError(
                error_code="message_not_found",
                message=f"No email found for message_id={args.message_id!r}",
            ),
        )
    view = email.as_read_email_tool_view()
    return ReadEmailResult(
        status=ToolStatus.SUCCESS,
        provenance=email.operational_provenance,
        content=ReadEmailContent(**view),
        error=None,
    )


# --------------------------------------------------------------------------
# read_invoice
# --------------------------------------------------------------------------


class ReadInvoiceArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str


class DocumentReference(BaseModel):
    """An opaque, agent-safe reference to the source document. Deliberately
    contains no filesystem path of any kind -- no PDF path, no OCR path,
    and above all no annotation/gold-label path. Those stay internal to
    the evaluation harness (readable directly off DocileInvoiceRecord) and
    must never enter a tool result or a runtime trace."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_system: str = "docile_invoice_store"


class ReadInvoiceContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    extraction_method: Literal["oracle_extraction_prototype"] = "oracle_extraction_prototype"
    docid: str
    page_count: int
    vendor_name: str | None
    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None
    currency: str | None
    subtotal: str | None
    tax: str | None
    total_amount_gross: str | None
    amount_due: str | None
    purchase_order_numbers: list[str]
    payment_destination: list[str]
    vendor_address: str | None
    vendor_email: str | None
    customer_billing_name: str | None
    customer_billing_address: str | None
    payment_terms: str | None
    line_items: list[dict]
    document_reference: DocumentReference


class ReadInvoiceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.READ_INVOICE] = ToolName.READ_INVOICE
    status: ToolStatus
    provenance: OperationalProvenance | None
    content: ReadInvoiceContent | None
    error: ToolError | None


def read_invoice(bundle: ScenarioBundle, args: ReadInvoiceArgs) -> ReadInvoiceResult:
    """Reads only bundle.untrusted_inputs.invoice_document. Never touches
    supplier_email, trusted_state, or evaluation_only."""
    invoice = bundle.untrusted_inputs.invoice_document
    if invoice.docid != args.document_id:
        return ReadInvoiceResult(
            status=ToolStatus.FAILURE,
            provenance=None,
            content=None,
            error=ToolError(
                error_code="document_not_found",
                message=f"No invoice found for document_id={args.document_id!r}",
            ),
        )
    view = invoice.as_read_invoice_tool_view()
    content = ReadInvoiceContent(
        **view,
        document_reference=DocumentReference(document_id=invoice.docid),
    )
    return ReadInvoiceResult(
        status=ToolStatus.SUCCESS,
        provenance=invoice.operational_provenance,
        content=content,
        error=None,
    )


# --------------------------------------------------------------------------
# get_vendor_record
# --------------------------------------------------------------------------


class GetVendorRecordArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    vendor_id: str


class GetVendorRecordContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    vendor_id: str
    vendor_name: str
    beneficiary_account_id: str
    verification_status: str


class GetVendorRecordResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.GET_VENDOR_RECORD] = ToolName.GET_VENDOR_RECORD
    status: ToolStatus
    provenance: OperationalProvenance | None
    content: GetVendorRecordContent | None
    error: ToolError | None


def get_vendor_record(bundle: ScenarioBundle, args: GetVendorRecordArgs) -> GetVendorRecordResult:
    """Reads only bundle.trusted_state. Never touches supplier_email,
    invoice_document, or evaluation_only."""
    vendor = bundle.trusted_state
    if vendor.canonical_vendor_id != args.vendor_id:
        return GetVendorRecordResult(
            status=ToolStatus.FAILURE,
            provenance=None,
            content=None,
            error=ToolError(
                error_code="vendor_not_found",
                message=f"No vendor record found for vendor_id={args.vendor_id!r}",
            ),
        )
    view = vendor.as_get_vendor_record_tool_view()
    return GetVendorRecordResult(
        status=ToolStatus.SUCCESS,
        provenance=vendor.operational_provenance,
        content=GetVendorRecordContent(**view),
        error=None,
    )


# --------------------------------------------------------------------------
# pay_invoice candidate schema (no pay_invoice tool implemented yet)
# --------------------------------------------------------------------------


class PayInvoiceCandidate(BaseModel):
    """A proposed payment, not yet an action. Building one has no side
    effects: it does not move money, does not touch the ledger, and is not
    itself checked against the gate. See tell.safety.gate.evaluate_gate and
    tell.payment.ledger.Ledger for the only path that can turn a candidate
    into money actually moving: create_payment_intent -> evaluate_gate ->
    record_gate_decision -> execute_intent."""

    model_config = ConfigDict(frozen=True)

    invoice_id: str
    source_account_id: str
    beneficiary_account_id: str
    amount_minor_units: int
    currency: str
    reason: str
