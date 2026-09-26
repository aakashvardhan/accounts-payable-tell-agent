"""Trusted, read-only lookup tools: invoice/payment-history and
dispute/case-status (Part 1 of the runtime-integration milestone; see
`results/routing_design/tell_runtime_integration_v1.md`).

These are new tools, distinct in kind from the three read-only tools in
`tell.agent.tools` (read_email, read_invoice, get_vendor_record): those
read from `ScenarioBundle.untrusted_inputs` / `.trusted_state` directly.
These two instead go through an injected *provider* -- a narrow,
explicit dependency-injection boundary (`InvoicePaymentHistoryProvider`,
`DisputeCaseProvider`, both `typing.Protocol`s) -- because a real
deployment would back them with a payment-history ledger and a case
database that do not exist as ScenarioBundle fields. No implementation of
either provider is shipped as production code here; tests supply an
in-memory fake. Nothing in this module creates, opens, or mutates a real
SQLite database or calls any external service.

Read-only by construction
--------------------------------------------------------------------------
Both provider Protocols expose exactly one method each, `lookup`, and
that method's return type (`InvoicePaymentHistoryResult` /
`DisputeCaseLookupResult`) has no field through which a caller could
request or record a write, an approval, a payment, or an alarm-state
change -- there is no such field to fill in, not merely an unused one.

No record vs. lookup failure
--------------------------------------------------------------------------
`LookupStatus.NO_RECORD` is an AUTHORITATIVE negative answer ("the trusted
store has no invoice/case matching this query"). `LookupStatus.LOOKUP_FAILED`
means the provider itself could not answer (a malformed/ambiguous
identifier it rejected, or an internal provider error) -- these are never
conflated: `check_trusted_invoice_payment_history` and
`check_trusted_dispute_case_status` return NO_RECORD only when the
provider itself reports that outcome, and LOOKUP_FAILED for every
exception the provider raises, with the exception's message preserved in
`ToolError.message` (never a raw traceback).

Untrusted free-form claims never become trusted facts
--------------------------------------------------------------------------
Neither query model accepts a beneficiary, an amount, a narrative claim,
or anything else the model might have read from an untrusted email/invoice
-- only the two stable identifiers (`invoice_document_id`, `vendor_id`)
already established elsewhere as the agent's own prior observations. The
provider is the only source of the returned facts; nothing about the
result is influenced by what the untrusted context claimed.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tell.agent.tools import ToolStatus
from tell.evaluation.scenario import OperationalProvenance


class LookupToolName(str, Enum):
    CHECK_TRUSTED_INVOICE_PAYMENT_HISTORY = "check_trusted_invoice_payment_history"
    CHECK_TRUSTED_DISPUTE_CASE_STATUS = "check_trusted_dispute_case_status"


class LookupStatus(str, Enum):
    FOUND = "found"
    NO_RECORD = "no_record"  # authoritative: the trusted store has nothing for this query
    LOOKUP_FAILED = "lookup_failed"  # the provider could not answer (malformed input, ambiguous match, provider error)


class LookupError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_code: Literal["ambiguous_identifier", "provider_error"]
    message: str


_DOCID_LEN = 24


def _validate_stable_id(value: str, *, field_name: str) -> str:
    v = value.strip()
    if not v or v != value:
        raise ValueError(f"{field_name} must be a non-empty identifier with no leading/trailing whitespace, got {value!r}")
    if any(c.isspace() for c in v):
        raise ValueError(f"{field_name} must not contain internal whitespace, got {value!r}")
    return v


# ---------------------------------------------------------------------
# A. Invoice / payment-history lookup
# ---------------------------------------------------------------------


class InvoicePaymentHistoryQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)

    @field_validator("invoice_document_id")
    @classmethod
    def _check_invoice_id(cls, v: str) -> str:
        return _validate_stable_id(v, field_name="invoice_document_id")

    @field_validator("vendor_id")
    @classmethod
    def _check_vendor_id(cls, v: str) -> str:
        return _validate_stable_id(v, field_name="vendor_id")


class PriorPaymentStatus(str, Enum):
    NOT_PAID = "not_paid"
    PAID = "paid"


class InvoicePaymentHistoryRecord(BaseModel):
    """Trusted facts only -- always TrustBoundary.TRUSTED (see
    `OperationalProvenance`); there is no field here that could carry an
    untrusted claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    prior_payment_status: PriorPaymentStatus
    amount_minor_units: int | None = Field(default=None, ge=0)
    currency: str | None = None
    payment_reference: str | None = None
    payment_timestamp: str | None = None
    record_id: str = Field(min_length=1)
    provenance: OperationalProvenance


class InvoicePaymentHistoryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: Literal[LookupToolName.CHECK_TRUSTED_INVOICE_PAYMENT_HISTORY] = LookupToolName.CHECK_TRUSTED_INVOICE_PAYMENT_HISTORY
    status: LookupStatus
    record: InvoicePaymentHistoryRecord | None
    error: LookupError | None


class AmbiguousIdentifierError(Exception):
    """A provider raises this (never returns it) when a query's
    identifiers match more than one trusted record and it cannot decide
    which is authoritative."""


class InvoicePaymentHistoryLookupOutcome(BaseModel):
    """What a provider returns internally: either a record, or an
    authoritative "no record" (record=None) -- or it raises (caught by the
    tool function below and turned into LOOKUP_FAILED). A provider never
    returns LOOKUP_FAILED itself; that status is assigned only by the tool
    function catching an exception, keeping "the provider doesn't know"
    (raise) and "the trusted store affirmatively has nothing" (this model
    with record=None) distinct at the type level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: InvoicePaymentHistoryRecord | None = None


class InvoicePaymentHistoryProvider(Protocol):
    """Explicit provider boundary -- production code injects a real
    implementation (backed by whatever trusted payment-history store
    exists); tests inject an in-memory fake. No implementation is shipped
    in this module."""

    def lookup(self, query: InvoicePaymentHistoryQuery) -> InvoicePaymentHistoryLookupOutcome: ...


def check_trusted_invoice_payment_history(provider: InvoicePaymentHistoryProvider, query: InvoicePaymentHistoryQuery) -> InvoicePaymentHistoryResult:
    """The tool function: validates `query` (already strictly typed by
    `InvoicePaymentHistoryQuery`), calls the injected provider exactly
    once, and never repairs, retries, or reinterprets what it returns."""
    try:
        outcome = provider.lookup(query)
    except AmbiguousIdentifierError as exc:
        return InvoicePaymentHistoryResult(status=LookupStatus.LOOKUP_FAILED, record=None, error=LookupError(error_code="ambiguous_identifier", message=str(exc)))
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any provider failure becomes LOOKUP_FAILED, never a raw traceback
        return InvoicePaymentHistoryResult(status=LookupStatus.LOOKUP_FAILED, record=None, error=LookupError(error_code="provider_error", message=str(exc)))
    if outcome.record is None:
        return InvoicePaymentHistoryResult(status=LookupStatus.NO_RECORD, record=None, error=None)
    if not isinstance(outcome.record, InvoicePaymentHistoryRecord):
        raise TypeError("provider returned a record of the wrong type for check_trusted_invoice_payment_history")
    return InvoicePaymentHistoryResult(status=LookupStatus.FOUND, record=outcome.record, error=None)


# ---------------------------------------------------------------------
# B. Dispute / case-status lookup
# ---------------------------------------------------------------------


class DisputeCaseQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)

    @field_validator("invoice_document_id")
    @classmethod
    def _check_invoice_id(cls, v: str) -> str:
        return _validate_stable_id(v, field_name="invoice_document_id")

    @field_validator("vendor_id")
    @classmethod
    def _check_vendor_id(cls, v: str) -> str:
        return _validate_stable_id(v, field_name="vendor_id")


class DisputeCaseStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    CLOSED_NO_ISSUE = "closed_no_issue"


class PermittedResolutionClassification(str, Enum):
    """What a registered human reviewer (never Agent S, never the model)
    is permitted to conclude about this case -- purely descriptive
    metadata from the trusted case store, not a decision this tool makes."""

    NONE_REQUIRED = "none_required"
    BENIGN_EXPLANATION_CONFIRMED = "benign_explanation_confirmed"
    CONFIRMED_SECURITY_ISSUE = "confirmed_security_issue"
    REQUIRES_HUMAN_REVIEW = "requires_human_review"


class DisputeCaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    invoice_document_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    status: DisputeCaseStatus
    permitted_resolution_classification: PermittedResolutionClassification
    provenance: OperationalProvenance


class DisputeCaseLookupResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: Literal[LookupToolName.CHECK_TRUSTED_DISPUTE_CASE_STATUS] = LookupToolName.CHECK_TRUSTED_DISPUTE_CASE_STATUS
    status: LookupStatus
    record: DisputeCaseRecord | None
    error: LookupError | None


class DisputeCaseLookupOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record: DisputeCaseRecord | None = None


class DisputeCaseProvider(Protocol):
    def lookup(self, query: DisputeCaseQuery) -> DisputeCaseLookupOutcome: ...


def check_trusted_dispute_case_status(provider: DisputeCaseProvider, query: DisputeCaseQuery) -> DisputeCaseLookupResult:
    try:
        outcome = provider.lookup(query)
    except AmbiguousIdentifierError as exc:
        return DisputeCaseLookupResult(status=LookupStatus.LOOKUP_FAILED, record=None, error=LookupError(error_code="ambiguous_identifier", message=str(exc)))
    except Exception as exc:  # noqa: BLE001
        return DisputeCaseLookupResult(status=LookupStatus.LOOKUP_FAILED, record=None, error=LookupError(error_code="provider_error", message=str(exc)))
    if outcome.record is None:
        return DisputeCaseLookupResult(status=LookupStatus.NO_RECORD, record=None, error=None)
    if not isinstance(outcome.record, DisputeCaseRecord):
        raise TypeError("provider returned a record of the wrong type for check_trusted_dispute_case_status")
    return DisputeCaseLookupResult(status=LookupStatus.FOUND, record=outcome.record, error=None)
