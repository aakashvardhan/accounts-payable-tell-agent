"""CPU-only tests for tell.agent.trusted_lookups (Part 1 of the
runtime-integration milestone). In-memory synthetic fixtures only; no real
SQLite database, no network call."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tell.agent.trusted_lookups import (
    AmbiguousIdentifierError,
    DisputeCaseLookupOutcome,
    DisputeCaseQuery,
    DisputeCaseRecord,
    DisputeCaseStatus,
    InvoicePaymentHistoryLookupOutcome,
    InvoicePaymentHistoryQuery,
    InvoicePaymentHistoryRecord,
    LookupStatus,
    PermittedResolutionClassification,
    PriorPaymentStatus,
    check_trusted_dispute_case_status,
    check_trusted_invoice_payment_history,
)
from tell.evaluation.scenario import OperationalProvenance, ProvenanceSource, SourceType, TrustBoundary


def _provenance(source_type: SourceType, source_id: str) -> OperationalProvenance:
    return OperationalProvenance(source_type=source_type, source_id=source_id, provenance=ProvenanceSource.SYNTHETIC_CONTROLLED, recorded_at="2024-01-01T00:00:00Z", trust_boundary=TrustBoundary.TRUSTED)


class InMemoryHistoryProvider:
    """Synthetic in-memory fixture -- no SQLite, no network."""

    def __init__(self, records: dict[tuple[str, str], InvoicePaymentHistoryRecord] | None = None, ambiguous_keys: set[tuple[str, str]] = frozenset(), error_keys: set[tuple[str, str]] = frozenset()):
        self.records = records or {}
        self.ambiguous_keys = ambiguous_keys
        self.error_keys = error_keys
        self.calls: list[InvoicePaymentHistoryQuery] = []

    def lookup(self, query: InvoicePaymentHistoryQuery) -> InvoicePaymentHistoryLookupOutcome:
        self.calls.append(query)
        key = (query.invoice_document_id, query.vendor_id)
        if key in self.ambiguous_keys:
            raise AmbiguousIdentifierError(f"multiple trusted records match {key}")
        if key in self.error_keys:
            raise RuntimeError("provider backend unavailable")
        return InvoicePaymentHistoryLookupOutcome(record=self.records.get(key))


class InMemoryDisputeProvider:
    def __init__(self, records: dict[tuple[str, str], DisputeCaseRecord] | None = None, ambiguous_keys: set[tuple[str, str]] = frozenset(), error_keys: set[tuple[str, str]] = frozenset()):
        self.records = records or {}
        self.ambiguous_keys = ambiguous_keys
        self.error_keys = error_keys
        self.calls: list[DisputeCaseQuery] = []

    def lookup(self, query: DisputeCaseQuery) -> DisputeCaseLookupOutcome:
        self.calls.append(query)
        key = (query.invoice_document_id, query.vendor_id)
        if key in self.ambiguous_keys:
            raise AmbiguousIdentifierError(f"multiple trusted cases match {key}")
        if key in self.error_keys:
            raise RuntimeError("provider backend unavailable")
        return DisputeCaseLookupOutcome(record=self.records.get(key))


PAID_RECORD = InvoicePaymentHistoryRecord(
    invoice_document_id="DOC-1", vendor_id="V1", prior_payment_status=PriorPaymentStatus.PAID,
    amount_minor_units=5000, currency="usd", payment_reference="REF-1", payment_timestamp="2024-01-01T00:00:00Z",
    record_id="HIST-1", provenance=_provenance(SourceType.INVOICE_PAYMENT_HISTORY, "HIST-1"),
)
OPEN_CASE = DisputeCaseRecord(
    case_id="CASE-1", invoice_document_id="DOC-1", vendor_id="V1", status=DisputeCaseStatus.OPEN,
    permitted_resolution_classification=PermittedResolutionClassification.REQUIRES_HUMAN_REVIEW,
    provenance=_provenance(SourceType.DISPUTE_CASE, "CASE-1"),
)


# ---------------------------------------------------------------------
# Read-only by construction
# ---------------------------------------------------------------------


def test_result_types_have_no_write_or_approval_field():
    for model in (InvoicePaymentHistoryRecord, DisputeCaseRecord):
        forbidden = {"approve", "approved", "write", "clear_alarm", "authorize", "execute_payment"}
        assert not (set(model.model_fields) & forbidden)


def test_query_rejects_malformed_identifiers():
    with pytest.raises(ValidationError):
        InvoicePaymentHistoryQuery(invoice_document_id="", vendor_id="V1")
    with pytest.raises(ValidationError):
        InvoicePaymentHistoryQuery(invoice_document_id="  DOC-1  ", vendor_id="V1")
    with pytest.raises(ValidationError):
        InvoicePaymentHistoryQuery(invoice_document_id="DOC 1", vendor_id="V1")
    with pytest.raises(ValidationError):
        DisputeCaseQuery(invoice_document_id="DOC-1", vendor_id="")


def test_query_does_not_accept_free_form_claims():
    assert set(InvoicePaymentHistoryQuery.model_fields) == {"invoice_document_id", "vendor_id"}
    assert set(DisputeCaseQuery.model_fields) == {"invoice_document_id", "vendor_id"}


# ---------------------------------------------------------------------
# Invoice/payment-history lookup: found / no-record / lookup-failed
# ---------------------------------------------------------------------


def test_history_lookup_found():
    provider = InMemoryHistoryProvider({("DOC-1", "V1"): PAID_RECORD})
    result = check_trusted_invoice_payment_history(provider, InvoicePaymentHistoryQuery(invoice_document_id="DOC-1", vendor_id="V1"))
    assert result.status is LookupStatus.FOUND
    assert result.record.prior_payment_status is PriorPaymentStatus.PAID
    assert result.record.provenance.trust_boundary is TrustBoundary.TRUSTED


def test_history_lookup_authoritative_no_record():
    provider = InMemoryHistoryProvider({})
    result = check_trusted_invoice_payment_history(provider, InvoicePaymentHistoryQuery(invoice_document_id="DOC-2", vendor_id="V1"))
    assert result.status is LookupStatus.NO_RECORD
    assert result.error is None


def test_history_lookup_ambiguous_is_lookup_failed_not_no_record():
    provider = InMemoryHistoryProvider(ambiguous_keys={("DOC-3", "V1")})
    result = check_trusted_invoice_payment_history(provider, InvoicePaymentHistoryQuery(invoice_document_id="DOC-3", vendor_id="V1"))
    assert result.status is LookupStatus.LOOKUP_FAILED
    assert result.error.error_code == "ambiguous_identifier"


def test_history_lookup_provider_error_is_lookup_failed_with_no_traceback():
    provider = InMemoryHistoryProvider(error_keys={("DOC-4", "V1")})
    result = check_trusted_invoice_payment_history(provider, InvoicePaymentHistoryQuery(invoice_document_id="DOC-4", vendor_id="V1"))
    assert result.status is LookupStatus.LOOKUP_FAILED
    assert result.error.error_code == "provider_error"
    assert "Traceback" not in result.error.message


def test_history_lookup_calls_provider_exactly_once():
    provider = InMemoryHistoryProvider({("DOC-1", "V1"): PAID_RECORD})
    check_trusted_invoice_payment_history(provider, InvoicePaymentHistoryQuery(invoice_document_id="DOC-1", vendor_id="V1"))
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------
# Dispute/case-status lookup
# ---------------------------------------------------------------------


def test_dispute_lookup_found():
    provider = InMemoryDisputeProvider({("DOC-1", "V1"): OPEN_CASE})
    result = check_trusted_dispute_case_status(provider, DisputeCaseQuery(invoice_document_id="DOC-1", vendor_id="V1"))
    assert result.status is LookupStatus.FOUND
    assert result.record.status is DisputeCaseStatus.OPEN
    assert result.record.permitted_resolution_classification is PermittedResolutionClassification.REQUIRES_HUMAN_REVIEW


def test_dispute_lookup_no_record():
    provider = InMemoryDisputeProvider({})
    result = check_trusted_dispute_case_status(provider, DisputeCaseQuery(invoice_document_id="DOC-9", vendor_id="V1"))
    assert result.status is LookupStatus.NO_RECORD


def test_dispute_lookup_ambiguous():
    provider = InMemoryDisputeProvider(ambiguous_keys={("DOC-1", "V1")})
    result = check_trusted_dispute_case_status(provider, DisputeCaseQuery(invoice_document_id="DOC-1", vendor_id="V1"))
    assert result.status is LookupStatus.LOOKUP_FAILED
    assert result.error.error_code == "ambiguous_identifier"


def test_no_network_or_database_reference_in_module():
    import inspect

    import tell.agent.trusted_lookups as mod

    source = inspect.getsource(mod)
    for banned in ("sqlite3", "requests", "socket", "http.client", "urllib"):
        assert banned not in source
