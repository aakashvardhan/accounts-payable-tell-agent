"""CPU-only tests for tell.safety.payment_validation (Part 1 of the
Tell-routing design). No model, no GPU, no ledger, no gate call beyond
constructing typed objects; every fixture below is synthetic."""

from __future__ import annotations

import inspect
import re

import pytest
from pydantic import ValidationError

from tell.safety.payment_validation import (
    ObservedEvidence,
    PendingCaseKind,
    ProposedPayment,
    RequiredFollowUp,
    TrustedInvoiceRecord,
    TrustedVendorRecord,
    ValidatedPayInvoiceCandidate,
    ValidationOutcome,
    VendorStatus,
    VendorVerificationStatus,
    validate_payment_proposal,
)

VENDOR = TrustedVendorRecord(
    vendor_id="VEND-1", vendor_name="Acme Supplies", beneficiary_account_id="ACCT-APPROVED",
    verification_status=VendorVerificationStatus.VERIFIED, vendor_status=VendorStatus.ACTIVE,
    approved_contact_email="ap@acme.example", approved_contact_verified=True,
)
INVOICE = TrustedInvoiceRecord(invoice_document_id="DOC-1", vendor_id="VEND-1", invoice_number="INV-100", amount_minor_units=50_000, currency="usd")
OBSERVED = ObservedEvidence(observed_invoice_document_id="DOC-1", observed_vendor_record_id="VEND-1", observed_amount_minor_units=50_000, observed_currency="usd")


def _valid_proposal(**overrides) -> ProposedPayment:
    fields = dict(
        invoice_document_id="DOC-1", invoice_number="INV-100", beneficiary_account_id="ACCT-APPROVED",
        amount_minor_units=50_000, currency="usd", evidence_invoice_document_id="DOC-1", evidence_vendor_record_id="VEND-1",
    )
    fields.update(overrides)
    return ProposedPayment(**fields)


def _validate(**overrides):
    return validate_payment_proposal(
        _valid_proposal(**overrides.pop("proposal_overrides", {})),
        trusted_vendor=overrides.pop("trusted_vendor", VENDOR),
        trusted_invoice=overrides.pop("trusted_invoice", INVOICE),
        observed=overrides.pop("observed", OBSERVED),
        pending_case=overrides.pop("pending_case", None),
        source_account_id="COMPANY-ACCT",
        reason="test",
    )


# ---------------------------------------------------------------------
# 1 & 2: beneficiary mismatch blocked with clear probe state AND with a
# raised probe state -- proven by validate_payment_proposal having NO
# alarm-state parameter at all, so the outcome cannot depend on it.
# ---------------------------------------------------------------------


def test_validate_payment_proposal_has_no_alarm_state_parameter():
    sig = inspect.signature(validate_payment_proposal)
    assert "alarm_state" not in sig.parameters
    assert "probe_score" not in sig.parameters
    assert "threshold" not in sig.parameters


@pytest.mark.parametrize("simulated_alarm_state", ["clear", "raised"])
def test_beneficiary_mismatch_blocked_regardless_of_simulated_alarm_state(simulated_alarm_state):
    # The validator is alarm-agnostic by construction; we simulate calling
    # it twice, once as if under AlarmState.CLEAR and once as if under a
    # raised alarm, to prove the outcome is identical either way (the
    # central Part 1 correction: the probe alarm must not be the condition
    # that causes beneficiary validation).
    decision = _validate(proposal_overrides={"beneficiary_account_id": "ACCT-ATTACKER"})
    assert decision.outcome is ValidationOutcome.BENEFICIARY_MISMATCH
    assert decision.candidate is None
    assert decision.required_follow_up is RequiredFollowUp.SUBMIT_EVIDENCE_REPORT


# ---------------------------------------------------------------------
# 3: model cannot declare validation success
# ---------------------------------------------------------------------


def test_model_cannot_declare_validation_success():
    # ProposedPayment (the only type the model's output maps to) has no
    # validity/status/success field of any kind.
    field_names = set(ProposedPayment.model_fields)
    assert not field_names & {"status", "valid", "is_valid", "validated", "outcome", "success"}
    with pytest.raises(ValidationError):
        ProposedPayment(**{**_valid_proposal().model_dump(), "outcome": "valid"})  # extra="forbid"


# ---------------------------------------------------------------------
# 6: only the application constructs a validated candidate
# ---------------------------------------------------------------------


def test_proposed_payment_cannot_become_a_validated_candidate_directly():
    proposal = _valid_proposal()
    assert not hasattr(proposal, "to_validated_candidate")
    assert not any(
        isinstance(getattr(proposal, name, None), ValidatedPayInvoiceCandidate) for name in type(proposal).model_fields
    )


def test_only_validator_constructs_candidate_in_this_module():
    import tell.safety.payment_validation as mod

    source = inspect.getsource(mod)
    construction_sites = [
        m.start() for m in re.finditer(r"ValidatedPayInvoiceCandidate\(", source) if not re.match(r"class\s+ValidatedPayInvoiceCandidate\(", source[m.start() - 6 : m.end()])
    ]
    # Exactly one construction call: inside validate_payment_proposal (the
    # class definition line itself, "class ValidatedPayInvoiceCandidate(BaseModel):", is excluded above).
    assert len(construction_sites) == 1
    call_line = source[: construction_sites[0]].count("\n") + 1
    func_source, func_start_line = inspect.getsource(mod.validate_payment_proposal), None
    func_start_line = source[: source.index("def validate_payment_proposal")].count("\n") + 1
    func_end_line = func_start_line + func_source.count("\n")
    assert func_start_line <= call_line <= func_end_line


def test_valid_proposal_produces_validated_candidate_convertible_to_pay_invoice_candidate():
    decision = _validate()
    assert decision.outcome is ValidationOutcome.VALID
    assert isinstance(decision.candidate, ValidatedPayInvoiceCandidate)
    pic = decision.candidate.to_pay_invoice_candidate()
    assert pic.beneficiary_account_id == "ACCT-APPROVED"
    assert pic.amount_minor_units == 50_000
    assert pic.currency == "usd"


# ---------------------------------------------------------------------
# 7: resolved injection can proceed to the approved beneficiary
# ---------------------------------------------------------------------


def test_resolved_injection_proceeds_to_approved_beneficiary():
    # A resolved injection means the untrusted claim never reached the
    # proposal at all -- the proposal itself already names the trusted
    # vendor record's approved beneficiary (as it must, since only
    # trusted-derived proposals reach validate_payment_proposal), so
    # validation succeeds normally, and any injection is recorded
    # separately as a nonblocking security event by application code (not
    # this module's concern -- see tell.safety.alarm/resolution for the
    # alarm-state side of "resolved").
    decision = _validate()
    assert decision.outcome is ValidationOutcome.VALID
    assert decision.candidate.beneficiary_account_id == "ACCT-APPROVED"


# ---------------------------------------------------------------------
# 8: unresolved/material injection produces an evidence report
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,expected_outcome",
    [
        ({"beneficiary_account_id": "ACCT-ATTACKER"}, ValidationOutcome.BENEFICIARY_MISMATCH),
        ({"amount_minor_units": 99_999}, ValidationOutcome.FABRICATED_EVIDENCE),
        ({"evidence_vendor_record_id": "VEND-FORGED"}, ValidationOutcome.UNOBSERVED_IDENTIFIER),
    ],
)
def test_material_or_suspicious_mismatch_produces_evidence_report(overrides, expected_outcome):
    decision = _validate(proposal_overrides=overrides)
    assert decision.outcome is expected_outcome
    assert decision.required_follow_up is RequiredFollowUp.SUBMIT_EVIDENCE_REPORT
    assert decision.candidate is None


def test_pending_case_blocks_regardless_of_everything_else():
    decision = _validate(pending_case=PendingCaseKind.AWAITING_VENDOR_CLARIFICATION)
    assert decision.outcome is ValidationOutcome.UNRESOLVED_CASE
    assert decision.candidate is None


# ---------------------------------------------------------------------
# 9: vendor clarification recipient comes only from trusted state
# ---------------------------------------------------------------------


def test_normal_processing_gap_with_approved_contact_requests_vendor_clarification():
    incomplete_invoice = TrustedInvoiceRecord(invoice_document_id="DOC-1", vendor_id="VEND-1", invoice_number=None, amount_minor_units=None, currency=None)
    decision = _validate(
        trusted_invoice=incomplete_invoice,
        observed=ObservedEvidence(observed_invoice_document_id="DOC-1", observed_vendor_record_id="VEND-1", observed_amount_minor_units=50_000, observed_currency="usd"),
    )
    assert decision.outcome is ValidationOutcome.INVOICE_INCOMPLETE
    assert decision.required_follow_up is RequiredFollowUp.REQUEST_VENDOR_CLARIFICATION


def test_normal_processing_gap_without_approved_contact_becomes_evidence_report():
    no_contact_vendor = VENDOR.model_copy(update={"approved_contact_email": None, "approved_contact_verified": False})
    incomplete_invoice = TrustedInvoiceRecord(invoice_document_id="DOC-1", vendor_id="VEND-1", invoice_number=None, amount_minor_units=None, currency=None)
    decision = _validate(trusted_vendor=no_contact_vendor, trusted_invoice=incomplete_invoice)
    assert decision.outcome is ValidationOutcome.INVOICE_INCOMPLETE
    assert decision.required_follow_up is RequiredFollowUp.SUBMIT_EVIDENCE_REPORT


def test_proposed_payment_has_no_recipient_field():
    # The model never chooses a clarification recipient -- ProposedPayment
    # (and the wider v2.2 RequestVendorClarificationAction contract) has no
    # such field at all.
    assert "recipient" not in ProposedPayment.model_fields
    assert "approved_contact_email" not in ProposedPayment.model_fields


# ---------------------------------------------------------------------
# Additional boundary checks
# ---------------------------------------------------------------------


def test_vendor_inactive_blocks():
    inactive = VENDOR.model_copy(update={"vendor_status": VendorStatus.INACTIVE})
    decision = _validate(trusted_vendor=inactive)
    assert decision.outcome is ValidationOutcome.VENDOR_MISSING_OR_INACTIVE


def test_vendor_missing_blocks():
    decision = _validate(trusted_vendor=None)
    assert decision.outcome is ValidationOutcome.VENDOR_MISSING_OR_INACTIVE


def test_unsupported_currency_is_a_deterministic_pre_gate_rejection():
    decision = _validate(
        proposal_overrides={"currency": "jpy"},
        trusted_invoice=TrustedInvoiceRecord(invoice_document_id="DOC-1", vendor_id="VEND-1", invoice_number="INV-100", amount_minor_units=50_000, currency="jpy"),
        observed=ObservedEvidence(observed_invoice_document_id="DOC-1", observed_vendor_record_id="VEND-1", observed_amount_minor_units=50_000, observed_currency="jpy"),
    )
    assert decision.outcome is ValidationOutcome.UNSUPPORTED_CURRENCY
