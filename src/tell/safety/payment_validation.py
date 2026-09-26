"""The always-on deterministic payment-policy validator (Part 1 of the
Tell-routing design; see results/routing_design/tell_agent_routing_v1.md).

Central correction this module exists to make
--------------------------------------------------------------------------
Before this module, the only deterministic thing standing between a
model-proposed payment and money moving was `tell.safety.gate.evaluate_gate`,
and that gate is *deliberately* unaware of the vendor record, the invoice,
or any other business fact (see its docstring) -- it only enforces the
alarm-resolution requirement. That means a workflow that never raised a
Tell alarm (`AlarmState.CLEAR`) had NO deterministic check at all that the
proposed beneficiary actually matched the trusted vendor record: the probe
alarm was, in effect, the only thing that could ever have caused
beneficiary validation, and only by accident (the gate still doesn't check
it even when raised). This module is the fix: it validates every proposed
payment against trusted, observed evidence, independent of the probe score
or alarm state, and runs whether or not Tell ever fires.

Responsibility boundary (do not blur this)
--------------------------------------------------------------------------
| Component                 | Responsibility                                    |
|----------------------------|---------------------------------------------------|
| This module                | Transaction and evidence integrity                 |
| Tell probe/router          | Decide whether Agent S must inspect (unrelated)    |
| tell.safety.resolution     | Decide whether trusted evidence resolves an alarm  |
| tell.safety.gate           | Block execution while alarm/review is unresolved   |
| tell.payment.ledger        | Financial invariants and execution                 |

This module does NOT decide routing, does NOT clear an alarm, and does NOT
execute anything. `validate_payment_proposal` below is a pure function: no
filesystem, no network, no ledger, no gate call. Its only output is a
`ValidationDecision`, which the caller (application code, never the model)
feeds to `tell.safety.gate.evaluate_gate` for the alarm-resolution check,
and only after a PERMIT there to `tell.payment.ledger.Ledger`.

Fields checked (Part 1's required list)
--------------------------------------------------------------------------
internal invoice document id, human-readable invoice number, vendor id,
approved beneficiary account, amount, currency, evidence references,
invoice completeness, vendor status, and pending clarification/verification
cases -- see `ValidationOutcome` for the exact reasons a proposal can fail.

Model/application field boundary
--------------------------------------------------------------------------
`ProposedPayment` is deliberately the only type the model's output can be
converted into on the way into this module -- it has no `source_account_id`
(never observable by the model; same reasoning as
`tell.agent.decision.ProposePaymentDecision`) and no status/validity field.
`ValidatedPayInvoiceCandidate` is deliberately a *different* type that
`ProposedPayment` cannot become except by passing through
`validate_payment_proposal` -- there is no method on `ProposedPayment` that
returns one, and no other function in this module constructs one. See
tests/test_payment_validation.py for tests proving the model cannot declare
validation success, cannot construct a validated candidate, and cannot
clear a pending case by itself.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from tell.agent.tools import PayInvoiceCandidate
from tell.payment.ledger import SUPPORTED_CURRENCIES


class VendorStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class VendorVerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    QUARANTINED = "quarantined"


class PendingCaseKind(str, Enum):
    """A case already open against this vendor/invoice pair that must
    resolve before any payment proposal for it can be validated -- Part 1
    requirement 4 ("... or unresolved case prevents candidate
    construction")."""

    AWAITING_VENDOR_CLARIFICATION = "awaiting_vendor_clarification"
    AWAITING_HUMAN_VERIFICATION = "awaiting_human_verification"


class TrustedVendorRecord(BaseModel):
    """Trusted, application-controlled vendor-master facts -- never derived
    from the untrusted email or invoice. Mirrors the fields already
    established by the frozen v2.2 corpus's vendor-record view
    (`scripts/enterprise_v2_2/contract.py:GetVendorRecordContentV22`,
    read-only reference; not imported, to keep `src/` independent of
    `scripts/`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor_id: str = Field(min_length=1)
    vendor_name: str
    beneficiary_account_id: str = Field(min_length=1)
    verification_status: VendorVerificationStatus
    vendor_status: VendorStatus
    approved_contact_email: str | None = None
    approved_contact_verified: bool = False

    @property
    def has_approved_contact(self) -> bool:
        return self.approved_contact_email is not None and self.approved_contact_verified


class TrustedInvoiceRecord(BaseModel):
    """Trusted, application-side record of what this invoice actually is --
    distinct from whatever the untrusted `read_invoice` tool result
    *claims*. `None` on `amount_minor_units`/`currency`/`invoice_number`
    means that field is not yet available (an ordinary processing gap, not
    a mismatch)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    invoice_number: str | None = None
    amount_minor_units: int | None = None
    currency: str | None = None


class ObservedEvidence(BaseModel):
    """What the agent's own tool calls actually returned in this run --
    used to catch a proposal that cites an id the agent never observed
    (`unobserved_identifier`) or that silently changes an observed value
    (`fabricated_evidence`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed_invoice_document_id: str | None = None
    observed_vendor_record_id: str | None = None
    observed_amount_minor_units: int | None = None
    observed_currency: str | None = None


class ProposedPayment(BaseModel):
    """Exactly what the model may propose -- business-level facts only,
    plus the two evidence citations the v2.2 action contract already
    requires (`ProposePaymentAction.evidence`). No status field, no
    validity claim, no beneficiary-authority claim: this type cannot
    represent "I already checked this."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    beneficiary_account_id: str = Field(min_length=1)
    amount_minor_units: int = Field(gt=0)
    currency: str
    evidence_invoice_document_id: str = Field(min_length=1)
    evidence_vendor_record_id: str = Field(min_length=1)


class ValidatedPayInvoiceCandidate(BaseModel):
    """The only typed object the executor may act on. Structurally
    convertible to `tell.agent.tools.PayInvoiceCandidate` (so the unchanged
    `tell.safety.gate.evaluate_gate` can consume it) but a DIFFERENT type
    from anything the model can produce -- `ProposedPayment` has no method
    that returns one. The only place in this codebase permitted to
    construct one is `validate_payment_proposal` below; see
    tests/test_payment_validation.py::test_only_validator_constructs_candidate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str
    source_account_id: str
    beneficiary_account_id: str
    amount_minor_units: int
    currency: str
    reason: str

    def to_pay_invoice_candidate(self) -> PayInvoiceCandidate:
        return PayInvoiceCandidate(
            invoice_id=self.invoice_document_id,
            source_account_id=self.source_account_id,
            beneficiary_account_id=self.beneficiary_account_id,
            amount_minor_units=self.amount_minor_units,
            currency=self.currency,
            reason=self.reason,
        )


class ValidationOutcome(str, Enum):
    VALID = "valid"
    UNRESOLVED_CASE = "unresolved_case"
    VENDOR_MISSING_OR_INACTIVE = "vendor_missing_or_inactive"
    VENDOR_UNVERIFIED = "vendor_unverified"
    FABRICATED_EVIDENCE = "fabricated_evidence"
    UNOBSERVED_IDENTIFIER = "unobserved_identifier"
    INVOICE_INCOMPLETE = "invoice_incomplete"
    BENEFICIARY_MISMATCH = "beneficiary_mismatch"
    AMOUNT_MISMATCH = "amount_mismatch"
    CURRENCY_MISMATCH = "currency_mismatch"
    UNSUPPORTED_CURRENCY = "unsupported_currency"


# Outcomes that reflect a normal processing gap (not a suspicious
# mismatch): eligible for vendor clarification when an approved contact
# exists. Every other non-VALID outcome routes to an evidence report --
# Part 1 requirements 5 and 6.
_CLARIFIABLE_OUTCOMES = frozenset({ValidationOutcome.INVOICE_INCOMPLETE})


class RequiredFollowUp(str, Enum):
    NONE = "none"
    REQUEST_VENDOR_CLARIFICATION = "request_vendor_clarification"
    SUBMIT_EVIDENCE_REPORT = "submit_evidence_report"


class ValidationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: ValidationOutcome
    candidate: ValidatedPayInvoiceCandidate | None
    required_follow_up: RequiredFollowUp
    reason_message: str

    @property
    def is_valid(self) -> bool:
        return self.outcome is ValidationOutcome.VALID


def _follow_up_for(outcome: ValidationOutcome, *, vendor: TrustedVendorRecord | None) -> RequiredFollowUp:
    if outcome is ValidationOutcome.VALID:
        return RequiredFollowUp.NONE
    if outcome in _CLARIFIABLE_OUTCOMES and vendor is not None and vendor.has_approved_contact:
        return RequiredFollowUp.REQUEST_VENDOR_CLARIFICATION
    return RequiredFollowUp.SUBMIT_EVIDENCE_REPORT


def validate_payment_proposal(
    proposal: ProposedPayment,
    *,
    trusted_vendor: TrustedVendorRecord | None,
    trusted_invoice: TrustedInvoiceRecord | None,
    observed: ObservedEvidence,
    pending_case: PendingCaseKind | None,
    source_account_id: str,
    reason: str,
) -> ValidationDecision:
    """Pure function: no filesystem, no network, no ledger, no gate call,
    and -- critically -- NO alarm-state parameter at all. This validator
    runs identically whether Tell's probe score is below or above
    threshold; see this module's docstring. Only a VALID outcome carries a
    `ValidatedPayInvoiceCandidate`; every other outcome carries `None` and
    a `required_follow_up` telling the caller whether a normal processing
    gap (vendor clarification) or a material/suspicious mismatch (evidence
    report) applies.
    """

    def blocked(outcome: ValidationOutcome, message: str) -> ValidationDecision:
        return ValidationDecision(
            outcome=outcome,
            candidate=None,
            required_follow_up=_follow_up_for(outcome, vendor=trusted_vendor),
            reason_message=message,
        )

    # 1. An unresolved case already blocks this vendor/invoice pair,
    #    independent of anything else the proposal says.
    if pending_case is not None:
        return blocked(
            ValidationOutcome.UNRESOLVED_CASE,
            f"A case is already pending ({pending_case.value}); no new payment proposal for this invoice may be validated until it resolves.",
        )

    # 2. Vendor must exist and be active.
    if trusted_vendor is None:
        return blocked(ValidationOutcome.VENDOR_MISSING_OR_INACTIVE, "No trusted vendor record exists for this proposal's vendor.")
    if trusted_vendor.vendor_status is not VendorStatus.ACTIVE:
        return blocked(ValidationOutcome.VENDOR_MISSING_OR_INACTIVE, f"Vendor {trusted_vendor.vendor_id} is not active (status={trusted_vendor.vendor_status.value}).")
    if trusted_vendor.verification_status is not VendorVerificationStatus.VERIFIED:
        return blocked(ValidationOutcome.VENDOR_UNVERIFIED, f"Vendor {trusted_vendor.vendor_id} verification_status={trusted_vendor.verification_status.value}, not verified.")

    # 3. Evidence must cite what the agent actually observed -- catches
    #    both fabrication (claims a value that contradicts what was
    #    observed) and an unobserved identifier (cites an id/record the
    #    agent never actually retrieved this run).
    if observed.observed_invoice_document_id is None or observed.observed_vendor_record_id is None:
        return blocked(ValidationOutcome.UNOBSERVED_IDENTIFIER, "Proposal was made before both read_invoice and get_vendor_record were observed this run.")
    if proposal.evidence_invoice_document_id != observed.observed_invoice_document_id or proposal.invoice_document_id != observed.observed_invoice_document_id:
        return blocked(ValidationOutcome.UNOBSERVED_IDENTIFIER, "invoice_document_id / evidence.invoice_document_id does not match the document_id actually observed via read_invoice this run.")
    if proposal.evidence_vendor_record_id != observed.observed_vendor_record_id or proposal.evidence_vendor_record_id != trusted_vendor.vendor_id:
        return blocked(ValidationOutcome.UNOBSERVED_IDENTIFIER, "evidence.vendor_record_id does not match the vendor_id actually observed via get_vendor_record this run.")
    if observed.observed_amount_minor_units is not None and observed.observed_amount_minor_units != proposal.amount_minor_units:
        return blocked(ValidationOutcome.FABRICATED_EVIDENCE, "Proposed amount does not match the amount actually observed via read_invoice this run.")
    if observed.observed_currency is not None and observed.observed_currency != proposal.currency:
        return blocked(ValidationOutcome.FABRICATED_EVIDENCE, "Proposed currency does not match the currency actually observed via read_invoice this run.")

    # 4. Invoice completeness, per the trusted (not merely observed) record.
    if trusted_invoice is None or trusted_invoice.invoice_document_id != proposal.invoice_document_id:
        return blocked(ValidationOutcome.INVOICE_INCOMPLETE, "No trusted invoice record matches this proposal's invoice_document_id.")
    if trusted_invoice.amount_minor_units is None or trusted_invoice.currency is None or not trusted_invoice.invoice_number:
        return blocked(ValidationOutcome.INVOICE_INCOMPLETE, "The trusted invoice record is missing a required field (amount, currency, or invoice number).")

    # 5. Currency: supported, and matches the trusted invoice record.
    if proposal.currency not in SUPPORTED_CURRENCIES:
        return blocked(ValidationOutcome.UNSUPPORTED_CURRENCY, f"Currency {proposal.currency!r} is not a supported currency.")
    if proposal.currency != trusted_invoice.currency:
        return blocked(ValidationOutcome.CURRENCY_MISMATCH, "Proposed currency does not match the trusted invoice record's currency.")

    # 6. Amount must match the trusted invoice record exactly.
    if proposal.amount_minor_units != trusted_invoice.amount_minor_units:
        return blocked(ValidationOutcome.AMOUNT_MISMATCH, "Proposed amount does not match the trusted invoice record's amount.")

    # 7. Beneficiary must match the trusted vendor record's approved
    #    beneficiary exactly -- this is the central correction: reached
    #    unconditionally, whether or not any Tell alarm ever fired.
    if proposal.beneficiary_account_id != trusted_vendor.beneficiary_account_id:
        return blocked(ValidationOutcome.BENEFICIARY_MISMATCH, "Proposed beneficiary_account_id does not match the trusted vendor record's approved beneficiary_account_id.")

    candidate = ValidatedPayInvoiceCandidate(
        invoice_document_id=proposal.invoice_document_id,
        source_account_id=source_account_id,
        beneficiary_account_id=trusted_vendor.beneficiary_account_id,
        amount_minor_units=trusted_invoice.amount_minor_units,
        currency=trusted_invoice.currency,
        reason=reason,
    )
    return ValidationDecision(
        outcome=ValidationOutcome.VALID,
        candidate=candidate,
        required_follow_up=RequiredFollowUp.NONE,
        reason_message="Proposal validated against trusted vendor and invoice records; no conflict found.",
    )
