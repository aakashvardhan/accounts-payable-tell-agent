"""Strict, typed contract for Tell's model-backed payment decision.

This module defines exactly what the base model is allowed to decide
after reading `read_email`, `read_invoice`, and `get_vendor_record`
results, and how a raw model completion is turned into one of those typed
decisions -- or rejected. It implements no prompting, no model runtime,
and no ledger/gate wiring; see `tell.agent.prompts`, `tell.agent.local_model`,
and `tell.safety.gate` for those.

--------------------------------------------------------------------------
Field boundary: what the model may decide vs. what the application adds
--------------------------------------------------------------------------
`PayInvoiceCandidate` (tell.agent.tools) has five fields: invoice_id,
source_account_id, beneficiary_account_id, amount_minor_units, currency,
reason. Of those, `source_account_id` is never returned by any read-only
tool -- `scripts/run_clean_payment_phase.py` reads it directly off
`ScenarioBundle.trusted_state.company_account_id`, entirely outside the
agent's tool-visible surface. It is a trusted, infrastructure-controlled
fact (which bank account the *company* pays from), not a business
judgment the model is in a position to make or verify. Asking the model
to reproduce it would either (a) require leaking `trusted_state` into the
prompt just so the model could echo a value it has no way to
independently confirm, manufacturing a fake capability, or (b) let the
model invent an account id it never observed. Neither is acceptable, so
`ProposePaymentDecision` below deliberately has no `source_account_id`
field. The same reasoning excludes `reason` (a free-text audit string)
from `ProposePaymentDecision` -- see `to_pay_invoice_candidate`.

The decision fields the model IS asked for are exactly the business-level
facts it can observe in the three tool results: which invoice, which
approved beneficiary (from the trusted vendor record, never the invoice
or email), how much, in what currency, and which evidence sources back
each of those claims.

--------------------------------------------------------------------------
Fail-closed validation
--------------------------------------------------------------------------
`parse_model_decision` never repairs, retries, or partially accepts a
malformed response. Malformed JSON, missing required fields, unknown
fields (both submodels use `extra="forbid"`), an action outside the two
supported values, a non-positive amount, or a currency outside
`tell.payment.ledger.SUPPORTED_CURRENCIES` all produce the same outcome:
`decision is None` and an `outcome` other than `VALID`. There is no code
path from a non-`VALID` `ParsedDecisionResult` to a `PayInvoiceCandidate`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from tell.agent.tools import PayInvoiceCandidate
from tell.payment.ledger import SUPPORTED_CURRENCIES


class DecisionAction(str, Enum):
    PROPOSE_PAYMENT = "propose_payment"
    REQUEST_REVIEW = "request_review"


class ReviewReasonCode(str, Enum):
    """Typed reasons for declining to propose a payment. Deliberately not
    free text / chain-of-thought -- a closed set the evaluator and, later,
    the safety LoRA and gate can act on programmatically."""

    MISSING_REQUIRED_INFORMATION = "missing_required_information"
    CONFLICTING_VENDOR_IDENTITY = "conflicting_vendor_identity"
    VENDOR_NOT_VERIFIED = "vendor_not_verified"
    BENEFICIARY_MISMATCH = "beneficiary_mismatch"
    OTHER = "other"


class EvidenceReferences(BaseModel):
    """Pointers back to the two agent-visible sources that justify a
    propose_payment decision. Both ids are values the model already saw
    verbatim in its tool results (read_invoice's document_reference.
    document_id and get_vendor_record's vendor_id) -- this field asks the
    model to cite what it used, not to invent a new identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_document_id: str = Field(min_length=1)
    vendor_record_id: str = Field(min_length=1)


def _validate_currency(value: str) -> str:
    if value not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"currency must be a lowercase 3-letter code in {sorted(SUPPORTED_CURRENCIES)}, got {value!r}"
        )
    return value


class ProposePaymentDecision(BaseModel):
    """The model's decision to proceed with a clean payment. Contains only
    business-level fields the model could have observed in its tool
    results. Deliberately excludes any trusted, infrastructure-controlled
    paying-account field and any free-form reason/chain-of-thought --
    see this module's top-of-file docstring for the field-boundary
    rationale.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal[DecisionAction.PROPOSE_PAYMENT] = DecisionAction.PROPOSE_PAYMENT
    invoice_id: str = Field(min_length=1, description="The invoice identifier from read_invoice's document_reference.document_id.")
    beneficiary_account_id: str = Field(
        min_length=1,
        description="The beneficiary_account_id returned by get_vendor_record. Never taken from the email or invoice.",
    )
    amount_minor_units: int = Field(gt=0, description="The invoice amount, converted to integer minor units (e.g. cents).")
    currency: str = Field(description="Lowercase ISO 4217 currency code, e.g. 'usd'.")
    evidence: EvidenceReferences

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, value: str) -> str:
        return _validate_currency(value)


class RequestReviewDecision(BaseModel):
    """The model's decision to decline proposing a payment. Payment fields
    are structurally absent (not optional-and-null): `extra="forbid"`
    means a response that tries to include invoice_id, amount, etc. here
    fails validation rather than silently carrying them through."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal[DecisionAction.REQUEST_REVIEW] = DecisionAction.REQUEST_REVIEW
    review_reason: ReviewReasonCode


ModelDecision = Annotated[
    Union[ProposePaymentDecision, RequestReviewDecision],
    Field(discriminator="action"),
]

_DECISION_ADAPTER: TypeAdapter = TypeAdapter(ModelDecision)


def decision_json_schema() -> dict:
    """The JSON schema for `ModelDecision`, generated from the Pydantic
    models themselves so the prompt shown to the model and the validator
    that checks its output can never drift apart."""
    return _DECISION_ADAPTER.json_schema()


class DecisionParseOutcome(str, Enum):
    VALID = "valid"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"


@dataclass(frozen=True)
class ParsedDecisionResult:
    outcome: DecisionParseOutcome
    decision: ProposePaymentDecision | RequestReviewDecision | None
    error_message: str | None
    cleaned_text: str

    @property
    def is_valid(self) -> bool:
        return self.outcome is DecisionParseOutcome.VALID


def clean_raw_output(raw_text: str) -> str:
    """The only transformation applied to the model's raw completion
    before JSON parsing: stripping leading/trailing whitespace.

    Markdown code fences (``` or ```json ... ```) are deliberately NOT
    stripped. The prompt instructs the model to output JSON only, with no
    Markdown -- a fenced response is treated as a parse failure rather
    than silently repaired. See tests/test_decision.py for an explicit
    case exercising this rejection.
    """
    return raw_text.strip()


def parse_model_decision(raw_text: str) -> ParsedDecisionResult:
    """Parses and strictly validates one raw model completion. Never
    retries and never repairs; the caller (the experiment script) attempts
    generation exactly once and hands its single raw output here."""
    cleaned = clean_raw_output(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return ParsedDecisionResult(
            outcome=DecisionParseOutcome.MALFORMED_JSON,
            decision=None,
            error_message=str(exc),
            cleaned_text=cleaned,
        )

    try:
        decision = _DECISION_ADAPTER.validate_python(data)
    except ValidationError as exc:
        return ParsedDecisionResult(
            outcome=DecisionParseOutcome.SCHEMA_VALIDATION_FAILED,
            decision=None,
            error_message=str(exc),
            cleaned_text=cleaned,
        )

    return ParsedDecisionResult(
        outcome=DecisionParseOutcome.VALID,
        decision=decision,
        error_message=None,
        cleaned_text=cleaned,
    )


def to_pay_invoice_candidate(
    decision: ProposePaymentDecision,
    *,
    source_account_id: str,
    reason: str,
) -> PayInvoiceCandidate:
    """Converts a validated propose_payment decision into a
    PayInvoiceCandidate. Building a candidate has no side effect (see
    tell.agent.tools) -- it does not create a payment intent and does not
    touch the ledger or gate.

    `source_account_id` and `reason` are supplied by the *application*,
    never by the model: `source_account_id` is the trusted,
    infrastructure-controlled paying account that no tool ever exposes to
    the agent (see this module's docstring), and `reason` is
    operator/application-authored audit text, not model chain-of-thought.

    Only accepts an already-validated `ProposePaymentDecision` -- there is
    no overload for `RequestReviewDecision` or for `None`, so an invalid
    or review decision cannot be converted by construction, not merely by
    caller discipline.
    """
    if not isinstance(decision, ProposePaymentDecision):
        raise TypeError(
            f"to_pay_invoice_candidate requires a validated ProposePaymentDecision, got {type(decision).__name__}"
        )
    return PayInvoiceCandidate(
        invoice_id=decision.invoice_id,
        source_account_id=source_account_id,
        beneficiary_account_id=decision.beneficiary_account_id,
        amount_minor_units=decision.amount_minor_units,
        currency=decision.currency,
        reason=reason,
    )
