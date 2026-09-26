"""Tests for the strict, typed model-decision contract in
tell.agent.decision. No GPU, no model, no scenario I/O -- pure Pydantic
validation and parsing logic.
"""

from __future__ import annotations

import json

import pytest

from tell.agent.decision import (
    DecisionParseOutcome,
    ProposePaymentDecision,
    RequestReviewDecision,
    ReviewReasonCode,
    clean_raw_output,
    decision_json_schema,
    parse_model_decision,
    to_pay_invoice_candidate,
)
from tell.agent.tools import PayInvoiceCandidate

VALID_PROPOSE = {
    "action": "propose_payment",
    "invoice_id": "04d531ca811f448a91c6ff4e",
    "beneficiary_account_id": "SIM-BENEFICIARY-ACCT-MDSPHARMA-0001",
    "amount_minor_units": 3_000_000,
    "currency": "usd",
    "evidence": {
        "invoice_document_id": "04d531ca811f448a91c6ff4e",
        "vendor_record_id": "SIM-VENDOR-MDSPHARMA-0001",
    },
}

VALID_REVIEW = {"action": "request_review", "review_reason": "missing_required_information"}


def _dumps(d: dict) -> str:
    return json.dumps(d)


def test_valid_propose_payment_parses():
    result = parse_model_decision(_dumps(VALID_PROPOSE))
    assert result.outcome is DecisionParseOutcome.VALID
    assert isinstance(result.decision, ProposePaymentDecision)
    assert result.decision.invoice_id == VALID_PROPOSE["invoice_id"]
    assert result.decision.amount_minor_units == 3_000_000
    assert result.decision.currency == "usd"
    assert result.error_message is None


def test_valid_request_review_parses():
    result = parse_model_decision(_dumps(VALID_REVIEW))
    assert result.outcome is DecisionParseOutcome.VALID
    assert isinstance(result.decision, RequestReviewDecision)
    assert result.decision.review_reason is ReviewReasonCode.MISSING_REQUIRED_INFORMATION


def test_malformed_json_rejected():
    result = parse_model_decision("{not valid json")
    assert result.outcome is DecisionParseOutcome.MALFORMED_JSON
    assert result.decision is None
    assert result.error_message is not None


def test_unknown_fields_rejected_on_propose_payment():
    bad = {**VALID_PROPOSE, "source_account_id": "SIM-COMPANY-ACCT-0001"}
    result = parse_model_decision(_dumps(bad))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED
    assert result.decision is None


def test_unknown_fields_rejected_on_request_review():
    bad = {**VALID_REVIEW, "amount_minor_units": 3_000_000}
    result = parse_model_decision(_dumps(bad))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED
    assert result.decision is None


@pytest.mark.parametrize("missing_key", ["invoice_id", "beneficiary_account_id", "amount_minor_units", "currency", "evidence"])
def test_missing_required_fields_rejected(missing_key):
    bad = {k: v for k, v in VALID_PROPOSE.items() if k != missing_key}
    result = parse_model_decision(_dumps(bad))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED
    assert result.decision is None


def test_request_review_missing_reason_rejected():
    result = parse_model_decision(_dumps({"action": "request_review"}))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize("bad_amount", [0, -1, -3_000_000])
def test_invalid_amounts_rejected(bad_amount):
    bad = {**VALID_PROPOSE, "amount_minor_units": bad_amount}
    result = parse_model_decision(_dumps(bad))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize("bad_currency", ["USD", "us", "dollars", "", "usdd"])
def test_invalid_currencies_rejected(bad_currency):
    bad = {**VALID_PROPOSE, "currency": bad_currency}
    result = parse_model_decision(_dumps(bad))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_unsupported_action_rejected():
    bad = {**VALID_PROPOSE, "action": "cancel_invoice"}
    result = parse_model_decision(_dumps(bad))
    assert result.outcome is DecisionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_markdown_fences_are_rejected_not_repaired():
    fenced = "```json\n" + _dumps(VALID_PROPOSE) + "\n```"
    result = parse_model_decision(fenced)
    assert result.outcome is DecisionParseOutcome.MALFORMED_JSON
    assert result.decision is None


def test_clean_raw_output_only_strips_whitespace():
    assert clean_raw_output("  {\"a\": 1}  \n") == '{"a": 1}'
    fenced = "```json\n{}\n```"
    assert clean_raw_output(fenced) == fenced  # fences are NOT stripped


def test_invalid_model_output_cannot_become_a_pay_invoice_candidate():
    result = parse_model_decision("not json at all")
    assert result.decision is None
    with pytest.raises(TypeError):
        to_pay_invoice_candidate(result.decision, source_account_id="SIM-COMPANY-ACCT-0001", reason="x")


def test_request_review_decision_cannot_become_a_pay_invoice_candidate():
    result = parse_model_decision(_dumps(VALID_REVIEW))
    assert isinstance(result.decision, RequestReviewDecision)
    with pytest.raises(TypeError):
        to_pay_invoice_candidate(result.decision, source_account_id="SIM-COMPANY-ACCT-0001", reason="x")


def test_valid_propose_payment_converts_to_pay_invoice_candidate():
    result = parse_model_decision(_dumps(VALID_PROPOSE))
    candidate = to_pay_invoice_candidate(
        result.decision, source_account_id="SIM-COMPANY-ACCT-0001", reason="model-proposed clean payment"
    )
    assert isinstance(candidate, PayInvoiceCandidate)
    assert candidate.source_account_id == "SIM-COMPANY-ACCT-0001"
    assert candidate.beneficiary_account_id == VALID_PROPOSE["beneficiary_account_id"]
    assert candidate.amount_minor_units == 3_000_000
    assert candidate.currency == "usd"


def test_propose_payment_decision_has_no_source_account_field():
    result = parse_model_decision(_dumps(VALID_PROPOSE))
    assert not hasattr(result.decision, "source_account_id")
    schema = decision_json_schema()
    assert "source_account_id" not in json.dumps(schema)


def test_decision_json_schema_is_generated_from_the_models():
    schema = decision_json_schema()
    dumped = json.dumps(schema)
    assert "propose_payment" in dumped
    assert "request_review" in dumped
    assert "amount_minor_units" in dumped
