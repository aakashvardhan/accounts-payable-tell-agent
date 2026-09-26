"""Tests for the strict, typed agent-action contract in
tell.agent.actions. No GPU, no model, no scenario I/O -- pure Pydantic
validation and parsing logic, mirroring tests/test_decision.py's rigor
for the six-action loop contract.
"""

from __future__ import annotations

import json

import pytest

from tell.agent.actions import (
    ActionParseOutcome,
    FailClosedAction,
    GetVendorRecordAction,
    ProposePaymentAction,
    ReadEmailAction,
    ReadInvoiceAction,
    RequestReviewAction,
    action_json_schema,
    clean_raw_action_output,
    is_read_action,
    is_terminal_action,
    parse_agent_action,
)

VALID_READ_EMAIL = {"action": "read_email", "message_id": "SIM-MSG-04D531CA-0001"}
VALID_READ_INVOICE = {"action": "read_invoice", "document_id": "04d531ca811f448a91c6ff4e"}
VALID_GET_VENDOR_RECORD = {"action": "get_vendor_record", "vendor_id": "SIM-VENDOR-MDSPHARMA-0001"}
VALID_PROPOSE_PAYMENT = {
    "action": "propose_payment",
    "invoice_document_id": "04d531ca811f448a91c6ff4e",
    "invoice_number": "33664",
    "beneficiary_account_id": "SIM-BENEFICIARY-ACCT-MDSPHARMA-0001",
    "amount_minor_units": 3_000_000,
    "currency": "usd",
    "evidence": {"invoice_document_id": "04d531ca811f448a91c6ff4e", "vendor_record_id": "SIM-VENDOR-MDSPHARMA-0001"},
}
VALID_REQUEST_REVIEW = {"action": "request_review", "review_reason": "beneficiary_mismatch"}
VALID_FAIL_CLOSED = {"action": "fail_closed", "failure_reason": "unresolvable_conflict"}


def _dumps(d: dict) -> str:
    return json.dumps(d)


@pytest.mark.parametrize(
    "payload,expected_type",
    [
        (VALID_READ_EMAIL, ReadEmailAction),
        (VALID_READ_INVOICE, ReadInvoiceAction),
        (VALID_GET_VENDOR_RECORD, GetVendorRecordAction),
        (VALID_PROPOSE_PAYMENT, ProposePaymentAction),
        (VALID_REQUEST_REVIEW, RequestReviewAction),
        (VALID_FAIL_CLOSED, FailClosedAction),
    ],
)
def test_valid_action_of_each_type_parses(payload, expected_type):
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.VALID
    assert isinstance(result.action, expected_type)


def test_read_actions_are_read_and_not_terminal():
    for payload in (VALID_READ_EMAIL, VALID_READ_INVOICE, VALID_GET_VENDOR_RECORD):
        result = parse_agent_action(_dumps(payload))
        assert is_read_action(result.action) is True
        assert is_terminal_action(result.action) is False


def test_terminal_actions_are_terminal_and_not_read():
    for payload in (VALID_PROPOSE_PAYMENT, VALID_REQUEST_REVIEW, VALID_FAIL_CLOSED):
        result = parse_agent_action(_dumps(payload))
        assert is_terminal_action(result.action) is True
        assert is_read_action(result.action) is False


def test_malformed_json_rejected():
    result = parse_agent_action("{not valid")
    assert result.outcome is ActionParseOutcome.MALFORMED_JSON
    assert result.action is None


def test_unknown_action_rejected():
    result = parse_agent_action(_dumps({"action": "delete_invoice", "id": "x"}))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED
    assert result.action is None


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_READ_EMAIL, "extra_field": "x"},
        {**VALID_PROPOSE_PAYMENT, "source_account_id": "SIM-COMPANY-ACCT-0001"},
        {**VALID_REQUEST_REVIEW, "amount_minor_units": 100},
    ],
)
def test_unknown_fields_rejected(payload):
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize("missing_key", ["message_id"])
def test_missing_required_field_rejected_read_email(missing_key):
    payload = {k: v for k, v in VALID_READ_EMAIL.items() if k != missing_key}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_missing_evidence_rejected_on_propose_payment():
    payload = {k: v for k, v in VALID_PROPOSE_PAYMENT.items() if k != "evidence"}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize("bad_amount", [0, -1])
def test_invalid_amount_rejected(bad_amount):
    payload = {**VALID_PROPOSE_PAYMENT, "amount_minor_units": bad_amount}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize("bad_currency", ["USD", "dollars", ""])
def test_invalid_currency_rejected(bad_currency):
    payload = {**VALID_PROPOSE_PAYMENT, "currency": bad_currency}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_markdown_fences_rejected_not_repaired():
    fenced = "```json\n" + _dumps(VALID_READ_EMAIL) + "\n```"
    result = parse_agent_action(fenced)
    assert result.outcome is ActionParseOutcome.MALFORMED_JSON


def test_clean_raw_action_output_only_strips_whitespace():
    assert clean_raw_action_output("  {\"a\": 1}  \n") == '{"a": 1}'
    fenced = "```json\n{}\n```"
    assert clean_raw_action_output(fenced) == fenced


def test_action_schema_has_no_source_account_field():
    schema = action_json_schema()
    dumped = json.dumps(schema)
    assert "source_account_id" not in dumped


def test_action_schema_contains_all_six_action_names():
    dumped = json.dumps(action_json_schema())
    for name in ("read_email", "read_invoice", "get_vendor_record", "propose_payment", "request_review", "fail_closed"):
        assert name in dumped


# ---------------------------------------------------------------------
# Version-2 identifier correction: invoice_document_id vs invoice_number
# ---------------------------------------------------------------------


def test_document_id_and_invoice_number_are_distinct_required_fields():
    result = parse_agent_action(_dumps(VALID_PROPOSE_PAYMENT))
    assert result.outcome is ActionParseOutcome.VALID
    assert result.action.invoice_document_id == "04d531ca811f448a91c6ff4e"
    assert result.action.invoice_number == "33664"
    assert result.action.invoice_document_id != result.action.invoice_number


def test_invoice_document_id_required():
    payload = {k: v for k, v in VALID_PROPOSE_PAYMENT.items() if k != "invoice_document_id"}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_invoice_number_required():
    payload = {k: v for k, v in VALID_PROPOSE_PAYMENT.items() if k != "invoice_number"}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_legacy_invoice_id_field_rejected_in_v2():
    legacy_payload = {
        "action": "propose_payment",
        "invoice_id": "33664",  # the old, now-removed field name
        "beneficiary_account_id": "SIM-BENEFICIARY-ACCT-MDSPHARMA-0001",
        "amount_minor_units": 3_000_000,
        "currency": "usd",
        "evidence": {"invoice_document_id": "04d531ca811f448a91c6ff4e", "vendor_record_id": "SIM-VENDOR-MDSPHARMA-0001"},
    }
    result = parse_agent_action(_dumps(legacy_payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED
    assert not hasattr(ProposePaymentAction, "invoice_id")


def test_mismatched_top_level_and_evidence_document_id_rejected():
    payload = {**VALID_PROPOSE_PAYMENT, "evidence": {"invoice_document_id": "SOME-OTHER-DOC-ID", "vendor_record_id": "SIM-VENDOR-MDSPHARMA-0001"}}
    result = parse_agent_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_source_account_still_absent_from_v2_schema_and_output():
    result = parse_agent_action(_dumps(VALID_PROPOSE_PAYMENT))
    assert not hasattr(result.action, "source_account_id")
    assert "source_account_id" not in json.dumps(action_json_schema())
