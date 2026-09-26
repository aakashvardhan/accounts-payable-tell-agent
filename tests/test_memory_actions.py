"""Tests for the memory-pilot action types added to tell.agent.actions
(WriteMemoryAction, SearchMemoryAction, FinishReviewAction, and the
SessionAAction/SessionBAction unions). No GPU, no model.
"""

from __future__ import annotations

import json

import pytest

from tell.agent.actions import (
    ActionParseOutcome,
    FailClosedAction,
    FinishReviewAction,
    GetVendorRecordAction,
    ProposePaymentAction,
    ReadEmailAction,
    ReadInvoiceAction,
    RequestReviewAction,
    SearchMemoryAction,
    WriteMemoryAction,
    parse_agent_action,
    parse_session_a_action,
    parse_session_b_action,
    session_a_action_json_schema,
    session_b_action_json_schema,
)


def _dumps(d: dict) -> str:
    return json.dumps(d)


VALID_WRITE_MEMORY = {
    "action": "write_memory",
    "vendor_id": "SIM-VENDOR-MDSPHARMA-0001",
    "memory_kind": "operational_note",
    "content": "Include the invoice number in future correspondence.",
    "reason": "durable_operational_note",
}
VALID_SEARCH_MEMORY = {"action": "search_memory", "vendor_id": "SIM-VENDOR-MDSPHARMA-0001", "query": "invoice number"}
VALID_FINISH_REVIEW = {"action": "finish_review"}


def test_write_memory_action_parses_with_session_a_parser():
    result = parse_session_a_action(_dumps(VALID_WRITE_MEMORY))
    assert result.outcome is ActionParseOutcome.VALID
    assert isinstance(result.action, WriteMemoryAction)
    assert result.action.vendor_id == "SIM-VENDOR-MDSPHARMA-0001"


def test_search_memory_action_parses_with_session_b_parser():
    result = parse_session_b_action(_dumps(VALID_SEARCH_MEMORY))
    assert result.outcome is ActionParseOutcome.VALID
    assert isinstance(result.action, SearchMemoryAction)


def test_finish_review_action_parses_with_session_a_parser():
    result = parse_session_a_action(_dumps(VALID_FINISH_REVIEW))
    assert result.outcome is ActionParseOutcome.VALID
    assert isinstance(result.action, FinishReviewAction)


def test_write_memory_has_no_application_derived_fields():
    """The model cannot supply memory_id, created_at, status, or any
    origin_* field -- they don't exist on the schema, so extra="forbid"
    rejects any attempt."""
    forbidden_extra_fields = ["memory_id", "created_at", "status", "origin_source_type", "origin_source_id", "origin_provenance", "origin_trust_boundary"]
    for field in forbidden_extra_fields:
        payload = {**VALID_WRITE_MEMORY, field: "attacker-supplied-value"}
        result = parse_session_a_action(_dumps(payload))
        assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED, f"{field} should have been rejected"


def test_write_memory_action_schema_has_no_provenance_fields():
    dumped = json.dumps(session_a_action_json_schema())
    for forbidden in ("origin_trust_boundary", "origin_provenance", "memory_id", "created_at"):
        assert forbidden not in dumped


@pytest.mark.parametrize("missing_key", ["vendor_id", "memory_kind", "content", "reason"])
def test_write_memory_missing_required_field_rejected(missing_key):
    payload = {k: v for k, v in VALID_WRITE_MEMORY.items() if k != missing_key}
    result = parse_session_a_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_write_memory_unknown_field_rejected():
    payload = {**VALID_WRITE_MEMORY, "priority": "urgent"}
    result = parse_session_a_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_search_memory_unknown_field_rejected():
    payload = {**VALID_SEARCH_MEMORY, "limit": 10}
    result = parse_session_b_action(_dumps(payload))
    assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED


def test_search_memory_query_and_kind_are_optional():
    result = parse_session_b_action(_dumps({"action": "search_memory", "vendor_id": "V1"}))
    assert result.outcome is ActionParseOutcome.VALID
    assert result.action.query is None
    assert result.action.memory_kind is None


def test_session_a_rejects_payment_and_search_actions():
    for payload in (
        {"action": "propose_payment", "invoice_document_id": "x", "invoice_number": "y", "beneficiary_account_id": "z", "amount_minor_units": 1, "currency": "usd", "evidence": {"invoice_document_id": "x", "vendor_record_id": "v"}},
        {"action": "get_vendor_record", "vendor_id": "V1"},
        {"action": "read_invoice", "document_id": "D1"},
        {"action": "search_memory", "vendor_id": "V1"},
    ):
        result = parse_session_a_action(_dumps(payload))
        assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED, f"{payload['action']} should be rejected in session A"


def test_session_b_rejects_memory_write_and_finish_review_actions():
    for payload in (VALID_WRITE_MEMORY, VALID_FINISH_REVIEW):
        result = parse_session_b_action(_dumps(payload))
        assert result.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED, f"{payload['action']} should be rejected in session B"


def test_session_a_accepts_read_email_request_review_fail_closed():
    for payload in (
        {"action": "read_email", "message_id": "M1"},
        {"action": "request_review", "review_reason": "beneficiary_mismatch"},
        {"action": "fail_closed", "failure_reason": "insufficient_information"},
    ):
        result = parse_session_a_action(_dumps(payload))
        assert result.outcome is ActionParseOutcome.VALID


def test_session_b_accepts_all_seven_action_types():
    payloads = [
        {"action": "read_email", "message_id": "M1"},
        {"action": "read_invoice", "document_id": "D1"},
        {"action": "search_memory", "vendor_id": "V1"},
        {"action": "get_vendor_record", "vendor_id": "V1"},
        {
            "action": "propose_payment",
            "invoice_document_id": "D1",
            "invoice_number": "N1",
            "beneficiary_account_id": "B1",
            "amount_minor_units": 100,
            "currency": "usd",
            "evidence": {"invoice_document_id": "D1", "vendor_record_id": "V1"},
        },
        {"action": "request_review", "review_reason": "other"},
        {"action": "fail_closed", "failure_reason": "other"},
    ]
    for payload in payloads:
        result = parse_session_b_action(_dumps(payload))
        assert result.outcome is ActionParseOutcome.VALID, f"{payload['action']} unexpectedly rejected in session B"


def test_original_agent_action_contract_unaffected():
    """The original six-action loop (tell.agent.loop) contract must
    behave exactly as before this addition."""
    result = parse_agent_action(json.dumps({"action": "read_email", "message_id": "M1"}))
    assert result.outcome is ActionParseOutcome.VALID
    assert isinstance(result.action, ReadEmailAction)
    # write_memory/search_memory/finish_review are NOT part of AgentAction.
    for payload in (VALID_WRITE_MEMORY, VALID_SEARCH_MEMORY, VALID_FINISH_REVIEW):
        r = parse_agent_action(_dumps(payload))
        assert r.outcome is ActionParseOutcome.SCHEMA_VALIDATION_FAILED
