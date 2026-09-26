"""Tests for tell.agent.prompts: the decision prompt must be built only
from agent-visible tool results, with no evaluation-only or path
leakage, and no infrastructure-controlled field (like source_account_id)
that no tool ever returns. No GPU, no model.
"""

from __future__ import annotations

import json

import pytest

from tell.agent.prompts import build_decision_messages, build_system_prompt
from tell.agent.tools import (
    GetVendorRecordArgs,
    ReadEmailArgs,
    ReadInvoiceArgs,
    ToolStatus,
    get_vendor_record,
    read_email,
    read_invoice,
)
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, ScenarioBundle, load_scenario

MESSAGE_ID = "SIM-MSG-04D531CA-0001"
DOCUMENT_ID = "04d531ca811f448a91c6ff4e"
VENDOR_ID = "SIM-VENDOR-MDSPHARMA-0001"

FORBIDDEN_EVALUATION_MARKERS = [
    "is_attacked", "attack_surface", "expected_safe_action",
    "expected_beneficiary_account_id", "expected_amount",
    "expected_currency", "expected_outcome", "evaluation_labels", "probe_label",
]

FORBIDDEN_PATH_MARKERS = [
    "annotation_path", "/annotations/", "ocr_path", "/ocr/", "pdf_path",
    "/data/docile/", "/home/hp5/tell/data", "/home/hp5/tell/results",
]


@pytest.fixture(scope="module")
def bundle() -> ScenarioBundle:
    return load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)


@pytest.fixture()
def tool_results(bundle: ScenarioBundle):
    email_result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    invoice_result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))
    assert email_result.status == ToolStatus.SUCCESS
    assert invoice_result.status == ToolStatus.SUCCESS
    assert vendor_result.status == ToolStatus.SUCCESS
    return email_result, invoice_result, vendor_result


def test_prompt_built_only_from_agent_visible_tool_results(tool_results):
    email_result, invoice_result, vendor_result = tool_results
    messages = build_decision_messages(email_result=email_result, invoice_result=invoice_result, vendor_result=vendor_result)
    user_content = messages[1]["content"]

    expected_observations = {
        "read_email": email_result.content.model_dump(mode="json"),
        "read_invoice": invoice_result.content.model_dump(mode="json"),
        "get_vendor_record": vendor_result.content.model_dump(mode="json"),
    }
    assert json.dumps(expected_observations, indent=2) in user_content


def test_no_evaluation_only_leakage(tool_results):
    email_result, invoice_result, vendor_result = tool_results
    messages = build_decision_messages(email_result=email_result, invoice_result=invoice_result, vendor_result=vendor_result)
    full_text = json.dumps(messages)
    for marker in FORBIDDEN_EVALUATION_MARKERS:
        assert marker not in full_text, f"prompt leaked {marker}"


def test_no_raw_document_path_leakage(tool_results):
    email_result, invoice_result, vendor_result = tool_results
    messages = build_decision_messages(email_result=email_result, invoice_result=invoice_result, vendor_result=vendor_result)
    full_text = json.dumps(messages)
    for marker in FORBIDDEN_PATH_MARKERS:
        assert marker not in full_text, f"prompt leaked path marker {marker}"


def test_source_account_not_requested_since_not_agent_visible(tool_results, bundle: ScenarioBundle):
    email_result, invoice_result, vendor_result = tool_results
    messages = build_decision_messages(email_result=email_result, invoice_result=invoice_result, vendor_result=vendor_result)
    full_text = json.dumps(messages)
    # The trusted, infrastructure-controlled paying account is never
    # returned by any tool, so it must never appear in the prompt, and the
    # schema shown to the model must not ask for it either.
    assert bundle.trusted_state.company_account_id not in full_text
    assert "source_account_id" not in full_text


def test_trust_boundary_language_present():
    system_prompt = build_system_prompt()
    assert "UNTRUSTED" in system_prompt
    assert "TRUSTED" in system_prompt
    assert "beneficiary" in system_prompt.lower()
    assert "request_review" in system_prompt


def test_output_format_instructions_present():
    system_prompt = build_system_prompt()
    lowered = system_prompt.lower()
    assert "json" in lowered
    assert "markdown" in lowered or "code fence" in lowered


def test_beneficiary_in_prompt_comes_only_from_vendor_record(tool_results):
    email_result, invoice_result, vendor_result = tool_results
    messages = build_decision_messages(email_result=email_result, invoice_result=invoice_result, vendor_result=vendor_result)
    user_content = messages[1]["content"]
    beneficiary_id = vendor_result.content.beneficiary_account_id
    assert beneficiary_id in user_content
    # It must appear exactly under get_vendor_record's observation, not
    # fabricated elsewhere by this function.
    observations = json.loads(user_content.split("Tool results for this task:\n\n", 1)[1].rsplit("\n\nProduce", 1)[0])
    assert observations["get_vendor_record"]["beneficiary_account_id"] == beneficiary_id
    assert beneficiary_id not in json.dumps(observations["read_email"])
    assert beneficiary_id not in json.dumps(observations["read_invoice"])


def test_raises_on_failed_tool_result(bundle: ScenarioBundle):
    email_result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))
    failed_invoice = read_invoice(bundle, ReadInvoiceArgs(document_id="does-not-exist"))
    assert failed_invoice.status == ToolStatus.FAILURE
    with pytest.raises(ValueError):
        build_decision_messages(email_result=email_result, invoice_result=failed_invoice, vendor_result=vendor_result)
