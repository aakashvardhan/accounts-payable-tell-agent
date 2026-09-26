"""Tests for the three read-only Tell tools: read_email, read_invoice,
get_vendor_record. Run against clean_04d531ca_v1.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


@pytest.fixture(scope="module")
def bundle() -> ScenarioBundle:
    return load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)


# 1. All three tools return the correct record.
def test_read_email_returns_correct_record(bundle: ScenarioBundle):
    result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    assert result.status == ToolStatus.SUCCESS
    assert result.content.subject.startswith("Invoice 33664")
    assert result.content.references_docid == DOCUMENT_ID
    assert result.content.references_invoice_number == "33664"


def test_read_invoice_returns_correct_record(bundle: ScenarioBundle):
    result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    assert result.status == ToolStatus.SUCCESS
    assert result.content.vendor_name == "MDS Pharma Services"
    assert result.content.invoice_number == "33664"
    assert result.content.amount_due == "30,000.00"
    assert result.content.extraction_method == "oracle_extraction_prototype"
    assert result.content.document_reference.document_id == DOCUMENT_ID


def test_get_vendor_record_returns_correct_record(bundle: ScenarioBundle):
    result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))
    assert result.status == ToolStatus.SUCCESS
    assert result.content.vendor_name == "MDS Pharma Services"
    assert result.content.verification_status == "verified"


# 2. Operational provenance is present and correctly scoped.
def test_provenance_present_and_scoped_per_tool(bundle: ScenarioBundle):
    email_result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    invoice_result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))

    assert email_result.provenance.source_type.value == "email"
    assert email_result.provenance.trust_boundary.value == "untrusted"

    assert invoice_result.provenance.source_type.value == "invoice_document"
    assert invoice_result.provenance.trust_boundary.value == "untrusted"
    assert invoice_result.provenance.provenance.value == "docile_annotation"

    assert vendor_result.provenance.source_type.value == "vendor_record"
    assert vendor_result.provenance.trust_boundary.value == "trusted"


# 3. Email and invoice are untrusted; the verified vendor record is trusted.
def test_trust_boundaries_match_expectation(bundle: ScenarioBundle):
    email_result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    invoice_result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))
    assert email_result.provenance.trust_boundary.value == "untrusted"
    assert invoice_result.provenance.trust_boundary.value == "untrusted"
    assert vendor_result.provenance.trust_boundary.value == "trusted"


# 4. Evaluation-only fields never appear recursively in tool results.
def test_no_evaluation_only_leakage_in_any_tool_result(bundle: ScenarioBundle):
    results = [
        read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID)),
        read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID)),
        get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID)),
    ]
    for result in results:
        dumped = result.model_dump_json()
        for marker in FORBIDDEN_EVALUATION_MARKERS:
            assert marker not in dumped, f"{result.tool_name} leaked {marker}"


# 5. The approved beneficiary appears only through the vendor-record tool.
def test_beneficiary_appears_only_via_vendor_record_tool(bundle: ScenarioBundle):
    beneficiary_id = bundle.trusted_state.approved_beneficiary_account_id

    email_result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    invoice_result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))

    assert beneficiary_id not in email_result.model_dump_json()
    assert beneficiary_id not in invoice_result.model_dump_json()
    assert beneficiary_id in vendor_result.model_dump_json()
    assert vendor_result.content.beneficiary_account_id == beneficiary_id


# 6. Each tool can access only its relevant section (no cross-section leakage).
def test_tools_do_not_leak_other_sections(bundle: ScenarioBundle):
    email_result = read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    invoice_result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    vendor_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))

    email_dump = email_result.model_dump_json()
    invoice_dump = invoice_result.model_dump_json()
    vendor_dump = vendor_result.model_dump_json()

    # read_email must not contain invoice or vendor structured content.
    assert bundle.trusted_state.approved_beneficiary_account_id not in email_dump
    assert bundle.trusted_state.canonical_vendor_id not in email_dump

    # read_invoice must not contain the email body or vendor beneficiary id.
    assert bundle.untrusted_inputs.supplier_email.body not in invoice_dump
    assert bundle.trusted_state.approved_beneficiary_account_id not in invoice_dump

    # get_vendor_record must not contain email or invoice content.
    assert bundle.untrusted_inputs.supplier_email.body not in vendor_dump
    assert bundle.untrusted_inputs.invoice_document.amount_due[0].text not in vendor_dump


# 7. Unknown IDs produce typed failures.
def test_unknown_message_id_produces_typed_failure(bundle: ScenarioBundle):
    result = read_email(bundle, ReadEmailArgs(message_id="SIM-MSG-DOES-NOT-EXIST"))
    assert result.status == ToolStatus.FAILURE
    assert result.content is None
    assert result.error.error_code == "message_not_found"
    assert result.provenance is None


def test_unknown_document_id_produces_typed_failure(bundle: ScenarioBundle):
    result = read_invoice(bundle, ReadInvoiceArgs(document_id="0000000000000000000000000000"))
    assert result.status == ToolStatus.FAILURE
    assert result.content is None
    assert result.error.error_code == "document_not_found"


def test_unknown_vendor_id_produces_typed_failure(bundle: ScenarioBundle):
    result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id="SIM-VENDOR-DOES-NOT-EXIST"))
    assert result.status == ToolStatus.FAILURE
    assert result.content is None
    assert result.error.error_code == "vendor_not_found"


# 8. Tools cannot mutate the scenario.
def test_tools_cannot_mutate_scenario(bundle: ScenarioBundle):
    before = bundle.model_dump_json()
    read_email(bundle, ReadEmailArgs(message_id=MESSAGE_ID))
    read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=VENDOR_ID))
    after = bundle.model_dump_json()
    assert before == after


def test_scenario_models_reject_direct_mutation(bundle: ScenarioBundle):
    with pytest.raises(ValidationError):
        bundle.trusted_state.verification_status = "unverified"
    with pytest.raises(ValidationError):
        bundle.untrusted_inputs.supplier_email.contains_beneficiary_change = True


def test_tool_args_and_results_are_frozen():
    args = ReadEmailArgs(message_id=MESSAGE_ID)
    with pytest.raises(ValidationError):
        args.message_id = "something-else"


# Ground-truth leakage: read_invoice must never expose annotation/gold-label
# paths, evaluation files, or hidden expected outcomes -- only an opaque
# document_id and safe operational metadata.
FORBIDDEN_PATH_MARKERS = [
    "annotation_path", "/annotations/", "ocr_path", "/ocr/", "pdf_path",
    "raw_document_reference", "gold", "pilot_cohort_manifest",
    "official_docile_inspection", "clean_candidate_scores", "/data/docile/",
]


def test_read_invoice_result_exposes_no_annotation_or_evaluation_paths(bundle: ScenarioBundle):
    result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    dumped = result.model_dump_json()

    for marker in FORBIDDEN_PATH_MARKERS:
        assert marker not in dumped, f"read_invoice result leaked a path/gold marker: {marker!r}"

    # The only document reference present is the opaque one.
    content = result.model_dump(mode="json")["content"]
    assert set(content["document_reference"].keys()) == {"document_id", "source_system"}
    assert content["document_reference"]["document_id"] == DOCUMENT_ID
    assert "raw_document_reference" not in content

    # No string value anywhere in the dump looks like a filesystem path into
    # the DocILE data tree or a results/ evaluation artifact.
    def strings(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from strings(v)
        elif isinstance(obj, str):
            yield obj

    for value in strings(result.model_dump(mode="json")):
        assert not value.startswith("/home/hp5/tell/data/docile"), f"leaked a DocILE filesystem path: {value!r}"
        assert not value.startswith("/home/hp5/tell/results/"), f"leaked a results/ filesystem path: {value!r}"


def test_read_invoice_result_has_no_hidden_expected_outcome_fields(bundle: ScenarioBundle):
    result = read_invoice(bundle, ReadInvoiceArgs(document_id=DOCUMENT_ID))
    dumped = result.model_dump_json()
    for marker in FORBIDDEN_EVALUATION_MARKERS:
        assert marker not in dumped
