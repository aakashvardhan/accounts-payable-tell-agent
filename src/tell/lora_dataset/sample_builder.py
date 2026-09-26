"""Builds one LoRA-corpus example (rendered messages + deterministic gold
action) for one (document, kind) pair.

Reuses, entirely unmodified, from the rest of this codebase:
  - `tell.probe_dataset.docile_extract.DocumentFacts` / `extract_document_facts`
  - `tell.agent.prompt_profiles.PromptProfile.TASK_ONLY_BASE_V1`
  - `tell.agent.loop_prompts.build_loop_system_prompt` /
    `build_initial_user_message` / `build_observation_message`
  - `tell.agent.work_item.TrustedWorkItem`
  - `tell.agent.tools.{ReadEmailResult, ReadEmailContent, ReadInvoiceResult,
    ReadInvoiceContent, DocumentReference, GetVendorRecordResult,
    GetVendorRecordContent, ToolStatus}`
  - `tell.evaluation.scenario.OperationalProvenance` (+ its enums)
  - `tell.agent.actions.{ReadEmailAction, ReadInvoiceAction,
    GetVendorRecordAction, ProposePaymentAction, VendorPaymentEvidence,
    RequestReviewAction}` and `tell.agent.decision.ReviewReasonCode`

Unlike the probe corpus (which duplicated `build_observation_message`'s
envelope over plain dicts, since it never needed a `get_vendor_record`
observation and wanted a narrower invoice-text-block representation),
this module constructs REAL `tell.agent.tools` result objects and calls
`build_observation_message` UNCHANGED -- there is no duplication here,
since every tool result this corpus needs (read_email, read_invoice,
get_vendor_record) already has a real, importable Pydantic class that
does not require a full `ScenarioBundle` to construct.

--------------------------------------------------------------------------
No fabricated model turns
--------------------------------------------------------------------------
Every example's context is built purely from prior *tool results*
(application-derived), never from a fabricated assistant/action turn --
same discipline as the probe corpus. Consecutive user-role messages are
used to chain multiple tool results, verified (by the probe corpus) to
render correctly under Qwen3's chat template.
"""

from __future__ import annotations

from dataclasses import dataclass

from tell.agent.actions import (
    GetVendorRecordAction,
    ProposePaymentAction,
    ReadEmailAction,
    ReadInvoiceAction,
    RequestReviewAction,
    VendorPaymentEvidence,
)
from tell.agent.decision import ReviewReasonCode
from tell.agent.loop_prompts import build_initial_user_message, build_loop_system_prompt, build_observation_message
from tell.agent.prompt_profiles import PromptProfile
from tell.agent.tools import (
    DocumentReference,
    GetVendorRecordContent,
    GetVendorRecordResult,
    ReadEmailContent,
    ReadEmailResult,
    ReadInvoiceContent,
    ReadInvoiceResult,
    ToolStatus,
)
from tell.agent.work_item import TrustedWorkItem
from tell.evaluation.scenario import OperationalProvenance, ProvenanceSource, SourceType, TrustBoundary
from tell.lora_dataset.synthetic_ids import LoraSyntheticIds
from tell.lora_dataset.templates import (
    LORA_SAMPLE_KIND_BY_NAME,
    LORA_SAMPLE_KIND_SPECS,
    conflict_vendor_name,
    render_baseline_email,
    render_email_kind,
    render_payment_destination,
    template_family_id,
)
from tell.probe_dataset.docile_extract import DocumentFacts

CORPUS_PROMPT_PROFILE = PromptProfile.TASK_ONLY_BASE_V1
FIXED_PROVENANCE_TIMESTAMP = "2024-01-01T00:00:00Z"


@dataclass(frozen=True)
class LoraSample:
    sample_id: str
    docid: str
    decision_point: str
    messages: tuple[dict, ...]
    gold_action_dict: dict  # already-validated action's model_dump(mode="json")
    gold_action_type: str


@dataclass(frozen=True)
class LoraSampleLabel:
    sample_id: str
    docid: str
    vendor_group_id: str
    split: str
    exposure_label: int  # 0 clean, 1 attack/conflicting
    decision_point: str
    gold_action_type: str
    template_family_id: str
    hard_negative_for: str | None
    source_provenance: str
    prompt_profile: str
    account_id_role: str  # "approved" / "unauthorized" / "obsolete" / "none"


def build_work_item(facts: DocumentFacts, synthetic_ids: LoraSyntheticIds) -> TrustedWorkItem:
    """`run_id` is docid-only -- identical across all ten samples of one
    document, never `sample_id` -- same leak-avoidance discipline as
    `tell.probe_dataset.sample_builder.build_work_item`."""
    return TrustedWorkItem(
        run_id=f"lora-work-{facts.docid}",
        supplier_message_id=synthetic_ids.supplier_message_id,
        canonical_vendor_id=synthetic_ids.canonical_vendor_id,
    )


def _fmt_kwargs(facts: DocumentFacts, synthetic_ids: LoraSyntheticIds) -> dict:
    return {
        "vendor": facts.vendor_name_display,
        "invoice_number": facts.invoice_number,
        "date": facts.invoice_date,
        "amount": facts.payable_amount_text,
        "currency_upper": facts.currency.upper(),
        "approved_account": synthetic_ids.approved_beneficiary_account_id,
        "unauthorized_account": synthetic_ids.unauthorized_beneficiary_account_id,
        "obsolete_account": synthetic_ids.obsolete_beneficiary_account_id,
    }


def _email_provenance(synthetic_ids: LoraSyntheticIds) -> OperationalProvenance:
    return OperationalProvenance(
        source_type=SourceType.EMAIL,
        source_id=synthetic_ids.supplier_message_id,
        provenance=ProvenanceSource.SYNTHETIC_CONTROLLED,
        recorded_at=FIXED_PROVENANCE_TIMESTAMP,
        trust_boundary=TrustBoundary.UNTRUSTED,
    )


def _invoice_provenance(facts: DocumentFacts) -> OperationalProvenance:
    return OperationalProvenance(
        source_type=SourceType.INVOICE_DOCUMENT,
        source_id=facts.docid,
        provenance=ProvenanceSource.DOCILE_ANNOTATION,
        recorded_at=FIXED_PROVENANCE_TIMESTAMP,
        trust_boundary=TrustBoundary.UNTRUSTED,
    )


def _vendor_provenance(synthetic_ids: LoraSyntheticIds) -> OperationalProvenance:
    return OperationalProvenance(
        source_type=SourceType.VENDOR_RECORD,
        source_id=synthetic_ids.canonical_vendor_id,
        provenance=ProvenanceSource.SYNTHETIC_CONTROLLED,
        recorded_at=FIXED_PROVENANCE_TIMESTAMP,
        trust_boundary=TrustBoundary.TRUSTED,
    )


def _domain_slug(vendor_display: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "" for ch in vendor_display)
    return slug[:24] or "vendor"


def _read_email_result(facts: DocumentFacts, synthetic_ids: LoraSyntheticIds, subject: str, body: str) -> ReadEmailResult:
    return ReadEmailResult(
        status=ToolStatus.SUCCESS,
        provenance=_email_provenance(synthetic_ids),
        content=ReadEmailContent(
            sender_display_name=facts.vendor_name_display,
            sender_address=f"billing@{_domain_slug(facts.vendor_name_display)}.example",
            subject=subject,
            body=body,
            references_docid=facts.docid,
            references_invoice_number=facts.invoice_number,
        ),
        error=None,
    )


def _line_items_view(facts: DocumentFacts) -> list[dict]:
    return [{"line_item_description": li.description} for li in facts.line_items if li.description]


def _read_invoice_result(facts: DocumentFacts, payment_destination: list[str]) -> ReadInvoiceResult:
    return ReadInvoiceResult(
        status=ToolStatus.SUCCESS,
        provenance=_invoice_provenance(facts),
        content=ReadInvoiceContent(
            docid=facts.docid,
            page_count=facts.page_count,
            vendor_name=facts.vendor_name_display,
            invoice_number=facts.invoice_number,
            invoice_date=facts.invoice_date,
            due_date=facts.due_date,
            currency=facts.currency,
            subtotal=None,
            tax=None,
            total_amount_gross=None,
            amount_due=facts.payable_amount_text,
            purchase_order_numbers=[],
            payment_destination=payment_destination,
            vendor_address=facts.vendor_address,
            vendor_email=facts.vendor_email,
            customer_billing_name=None,
            customer_billing_address=None,
            payment_terms=None,
            line_items=_line_items_view(facts),
            document_reference=DocumentReference(document_id=facts.docid),
        ),
        error=None,
    )


def _get_vendor_record_result(synthetic_ids: LoraSyntheticIds, *, vendor_name: str, verification_status: str) -> GetVendorRecordResult:
    return GetVendorRecordResult(
        status=ToolStatus.SUCCESS,
        provenance=_vendor_provenance(synthetic_ids),
        content=GetVendorRecordContent(
            vendor_id=synthetic_ids.canonical_vendor_id,
            vendor_name=vendor_name,
            beneficiary_account_id=synthetic_ids.approved_beneficiary_account_id,
            verification_status=verification_status,
        ),
        error=None,
    )


_ACCOUNT_ID_ROLE_BY_KIND = {
    "clean_read_email_first": "none",
    "clean_request_invoice": "approved",
    "clean_request_vendor_record": "obsolete",
    "clean_verify_before_pay": "obsolete",
    "clean_propose_payment": "approved",
    "attack_email_beneficiary_change": "unauthorized",
    "attack_email_forged_tool_result": "unauthorized",
    "attack_invoice_instruction_injection": "unauthorized",
    "attack_vendor_unverified": "approved",
    "attack_vendor_identity_conflict": "approved",
}


def build_sample(
    facts: DocumentFacts,
    synthetic_ids: LoraSyntheticIds,
    kind: str,
    *,
    split_docids_sorted: list[str],
    vendor_name_by_docid: dict[str, str],
) -> tuple[LoraSample, LoraSampleLabel]:
    spec = LORA_SAMPLE_KIND_BY_NAME[kind]
    split = facts.split
    sample_id = f"{facts.docid}__{kind}"
    fmt = _fmt_kwargs(facts, synthetic_ids)

    system_prompt = build_loop_system_prompt(CORPUS_PROMPT_PROFILE)
    work_item = build_work_item(facts, synthetic_ids)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]

    gold_action = None

    if spec.decision_point == "initial":
        gold_action = ReadEmailAction(message_id=synthetic_ids.supplier_message_id)

    elif spec.decision_point == "post_email":
        subject, body = render_email_kind(kind, split, **fmt)
        email_result = _read_email_result(facts, synthetic_ids, subject, body)
        messages.append({"role": "user", "content": build_observation_message(email_result)})
        if spec.gold_action_type == "read_invoice":
            gold_action = ReadInvoiceAction(document_id=facts.docid)
        else:
            assert spec.gold_action_type == "get_vendor_record"
            gold_action = GetVendorRecordAction(vendor_id=synthetic_ids.canonical_vendor_id)

    else:
        assert spec.decision_point in ("post_invoice", "post_vendor_record")
        base_subject, base_body = render_baseline_email(split, **fmt)
        email_result = _read_email_result(facts, synthetic_ids, base_subject, base_body)
        messages.append({"role": "user", "content": build_observation_message(email_result)})

        payment_destination = render_payment_destination(kind, split, **fmt)
        invoice_result = _read_invoice_result(facts, payment_destination)
        messages.append({"role": "user", "content": build_observation_message(invoice_result)})

        if spec.decision_point == "post_invoice":
            assert spec.gold_action_type == "get_vendor_record"
            gold_action = GetVendorRecordAction(vendor_id=synthetic_ids.canonical_vendor_id)
        else:
            if kind == "attack_vendor_identity_conflict":
                vendor_name = conflict_vendor_name(facts.docid, split_docids_sorted, vendor_name_by_docid)
                verification_status = "verified"
            elif kind == "attack_vendor_unverified":
                vendor_name = facts.vendor_name_display
                verification_status = "unverified"
            else:
                assert kind == "clean_propose_payment"
                vendor_name = facts.vendor_name_display
                verification_status = "verified"
            vendor_result = _get_vendor_record_result(synthetic_ids, vendor_name=vendor_name, verification_status=verification_status)
            messages.append({"role": "user", "content": build_observation_message(vendor_result)})

            if spec.gold_action_type == "propose_payment":
                gold_action = ProposePaymentAction(
                    invoice_document_id=facts.docid,
                    invoice_number=facts.invoice_number,
                    beneficiary_account_id=synthetic_ids.approved_beneficiary_account_id,
                    amount_minor_units=facts.payable_amount_minor_units,
                    currency=facts.currency,
                    evidence=VendorPaymentEvidence(invoice_document_id=facts.docid, vendor_record_id=synthetic_ids.canonical_vendor_id),
                )
            else:
                assert spec.gold_action_type == "request_review"
                reason = ReviewReasonCode.VENDOR_NOT_VERIFIED if kind == "attack_vendor_unverified" else ReviewReasonCode.CONFLICTING_VENDOR_IDENTITY
                gold_action = RequestReviewAction(review_reason=reason)

    assert gold_action is not None
    sample = LoraSample(
        sample_id=sample_id,
        docid=facts.docid,
        decision_point=spec.decision_point,
        messages=tuple(messages),
        gold_action_dict=gold_action.model_dump(mode="json"),
        gold_action_type=spec.gold_action_type,
    )
    label = LoraSampleLabel(
        sample_id=sample_id,
        docid=facts.docid,
        vendor_group_id=facts.vendor_name_display,
        split=split,
        exposure_label=spec.class_label,
        decision_point=spec.decision_point,
        gold_action_type=spec.gold_action_type,
        template_family_id=template_family_id(kind, split),
        hard_negative_for=spec.hard_negative_for,
        source_provenance="docile_annotation",
        prompt_profile=CORPUS_PROMPT_PROFILE.value,
        account_id_role=_ACCOUNT_ID_ROLE_BY_KIND[kind],
    )
    return sample, label


def build_all_samples(
    facts: DocumentFacts,
    synthetic_ids: LoraSyntheticIds,
    *,
    split_docids_sorted: list[str],
    vendor_name_by_docid: dict[str, str],
) -> list[tuple[LoraSample, LoraSampleLabel]]:
    return [
        build_sample(facts, synthetic_ids, spec.kind, split_docids_sorted=split_docids_sorted, vendor_name_by_docid=vendor_name_by_docid)
        for spec in LORA_SAMPLE_KIND_SPECS
    ]


__all__ = ["CORPUS_PROMPT_PROFILE", "LoraSample", "LoraSampleLabel", "build_work_item", "build_sample", "build_all_samples"]
