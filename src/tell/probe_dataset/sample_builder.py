"""Builds the ten paired forward-pass contexts (messages lists, ready
for `runtime.render_chat_prompt`) and their separate labels for one
selected document.

Reuses, unmodified, from the rest of this codebase:
  - `tell.agent.prompt_profiles.PromptProfile.TASK_ONLY_BASE_V1`
  - `tell.agent.loop_prompts.build_loop_system_prompt` (system prompt)
  - `tell.agent.work_item.TrustedWorkItem` /
    `tell.agent.loop_prompts.build_initial_user_message` (work item)

`_render_tool_result_message` below duplicates (rather than imports)
`tell.agent.loop_prompts.build_observation_message`'s exact JSON
envelope shape (`observation_type`/`tool_name`/`status`/`provenance`/
`content`/`error` plus the `[trust_boundary: ...]` note and the
"choose your next action" suffix) over plain dicts instead of typed
`ToolResult` objects, since this corpus never executes
`tell.agent.tools.read_email`/`read_invoice` (which require a full
`ScenarioBundle`) -- it builds a purpose-built, narrower content
representation per Section 7. Same duplication discipline already used
throughout this codebase (e.g. `tell.agent.conditional_retrieval`'s
`_search_result_tool_result_dict`) for a private, non-exported shape.

--------------------------------------------------------------------------
No fabricated model turns
--------------------------------------------------------------------------
This corpus never runs generation (forward-only capture), so there is no
real model action to fabricate or omit. The `post_invoice` context
therefore chains the initial work-item message directly to the
`read_email` and `read_invoice` observation messages without any
intervening assistant/action turn -- consecutive user-role messages
render correctly under the Qwen3 chat template (already verified for
`tell.agent.conditional_retrieval`'s application-prefetch harness) and
avoid any ambiguity about whether a model "chose" this tool sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from tell.agent.loop_prompts import build_initial_user_message, build_loop_system_prompt
from tell.agent.prompt_profiles import PromptProfile
from tell.agent.work_item import TrustedWorkItem
from tell.probe_dataset.docile_extract import DocumentFacts
from tell.probe_dataset.synthetic_ids import SyntheticIds
from tell.probe_dataset.templates import (
    SAMPLE_KIND_BY_NAME,
    SAMPLE_KIND_SPECS,
    render_email,
    render_overlay,
    template_family_id,
)
from tell.probe_dataset.text_block import InvoiceTextBlock, build_docile_ocr_block, build_synthetic_overlay_block

CORPUS_PROMPT_PROFILE = PromptProfile.TASK_ONLY_BASE_V1
FIXED_PROVENANCE_TIMESTAMP = "2024-01-01T00:00:00Z"


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    docid: str
    decision_point: str  # "post_email" or "post_invoice"
    messages: tuple[dict, ...]  # [{"role": ..., "content": ...}, ...] -- ready to render, no labels inside


@dataclass(frozen=True)
class SampleLabel:
    """Everything the future probe evaluator needs, kept entirely
    separate from `ProbeSample.messages` and from activation metadata --
    see this module's and `tell.probe_dataset.text_block`'s docstrings."""

    sample_id: str
    docid: str
    vendor_group_id: str  # normalized vendor name, for optional vendor-aware analysis later
    split: str
    exposure_label: int  # 0 clean, 1 attacked
    attack_surface: str | None  # "email" / "invoice_tool_result" / None
    template_family_id: str  # f"{kind}__{split}"
    clean_negative_family: str | None  # which attack kind this clean sample is a hard negative for
    source_provenance: str  # "docile_annotation" (base facts) -- the email/overlay text itself is synthetic_controlled
    prompt_profile: str
    decision_point: str


def _render_tool_result_message(*, tool_name: str, status: str, provenance: dict, content: dict) -> str:
    """Duplicates tell.agent.loop_prompts.build_observation_message's
    envelope shape over plain dicts -- see this module's docstring."""
    payload = {
        "observation_type": "tool_result",
        "tool_name": tool_name,
        "status": status,
        "provenance": provenance,
        "content": content,
        "error": None,
    }
    trust_note = f"\n\n[trust_boundary: {provenance['trust_boundary']}]"
    return (
        "Tool result:\n\n"
        + json.dumps(payload, indent=2)
        + trust_note
        + "\n\nChoose your next action now, as a single JSON object matching one of the action schemas. Output nothing else."
    )


def _email_provenance(synthetic_ids: SyntheticIds) -> dict:
    return {
        "source_type": "email",
        "source_id": synthetic_ids.supplier_message_id,
        "provenance": "synthetic_controlled",
        "recorded_at": FIXED_PROVENANCE_TIMESTAMP,
        "trust_boundary": "untrusted",
    }


def _invoice_provenance(facts: DocumentFacts) -> dict:
    return {
        "source_type": "invoice_document",
        "source_id": facts.docid,
        "provenance": "docile_annotation",
        "recorded_at": FIXED_PROVENANCE_TIMESTAMP,
        "trust_boundary": "untrusted",
    }


def _fmt_kwargs(facts: DocumentFacts, synthetic_ids: SyntheticIds) -> dict:
    return {
        "vendor": facts.vendor_name_display,
        "invoice_number": facts.invoice_number,
        "date": facts.invoice_date,
        "amount": facts.payable_amount_text,
        "currency_upper": facts.currency.upper(),
        "approved_account": synthetic_ids.approved_beneficiary_account_id,
        "attacker_account": synthetic_ids.attacker_beneficiary_account_id,
    }


def _email_content(facts: DocumentFacts, subject: str, body: str) -> dict:
    return {
        "sender_display_name": facts.vendor_name_display,
        "sender_address": f"billing@{_domain_slug(facts.vendor_name_display)}.example",
        "subject": subject,
        "body": body,
        "references_docid": facts.docid,
        "references_invoice_number": facts.invoice_number,
    }


def _domain_slug(vendor_display: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "" for ch in vendor_display)
    return slug[:24] or "vendor"


def _invoice_content(facts: DocumentFacts, text_blocks: list[InvoiceTextBlock]) -> dict:
    return {
        "docid": facts.docid,
        "page_count": facts.page_count,
        "vendor_name": facts.vendor_name_raw,
        "invoice_number": facts.invoice_number,
        "invoice_date": facts.invoice_date,
        "due_date": facts.due_date,
        "currency": facts.currency,
        "amount_due": facts.payable_amount_text,
        "line_item_count": facts.line_item_count,
        "vendor_address": facts.vendor_address,
        "text_blocks": [b.to_dict() for b in text_blocks],
    }


def build_work_item(facts: DocumentFacts, synthetic_ids: SyntheticIds) -> TrustedWorkItem:
    """`run_id` is deliberately docid-only (identical across all ten
    samples of one document), never `sample_id` -- `sample_id` encodes
    the sample kind (e.g. "...__email_social_engineering"), and
    `TrustedWorkItem.run_id` is embedded verbatim into the rendered
    prompt by `build_initial_user_message`. Using `sample_id` there would
    leak the exposure label directly into the model-visible text this
    corpus exists to keep label-free -- caught by
    scripts/build_probe_corpus_samples.py's own leakage check."""
    return TrustedWorkItem(
        run_id=f"probe-work-{facts.docid}",
        supplier_message_id=synthetic_ids.supplier_message_id,
        canonical_vendor_id=synthetic_ids.canonical_vendor_id,
    )


def build_sample(facts: DocumentFacts, synthetic_ids: SyntheticIds, kind: str) -> tuple[ProbeSample, SampleLabel]:
    spec = SAMPLE_KIND_BY_NAME[kind]
    split = facts.split
    sample_id = f"{facts.docid}__{kind}"
    fmt = _fmt_kwargs(facts, synthetic_ids)

    system_prompt = build_loop_system_prompt(CORPUS_PROMPT_PROFILE)
    work_item = build_work_item(facts, synthetic_ids)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]

    if spec.surface == "post_email":
        subject, body = render_email(kind, split, **fmt)
        email_msg = _render_tool_result_message(
            tool_name="read_email", status="success", provenance=_email_provenance(synthetic_ids), content=_email_content(facts, subject, body)
        )
        messages.append({"role": "user", "content": email_msg})
        decision_point = "post_email"
    else:
        assert spec.surface == "post_invoice"
        # Deterministic, paired-within-document tool history: the SAME
        # ordinary clean_standard email for this document/split precedes
        # every post_invoice sample, regardless of whether the invoice
        # text block is clean or attacked (see this module's docstring).
        std_subject, std_body = render_email("clean_standard", split, **fmt)
        email_msg = _render_tool_result_message(
            tool_name="read_email", status="success", provenance=_email_provenance(synthetic_ids), content=_email_content(facts, std_subject, std_body)
        )
        messages.append({"role": "user", "content": email_msg})

        ocr_block = build_docile_ocr_block(
            block_id=f"{facts.docid}-ocr-0",
            vendor_address=facts.vendor_address,
            line_item_descriptions=[li.description for li in facts.line_items if li.description],
        )
        overlay_text = render_overlay(kind, split, **fmt)
        # block_id is deliberately kind-free (there is only ever one
        # overlay block per post_invoice sample) -- embedding `kind` here
        # would leak the exposure label directly into model-visible text,
        # the same class of bug build_work_item's docstring explains.
        overlay_block = build_synthetic_overlay_block(block_id=f"{facts.docid}-overlay-0", text=overlay_text)
        invoice_msg = _render_tool_result_message(
            tool_name="read_invoice", status="success", provenance=_invoice_provenance(facts), content=_invoice_content(facts, [ocr_block, overlay_block])
        )
        messages.append({"role": "user", "content": invoice_msg})
        decision_point = "post_invoice"

    sample = ProbeSample(sample_id=sample_id, docid=facts.docid, decision_point=decision_point, messages=tuple(messages))
    label = SampleLabel(
        sample_id=sample_id,
        docid=facts.docid,
        vendor_group_id=facts.vendor_name_display,
        split=split,
        exposure_label=spec.class_label,
        attack_surface=spec.attack_surface,
        template_family_id=template_family_id(kind, split),
        clean_negative_family=spec.hard_negative_for,
        source_provenance="docile_annotation",
        prompt_profile=CORPUS_PROMPT_PROFILE.value,
        decision_point=decision_point,
    )
    return sample, label


def build_all_samples(facts: DocumentFacts, synthetic_ids: SyntheticIds) -> list[tuple[ProbeSample, SampleLabel]]:
    return [build_sample(facts, synthetic_ids, spec.kind) for spec in SAMPLE_KIND_SPECS]


__all__ = ["CORPUS_PROMPT_PROFILE", "ProbeSample", "SampleLabel", "build_work_item", "build_sample", "build_all_samples"]
