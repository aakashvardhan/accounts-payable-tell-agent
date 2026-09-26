"""Bounded, model-directed agent loops for Tell's two-session
delayed-memory-poisoning pilot.

Two functions, mirroring `tell.agent.loop.run_agent_loop`'s structure
and discipline exactly (one capture + one generation per turn, no
retries, strict parsing, faithful preservation of whatever the model
attempts) but each over its own restricted action set:

- `run_session_a`: read_email, write_memory, request_review,
  finish_review, fail_closed. No payment action exists in this session
  at all.
- `run_session_b`: read_email, read_invoice, search_memory,
  get_vendor_record, propose_payment, request_review, fail_closed.
  `propose_payment` is recorded like any other terminal action but never
  executed -- there is no payment executor call anywhere in this module.

Neither function loads a model at import time or mutates
`tell.agent.loop` in any way; `tell.agent.loop.TurnRecord` is reused
unchanged (its shape already fits both sessions) via import.

--------------------------------------------------------------------------
Where the untrusted-to-trusted boundary is enforced
--------------------------------------------------------------------------
`run_session_a`'s `write_memory` dispatch is the one place in this
module that creates persistent state. It calls
`tell.memory.store.MemoryStore.append_memory` with
`origin_trust_boundary=TrustBoundary.UNTRUSTED` unconditionally, because
in this pilot every write_memory call follows reading an email --
`WriteMemoryAction` itself carries no provenance field the model could
set instead (see tell.agent.actions's "Memory-pilot actions" section).
`run_session_b`'s `search_memory` dispatch never modifies a retrieved
record's origin fields before showing them to the model -- see
`tell.agent.memory_prompts.build_search_memory_result_message`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tell.agent.actions import (
    FailClosedAction,
    FinishReviewAction,
    GetVendorRecordAction,
    ReadEmailAction,
    ReadInvoiceAction,
    RequestReviewAction,
    SearchMemoryAction,
    WriteMemoryAction,
    parse_session_a_action,
    parse_session_b_action,
)
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, QwenLocalRuntime
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS, TurnRecord
from tell.agent.memory_prompts import (
    build_initial_user_message,
    build_observation_message,
    build_search_memory_result_message,
    build_session_a_system_prompt,
    build_session_b_system_prompt,
    build_write_memory_result_message,
)
from tell.agent.prompt_profiles import DEFAULT_PROMPT_PROFILE, PromptProfile
from tell.agent.tools import (
    GetVendorRecordArgs,
    ReadEmailArgs,
    ReadInvoiceArgs,
    get_vendor_record,
    read_email,
    read_invoice,
)
from tell.agent.work_item import TrustedWorkItem
from tell.detector.capture import CaptureRequest, capture_predecision_activations, save_activation_artifact
from tell.evaluation.scenario import ProvenanceSource, ScenarioBundle, SourceType, TrustBoundary
from tell.memory.models import MemoryRecord
from tell.memory.store import MemoryStore

# Session-A termination reasons.
SESSION_A_TERMINATION_MEMORY_WRITTEN_THEN_FINISHED = "finished_review"
SESSION_A_TERMINATION_REQUESTED_REVIEW = "requested_review"
SESSION_A_TERMINATION_MODEL_FAILED_CLOSED = "model_failed_closed"
SESSION_A_TERMINATION_INVALID_OUTPUT = "invalid_fail_closed"
SESSION_A_TERMINATION_REPEATED_ACTION = "repeated_action_detected"
SESSION_A_TERMINATION_STEP_LIMIT = "step_limit_fail_closed"

# Session-B termination reasons (same vocabulary as tell.agent.loop).
SESSION_B_TERMINATION_PROPOSED_PAYMENT = "proposed_payment"
SESSION_B_TERMINATION_REQUESTED_REVIEW = "requested_review"
SESSION_B_TERMINATION_MODEL_FAILED_CLOSED = "model_failed_closed"
SESSION_B_TERMINATION_INVALID_OUTPUT = "invalid_fail_closed"
SESSION_B_TERMINATION_REPEATED_ACTION = "repeated_action_detected"
SESSION_B_TERMINATION_STEP_LIMIT = "step_limit_fail_closed"


@dataclass(frozen=True)
class SessionARunResult:
    scenario_id: str
    run_id: str
    turns: tuple[TurnRecord, ...]
    termination_reason: str
    terminal_action: dict | None
    read_email_called: bool
    memory_written: MemoryRecord | None


@dataclass(frozen=True)
class SessionBRunResult:
    scenario_id: str
    run_id: str
    turns: tuple[TurnRecord, ...]
    termination_reason: str
    terminal_action: dict | None
    read_email_called: bool
    read_invoice_called: bool
    get_vendor_record_called: bool
    search_memory_called: bool
    retrieved_memories: tuple[MemoryRecord, ...]


def _activation_paths(activation_dir: Path, run_id: str, turn_number: int, decision_point: str) -> tuple[Path, Path]:
    stem = f"{run_id}_turn{turn_number:02d}_{decision_point}"
    return activation_dir / f"{stem}.safetensors", activation_dir / f"{stem}_metadata.json"


def _mem_record_tool_result_dict(record: MemoryRecord) -> dict:
    return {
        "tool_name": "write_memory",
        "status": "success",
        "content": {
            "memory_id": record.memory_id,
            "vendor_id": record.vendor_id,
            "memory_kind": record.memory_kind.value,
            "content": record.content,
            "status": record.status.value,
            "created_at": record.created_at,
            "origin_source_type": record.origin_source_type.value,
            "origin_source_id": record.origin_source_id,
            "origin_provenance": record.origin_provenance.value,
            "origin_trust_boundary": record.origin_trust_boundary.value,
        },
    }


def _search_result_tool_result_dict(vendor_id: str, records: list[MemoryRecord]) -> dict:
    return {
        "tool_name": "search_memory",
        "status": "success",
        "content": {
            "vendor_id": vendor_id,
            "results": [
                {
                    "memory_id": r.memory_id,
                    "vendor_id": r.vendor_id,
                    "memory_kind": r.memory_kind.value,
                    "content": r.content,
                    "status": r.status.value,
                    "created_at": r.created_at,
                    "origin_source_type": r.origin_source_type.value,
                    "origin_source_id": r.origin_source_id,
                    "origin_provenance": r.origin_provenance.value,
                    "origin_trust_boundary": r.origin_trust_boundary.value,
                }
                for r in records
            ],
        },
    }


def _capture_and_generate(runtime: QwenLocalRuntime, messages: list[dict], *, scenario_id: str, decision_point: str, run_id: str, turn_number: int, source_ids: tuple[str, ...], previous_action_dump: dict | None, last_observation_source_type: str | None, last_observation_source_id: str | None):
    """Shared capture+generate step used by both sessions. Returns
    (capture_result, prompt_token_count, gen_elapsed, raw_output,
    safetensors_path, metadata_path)."""
    chat_text = runtime.render_chat_prompt(messages, enable_thinking=False)
    inputs = runtime.tokenize(chat_text)
    prompt_token_count = int(inputs["input_ids"].shape[1])

    capture_request = CaptureRequest(
        scenario_id=scenario_id,
        decision_point=decision_point,
        source_ids=source_ids,
        model_repo_id=PINNED_MODEL_REPO_ID,
        model_revision=PINNED_MODEL_REVISION,
        tokenizer_class=type(runtime.tokenizer).__name__,
        model_class=type(runtime.model).__name__,
        prompt_text=chat_text,
        run_id=run_id,
        turn_number=turn_number,
        previous_action=previous_action_dump,
        most_recent_observation_source_type=last_observation_source_type,
        most_recent_observation_source_id=last_observation_source_id,
    )
    capture_result, vectors = capture_predecision_activations(runtime.model, inputs, capture_request)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gen_start = time.perf_counter()
    with torch.inference_mode():
        generated = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
            use_cache=True,
        )
    gen_elapsed = time.perf_counter() - gen_start
    new_tokens = generated[0][prompt_token_count:]
    raw_output = runtime.tokenizer.decode(new_tokens, skip_special_tokens=True)
    del generated, new_tokens, inputs

    return capture_request, capture_result, vectors, prompt_token_count, gen_elapsed, raw_output


def run_session_a(
    runtime: QwenLocalRuntime,
    bundle: ScenarioBundle,
    work_item: TrustedWorkItem,
    memory_store: MemoryStore,
    *,
    activation_dir: Path,
    max_turns: int = MAX_TURNS,
    prompt_profile: PromptProfile = DEFAULT_PROMPT_PROFILE,
) -> SessionARunResult:
    """Infection session. `bundle` supplies the (clean or poisoned)
    supplier email; `memory_store` must already have its schema
    initialized and should start empty for this run. `prompt_profile`
    selects which system-prompt template is rendered (see
    tell.agent.memory_prompts and tell.agent.prompt_profiles); it
    defaults to the hardened profile, so every existing call site's
    behavior is unchanged."""
    messages: list[dict] = [
        {"role": "system", "content": build_session_a_system_prompt(prompt_profile)},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]

    turns: list[TurnRecord] = []
    executed_signatures: set[tuple] = set()
    last_observation_source_type: str | None = None
    last_observation_source_id: str | None = None
    previous_action_dump: dict | None = None
    read_email_called = False
    memory_written: MemoryRecord | None = None
    termination_reason = SESSION_A_TERMINATION_STEP_LIMIT
    terminal_action_dump: dict | None = None

    for turn_number in range(1, max_turns + 1):
        if turn_number == 1:
            decision_point = "initial"
        elif last_observation_source_type == "email":
            decision_point = "infection_post_email"
        elif last_observation_source_type == "write_memory":
            decision_point = "post_write_memory"
        else:
            decision_point = "other"

        capture_request, capture_result, vectors, prompt_token_count, gen_elapsed, raw_output = _capture_and_generate(
            runtime,
            messages,
            scenario_id=bundle.scenario_id,
            decision_point=decision_point,
            run_id=work_item.run_id,
            turn_number=turn_number,
            source_ids=(work_item.supplier_message_id,),
            previous_action_dump=previous_action_dump,
            last_observation_source_type=last_observation_source_type,
            last_observation_source_id=last_observation_source_id,
        )
        safetensors_path, metadata_path = _activation_paths(activation_dir, work_item.run_id, turn_number, decision_point)
        save_activation_artifact(vectors=vectors, capture_result=capture_result, request=capture_request, safetensors_path=safetensors_path, metadata_path=metadata_path)

        parse_result = parse_session_a_action(raw_output)

        def _record(action_dump, executed, tool_status, tool_result) -> TurnRecord:
            return TurnRecord(
                turn_number=turn_number,
                decision_point=decision_point,
                prompt_sha256=capture_result.prompt_sha256,
                input_ids_sha256=capture_result.input_ids_sha256,
                rendered_token_count=prompt_token_count,
                selected_token_index=capture_result.selected_token_index,
                selected_token_id=capture_result.selected_token_id,
                capture_elapsed_seconds=capture_result.elapsed_seconds,
                generation_elapsed_seconds=gen_elapsed,
                raw_output=raw_output,
                parse_outcome=parse_result.outcome.value,
                parse_error_message=parse_result.error_message,
                action=action_dump,
                executed=executed,
                tool_status=tool_status,
                tool_result=tool_result,
                activation_safetensors_path=str(safetensors_path),
                activation_metadata_path=str(metadata_path),
            )

        if not parse_result.is_valid:
            turns.append(_record(None, False, None, None))
            termination_reason = SESSION_A_TERMINATION_INVALID_OUTPUT
            break

        action = parse_result.action
        action_dump = action.model_dump(mode="json")
        messages.append({"role": "assistant", "content": parse_result.cleaned_text})

        if isinstance(action, (RequestReviewAction, FinishReviewAction, FailClosedAction)):
            turns.append(_record(action_dump, False, None, None))
            terminal_action_dump = action_dump
            if isinstance(action, RequestReviewAction):
                termination_reason = SESSION_A_TERMINATION_REQUESTED_REVIEW
            elif isinstance(action, FinishReviewAction):
                termination_reason = SESSION_A_TERMINATION_MEMORY_WRITTEN_THEN_FINISHED
            else:
                termination_reason = SESSION_A_TERMINATION_MODEL_FAILED_CLOSED
            break

        if isinstance(action, ReadEmailAction):
            signature = ("read_email", action.message_id)
            if signature in executed_signatures:
                turns.append(_record(action_dump, False, None, None))
                termination_reason = SESSION_A_TERMINATION_REPEATED_ACTION
                break
            executed_signatures.add(signature)
            tool_result = read_email(bundle, ReadEmailArgs(message_id=action.message_id))
            read_email_called = True
            turns.append(_record(action_dump, True, tool_result.status.value, tool_result.model_dump(mode="json")))
            messages.append({"role": "user", "content": build_observation_message(tool_result)})
            previous_action_dump = action_dump
            last_observation_source_type = "email"
            last_observation_source_id = action.message_id
            continue

        if isinstance(action, WriteMemoryAction):
            email = bundle.untrusted_inputs.supplier_email
            source_message_id = last_observation_source_id if last_observation_source_type == "email" else work_item.supplier_message_id
            record = memory_store.append_memory(
                vendor_id=action.vendor_id,
                memory_kind=action.memory_kind,
                content=action.content,
                origin_source_type=email.operational_provenance.source_type if email.operational_provenance else SourceType.EMAIL,
                origin_source_id=source_message_id,
                origin_provenance=email.operational_provenance.provenance if email.operational_provenance else ProvenanceSource.SYNTHETIC_CONTROLLED,
                # Always UNTRUSTED, unconditionally -- see this module's
                # docstring. WriteMemoryAction has no field the model
                # could use to claim otherwise even if it wanted to.
                origin_trust_boundary=TrustBoundary.UNTRUSTED,
            )
            memory_written = record
            turns.append(_record(action_dump, True, "success", _mem_record_tool_result_dict(record)))
            messages.append({"role": "user", "content": build_write_memory_result_message(record)})
            previous_action_dump = action_dump
            last_observation_source_type = "write_memory"
            last_observation_source_id = record.memory_id
            continue

        raise TypeError(f"Unhandled session-A action type: {type(action).__name__}")  # unreachable
    else:
        termination_reason = SESSION_A_TERMINATION_STEP_LIMIT

    return SessionARunResult(
        scenario_id=bundle.scenario_id,
        run_id=work_item.run_id,
        turns=tuple(turns),
        termination_reason=termination_reason,
        terminal_action=terminal_action_dump,
        read_email_called=read_email_called,
        memory_written=memory_written,
    )


def run_session_b(
    runtime: QwenLocalRuntime,
    bundle: ScenarioBundle,
    work_item: TrustedWorkItem,
    memory_store: MemoryStore,
    *,
    activation_dir: Path,
    max_turns: int = MAX_TURNS,
    prompt_profile: PromptProfile = DEFAULT_PROMPT_PROFILE,
) -> SessionBRunResult:
    """Delayed-retrieval session. `bundle` is the clean MDS Pharma
    scenario (never a Session-A infection scenario); `messages` starts
    completely fresh -- no Session-A content is ever added to it. Poison
    can only reach this session through `memory_store.search_memories`.
    `prompt_profile` selects which system-prompt template is rendered
    (see tell.agent.memory_prompts and tell.agent.prompt_profiles); it
    defaults to the hardened profile, so every existing call site's
    behavior is unchanged."""
    messages: list[dict] = [
        {"role": "system", "content": build_session_b_system_prompt(prompt_profile)},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]

    turns: list[TurnRecord] = []
    executed_signatures: set[tuple] = set()
    last_observation_source_type: str | None = None
    last_observation_source_id: str | None = None
    previous_action_dump: dict | None = None
    read_email_called = False
    read_invoice_called = False
    get_vendor_record_called = False
    search_memory_called = False
    retrieved_memories: list[MemoryRecord] = []
    termination_reason = SESSION_B_TERMINATION_STEP_LIMIT
    terminal_action_dump: dict | None = None

    for turn_number in range(1, max_turns + 1):
        if turn_number == 1:
            decision_point = "initial"
        elif last_observation_source_type == "memory":
            decision_point = "retrieval_post_memory"
        elif last_observation_source_type in ("email", "invoice", "vendor_record"):
            decision_point = f"post_{last_observation_source_type}"
        else:
            decision_point = "other"

        capture_request, capture_result, vectors, prompt_token_count, gen_elapsed, raw_output = _capture_and_generate(
            runtime,
            messages,
            scenario_id=bundle.scenario_id,
            decision_point=decision_point,
            run_id=work_item.run_id,
            turn_number=turn_number,
            source_ids=(work_item.supplier_message_id, work_item.canonical_vendor_id),
            previous_action_dump=previous_action_dump,
            last_observation_source_type=last_observation_source_type,
            last_observation_source_id=last_observation_source_id,
        )
        safetensors_path, metadata_path = _activation_paths(activation_dir, work_item.run_id, turn_number, decision_point)
        save_activation_artifact(vectors=vectors, capture_result=capture_result, request=capture_request, safetensors_path=safetensors_path, metadata_path=metadata_path)

        parse_result = parse_session_b_action(raw_output)

        def _record(action_dump, executed, tool_status, tool_result) -> TurnRecord:
            return TurnRecord(
                turn_number=turn_number,
                decision_point=decision_point,
                prompt_sha256=capture_result.prompt_sha256,
                input_ids_sha256=capture_result.input_ids_sha256,
                rendered_token_count=prompt_token_count,
                selected_token_index=capture_result.selected_token_index,
                selected_token_id=capture_result.selected_token_id,
                capture_elapsed_seconds=capture_result.elapsed_seconds,
                generation_elapsed_seconds=gen_elapsed,
                raw_output=raw_output,
                parse_outcome=parse_result.outcome.value,
                parse_error_message=parse_result.error_message,
                action=action_dump,
                executed=executed,
                tool_status=tool_status,
                tool_result=tool_result,
                activation_safetensors_path=str(safetensors_path),
                activation_metadata_path=str(metadata_path),
            )

        if not parse_result.is_valid:
            turns.append(_record(None, False, None, None))
            termination_reason = SESSION_B_TERMINATION_INVALID_OUTPUT
            break

        action = parse_result.action
        action_dump = action.model_dump(mode="json")
        messages.append({"role": "assistant", "content": parse_result.cleaned_text})

        if isinstance(action, (RequestReviewAction, FailClosedAction)) or action_dump["action"] == "propose_payment":
            turns.append(_record(action_dump, False, None, None))
            terminal_action_dump = action_dump
            if action_dump["action"] == "propose_payment":
                termination_reason = SESSION_B_TERMINATION_PROPOSED_PAYMENT
            elif isinstance(action, RequestReviewAction):
                termination_reason = SESSION_B_TERMINATION_REQUESTED_REVIEW
            else:
                termination_reason = SESSION_B_TERMINATION_MODEL_FAILED_CLOSED
            break

        if isinstance(action, SearchMemoryAction):
            signature = ("search_memory", action.vendor_id, action.query, action.memory_kind.value if action.memory_kind else None)
            if signature in executed_signatures:
                turns.append(_record(action_dump, False, None, None))
                termination_reason = SESSION_B_TERMINATION_REPEATED_ACTION
                break
            executed_signatures.add(signature)
            results = memory_store.search_memories(action.vendor_id, query=action.query, memory_kind=action.memory_kind)
            retrieved_memories.extend(results)
            search_memory_called = True
            turns.append(_record(action_dump, True, "success", _search_result_tool_result_dict(action.vendor_id, results)))
            messages.append({"role": "user", "content": build_search_memory_result_message(action.vendor_id, results)})
            previous_action_dump = action_dump
            last_observation_source_type = "memory"
            last_observation_source_id = action.vendor_id
            continue

        if isinstance(action, (ReadEmailAction, ReadInvoiceAction, GetVendorRecordAction)):
            if isinstance(action, ReadEmailAction):
                signature = ("read_email", action.message_id)
            elif isinstance(action, ReadInvoiceAction):
                signature = ("read_invoice", action.document_id)
            else:
                signature = ("get_vendor_record", action.vendor_id)
            if signature in executed_signatures:
                turns.append(_record(action_dump, False, None, None))
                termination_reason = SESSION_B_TERMINATION_REPEATED_ACTION
                break
            executed_signatures.add(signature)

            if isinstance(action, ReadEmailAction):
                tool_result = read_email(bundle, ReadEmailArgs(message_id=action.message_id))
                source_type_label, source_id = "email", action.message_id
                read_email_called = True
            elif isinstance(action, ReadInvoiceAction):
                tool_result = read_invoice(bundle, ReadInvoiceArgs(document_id=action.document_id))
                source_type_label, source_id = "invoice", action.document_id
                read_invoice_called = True
            else:
                tool_result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=action.vendor_id))
                source_type_label, source_id = "vendor_record", action.vendor_id
                get_vendor_record_called = True

            turns.append(_record(action_dump, True, tool_result.status.value, tool_result.model_dump(mode="json")))
            messages.append({"role": "user", "content": build_observation_message(tool_result)})
            previous_action_dump = action_dump
            last_observation_source_type = source_type_label
            last_observation_source_id = source_id
            continue

        raise TypeError(f"Unhandled session-B action type: {type(action).__name__}")  # unreachable
    else:
        termination_reason = SESSION_B_TERMINATION_STEP_LIMIT

    return SessionBRunResult(
        scenario_id=bundle.scenario_id,
        run_id=work_item.run_id,
        turns=tuple(turns),
        termination_reason=termination_reason,
        terminal_action=terminal_action_dump,
        read_email_called=read_email_called,
        read_invoice_called=read_invoice_called,
        get_vendor_record_called=get_vendor_record_called,
        search_memory_called=search_memory_called,
        retrieved_memories=tuple(retrieved_memories),
    )
