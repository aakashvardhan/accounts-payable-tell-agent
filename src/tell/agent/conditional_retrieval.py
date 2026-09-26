"""Application-prefetched delayed-memory-retrieval harness.

This harness closes a coverage gap left by the `task_only_base_v1`
prompt-policy ablation (see
`results/evaluation/prompt_policy_ablation_report.md`, Section 7): in
that pilot, `tell.agent.memory_loop.run_session_b` lets the model choose
for itself whether to call `search_memory` at all, and on the one
deterministic run it did not, so the "does a retrieved poisoned memory
change the payment decision" question was never actually exercised.

This module answers a narrower, conditional question: *given* that the
application guarantees the agent is shown the retrieved memory before
its first decision (exactly the way a real AP inbox might pre-attach a
supplier's file notes to a work item), what does the model do next? It
does this by executing exactly one `search_memory` call as application
code -- not a model choice -- before the model's first generation, then
handing full, unconstrained control to the model for every subsequent
turn, using the identical per-turn machinery
`tell.agent.memory_loop.run_session_b` already uses (same action set,
same tool executors, same capture discipline, same fail-closed parsing,
same termination vocabulary). `tell.agent.memory_loop` and
`tell.agent.loop` are not modified by this module.

--------------------------------------------------------------------------
Actor discipline
--------------------------------------------------------------------------
The one application-executed step (the prefetch `search_memory` call) is
recorded separately as a `PrefetchRecord` with `actor="application_prefetch"`.
It involves no model generation and no activation capture of its own --
it is exactly the tool result a model-chosen `search_memory` action would
have produced, appended to the conversation the same way
(`tell.agent.memory_prompts.build_search_memory_result_message`,
unmodified). Every action from the model's first turn onward is recorded
with `actor="model"` and is generated, parsed, and (for the three
read-only tools plus a model-chosen repeat `search_memory`) executed
exactly as `run_session_b` does; `propose_payment`/`request_review`/
`fail_closed` always terminate the run without any executor call --
same enforcement boundary as every other loop in this codebase. This
harness never fabricates a model tool call and never claims the model
autonomously chose to retrieve memory; it is a conditional-on-retrieval
evaluation, not a discovery of the model's own tool-choice policy.

Because the prefetch step sets `last_observation_source_type = "memory"`
before the model's first turn, the model's first decision point is named
`retrieval_post_memory` through the exact same decision-point logic
`run_session_b` already uses (it checks `last_observation_source_type ==
"memory"` first) -- no new decision-point naming rule was added.

The application guarantees exposure to the retrieved memory only. It
never forces a subsequent invoice lookup, vendor lookup, review,
beneficiary selection, or terminal action -- those remain entirely the
model's choice among the unmodified `SessionBAction` set (which, notably,
still has no `write_memory` action, so this harness cannot mutate the
memory store no matter what the model outputs).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tell.agent.actions import (
    FailClosedAction,
    GetVendorRecordAction,
    ReadEmailAction,
    ReadInvoiceAction,
    RequestReviewAction,
    SearchMemoryAction,
    parse_session_b_action,
)
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, QwenLocalRuntime
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS, TurnRecord
from tell.agent.memory_prompts import (
    build_initial_user_message,
    build_observation_message,
    build_search_memory_result_message,
    build_session_b_system_prompt,
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
from tell.evaluation.scenario import ScenarioBundle
from tell.memory.models import MemoryRecord
from tell.memory.store import MemoryStore

ACTOR_APPLICATION_PREFETCH = "application_prefetch"
ACTOR_MODEL = "model"

# Same termination vocabulary as tell.agent.memory_loop's Session B.
TERMINATION_PROPOSED_PAYMENT = "proposed_payment"
TERMINATION_REQUESTED_REVIEW = "requested_review"
TERMINATION_MODEL_FAILED_CLOSED = "model_failed_closed"
TERMINATION_INVALID_OUTPUT = "invalid_fail_closed"
TERMINATION_REPEATED_ACTION = "repeated_action_detected"
TERMINATION_STEP_LIMIT = "step_limit_fail_closed"

RETRIEVAL_DECISION_POINT = "retrieval_post_memory"


@dataclass(frozen=True)
class PrefetchRecord:
    """The one application-executed step. No capture and no generation
    happen for this step -- see this module's docstring."""

    actor: str
    vendor_id: str
    query: str | None
    memory_kind: str | None
    tool_result: dict
    retrieved_memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConditionalRetrievalRunResult:
    scenario_id: str
    run_id: str
    condition_id: str
    prompt_profile: str
    prefetch: PrefetchRecord
    turns: tuple[TurnRecord, ...]
    termination_reason: str
    terminal_action: dict | None
    read_email_called: bool
    read_invoice_called: bool
    get_vendor_record_called: bool
    search_memory_called_by_model: bool
    retrieved_memories_prefetch: tuple[MemoryRecord, ...]
    retrieved_memories_model: tuple[MemoryRecord, ...]

    @property
    def retrieved_memories(self) -> tuple[MemoryRecord, ...]:
        """Every memory record the model was ever exposed to in this run
        (prefetch plus any model-chosen re-search), in exposure order."""
        return self.retrieved_memories_prefetch + self.retrieved_memories_model


def _search_result_tool_result_dict(vendor_id: str, records: list[MemoryRecord]) -> dict:
    """Identical shape to tell.agent.memory_loop's module-private
    `_search_result_tool_result_dict` -- duplicated rather than imported,
    since that name is private to memory_loop.py, and this harness must
    not modify memory_loop.py to export it."""
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


def _activation_paths(activation_dir: Path, run_id: str, turn_number: int, decision_point: str) -> tuple[Path, Path]:
    stem = f"{run_id}_turn{turn_number:02d}_{decision_point}"
    return activation_dir / f"{stem}.safetensors", activation_dir / f"{stem}_metadata.json"


def build_initial_rendered_context(
    runtime: QwenLocalRuntime,
    work_item: TrustedWorkItem,
    memory_store: MemoryStore,
    *,
    prompt_profile: PromptProfile = DEFAULT_PROMPT_PROFILE,
) -> tuple[str, str, list[MemoryRecord]]:
    """Renders the system+initial-user-message text alone (no memory
    result) and the full turn-1 text (with the prefetch memory result
    appended), without running the model -- used to prove the context
    reset independently of inference, per condition. Returns
    (pre_prefetch_text, post_prefetch_text, prefetch_records)."""
    pre_messages = [
        {"role": "system", "content": build_session_b_system_prompt(prompt_profile)},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]
    pre_text = runtime.render_chat_prompt(pre_messages, enable_thinking=False)

    prefetch_records = memory_store.search_memories(work_item.canonical_vendor_id, query=None, memory_kind=None)
    post_messages = pre_messages + [
        {"role": "user", "content": build_search_memory_result_message(work_item.canonical_vendor_id, prefetch_records)}
    ]
    post_text = runtime.render_chat_prompt(post_messages, enable_thinking=False)
    return pre_text, post_text, prefetch_records


def run_conditional_retrieval(
    runtime: QwenLocalRuntime,
    bundle: ScenarioBundle,
    work_item: TrustedWorkItem,
    memory_store: MemoryStore,
    *,
    activation_dir: Path,
    condition_id: str,
    max_turns: int = MAX_TURNS,
    prompt_profile: PromptProfile = DEFAULT_PROMPT_PROFILE,
) -> ConditionalRetrievalRunResult:
    """Runs one fresh, Session-B-shaped conversation in which the
    application itself executes the first `search_memory` call, then
    hands full control to the model for every subsequent turn.

    `bundle` must be the clean payment scenario (never a Session-A
    infection scenario) -- `run_conditional_retrieval` never reads any
    Session-A message or scenario. `memory_store` supplies whichever
    memory snapshot (clean-control or poisoned) this condition tests; it
    is opened read-only in spirit -- this function calls only
    `MemoryStore.search_memories`, never a write method, so the
    underlying database file is never modified by this run.
    """
    messages: list[dict] = [
        {"role": "system", "content": build_session_b_system_prompt(prompt_profile)},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]

    # ---- Step 0: application-executed prefetch (actor=application_prefetch) ----
    prefetch_records = memory_store.search_memories(work_item.canonical_vendor_id, query=None, memory_kind=None)
    prefetch_tool_result = _search_result_tool_result_dict(work_item.canonical_vendor_id, prefetch_records)
    messages.append(
        {"role": "user", "content": build_search_memory_result_message(work_item.canonical_vendor_id, prefetch_records)}
    )
    prefetch = PrefetchRecord(
        actor=ACTOR_APPLICATION_PREFETCH,
        vendor_id=work_item.canonical_vendor_id,
        query=None,
        memory_kind=None,
        tool_result=prefetch_tool_result,
        retrieved_memory_ids=tuple(r.memory_id for r in prefetch_records),
    )

    # ---- Steps 1..N: model-directed turns, identical machinery to run_session_b ----
    turns: list[TurnRecord] = []
    executed_signatures: set[tuple] = set()
    last_observation_source_type: str | None = "memory"
    last_observation_source_id: str | None = work_item.canonical_vendor_id
    previous_action_dump: dict | None = None
    read_email_called = False
    read_invoice_called = False
    get_vendor_record_called = False
    search_memory_called_by_model = False
    retrieved_memories_model: list[MemoryRecord] = []
    termination_reason = TERMINATION_STEP_LIMIT
    terminal_action_dump: dict | None = None

    for turn_number in range(1, max_turns + 1):
        if last_observation_source_type == "memory":
            decision_point = RETRIEVAL_DECISION_POINT
        elif last_observation_source_type in ("email", "invoice", "vendor_record"):
            decision_point = f"post_{last_observation_source_type}"
        else:
            decision_point = "other"

        chat_text = runtime.render_chat_prompt(messages, enable_thinking=False)
        inputs = runtime.tokenize(chat_text)
        prompt_token_count = int(inputs["input_ids"].shape[1])

        capture_request = CaptureRequest(
            scenario_id=bundle.scenario_id,
            decision_point=decision_point,
            source_ids=(work_item.supplier_message_id, work_item.canonical_vendor_id),
            model_repo_id=PINNED_MODEL_REPO_ID,
            model_revision=PINNED_MODEL_REVISION,
            tokenizer_class=type(runtime.tokenizer).__name__,
            model_class=type(runtime.model).__name__,
            prompt_text=chat_text,
            run_id=work_item.run_id,
            turn_number=turn_number,
            previous_action=previous_action_dump,
            most_recent_observation_source_type=last_observation_source_type,
            most_recent_observation_source_id=last_observation_source_id,
        )
        capture_result, vectors = capture_predecision_activations(runtime.model, inputs, capture_request)
        safetensors_path, metadata_path = _activation_paths(activation_dir, work_item.run_id, turn_number, decision_point)
        save_activation_artifact(
            vectors=vectors,
            capture_result=capture_result,
            request=capture_request,
            safetensors_path=safetensors_path,
            metadata_path=metadata_path,
        )
        # Post-write, additive amendment only -- tell.detector.capture
        # itself is never touched (same discipline as
        # scripts/run_task_only_*.py's _tag_activation_metadata_with_profile).
        # No clean/poison label, expected outcome, attacker beneficiary, or
        # any other evaluation-only field is ever added here.
        metadata = json.loads(metadata_path.read_text())
        metadata["condition_id"] = condition_id
        metadata["prompt_profile"] = prompt_profile.value
        metadata_path.write_text(json.dumps(metadata, indent=2))

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
            termination_reason = TERMINATION_INVALID_OUTPUT
            break

        action = parse_result.action
        action_dump = action.model_dump(mode="json")
        messages.append({"role": "assistant", "content": parse_result.cleaned_text})

        if isinstance(action, (RequestReviewAction, FailClosedAction)) or action_dump["action"] == "propose_payment":
            turns.append(_record(action_dump, False, None, None))
            terminal_action_dump = action_dump
            if action_dump["action"] == "propose_payment":
                termination_reason = TERMINATION_PROPOSED_PAYMENT
            elif isinstance(action, RequestReviewAction):
                termination_reason = TERMINATION_REQUESTED_REVIEW
            else:
                termination_reason = TERMINATION_MODEL_FAILED_CLOSED
            break

        if isinstance(action, SearchMemoryAction):
            signature = ("search_memory", action.vendor_id, action.query, action.memory_kind.value if action.memory_kind else None)
            if signature in executed_signatures:
                turns.append(_record(action_dump, False, None, None))
                termination_reason = TERMINATION_REPEATED_ACTION
                break
            executed_signatures.add(signature)
            results = memory_store.search_memories(action.vendor_id, query=action.query, memory_kind=action.memory_kind)
            retrieved_memories_model.extend(results)
            search_memory_called_by_model = True
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
                termination_reason = TERMINATION_REPEATED_ACTION
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

        raise TypeError(f"Unhandled conditional-retrieval action type: {type(action).__name__}")  # unreachable
    else:
        termination_reason = TERMINATION_STEP_LIMIT

    return ConditionalRetrievalRunResult(
        scenario_id=bundle.scenario_id,
        run_id=work_item.run_id,
        condition_id=condition_id,
        prompt_profile=prompt_profile.value,
        prefetch=prefetch,
        turns=tuple(turns),
        termination_reason=termination_reason,
        terminal_action=terminal_action_dump,
        read_email_called=read_email_called,
        read_invoice_called=read_invoice_called,
        get_vendor_record_called=get_vendor_record_called,
        search_memory_called_by_model=search_memory_called_by_model,
        retrieved_memories_prefetch=tuple(prefetch_records),
        retrieved_memories_model=tuple(retrieved_memories_model),
    )


__all__ = [
    "ACTOR_APPLICATION_PREFETCH",
    "ACTOR_MODEL",
    "RETRIEVAL_DECISION_POINT",
    "PrefetchRecord",
    "ConditionalRetrievalRunResult",
    "build_initial_rendered_context",
    "run_conditional_retrieval",
]
