"""Bounded, model-directed, read-only agentic loop for Tell's autonomous
AP experiment.

At each turn the loop:
  1. renders the current conversation (system policy + work item +
     every action/observation pair so far);
  2. tokenizes it once;
  3. captures the pre-action hidden state at indices 0/9/18/27/36 from
     that exact tokenized context;
  4. generates exactly one action from Qwen (no retry);
  5. validates it strictly (`tell.agent.actions.parse_agent_action`);
  6. if it is one of the three read-only tools, executes it and appends
     the typed result (with its own operational provenance, never
     reinterpreted) as the next observation;
  7. if it is a terminal action (propose_payment / request_review /
     fail_closed), stops;
  8. also stops on invalid output, a repeated read action, or after
     `MAX_TURNS` turns without a terminal action.

This module never loads a model at import time -- it accepts an
already-loaded `QwenLocalRuntime` (see tell.agent.local_model) so unit
tests can exercise everything except `run_agent_loop` itself without any
GPU dependency.

Enforcement boundary, deliberately narrow
--------------------------------------------------------------------------
This loop enforces exactly one thing deterministically at the READ-ACTION
level: which of the six schema-valid action types may actually execute
(only the three read-only tools). It does NOT force the model to call
get_vendor_record, and it does NOT check whether cited evidence is real
before RECORDING a terminal action as the model's choice -- that part of
the historical behavior (`tell.evaluation.agentic_outcomes` grading a run
after the fact) is unchanged.

What DID change in this milestone (real-loop payment integration)
--------------------------------------------------------------------------
Previously, a `propose_payment` terminal action just ended the run; no
validation, gate, or executor of any kind ran, and
`tell.evaluation.agentic_outcomes` graded the raw proposal separately,
after the fact. Now, whenever the model's terminal action is
`propose_payment`, OR whenever `tell_router` routes the workflow to Agent
S (regardless of what the model's own terminal action was), this module
calls `tell.agent.routing_orchestrator.route_and_validate_payment` --
the SAME tested orchestrator from the previous milestone, not a new
parallel implementation -- before `run_agent_loop` returns. The result is
recorded on `LoopRunResult.payment_routing_record`, and
`termination_reason` is updated to reflect the VALIDATED outcome (executed
/ escalated / hard-blocked / still-blocked), not merely "the model
proposed a payment." Every other termination path (request_review or
fail_closed chosen by the model while routed to Agent 1, invalid output,
a repeated read action, or the step limit) is completely unchanged.

Typed injection seams (no model / no GPU required to test this milestone)
--------------------------------------------------------------------------
- `TurnGenerator` / `ModelTurnGenerator`: the ONLY code that touches
  `runtime.model`/`runtime.tokenizer`. `ModelTurnGenerator` is byte-for-
  byte this module's previous inline tokenize/capture/generate/decode
  sequence -- using it (the default) reproduces the exact prior behavior.
  A test-only fake implementation returns a canned raw action string per
  turn with no model, no tokenizer, and no activation capture at all.
- `TellRouter` / `DefaultTellRouter`: no calibrated Tell threshold exists
  yet (TO MEASURE) -- the default always routes to Agent 1, which is
  exactly this loop's entire behavior before this integration existed.
  Tests inject a fake router to exercise the Agent-S path.
- `AgentSInvestigator` / `UnimplementedAgentSInvestigator`: Agent S's real
  trusted-evidence-gathering reasoning is not implemented in this
  milestone (it would require the safety LoRA and inference). The default
  raises `NotImplementedError` and is only ever reached if a `TellRouter`
  actually routes to Agent S -- `DefaultTellRouter` never does, so
  ordinary real-model use of this loop cannot reach it. Tests inject a
  fake investigator.
- `PaymentExecutor` (from `tell.agent.routing_orchestrator`) /
  `NullPaymentExecutor`: no real ledger/payment wiring exists yet
  (forbidden: no real database mutation in this milestone) -- the default
  executes nothing. Tests inject a spy.

None of these defaults change behavior for any existing caller
(`scripts/run_autonomous_read_loop.py` and friends): `ModelTurnGenerator`
reproduces the previous code exactly, and `DefaultTellRouter` always
choosing Agent 1 means the payment path behaves exactly as the previous
milestone's `route_and_validate_payment(routing_decision=AGENT_1, ...)`
call would for a below-threshold workflow -- there is no
default-disabled flag; this IS the loop's real payment path now.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from tell.agent.actions import (
    AgentAction,
    GetVendorRecordAction,
    ParsedActionResult,
    ReadEmailAction,
    ReadInvoiceAction,
    is_read_action,
    is_terminal_action,
    parse_agent_action,
)
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, QwenLocalRuntime
from tell.agent.loop_prompts import build_initial_user_message, build_loop_system_prompt, build_observation_message
from tell.agent.prompt_profiles import DEFAULT_PROMPT_PROFILE, PromptProfile
from tell.agent.routing_orchestrator import (
    PaymentExecutor,
    ResolutionInput,
    RoutingAuditRecord,
    RoutingDecision,
    route_and_validate_payment,
)
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
from tell.routing.records import FinalActionType, RoutedAgent
from tell.safety.payment_validation import (
    ObservedEvidence,
    ProposedPayment,
    TrustedInvoiceRecord,
    TrustedVendorRecord,
    VendorStatus,
    VendorVerificationStatus,
)

MAX_TURNS = 8
MAX_NEW_TOKENS = 256

# Turn-level, non-model-related termination reasons. "proposed_payment",
# "requested_review", and "model_failed_closed" reflect the model's own
# chosen terminal action (unchanged); the payment-path outcomes below
# reflect the VALIDATED result once routing/validation/gate have run.
TERMINATION_PROPOSED_PAYMENT = "proposed_payment"
TERMINATION_REQUESTED_REVIEW = "requested_review"
TERMINATION_MODEL_FAILED_CLOSED = "model_failed_closed"
TERMINATION_INVALID_OUTPUT = "invalid_fail_closed"
TERMINATION_REPEATED_ACTION = "repeated_action_detected"
TERMINATION_STEP_LIMIT = "step_limit_fail_closed"
# New in this milestone: what actually happened to a payment proposal
# after tell.agent.routing_orchestrator ran. TERMINATION_PROPOSED_PAYMENT
# above is still recorded on the turn itself (the model's own choice);
# these describe run_agent_loop's final, validated termination_reason.
TERMINATION_PAYMENT_EXECUTED = "payment_validated_and_executed"
TERMINATION_PAYMENT_ESCALATED_VENDOR_CLARIFICATION = "payment_escalated_vendor_clarification"
TERMINATION_PAYMENT_ESCALATED_EVIDENCE_REPORT = "payment_escalated_evidence_report"
TERMINATION_PAYMENT_HARD_BLOCKED = "payment_hard_blocked"
TERMINATION_PAYMENT_BLOCKED_UNRESOLVED = "payment_blocked_alarm_unresolved"


@dataclass(frozen=True)
class TurnRecord:
    turn_number: int
    decision_point: str
    prompt_sha256: str
    input_ids_sha256: str
    rendered_token_count: int
    selected_token_index: int
    selected_token_id: int
    capture_elapsed_seconds: float
    generation_elapsed_seconds: float
    raw_output: str
    parse_outcome: str
    parse_error_message: str | None
    action: dict | None
    executed: bool
    tool_status: str | None
    tool_result: dict | None
    activation_safetensors_path: str
    activation_metadata_path: str


@dataclass(frozen=True)
class LoopRunResult:
    scenario_id: str
    run_id: str
    turns: tuple[TurnRecord, ...]
    termination_reason: str
    terminal_action: dict | None
    read_email_called: bool
    read_invoice_called: bool
    get_vendor_record_called: bool
    routing_decision: RoutingDecision | None = None
    payment_routing_record: RoutingAuditRecord | None = None


def _activation_paths(activation_dir: Path, run_id: str, turn_number: int, decision_point: str) -> tuple[Path, Path]:
    stem = f"{run_id}_turn{turn_number:02d}_{decision_point}"
    return activation_dir / f"{stem}.safetensors", activation_dir / f"{stem}_metadata.json"


def _execute_read_action(bundle: ScenarioBundle, action: AgentAction):
    """Executes exactly one of the three read-only tools. Returns
    (tool_result, source_type_label, source_id) -- never reinterprets or
    upgrades a tool result's own operational provenance."""
    if isinstance(action, ReadEmailAction):
        result = read_email(bundle, ReadEmailArgs(message_id=action.message_id))
        return result, "email", action.message_id
    if isinstance(action, ReadInvoiceAction):
        result = read_invoice(bundle, ReadInvoiceArgs(document_id=action.document_id))
        return result, "invoice", action.document_id
    if isinstance(action, GetVendorRecordAction):
        result = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=action.vendor_id))
        return result, "vendor_record", action.vendor_id
    raise TypeError(f"Not a read action: {type(action).__name__}")  # unreachable given is_read_action guard


# ---------------------------------------------------------------------
# Typed injection seams -- see this module's docstring.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TurnGenerationResult:
    raw_output: str
    generation_elapsed_seconds: float
    prompt_sha256: str
    input_ids_sha256: str
    rendered_token_count: int
    selected_token_index: int
    selected_token_id: int
    capture_elapsed_seconds: float
    activation_safetensors_path: str
    activation_metadata_path: str


class TurnGenerator(Protocol):
    def generate_turn(
        self,
        *,
        runtime: QwenLocalRuntime,
        bundle: ScenarioBundle,
        work_item: TrustedWorkItem,
        messages: list[dict],
        turn_number: int,
        decision_point: str,
        previous_action_dump: dict | None,
        last_observation_source_type: str | None,
        last_observation_source_id: str | None,
        activation_dir: Path,
    ) -> TurnGenerationResult: ...


class ModelTurnGenerator:
    """The real thing: identical to this module's previous inline
    tokenize/capture/generate/decode sequence. Using this (the default)
    changes nothing about existing behavior."""

    def generate_turn(
        self,
        *,
        runtime: QwenLocalRuntime,
        bundle: ScenarioBundle,
        work_item: TrustedWorkItem,
        messages: list[dict],
        turn_number: int,
        decision_point: str,
        previous_action_dump: dict | None,
        last_observation_source_type: str | None,
        last_observation_source_id: str | None,
        activation_dir: Path,
    ) -> TurnGenerationResult:
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

        return TurnGenerationResult(
            raw_output=raw_output,
            generation_elapsed_seconds=gen_elapsed,
            prompt_sha256=capture_result.prompt_sha256,
            input_ids_sha256=capture_result.input_ids_sha256,
            rendered_token_count=prompt_token_count,
            selected_token_index=capture_result.selected_token_index,
            selected_token_id=capture_result.selected_token_id,
            capture_elapsed_seconds=capture_result.elapsed_seconds,
            activation_safetensors_path=str(safetensors_path),
            activation_metadata_path=str(metadata_path),
        )


class TellRouter(Protocol):
    def decide(self, bundle: ScenarioBundle, work_item: TrustedWorkItem) -> RoutingDecision: ...


class DefaultTellRouter:
    """No calibrated Tell threshold exists yet -- see
    results/routing_design/tell_threshold_selection_v1.md. Always routes
    to Agent 1: this reproduces the loop's entire pre-integration
    behavior exactly. Inject a fake `TellRouter` to test the Agent-S path
    without a real probe."""

    def decide(self, bundle: ScenarioBundle, work_item: TrustedWorkItem) -> RoutingDecision:
        return RoutingDecision(
            routed_agent=RoutedAgent.AGENT_1,
            reason="no calibrated Tell threshold exists yet (TO MEASURE); DefaultTellRouter always routes to Agent 1",
        )


@dataclass(frozen=True)
class AgentSFinding:
    resolution: ResolutionInput | None
    evidence_sources: tuple[str, ...] = ()


class AgentSInvestigator(Protocol):
    def investigate(self, bundle: ScenarioBundle, work_item: TrustedWorkItem, terminal_action: dict | None) -> AgentSFinding: ...


class UnimplementedAgentSInvestigator:
    """Agent S's real trusted-evidence-gathering reasoning (the safety
    LoRA plus tell.agent.trusted_lookups) is not implemented in this
    CPU-only milestone -- that requires inference. This default is only
    ever reached if a `TellRouter` actually routes to Agent S;
    `DefaultTellRouter` never does, so no ordinary real-model use of this
    loop can reach it. Inject a fake `AgentSInvestigator` for testing."""

    def investigate(self, bundle: ScenarioBundle, work_item: TrustedWorkItem, terminal_action: dict | None) -> AgentSFinding:
        raise NotImplementedError(
            "Agent S investigation is not implemented in this milestone (would require inference); inject a fake AgentSInvestigator for CPU testing."
        )


class NullPaymentExecutor:
    """No real ledger/payment wiring exists yet in this milestone
    (forbidden: no real database mutation). Executes nothing. Inject a
    spy `PaymentExecutor` for testing."""

    def execute(self, action) -> None:
        return None


class TrustedContextProvider(Protocol):
    """The fifth injection seam: how `run_agent_loop` derives
    `(TrustedVendorRecord, TrustedInvoiceRecord)` for a given `bundle`.
    The default, `_trusted_context_from_bundle`, is real, principled
    derivation from `ScenarioBundle.trusted_state` (see that function's
    docstring for its documented simplifications) -- using it changes
    nothing about existing behavior. This `ScenarioBundle` schema has no
    vendor-approved-contact concept at all, so the default can never
    produce a vendor record eligible for `request_vendor_clarification`
    (every gap escalates to an evidence report instead); tests inject a
    fake provider to exercise the vendor-clarification path without
    changing the frozen scenario-fixture schema."""

    def __call__(self, bundle: ScenarioBundle, observed_invoice_number: str | None) -> tuple[TrustedVendorRecord, TrustedInvoiceRecord]: ...


def _to_proposed_payment(action_dump: dict) -> ProposedPayment:
    return ProposedPayment(
        invoice_document_id=action_dump["invoice_document_id"],
        invoice_number=action_dump["invoice_number"],
        beneficiary_account_id=action_dump["beneficiary_account_id"],
        amount_minor_units=action_dump["amount_minor_units"],
        currency=action_dump["currency"],
        evidence_invoice_document_id=action_dump["evidence"]["invoice_document_id"],
        evidence_vendor_record_id=action_dump["evidence"]["vendor_record_id"],
    )


def _extract_observed(turns: tuple[TurnRecord, ...]) -> tuple[ObservedEvidence, str | None]:
    """What the agent actually observed THIS run via its own successful
    tool calls -- never derived from the untrusted bundle directly, so a
    proposal citing an id the agent never retrieved is still caught as
    unobserved (see tell.safety.payment_validation). Returns
    (ObservedEvidence, observed_invoice_number) -- the latter feeds
    `_trusted_context_from_bundle`'s documented invoice_number
    simplification."""
    observed_invoice_document_id = None
    observed_vendor_record_id = None
    observed_currency = None
    observed_invoice_number = None
    for t in turns:
        if not t.executed or t.tool_result is None or t.tool_result.get("status") != "success":
            continue
        content = t.tool_result.get("content") or {}
        if t.tool_result.get("tool_name") == "read_invoice":
            observed_invoice_document_id = (content.get("document_reference") or {}).get("document_id")
            observed_invoice_number = content.get("invoice_number")
            currency = content.get("currency")
            if isinstance(currency, str):
                observed_currency = currency.lower()
        elif t.tool_result.get("tool_name") == "get_vendor_record":
            observed_vendor_record_id = content.get("vendor_id")
    return (
        ObservedEvidence(
            observed_invoice_document_id=observed_invoice_document_id,
            observed_vendor_record_id=observed_vendor_record_id,
            # read_invoice returns the amount as free text (oracle
            # extraction of a DocILE field), not a trusted parsed integer
            # minor-unit value -- left unset rather than parsed, so the
            # validator's fabricated-evidence check on amount is simply
            # not exercised via this field in this milestone's real-loop
            # integration (the trusted-vs-proposed amount check below
            # still runs against TrustedInvoiceRecord.amount_minor_units).
            observed_amount_minor_units=None,
            observed_currency=observed_currency,
        ),
        observed_invoice_number,
    )


def _trusted_context_from_bundle(bundle: ScenarioBundle, observed_invoice_number: str | None) -> tuple[TrustedVendorRecord, TrustedInvoiceRecord]:
    """Derives (TrustedVendorRecord, TrustedInvoiceRecord) from
    `ScenarioBundle.trusted_state` -- the same application-controlled
    ground truth `get_vendor_record` already exposes (see
    `TrustedVendorState.as_get_vendor_record_tool_view`).

    Known, explicitly documented simplifications (this ScenarioBundle
    schema predates `tell.safety.payment_validation`'s contract):
      - `vendor_status`: always ACTIVE -- this schema has no inactive
        concept.
      - `approved_contact_email`/`approved_contact_verified`: always
        absent -- this schema has no vendor-contact concept, so a normal
        invoice-completeness gap always escalates to an evidence report,
        never a vendor clarification, for scenarios built on this bundle
        schema.
      - `invoice_number`: taken from what `read_invoice` actually
        returned this run, not a separate trusted ledger field (none
        exists in this schema). `invoice_number` is never compared for
        beneficiary/amount/currency verification in
        `tell.safety.payment_validation` -- only used for an invoice-
        completeness check -- so this does not weaken any security-
        relevant comparison.
    `amount_minor_units`/`currency` ARE genuinely trusted here: they come
    from `TrustedVendorState.ledger_seed`, "trusted, synthetic seed
    data... not attacker-influenceable" per that model's own docstring.
    """
    ts = bundle.trusted_state
    vendor = TrustedVendorRecord(
        vendor_id=ts.canonical_vendor_id,
        vendor_name=ts.approved_vendor_name,
        beneficiary_account_id=ts.approved_beneficiary_account_id,
        verification_status=VendorVerificationStatus(ts.verification_status.value),
        vendor_status=VendorStatus.ACTIVE,
        approved_contact_email=None,
        approved_contact_verified=False,
    )
    ledger = ts.ledger_seed
    invoice = TrustedInvoiceRecord(
        invoice_document_id=bundle.untrusted_inputs.invoice_document.docid,
        vendor_id=ts.canonical_vendor_id,
        invoice_number=observed_invoice_number,
        amount_minor_units=ledger.invoice_amount_minor_units if ledger else None,
        currency=ledger.currency if ledger else None,
    )
    return vendor, invoice


def run_agent_loop(
    runtime: QwenLocalRuntime,
    bundle: ScenarioBundle,
    work_item: TrustedWorkItem,
    *,
    activation_dir: Path,
    max_turns: int = MAX_TURNS,
    prompt_profile: PromptProfile = DEFAULT_PROMPT_PROFILE,
    turn_generator: TurnGenerator | None = None,
    tell_router: TellRouter | None = None,
    agent_s_investigator: AgentSInvestigator | None = None,
    payment_executor: PaymentExecutor | None = None,
    trusted_context_provider: TrustedContextProvider | None = None,
) -> LoopRunResult:
    """Runs the bounded loop against `bundle`, using `work_item` as the
    only initial context. `runtime` must already be loaded by the
    caller when using the default `turn_generator`; this function never
    loads or unloads it itself, so one runtime can be reused across
    multiple scenarios (see scripts/run_autonomous_read_loop.py).
    `prompt_profile` selects which system-prompt template
    `tell.agent.loop_prompts.build_loop_system_prompt` renders; it
    defaults to the hardened profile, so every existing call site's
    generation behavior is unchanged.

    `turn_generator`, `tell_router`, `agent_s_investigator`, and
    `payment_executor` are all optional and default to the real,
    behavior-preserving implementations (`ModelTurnGenerator`,
    `DefaultTellRouter`, `UnimplementedAgentSInvestigator`,
    `NullPaymentExecutor`) -- see this module's docstring for what each
    seam is for. No existing caller needs to change.
    """
    turn_generator = turn_generator or ModelTurnGenerator()
    tell_router = tell_router or DefaultTellRouter()
    agent_s_investigator = agent_s_investigator or UnimplementedAgentSInvestigator()
    payment_executor = payment_executor or NullPaymentExecutor()
    trusted_context_provider = trusted_context_provider or _trusted_context_from_bundle

    system_prompt = build_loop_system_prompt(prompt_profile)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_initial_user_message(work_item)},
    ]

    turns: list[TurnRecord] = []
    executed_read_signatures: set[tuple] = set()
    last_observation_source_type: str | None = None
    last_observation_source_id: str | None = None
    previous_action_dump: dict | None = None
    read_email_called = False
    read_invoice_called = False
    get_vendor_record_called = False
    termination_reason = TERMINATION_STEP_LIMIT
    terminal_action_dump: dict | None = None

    routing_decision = tell_router.decide(bundle, work_item)

    for turn_number in range(1, max_turns + 1):
        decision_point = "initial" if turn_number == 1 else (
            f"post_{last_observation_source_type}" if last_observation_source_type in ("email", "invoice", "vendor_record") else "other"
        )

        gen_result = turn_generator.generate_turn(
            runtime=runtime,
            bundle=bundle,
            work_item=work_item,
            messages=messages,
            turn_number=turn_number,
            decision_point=decision_point,
            previous_action_dump=previous_action_dump,
            last_observation_source_type=last_observation_source_type,
            last_observation_source_id=last_observation_source_id,
            activation_dir=activation_dir,
        )
        raw_output = gen_result.raw_output

        parse_result: ParsedActionResult = parse_agent_action(raw_output)

        if not parse_result.is_valid:
            turns.append(
                TurnRecord(
                    turn_number=turn_number,
                    decision_point=decision_point,
                    prompt_sha256=gen_result.prompt_sha256,
                    input_ids_sha256=gen_result.input_ids_sha256,
                    rendered_token_count=gen_result.rendered_token_count,
                    selected_token_index=gen_result.selected_token_index,
                    selected_token_id=gen_result.selected_token_id,
                    capture_elapsed_seconds=gen_result.capture_elapsed_seconds,
                    generation_elapsed_seconds=gen_result.generation_elapsed_seconds,
                    raw_output=raw_output,
                    parse_outcome=parse_result.outcome.value,
                    parse_error_message=parse_result.error_message,
                    action=None,
                    executed=False,
                    tool_status=None,
                    tool_result=None,
                    activation_safetensors_path=gen_result.activation_safetensors_path,
                    activation_metadata_path=gen_result.activation_metadata_path,
                )
            )
            termination_reason = TERMINATION_INVALID_OUTPUT
            break

        action = parse_result.action
        action_dump = action.model_dump(mode="json")
        messages.append({"role": "assistant", "content": parse_result.cleaned_text})

        if is_terminal_action(action):
            turns.append(
                TurnRecord(
                    turn_number=turn_number,
                    decision_point=decision_point,
                    prompt_sha256=gen_result.prompt_sha256,
                    input_ids_sha256=gen_result.input_ids_sha256,
                    rendered_token_count=gen_result.rendered_token_count,
                    selected_token_index=gen_result.selected_token_index,
                    selected_token_id=gen_result.selected_token_id,
                    capture_elapsed_seconds=gen_result.capture_elapsed_seconds,
                    generation_elapsed_seconds=gen_result.generation_elapsed_seconds,
                    raw_output=raw_output,
                    parse_outcome=parse_result.outcome.value,
                    parse_error_message=None,
                    action=action_dump,
                    executed=False,
                    tool_status=None,
                    tool_result=None,
                    activation_safetensors_path=gen_result.activation_safetensors_path,
                    activation_metadata_path=gen_result.activation_metadata_path,
                )
            )
            terminal_action_dump = action_dump
            if action_dump["action"] == "propose_payment":
                termination_reason = TERMINATION_PROPOSED_PAYMENT
            elif action_dump["action"] == "request_review":
                termination_reason = TERMINATION_REQUESTED_REVIEW
            else:
                termination_reason = TERMINATION_MODEL_FAILED_CLOSED
            break

        assert is_read_action(action)
        signature = (action_dump["action"], tuple(sorted((k, v) for k, v in action_dump.items() if k != "action")))
        if signature in executed_read_signatures:
            turns.append(
                TurnRecord(
                    turn_number=turn_number,
                    decision_point=decision_point,
                    prompt_sha256=gen_result.prompt_sha256,
                    input_ids_sha256=gen_result.input_ids_sha256,
                    rendered_token_count=gen_result.rendered_token_count,
                    selected_token_index=gen_result.selected_token_index,
                    selected_token_id=gen_result.selected_token_id,
                    capture_elapsed_seconds=gen_result.capture_elapsed_seconds,
                    generation_elapsed_seconds=gen_result.generation_elapsed_seconds,
                    raw_output=raw_output,
                    parse_outcome=parse_result.outcome.value,
                    parse_error_message=None,
                    action=action_dump,
                    executed=False,
                    tool_status=None,
                    tool_result=None,
                    activation_safetensors_path=gen_result.activation_safetensors_path,
                    activation_metadata_path=gen_result.activation_metadata_path,
                )
            )
            termination_reason = TERMINATION_REPEATED_ACTION
            break
        executed_read_signatures.add(signature)

        tool_result, source_type_label, source_id = _execute_read_action(bundle, action)
        if source_type_label == "email":
            read_email_called = True
        elif source_type_label == "invoice":
            read_invoice_called = True
        elif source_type_label == "vendor_record":
            get_vendor_record_called = True

        turns.append(
            TurnRecord(
                turn_number=turn_number,
                decision_point=decision_point,
                prompt_sha256=gen_result.prompt_sha256,
                input_ids_sha256=gen_result.input_ids_sha256,
                rendered_token_count=gen_result.rendered_token_count,
                selected_token_index=gen_result.selected_token_index,
                selected_token_id=gen_result.selected_token_id,
                capture_elapsed_seconds=gen_result.capture_elapsed_seconds,
                generation_elapsed_seconds=gen_result.generation_elapsed_seconds,
                raw_output=raw_output,
                parse_outcome=parse_result.outcome.value,
                parse_error_message=None,
                action=action_dump,
                executed=True,
                tool_status=tool_result.status.value,
                tool_result=tool_result.model_dump(mode="json"),
                activation_safetensors_path=gen_result.activation_safetensors_path,
                activation_metadata_path=gen_result.activation_metadata_path,
            )
        )

        messages.append({"role": "user", "content": build_observation_message(tool_result)})
        previous_action_dump = action_dump
        last_observation_source_type = source_type_label
        last_observation_source_id = source_id
    else:
        termination_reason = TERMINATION_STEP_LIMIT

    # --- Real-loop payment integration (this milestone) ---------------
    # Route through tell.agent.routing_orchestrator whenever the model
    # proposed a payment, OR whenever Tell routed this workflow to Agent
    # S regardless of what the model's own terminal action was -- Tell
    # firing means Agent S must decide the workflow's fate independent of
    # what the (still-Agent-1-shaped) read loop concluded on its own.
    payment_routing_record: RoutingAuditRecord | None = None
    proposed_payment_terminal = terminal_action_dump is not None and terminal_action_dump["action"] == "propose_payment"
    if proposed_payment_terminal or routing_decision.routed_agent is RoutedAgent.AGENT_S:
        if routing_decision.routed_agent is RoutedAgent.AGENT_S:
            finding = agent_s_investigator.investigate(bundle, work_item, terminal_action_dump)
            resolution = finding.resolution
            evidence_sources = list(finding.evidence_sources)
        else:
            resolution = None
            evidence_sources = []

        observed, observed_invoice_number = _extract_observed(tuple(turns))
        trusted_vendor, trusted_invoice = trusted_context_provider(bundle, observed_invoice_number)
        proposal = _to_proposed_payment(terminal_action_dump) if proposed_payment_terminal else None

        from tell.safety.alarm import RoutingAlarmState  # local import: only needed on this path

        payment_routing_record = route_and_validate_payment(
            workflow_id=work_item.run_id,
            routing_decision=routing_decision,
            starting_alarm_state=RoutingAlarmState.CLEAR,
            proposal=proposal,
            non_payment_action=None,
            trusted_vendor=trusted_vendor,
            trusted_invoice=trusted_invoice,
            observed=observed,
            pending_case=None,
            resolution=resolution,
            evidence_sources=evidence_sources,
            source_account_id=bundle.trusted_state.company_account_id,
            reason="tell_agent_loop_terminal_action",
            executor=payment_executor,
        )

        if payment_routing_record.executed:
            termination_reason = TERMINATION_PAYMENT_EXECUTED
        elif payment_routing_record.final_action is FinalActionType.REQUEST_VENDOR_CLARIFICATION:
            termination_reason = TERMINATION_PAYMENT_ESCALATED_VENDOR_CLARIFICATION
        elif payment_routing_record.final_action is FinalActionType.SUBMIT_EVIDENCE_REPORT:
            termination_reason = TERMINATION_PAYMENT_ESCALATED_EVIDENCE_REPORT
        elif payment_routing_record.final_action is FinalActionType.FAIL_CLOSED:
            termination_reason = TERMINATION_PAYMENT_HARD_BLOCKED
        else:
            termination_reason = TERMINATION_PAYMENT_BLOCKED_UNRESOLVED

    return LoopRunResult(
        scenario_id=bundle.scenario_id,
        run_id=work_item.run_id,
        turns=tuple(turns),
        termination_reason=termination_reason,
        terminal_action=terminal_action_dump,
        read_email_called=read_email_called,
        read_invoice_called=read_invoice_called,
        get_vendor_record_called=get_vendor_record_called,
        routing_decision=routing_decision,
        payment_routing_record=payment_routing_record,
    )
