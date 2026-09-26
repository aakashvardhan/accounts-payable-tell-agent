"""The deterministic runtime orchestrator wiring Tell's routing decision,
the always-on payment validator, the alarm/resolution state machine, and
the (unchanged) gate into one coherent, CPU-only, model-free control flow
(Part 3 of the runtime-integration milestone; see
`results/routing_design/tell_runtime_integration_v1.md`).

Why this is a NEW module, not an edit to `tell.agent.loop`
--------------------------------------------------------------------------
`tell.agent.loop.run_agent_loop` (and `conditional_retrieval`,
`memory_loop`) unconditionally load a real Qwen model and call
`.generate()` -- there is no code path through them that doesn't require a
GPU and pinned model weights, and this milestone is explicitly CPU-only
with no model loaded (see this task's forbidden-actions list: no
inference, no GPU). Rather than bolt a probe/validator/gate integration
onto a model-calling loop that cannot be exercised in this milestone's
tests at all, this module defines the SAME control flow those loops will
eventually need, taking every model-shaped decision (which agent path was
routed to, what Agent S concluded from trusted evidence) as an explicit,
typed input -- the "typed test seam" the task requires. `tell.agent.loop`
itself is completely untouched; nothing about its behavior changes.

Non-negotiable ownership boundaries this module enforces by construction
--------------------------------------------------------------------------
- Only `tell.safety.payment_validation.validate_payment_proposal` ever
  constructs a `ValidatedPayInvoiceCandidate`. This module calls that
  function and reads its `.candidate` field; it never constructs one
  itself, and neither does the caller-supplied `routing_decision` or
  `resolution` input (both are pure data, not executable code with access
  to that class).
- Only `tell.safety.resolution.resolve_from_human_decision` /
  `resolve_autonomous_injection` ever move a `RoutingAlarmState` to
  `RESOLVED` or `HARD_BLOCKED`. This module dispatches to those functions
  based on the caller's typed `ResolutionInput`; it never sets
  `alarm_state` directly to any value outside what
  `tell.safety.alarm.apply_transition` / the resolution module return.
- `evaluate_gate` (`tell.safety.gate`, unchanged) is still the only
  function that produces a `PermittedExecutorAction`, and it is called
  with the SAME alarm-agnostic validator result regardless of which agent
  path produced the proposal -- a beneficiary mismatch on the Agent-1
  path (Tell never fired) is validated exactly like one Agent S produced.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field

from tell.routing.records import FinalActionType, ResolutionOwner, RoutedAgent
from tell.safety.alarm import RoutingAlarmState, apply_transition, to_gate_alarm_state
from tell.safety.gate import GateDecision, GateDecisionType, PermittedExecutorAction, evaluate_gate
from tell.safety.payment_validation import (
    ObservedEvidence,
    PendingCaseKind,
    ProposedPayment,
    RequiredFollowUp,
    TrustedInvoiceRecord,
    TrustedVendorRecord,
    ValidationDecision,
    ValidationOutcome,
    validate_payment_proposal,
)
from tell.safety.resolution import (
    ReviewerActorType,
    ReviewerDecisionType,
    resolve_autonomous_injection,
    resolve_from_human_decision,
)


class RoutingDecision(BaseModel):
    """The typed test seam for "which agent path handled this workflow".
    No numeric Tell threshold is calibrated or invoked in this milestone
    -- `probe_score`/`threshold` are optional, informational-only fields;
    `routed_agent` is supplied directly by the caller (a future milestone
    computes it from a real, calibrated threshold; this milestone's tests
    inject it)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    routed_agent: RoutedAgent
    probe_score: float | None = None
    threshold: float | None = None
    reason: str = ""


class NonPaymentAction(str, Enum):
    """A terminal action the selected agent path can choose WITHOUT ever
    producing a payment proposal (e.g. Agent S concludes immediately that
    an evidence report is required)."""

    REQUEST_VENDOR_CLARIFICATION = "request_vendor_clarification"
    SUBMIT_EVIDENCE_REPORT = "submit_evidence_report"
    FAIL_CLOSED = "fail_closed"


# ---------------------------------------------------------------------
# Resolution input: a tagged union mirroring exactly
# tell.safety.resolution's two real resolution functions' parameters, plus
# the two escalation transitions Agent S may trigger without resolving
# anything. This is the ONLY way alarm state can move past RESOLVING in
# this module -- there is no "just set the new state" field anywhere.
# ---------------------------------------------------------------------


class AutonomousResolutionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["autonomous"] = "autonomous"
    injection_contradicted_by_trusted_vendor_record: bool


class HumanReviewerResolutionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["human_reviewer"] = "human_reviewer"
    actor_type: ReviewerActorType
    reviewer_id: str
    registered_reviewer_ids: frozenset[str]
    decision: ReviewerDecisionType
    supporting_trusted_evidence: list[str] = Field(default_factory=list)


class EscalateVendorClarificationInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["escalate_vendor_clarification"] = "escalate_vendor_clarification"


class EscalateEvidenceReportInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["escalate_evidence_report"] = "escalate_evidence_report"


ResolutionInput = Union[AutonomousResolutionInput, HumanReviewerResolutionInput, EscalateVendorClarificationInput, EscalateEvidenceReportInput]


class PaymentExecutor(Protocol):
    """Explicit injection boundary for the executor. Production code will
    eventually inject something backed by `tell.payment.ledger.Ledger`;
    this milestone ships no such implementation and tests inject a spy
    that records calls without any real side effect."""

    def execute(self, action: PermittedExecutorAction) -> None: ...


class RoutingAuditRecord(BaseModel):
    """Typed attribution record (Part 3's audit requirement). Deliberately
    carries no secrets and no untrusted document content -- only
    identifiers, enums, and booleans."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    selected_branch: RoutedAgent
    alarm_state_before: RoutingAlarmState
    alarm_state_after: RoutingAlarmState
    alarm_transitions: list[tuple[RoutingAlarmState, RoutingAlarmState]]
    evidence_sources: list[str]
    resolution_owner: ResolutionOwner
    validator_outcome: ValidationOutcome | None
    candidate_constructed: bool
    candidate_constructed_by: Literal["validator", "none"]
    gate_decision: GateDecisionType | None
    gate_reason_code: str | None
    final_action: FinalActionType
    executed: bool


def _apply_resolution(alarm_state: RoutingAlarmState, resolution: ResolutionInput | None, transitions: list[tuple[RoutingAlarmState, RoutingAlarmState]]) -> RoutingAlarmState:
    if resolution is None:
        return alarm_state
    if isinstance(resolution, AutonomousResolutionInput):
        new_state = resolve_autonomous_injection(alarm_state, injection_contradicted_by_trusted_vendor_record=resolution.injection_contradicted_by_trusted_vendor_record)
    elif isinstance(resolution, HumanReviewerResolutionInput):
        new_state = resolve_from_human_decision(
            alarm_state,
            actor_type=resolution.actor_type,
            reviewer_id=resolution.reviewer_id,
            registered_reviewer_ids=resolution.registered_reviewer_ids,
            decision=resolution.decision,
            supporting_trusted_evidence=resolution.supporting_trusted_evidence,
        )
    elif isinstance(resolution, EscalateVendorClarificationInput):
        new_state = apply_transition(alarm_state, RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION).to
    elif isinstance(resolution, EscalateEvidenceReportInput):
        new_state = apply_transition(alarm_state, RoutingAlarmState.AWAITING_HUMAN_VERIFICATION).to
    else:  # pragma: no cover - exhaustive tagged union
        raise TypeError(f"unknown resolution input: {resolution!r}")
    transitions.append((alarm_state, new_state))
    return new_state


def route_and_validate_payment(
    *,
    workflow_id: str,
    routing_decision: RoutingDecision,
    starting_alarm_state: RoutingAlarmState,
    proposal: ProposedPayment | None,
    non_payment_action: NonPaymentAction | None,
    trusted_vendor: TrustedVendorRecord | None,
    trusted_invoice: TrustedInvoiceRecord | None,
    observed: ObservedEvidence,
    pending_case: PendingCaseKind | None,
    resolution: ResolutionInput | None,
    evidence_sources: list[str],
    source_account_id: str,
    reason: str,
    executor: PaymentExecutor,
) -> RoutingAuditRecord:
    """The single, minimal integration point. Required order (matching
    Part 3's contract exactly):

      1. `routing_decision` names the already-selected branch (Agent 1 or
         Agent S) -- this function does not select it.
      2. If routed to Agent S and the alarm is still `clear`, it moves to
         `raised` (Tell) then `resolving` (Agent S takes the handoff) --
         both deterministic, not judgment calls.
      3. `resolution` (if given) is applied via `tell.safety.resolution`
         /`tell.safety.alarm` ONLY -- this function never sets alarm state
         directly.
      4. If `proposal` is given, `validate_payment_proposal` runs
         regardless of alarm state (Part 1's always-on guarantee).
      5. Only a VALID validator outcome reaches `evaluate_gate`, using the
         alarm state AFTER step 3, collapsed via `to_gate_alarm_state`.
      6. Only a gate PERMIT reaches `executor.execute`.
    """
    transitions: list[tuple[RoutingAlarmState, RoutingAlarmState]] = []
    alarm_state = starting_alarm_state

    if routing_decision.routed_agent is RoutedAgent.AGENT_S and alarm_state is RoutingAlarmState.CLEAR:
        raised = apply_transition(alarm_state, RoutingAlarmState.RAISED).to
        transitions.append((alarm_state, raised))
        alarm_state = raised
        resolving = apply_transition(alarm_state, RoutingAlarmState.RESOLVING).to
        transitions.append((alarm_state, resolving))
        alarm_state = resolving

    alarm_state = _apply_resolution(alarm_state, resolution, transitions)
    recovery_resolved = alarm_state is RoutingAlarmState.RESOLVED

    def record(
        *,
        validator_outcome: ValidationOutcome | None,
        candidate_constructed: bool,
        gate_decision: GateDecision | None,
        final_action: FinalActionType,
        executed: bool,
        resolution_owner: ResolutionOwner,
    ) -> RoutingAuditRecord:
        return RoutingAuditRecord(
            workflow_id=workflow_id,
            selected_branch=routing_decision.routed_agent,
            alarm_state_before=starting_alarm_state,
            alarm_state_after=alarm_state,
            alarm_transitions=transitions,
            evidence_sources=list(evidence_sources),
            resolution_owner=resolution_owner,
            validator_outcome=validator_outcome,
            candidate_constructed=candidate_constructed,
            candidate_constructed_by="validator" if candidate_constructed else "none",
            gate_decision=gate_decision.decision if gate_decision else None,
            gate_reason_code=gate_decision.reason_code.value if gate_decision else None,
            final_action=final_action,
            executed=executed,
        )

    # What the alarm state ALONE implies, if nothing overrides it below --
    # used both as the answer when there is no proposal, and as the
    # fallback when a proposal exists but the gate still blocks it (Part 4
    # scenario 7: a candidate may be validly constructed and still be
    # gate-blocked while the alarm is unresolved; Agent S cannot override
    # that by supplying a proposal).
    pending_states = (RoutingAlarmState.RAISED, RoutingAlarmState.RESOLVING, RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION, RoutingAlarmState.AWAITING_HUMAN_VERIFICATION)
    if alarm_state in pending_states:
        alarm_only_final = {
            RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION: FinalActionType.REQUEST_VENDOR_CLARIFICATION,
            RoutingAlarmState.AWAITING_HUMAN_VERIFICATION: FinalActionType.SUBMIT_EVIDENCE_REPORT,
        }.get(alarm_state, FinalActionType.UNRESOLVED)
        alarm_only_owner = ResolutionOwner.UNRESOLVED
    elif alarm_state is RoutingAlarmState.HARD_BLOCKED:
        alarm_only_final = FinalActionType.FAIL_CLOSED
        alarm_only_owner = ResolutionOwner.HUMAN_REVIEWER if isinstance(resolution, HumanReviewerResolutionInput) else ResolutionOwner.DETERMINISTIC_POLICY
    else:
        alarm_only_final = None  # CLEAR or RESOLVED: alarm state alone does not determine the outcome
        alarm_only_owner = None

    if proposal is None:
        if non_payment_action is not None:
            final = {
                NonPaymentAction.REQUEST_VENDOR_CLARIFICATION: FinalActionType.REQUEST_VENDOR_CLARIFICATION,
                NonPaymentAction.SUBMIT_EVIDENCE_REPORT: FinalActionType.SUBMIT_EVIDENCE_REPORT,
                NonPaymentAction.FAIL_CLOSED: FinalActionType.FAIL_CLOSED,
            }[non_payment_action]
            owner = ResolutionOwner.DETERMINISTIC_POLICY if final is FinalActionType.FAIL_CLOSED else ResolutionOwner.UNRESOLVED
            return record(validator_outcome=None, candidate_constructed=False, gate_decision=None, final_action=final, executed=False, resolution_owner=owner)
        final = alarm_only_final if alarm_only_final is not None else FinalActionType.UNRESOLVED
        owner = alarm_only_owner if alarm_only_owner is not None else ResolutionOwner.UNRESOLVED
        return record(validator_outcome=None, candidate_constructed=False, gate_decision=None, final_action=final, executed=False, resolution_owner=owner)

    # A proposal exists: the always-on validator runs regardless of alarm
    # state (Part 1's central guarantee -- this is the SAME call whether
    # alarm_state is CLEAR, RESOLVED, or still pending/hard-blocked).
    validation: ValidationDecision = validate_payment_proposal(
        proposal, trusted_vendor=trusted_vendor, trusted_invoice=trusted_invoice, observed=observed,
        pending_case=pending_case, source_account_id=source_account_id, reason=reason,
    )

    if not validation.is_valid:
        if validation.required_follow_up is RequiredFollowUp.REQUEST_VENDOR_CLARIFICATION:
            final_action = FinalActionType.REQUEST_VENDOR_CLARIFICATION
        else:
            final_action = FinalActionType.SUBMIT_EVIDENCE_REPORT
        # Caught by the always-on validator independent of alarm/routing --
        # this is a deterministic-policy block, whether or not Tell ever
        # fired (Part 1's central guarantee).
        return record(validator_outcome=validation.outcome, candidate_constructed=False, gate_decision=None, final_action=final_action, executed=False, resolution_owner=ResolutionOwner.DETERMINISTIC_POLICY)

    gate_alarm_state, _ = to_gate_alarm_state(alarm_state, recovery_resolved=recovery_resolved)
    pic = validation.candidate.to_pay_invoice_candidate()
    gate_decision = evaluate_gate(pic, gate_alarm_state, recovery_resolved=recovery_resolved)

    if gate_decision.decision is not GateDecisionType.PERMIT:
        # A validly-constructed candidate is still blocked while the alarm
        # is unresolved/hard-blocked -- Agent S (or Agent 1) cannot
        # override this by supplying a proposal; the gate's own decision
        # is final.
        final = alarm_only_final if alarm_only_final is not None else FinalActionType.UNRESOLVED
        owner = alarm_only_owner if alarm_only_owner is not None else ResolutionOwner.UNRESOLVED
        return record(validator_outcome=validation.outcome, candidate_constructed=True, gate_decision=gate_decision, final_action=final, executed=False, resolution_owner=owner)

    executor.execute(gate_decision.permitted_action)
    owner = ResolutionOwner.AGENT_S if routing_decision.routed_agent is RoutedAgent.AGENT_S else ResolutionOwner.AGENT_1
    return record(validator_outcome=validation.outcome, candidate_constructed=True, gate_decision=gate_decision, final_action=FinalActionType.PROPOSE_PAYMENT_EXECUTED, executed=True, resolution_owner=owner)
