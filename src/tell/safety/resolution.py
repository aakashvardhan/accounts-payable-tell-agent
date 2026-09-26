"""The resolution validator (Part 1 / Part 5 / Part 8 of the Tell-routing
design): deterministic application code that decides whether trusted
evidence resolves a raised alarm. Never the model, and never Agent S
itself -- see `ReviewerActorType` and the actor check in
`resolve_from_human_decision` below, and
tests/test_alarm_states.py::test_agent_s_cannot_clear_its_own_alarm.

Responsibility boundary
--------------------------------------------------------------------------
This module owns exactly one thing: given a `RoutingAlarmState` currently
at `RESOLVING` or `AWAITING_HUMAN_VERIFICATION`, and either (a) a typed
decision from a registered human reviewer, or (b) a deterministic finding
that trusted evidence already contradicts an injection with no other
block present, decide the NEXT `RoutingAlarmState` via
`tell.safety.alarm.apply_transition` (the closed transition table -- an
attempt to move to any state not in that table raises). It does not touch
the ledger, the gate, or `tell.safety.payment_validation`; those stay
exactly as defined in their own modules and are unaffected by this one.
"""

from __future__ import annotations

from enum import Enum

from tell.safety.alarm import InvalidAlarmTransitionError, RoutingAlarmState, apply_transition


class ReviewerActorType(str, Enum):
    """Every possible caller identity this module can be told about. Only
    `HUMAN_REVIEWER` may ever resolve a case via
    `resolve_from_human_decision` -- see that function's actor check."""

    HUMAN_REVIEWER = "human_reviewer"
    MODEL = "model"
    AGENT_S = "agent_s"
    APPLICATION = "application"


class UnauthorizedResolutionError(Exception):
    """Raised whenever a non-human-reviewer actor (including Agent S and
    the model) attempts to resolve a case that requires human sign-off."""


class ReviewerDecisionType(str, Enum):
    """A registered human reviewer's typed decision on an evidence-report
    case. No free-text field exists anywhere in this module -- a decision
    is exactly one of these six values, never chain-of-thought."""

    APPROVED_CANONICAL_PAYMENT = "approved_canonical_payment"
    VENDOR_CLARIFICATION_REQUIRED = "vendor_clarification_required"
    TRUSTED_VENDOR_UPDATE_REQUIRED = "trusted_vendor_update_required"
    REJECTED_AS_THREAT = "rejected_as_threat"
    INVOICE_REJECTED = "invoice_rejected"
    ADDITIONAL_EVIDENCE_REQUIRED = "additional_evidence_required"


# decision -> the RoutingAlarmState.AWAITING_HUMAN_VERIFICATION transition
# target. Every target here must be a real edge in
# tell.safety.alarm.VALID_TRANSITIONS or apply_transition raises.
_DECISION_TARGET: dict[ReviewerDecisionType, RoutingAlarmState] = {
    ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT: RoutingAlarmState.RESOLVED,
    ReviewerDecisionType.REJECTED_AS_THREAT: RoutingAlarmState.HARD_BLOCKED,
    ReviewerDecisionType.INVOICE_REJECTED: RoutingAlarmState.HARD_BLOCKED,
    ReviewerDecisionType.VENDOR_CLARIFICATION_REQUIRED: RoutingAlarmState.RESOLVING,
    ReviewerDecisionType.TRUSTED_VENDOR_UPDATE_REQUIRED: RoutingAlarmState.RESOLVING,
    ReviewerDecisionType.ADDITIONAL_EVIDENCE_REQUIRED: RoutingAlarmState.RESOLVING,
}


def resolve_from_human_decision(
    current_state: RoutingAlarmState,
    *,
    actor_type: ReviewerActorType,
    reviewer_id: str,
    registered_reviewer_ids: frozenset[str],
    decision: ReviewerDecisionType,
    supporting_trusted_evidence: list[str],
) -> RoutingAlarmState:
    """The only function in this codebase that may move a case out of
    `AWAITING_HUMAN_VERIFICATION`. Raises `UnauthorizedResolutionError` for
    any actor_type other than `HUMAN_REVIEWER` (this covers the model and
    Agent S identically -- neither may ever approve its own escalation),
    and for a `reviewer_id` not in the caller-supplied registry of actually
    registered reviewers. An `APPROVED_CANONICAL_PAYMENT` decision without
    at least one cited trusted-evidence reference is rejected outright: an
    approval must point at something.
    """
    if actor_type is not ReviewerActorType.HUMAN_REVIEWER:
        raise UnauthorizedResolutionError(f"actor_type={actor_type.value!r} may not resolve a case; only a registered human reviewer can")
    if reviewer_id not in registered_reviewer_ids:
        raise UnauthorizedResolutionError(f"reviewer_id {reviewer_id!r} is not a registered reviewer")
    if current_state is not RoutingAlarmState.AWAITING_HUMAN_VERIFICATION:
        raise InvalidAlarmTransitionError(f"a human-reviewer decision only applies to a case in awaiting_human_verification, not {current_state.value}")
    if decision is ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT and not supporting_trusted_evidence:
        raise ValueError("an approval must cite at least one supporting trusted-evidence reference")

    target = _DECISION_TARGET[decision]
    return apply_transition(current_state, target).to


def resolve_autonomous_injection(current_state: RoutingAlarmState, *, injection_contradicted_by_trusted_vendor_record: bool) -> RoutingAlarmState:
    """The deterministic autonomous-resolution path (Part 5, step 7: "the
    deterministic resolution validator may clear the alarm only from typed
    trusted evidence"). `injection_contradicted_by_trusted_vendor_record`
    must already have been established by trusted, deterministic evidence
    (e.g. `tell.safety.payment_validation.validate_payment_proposal`
    finding that the untrusted claim's beneficiary does not match, and
    nothing else, blocks the proposal) -- this function does not compute
    that boolean itself and does not accept Agent S's own say-so for it;
    it only performs the state transition once the caller (application
    code) has already verified the condition.
    """
    if current_state is not RoutingAlarmState.RESOLVING:
        raise InvalidAlarmTransitionError(f"autonomous resolution only applies to a case in resolving, not {current_state.value}")
    if not injection_contradicted_by_trusted_vendor_record:
        raise ValueError("cannot autonomously resolve without a verified trusted-record contradiction")
    return apply_transition(current_state, RoutingAlarmState.RESOLVED).to
