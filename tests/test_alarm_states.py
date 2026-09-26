"""CPU-only tests for tell.safety.alarm and tell.safety.resolution (Parts
1, 5, and 8 of the Tell-routing design). No model, no GPU, no ledger."""

from __future__ import annotations

import pytest

from tell.safety.alarm import (
    VALID_TRANSITIONS,
    AlarmStateOwner,
    InvalidAlarmTransitionError,
    RoutingAlarmState,
    apply_transition,
    to_gate_alarm_state,
)
from tell.safety.gate import AlarmState as GateAlarmState
from tell.safety.resolution import (
    ReviewerActorType,
    ReviewerDecisionType,
    UnauthorizedResolutionError,
    resolve_autonomous_injection,
    resolve_from_human_decision,
)

REVIEWERS = frozenset({"SIM-REVIEWER-AP-001"})


def test_every_state_reachable_and_every_transition_documented():
    states_with_outgoing = {t.frm for t in VALID_TRANSITIONS}
    # every non-terminal state has at least one documented outgoing edge
    for state in RoutingAlarmState:
        if state in (RoutingAlarmState.RESOLVED, RoutingAlarmState.HARD_BLOCKED):
            continue
        assert state in states_with_outgoing, f"{state} has no documented outgoing transition"


def test_apply_transition_succeeds_for_documented_edge():
    t = apply_transition(RoutingAlarmState.CLEAR, RoutingAlarmState.RAISED)
    assert t.owner is AlarmStateOwner.TELL_ROUTER


@pytest.mark.parametrize(
    "frm,to",
    [
        (RoutingAlarmState.CLEAR, RoutingAlarmState.RESOLVED),
        (RoutingAlarmState.RESOLVED, RoutingAlarmState.CLEAR),
        (RoutingAlarmState.HARD_BLOCKED, RoutingAlarmState.CLEAR),
        (RoutingAlarmState.RAISED, RoutingAlarmState.RESOLVED),
    ],
)
def test_apply_transition_rejects_undocumented_edge(frm, to):
    with pytest.raises(InvalidAlarmTransitionError):
        apply_transition(frm, to)


@pytest.mark.parametrize(
    "state,expected_gate_state",
    [
        (RoutingAlarmState.CLEAR, GateAlarmState.CLEAR),
        (RoutingAlarmState.RAISED, GateAlarmState.UNRESOLVED),
        (RoutingAlarmState.RESOLVING, GateAlarmState.UNRESOLVED),
        (RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION, GateAlarmState.UNRESOLVED),
        (RoutingAlarmState.AWAITING_HUMAN_VERIFICATION, GateAlarmState.UNRESOLVED),
        (RoutingAlarmState.HARD_BLOCKED, GateAlarmState.UNRESOLVED),
    ],
)
def test_to_gate_alarm_state_collapse(state, expected_gate_state):
    gate_state, resolved = to_gate_alarm_state(state)
    assert gate_state is expected_gate_state
    assert resolved is False


def test_resolved_maps_to_recovered_only_with_explicit_resolved_flag():
    gate_state, resolved = to_gate_alarm_state(RoutingAlarmState.RESOLVED, recovery_resolved=True)
    assert gate_state is GateAlarmState.RECOVERED
    assert resolved is True
    gate_state, resolved = to_gate_alarm_state(RoutingAlarmState.RESOLVED)
    assert gate_state is GateAlarmState.RECOVERED
    assert resolved is False  # gate still requires explicit recovery_resolved=True to permit


# ---------------------------------------------------------------------
# 4 & 5: model cannot clear an alarm; model cannot approve its own
# evidence report -- both mean the SAME thing here: no function in
# tell.safety.resolution accepts a non-human-reviewer actor and resolves
# a case.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("actor", [ReviewerActorType.MODEL, ReviewerActorType.AGENT_S, ReviewerActorType.APPLICATION])
def test_only_human_reviewer_actor_type_may_resolve_a_case(actor):
    with pytest.raises(UnauthorizedResolutionError):
        resolve_from_human_decision(
            RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
            actor_type=actor,
            reviewer_id="SIM-REVIEWER-AP-001",
            registered_reviewer_ids=REVIEWERS,
            decision=ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT,
            supporting_trusted_evidence=["VEND-1"],
        )


def test_unregistered_reviewer_id_rejected_even_as_human_reviewer():
    with pytest.raises(UnauthorizedResolutionError):
        resolve_from_human_decision(
            RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
            actor_type=ReviewerActorType.HUMAN_REVIEWER,
            reviewer_id="NOT-REGISTERED",
            registered_reviewer_ids=REVIEWERS,
            decision=ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT,
            supporting_trusted_evidence=["VEND-1"],
        )


def test_registered_human_reviewer_can_resolve_and_clear():
    new_state = resolve_from_human_decision(
        RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
        actor_type=ReviewerActorType.HUMAN_REVIEWER,
        reviewer_id="SIM-REVIEWER-AP-001",
        registered_reviewer_ids=REVIEWERS,
        decision=ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT,
        supporting_trusted_evidence=["VEND-1"],
    )
    assert new_state is RoutingAlarmState.RESOLVED


def test_approval_without_supporting_evidence_rejected():
    with pytest.raises(ValueError):
        resolve_from_human_decision(
            RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
            actor_type=ReviewerActorType.HUMAN_REVIEWER,
            reviewer_id="SIM-REVIEWER-AP-001",
            registered_reviewer_ids=REVIEWERS,
            decision=ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT,
            supporting_trusted_evidence=[],
        )


def test_human_reviewer_can_reject_as_threat_hard_blocking():
    new_state = resolve_from_human_decision(
        RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
        actor_type=ReviewerActorType.HUMAN_REVIEWER,
        reviewer_id="SIM-REVIEWER-AP-001",
        registered_reviewer_ids=REVIEWERS,
        decision=ReviewerDecisionType.REJECTED_AS_THREAT,
        supporting_trusted_evidence=[],
    )
    assert new_state is RoutingAlarmState.HARD_BLOCKED


# ---------------------------------------------------------------------
# Agent S cannot clear its own alarm via the autonomous path either --
# resolve_autonomous_injection takes no actor identity, only a
# caller-verified boolean; it cannot be invoked with "Agent S says so".
# ---------------------------------------------------------------------


def test_agent_s_cannot_clear_its_own_alarm():
    import inspect

    sig = inspect.signature(resolve_autonomous_injection)
    assert "actor_type" not in sig.parameters
    with pytest.raises(ValueError):
        resolve_autonomous_injection(RoutingAlarmState.RESOLVING, injection_contradicted_by_trusted_vendor_record=False)


def test_autonomous_resolution_succeeds_only_from_resolving_state_with_verified_contradiction():
    new_state = resolve_autonomous_injection(RoutingAlarmState.RESOLVING, injection_contradicted_by_trusted_vendor_record=True)
    assert new_state is RoutingAlarmState.RESOLVED
    with pytest.raises(InvalidAlarmTransitionError):
        resolve_autonomous_injection(RoutingAlarmState.RAISED, injection_contradicted_by_trusted_vendor_record=True)
