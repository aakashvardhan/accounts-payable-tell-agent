"""Tell's routing alarm-state model (Part 5 of the Tell-routing design;
see results/routing_design/tell_agent_routing_v1.md for the full design
rationale).

This module is purely additive: it does NOT modify `tell.safety.gate`.
`tell.safety.gate.AlarmState` stays exactly as it is (`CLEAR`,
`UNRESOLVED`, `RECOVERED`) -- see that module's docstring for why it is
deliberately narrow and unaware of vendor/beneficiary state. This module
defines a richer, seven-state routing-level alarm lifecycle used by Tell's
probe-router and the resolution validator, and provides one pure function,
`to_gate_alarm_state`, that collapses a `RoutingAlarmState` down to the
gate's own three-value enum at the one boundary where the gate is actually
consulted. The gate's logic, its tests, and its narrow scope are
untouched.

Why a separate enum instead of extending `tell.safety.gate.AlarmState`
--------------------------------------------------------------------------
`tell.safety.gate.evaluate_gate` is a pure function whose entire contract
(and the completed, tested experiment built on it -- see tests/test_gate.py
and scripts/enterprise_v2_2/runtime.py, which imports `AlarmState` by
name) is keyed to exactly three states. Adding routing-level states
(`raised`, `resolving`, `awaiting_vendor_clarification`,
`awaiting_human_verification`, `hard_blocked`) directly to that enum would
force `evaluate_gate` to grow new branches or silently treat unfamiliar
states as one of the three -- either changes gate.py's behavior or its
meaning. Keeping the richer state machine in a new module and collapsing
it at the boundary keeps the gate exactly as it was validated.
"""

from __future__ import annotations

from enum import Enum

from tell.safety.gate import AlarmState as GateAlarmState


class RoutingAlarmState(str, Enum):
    """The seven states a workflow's alarm can be in from Tell's routing
    perspective, per Part 5 of the routing design. Distinct from (and a
    strict refinement of) `tell.safety.gate.AlarmState`."""

    CLEAR = "clear"
    RAISED = "raised"
    RESOLVING = "resolving"
    AWAITING_VENDOR_CLARIFICATION = "awaiting_vendor_clarification"
    AWAITING_HUMAN_VERIFICATION = "awaiting_human_verification"
    RESOLVED = "resolved"
    HARD_BLOCKED = "hard_blocked"


class AlarmStateOwner(str, Enum):
    """Who is authorized to move a workflow OUT of a given state. See
    `VALID_TRANSITIONS` below for the full transition table; this enum is
    just the vocabulary of owners used there and in resolution ownership
    (Part 8)."""

    TELL_ROUTER = "tell_router"
    AGENT_S = "agent_s"
    VENDOR = "vendor"
    HUMAN_REVIEWER = "human_reviewer"
    RESOLUTION_VALIDATOR = "resolution_validator"
    DETERMINISTIC_POLICY = "deterministic_policy"


class AlarmTransition:
    """One documented, allowed edge in the alarm-state graph."""

    __slots__ = ("frm", "to", "owner", "description")

    def __init__(self, frm: RoutingAlarmState, to: RoutingAlarmState, owner: AlarmStateOwner, description: str) -> None:
        self.frm = frm
        self.to = to
        self.owner = owner
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"AlarmTransition({self.frm.value} -> {self.to.value}, owner={self.owner.value})"


# The complete, closed set of valid transitions. Anything not listed here
# is invalid and `apply_transition` (below) raises rather than allowing it
# silently -- see Part 5 "Document valid transitions and their owners."
VALID_TRANSITIONS: tuple[AlarmTransition, ...] = (
    AlarmTransition(
        RoutingAlarmState.CLEAR, RoutingAlarmState.RAISED, AlarmStateOwner.TELL_ROUTER,
        "The probe's activation-risk score meets or exceeds the calibrated threshold; side effects freeze and the workflow hands off to Agent S.",
    ),
    AlarmTransition(
        RoutingAlarmState.RAISED, RoutingAlarmState.RESOLVING, AlarmStateOwner.AGENT_S,
        "Agent S has taken the handoff and is inspecting trusted evidence to select a safe next action.",
    ),
    AlarmTransition(
        RoutingAlarmState.RESOLVING, RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION, AlarmStateOwner.AGENT_S,
        "Agent S selected request_vendor_clarification; the application resolved and queued the message to a trusted approved contact.",
    ),
    AlarmTransition(
        RoutingAlarmState.RESOLVING, RoutingAlarmState.AWAITING_HUMAN_VERIFICATION, AlarmStateOwner.AGENT_S,
        "Agent S selected submit_evidence_report; the case is queued for a registered human reviewer.",
    ),
    AlarmTransition(
        RoutingAlarmState.RESOLVING, RoutingAlarmState.RESOLVED, AlarmStateOwner.RESOLUTION_VALIDATOR,
        "Trusted evidence available to Agent S already fully resolves the alarm (e.g. a resolved injection the vendor record contradicts) without a vendor or human round trip.",
    ),
    AlarmTransition(
        RoutingAlarmState.RESOLVING, RoutingAlarmState.HARD_BLOCKED, AlarmStateOwner.DETERMINISTIC_POLICY,
        "A deterministic hard failure (unsupported currency, inactive vendor, insufficient funds, invalid account) is present; no model or reviewer action can clear it.",
    ),
    AlarmTransition(
        RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION, RoutingAlarmState.RESOLVING, AlarmStateOwner.VENDOR,
        "A vendor reply satisfying the clarification's resume condition has arrived (re-entering the pipeline as new untrusted evidence for Agent S to re-inspect).",
    ),
    AlarmTransition(
        RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION, RoutingAlarmState.AWAITING_HUMAN_VERIFICATION, AlarmStateOwner.DETERMINISTIC_POLICY,
        "No approved contact existed at submission time (or the clarification attempt itself was rejected); the application converts it into an internal evidence report.",
    ),
    AlarmTransition(
        RoutingAlarmState.AWAITING_HUMAN_VERIFICATION, RoutingAlarmState.RESOLVING, AlarmStateOwner.HUMAN_REVIEWER,
        "A registered human reviewer has recorded a typed decision that requires Agent S / the resolution validator to re-evaluate (e.g. additional_evidence_required).",
    ),
    AlarmTransition(
        RoutingAlarmState.AWAITING_HUMAN_VERIFICATION, RoutingAlarmState.RESOLVED, AlarmStateOwner.RESOLUTION_VALIDATOR,
        "A registered human reviewer approved the canonical payment with supporting trusted evidence; the resolution validator records the alarm as resolved.",
    ),
    AlarmTransition(
        RoutingAlarmState.AWAITING_HUMAN_VERIFICATION, RoutingAlarmState.HARD_BLOCKED, AlarmStateOwner.HUMAN_REVIEWER,
        "A registered human reviewer rejected the invoice or the claim as a confirmed threat; no payment will ever be made against this workflow.",
    ),
)

_TRANSITION_INDEX = {(t.frm, t.to): t for t in VALID_TRANSITIONS}


class InvalidAlarmTransitionError(Exception):
    pass


def apply_transition(current: RoutingAlarmState, target: RoutingAlarmState) -> AlarmTransition:
    """Returns the matching `AlarmTransition` or raises. Callers use this
    instead of setting `.alarm_state = target` directly, so that the closed
    transition table is the only way a workflow's alarm state can change."""
    key = (current, target)
    if key not in _TRANSITION_INDEX:
        raise InvalidAlarmTransitionError(f"{current.value} -> {target.value} is not a valid routing-alarm transition")
    return _TRANSITION_INDEX[key]


def to_gate_alarm_state(state: RoutingAlarmState, *, recovery_resolved: bool = False) -> tuple[GateAlarmState, bool]:
    """Collapses a `RoutingAlarmState` to the (unchanged) three-value
    `tell.safety.gate.AlarmState` plus the `recovery_resolved` flag
    `evaluate_gate` expects, at the one point the gate is actually called.
    `CLEAR` -> gate CLEAR. `RESOLVED` -> gate RECOVERED with
    recovery_resolved=True (only a `RoutingAlarmState.RESOLVED` produced by
    the resolution validator, never by Agent S itself, may set this).
    Every other state -> gate UNRESOLVED: side effects stay blocked."""
    if state is RoutingAlarmState.CLEAR:
        return GateAlarmState.CLEAR, False
    if state is RoutingAlarmState.RESOLVED:
        return GateAlarmState.RECOVERED, recovery_resolved
    return GateAlarmState.UNRESOLVED, False


TERMINAL_STATES = (RoutingAlarmState.RESOLVED, RoutingAlarmState.HARD_BLOCKED)
BLOCKING_STATES = (
    RoutingAlarmState.RAISED,
    RoutingAlarmState.RESOLVING,
    RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION,
    RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
    RoutingAlarmState.HARD_BLOCKED,
)
