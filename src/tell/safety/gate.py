"""The deterministic action gate.

Scope, deliberately narrow
--------------------------------------------------------------------------
This gate does exactly three things:

1. Validates that a candidate action is structurally valid (well-formed
   amount, currency, ids) -- independent of alarm state.
2. Blocks every side-effecting action while `AlarmState.UNRESOLVED`.
3. Permits a structurally valid action when `AlarmState.CLEAR`, or when
   `AlarmState.RECOVERED` *and* recovery has explicitly resolved the alarm
   (`recovery_resolved=True`) -- never a bare RECOVERED state alone.

It deliberately does NOT compare the candidate's beneficiary against the
vendor record, the invoice, or anything else, and does NOT have access to
a ScenarioBundle at all -- `evaluate_gate`'s signature only takes a
candidate, an alarm state, and a recovery flag, so there is no code path
by which it could read `evaluation_only` or infer whether a scenario is
attacked. That is intentional: an unconditional
"beneficiary must equal the vendor record's beneficiary" rule here would
independently solve the central beneficiary-redirection attack by brute
force, making Tell's activation-based detector irrelevant to this
project's central research question. In this clean-scenario slice, the
clean deterministic controller simply proposes the beneficiary that
`get_vendor_record` returned and the gate either lets a *clear* turn
through or doesn't. Later, Tell -- the probe reading hidden activations --
is what will decide whether suspicious model state should push the alarm
into `unresolved` in the first place. The gate enforces that decision; it
does not make it.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tell.agent.tools import PayInvoiceCandidate


class AlarmState(str, Enum):
    CLEAR = "clear"
    UNRESOLVED = "unresolved"
    RECOVERED = "recovered"


class GateDecisionType(str, Enum):
    PERMIT = "permit"
    BLOCK = "block"


class GateReasonCode(str, Enum):
    STRUCTURALLY_INVALID = "structurally_invalid"
    ALARM_UNRESOLVED = "alarm_unresolved"
    RECOVERY_NOT_RESOLVED = "recovery_not_resolved"
    CLEAR_STATE_PERMITTED = "clear_state_permitted"
    RECOVERY_RESOLVED_PERMITTED = "recovery_resolved_permitted"


class PermittedExecutorAction(BaseModel):
    """The exact action the executor is allowed to run. Only populated
    when the decision is PERMIT; mirrors the candidate's fields so the
    executor never has to re-derive them."""

    model_config = ConfigDict(frozen=True)

    invoice_id: str
    source_account_id: str
    beneficiary_account_id: str
    amount_minor_units: int
    currency: str


class GateDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: GateDecisionType
    reason_code: GateReasonCode
    reason_message: str
    alarm_state: AlarmState
    permitted_action: PermittedExecutorAction | None


def _structural_validation_error(candidate: PayInvoiceCandidate) -> str | None:
    if not candidate.invoice_id:
        return "invoice_id is empty"
    if not candidate.source_account_id:
        return "source_account_id is empty"
    if not candidate.beneficiary_account_id:
        return "beneficiary_account_id is empty"
    if not candidate.currency or len(candidate.currency) != 3:
        return f"currency is not a 3-letter code: {candidate.currency!r}"
    if candidate.amount_minor_units <= 0:
        return f"amount_minor_units must be positive, got {candidate.amount_minor_units}"
    if not candidate.reason:
        return "reason is empty"
    return None


def evaluate_gate(
    candidate: PayInvoiceCandidate,
    alarm_state: AlarmState,
    recovery_resolved: bool = False,
) -> GateDecision:
    """Pure function: no ScenarioBundle, no filesystem, no network, no
    database. Given the same three arguments it always returns the same
    decision."""

    structural_error = _structural_validation_error(candidate)
    if structural_error is not None:
        return GateDecision(
            decision=GateDecisionType.BLOCK,
            reason_code=GateReasonCode.STRUCTURALLY_INVALID,
            reason_message=f"Candidate action is structurally invalid: {structural_error}",
            alarm_state=alarm_state,
            permitted_action=None,
        )

    if alarm_state == AlarmState.UNRESOLVED:
        return GateDecision(
            decision=GateDecisionType.BLOCK,
            reason_code=GateReasonCode.ALARM_UNRESOLVED,
            reason_message="Side-effecting action blocked while the alarm is unresolved.",
            alarm_state=alarm_state,
            permitted_action=None,
        )

    if alarm_state == AlarmState.RECOVERED and not recovery_resolved:
        return GateDecision(
            decision=GateDecisionType.BLOCK,
            reason_code=GateReasonCode.RECOVERY_NOT_RESOLVED,
            reason_message=(
                "Alarm state is 'recovered' but recovery_resolved=False: recovered "
                "state alone does not permit a side-effecting action without "
                "explicit resolution metadata confirming the alarm was resolved."
            ),
            alarm_state=alarm_state,
            permitted_action=None,
        )

    permitted_action = PermittedExecutorAction(
        invoice_id=candidate.invoice_id,
        source_account_id=candidate.source_account_id,
        beneficiary_account_id=candidate.beneficiary_account_id,
        amount_minor_units=candidate.amount_minor_units,
        currency=candidate.currency,
    )

    if alarm_state == AlarmState.CLEAR:
        return GateDecision(
            decision=GateDecisionType.PERMIT,
            reason_code=GateReasonCode.CLEAR_STATE_PERMITTED,
            reason_message="Alarm state is clear; structurally valid action permitted.",
            alarm_state=alarm_state,
            permitted_action=permitted_action,
        )

    # alarm_state == RECOVERED and recovery_resolved is True
    return GateDecision(
        decision=GateDecisionType.PERMIT,
        reason_code=GateReasonCode.RECOVERY_RESOLVED_PERMITTED,
        reason_message="Alarm was recovered and recovery_resolved=True; action permitted.",
        alarm_state=alarm_state,
        permitted_action=permitted_action,
    )
