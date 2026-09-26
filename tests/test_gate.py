"""Tests for the deterministic action gate (tell.safety.gate)."""

from __future__ import annotations

import inspect

import pytest

from tell.agent.tools import PayInvoiceCandidate
from tell.safety.gate import (
    AlarmState,
    GateDecisionType,
    GateReasonCode,
    evaluate_gate,
)

VALID_CANDIDATE = PayInvoiceCandidate(
    invoice_id="04d531ca811f448a91c6ff4e",
    source_account_id="SIM-COMPANY-ACCT-0001",
    beneficiary_account_id="SIM-BENEFICIARY-ACCT-MDSPHARMA-0001",
    amount_minor_units=3_000_000,
    currency="usd",
    reason="clean invoice payment",
)


def test_clear_state_permits_valid_candidate():
    decision = evaluate_gate(VALID_CANDIDATE, AlarmState.CLEAR)
    assert decision.decision == GateDecisionType.PERMIT
    assert decision.reason_code == GateReasonCode.CLEAR_STATE_PERMITTED
    assert decision.permitted_action is not None
    assert decision.permitted_action.beneficiary_account_id == VALID_CANDIDATE.beneficiary_account_id
    assert decision.permitted_action.amount_minor_units == VALID_CANDIDATE.amount_minor_units


def test_unresolved_alarm_blocks_payment():
    decision = evaluate_gate(VALID_CANDIDATE, AlarmState.UNRESOLVED)
    assert decision.decision == GateDecisionType.BLOCK
    assert decision.reason_code == GateReasonCode.ALARM_UNRESOLVED
    assert decision.permitted_action is None


def test_recovered_without_resolution_blocks():
    decision = evaluate_gate(VALID_CANDIDATE, AlarmState.RECOVERED, recovery_resolved=False)
    assert decision.decision == GateDecisionType.BLOCK
    assert decision.reason_code == GateReasonCode.RECOVERY_NOT_RESOLVED
    assert decision.permitted_action is None


def test_recovered_with_explicit_resolution_permits():
    decision = evaluate_gate(VALID_CANDIDATE, AlarmState.RECOVERED, recovery_resolved=True)
    assert decision.decision == GateDecisionType.PERMIT
    assert decision.reason_code == GateReasonCode.RECOVERY_RESOLVED_PERMITTED
    assert decision.permitted_action is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amount_minor_units": 0},
        {"amount_minor_units": -1},
        {"invoice_id": ""},
        {"source_account_id": ""},
        {"beneficiary_account_id": ""},
        {"currency": "us"},
        {"reason": ""},
    ],
)
def test_structurally_invalid_candidate_always_blocked_regardless_of_alarm_state(kwargs):
    data = VALID_CANDIDATE.model_dump()
    data.update(kwargs)
    candidate = PayInvoiceCandidate(**data)
    for alarm_state in (AlarmState.CLEAR, AlarmState.UNRESOLVED, AlarmState.RECOVERED):
        decision = evaluate_gate(candidate, alarm_state, recovery_resolved=True)
        assert decision.decision == GateDecisionType.BLOCK
        assert decision.reason_code == GateReasonCode.STRUCTURALLY_INVALID
        assert decision.permitted_action is None


def test_gate_has_no_unconditional_beneficiary_equality_rule():
    """The gate must permit a CLEAR-state candidate regardless of which
    beneficiary it names -- it is not the gate's job to compare the
    candidate's beneficiary against any trusted record. If this test ever
    fails because someone added a beneficiary check to evaluate_gate, that
    violates the documented scope in gate.py."""
    attacker_beneficiary_candidate = PayInvoiceCandidate(
        invoice_id=VALID_CANDIDATE.invoice_id,
        source_account_id=VALID_CANDIDATE.source_account_id,
        beneficiary_account_id="SIM-COMPLETELY-DIFFERENT-ACCOUNT-0099",
        amount_minor_units=VALID_CANDIDATE.amount_minor_units,
        currency=VALID_CANDIDATE.currency,
        reason="a different beneficiary entirely",
    )
    decision = evaluate_gate(attacker_beneficiary_candidate, AlarmState.CLEAR)
    assert decision.decision == GateDecisionType.PERMIT, (
        "evaluate_gate must not independently reject a candidate based on "
        "beneficiary identity alone -- that is Tell's job via alarm_state, "
        "not the gate's"
    )


def test_evaluate_gate_signature_has_no_scenario_bundle_or_evaluation_access():
    """Structural guarantee that the gate cannot read evaluation_only or
    infer whether a scenario is attacked: its only parameters are a
    candidate, an alarm state, and a boolean flag."""
    sig = inspect.signature(evaluate_gate)
    param_names = list(sig.parameters.keys())
    assert param_names == ["candidate", "alarm_state", "recovery_resolved"]
    for param in sig.parameters.values():
        annotation = str(param.annotation)
        assert "ScenarioBundle" not in annotation
        assert "EvaluationOnly" not in annotation


def test_gate_is_a_pure_function_same_inputs_same_output():
    d1 = evaluate_gate(VALID_CANDIDATE, AlarmState.CLEAR)
    d2 = evaluate_gate(VALID_CANDIDATE, AlarmState.CLEAR)
    assert d1 == d2
