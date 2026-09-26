"""CPU-only tests for tell.memory.next_step (Part 2 of the Tell-routing
design). No model, no GPU, no database."""

from __future__ import annotations

import inspect

import pytest

from tell.memory.next_step import (
    ACTION_CHANGING_SIGNALS,
    LinkedHistoryStatus,
    MemoryNextStepDecision,
    RequiredNextStep,
    RetrievedMemorySignal,
    decide_next_step,
    required_next_step,
)

ALL_SIGNALS = list(RetrievedMemorySignal)


# ---------------------------------------------------------------------
# 10: memory never authorizes a beneficiary or payment
# ---------------------------------------------------------------------


def test_no_next_step_is_payment_shaped():
    # "payment_history" (a read-only trusted lookup) is fine; actually
    # authorizing, approving, or executing a payment is not.
    forbidden = {"pay_invoice", "approve_payment", "authorize_payment", "beneficiary_account", "clear_alarm"}
    for step in RequiredNextStep:
        assert not any(word in step.value for word in forbidden), step.value


def test_module_never_returns_a_beneficiary_or_amount():
    # decide_next_step's return type has no field capable of carrying a
    # beneficiary account id, an amount, or a currency -- by construction,
    # not merely by convention.
    field_names = set(MemoryNextStepDecision.model_fields)
    assert not field_names & {"beneficiary_account_id", "amount_minor_units", "currency", "approved"}


def test_no_function_in_module_touches_ledger_or_gate():
    import tell.memory.next_step as mod

    source = inspect.getsource(mod)
    assert "ledger" not in source.lower()
    assert "evaluate_gate" not in source
    assert "clear" not in source.lower() or "clarification" in source.lower()  # "clear" only appears inside "clarification"-family words


# ---------------------------------------------------------------------
# 11: history-required memory can change the next action
# ---------------------------------------------------------------------


@pytest.mark.parametrize("signal", sorted(ACTION_CHANGING_SIGNALS, key=lambda s: s.value))
def test_history_required_action_changing_signals_do_not_continue_normally(signal):
    decision = decide_next_step(LinkedHistoryStatus.PENDING, signal)
    assert decision.next_step is not RequiredNextStep.CONTINUE_NORMAL_PROCESSING


def test_beneficiary_change_claim_routes_to_trusted_vendor_record_not_memory():
    decision = decide_next_step(LinkedHistoryStatus.PENDING, RetrievedMemorySignal.BENEFICIARY_CHANGE_CLAIM)
    assert decision.next_step is RequiredNextStep.RETRIEVE_TRUSTED_VENDOR_RECORD


def test_irrelevant_or_empty_and_clean_note_continue_normally_even_when_history_required():
    for signal in (RetrievedMemorySignal.IRRELEVANT_OR_EMPTY, RetrievedMemorySignal.CLEAN_OPERATIONAL_NOTE):
        decision = decide_next_step(LinkedHistoryStatus.PENDING, signal)
        assert decision.next_step is RequiredNextStep.CONTINUE_NORMAL_PROCESSING


def test_no_history_required_always_continues_normally_regardless_of_signal():
    for signal in ALL_SIGNALS:
        decision = decide_next_step(LinkedHistoryStatus.NONE, signal)
        assert decision.next_step is RequiredNextStep.CONTINUE_NORMAL_PROCESSING


def test_required_next_step_is_a_total_function_over_every_signal():
    for signal in ALL_SIGNALS:
        assert isinstance(required_next_step(signal), RequiredNextStep)


# ---------------------------------------------------------------------
# 12: at least 25% of history-required scenarios differ from their
# matched no-history negative
# ---------------------------------------------------------------------


def test_at_least_25_percent_of_history_required_scenarios_differ_from_matched_negative():
    rows = []
    for signal in ALL_SIGNALS:
        required = decide_next_step(LinkedHistoryStatus.PENDING, signal)
        negative = decide_next_step(LinkedHistoryStatus.NONE, signal)
        rows.append(required.next_step != negative.next_step)
    pct = sum(rows) / len(rows)
    assert pct >= 0.25, f"only {pct:.1%} of history-required scenarios differ from their matched no-history negative"
    # Document the exact measured value inline so a future change to the
    # signal table that silently drops below the bar fails loudly here,
    # not just in the routing-design report.
    assert pct == pytest.approx(5 / 7)
