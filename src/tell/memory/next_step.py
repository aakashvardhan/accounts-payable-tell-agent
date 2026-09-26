"""Operational meaning for retrieved memory (Part 2 of the Tell-routing
design; see results/routing_design/tell_agent_routing_v1.md).

Memory stays untrusted and optional -- nothing here lets a memory record
authorize a payment, update the vendor master, clear an alarm, approve a
beneficiary, prove an invoice was paid, or release the gate (there is no
function in this module that returns anything payment- or alarm-shaped).
What this module adds is that `search_memory` is not ceremonial: given
what a retrieved memory record actually claims, and whether the trusted
work item marked linked history as pending, `required_next_step` returns
a typed action the caller (application code, or -- via the LoRA -- the
model) is expected to take next, per the table in Part 2 of the design.

Every `RequiredNextStep` value routes to a TRUSTED check or a normal
processing action, never to treating the memory's content as authority:
"beneficiary_change_claim" -> retrieve_trusted_vendor_record (verify
against the vendor master, not against the memory), "poisoned_or_
conflicting" -> verify_using_trusted_evidence, and so on. See
tests/test_memory_next_step.py for the required 25%-of-scenarios-differ
measurement (Part 2's explicit acceptance bar).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class RetrievedMemorySignal(str, Enum):
    """What a retrieved memory record (or set of records) appears to say,
    classified from its `memory_kind` and content alone -- never from any
    trust label, since memory carries none. Exactly the left column of
    Part 2's required-behavior table."""

    POSSIBLE_DUPLICATE_INVOICE_OR_PAYMENT_CLAIM = "possible_duplicate_invoice_or_payment_claim"
    PREVIOUS_DISPUTE = "previous_dispute"
    PREVIOUS_VENDOR_CLARIFICATION = "previous_vendor_clarification"
    BENEFICIARY_CHANGE_CLAIM = "beneficiary_change_claim"
    IRRELEVANT_OR_EMPTY = "irrelevant_or_empty"
    POISONED_OR_CONFLICTING = "poisoned_or_conflicting"
    CLEAN_OPERATIONAL_NOTE = "clean_operational_note"


class RequiredNextStep(str, Enum):
    """The right column of Part 2's table. None of these values is, or
    leads directly to, a payment, an alarm-clear, or a beneficiary
    approval -- every non-trivial one names a TRUSTED check to perform."""

    CHECK_TRUSTED_INVOICE_PAYMENT_HISTORY = "check_trusted_invoice_payment_history"
    CHECK_TRUSTED_DISPUTE_CASE_STATUS = "check_trusted_dispute_case_status"
    CHECK_CLARIFICATION_RESUME_CONDITION = "check_clarification_resume_condition"
    RETRIEVE_TRUSTED_VENDOR_RECORD = "retrieve_trusted_vendor_record"
    CONTINUE_NORMAL_PROCESSING = "continue_normal_processing"
    VERIFY_USING_TRUSTED_EVIDENCE = "verify_using_trusted_evidence"


# The exact mapping from Part 2's table. Every "no trusted follow-up
# needed" row maps to CONTINUE_NORMAL_PROCESSING; every other row names a
# specific trusted lookup. `POSSIBLE_DUPLICATE...` deliberately maps to a
# trusted invoice/payment-history check *or* an evidence report, per Part
# 2's table cell -- represented here by the same CHECK_TRUSTED_INVOICE_PAYMENT_HISTORY
# step, since checking trusted history is the always-safe first action;
# whether it turns out to require an evidence report is that check's own
# outcome (see tell.safety.payment_validation.PendingCaseKind /
# ValidationOutcome for what happens if the check finds a real conflict),
# not something a memory signal alone can decide.
_SIGNAL_TO_STEP: dict[RetrievedMemorySignal, RequiredNextStep] = {
    RetrievedMemorySignal.POSSIBLE_DUPLICATE_INVOICE_OR_PAYMENT_CLAIM: RequiredNextStep.CHECK_TRUSTED_INVOICE_PAYMENT_HISTORY,
    RetrievedMemorySignal.PREVIOUS_DISPUTE: RequiredNextStep.CHECK_TRUSTED_DISPUTE_CASE_STATUS,
    RetrievedMemorySignal.PREVIOUS_VENDOR_CLARIFICATION: RequiredNextStep.CHECK_CLARIFICATION_RESUME_CONDITION,
    RetrievedMemorySignal.BENEFICIARY_CHANGE_CLAIM: RequiredNextStep.RETRIEVE_TRUSTED_VENDOR_RECORD,
    RetrievedMemorySignal.IRRELEVANT_OR_EMPTY: RequiredNextStep.CONTINUE_NORMAL_PROCESSING,
    RetrievedMemorySignal.POISONED_OR_CONFLICTING: RequiredNextStep.VERIFY_USING_TRUSTED_EVIDENCE,
    RetrievedMemorySignal.CLEAN_OPERATIONAL_NOTE: RequiredNextStep.CONTINUE_NORMAL_PROCESSING,
}

# Signals for which "history was required" (the trusted work item flagged
# linked_history.check_status == "pending") is meant to change the next
# action relative to the no-history baseline (CONTINUE_NORMAL_PROCESSING).
# Used only for measurement (Part 2's 25% bar), not for routing itself.
ACTION_CHANGING_SIGNALS = frozenset(
    {
        RetrievedMemorySignal.POSSIBLE_DUPLICATE_INVOICE_OR_PAYMENT_CLAIM,
        RetrievedMemorySignal.PREVIOUS_DISPUTE,
        RetrievedMemorySignal.PREVIOUS_VENDOR_CLARIFICATION,
        RetrievedMemorySignal.BENEFICIARY_CHANGE_CLAIM,
        RetrievedMemorySignal.POISONED_OR_CONFLICTING,
    }
)


def required_next_step(signal: RetrievedMemorySignal) -> RequiredNextStep:
    return _SIGNAL_TO_STEP[signal]


class LinkedHistoryStatus(str, Enum):
    """The trusted work item's own flag -- not derived from memory content.
    Mirrors the "linked_history.check_status" concept already used by the
    frozen v2.2 LoRA pretraining contract's search_memory-target check
    (results/lora_training/pretraining_contract_v1/training_distribution_report.md
    section 8: "search_memory is a target only when linked history is
    pending"). This module does not introduce a new trust boundary; it
    only gives the retrieval outcome an operational next step."""

    NONE = "none"
    PENDING = "pending"


class MemoryNextStepDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    linked_history_status: LinkedHistoryStatus
    signal: RetrievedMemorySignal | None
    next_step: RequiredNextStep


def decide_next_step(linked_history_status: LinkedHistoryStatus, signal: RetrievedMemorySignal | None) -> MemoryNextStepDecision:
    """The one entry point application/LoRA-training-target code should
    use. When no history was required (`LinkedHistoryStatus.NONE`), memory
    is optional and the answer is always `CONTINUE_NORMAL_PROCESSING`
    regardless of what search_memory happens to return (mirrors the v2.2
    policy: "search_memory is never a gold target because no work item
    states a historical dependency" -- see
    results/enterprise_corpus/v2_2/resolution_policy.md). When history WAS
    required, an absent/irrelevant signal still continues normally; every
    other signal routes to its trusted check."""
    if linked_history_status is LinkedHistoryStatus.NONE:
        return MemoryNextStepDecision(linked_history_status=linked_history_status, signal=signal, next_step=RequiredNextStep.CONTINUE_NORMAL_PROCESSING)
    if signal is None:
        return MemoryNextStepDecision(linked_history_status=linked_history_status, signal=signal, next_step=RequiredNextStep.CONTINUE_NORMAL_PROCESSING)
    return MemoryNextStepDecision(linked_history_status=linked_history_status, signal=signal, next_step=required_next_step(signal))
