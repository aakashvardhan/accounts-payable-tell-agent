"""CPU-only full-loop tests for tell.agent.routing_orchestrator (Part 3/4
of the runtime-integration milestone). No model is loaded; every
scenario injects a deterministic RoutingDecision/ResolutionInput through
the typed test seam. No real payment, email, database, or network side
effect occurs anywhere in this file -- SpyExecutor only records calls."""

from __future__ import annotations

import pytest

from tell.agent.routing_orchestrator import (
    AutonomousResolutionInput,
    EscalateEvidenceReportInput,
    EscalateVendorClarificationInput,
    HumanReviewerResolutionInput,
    NonPaymentAction,
    RoutingDecision,
    route_and_validate_payment,
)
from tell.routing.records import FinalActionType, ResolutionOwner, RoutedAgent
from tell.safety.alarm import RoutingAlarmState
from tell.safety.gate import GateDecisionType
from tell.safety.payment_validation import (
    ObservedEvidence,
    PendingCaseKind,
    ProposedPayment,
    TrustedInvoiceRecord,
    TrustedVendorRecord,
    ValidationOutcome,
    VendorStatus,
    VendorVerificationStatus,
)
from tell.safety.resolution import ReviewerActorType, ReviewerDecisionType, UnauthorizedResolutionError

REGISTERED_REVIEWERS = frozenset({"SIM-REVIEWER-AP-001"})


class SpyExecutor:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, action) -> None:
        self.calls.append(action)


def _vendor(**overrides) -> TrustedVendorRecord:
    fields = dict(vendor_id="V1", vendor_name="Acme", beneficiary_account_id="ACCT-OK", verification_status=VendorVerificationStatus.VERIFIED, vendor_status=VendorStatus.ACTIVE, approved_contact_email="ap@acme.example", approved_contact_verified=True)
    fields.update(overrides)
    return TrustedVendorRecord(**fields)


def _invoice(**overrides) -> TrustedInvoiceRecord:
    fields = dict(invoice_document_id="DOC-1", vendor_id="V1", invoice_number="INV-1", amount_minor_units=5000, currency="usd")
    fields.update(overrides)
    return TrustedInvoiceRecord(**fields)


def _observed(**overrides) -> ObservedEvidence:
    fields = dict(observed_invoice_document_id="DOC-1", observed_vendor_record_id="V1", observed_amount_minor_units=5000, observed_currency="usd")
    fields.update(overrides)
    return ObservedEvidence(**fields)


def _proposal(**overrides) -> ProposedPayment:
    fields = dict(invoice_document_id="DOC-1", invoice_number="INV-1", beneficiary_account_id="ACCT-OK", amount_minor_units=5000, currency="usd", evidence_invoice_document_id="DOC-1", evidence_vendor_record_id="V1")
    fields.update(overrides)
    return ProposedPayment(**fields)


def _run(**kw):
    defaults = dict(
        workflow_id="wf", starting_alarm_state=RoutingAlarmState.CLEAR, proposal=None, non_payment_action=None,
        trusted_vendor=_vendor(), trusted_invoice=_invoice(), observed=_observed(), pending_case=None, resolution=None,
        evidence_sources=[], source_account_id="COMPANY-ACCT", reason="test", executor=SpyExecutor(),
    )
    defaults.update(kw)
    executor = defaults["executor"]
    record = route_and_validate_payment(**defaults)
    return record, executor


# ---------------------------------------------------------------------
# 1. Agent 1 valid payment
# ---------------------------------------------------------------------


def test_scenario_1_agent_1_valid_payment():
    record, executor = _run(routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_1, reason="below threshold"), proposal=_proposal())
    assert record.final_action is FinalActionType.PROPOSE_PAYMENT_EXECUTED
    assert record.executed is True
    assert record.resolution_owner is ResolutionOwner.AGENT_1
    assert record.candidate_constructed_by == "validator"
    assert len(executor.calls) == 1
    assert executor.calls[0].beneficiary_account_id == "ACCT-OK"


# ---------------------------------------------------------------------
# 2. Beneficiary mismatch with no alarm
# ---------------------------------------------------------------------


def test_scenario_2_beneficiary_mismatch_no_alarm():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_1, reason="below threshold"),
        proposal=_proposal(beneficiary_account_id="ACCT-ATTACKER"),
    )
    assert record.executed is False
    assert len(executor.calls) == 0
    assert record.candidate_constructed is False
    assert record.validator_outcome is ValidationOutcome.BENEFICIARY_MISMATCH
    assert record.alarm_state_before is RoutingAlarmState.CLEAR
    assert record.alarm_state_after is RoutingAlarmState.CLEAR  # Tell never fired


# ---------------------------------------------------------------------
# 3. Tell alarm resolved as benign by Agent S
# ---------------------------------------------------------------------


def test_scenario_3_benign_alarm_resolved_by_agent_s():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, probe_score=0.9, threshold=0.5, reason="above threshold"),
        proposal=_proposal(),
        resolution=AutonomousResolutionInput(injection_contradicted_by_trusted_vendor_record=True),
        evidence_sources=["trusted_vendor_record"],
    )
    assert record.alarm_state_before is RoutingAlarmState.CLEAR
    assert record.alarm_state_after is RoutingAlarmState.RESOLVED
    assert (RoutingAlarmState.RAISED, RoutingAlarmState.RESOLVING) in record.alarm_transitions or any(a == RoutingAlarmState.RAISED for a, b in record.alarm_transitions)
    assert record.executed is True
    assert record.resolution_owner is ResolutionOwner.AGENT_S
    assert len(executor.calls) == 1


def test_scenario_3_human_path_variant_benign_approved_by_reviewer():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        starting_alarm_state=RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
        proposal=_proposal(),
        resolution=HumanReviewerResolutionInput(
            actor_type=ReviewerActorType.HUMAN_REVIEWER, reviewer_id="SIM-REVIEWER-AP-001", registered_reviewer_ids=REGISTERED_REVIEWERS,
            decision=ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT, supporting_trusted_evidence=["V1"],
        ),
        evidence_sources=["trusted_dispute_case_status"],
    )
    assert record.alarm_state_after is RoutingAlarmState.RESOLVED
    assert record.executed is True
    assert record.resolution_owner is ResolutionOwner.AGENT_S
    assert len(executor.calls) == 1


# ---------------------------------------------------------------------
# 4. Genuine attack contained by Agent S
# ---------------------------------------------------------------------


def test_scenario_4_genuine_attack_contained_no_payment():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, probe_score=0.95, threshold=0.5, reason="above threshold"),
        proposal=None,
        resolution=EscalateEvidenceReportInput(),
        evidence_sources=["trusted_dispute_case_status"],
    )
    assert record.final_action is FinalActionType.SUBMIT_EVIDENCE_REPORT
    assert record.executed is False
    assert len(executor.calls) == 0
    assert record.candidate_constructed is False
    assert record.alarm_state_after is RoutingAlarmState.AWAITING_HUMAN_VERIFICATION


def test_scenario_4_variant_confirmed_threat_hard_blocked():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        starting_alarm_state=RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
        proposal=None,
        resolution=HumanReviewerResolutionInput(
            actor_type=ReviewerActorType.HUMAN_REVIEWER, reviewer_id="SIM-REVIEWER-AP-001", registered_reviewer_ids=REGISTERED_REVIEWERS,
            decision=ReviewerDecisionType.REJECTED_AS_THREAT, supporting_trusted_evidence=[],
        ),
    )
    assert record.final_action is FinalActionType.FAIL_CLOSED
    assert record.alarm_state_after is RoutingAlarmState.HARD_BLOCKED
    assert record.resolution_owner is ResolutionOwner.HUMAN_REVIEWER
    assert not record.executed and len(executor.calls) == 0


# ---------------------------------------------------------------------
# 5. Vendor clarification
# ---------------------------------------------------------------------


def test_scenario_5_vendor_clarification_normal_gap_with_approved_contact():
    incomplete_invoice = _invoice(invoice_number=None, amount_minor_units=None, currency=None)
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_1, reason="below threshold"),
        proposal=_proposal(), trusted_invoice=incomplete_invoice,
    )
    assert record.final_action is FinalActionType.REQUEST_VENDOR_CLARIFICATION
    assert record.validator_outcome is ValidationOutcome.INVOICE_INCOMPLETE
    assert record.candidate_constructed is False
    assert not record.executed and len(executor.calls) == 0


# ---------------------------------------------------------------------
# 6. Human evidence report
# ---------------------------------------------------------------------


def test_scenario_6_human_evidence_report_on_conflicting_evidence():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_1, reason="below threshold"),
        proposal=_proposal(amount_minor_units=99999),  # fabricated evidence vs. observed
    )
    assert record.final_action is FinalActionType.SUBMIT_EVIDENCE_REPORT
    assert record.validator_outcome is ValidationOutcome.FABRICATED_EVIDENCE
    assert not record.executed and len(executor.calls) == 0


def test_scenario_6_variant_no_approved_contact_becomes_evidence_report():
    no_contact_vendor = _vendor(approved_contact_email=None, approved_contact_verified=False)
    incomplete_invoice = _invoice(invoice_number=None, amount_minor_units=None, currency=None)
    record, _ = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_1, reason="below threshold"),
        proposal=_proposal(), trusted_vendor=no_contact_vendor, trusted_invoice=incomplete_invoice,
    )
    assert record.final_action is FinalActionType.SUBMIT_EVIDENCE_REPORT


# ---------------------------------------------------------------------
# 7. Deterministic hard block
# ---------------------------------------------------------------------


def test_scenario_7_valid_candidate_still_blocked_while_unresolved():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        proposal=_proposal(), resolution=None,  # Agent S never resolves it -- stuck at RESOLVING
    )
    assert record.candidate_constructed is True  # the validator says it's fine
    assert record.gate_decision is GateDecisionType.BLOCK  # the gate still blocks it
    assert not record.executed and len(executor.calls) == 0
    assert record.alarm_state_after is RoutingAlarmState.RESOLVING


def test_scenario_7_hard_blocked_state_cannot_be_overridden_by_a_new_proposal():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        starting_alarm_state=RoutingAlarmState.HARD_BLOCKED, proposal=_proposal(),
    )
    assert record.gate_decision is GateDecisionType.BLOCK
    assert not record.executed and len(executor.calls) == 0


# ---------------------------------------------------------------------
# Attempted Agent-S candidate construction / alarm clearing (must be
# impossible through this module's public API)
# ---------------------------------------------------------------------


def test_agent_s_routing_decision_has_no_way_to_construct_a_candidate():
    import inspect

    sig = inspect.signature(RoutingDecision)
    # RoutingDecision (the thing Agent S's routing is expressed as) carries
    # no candidate-shaped field at all.
    assert "candidate" not in RoutingDecision.model_fields


def test_agent_s_cannot_clear_alarm_without_a_registered_human_reviewer():
    with pytest.raises(UnauthorizedResolutionError):
        _run(
            routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
            starting_alarm_state=RoutingAlarmState.AWAITING_HUMAN_VERIFICATION,
            proposal=_proposal(),
            resolution=HumanReviewerResolutionInput(
                actor_type=ReviewerActorType.AGENT_S, reviewer_id="SIM-REVIEWER-AP-001", registered_reviewer_ids=REGISTERED_REVIEWERS,
                decision=ReviewerDecisionType.APPROVED_CANONICAL_PAYMENT, supporting_trusted_evidence=["V1"],
            ),
        )


def test_no_resolution_input_can_directly_request_resolved_state():
    # The tagged union has exactly four kinds, none of which is "just set
    # alarm_state=resolved" -- every path to RESOLVED goes through
    # tell.safety.resolution's actor/evidence checks.
    import tell.agent.routing_orchestrator as mod

    kinds = {"autonomous", "human_reviewer", "escalate_vendor_clarification", "escalate_evidence_report"}
    actual = {
        mod.AutonomousResolutionInput.model_fields["kind"].default,
        mod.HumanReviewerResolutionInput.model_fields["kind"].default,
        mod.EscalateVendorClarificationInput.model_fields["kind"].default,
        mod.EscalateEvidenceReportInput.model_fields["kind"].default,
    }
    assert actual == kinds


# ---------------------------------------------------------------------
# Provenance propagation / pending case / closed alarm transitions
# ---------------------------------------------------------------------


def test_pending_case_blocks_even_with_agent_s_routing():
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        proposal=_proposal(), resolution=AutonomousResolutionInput(injection_contradicted_by_trusted_vendor_record=True),
        pending_case=PendingCaseKind.AWAITING_VENDOR_CLARIFICATION,
    )
    assert record.validator_outcome is ValidationOutcome.UNRESOLVED_CASE
    assert not record.executed and len(executor.calls) == 0


def test_evidence_sources_propagate_into_audit_record():
    record, _ = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        proposal=None, resolution=EscalateVendorClarificationInput(),
        evidence_sources=["trusted_invoice_payment_history", "trusted_dispute_case_status"],
    )
    assert record.evidence_sources == ["trusted_invoice_payment_history", "trusted_dispute_case_status"]


def test_alarm_transitions_recorded_in_audit_record():
    record, _ = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        proposal=None, resolution=EscalateVendorClarificationInput(),
    )
    froms = [f for f, _ in record.alarm_transitions]
    tos = [t for _, t in record.alarm_transitions]
    assert RoutingAlarmState.CLEAR in froms
    assert RoutingAlarmState.AWAITING_VENDOR_CLARIFICATION in tos


def test_vendor_clarification_case_can_then_resume_and_pay():
    # Agent S escalates to vendor clarification, the vendor replies
    # satisfying the resume condition (modeled here by starting a second
    # call at RESOLVING, matching alarm.VALID_TRANSITIONS'
    # awaiting_vendor_clarification -> resolving edge), then autonomous
    # resolution completes it.
    record, executor = _run(
        routing_decision=RoutingDecision(routed_agent=RoutedAgent.AGENT_S, reason="above threshold"),
        starting_alarm_state=RoutingAlarmState.RESOLVING,
        proposal=_proposal(), resolution=AutonomousResolutionInput(injection_contradicted_by_trusted_vendor_record=True),
    )
    assert record.executed is True


# ---------------------------------------------------------------------
# No real side effects anywhere in this module
# ---------------------------------------------------------------------


def test_no_sqlite_email_or_network_reference_in_orchestrator():
    import inspect

    import tell.agent.routing_orchestrator as mod

    source = inspect.getsource(mod)
    for banned in ("sqlite3", "smtplib", "requests", "socket", "Ledger("):
        assert banned not in source
