"""CPU-only, in-memory tests for the three-zone router and Tell Verify
(tell.routing.typed_verifier). No model, GPU, database, or real payment
anywhere in this file -- every fixture is a plain in-memory typed object.
"""

from __future__ import annotations

import inspect
import json
import math

import pytest

from tell.agent.routing_orchestrator import route_and_validate_payment, RoutingDecision, NonPaymentAction
from tell.routing.records import RoutedAgent
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
from tell.routing.typed_verifier import (
    DisputeCaseLookup,
    EvidenceStatus,
    LocalEvidenceAdjudicator,
    MemoryProvenanceStatus,
    ObservationSourceType,
    PaymentHistoryLookup,
    ThreeZoneConfig,
    ThreeZoneConfigError,
    VerificationContext,
    VerificationDecision,
    VerifyReasonCode,
    VerifyRoute,
    ZoneRoute,
    load_three_zone_config,
    route_by_score,
)

CONFIG_PATH = "configs/routing/tell_three_zone_v1.json"
LOW = 0.1708046793937683
HIGH = 0.5134634443863925


def _config() -> ThreeZoneConfig:
    return load_three_zone_config(CONFIG_PATH)


def _vendor(**overrides) -> TrustedVendorRecord:
    fields = dict(vendor_id="V1", vendor_name="Acme", beneficiary_account_id="ACCT-OK",
                 verification_status=VendorVerificationStatus.VERIFIED, vendor_status=VendorStatus.ACTIVE,
                 approved_contact_email="ap@acme.example", approved_contact_verified=True)
    fields.update(overrides)
    return TrustedVendorRecord(**fields)


def _invoice(**overrides) -> TrustedInvoiceRecord:
    fields = dict(invoice_document_id="DOC-1", vendor_id="V1", invoice_number="INV-1", amount_minor_units=5000, currency="usd")
    fields.update(overrides)
    return TrustedInvoiceRecord(**fields)


def _history(**overrides) -> PaymentHistoryLookup:
    fields = dict(lookup_succeeded=True)
    fields.update(overrides)
    return PaymentHistoryLookup(**fields)


def _dispute(**overrides) -> DisputeCaseLookup:
    fields = dict(lookup_succeeded=True)
    fields.update(overrides)
    return DisputeCaseLookup(**fields)


def _ctx(**overrides) -> VerificationContext:
    fields = dict(
        tell_score=0.3, observation_source_type=ObservationSourceType.INVOICE, observation_provenance="test-fixture",
        scenario_id="scn-1", canonical_vendor_id="V1", observed_vendor_id="V1",
        canonical_invoice_id="DOC-1", observed_invoice_id="DOC-1",
        trusted_vendor=_vendor(), trusted_invoice=_invoice(),
        observed=ObservedEvidence(observed_invoice_document_id="DOC-1", observed_vendor_record_id="V1",
                                  observed_amount_minor_units=5000, observed_currency="usd"),
        trusted_beneficiary_account_id="ACCT-OK", observed_beneficiary_account_id="ACCT-OK",
        trusted_amount_minor_units=5000, observed_amount_minor_units=5000, trusted_currency="usd", observed_currency="usd",
        payment_history=_history(), dispute_case=_dispute(), memory_provenance=MemoryProvenanceStatus.NO_MEMORY_CONSULTED,
        approved_vendor_contact_available=True,
    )
    fields.update(overrides)
    return VerificationContext(**fields)


# --- 1-7: three-zone score routing ------------------------------------------------


def test_score_just_below_low_routes_agent1():
    assert route_by_score(LOW - 1e-9, _config()) is ZoneRoute.AGENT_1


def test_score_exactly_low_enters_tell_verify():
    assert route_by_score(LOW, _config()) is ZoneRoute.TELL_VERIFY


def test_score_inside_middle_band_enters_tell_verify():
    assert route_by_score((LOW + HIGH) / 2, _config()) is ZoneRoute.TELL_VERIFY


def test_score_just_below_high_enters_tell_verify():
    assert route_by_score(HIGH - 1e-9, _config()) is ZoneRoute.TELL_VERIFY


def test_score_exactly_high_routes_agent_s():
    assert route_by_score(HIGH, _config()) is ZoneRoute.AGENT_S


def test_score_above_high_routes_agent_s():
    assert route_by_score(0.99, _config()) is ZoneRoute.AGENT_S


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_invalid_score_fails_closed(bad):
    assert route_by_score(bad, _config()) is ZoneRoute.FAIL_CLOSED


# --- config validity ---------------------------------------------------------------


def test_config_loads_and_matches_frozen_thresholds():
    cfg = _config()
    assert cfg.low_threshold == pytest.approx(LOW)
    assert cfg.high_threshold == pytest.approx(HIGH)


def test_config_rejects_low_not_less_than_high(tmp_path):
    raw = json.loads(open(CONFIG_PATH).read())
    raw["zones"]["low_threshold"], raw["zones"]["high_threshold"] = 0.6, 0.5
    raw["primary_threshold_artifact"]["value"] = 0.6
    raw["operational_threshold_artifact"]["value"] = 0.5
    p = tmp_path / "bad.json"; p.write_text(json.dumps(raw))
    with pytest.raises(ThreeZoneConfigError):
        load_three_zone_config(p)


def test_config_rejects_threshold_outside_unit_interval(tmp_path):
    raw = json.loads(open(CONFIG_PATH).read())
    raw["zones"]["high_threshold"] = 1.5
    raw["operational_threshold_artifact"]["value"] = 1.5
    p = tmp_path / "bad.json"; p.write_text(json.dumps(raw))
    with pytest.raises(ThreeZoneConfigError):
        load_three_zone_config(p)


def test_config_rejects_missing_provenance(tmp_path):
    raw = json.loads(open(CONFIG_PATH).read())
    raw["provenance"]["derived_from"] = []
    p = tmp_path / "bad.json"; p.write_text(json.dumps(raw))
    with pytest.raises(ThreeZoneConfigError):
        load_three_zone_config(p)


def test_config_rejects_probe_hash_mismatch():
    with pytest.raises(ThreeZoneConfigError):
        load_three_zone_config(CONFIG_PATH, expected_probe_weights_sha256="0" * 64)


# --- 8-16: deterministic verification policy ----------------------------------------


def test_complete_consistent_evidence_returns_agent1():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx())
    assert d.route is VerifyRoute.CONTINUE_AGENT_1
    assert d.evidence_status is EvidenceStatus.COMPLETE
    assert VerifyReasonCode.EVIDENCE_COMPLETE_AND_CONSISTENT in d.reason_codes


def test_beneficiary_conflict_escalates_agent_s():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(observed_beneficiary_account_id="ACCT-ATTACKER"))
    assert d.route is VerifyRoute.ESCALATE_AGENT_S
    assert VerifyReasonCode.BENEFICIARY_CONFLICT in d.reason_codes


def test_vendor_identity_conflict_escalates_agent_s():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(observed_vendor_id="V-DIFFERENT"))
    assert d.route is VerifyRoute.ESCALATE_AGENT_S
    assert VerifyReasonCode.VENDOR_IDENTITY_CONFLICT in d.reason_codes


def test_invoice_identity_conflict_escalates_agent_s():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(observed_invoice_id="DOC-DIFFERENT"))
    assert d.route is VerifyRoute.ESCALATE_AGENT_S
    assert VerifyReasonCode.INVOICE_IDENTITY_CONFLICT in d.reason_codes


def test_duplicate_payment_escalates_agent_s():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(payment_history=_history(duplicate_payment_detected=True)))
    assert d.route is VerifyRoute.ESCALATE_AGENT_S
    assert VerifyReasonCode.DUPLICATE_PAYMENT_DETECTED in d.reason_codes


def test_active_dispute_escalates_agent_s():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(dispute_case=_dispute(active_dispute=True)))
    assert d.route is VerifyRoute.ESCALATE_AGENT_S
    assert VerifyReasonCode.ACTIVE_DISPUTE in d.reason_codes


def test_untrusted_conflicting_memory_escalates_agent_s():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(memory_provenance=MemoryProvenanceStatus.CONFLICTS_WITH_TRUSTED_RECORD))
    assert d.route is VerifyRoute.ESCALATE_AGENT_S
    assert VerifyReasonCode.MEMORY_CONFLICTS_WITH_TRUSTED_RECORD in d.reason_codes


def test_ordinary_missing_information_with_contact_requests_clarification():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(trusted_invoice=_invoice(invoice_number=None)))
    assert d.route is VerifyRoute.REQUEST_VENDOR_CLARIFICATION
    assert d.evidence_status is EvidenceStatus.INCOMPLETE


def test_lookup_failure_produces_evidence_report():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(dispute_case=_dispute(lookup_succeeded=False)))
    assert d.route is VerifyRoute.SUBMIT_EVIDENCE_REPORT
    assert d.evidence_status is EvidenceStatus.LOOKUP_FAILED
    assert VerifyReasonCode.LOOKUP_FAILED in d.reason_codes


def test_missing_evidence_without_contact_produces_evidence_report():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(trusted_invoice=_invoice(invoice_number=None), approved_vendor_contact_available=False))
    assert d.route is VerifyRoute.SUBMIT_EVIDENCE_REPORT
    assert VerifyReasonCode.NO_APPROVED_CONTACT_FOR_CLARIFICATION in d.reason_codes


def test_malformed_trusted_vendor_record_produces_evidence_report():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(trusted_vendor=None))
    assert d.route is VerifyRoute.SUBMIT_EVIDENCE_REPORT


def test_invalid_score_inside_context_fails_closed_to_evidence_report():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(tell_score=float("nan")))
    assert d.route is VerifyRoute.SUBMIT_EVIDENCE_REPORT
    assert VerifyReasonCode.INVALID_TELL_SCORE in d.reason_codes


# --- 17-18: ownership boundaries (static + behavioral proof) ------------------------


def test_verifier_cannot_construct_candidate_or_clear_alarm():
    import ast

    import tell.routing.typed_verifier as mod

    tree = ast.parse(inspect.getsource(mod))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.add(node.module or "")
            imported_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(a.name for a in node.names)
    # No import of the validator, the gate, the ledger, or the resolution/
    # alarm-transition machinery -- so there is no code path by which this
    # module could construct a ValidatedPayInvoiceCandidate or move a
    # RoutingAlarmState to RESOLVED/HARD_BLOCKED.
    for forbidden in ("validate_payment_proposal", "evaluate_gate", "ValidatedPayInvoiceCandidate",
                     "resolve_from_human_decision", "resolve_autonomous_injection", "apply_transition",
                     "tell.payment.ledger", "tell.safety.gate", "tell.safety.resolution", "tell.safety.alarm"):
        assert forbidden not in imported_names, f"typed_verifier.py must never import {forbidden!r}"
    assert not hasattr(LocalEvidenceAdjudicator, "construct_candidate")
    assert not hasattr(LocalEvidenceAdjudicator, "clear_alarm")
    assert not hasattr(VerificationDecision, "to_pay_invoice_candidate")


def test_verification_decision_side_effect_authorized_always_false():
    d = LocalEvidenceAdjudicator().adjudicate(_ctx())
    assert d.side_effect_authorized is False
    with pytest.raises(Exception):
        VerificationDecision(route=VerifyRoute.CONTINUE_AGENT_1, evidence_status=EvidenceStatus.COMPLETE,
                            reason_codes=(), checked_trusted_sources=(), source_provenance="x", tell_score=0.1,
                            score_band=ZoneRoute.AGENT_1, evidence_fields_checked=0, scenario_id="s", side_effect_authorized=True)


def test_context_rejects_label_like_fields():
    # extra="forbid" means any ground-truth/label-like field the caller
    # tries to smuggle in raises, rather than silently being accepted.
    with pytest.raises(Exception):
        VerificationContext(**{**_ctx().model_dump(), "ground_truth_class": "malicious"})
    with pytest.raises(Exception):
        VerificationContext(**{**_ctx().model_dump(), "attack_surface": "email_injection"})


# --- 19-21: middle-band clearance still passes through validator + gate ------------


def _run_pipeline(routed_agent: RoutedAgent, *, proposal=None, non_payment_action=None, starting_alarm=RoutingAlarmState.CLEAR, **kw):
    class SpyExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, action):
            self.calls.append(action)

    executor = SpyExecutor()
    record = route_and_validate_payment(
        workflow_id="wf-1", routing_decision=RoutingDecision(routed_agent=routed_agent),
        starting_alarm_state=starting_alarm, proposal=proposal, non_payment_action=non_payment_action,
        trusted_vendor=kw.get("trusted_vendor", _vendor()), trusted_invoice=kw.get("trusted_invoice", _invoice()),
        observed=kw.get("observed", ObservedEvidence(observed_invoice_document_id="DOC-1", observed_vendor_record_id="V1",
                                                     observed_amount_minor_units=5000, observed_currency="usd")),
        pending_case=kw.get("pending_case"), resolution=kw.get("resolution"), evidence_sources=["tell_verify"],
        source_account_id="SIM-SOURCE", reason="tell-verify-cleared", executor=executor,
    )
    return record, executor


def test_verifier_cleared_middle_band_case_still_passes_through_validator():
    """Tell Verify returning CONTINUE_AGENT_1 in the middle band does not
    skip validate_payment_proposal -- a bad beneficiary is still caught."""
    d = LocalEvidenceAdjudicator().adjudicate(_ctx(tell_score=0.3))
    assert d.route is VerifyRoute.CONTINUE_AGENT_1  # verifier says "fine"

    bad_proposal = ProposedPayment(invoice_document_id="DOC-1", invoice_number="INV-1", beneficiary_account_id="ACCT-ATTACKER",
                                   amount_minor_units=5000, currency="usd", evidence_invoice_document_id="DOC-1", evidence_vendor_record_id="V1")
    record, executor = _run_pipeline(RoutedAgent.AGENT_1, proposal=bad_proposal)
    assert record.validator_outcome is ValidationOutcome.BENEFICIARY_MISMATCH
    assert not record.executed
    assert executor.calls == []


def test_validator_approved_candidate_still_passes_through_gate():
    good_proposal = ProposedPayment(invoice_document_id="DOC-1", invoice_number="INV-1", beneficiary_account_id="ACCT-OK",
                                    amount_minor_units=5000, currency="usd", evidence_invoice_document_id="DOC-1", evidence_vendor_record_id="V1")
    record, executor = _run_pipeline(RoutedAgent.AGENT_1, proposal=good_proposal, starting_alarm=RoutingAlarmState.CLEAR)
    assert record.validator_outcome is ValidationOutcome.VALID
    assert record.gate_decision is GateDecisionType.PERMIT
    assert record.executed
    assert len(executor.calls) == 1


def test_unresolved_alarm_cannot_reach_executor():
    good_proposal = ProposedPayment(invoice_document_id="DOC-1", invoice_number="INV-1", beneficiary_account_id="ACCT-OK",
                                    amount_minor_units=5000, currency="usd", evidence_invoice_document_id="DOC-1", evidence_vendor_record_id="V1")
    # Escalated to Agent S (as Tell Verify would recommend on conflict) but
    # never resolved -- the gate must still block execution.
    record, executor = _run_pipeline(RoutedAgent.AGENT_S, proposal=good_proposal, resolution=None)
    assert not record.executed
    assert executor.calls == []
    assert record.gate_decision is not GateDecisionType.PERMIT


# --- 22: existing low/high routing behavior remains compatible ---------------------


def test_low_and_high_zone_routes_are_compatible_with_existing_routed_agent_enum():
    cfg = _config()
    assert route_by_score(0.01, cfg) is ZoneRoute.AGENT_1
    assert route_by_score(0.9, cfg) is ZoneRoute.AGENT_S
    # ZoneRoute.AGENT_1 / AGENT_S values match tell.routing.records.RoutedAgent's
    # values exactly, so a caller can map one to the other without a lookup table.
    assert ZoneRoute.AGENT_1.value == RoutedAgent.AGENT_1.value
    assert ZoneRoute.AGENT_S.value == RoutedAgent.AGENT_S.value
