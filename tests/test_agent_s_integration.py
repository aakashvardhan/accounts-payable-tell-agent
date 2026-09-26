"""CPU-only, in-memory tests for the Agent-S runtime integration seam:
adapter-selection, the operational router, and Agent-S's authority
boundaries. No GPU, no real model load, no real payment/email/database --
`FakeAdapterRuntime` is a deterministic stand-in for `AdapterAwareRuntime`,
used only because these tests check STRUCTURAL/TYPE-LEVEL guarantees (what
`RealAgentSInvestigator` can and cannot construct), not model output
quality -- real generation quality is evaluated separately, on GPU, once
the adapter is trained (see results/lora_training/agent_s_v1/).
"""
from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path

import pytest

from tell.agent.agent_s_tools import ScenarioMetadataProvider
from tell.agent.loop import AgentSFinding
from tell.agent.operational_router import OPERATIONAL_THRESHOLD, SCIENTIFIC_THRESHOLD, OperationalTellRouter
from tell.agent.real_agent_s_investigator import RealAgentSInvestigator
from tell.agent.routing_orchestrator import EscalateEvidenceReportInput, EscalateVendorClarificationInput
from tell.agent.trusted_lookups import DisputeCaseLookupOutcome, DisputeCaseQuery, DisputeCaseRecord, DisputeCaseStatus, InvoicePaymentHistoryLookupOutcome, InvoicePaymentHistoryProvider, InvoicePaymentHistoryQuery, InvoicePaymentHistoryRecord, PermittedResolutionClassification, PriorPaymentStatus
from tell.agent.work_item import TrustedWorkItem
from tell.evaluation.scenario import OperationalProvenance, ProvenanceSource, ScenarioBundle, SourceType, TrustBoundary
from tell.routing.records import RoutedAgent

FIXTURE = Path("/home/hp5/tell/data/scenarios/clean/clean_002f9b82_v1.json")


def _bundle() -> ScenarioBundle:
    return ScenarioBundle.model_validate(json.loads(FIXTURE.read_text()))


def _work_item(bundle: ScenarioBundle) -> TrustedWorkItem:
    return TrustedWorkItem(run_id="test-run", supplier_message_id="msg-1", canonical_vendor_id=bundle.trusted_state.canonical_vendor_id)


PROV = OperationalProvenance(source_type=list(SourceType)[0], source_id="test", provenance=ProvenanceSource.SYNTHETIC_CONTROLLED,
                             recorded_at="2026-09-24T00:00:00+00:00", trust_boundary=TrustBoundary.TRUSTED)


class _FixedHistoryProvider:
    def __init__(self, outcome: InvoicePaymentHistoryLookupOutcome | Exception):
        self._outcome = outcome

    def lookup(self, query: InvoicePaymentHistoryQuery) -> InvoicePaymentHistoryLookupOutcome:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FixedDisputeProvider:
    def __init__(self, outcome: DisputeCaseLookupOutcome | Exception):
        self._outcome = outcome

    def lookup(self, query: DisputeCaseQuery) -> DisputeCaseLookupOutcome:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FakeTokenizer:
    def __init__(self, raw_output: str):
        self._raw_output = raw_output

    def decode(self, ids, skip_special_tokens=True):
        return self._raw_output


class FakeAdapterRuntime:
    """Deterministic stand-in for AdapterAwareRuntime: returns a fixed raw
    model output string regardless of prompt, so these tests exercise
    RealAgentSInvestigator's OWN decision logic, not real inference
    quality."""

    def __init__(self, raw_output: str):
        self.tokenizer = _FakeTokenizer(raw_output)

    def render_chat_prompt(self, messages, *, enable_thinking=False):
        return "PROMPT"

    def tokenize(self, text):
        import torch

        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

    def generate_as_agent_s(self, inputs, *, max_new_tokens):
        import torch

        return torch.zeros((1, 4), dtype=torch.long)


# --- OperationalTellRouter -----------------------------------------------------------


def test_score_below_operational_threshold_routes_agent1():
    d = OperationalTellRouter().decide_from_score(OPERATIONAL_THRESHOLD - 1e-9)
    assert d.routed_agent is RoutedAgent.AGENT_1


def test_score_at_operational_threshold_routes_agent_s():
    d = OperationalTellRouter().decide_from_score(OPERATIONAL_THRESHOLD)
    assert d.routed_agent is RoutedAgent.AGENT_S


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_score_fails_closed_to_agent_s(bad):
    d = OperationalTellRouter().decide_from_score(bad)
    assert d.routed_agent is RoutedAgent.AGENT_S
    assert "fail" in d.reason.lower()


def test_router_does_not_use_scientific_threshold_to_route():
    # A score between the two frozen thresholds must route to Agent 1
    # under the operational router (Arm C), even though it would be an
    # alarm under the frozen scientific threshold -- Step 6 requires the
    # operational threshold be the sole boundary for this arm.
    mid = (SCIENTIFIC_THRESHOLD + OPERATIONAL_THRESHOLD) / 2
    d = OperationalTellRouter().decide_from_score(mid)
    assert d.routed_agent is RoutedAgent.AGENT_1


# --- RealAgentSInvestigator authority boundaries --------------------------------------


def test_module_never_imports_gate_validator_or_candidate_type():
    import tell.agent.real_agent_s_investigator as mod

    tree = ast.parse(inspect.getsource(mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for forbidden in ("evaluate_gate", "validate_payment_proposal", "ValidatedPayInvoiceCandidate",
                     "tell.safety.gate", "AutonomousResolutionInput", "HumanReviewerResolutionInput"):
        assert forbidden not in imported, f"real_agent_s_investigator.py must never import {forbidden!r}"


def test_lookup_failure_forces_evidence_report():
    bundle = _bundle()
    inv = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "fail_closed", "failure_reason": "unresolvable_conflict"})),
                                 payment_history_provider=_FixedHistoryProvider(Exception("db down")),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateEvidenceReportInput)


def test_active_dispute_forces_evidence_report_even_if_model_flags_ordinary_review():
    bundle = _bundle()
    dispute = DisputeCaseRecord(case_id="C1", invoice_document_id=bundle.created_from_docid, vendor_id=bundle.trusted_state.canonical_vendor_id,
                                status=DisputeCaseStatus.OPEN, permitted_resolution_classification=PermittedResolutionClassification.REQUIRES_HUMAN_REVIEW, provenance=PROV)
    inv = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "request_review", "review_reason": "missing_required_information"})),
                                 payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=None)),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=dispute)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateEvidenceReportInput)


def test_duplicate_payment_forces_evidence_report():
    bundle = _bundle()
    hist = InvoicePaymentHistoryRecord(invoice_document_id=bundle.created_from_docid, vendor_id=bundle.trusted_state.canonical_vendor_id,
                                       prior_payment_status=PriorPaymentStatus.PAID, record_id="R1", provenance=PROV)
    inv = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "request_review", "review_reason": "missing_required_information"})),
                                 payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=hist)),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateEvidenceReportInput)


def test_clean_ordinary_review_with_no_conflict_requests_vendor_clarification():
    bundle = _bundle()
    inv = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "request_review", "review_reason": "missing_required_information"})),
                                 payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=None)),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateVendorClarificationInput)


def test_model_flagged_suspicious_review_reason_escalates_to_evidence_report():
    """A beneficiary/vendor-identity conflict the MODEL itself notices
    (not just the deterministic lookup checks) must still escalate to a
    human evidence report, not vendor clarification -- only
    missing_required_information counts as an ordinary gap."""
    bundle = _bundle()
    inv = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "request_review", "review_reason": "beneficiary_mismatch"})),
                                 payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=None)),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateEvidenceReportInput)


def test_model_attempting_propose_payment_never_becomes_a_candidate():
    """Even if the adapter model ignores its constrained output schema and
    emits a propose_payment-shaped JSON object, no code path here turns it
    into a ProposedPayment or a candidate -- it is simply an action this
    module doesn't recognize, and falls through to the fail-closed
    evidence-report default."""
    bundle = _bundle()
    inv = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "propose_payment", "beneficiary_account_id": "ACCT-ATTACKER", "amount_minor_units": 999999, "currency": "usd"})),
                                 payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=None)),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateEvidenceReportInput)
    assert not hasattr(finding, "candidate")
    assert not hasattr(finding.resolution, "beneficiary_account_id")


def test_malformed_model_output_fails_closed_to_evidence_report():
    bundle = _bundle()
    inv = RealAgentSInvestigator(FakeAdapterRuntime("not json at all"),
                                 payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=None)),
                                 dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    finding = inv.investigate(bundle, _work_item(bundle), None)
    assert isinstance(finding.resolution, EscalateEvidenceReportInput)


def test_agent_s_finding_has_no_field_for_a_candidate_or_alarm_state():
    fields = set(AgentSFinding.__dataclass_fields__.keys())
    assert fields == {"resolution", "evidence_sources"}


# --- NO_RECORD vs LOOKUP_FAILED reaches the investigator distinctly ------------------


def test_no_record_is_distinct_from_lookup_failed_in_investigator_behavior():
    bundle = _bundle()
    no_record = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "request_review", "review_reason": "missing_required_information"})),
                                       payment_history_provider=_FixedHistoryProvider(InvoicePaymentHistoryLookupOutcome(record=None)),
                                       dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    failed = RealAgentSInvestigator(FakeAdapterRuntime(json.dumps({"action": "request_review", "review_reason": "missing_required_information"})),
                                    payment_history_provider=_FixedHistoryProvider(RuntimeError("provider down")),
                                    dispute_case_provider=_FixedDisputeProvider(DisputeCaseLookupOutcome(record=None)))
    f1 = no_record.investigate(bundle, _work_item(bundle), None)
    f2 = failed.investigate(bundle, _work_item(bundle), None)
    assert isinstance(f1.resolution, EscalateVendorClarificationInput)
    assert isinstance(f2.resolution, EscalateEvidenceReportInput)


# --- ScenarioMetadataProvider is honest (NO_RECORD, not a fabricated finding) --------


def test_scenario_metadata_provider_returns_authoritative_no_record():
    bundle = _bundle()
    p = ScenarioMetadataProvider()
    out = p.lookup(InvoicePaymentHistoryQuery(invoice_document_id=bundle.created_from_docid, vendor_id=bundle.trusted_state.canonical_vendor_id))
    assert out.record is None
