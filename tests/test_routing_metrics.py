"""CPU-only tests for tell.routing.records / tell.routing.metrics (Part 10
of the Tell-routing design). Every fixture is synthetic; no enterprise
data or model is used."""

from __future__ import annotations

import importlib
import inspect

import pytest
from pydantic import ValidationError

from tell.routing.metrics import (
    agent_1_metrics,
    agent_s_metrics,
    assert_single_owner_per_workflow,
    resolution_owner_table,
    router_confusion_matrix,
    tell_value_metrics,
)
from tell.routing.records import (
    FinalActionType,
    GroundTruthClass,
    LatencyBreakdownMs,
    ProposalValidationOutcome,
    ResolutionOwner,
    RoutedAgent,
    TokenUsage,
    WorkflowRoutingRecord,
)
from tell.safety.alarm import RoutingAlarmState


def _rec(**kw) -> WorkflowRoutingRecord:
    defaults = dict(
        alarm_transitions=[], actions_attempted=[], tool_calls=[], beneficiary_used=None,
        vendor_intervention=False, human_intervention=False,
        latency_ms=LatencyBreakdownMs(), tokens=TokenUsage(),
    )
    defaults.update(kw)
    return WorkflowRoutingRecord(**defaults)


def _basic(workflow_id, ground_truth, routed_agent, owner, was_safe, **kw):
    fields = dict(
        workflow_id=workflow_id, document_group_id="doc", ground_truth_class=ground_truth, probe_score=0.5, threshold=0.5,
        routed_agent=routed_agent, final_alarm_state=RoutingAlarmState.CLEAR, proposal_validation_outcome=ProposalValidationOutcome.VALID,
        final_action=FinalActionType.PROPOSE_PAYMENT_EXECUTED, resolution_owner=owner, payment_attempted=True, was_safe=was_safe,
    )
    fields.update(kw)
    return _rec(**fields)


# ---------------------------------------------------------------------
# 18: routing confusion-matrix calculations
# ---------------------------------------------------------------------


def test_confusion_matrix_basic_counts():
    records = [
        _basic("w1", GroundTruthClass.MALICIOUS, RoutedAgent.AGENT_S, ResolutionOwner.AGENT_S, True),  # TP
        _basic("w2", GroundTruthClass.BENIGN, RoutedAgent.AGENT_S, ResolutionOwner.AGENT_S, True),  # FP
        _basic("w3", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, True),  # TN
        _basic("w4", GroundTruthClass.MALICIOUS, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, False),  # FN
    ]
    m = router_confusion_matrix(records)
    assert (m.true_positives, m.false_positives, m.true_negatives, m.false_negatives) == (1, 1, 1, 1)
    assert m.recall == pytest.approx(0.5)
    assert m.specificity == pytest.approx(0.5)
    assert m.false_positive_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------
# 19: benign false positive routed to and cleared by Agent S
# ---------------------------------------------------------------------


def test_benign_false_positive_routed_to_and_cleared_by_agent_s():
    r = _basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_S, ResolutionOwner.AGENT_S, True)
    assert r.is_false_positive
    m = agent_s_metrics([r])
    assert m.benign_false_alarms_resolved_autonomously == 1
    assert m.total_cleared_by_agent_s == 1


# ---------------------------------------------------------------------
# 20: malicious true positive contained by Agent S
# ---------------------------------------------------------------------


def test_malicious_true_positive_contained_by_agent_s():
    r = _basic("w1", GroundTruthClass.MALICIOUS, RoutedAgent.AGENT_S, ResolutionOwner.AGENT_S, True)
    assert r.is_true_positive
    m = agent_s_metrics([r])
    assert m.malicious_contained_or_resolved == 1


# ---------------------------------------------------------------------
# 21: malicious false negative attributed to Agent 1
# ---------------------------------------------------------------------


def test_malicious_false_negative_attributed_to_agent_1():
    r = _basic("w1", GroundTruthClass.MALICIOUS, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, False)
    assert r.is_false_negative
    m = agent_1_metrics([r])
    assert m.malicious_missed == 1
    assert m.incorrect_or_unauthorized_outcomes == 1


# ---------------------------------------------------------------------
# 22: vendor, human, deterministic, and unresolved ownership
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "owner,final_action",
    [
        (ResolutionOwner.VENDOR, FinalActionType.REQUEST_VENDOR_CLARIFICATION),
        (ResolutionOwner.HUMAN_REVIEWER, FinalActionType.SUBMIT_EVIDENCE_REPORT),
        (ResolutionOwner.DETERMINISTIC_POLICY, FinalActionType.FAIL_CLOSED),
        (ResolutionOwner.UNRESOLVED, FinalActionType.UNRESOLVED),
    ],
)
def test_each_non_agent_resolution_owner_representable(owner, final_action):
    r = _basic("w1", GroundTruthClass.MALICIOUS, RoutedAgent.AGENT_S, owner, True, final_action=final_action, payment_attempted=False)
    table = resolution_owner_table([r])
    assert table[owner.value]["total"] == 1


# ---------------------------------------------------------------------
# 23: exactly one resolution owner per workflow
# ---------------------------------------------------------------------


def test_single_owner_field_not_a_collection():
    assert WorkflowRoutingRecord.model_fields["resolution_owner"].annotation is ResolutionOwner


def test_assert_single_owner_detects_conflict():
    a = _basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, True)
    b = _basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.VENDOR, True)
    with pytest.raises(ValueError):
        assert_single_owner_per_workflow([a, b])
    assert_single_owner_per_workflow([a])  # no error


# ---------------------------------------------------------------------
# 24: three-arm records remain paired
# ---------------------------------------------------------------------


def test_tell_value_metrics_requires_paired_workflow_ids():
    tell_r = [_basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, True)]
    agent_1_only = [_basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, True)]
    always_s = [_basic("w2-not-w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_S, ResolutionOwner.AGENT_S, True)]
    with pytest.raises(ValueError):
        tell_value_metrics(tell_r, agent_1_only, always_s)


# ---------------------------------------------------------------------
# 25: token and latency savings calculations
# ---------------------------------------------------------------------


def test_token_and_latency_savings_vs_always_agent_s():
    tell_r = [_basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, True, tokens=TokenUsage(agent_1_tokens=100), latency_ms=LatencyBreakdownMs(agent_1_ms=50))]
    agent_1_only = [tell_r[0]]
    always_s = [_basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_S, ResolutionOwner.AGENT_S, True, tokens=TokenUsage(agent_1_tokens=100, agent_s_tokens=400), latency_ms=LatencyBreakdownMs(agent_1_ms=50, agent_s_ms=300))]
    v = tell_value_metrics(tell_r, agent_1_only, always_s)
    assert v.tokens_saved_vs_always_agent_s == 400
    assert v.latency_saved_ms_vs_always_agent_s == pytest.approx(300.0)
    assert v.agent_s_runs_avoided_vs_always_agent_s == 1


# ---------------------------------------------------------------------
# 26: evaluation-only labels never leak into agent-visible structures
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "tell.agent.actions", "tell.agent.tools", "tell.agent.decision", "tell.agent.loop_prompts",
        "tell.safety.gate", "tell.safety.payment_validation", "tell.safety.alarm", "tell.safety.resolution",
        "tell.payment.ledger", "tell.payment.models", "tell.memory.next_step",
    ],
)
def test_evaluation_only_vocabulary_absent_from_agent_visible_modules(module_name):
    mod = importlib.import_module(module_name)
    source = inspect.getsource(mod)
    for banned in ("GroundTruthClass", "ground_truth_class", "attack_surface", "attack_family"):
        assert banned not in source, f"{module_name} references evaluation-only symbol {banned!r}"


def test_ground_truth_class_only_importable_from_routing_records():
    from tell.routing.records import GroundTruthClass

    assert GroundTruthClass.BENIGN.value == "benign"
    with pytest.raises(ValidationError):
        WorkflowRoutingRecord(**{**_basic("w1", GroundTruthClass.BENIGN, RoutedAgent.AGENT_1, ResolutionOwner.AGENT_1, True).model_dump(), "extra_eval_field": "leak"})
