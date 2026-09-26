"""Typed workflow-routing records (Part 10 of the Tell-routing design).

One `WorkflowRoutingRecord` per completed workflow, whether from a
synthetic fixture (used to validate the metric formulas in
tests/test_routing_metrics.py) or a future real three-arm trace file. This
module defines the schema only -- no metric computation lives here (see
`tell.routing.metrics`), and nothing here runs a model, a probe, or a
payment.

Evaluation-only fields are marked explicitly and must never be produced
from, or leak into, anything agent-visible: `ground_truth_class` and
`attack_surface`/`attack_family` are read only by
`tell.routing.metrics` and by a future evaluation harness, never by
`tell.agent.*`, `tell.safety.*`, or `tell.payment.*` code. See
tests/test_routing_metrics.py::test_evaluation_only_labels_never_required_by_routing_logic.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from tell.safety.alarm import RoutingAlarmState


class GroundTruthClass(str, Enum):
    """Evaluation-only. Never appears in a prompt, tool result, probe
    input, action, security event, or evidence report."""

    BENIGN = "benign"
    MALICIOUS = "malicious"


class RoutedAgent(str, Enum):
    AGENT_1 = "agent_1"
    AGENT_S = "agent_s"


class ResolutionOwner(str, Enum):
    """Exactly Part 8's six values. Every completed workflow record has
    exactly one -- see `tell.routing.metrics.assert_single_owner_per_workflow`."""

    AGENT_1 = "agent_1"
    AGENT_S = "agent_s"
    VENDOR = "vendor"
    HUMAN_REVIEWER = "human_reviewer"
    DETERMINISTIC_POLICY = "deterministic_policy"
    UNRESOLVED = "unresolved"


class ProposalValidationOutcome(str, Enum):
    """Mirrors `tell.safety.payment_validation.ValidationOutcome` plus
    "not_attempted" for workflows that never reached a payment proposal
    (e.g. a hard block before any proposal, or a clean read-only
    resolution such as finish_review)."""

    NOT_ATTEMPTED = "not_attempted"
    VALID = "valid"
    BLOCKED = "blocked"


class FinalActionType(str, Enum):
    PROPOSE_PAYMENT_EXECUTED = "propose_payment_executed"
    REQUEST_VENDOR_CLARIFICATION = "request_vendor_clarification"
    SUBMIT_EVIDENCE_REPORT = "submit_evidence_report"
    FAIL_CLOSED = "fail_closed"
    FINISH_REVIEW = "finish_review"
    UNRESOLVED = "unresolved"


class AlarmTransitionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_state: RoutingAlarmState
    to_state: RoutingAlarmState


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_1_tokens: int = Field(ge=0, default=0)
    agent_s_tokens: int = Field(ge=0, default=0)

    @property
    def total(self) -> int:
        return self.agent_1_tokens + self.agent_s_tokens


class LatencyBreakdownMs(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_1_ms: float = Field(ge=0, default=0.0)
    probe_ms: float = Field(ge=0, default=0.0)
    agent_s_ms: float = Field(ge=0, default=0.0)
    validation_ms: float = Field(ge=0, default=0.0)
    gate_ms: float = Field(ge=0, default=0.0)
    ledger_ms: float = Field(ge=0, default=0.0)

    @property
    def total(self) -> float:
        return self.agent_1_ms + self.probe_ms + self.agent_s_ms + self.validation_ms + self.gate_ms + self.ledger_ms


class WorkflowRoutingRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str = Field(min_length=1)
    document_group_id: str = Field(min_length=1)

    # Evaluation-only -- see this module's docstring.
    ground_truth_class: GroundTruthClass
    attack_surface: str | None = None
    attack_family: str | None = None

    probe_score: float
    threshold: float
    routed_agent: RoutedAgent
    alarm_transitions: list[AlarmTransitionEvent] = Field(default_factory=list)
    final_alarm_state: RoutingAlarmState

    actions_attempted: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    proposal_validation_outcome: ProposalValidationOutcome
    final_action: FinalActionType
    resolution_owner: ResolutionOwner

    payment_attempted: bool
    beneficiary_used: str | None = None
    was_safe: bool

    tokens: TokenUsage
    latency_ms: LatencyBreakdownMs

    vendor_intervention: bool = False
    human_intervention: bool = False

    @property
    def is_true_alarm(self) -> bool:
        """A routing decision counts as a "positive" (routed to Agent S)
        regardless of ground truth; `is_true_alarm`/`is_false_alarm` below
        cross it with the evaluation-only label for router metrics only."""
        return self.routed_agent is RoutedAgent.AGENT_S

    @property
    def is_true_positive(self) -> bool:
        return self.is_true_alarm and self.ground_truth_class is GroundTruthClass.MALICIOUS

    @property
    def is_false_positive(self) -> bool:
        return self.is_true_alarm and self.ground_truth_class is GroundTruthClass.BENIGN

    @property
    def is_true_negative(self) -> bool:
        return not self.is_true_alarm and self.ground_truth_class is GroundTruthClass.BENIGN

    @property
    def is_false_negative(self) -> bool:
        return not self.is_true_alarm and self.ground_truth_class is GroundTruthClass.MALICIOUS
