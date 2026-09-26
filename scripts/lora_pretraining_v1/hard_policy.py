"""Hard-failure ownership: every hard failure belongs to a deterministic
component, never to a LoRA-generated `fail_closed`.

The LoRA proposes task or resolution actions. It does not replace
deterministic transaction policy. The pieces reused here are unchanged:

  - `tell.safety.gate.evaluate_gate` (structural validity + alarm state);
  - `tell.payment.ledger.Ledger` (currency, accounts, funds, duplicate
    execution, intent-not-permitted -- all raise `LedgerError` subclasses);
  - `enterprise_v2_2.runtime.ResolutionCoordinator` (pending clarification /
    human verification keep `AlarmState.UNRESOLVED`, so the gate blocks).

New here (composition only, no existing module is edited):

  - `validate_candidate`: the deterministic pre-gate validation layer --
    currency must be one the ledger supports, and a payment to an inactive
    vendor is refused (remediation path: evidence report).
  - `decide_payment`: validation first, then the unchanged gate.
  - `settle_through_ledger`: runs the ledger's only money path and turns a
    ledger exception into a typed `HardRejection` owned by the ledger.
  - `HardRejectionRegistry`: a recorded hard rejection is sticky for its
    invoice. Nothing a model emits can clear it; only the application or a
    registered human reviewer can.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path("/home/hp5/tell/src")))
sys.path.insert(0, str(Path("/home/hp5/tell/scripts")))

from tell.agent.tools import PayInvoiceCandidate  # noqa: E402
from tell.payment import ledger as L  # noqa: E402
from tell.safety.gate import AlarmState, GateDecision, GateDecisionType, evaluate_gate  # noqa: E402


class HardFailure(str, Enum):
    UNSUPPORTED_CURRENCY = "unsupported_currency"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_ACCOUNT = "invalid_account"
    DUPLICATE_EXECUTION = "duplicate_execution"
    INTENT_NOT_PERMITTED = "intent_not_permitted"
    PENDING_VENDOR_CLARIFICATION = "pending_vendor_clarification"
    PENDING_HUMAN_VERIFICATION = "pending_human_verification"
    VENDOR_RECORD_MISSING_OR_INACTIVE = "vendor_record_missing_or_inactive"
    BENEFICIARY_CONFLICT = "beneficiary_conflict"


class Owner(str, Enum):
    DETERMINISTIC_VALIDATION = "deterministic_validation"
    GATE = "deterministic_action_gate"
    LEDGER = "ledger"
    EVIDENCE_REPORT = "evidence_report"


@dataclass(frozen=True)
class Ownership:
    owner: Owner
    outcome: str
    enforced_by: str
    model_role: str


OWNERSHIP: dict[HardFailure, Ownership] = {
    HardFailure.UNSUPPORTED_CURRENCY: Ownership(
        Owner.DETERMINISTIC_VALIDATION, "rejected before the gate; the ledger also refuses the intent (UnsupportedCurrencyError)",
        "lora_pretraining_v1.hard_policy.validate_candidate; tell.payment.ledger.Ledger.create_payment_intent",
        "none -- no fail_closed target; any proposed payment is rejected deterministically"),
    HardFailure.INSUFFICIENT_FUNDS: Ownership(
        Owner.LEDGER, "execution rejected (InsufficientFundsError); no journal entry", "tell.payment.ledger.Ledger.execute_intent",
        "none -- the model cannot observe or waive the payer balance"),
    HardFailure.INVALID_ACCOUNT: Ownership(
        Owner.LEDGER, "intent rejected (UnknownAccountError)", "tell.payment.ledger.Ledger.create_payment_intent / execute_intent",
        "none"),
    HardFailure.DUPLICATE_EXECUTION: Ownership(
        Owner.LEDGER, "second execution rejected (DuplicateExecutionError)", "tell.payment.ledger.Ledger.execute_intent",
        "none"),
    HardFailure.INTENT_NOT_PERMITTED: Ownership(
        Owner.GATE, "gate blocks; the ledger refuses to execute a non-permitted intent (IntentNotPermittedError)",
        "tell.safety.gate.evaluate_gate; tell.payment.ledger.Ledger.execute_intent", "none"),
    HardFailure.PENDING_VENDOR_CLARIFICATION: Ownership(
        Owner.GATE, "gate remains blocked (case alarm_state UNRESOLVED) until a typed human reviewer decision",
        "enterprise_v2_2.runtime.ResolutionCoordinator.gate_payment", "request_vendor_clarification opens the case; cannot close it"),
    HardFailure.PENDING_HUMAN_VERIFICATION: Ownership(
        Owner.GATE, "gate remains blocked (case alarm_state UNRESOLVED) until a typed human reviewer decision",
        "enterprise_v2_2.runtime.ResolutionCoordinator.gate_payment / apply_reviewer_decision", "submit_evidence_report opens the case; cannot close it"),
    HardFailure.VENDOR_RECORD_MISSING_OR_INACTIVE: Ownership(
        Owner.EVIDENCE_REPORT, "remediable: evidence report to internal AP operations; payment is refused meanwhile (inactive vendor also refused by validate_candidate)",
        "v2.2 policy T4 (missing record); lora_pretraining_v1.hard_policy.validate_candidate (inactive)",
        "submit_evidence_report (supersedes the v2.2 T2 fail_closed branch, which has zero v2.2 LoRA examples)"),
    HardFailure.BENEFICIARY_CONFLICT: Ownership(
        Owner.EVIDENCE_REPORT, "evidence report to internal security review; gate blocked while the alarm/case is unresolved",
        "v2.2 policy T3; enterprise_v2_2.runtime.ResolutionCoordinator", "submit_evidence_report"),
}


@dataclass(frozen=True)
class HardRejection:
    condition: HardFailure
    owner: Owner
    detail: str
    remediation: str | None = None


def validate_candidate(candidate: PayInvoiceCandidate, *, trusted_vendor_status: str | None = None) -> HardRejection | None:
    """Deterministic pre-gate validation. Pure; no model input besides the
    candidate itself, whose fields are checked, never trusted."""
    if (candidate.currency or "").lower() not in L.SUPPORTED_CURRENCIES:
        return HardRejection(HardFailure.UNSUPPORTED_CURRENCY, Owner.DETERMINISTIC_VALIDATION, f"currency {candidate.currency!r} is not supported")
    if trusted_vendor_status is not None and trusted_vendor_status != "active":
        return HardRejection(HardFailure.VENDOR_RECORD_MISSING_OR_INACTIVE, Owner.DETERMINISTIC_VALIDATION,
                             f"trusted vendor status is {trusted_vendor_status!r}", remediation="submit_evidence_report")
    return None


@dataclass(frozen=True)
class PaymentDecision:
    permitted: bool
    rejection: HardRejection | None
    gate: GateDecision | None


def decide_payment(candidate: PayInvoiceCandidate, alarm_state: AlarmState, recovery_resolved: bool = False, *,
                   trusted_vendor_status: str | None = None, registry: "HardRejectionRegistry | None" = None) -> PaymentDecision:
    if registry is not None and (sticky := registry.get(candidate.invoice_id)) is not None:
        return PaymentDecision(False, sticky, None)
    rej = validate_candidate(candidate, trusted_vendor_status=trusted_vendor_status)
    if rej is not None:
        if registry is not None:
            registry.record(candidate.invoice_id, rej)
        return PaymentDecision(False, rej, None)
    g = evaluate_gate(candidate, alarm_state, recovery_resolved=recovery_resolved)
    if g.decision != GateDecisionType.PERMIT:
        return PaymentDecision(False, HardRejection(HardFailure.INTENT_NOT_PERMITTED, Owner.GATE, g.reason_code.value), g)
    return PaymentDecision(True, None, g)


_LEDGER_ERRORS = (
    (L.UnsupportedCurrencyError, HardFailure.UNSUPPORTED_CURRENCY),
    (L.InsufficientFundsError, HardFailure.INSUFFICIENT_FUNDS),
    (L.UnknownAccountError, HardFailure.INVALID_ACCOUNT),
    (L.DuplicateExecutionError, HardFailure.DUPLICATE_EXECUTION),
    (L.IntentNotPermittedError, HardFailure.INTENT_NOT_PERMITTED),
)


def ledger_rejection(exc: L.LedgerError) -> HardRejection:
    for cls, cond in _LEDGER_ERRORS:
        if isinstance(exc, cls):
            return HardRejection(cond, Owner.LEDGER, str(exc))
    raise exc


def settle_through_ledger(ledger: L.Ledger, *, intent_id: str, candidate: PayInvoiceCandidate, decision: PaymentDecision,
                          registry: "HardRejectionRegistry | None" = None) -> HardRejection | None:
    """The ledger's only money path: create intent -> record gate decision
    -> execute. Returns None on success, or the ledger-owned rejection."""
    try:
        if ledger._get_intent_row(intent_id) is None:
            ledger.create_payment_intent(intent_id=intent_id, invoice_id=candidate.invoice_id, source_account_id=candidate.source_account_id,
                                         beneficiary_account_id=candidate.beneficiary_account_id, amount_minor_units=candidate.amount_minor_units,
                                         currency=candidate.currency, reason=candidate.reason)
            ledger.record_gate_decision(intent_id=intent_id, alarm_state=(decision.gate.alarm_state.value if decision.gate else "unresolved"),
                                        decision="permit" if decision.permitted else "block",
                                        reason=decision.rejection.detail if decision.rejection else "permitted")
        ledger.execute_intent(intent_id)
        return None
    except L.LedgerError as exc:
        rej = ledger_rejection(exc)
        if registry is not None:
            registry.record(candidate.invoice_id, rej)
        return rej


MODEL_ACTORS = frozenset({"model", "agent", "safety_lora", "probe"})
CLEARING_ACTORS = frozenset({"application", "human_reviewer"})


@dataclass
class HardRejectionRegistry:
    """Sticky per-invoice hard rejections. A model action never clears one."""

    _by_invoice: dict[str, HardRejection] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def record(self, invoice_id: str, rej: HardRejection) -> None:
        self._by_invoice.setdefault(invoice_id, rej)
        self.history.append({"event": "hard_rejection_recorded", "invoice_id": invoice_id, "condition": rej.condition.value, "owner": rej.owner.value})

    def get(self, invoice_id: str) -> HardRejection | None:
        return self._by_invoice.get(invoice_id)

    def clear(self, invoice_id: str, *, actor_type: str, trusted_reason: str) -> None:
        if actor_type in MODEL_ACTORS or actor_type not in CLEARING_ACTORS:
            raise PermissionError("a hard rejection can only be cleared by the application or a human reviewer, never by a model action")
        if not trusted_reason:
            raise ValueError("clearing a hard rejection requires a trusted reason")
        self._by_invoice.pop(invoice_id, None)
        self.history.append({"event": "hard_rejection_cleared", "invoice_id": invoice_id, "actor_type": actor_type, "reason": trusted_reason})


def apply_model_action_after_rejection(registry: HardRejectionRegistry, invoice_id: str, parsed_action) -> HardRejection | None:
    """Whatever the model emits after a hard rejection -- another payment,
    `fail_closed`, a report, a clarification -- the recorded rejection is
    returned unchanged. Model actions have no path to `registry.clear`."""
    _ = parsed_action
    return registry.get(invoice_id)
