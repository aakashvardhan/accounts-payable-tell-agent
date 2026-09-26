"""Tell Verify: a local, typed, deterministic middle-band evidence
adjudicator for three-zone Tell routing (secondary, post-hoc policy layered
on the frozen probe -- see `results/routing_design/tell_typed_verifier_v1.md`).

Independent design note
--------------------------------------------------------------------------
This module is inspired only by the general, common idea that software
consuming constrained typed decisions from an upstream signal is safer
than freeform text -- an idea with no single owner. It is implemented from
scratch against this repository's own types
(`tell.safety.payment_validation`, `tell.agent.routing_orchestrator`,
`tell.routing.records`) and contains no code, prompts, weights, API calls,
or branding from any third-party product. It calls no external service and
performs no model inference; it is pure Python control flow over typed,
already-computed inputs.

Ownership boundaries (enforced by construction, not by convention)
--------------------------------------------------------------------------
`VerificationDecision` cannot become a `ValidatedPayInvoiceCandidate`: it
has no field of that type, no method that constructs one, and this module
never imports `tell.safety.gate` or calls
`tell.safety.payment_validation.validate_payment_proposal`. It also never
imports anything from `tell.safety.resolution` or `tell.safety.alarm`, so
it has no way to move a `RoutingAlarmState` to `RESOLVED`/`HARD_BLOCKED`
even by accident -- see tests/test_typed_verifier.py::
test_verifier_cannot_construct_candidate_or_clear_alarm for a static proof
by introspection. Every `VerificationDecision` this module returns carries
`side_effect_authorized=False`; nothing in this file ever sets it to True.
The caller is unconditionally required to still run
`validate_payment_proposal` and `evaluate_gate` before anything executes --
Tell Verify's route is a *recommendation*, never an authorization.

Three-zone routing
--------------------------------------------------------------------------
    score <  low_threshold                    -> ZoneRoute.AGENT_1
    low_threshold <= score < high_threshold    -> ZoneRoute.TELL_VERIFY
    score >= high_threshold                    -> ZoneRoute.AGENT_S
    score is NaN / +-inf / outside [0, 1]      -> ZoneRoute.FAIL_CLOSED

`route_by_score` is the only place that inequality is implemented; the
frozen `configs/routing/tell_three_zone_v1.json` supplies the two
thresholds and is validated by `load_three_zone_config` before use.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tell.safety.payment_validation import (
    ObservedEvidence,
    PendingCaseKind,
    TrustedInvoiceRecord,
    TrustedVendorRecord,
)


# ---------------------------------------------------------------------------
# Part 1: typed, validated three-zone configuration
# ---------------------------------------------------------------------------


class ZoneRoute(str, Enum):
    AGENT_1 = "agent_1"
    TELL_VERIFY = "tell_verify"
    AGENT_S = "agent_s"
    FAIL_CLOSED = "fail_closed"


class ThreeZoneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    low_threshold: float
    high_threshold: float
    frozen_probe_weights_sha256: str
    primary_threshold_value: float
    operational_threshold_value: float
    provenance: tuple[str, ...]


class ThreeZoneConfigError(ValueError):
    """Raised by `load_three_zone_config` when the config file fails any
    validity constraint. Never silently relaxed or auto-corrected."""


def load_three_zone_config(path: str | Path, *, expected_probe_weights_sha256: str | None = None) -> ThreeZoneConfig:
    raw = json.loads(Path(path).read_text())
    try:
        low = float(raw["zones"]["low_threshold"])
        high = float(raw["zones"]["high_threshold"])
        probe_hash = str(raw["frozen_probe"]["weights_sha256"])
        primary_value = float(raw["primary_threshold_artifact"]["value"])
        operational_value = float(raw["operational_threshold_artifact"]["value"])
        provenance = tuple(raw["provenance"]["derived_from"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ThreeZoneConfigError(f"malformed or incomplete three-zone config: {exc}") from exc

    if not (0.0 <= low <= 1.0) or not (0.0 <= high <= 1.0):
        raise ThreeZoneConfigError(f"thresholds must be in [0, 1]: low={low}, high={high}")
    if not low < high:
        raise ThreeZoneConfigError(f"low_threshold must be strictly less than high_threshold: low={low}, high={high}")
    if not provenance:
        raise ThreeZoneConfigError("provenance.derived_from must be non-empty")
    if abs(primary_value - low) > 1e-12:
        raise ThreeZoneConfigError(f"primary_threshold_artifact.value ({primary_value}) does not equal low_threshold ({low})")
    if abs(operational_value - high) > 1e-12:
        raise ThreeZoneConfigError(f"operational_threshold_artifact.value ({operational_value}) does not equal high_threshold ({high})")
    if expected_probe_weights_sha256 is not None and probe_hash != expected_probe_weights_sha256:
        raise ThreeZoneConfigError(f"frozen_probe.weights_sha256 ({probe_hash}) does not match the loaded probe ({expected_probe_weights_sha256})")

    return ThreeZoneConfig(low_threshold=low, high_threshold=high, frozen_probe_weights_sha256=probe_hash,
                           primary_threshold_value=primary_value, operational_threshold_value=operational_value,
                           provenance=provenance)


def route_by_score(score: float, config: ThreeZoneConfig) -> ZoneRoute:
    """The only place the three-zone boundary inequalities are implemented.
    `score >= high` is checked before the low comparison so a NaN (which
    compares False to everything) falls through to FAIL_CLOSED rather than
    silently landing in AGENT_1."""

    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not (0.0 <= score <= 1.0):
        return ZoneRoute.FAIL_CLOSED
    if score >= config.high_threshold:
        return ZoneRoute.AGENT_S
    if score >= config.low_threshold:
        return ZoneRoute.TELL_VERIFY
    return ZoneRoute.AGENT_1


# ---------------------------------------------------------------------------
# Part 2: typed verification input/output
# ---------------------------------------------------------------------------


class PaymentHistoryLookup(BaseModel):
    """Result of a read-only trusted payment-history lookup. This module
    never constructs one from untrusted text -- the caller must obtain it
    from the existing trusted lookup interface and pass it in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lookup_succeeded: bool
    duplicate_payment_detected: bool = False
    prior_payment_to_different_beneficiary: bool = False


class DisputeCaseLookup(BaseModel):
    """Result of a read-only trusted dispute/case lookup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lookup_succeeded: bool
    active_dispute: bool = False
    pending_case: PendingCaseKind | None = None


class MemoryProvenanceStatus(str, Enum):
    NO_MEMORY_CONSULTED = "no_memory_consulted"
    TRUSTED_CLEAN = "trusted_clean"
    UNTRUSTED_UNVERIFIED = "untrusted_unverified"
    QUARANTINED = "quarantined"
    CONFLICTS_WITH_TRUSTED_RECORD = "conflicts_with_trusted_record"


class ObservationSourceType(str, Enum):
    EMAIL = "email"
    INVOICE = "invoice"
    TOOL_RESULT = "tool_result"
    MEMORY = "memory"
    VENDOR_RECORD = "vendor_record"


class VerificationContext(BaseModel):
    """Everything Tell Verify may look at. Deliberately excludes
    ground-truth attack labels, expected-safe-action labels, evaluation
    partition labels, future model actions, and validator/gate/final
    outcomes -- see this module's docstring and
    tests/test_typed_verifier.py::test_context_rejects_label_like_fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tell_score: float
    observation_source_type: ObservationSourceType
    observation_provenance: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)

    canonical_vendor_id: str | None = None
    observed_vendor_id: str | None = None
    canonical_invoice_id: str | None = None
    observed_invoice_id: str | None = None

    trusted_vendor: TrustedVendorRecord | None = None
    trusted_invoice: TrustedInvoiceRecord | None = None
    observed: ObservedEvidence = Field(default_factory=ObservedEvidence)

    trusted_beneficiary_account_id: str | None = None
    observed_beneficiary_account_id: str | None = None
    trusted_amount_minor_units: int | None = None
    observed_amount_minor_units: int | None = None
    trusted_currency: str | None = None
    observed_currency: str | None = None

    payment_history: PaymentHistoryLookup
    dispute_case: DisputeCaseLookup
    memory_provenance: MemoryProvenanceStatus

    approved_vendor_contact_available: bool = False
    proposed_action_type: str | None = None


class EvidenceStatus(str, Enum):
    COMPLETE = "complete"
    CONFLICTING = "conflicting"
    INCOMPLETE = "incomplete"
    LOOKUP_FAILED = "lookup_failed"


class VerifyRoute(str, Enum):
    CONTINUE_AGENT_1 = "continue_agent_1"
    ESCALATE_AGENT_S = "escalate_agent_s"
    REQUEST_VENDOR_CLARIFICATION = "request_vendor_clarification"
    SUBMIT_EVIDENCE_REPORT = "submit_evidence_report"


class VerifyReasonCode(str, Enum):
    EVIDENCE_COMPLETE_AND_CONSISTENT = "evidence_complete_and_consistent"
    VENDOR_IDENTITY_CONFLICT = "vendor_identity_conflict"
    INVOICE_IDENTITY_CONFLICT = "invoice_identity_conflict"
    BENEFICIARY_CONFLICT = "beneficiary_conflict"
    AMOUNT_OR_CURRENCY_CONFLICT = "amount_or_currency_conflict"
    DUPLICATE_PAYMENT_DETECTED = "duplicate_payment_detected"
    PRIOR_PAYMENT_BENEFICIARY_CONFLICT = "prior_payment_beneficiary_conflict"
    ACTIVE_DISPUTE = "active_dispute"
    PENDING_CASE_UNRESOLVED = "pending_case_unresolved"
    UNTRUSTED_MEMORY_PROVENANCE = "untrusted_memory_provenance"
    MEMORY_CONFLICTS_WITH_TRUSTED_RECORD = "memory_conflicts_with_trusted_record"
    LOOKUP_FAILED = "lookup_failed"
    ORDINARY_MISSING_INFORMATION = "ordinary_missing_information"
    NO_APPROVED_CONTACT_FOR_CLARIFICATION = "no_approved_contact_for_clarification"
    MALFORMED_TRUSTED_RECORD = "malformed_trusted_record"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    INVALID_TELL_SCORE = "invalid_tell_score"
    VERIFIER_EXCEPTION = "verifier_exception"
    UNSUPPORTED_STATE = "unsupported_state"


class VerificationDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route: VerifyRoute
    evidence_status: EvidenceStatus
    reason_codes: tuple[VerifyReasonCode, ...]
    checked_trusted_sources: tuple[str, ...]
    source_provenance: str
    tell_score: float
    score_band: ZoneRoute
    evidence_fields_checked: int
    scenario_id: str
    side_effect_authorized: bool = False

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        if self.side_effect_authorized:
            raise ValueError("Tell Verify may never set side_effect_authorized=True")


# ---------------------------------------------------------------------------
# Part 3: deterministic policy
# ---------------------------------------------------------------------------


class LocalEvidenceAdjudicator:
    """`Tell Verify`. Stateless: every method is a pure function of its
    typed input. Holds no ledger connection, no memory-store handle, no
    executor, no gate -- it cannot reach any of the systems it is
    forbidden to touch because it never imports them."""

    def adjudicate(self, ctx: VerificationContext) -> VerificationDecision:
        try:
            return self._adjudicate(ctx)
        except Exception:
            # Fail-closed: any unexpected condition becomes an evidence
            # report, never a silent Agent-1 continuation.
            return VerificationDecision(
                route=VerifyRoute.SUBMIT_EVIDENCE_REPORT, evidence_status=EvidenceStatus.LOOKUP_FAILED,
                reason_codes=(VerifyReasonCode.VERIFIER_EXCEPTION,), checked_trusted_sources=(),
                source_provenance=getattr(ctx, "observation_provenance", "unknown"),
                tell_score=getattr(ctx, "tell_score", float("nan")), score_band=ZoneRoute.FAIL_CLOSED,
                evidence_fields_checked=0, scenario_id=getattr(ctx, "scenario_id", "unknown"),
            )

    def _adjudicate(self, ctx: VerificationContext) -> VerificationDecision:
        sources: list[str] = []
        checked = 0
        reasons: list[VerifyReasonCode] = []

        if not math.isfinite(ctx.tell_score) or not (0.0 <= ctx.tell_score <= 1.0):
            return self._decision(ctx, VerifyRoute.SUBMIT_EVIDENCE_REPORT, EvidenceStatus.LOOKUP_FAILED,
                                  [VerifyReasonCode.INVALID_TELL_SCORE], sources, 0, ZoneRoute.FAIL_CLOSED)

        # --- lookup failures fail closed immediately ---
        if not ctx.payment_history.lookup_succeeded or not ctx.dispute_case.lookup_succeeded:
            reasons.append(VerifyReasonCode.LOOKUP_FAILED)
            return self._decision(ctx, VerifyRoute.SUBMIT_EVIDENCE_REPORT, EvidenceStatus.LOOKUP_FAILED, reasons, sources, checked)

        # --- vendor identity ---
        checked += 1
        sources.append("trusted_vendor_record")
        if ctx.trusted_vendor is None:
            return self._decision(ctx, VerifyRoute.SUBMIT_EVIDENCE_REPORT, EvidenceStatus.INCOMPLETE,
                                  [VerifyReasonCode.MALFORMED_TRUSTED_RECORD], sources, checked)
        if ctx.canonical_vendor_id is not None and ctx.observed_vendor_id is not None and ctx.canonical_vendor_id != ctx.observed_vendor_id:
            reasons.append(VerifyReasonCode.VENDOR_IDENTITY_CONFLICT)

        # --- invoice identity ---
        checked += 1
        sources.append("trusted_invoice_record")
        if ctx.canonical_invoice_id is not None and ctx.observed_invoice_id is not None and ctx.canonical_invoice_id != ctx.observed_invoice_id:
            reasons.append(VerifyReasonCode.INVOICE_IDENTITY_CONFLICT)

        # --- beneficiary consistency, where available ---
        checked += 1
        if ctx.trusted_beneficiary_account_id is not None and ctx.observed_beneficiary_account_id is not None:
            if ctx.trusted_beneficiary_account_id != ctx.observed_beneficiary_account_id:
                reasons.append(VerifyReasonCode.BENEFICIARY_CONFLICT)

        # --- amount / currency consistency, where available ---
        checked += 1
        amount_conflict = (ctx.trusted_amount_minor_units is not None and ctx.observed_amount_minor_units is not None
                          and ctx.trusted_amount_minor_units != ctx.observed_amount_minor_units)
        currency_conflict = (ctx.trusted_currency is not None and ctx.observed_currency is not None
                            and ctx.trusted_currency != ctx.observed_currency)
        if amount_conflict or currency_conflict:
            reasons.append(VerifyReasonCode.AMOUNT_OR_CURRENCY_CONFLICT)

        # --- payment history ---
        checked += 1
        sources.append("trusted_payment_history")
        if ctx.payment_history.duplicate_payment_detected:
            reasons.append(VerifyReasonCode.DUPLICATE_PAYMENT_DETECTED)
        if ctx.payment_history.prior_payment_to_different_beneficiary:
            reasons.append(VerifyReasonCode.PRIOR_PAYMENT_BENEFICIARY_CONFLICT)

        # --- dispute / pending case ---
        checked += 1
        sources.append("trusted_dispute_case_lookup")
        if ctx.dispute_case.active_dispute:
            reasons.append(VerifyReasonCode.ACTIVE_DISPUTE)
        if ctx.dispute_case.pending_case is not None:
            reasons.append(VerifyReasonCode.PENDING_CASE_UNRESOLVED)

        # --- memory provenance ---
        checked += 1
        if ctx.memory_provenance is MemoryProvenanceStatus.CONFLICTS_WITH_TRUSTED_RECORD:
            reasons.append(VerifyReasonCode.MEMORY_CONFLICTS_WITH_TRUSTED_RECORD)
        elif ctx.memory_provenance in (MemoryProvenanceStatus.UNTRUSTED_UNVERIFIED, MemoryProvenanceStatus.QUARANTINED):
            reasons.append(VerifyReasonCode.UNTRUSTED_MEMORY_PROVENANCE)

        conflict_reasons = {
            VerifyReasonCode.VENDOR_IDENTITY_CONFLICT, VerifyReasonCode.INVOICE_IDENTITY_CONFLICT,
            VerifyReasonCode.BENEFICIARY_CONFLICT, VerifyReasonCode.AMOUNT_OR_CURRENCY_CONFLICT,
            VerifyReasonCode.DUPLICATE_PAYMENT_DETECTED, VerifyReasonCode.PRIOR_PAYMENT_BENEFICIARY_CONFLICT,
            VerifyReasonCode.ACTIVE_DISPUTE, VerifyReasonCode.PENDING_CASE_UNRESOLVED,
            VerifyReasonCode.UNTRUSTED_MEMORY_PROVENANCE, VerifyReasonCode.MEMORY_CONFLICTS_WITH_TRUSTED_RECORD,
        }
        if any(r in conflict_reasons for r in reasons):
            return self._decision(ctx, VerifyRoute.ESCALATE_AGENT_S, EvidenceStatus.CONFLICTING, reasons, sources, checked)

        # --- no conflict found: is evidence complete? ---
        missing_ordinary = (
            (ctx.trusted_invoice is None or ctx.trusted_invoice.amount_minor_units is None
             or ctx.trusted_invoice.currency is None or not ctx.trusted_invoice.invoice_number)
            or ctx.observed.observed_invoice_document_id is None or ctx.observed.observed_vendor_record_id is None
        )
        if missing_ordinary:
            reasons.append(VerifyReasonCode.ORDINARY_MISSING_INFORMATION)
            if ctx.approved_vendor_contact_available:
                return self._decision(ctx, VerifyRoute.REQUEST_VENDOR_CLARIFICATION, EvidenceStatus.INCOMPLETE, reasons, sources, checked)
            reasons.append(VerifyReasonCode.NO_APPROVED_CONTACT_FOR_CLARIFICATION)
            return self._decision(ctx, VerifyRoute.SUBMIT_EVIDENCE_REPORT, EvidenceStatus.INCOMPLETE, reasons, sources, checked)

        reasons.append(VerifyReasonCode.EVIDENCE_COMPLETE_AND_CONSISTENT)
        return self._decision(ctx, VerifyRoute.CONTINUE_AGENT_1, EvidenceStatus.COMPLETE, reasons, sources, checked)

    @staticmethod
    def _decision(ctx: VerificationContext, route: VerifyRoute, status: EvidenceStatus,
                 reasons: list[VerifyReasonCode], sources: list[str], checked: int,
                 band: ZoneRoute = ZoneRoute.TELL_VERIFY) -> VerificationDecision:
        return VerificationDecision(
            route=route, evidence_status=status, reason_codes=tuple(reasons), checked_trusted_sources=tuple(sources),
            source_provenance=ctx.observation_provenance, tell_score=ctx.tell_score, score_band=band,
            evidence_fields_checked=checked, scenario_id=ctx.scenario_id,
        )
