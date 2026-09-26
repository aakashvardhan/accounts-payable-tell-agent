"""Versioned LoRA pre-training contract v1 over the frozen v2.2 action contract.

Decision on `fail_closed` -- option B (kept for backward compatibility,
excluded from the LoRA target space, documented as a runtime outcome)
--------------------------------------------------------------------------
Option A (remove `fail_closed` from the model-generatable union) would
change the Session A/B system prompts, because both prompts embed the full
JSON schema and the action list. Every one of the 2,500 frozen v2.2 LoRA
inputs (and every probe v2.2 input) contains that prompt, and its sha256 is
recorded in the frozen v2.2 manifests. Removing the action therefore cannot
be done without re-rendering the v2.2 inputs, i.e. without modifying a
historical protocol. So:

  - the v2.2 schema and prompts are used byte-for-byte (same sha256);
  - `fail_closed` has zero targets in v2.2 LoRA train/validation/test and
    zero in the training supplement, and the sampler refuses any pool that
    contains one (`assert_no_excluded_targets`);
  - a model-emitted `fail_closed` is parsed (backward compatible), never
    executed, and maps to the runtime outcome `model_stop_no_side_effect`:
    it cannot release, block, clear, or override anything, and it is
    counted as an off-target action by evaluation;
  - every hard failure is owned by a deterministic component
    (`hard_policy.OWNERSHIP`).

A future corpus version that re-renders its prompts should adopt option A.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/hp5/tell/src")))
sys.path.insert(0, str(Path("/home/hp5/tell/scripts")))

from enterprise_v2_2 import contract as C22  # noqa: E402  (frozen; reused read-only)

CONTRACT_VERSION = "tell_lora_pretraining_contract_v1"
BASE_ACTION_CONTRACT = C22.CONTRACT_VERSION  # "tell_action_contract_v2_2"
FAIL_CLOSED_OPTION = "B"

# Parseable by the v2.2 schema (backward compatible) ...
MODEL_PARSEABLE_ACTIONS = {
    "session_b_processing": ("read_email", "read_invoice", "search_memory", "get_vendor_record", "propose_payment",
                             "request_vendor_clarification", "submit_evidence_report", "fail_closed"),
    "session_a_intake": ("read_email", "write_memory", "submit_evidence_report", "finish_review", "fail_closed"),
}
# ... but only these may appear as LoRA training targets.
LORA_TARGET_ACTIONS = {
    "session_b_processing": ("read_email", "read_invoice", "search_memory", "get_vendor_record", "propose_payment",
                             "request_vendor_clarification", "submit_evidence_report"),
    "session_a_intake": ("submit_evidence_report", "finish_review"),
}
EXCLUDED_FROM_LORA_TARGETS = {
    "fail_closed": "runtime outcome owned by deterministic validation / gate / ledger (see hard_policy.OWNERSHIP); "
                   "kept parseable only for v2.2 schema/prompt backward compatibility",
    "write_memory": "Session A intake never targets a memory write in v2.2 (no durable-note targets exist); out of scope for this LoRA",
    "request_review": "v1/v2.1 generic review action; not in the v2.2 schema (replaced by request_vendor_clarification / submit_evidence_report)",
}
TOOL_ACTIONS = ("read_email", "read_invoice", "search_memory", "get_vendor_record")
TERMINAL_ACTIONS = ("propose_payment", "request_vendor_clarification", "submit_evidence_report", "finish_review", "fail_closed")

MODEL_FAIL_CLOSED_RUNTIME_OUTCOME = "model_stop_no_side_effect"


def is_lora_target(action: str, contract: str) -> bool:
    return action in LORA_TARGET_ACTIONS[contract]


def assert_no_excluded_targets(rows) -> None:
    """rows: iterable of (sample_id, contract, gold_action_type)."""
    bad = [(sid, a) for sid, contract, a in rows if not is_lora_target(a, contract)]
    if bad:
        raise ValueError(f"excluded / non-target actions in LoRA pool: {bad[:5]} (n={len(bad)})")


def runtime_outcome_for_model_action(action: str) -> str:
    """How the runtime treats a parsed model action. Only tool reads and
    typed resolution actions have an effect; `fail_closed` never does."""
    if action == "fail_closed":
        return MODEL_FAIL_CLOSED_RUNTIME_OUTCOME
    if action == "propose_payment":
        return "payment_candidate_to_hard_validation_then_gate_then_ledger"
    if action in ("request_vendor_clarification", "submit_evidence_report"):
        return "resolution_coordinator_case"
    if action in TOOL_ACTIONS:
        return "read_only_tool_call"
    return "no_side_effect"
