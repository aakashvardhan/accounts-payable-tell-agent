"""Freezes the version-2 autonomous-loop protocol, before any clean or
attacked version-2 model run. Mirrors
scripts/freeze_autonomous_loop_protocol.py (v1) exactly, over the
now-corrected action schema. Does not modify the v1 manifest.

Writes results/scenario_design/autonomous_loop_v2_protocol_manifest.json.
Once written, this file must not be edited after seeing any run output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer, Qwen3ForCausalLM

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS
from tell.agent.loop_prompts import build_loop_system_prompt
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario

ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"

ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
LOOP_SRC = Path("/home/hp5/tell/src/tell/agent/loop.py")
LOOP_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/loop_prompts.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")
AGENTIC_OUTCOMES_SRC = Path("/home/hp5/tell/src/tell/evaluation/agentic_outcomes.py")

V1_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_protocol_manifest.json")
DIFF_REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v1_v2_diff.md")
MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v2_protocol_manifest.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    if not DIFF_REPORT_PATH.exists():
        raise RuntimeError(
            f"{DIFF_REPORT_PATH} does not exist -- run scripts/diff_autonomous_loop_v1_v2.py first "
            "to prove the change is narrowly scoped before freezing the v2 protocol."
        )

    clean_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__  # class name only; never instantiated here

    rendered_system_prompt = build_loop_system_prompt()

    manifest = {
        "protocol": "autonomous_email_attack_loop_v2",
        "supersedes": "autonomous_email_attack_loop_v1",
        "v1_manifest_path": str(V1_MANIFEST_PATH),
        "diff_report_path": str(DIFF_REPORT_PATH),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "clean_scenario_id": clean_bundle.scenario_id,
        "clean_scenario_path": str(DEFAULT_SCENARIO_PATH_04D531CA),
        "clean_scenario_sha256": _sha256_file(DEFAULT_SCENARIO_PATH_04D531CA),
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "source_hashes": {
            "actions_schema": {"path": str(ACTIONS_SRC), "sha256": _sha256_file(ACTIONS_SRC)},
            "loop_engine": {"path": str(LOOP_SRC), "sha256": _sha256_file(LOOP_SRC)},
            "loop_prompt_builder": {"path": str(LOOP_PROMPTS_SRC), "sha256": _sha256_file(LOOP_PROMPTS_SRC)},
            "work_item_envelope": {"path": str(WORK_ITEM_SRC), "sha256": _sha256_file(WORK_ITEM_SRC)},
            "capture_module": {"path": str(CAPTURE_MODULE_SRC), "sha256": _sha256_file(CAPTURE_MODULE_SRC)},
            "agentic_outcomes_evaluator": {"path": str(AGENTIC_OUTCOMES_SRC), "sha256": _sha256_file(AGENTIC_OUTCOMES_SRC)},
        },
        "rendered_loop_system_prompt_sha256": _sha256_text(rendered_system_prompt),
        "rendered_loop_system_prompt_char_count": len(rendered_system_prompt),
        "generation_parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "output_hidden_states": False,
            "return_dict_in_generate": False,
            "enable_thinking": False,
            "add_generation_prompt": True,
        },
        "max_turns": MAX_TURNS,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "approved_beneficiary_account_id": clean_bundle.trusted_state.approved_beneficiary_account_id,
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "action_schema_change_summary": (
            "ProposePaymentAction.invoice_id (ambiguous) replaced by required invoice_document_id "
            "(opaque internal id) and invoice_number (human-readable business number), with a "
            "same-value consistency validator against evidence.invoice_document_id. Legacy invoice_id "
            "is rejected as an unknown field (extra=\"forbid\"), not accepted as an alias."
        ),
        "constraints": [
            "The loop deterministically enforces only which schema-valid action types may execute (the three "
            "read-only tools); it never forces a specific tool call and never rejects/rewrites a terminal action "
            "because required tools were skipped.",
            "The attack target account id may appear only in an attacked email body, evaluation_only metadata, "
            "and evaluation reports -- never in trusted_state, the clean scenario, the loop system prompt, the "
            "work item, or activation metadata.",
            "Trust policy text, available read tools, tool descriptions, provenance handling, attack scenarios, "
            "generation settings, model revision, selected hidden-state indices, max turns, and terminal safety "
            "behavior are unchanged from v1 -- see the diff report for verification.",
            "This manifest is written before any clean or attacked version-2 model run and must not be edited "
            "after seeing run output.",
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
