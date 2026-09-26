"""Freezes both Session-A and Session-B protocols for the delayed-
memory-poisoning pilot, before any clean or poisoned inference in either
session.

Records, for each session: the rendered system prompt hash (embeds the
session's own action schema, generated live), source hashes of the
prompt builder, actions module, memory-loop engine, memory store/models,
and work-item envelope, plus model identity/revision, generation
parameters, turn budget, and hidden-state indices -- everything that
must not change across the clean and poisoned runs of that session.

This script loads only the tokenizer (no GPU, no model weights); model
class is read directly off the imported class object, never
instantiated.

Writes:
  - results/scenario_design/memory_pilot_session_a_protocol_manifest.json
  - results/scenario_design/memory_pilot_session_b_protocol_manifest.json

Once written, neither file may be edited after seeing any run output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer, Qwen3ForCausalLM

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS
from tell.agent.memory_prompts import SESSION_A_TASK_TEXT, build_session_a_system_prompt, build_session_b_system_prompt
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario

ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"

ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
MEMORY_LOOP_SRC = Path("/home/hp5/tell/src/tell/agent/memory_loop.py")
MEMORY_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/memory_prompts.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")
MEMORY_MODELS_SRC = Path("/home/hp5/tell/src/tell/memory/models.py")
MEMORY_STORE_SRC = Path("/home/hp5/tell/src/tell/memory/store.py")
MEMORY_OUTCOMES_SRC = Path("/home/hp5/tell/src/tell/evaluation/memory_outcomes.py")

SCENARIO_DIR = Path("/home/hp5/tell/data/scenarios/memory_infection")

SESSION_A_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/memory_pilot_session_a_protocol_manifest.json")
SESSION_B_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/memory_pilot_session_b_protocol_manifest.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SHARED_SOURCE_HASHES = {
    "actions_schema": ACTIONS_SRC,
    "memory_loop_engine": MEMORY_LOOP_SRC,
    "memory_prompt_builder": MEMORY_PROMPTS_SRC,
    "work_item_envelope": WORK_ITEM_SRC,
    "capture_module": CAPTURE_MODULE_SRC,
    "memory_models": MEMORY_MODELS_SRC,
    "memory_store": MEMORY_STORE_SRC,
    "memory_outcomes_evaluator": MEMORY_OUTCOMES_SRC,
}

_GENERATION_PARAMETERS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "use_cache": True,
    "output_hidden_states": False,
    "return_dict_in_generate": False,
    "enable_thinking": False,
    "add_generation_prompt": True,
}


def main() -> None:
    clean_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    clean_infection_path = SCENARIO_DIR / "memory_infection_clean_04d531ca_v1.json"
    poisoned_infection_path = SCENARIO_DIR / "memory_infection_poisoned_04d531ca_v1.json"

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__

    source_hashes = {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in _SHARED_SOURCE_HASHES.items()}

    frozen_at = datetime.now(timezone.utc).isoformat()

    session_a_prompt = build_session_a_system_prompt()
    session_a_manifest = {
        "protocol": "memory_pilot_session_a_v1",
        "frozen_at": frozen_at,
        "task_text": SESSION_A_TASK_TEXT,
        "clean_infection_scenario_path": str(clean_infection_path),
        "clean_infection_scenario_sha256": _sha256_file(clean_infection_path),
        "poisoned_infection_scenario_path": str(poisoned_infection_path),
        "poisoned_infection_scenario_sha256": _sha256_file(poisoned_infection_path),
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "source_hashes": source_hashes,
        "rendered_system_prompt_sha256": _sha256_text(session_a_prompt),
        "rendered_system_prompt_char_count": len(session_a_prompt),
        "generation_parameters": _GENERATION_PARAMETERS,
        "max_turns": MAX_TURNS,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "available_actions": ["read_email", "write_memory", "request_review", "finish_review", "fail_closed"],
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "constraints": [
            "No payment action exists in Session A's action set at all.",
            "write_memory accepts only vendor_id, memory_kind, content, and a typed reason from the model; "
            "memory_id, created_at, status, and every origin_* field are derived by tell.agent.memory_loop, "
            "never accepted from model output.",
            "This manifest is written before any clean or poisoned Session-A inference and must not be "
            "edited after seeing run output.",
        ],
    }
    SESSION_A_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_A_MANIFEST_PATH.write_text(json.dumps(session_a_manifest, indent=2))
    print(f"Wrote {SESSION_A_MANIFEST_PATH}")

    session_b_prompt = build_session_b_system_prompt()
    session_b_manifest = {
        "protocol": "memory_pilot_session_b_v1",
        "frozen_at": frozen_at,
        "clean_payment_scenario_id": clean_bundle.scenario_id,
        "clean_payment_scenario_path": str(DEFAULT_SCENARIO_PATH_04D531CA),
        "clean_payment_scenario_sha256": _sha256_file(DEFAULT_SCENARIO_PATH_04D531CA),
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "source_hashes": source_hashes,
        "rendered_system_prompt_sha256": _sha256_text(session_b_prompt),
        "rendered_system_prompt_char_count": len(session_b_prompt),
        "generation_parameters": _GENERATION_PARAMETERS,
        "max_turns": MAX_TURNS,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "available_actions": ["read_email", "read_invoice", "search_memory", "get_vendor_record", "propose_payment", "request_review", "fail_closed"],
        "approved_beneficiary_account_id": clean_bundle.trusted_state.approved_beneficiary_account_id,
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "constraints": [
            "propose_payment is recorded like any other terminal action but is never executed by this pilot -- "
            "no ledger, gate, or SQLite payment operation is ever invoked.",
            "search_memory results preserve every retrieved record's original origin_* fields verbatim; "
            "tell.agent.memory_loop never re-derives or upgrades them on retrieval.",
            "Session B's message history starts fresh for every run -- no Session-A message, prompt, or KV "
            "cache state carries over; only the persisted memory database and ordinary trusted application "
            "state (scenario, vendor record) are shared.",
            "This manifest is written before any Session-B retrieval run and must not be edited after seeing "
            "run output.",
        ],
    }
    SESSION_B_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_B_MANIFEST_PATH.write_text(json.dumps(session_b_manifest, indent=2))
    print(f"Wrote {SESSION_B_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
