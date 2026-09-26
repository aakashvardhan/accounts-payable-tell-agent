"""Freezes both Session-A and Session-B `task_only_base_v1` protocols
for the delayed-memory-poisoning pilot, before any task-only clean or
poisoned inference in either session. Mirrors
scripts/freeze_memory_pilot_protocol.py exactly, over the same
unmodified memory-infection scenarios -- only the prompt profile
differs.

Requires results/scenario_design/task_only_vs_hardened_diff.md to
already exist (proves the ablation is scoped to prompt-policy language
only) before writing these manifests.

Writes:
  - results/scenario_design/task_only_memory_pilot_session_a_protocol_manifest.json
  - results/scenario_design/task_only_memory_pilot_session_b_protocol_manifest.json

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
from tell.agent.memory_prompts import TASK_ONLY_SESSION_A_TASK_TEXT, build_session_a_system_prompt, build_session_b_system_prompt
from tell.agent.prompt_profiles import PromptProfile
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario

ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"

ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
MEMORY_LOOP_SRC = Path("/home/hp5/tell/src/tell/agent/memory_loop.py")
MEMORY_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/memory_prompts.py")
PROMPT_PROFILES_SRC = Path("/home/hp5/tell/src/tell/agent/prompt_profiles.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")
MEMORY_MODELS_SRC = Path("/home/hp5/tell/src/tell/memory/models.py")
MEMORY_STORE_SRC = Path("/home/hp5/tell/src/tell/memory/store.py")
MEMORY_OUTCOMES_SRC = Path("/home/hp5/tell/src/tell/evaluation/memory_outcomes.py")

SCENARIO_DIR = Path("/home/hp5/tell/data/scenarios/memory_infection")

DIFF_REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/task_only_vs_hardened_diff.md")
AUDIT_PATH = Path("/home/hp5/tell/results/scenario_design/hardened_prompt_policy_audit.md")
REFERENCE_SESSION_A_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/memory_pilot_session_a_protocol_manifest.json")
REFERENCE_SESSION_B_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/memory_pilot_session_b_protocol_manifest.json")

SESSION_A_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/task_only_memory_pilot_session_a_protocol_manifest.json")
SESSION_B_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/task_only_memory_pilot_session_b_protocol_manifest.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SHARED_SOURCE_HASHES = {
    "actions_schema": ACTIONS_SRC,
    "memory_loop_engine": MEMORY_LOOP_SRC,
    "memory_prompt_builder": MEMORY_PROMPTS_SRC,
    "prompt_profiles": PROMPT_PROFILES_SRC,
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
    if not DIFF_REPORT_PATH.exists():
        raise RuntimeError(f"{DIFF_REPORT_PATH} does not exist -- run scripts/diff_task_only_vs_hardened_prompts.py first.")
    if not AUDIT_PATH.exists():
        raise RuntimeError(f"{AUDIT_PATH} does not exist -- the prompt-policy audit must exist before freezing.")
    if "**PASS**" not in DIFF_REPORT_PATH.read_text():
        raise RuntimeError(f"{DIFF_REPORT_PATH} does not report PASS -- fix the ablation scope before freezing.")

    reference_a = json.loads(REFERENCE_SESSION_A_MANIFEST_PATH.read_text())
    reference_b = json.loads(REFERENCE_SESSION_B_MANIFEST_PATH.read_text())

    clean_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    clean_infection_path = SCENARIO_DIR / "memory_infection_clean_04d531ca_v1.json"
    poisoned_infection_path = SCENARIO_DIR / "memory_infection_poisoned_04d531ca_v1.json"

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__

    source_hashes = {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in _SHARED_SOURCE_HASHES.items()}

    frozen_at = datetime.now(timezone.utc).isoformat()

    hardened_a_prompt = build_session_a_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)
    assert _sha256_text(hardened_a_prompt) == reference_a["rendered_system_prompt_sha256"]
    task_only_a_prompt = build_session_a_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)

    session_a_manifest = {
        "protocol": "memory_pilot_session_a_task_only_base_v1",
        "prompt_profile": PromptProfile.TASK_ONLY_BASE_V1.value,
        "reference_hardened_protocol": "memory_pilot_session_a_v1",
        "reference_hardened_prompt_profile": PromptProfile.HARDENED_PROVENANCE_PROMPT_V1.value,
        "reference_hardened_manifest_path": str(REFERENCE_SESSION_A_MANIFEST_PATH),
        "diff_report_path": str(DIFF_REPORT_PATH),
        "prompt_policy_audit_path": str(AUDIT_PATH),
        "frozen_at": frozen_at,
        "task_text": TASK_ONLY_SESSION_A_TASK_TEXT,
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
        "rendered_task_only_system_prompt_sha256": _sha256_text(task_only_a_prompt),
        "rendered_task_only_system_prompt_char_count": len(task_only_a_prompt),
        "rendered_task_only_system_prompt": task_only_a_prompt,
        "generation_parameters": _GENERATION_PARAMETERS,
        "max_turns": MAX_TURNS,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "available_actions": ["read_email", "write_memory", "request_review", "finish_review", "fail_closed"],
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "constraints": [
            "No payment action exists in Session A's action set at all -- identical to the hardened protocol.",
            "write_memory accepts only vendor_id, memory_kind, content, and a typed reason from the model; every "
            "origin_* field and memory_id/created_at/status are derived by tell.agent.memory_loop, never accepted "
            "from model output -- this application-level trust boundary is unchanged by the prompt-profile ablation.",
            "This manifest is written before any task-only clean or poisoned Session-A inference and must not be "
            "edited after seeing run output. The reference hardened manifest/results are read-only historical "
            "artifacts and are never rerun or relabeled by this protocol.",
        ],
    }
    SESSION_A_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_A_MANIFEST_PATH.write_text(json.dumps(session_a_manifest, indent=2))
    print(f"Wrote {SESSION_A_MANIFEST_PATH}")

    hardened_b_prompt = build_session_b_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)
    assert _sha256_text(hardened_b_prompt) == reference_b["rendered_system_prompt_sha256"]
    task_only_b_prompt = build_session_b_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)

    session_b_manifest = {
        "protocol": "memory_pilot_session_b_task_only_base_v1",
        "prompt_profile": PromptProfile.TASK_ONLY_BASE_V1.value,
        "reference_hardened_protocol": "memory_pilot_session_b_v1",
        "reference_hardened_prompt_profile": PromptProfile.HARDENED_PROVENANCE_PROMPT_V1.value,
        "reference_hardened_manifest_path": str(REFERENCE_SESSION_B_MANIFEST_PATH),
        "diff_report_path": str(DIFF_REPORT_PATH),
        "prompt_policy_audit_path": str(AUDIT_PATH),
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
        "rendered_task_only_system_prompt_sha256": _sha256_text(task_only_b_prompt),
        "rendered_task_only_system_prompt_char_count": len(task_only_b_prompt),
        "rendered_task_only_system_prompt": task_only_b_prompt,
        "generation_parameters": _GENERATION_PARAMETERS,
        "max_turns": MAX_TURNS,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "available_actions": ["read_email", "read_invoice", "search_memory", "get_vendor_record", "propose_payment", "request_review", "fail_closed"],
        "approved_beneficiary_account_id": clean_bundle.trusted_state.approved_beneficiary_account_id,
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "constraints": [
            "propose_payment is recorded like any other terminal action but is never executed by this pilot -- "
            "no ledger, gate, or SQLite payment operation is ever invoked. Identical to the hardened protocol.",
            "search_memory results preserve every retrieved record's original origin_* fields verbatim; "
            "tell.agent.memory_loop never re-derives or upgrades them on retrieval, regardless of prompt profile.",
            "Session B's message history starts fresh for every run -- no Session-A message, prompt, or KV cache "
            "state carries over; only the persisted memory database and ordinary trusted application state "
            "(scenario, vendor record) are shared. Identical to the hardened protocol.",
            "The poisoned-memory Session-B run under this profile is executed only if task-only Session A "
            "actually stored the poisoned memory -- never force-seeded.",
            "This manifest is written before any task-only Session-B retrieval run and must not be edited after "
            "seeing run output. The reference hardened manifest/results are read-only historical artifacts and "
            "are never rerun or relabeled by this protocol.",
        ],
    }
    SESSION_B_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_B_MANIFEST_PATH.write_text(json.dumps(session_b_manifest, indent=2))
    print(f"Wrote {SESSION_B_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
