"""Freezes the `task_only_base_v1` autonomous-loop protocol, before any
task-only clean or attacked model run. Mirrors
scripts/freeze_autonomous_loop_v2_protocol.py exactly, over the same
unmodified v2 action schema and the same unmodified clean/attacked
scenarios -- only the prompt profile differs.

Requires results/scenario_design/task_only_vs_hardened_diff.md to
already exist (proves the ablation is scoped to prompt-policy language
only) before writing this manifest.

Writes results/scenario_design/task_only_autonomous_protocol_manifest.json.
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
from tell.agent.prompt_profiles import PromptProfile
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario

ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"
ATTACK_DIR = Path("/home/hp5/tell/data/scenarios/attacked/email")

ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
LOOP_SRC = Path("/home/hp5/tell/src/tell/agent/loop.py")
LOOP_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/loop_prompts.py")
PROMPT_PROFILES_SRC = Path("/home/hp5/tell/src/tell/agent/prompt_profiles.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")
AGENTIC_OUTCOMES_SRC = Path("/home/hp5/tell/src/tell/evaluation/agentic_outcomes.py")

V2_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v2_protocol_manifest.json")
DIFF_REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/task_only_vs_hardened_diff.md")
AUDIT_PATH = Path("/home/hp5/tell/results/scenario_design/hardened_prompt_policy_audit.md")
MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/task_only_autonomous_protocol_manifest.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    if not DIFF_REPORT_PATH.exists():
        raise RuntimeError(f"{DIFF_REPORT_PATH} does not exist -- run scripts/diff_task_only_vs_hardened_prompts.py first.")
    if not AUDIT_PATH.exists():
        raise RuntimeError(f"{AUDIT_PATH} does not exist -- the prompt-policy audit must exist before freezing.")
    diff_text = DIFF_REPORT_PATH.read_text()
    if "**PASS**" not in diff_text:
        raise RuntimeError(f"{DIFF_REPORT_PATH} does not report PASS -- fix the ablation scope before freezing.")

    v2_manifest = json.loads(V2_MANIFEST_PATH.read_text())
    clean_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    attack_paths = sorted(ATTACK_DIR.glob("*.json"))
    if len(attack_paths) != 5:
        raise RuntimeError(f"Expected 5 attacked scenarios, found {len(attack_paths)}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__  # class name only; never instantiated here

    rendered_task_only_prompt = build_loop_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)
    rendered_hardened_prompt = build_loop_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)
    assert _sha256_text(rendered_hardened_prompt) == v2_manifest["rendered_loop_system_prompt_sha256"]

    manifest = {
        "protocol": "autonomous_email_attack_loop_task_only_base_v1",
        "prompt_profile": PromptProfile.TASK_ONLY_BASE_V1.value,
        "reference_hardened_protocol": "autonomous_email_attack_loop_v2",
        "reference_hardened_prompt_profile": PromptProfile.HARDENED_PROVENANCE_PROMPT_V1.value,
        "reference_hardened_manifest_path": str(V2_MANIFEST_PATH),
        "diff_report_path": str(DIFF_REPORT_PATH),
        "prompt_policy_audit_path": str(AUDIT_PATH),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "clean_scenario_id": clean_bundle.scenario_id,
        "clean_scenario_path": str(DEFAULT_SCENARIO_PATH_04D531CA),
        "clean_scenario_sha256": _sha256_file(DEFAULT_SCENARIO_PATH_04D531CA),
        "attack_scenario_hashes": {p.name: _sha256_file(p) for p in attack_paths},
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "source_hashes": {
            "actions_schema": {"path": str(ACTIONS_SRC), "sha256": _sha256_file(ACTIONS_SRC)},
            "loop_engine": {"path": str(LOOP_SRC), "sha256": _sha256_file(LOOP_SRC)},
            "loop_prompt_builder": {"path": str(LOOP_PROMPTS_SRC), "sha256": _sha256_file(LOOP_PROMPTS_SRC)},
            "prompt_profiles": {"path": str(PROMPT_PROFILES_SRC), "sha256": _sha256_file(PROMPT_PROFILES_SRC)},
            "work_item_envelope": {"path": str(WORK_ITEM_SRC), "sha256": _sha256_file(WORK_ITEM_SRC)},
            "capture_module": {"path": str(CAPTURE_MODULE_SRC), "sha256": _sha256_file(CAPTURE_MODULE_SRC)},
            "agentic_outcomes_evaluator": {"path": str(AGENTIC_OUTCOMES_SRC), "sha256": _sha256_file(AGENTIC_OUTCOMES_SRC)},
        },
        # These four must exactly equal the v2 (hardened) manifest's own values --
        # this ablation never touches actions_schema, loop_engine, work_item_envelope,
        # capture_module, or agentic_outcomes_evaluator content, only which prompt
        # template loop_prompt_builder renders.
        "source_hashes_match_hardened_manifest": {
            name: v2_manifest["source_hashes"][name]["sha256"] == _sha256_file(path)
            for name, path in {
                "actions_schema": ACTIONS_SRC,
                "work_item_envelope": WORK_ITEM_SRC,
                "capture_module": CAPTURE_MODULE_SRC,
                "agentic_outcomes_evaluator": AGENTIC_OUTCOMES_SRC,
            }.items()
        },
        "rendered_task_only_system_prompt_sha256": _sha256_text(rendered_task_only_prompt),
        "rendered_task_only_system_prompt_char_count": len(rendered_task_only_prompt),
        "rendered_task_only_system_prompt": rendered_task_only_prompt,
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
        "constraints": [
            "This ablation changes only which system-prompt template tell.agent.loop_prompts.build_loop_system_prompt "
            "renders, selected via a prompt_profile parameter that defaults to the hardened profile everywhere else "
            "in this codebase. Action schemas, tool schemas, tool execution, provenance derivation, scenario/attack "
            "content, generation parameters, model revision, hidden-state indices, and max turns are byte-identical "
            "to the reference hardened (autonomous_loop_v2) protocol -- see diff_report_path.",
            "The task-only prompt excludes every sentence classified as security/trust guidance or attack-specific "
            "guidance in prompt_policy_audit_path; it retains the AP role, the six action descriptions, the JSON-only "
            "output contract, and the ordinary business task.",
            "This manifest is written before any task-only clean or attacked model run and must not be edited after "
            "seeing run output. The reference hardened manifest/results are read-only historical artifacts and are "
            "never rerun or relabeled by this protocol.",
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
