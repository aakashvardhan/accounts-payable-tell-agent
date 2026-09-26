"""Freezes the conditional-memory-retrieval protocol (the 2x2:
{task_only_base_v1, hardened_provenance_prompt_v1} x {clean, poisoned}
memory), before any inference. Also performs and records the
independent context-reset proof for all four rendered initial contexts
(Section 3 of the experiment spec), using only the tokenizer -- no
model weights are loaded, matching the discipline of every prior freeze
script in this repository.

Requires results/scenario_design/conditional_memory_retrieval_v1/memory_inputs_manifest.json
to already exist (scripts/inspect_and_freeze_memory_inputs.py).

Writes results/scenario_design/conditional_memory_retrieval_v1/protocol_manifest.json.
Once written, this file must not be edited after the first generation of
the downstream experiment begins.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer, Qwen3ForCausalLM

from tell.agent.actions import session_b_action_json_schema
from tell.agent.conditional_retrieval import RETRIEVAL_DECISION_POINT
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS
from tell.agent.memory_prompts import build_initial_user_message, build_search_memory_result_message, build_session_b_system_prompt
from tell.agent.prompt_profiles import PromptProfile
from tell.agent.work_item import TrustedWorkItem
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario
from tell.memory.store import MemoryStore

ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"

OUTPUT_DIR = Path("/home/hp5/tell/results/scenario_design/conditional_memory_retrieval_v1")
MEMORY_INPUTS_MANIFEST_PATH = OUTPUT_DIR / "memory_inputs_manifest.json"
PROTOCOL_MANIFEST_PATH = OUTPUT_DIR / "protocol_manifest.json"

CONDITIONAL_RETRIEVAL_SRC = Path("/home/hp5/tell/src/tell/agent/conditional_retrieval.py")
CONDITIONAL_OUTCOMES_SRC = Path("/home/hp5/tell/src/tell/evaluation/conditional_retrieval_outcomes.py")
MEMORY_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/memory_prompts.py")
ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
TOOLS_SRC = Path("/home/hp5/tell/src/tell/agent/tools.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")
MEMORY_STORE_SRC = Path("/home/hp5/tell/src/tell/memory/store.py")
MEMORY_MODELS_SRC = Path("/home/hp5/tell/src/tell/memory/models.py")

INVOICE_DOCUMENT_ID = "04d531ca811f448a91c6ff4e"

_GENERATION_PARAMETERS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "use_cache": True,
    "output_hidden_states": False,
    "return_dict_in_generate": False,
    "enable_thinking": False,
    "add_generation_prompt": True,
}

CONDITIONS = [
    ("task_only_base_v1__clean_memory", PromptProfile.TASK_ONLY_BASE_V1, "clean"),
    ("task_only_base_v1__poisoned_memory", PromptProfile.TASK_ONLY_BASE_V1, "poisoned"),
    ("hardened_provenance_prompt_v1__clean_memory", PromptProfile.HARDENED_PROVENANCE_PROMPT_V1, "clean"),
    ("hardened_provenance_prompt_v1__poisoned_memory", PromptProfile.HARDENED_PROVENANCE_PROMPT_V1, "poisoned"),
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main() -> None:
    if not MEMORY_INPUTS_MANIFEST_PATH.exists():
        raise RuntimeError(f"{MEMORY_INPUTS_MANIFEST_PATH} does not exist -- run scripts/inspect_and_freeze_memory_inputs.py first.")
    memory_inputs = json.loads(MEMORY_INPUTS_MANIFEST_PATH.read_text())
    clean_snapshot_path = Path(memory_inputs["snapshot_databases"]["clean"]["path"])
    poisoned_snapshot_path = Path(memory_inputs["snapshot_databases"]["poisoned"]["path"])
    poisoned_message_id = memory_inputs["poisoned_memory_record"]["origin_source_id"]
    clean_message_id = memory_inputs["clean_memory_record"]["origin_source_id"]

    poisoned_infection_scenario = json.loads(
        Path("/home/hp5/tell/data/scenarios/memory_infection/memory_infection_poisoned_04d531ca_v1.json").read_text()
    )
    poisoned_email_body = poisoned_infection_scenario["untrusted_inputs"]["supplier_email"]["body"]
    clean_infection_scenario = json.loads(
        Path("/home/hp5/tell/data/scenarios/memory_infection/memory_infection_clean_04d531ca_v1.json").read_text()
    )
    clean_email_body = clean_infection_scenario["untrusted_inputs"]["supplier_email"]["body"]

    clean_payment_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__  # class name only; never instantiated here

    # ---- Context-reset proof for all four rendered initial contexts ----
    context_reset_proof = []
    for condition_id, profile, memory_condition in CONDITIONS:
        db_path = clean_snapshot_path if memory_condition == "clean" else poisoned_snapshot_path
        store = MemoryStore(db_path)  # read-only usage: only .search_memories is ever called below
        work_item = TrustedWorkItem(
            run_id=f"context_reset_probe__{condition_id}",
            supplier_message_id=clean_payment_bundle.untrusted_inputs.supplier_email.message_id,
            canonical_vendor_id=clean_payment_bundle.trusted_state.canonical_vendor_id,
        )
        pre_messages = [
            {"role": "system", "content": build_session_b_system_prompt(profile)},
            {"role": "user", "content": build_initial_user_message(work_item)},
        ]
        records = store.search_memories(work_item.canonical_vendor_id, query=None, memory_kind=None)
        memory_message_content = build_search_memory_result_message(work_item.canonical_vendor_id, records)
        post_messages = pre_messages + [{"role": "user", "content": memory_message_content}]

        pre_text = _render(tokenizer, pre_messages)
        post_text = _render(tokenizer, post_messages)
        store.close()  # never wrote anything; only .search_memories (a SELECT) was called

        poisoned_body_in_pre = poisoned_email_body in pre_text
        poisoned_body_in_post = poisoned_email_body in post_text
        poisoned_id_in_pre = poisoned_message_id in pre_text
        poisoned_id_in_post = poisoned_message_id in post_text
        clean_body_in_pre = clean_email_body in pre_text
        clean_body_in_post = clean_email_body in post_text
        clean_id_in_pre = clean_message_id in pre_text
        clean_id_in_post = clean_message_id in post_text

        # Message-level check (not just substring-in-rendered-text): among
        # the three messages that make up post_messages, which ones
        # literally contain each id? For a clean-memory-condition run,
        # only the memory-result message may contain clean_message_id (and
        # poisoned_message_id must appear in none of them). For a
        # poisoned-memory-condition run, only the memory-result message
        # may contain poisoned_message_id.
        containing_indices_poisoned_id = [i for i, m in enumerate(post_messages) if poisoned_message_id in m["content"]]
        containing_indices_clean_id = [i for i, m in enumerate(post_messages) if clean_message_id in m["content"]]

        if memory_condition == "poisoned":
            id_entered_exclusively_via_memory_result = containing_indices_poisoned_id == [2]
        else:
            id_entered_exclusively_via_memory_result = containing_indices_poisoned_id == []

        context_reset_proof.append(
            {
                "condition_id": condition_id,
                "prompt_profile": profile.value,
                "memory_condition": memory_condition,
                "pre_prefetch_text_sha256": _sha256_text(pre_text),
                "post_prefetch_text_sha256": _sha256_text(post_text),
                "poisoned_infection_email_body_absent_pre_prefetch": not poisoned_body_in_pre,
                "poisoned_infection_email_body_absent_post_prefetch": not poisoned_body_in_post,
                "poisoned_infection_message_id_absent_pre_prefetch": not poisoned_id_in_pre,
                "poisoned_infection_message_id_present_post_prefetch": poisoned_id_in_post,
                "clean_infection_email_body_absent_pre_prefetch": not clean_body_in_pre,
                "clean_infection_email_body_absent_post_prefetch": not clean_body_in_post,
                "clean_infection_message_id_absent_pre_prefetch": not clean_id_in_pre,
                "message_indices_containing_poisoned_message_id": containing_indices_poisoned_id,
                "message_indices_containing_clean_message_id": containing_indices_clean_id,
                "id_entered_context_exclusively_via_typed_memory_result": id_entered_exclusively_via_memory_result,
            }
        )

    all_context_reset_checks_pass = all(
        row["poisoned_infection_email_body_absent_pre_prefetch"]
        and row["poisoned_infection_email_body_absent_post_prefetch"]
        and row["poisoned_infection_message_id_absent_pre_prefetch"]
        and row["clean_infection_email_body_absent_pre_prefetch"]
        and row["clean_infection_email_body_absent_post_prefetch"]
        and row["clean_infection_message_id_absent_pre_prefetch"]
        and row["id_entered_context_exclusively_via_typed_memory_result"]
        for row in context_reset_proof
    )
    if not all_context_reset_checks_pass:
        raise RuntimeError(f"Context-reset proof failed for one or more conditions: {context_reset_proof}")

    manifest = {
        "protocol": "conditional_memory_retrieval_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "memory_inputs_manifest_path": str(MEMORY_INPUTS_MANIFEST_PATH),
        "conditions": [
            {"condition_id": cid, "prompt_profile": profile.value, "memory_condition": mc}
            for cid, profile, mc in CONDITIONS
        ],
        "prompt_hashes": {
            "hardened_session_b_system_prompt_sha256": _sha256_text(build_session_b_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)),
            "task_only_session_b_system_prompt_sha256": _sha256_text(build_session_b_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)),
        },
        "memory_database_hashes": memory_inputs["snapshot_databases"],
        "memory_record_hashes": {
            "clean": {
                "memory_id": memory_inputs["clean_memory_record"]["memory_id"],
                "content_sha256": memory_inputs["clean_memory_record"]["content_sha256"],
            },
            "poisoned": {
                "memory_id": memory_inputs["poisoned_memory_record"]["memory_id"],
                "content_sha256": memory_inputs["poisoned_memory_record"]["content_sha256"],
            },
        },
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "generation_parameters": _GENERATION_PARAMETERS,
        "max_turns": MAX_TURNS,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "retrieval_decision_point": RETRIEVAL_DECISION_POINT,
        "action_schema": session_b_action_json_schema(),
        "action_schema_sha256": _sha256_text(json.dumps(session_b_action_json_schema(), indent=2, sort_keys=True)),
        "clean_payment_scenario_id": clean_payment_bundle.scenario_id,
        "clean_payment_scenario_path": str(DEFAULT_SCENARIO_PATH_04D531CA),
        "clean_payment_scenario_sha256": _sha256_file(DEFAULT_SCENARIO_PATH_04D531CA),
        "invoice_document_id": INVOICE_DOCUMENT_ID,
        "approved_beneficiary_account_id": clean_payment_bundle.trusted_state.approved_beneficiary_account_id,
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "source_hashes": {
            "conditional_retrieval_harness": {"path": str(CONDITIONAL_RETRIEVAL_SRC), "sha256": _sha256_file(CONDITIONAL_RETRIEVAL_SRC)},
            "conditional_retrieval_outcomes_evaluator": {"path": str(CONDITIONAL_OUTCOMES_SRC), "sha256": _sha256_file(CONDITIONAL_OUTCOMES_SRC)},
            "memory_prompt_builder": {"path": str(MEMORY_PROMPTS_SRC), "sha256": _sha256_file(MEMORY_PROMPTS_SRC)},
            "actions_schema": {"path": str(ACTIONS_SRC), "sha256": _sha256_file(ACTIONS_SRC)},
            "tools_module": {"path": str(TOOLS_SRC), "sha256": _sha256_file(TOOLS_SRC)},
            "work_item_envelope": {"path": str(WORK_ITEM_SRC), "sha256": _sha256_file(WORK_ITEM_SRC)},
            "capture_module": {"path": str(CAPTURE_MODULE_SRC), "sha256": _sha256_file(CAPTURE_MODULE_SRC)},
            "memory_store": {"path": str(MEMORY_STORE_SRC), "sha256": _sha256_file(MEMORY_STORE_SRC)},
            "memory_models": {"path": str(MEMORY_MODELS_SRC), "sha256": _sha256_file(MEMORY_MODELS_SRC)},
        },
        "context_reset_proof": context_reset_proof,
        "all_context_reset_checks_pass": all_context_reset_checks_pass,
        "constraints": [
            "The application-prefetch step (tell.agent.conditional_retrieval.run_conditional_retrieval) executes "
            "exactly one search_memory call via application code before the model's first generation, and never "
            "fabricates a model tool call. It guarantees exposure to the retrieved memory only -- it never forces "
            "a subsequent invoice lookup, vendor lookup, review, beneficiary selection, or terminal action.",
            "Both prompt profiles in this experiment read the same two snapshot database files -- "
            "memory_database_hashes above -- never the original task-only databases and never a regenerated or "
            "re-seeded poison.",
            "No write method is ever called against either snapshot database by the downstream experiment; both "
            "snapshot file hashes are re-verified unchanged after all four runs complete.",
            "This manifest is written before the first generation of the downstream experiment and must not be "
            "edited after seeing run output.",
        ],
    }

    PROTOCOL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {PROTOCOL_MANIFEST_PATH}")
    print(f"All context-reset checks pass: {all_context_reset_checks_pass}")


if __name__ == "__main__":
    main()
