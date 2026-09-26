"""Freezes the LoRA corpus v1 manifest (corpus hashes, split manifest,
prompt profile, tokenizer revision, model revision, action-schema hash)
before any GPU training. Requires scripts/build_lora_corpus_samples.py to
have already run and passed every validation.

Writes results/lora_dataset/v1/lora_corpus_v1_manifest.json. Once
written, this file must not be edited after training begins.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer, Qwen3ForCausalLM

from tell.agent.actions import action_json_schema
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH
from tell.agent.loop_prompts import build_loop_system_prompt
from tell.agent.prompt_profiles import PromptProfile
from tell.lora_dataset.synthetic_ids import ACCOUNT_ID_HEX_LEN, ACCOUNT_ID_PREFIX, CORPUS_SALT_LORA_V1

OUTPUT_DIR = Path("/home/hp5/tell/results/lora_dataset/v1")
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
SELECTION_MANIFEST_PATH = OUTPUT_DIR / "document_selection_manifest.json"
MANIFEST_PATH = OUTPUT_DIR / "lora_corpus_v1_manifest.json"

SOURCE_FILES = {
    "docile_extract": Path("/home/hp5/tell/src/tell/probe_dataset/docile_extract.py"),
    "synthetic_ids": Path("/home/hp5/tell/src/tell/lora_dataset/synthetic_ids.py"),
    "templates": Path("/home/hp5/tell/src/tell/lora_dataset/templates.py"),
    "sample_builder": Path("/home/hp5/tell/src/tell/lora_dataset/sample_builder.py"),
    "masking": Path("/home/hp5/tell/src/tell/lora_dataset/masking.py"),
    "actions_schema": Path("/home/hp5/tell/src/tell/agent/actions.py"),
    "loop_prompts": Path("/home/hp5/tell/src/tell/agent/loop_prompts.py"),
    "prompt_profiles": Path("/home/hp5/tell/src/tell/agent/prompt_profiles.py"),
    "work_item": Path("/home/hp5/tell/src/tell/agent/work_item.py"),
    "tools": Path("/home/hp5/tell/src/tell/agent/tools.py"),
    "build_lora_corpus_samples": Path("/home/hp5/tell/scripts/build_lora_corpus_samples.py"),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    for p in (RENDERED_SAMPLES_PATH, LABELS_PATH, SPLIT_MANIFEST_PATH, TEMPLATE_MANIFEST_PATH, SELECTION_MANIFEST_PATH):
        if not p.exists():
            raise RuntimeError(f"{p} does not exist -- run scripts/build_lora_corpus_samples.py first.")

    n_samples = sum(1 for _ in RENDERED_SAMPLES_PATH.open())
    n_labels = sum(1 for _ in LABELS_PATH.open())
    if n_samples != 600 or n_labels != 600:
        raise RuntimeError(f"Expected 600 rendered samples and 600 labels, found {n_samples} / {n_labels}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__  # class name only; never instantiated here

    rendered_prompt = build_loop_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)

    manifest = {
        "protocol": "lora_corpus_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prompt_profile": PromptProfile.TASK_ONLY_BASE_V1.value,
        "rendered_task_only_system_prompt_sha256": _sha256_text(rendered_prompt),
        "action_schema_sha256": _sha256_text(json.dumps(action_json_schema(), indent=2, sort_keys=True)),
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "tokenizer_revision": PINNED_MODEL_REVISION,
        "n_samples": n_samples,
        "n_labels": n_labels,
        "account_identifier_generation": {
            "corpus_salt": CORPUS_SALT_LORA_V1,
            "account_id_prefix": ACCOUNT_ID_PREFIX,
            "account_id_hex_length": ACCOUNT_ID_HEX_LEN,
            "roles": ["approved", "unauthorized", "obsolete"],
        },
        "sft_representation": {
            "completion_format": "compact JSON (json.dumps(gold_action, separators=(',', ':'))), no chain-of-thought",
            "eos_token": "<|im_end|>",
            "masking_rule": "prompt tokens (system+user turns, incl. the assistant preamble emitted by add_generation_prompt=True) are -100; only completion+EOS tokens contribute to loss",
            "max_seq_len": 6144,
            "overlength_policy": "rejected outright, never truncated (0 rejections occurred)",
        },
        "rendered_samples_path": str(RENDERED_SAMPLES_PATH),
        "rendered_samples_sha256": _sha256_file(RENDERED_SAMPLES_PATH),
        "labels_path": str(LABELS_PATH),
        "labels_sha256": _sha256_file(LABELS_PATH),
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST_PATH),
        "template_manifest_sha256": _sha256_file(TEMPLATE_MANIFEST_PATH),
        "document_selection_manifest_sha256": _sha256_file(SELECTION_MANIFEST_PATH),
        "source_hashes": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in SOURCE_FILES.items()},
        "constraints": [
            "Every sample renders under PromptProfile.TASK_ONLY_BASE_V1 only.",
            "Gold actions are deterministic from trusted scenario state and observed tool results, never free-form model generation.",
            "This manifest is written before any LoRA training and must not be edited after seeing training output.",
            "The probe corpus (v1/v1.1) and its trained probe/threshold/reports are never read or modified by this corpus.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
