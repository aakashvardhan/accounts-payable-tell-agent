"""Freezes the probe-activation-corpus-v1 protocol (Section 1), before
any model inference. Requires `scripts/build_probe_corpus_samples.py` to
have already run and passed every pre-capture validation.

Records: prompt profile, rendered system-prompt hash, model/tokenizer
identity, capture hidden-state indices (9/18/27/36 -- embedding index 0
is deliberately excluded, see Section 1 of the corpus spec), token-
position rule, generation-free capture method, action-schema hash,
source hashes of every module this corpus depends on, and the corpus
sample/label file hashes.

Writes results/probe_dataset/corpus_protocol_manifest.json. Once
written, this file must not be edited after the first forward pass of
the capture run begins.
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

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset")
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
SELECTION_MANIFEST_PATH = OUTPUT_DIR / "document_selection_manifest.json"
MANIFEST_PATH = OUTPUT_DIR / "corpus_protocol_manifest.json"

CAPTURE_HIDDEN_STATE_INDICES = (9, 18, 27, 36)
HIDDEN_SIZE = 4096

SOURCE_FILES = {
    "docile_extract": Path("/home/hp5/tell/src/tell/probe_dataset/docile_extract.py"),
    "synthetic_ids": Path("/home/hp5/tell/src/tell/probe_dataset/synthetic_ids.py"),
    "text_block": Path("/home/hp5/tell/src/tell/probe_dataset/text_block.py"),
    "templates": Path("/home/hp5/tell/src/tell/probe_dataset/templates.py"),
    "sample_builder": Path("/home/hp5/tell/src/tell/probe_dataset/sample_builder.py"),
    "actions_schema": Path("/home/hp5/tell/src/tell/agent/actions.py"),
    "loop_prompts": Path("/home/hp5/tell/src/tell/agent/loop_prompts.py"),
    "prompt_profiles": Path("/home/hp5/tell/src/tell/agent/prompt_profiles.py"),
    "work_item": Path("/home/hp5/tell/src/tell/agent/work_item.py"),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    for p in (RENDERED_SAMPLES_PATH, LABELS_PATH, SPLIT_MANIFEST_PATH, TEMPLATE_MANIFEST_PATH):
        if not p.exists():
            raise RuntimeError(f"{p} does not exist -- run scripts/build_probe_corpus_samples.py first.")

    n_samples = sum(1 for _ in RENDERED_SAMPLES_PATH.open())
    n_labels = sum(1 for _ in LABELS_PATH.open())
    if n_samples != 200 or n_labels != 200:
        raise RuntimeError(f"Expected 200 rendered samples and 200 labels, found {n_samples} / {n_labels}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__  # class name only; never instantiated here

    rendered_prompt = build_loop_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)

    manifest = {
        "protocol": "probe_activation_corpus_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prompt_profile": PromptProfile.TASK_ONLY_BASE_V1.value,
        "rendered_task_only_system_prompt_sha256": _sha256_text(rendered_prompt),
        "action_schema_sha256": _sha256_text(json.dumps(action_json_schema(), indent=2, sort_keys=True)),
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_snapshot_path": str(PINNED_SNAPSHOT_PATH),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "capture_hidden_state_indices": list(CAPTURE_HIDDEN_STATE_INDICES),
        "capture_excludes_embedding_index_0": True,
        "capture_excludes_embedding_index_0_reason": (
            "Index 0 is the raw embedding of the final prompt token; it has already been shown (see "
            "tell.detector.capture's docstring and every prior activation-comparison report in this repo) to be "
            "identical whenever the final token is identical, regardless of context content -- it does not "
            "represent contextual processing and would not be a useful probe feature."
        ),
        "hidden_size": HIDDEN_SIZE,
        "token_position_rule": "final non-padding prompt token, immediately before the next model action (no generation)",
        "capture_method": "forward pass only (output_hidden_states=True, use_cache=False, torch.inference_mode()); no generation, no model actions",
        "generation_free": True,
        "n_samples": n_samples,
        "n_labels": n_labels,
        "rendered_samples_path": str(RENDERED_SAMPLES_PATH),
        "rendered_samples_sha256": _sha256_file(RENDERED_SAMPLES_PATH),
        "labels_path": str(LABELS_PATH),
        "labels_sha256": _sha256_file(LABELS_PATH),
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST_PATH),
        "template_manifest_sha256": _sha256_file(TEMPLATE_MANIFEST_PATH),
        "document_selection_manifest_sha256": _sha256_file(SELECTION_MANIFEST_PATH),
        "source_hashes": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in SOURCE_FILES.items()},
        "constraints": [
            "Every sample renders under PromptProfile.TASK_ONLY_BASE_V1 only -- no hardened-prompt activation is "
            "ever mixed into this corpus.",
            "No generation ever occurs during capture -- every forward pass is output_hidden_states=True, "
            "use_cache=False, and no model action is produced or parsed.",
            "Only the four selected hidden-state indices (9, 18, 27, 36) and only the final non-padding prompt "
            "token's vector are ever persisted -- no full-sequence tensor is ever written to disk.",
            "This manifest is written before the first capture forward pass and must not be edited after seeing "
            "capture output.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
