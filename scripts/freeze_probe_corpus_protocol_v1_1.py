"""Freezes the probe-activation-corpus-v1.1 protocol, before any model
inference. Requires `scripts/build_probe_corpus_samples_v1_1.py` to have
already run and passed every pre-capture validation.

Mirrors `freeze_probe_corpus_protocol.py` (v1) exactly in structure and
in every reused property (prompt profile, capture hidden-state indices,
token-position rule, generation-free capture method, model identity) --
only the source hashes, corpus paths, and the new v1.1 identifier-
generation fields differ.

Writes results/probe_dataset/v1_1/corpus_protocol_manifest_v1_1.json.
Once written, this file must not be edited after the first forward pass
of the v1.1 capture run begins.
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
from tell.probe_dataset.synthetic_ids_v1_1 import ACCOUNT_ID_HEX_LEN, ACCOUNT_ID_PREFIX, CORPUS_SALT_V1_1

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset/v1_1")
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TEMPLATE_MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"
SELECTION_MANIFEST_PATH = OUTPUT_DIR / "document_selection_manifest.json"
MANIFEST_PATH = OUTPUT_DIR / "corpus_protocol_manifest_v1_1.json"

V1_PROTOCOL_MANIFEST_PATH = Path("/home/hp5/tell/results/probe_dataset/corpus_protocol_manifest.json")

CAPTURE_HIDDEN_STATE_INDICES = (9, 18, 27, 36)
HIDDEN_SIZE = 4096

SOURCE_FILES = {
    "docile_extract": Path("/home/hp5/tell/src/tell/probe_dataset/docile_extract.py"),
    "synthetic_ids_v1_1": Path("/home/hp5/tell/src/tell/probe_dataset/synthetic_ids_v1_1.py"),
    "text_block": Path("/home/hp5/tell/src/tell/probe_dataset/text_block.py"),
    "templates_v1_1": Path("/home/hp5/tell/src/tell/probe_dataset/templates_v1_1.py"),
    "sample_builder_v1_1": Path("/home/hp5/tell/src/tell/probe_dataset/sample_builder_v1_1.py"),
    "actions_schema": Path("/home/hp5/tell/src/tell/agent/actions.py"),
    "loop_prompts": Path("/home/hp5/tell/src/tell/agent/loop_prompts.py"),
    "prompt_profiles": Path("/home/hp5/tell/src/tell/agent/prompt_profiles.py"),
    "work_item": Path("/home/hp5/tell/src/tell/agent/work_item.py"),
    "build_probe_corpus_samples_v1_1": Path("/home/hp5/tell/scripts/build_probe_corpus_samples_v1_1.py"),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    for p in (RENDERED_SAMPLES_PATH, LABELS_PATH, SPLIT_MANIFEST_PATH, TEMPLATE_MANIFEST_PATH, SELECTION_MANIFEST_PATH):
        if not p.exists():
            raise RuntimeError(f"{p} does not exist -- run scripts/build_probe_corpus_samples_v1_1.py first.")
    if not V1_PROTOCOL_MANIFEST_PATH.exists():
        raise RuntimeError(f"{V1_PROTOCOL_MANIFEST_PATH} (v1's frozen protocol) does not exist.")

    n_samples = sum(1 for _ in RENDERED_SAMPLES_PATH.open())
    n_labels = sum(1 for _ in LABELS_PATH.open())
    if n_samples != 200 or n_labels != 200:
        raise RuntimeError(f"Expected 200 rendered samples and 200 labels, found {n_samples} / {n_labels}")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    tokenizer_class = type(tokenizer).__name__
    model_class = Qwen3ForCausalLM.__name__  # class name only; never instantiated here

    rendered_prompt = build_loop_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)
    v1_protocol = json.loads(V1_PROTOCOL_MANIFEST_PATH.read_text())

    manifest = {
        "protocol": "probe_activation_corpus_v1_1",
        "supersedes": "probe_activation_corpus_v1 (preserved unmodified, not overwritten)",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "reused_from_v1": {
            "prompt_profile": True,
            "document_selection_and_splits": True,
            "template_families_and_hard_negative_pairing": True,
            "capture_hidden_state_indices": True,
            "model_revision": True,
            "token_position_rule": True,
        },
        "v1_protocol_manifest_sha256": _sha256_file(V1_PROTOCOL_MANIFEST_PATH),
        "v1_rendered_task_only_system_prompt_sha256_matches": v1_protocol["rendered_task_only_system_prompt_sha256"] == _sha256_text(rendered_prompt),
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
        "hidden_size": HIDDEN_SIZE,
        "token_position_rule": "final non-padding prompt token, immediately before the next model action (no generation)",
        "capture_method": "forward pass only (output_hidden_states=True, use_cache=False, torch.inference_mode()); no generation, no model actions",
        "generation_free": True,
        "n_samples": n_samples,
        "n_labels": n_labels,
        "account_identifier_generation": {
            "corpus_salt": CORPUS_SALT_V1_1,
            "account_id_prefix": ACCOUNT_ID_PREFIX,
            "account_id_hex_length": ACCOUNT_ID_HEX_LEN,
            "roles": ["approved", "unauthorized", "obsolete"],
            "derivation": "sha256(f'{salt}:{docid}:{role}').hexdigest()[:12].upper(), prefixed by account_id_prefix",
            "forbidden_words_checked": ["ATTACKER", "MALICIOUS", "POISON", "FAKE"],
            "note": (
                "Replaces v1's single fixed literal SIM-ATTACKER-ACCT-0001 (synthetic_ids.py, "
                "ATTACKER_BENEFICIARY_ACCOUNT_ID, left unmodified for v1's own artifacts) with three "
                "per-document, deterministically derived, identically-formatted identifiers -- see "
                "src/tell/probe_dataset/synthetic_ids_v1_1.py."
            ),
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
            "No generation ever occurs during capture -- every forward pass is output_hidden_states=True, "
            "use_cache=False, and no model action is produced or parsed.",
            "Only the four selected hidden-state indices (9, 18, 27, 36) and only the final non-padding prompt "
            "token's vector are ever persisted.",
            "This manifest is written before the first v1.1 capture forward pass and must not be edited after "
            "seeing capture output.",
            "v1's own artifacts (results/probe_dataset/*.jsonl, *.safetensors, corpus_protocol_manifest.json) "
            "are never read for writing, only for the reused-selection hash check above.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
