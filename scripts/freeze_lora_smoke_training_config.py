"""Freezes the smoke-training configuration under configs/lora/, before
any GPU training. Requires the smoke/diagnostic subset manifests to
already exist.

Writes configs/lora/smoke_v1_config.json.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CORPUS_MANIFEST_PATH = Path("/home/hp5/tell/results/lora_dataset/v1/lora_corpus_v1_manifest.json")
SMOKE_SUBSET_PATH = Path("/home/hp5/tell/results/lora_training/smoke_v1/smoke_subset_manifest.json")
DIAGNOSTIC_SUBSET_PATH = Path("/home/hp5/tell/results/lora_training/smoke_v1/diagnostic_subset_manifest.json")

OUTPUT_PATH = Path("/home/hp5/tell/configs/lora/smoke_v1_config.json")

ADAPTER_OUTPUT_DIR = "/home/hp5/tell/results/lora_training/smoke_v1/adapter"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": "CAUSAL_LM",
    "bias": "none",
}

TRAINING_CONFIG = {
    "dtype": "bfloat16",
    "device": "cuda:0 (single local CUDA device)",
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "n_optimizer_steps": 20,
    "learning_rate": 2e-4,
    "optimizer": "AdamW (default betas/eps)",
    "lr_schedule": "constant",
    "gradient_checkpointing": True,
    "seed": 20260101,
    "max_seq_len": 6144,
    "checkpoint_upload": "none (local save only, no external experiment tracking service)",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for p in (CORPUS_MANIFEST_PATH, SMOKE_SUBSET_PATH, DIAGNOSTIC_SUBSET_PATH):
        if not p.exists():
            raise RuntimeError(f"{p} does not exist -- build/select the corpus and smoke/diagnostic subsets first.")

    config = {
        "config_id": "lora_smoke_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "corpus_manifest_sha256": _sha256_file(CORPUS_MANIFEST_PATH),
        "smoke_subset_manifest_sha256": _sha256_file(SMOKE_SUBSET_PATH),
        "diagnostic_subset_manifest_sha256": _sha256_file(DIAGNOSTIC_SUBSET_PATH),
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "adapter_output_dir": ADAPTER_OUTPUT_DIR,
        "package_versions": {"note": "recorded separately in results/lora_training/smoke_v1/environment_report.json by scripts/run_lora_smoke_training.py at run time"},
        "constraints": [
            "No bitsandbytes, no vLLM, no quantized training.",
            "Base-model parameters frozen; only adapter parameters trainable.",
            "No automatic checkpoint upload to any external service.",
            "This is a smoke run only -- not full LoRA training.",
            "Does not train, retrain, or modify the Tell probe (results/probe_dataset, results/probe_training) in any way.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
