"""CPU-only. Copies ONE completed, immutable training checkpoint's
inference-only adapter files into a new, immutable, hashed candidate
directory -- without ever loading the adapter, and without mutating the
training checkpoint it copies from. Intended for the epoch-1 checkpoint
(or any other epoch-boundary checkpoint) once training has moved past it,
so it can be smoke-tested later without touching the still-growing
`checkpoints/` directory or an actively-written checkpoint.

Usage:
    .venv/bin/python scripts/agent_s_v1/snapshot_candidate.py \\
        --checkpoint results/lora_training/agent_s_v1/checkpoints/step_XXXXXX_epoch_end \\
        --name epoch1_candidate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

REPO = Path("/home/hp5/tell")
OUT_DIR = REPO / "results/lora_training/agent_s_v1"
CANDIDATES_DIR = OUT_DIR / "candidates"
INFERENCE_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json",
                   "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json")
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
STABILITY_CHECK_DELAY_SECONDS = 2.0


class SnapshotError(ValueError):
    pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha(p.read_bytes())


def reject_mutable_alias(path: Path) -> None:
    if "latest" in {part.lower() for part in path.parts}:
        raise SnapshotError(f"refusing mutable alias path (contains 'latest'): {path}")
    if path.is_symlink():
        raise SnapshotError(f"refusing a symlinked checkpoint path (not an immutable, named directory): {path}")


def verify_epoch_complete_marker(checkpoint_dir: Path) -> dict:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        raise SnapshotError(f"no trainer_state.json in {checkpoint_dir}; cannot confirm this is a real checkpoint")
    state = json.loads(state_path.read_text())
    is_epoch_end = checkpoint_dir.name.endswith("_epoch_end") and state.get("position_in_epoch") == 0
    if not is_epoch_end:
        raise SnapshotError(f"{checkpoint_dir} has no epoch-complete marker (expected an '_epoch_end'-suffixed directory "
                           f"with trainer_state.json position_in_epoch==0; found position_in_epoch={state.get('position_in_epoch')})")
    return state


def verify_files_present(checkpoint_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (checkpoint_dir / f).exists()]
    if missing:
        raise SnapshotError(f"{checkpoint_dir} is missing required files: {missing}")


def verify_not_still_being_written(checkpoint_dir: Path) -> None:
    """Two-pass stability check: record size+mtime for every file, wait a
    short bounded interval, re-check. Any change means an active writer,
    so this refuses rather than snapshotting a partial file."""
    def snapshot():
        return {f.name: (f.stat().st_size, f.stat().st_mtime) for f in checkpoint_dir.iterdir() if f.is_file()}

    before = snapshot()
    time.sleep(STABILITY_CHECK_DELAY_SECONDS)
    after = snapshot()
    changed = [name for name in before if before.get(name) != after.get(name)]
    new_files = set(after) - set(before)
    if changed or new_files:
        raise SnapshotError(f"{checkpoint_dir} changed during the {STABILITY_CHECK_DELAY_SECONDS}s stability check "
                           f"(changed={changed}, new={sorted(new_files)}); refusing to snapshot a possibly-mid-write checkpoint")


def snapshot(checkpoint_path: str, name: str, *, labels: tuple[str, ...] = ()) -> Path:
    checkpoint_dir = Path(checkpoint_path)
    reject_mutable_alias(checkpoint_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise SnapshotError(f"checkpoint directory does not exist: {checkpoint_dir}")

    trainer_state = verify_epoch_complete_marker(checkpoint_dir)
    verify_files_present(checkpoint_dir)
    verify_not_still_being_written(checkpoint_dir)

    candidate_dir = CANDIDATES_DIR / name
    if candidate_dir.exists():
        raise SnapshotError(f"refusing to overwrite existing candidate directory: {candidate_dir}")
    candidate_dir.mkdir(parents=True)

    copied = []
    for fname in INFERENCE_FILES:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, candidate_dir / fname)  # never touches the source checkpoint
            copied.append(fname)

    resolved_config_path = OUT_DIR / "resolved_training_config.json"
    sampler_manifest_path = OUT_DIR / "epoch_manifests" / "manifest.json"
    resolved_config = json.loads(resolved_config_path.read_text()) if resolved_config_path.exists() else {}
    provenance = {
        "labels": list(labels),
        "source_checkpoint": str(checkpoint_dir), "epoch": trainer_state["epoch"], "global_step": trainer_state["global_step"],
        "examples_seen": trainer_state["examples_seen"],
        "base_model_repo_id": resolved_config.get("model_repo_id"), "base_model_revision": resolved_config.get("model_revision"),
        "tokenizer_revision": resolved_config.get("tokenizer_revision"),
        "resolved_training_config_sha256": sha_file(resolved_config_path) if resolved_config_path.exists() else None,
        "sampler_manifest_sha256": sha_file(sampler_manifest_path) if sampler_manifest_path.exists() else None,
        "adapter_config_sha256": sha_file(candidate_dir / "adapter_config.json") if (candidate_dir / "adapter_config.json").exists() else None,
        "adapter_weights_sha256": sha_file(candidate_dir / "adapter_model.safetensors") if (candidate_dir / "adapter_model.safetensors").exists() else None,
        "copied_files": copied, "snapshot_created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "adapter_never_loaded_during_snapshot": True,
    }
    (candidate_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))
    hashes = "".join(f"{sha_file(p)}  {p.name}\n" for p in sorted(candidate_dir.iterdir()) if p.suffix != ".sha256")
    (candidate_dir / "CANDIDATE.sha256").write_text(hashes)
    print(json.dumps({"candidate_dir": str(candidate_dir), "provenance": provenance}, indent=1))
    return candidate_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", action="append", default=[])
    args = ap.parse_args()
    snapshot(args.checkpoint, args.name, labels=tuple(args.label))
