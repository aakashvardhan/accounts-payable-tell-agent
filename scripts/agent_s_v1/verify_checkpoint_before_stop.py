"""CPU-only, read-only. Verifies an epoch-boundary checkpoint is complete
and stable BEFORE the trainer is stopped. Refuses (non-zero exit, no
stopping recommended) on any incompleteness -- never assumes a checkpoint
is good merely because its directory exists."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "scripts"))
from agent_s_v1.snapshot_candidate import verify_not_still_being_written  # noqa: E402

REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "rng_state.pt", "trainer_state.json")


class CheckpointNotReady(Exception):
    pass


def verify(checkpoint_dir: Path) -> dict:
    if not checkpoint_dir.exists():
        raise CheckpointNotReady(f"checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.name.endswith("_epoch_end"):
        raise CheckpointNotReady(f"{checkpoint_dir} is not an epoch-boundary checkpoint (name must end in _epoch_end)")

    missing = [f for f in REQUIRED_FILES if not (checkpoint_dir / f).exists()]
    if missing:
        raise CheckpointNotReady(f"missing files: {missing}")

    state = json.loads((checkpoint_dir / "trainer_state.json").read_text())
    if state.get("position_in_epoch") != 0:
        raise CheckpointNotReady(f"trainer_state.json position_in_epoch={state.get('position_in_epoch')}, expected 0 (epoch boundary)")
    for key in ("epoch", "global_step", "examples_seen", "max_seq_len", "grad_accum", "learning_rate", "seed"):
        if key not in state:
            raise CheckpointNotReady(f"trainer_state.json missing expected key: {key}")

    verify_not_still_being_written(checkpoint_dir)

    abort_path = REPO / "results/lora_training/agent_s_v1/abort_report.json"
    if abort_path.exists():
        raise CheckpointNotReady(f"an abort_report.json exists ({abort_path}) -- training hit an error condition")

    for f in (checkpoint_dir / "adapter_model.safetensors", checkpoint_dir / "optimizer.pt"):
        if f.stat().st_size == 0:
            raise CheckpointNotReady(f"{f} is zero bytes")

    return {"checkpoint_dir": str(checkpoint_dir), "epoch": state["epoch"], "global_step": state["global_step"],
           "examples_seen": state["examples_seen"], "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "all_required_files_present": True, "stable": True, "no_abort_marker": True, "ready_to_stop": True}


if __name__ == "__main__":
    ckpt = Path(sys.argv[1])
    try:
        report = verify(ckpt)
        print(json.dumps(report, indent=1))
        sys.exit(0)
    except CheckpointNotReady as e:
        print(json.dumps({"ready_to_stop": False, "reason": str(e)}, indent=1))
        sys.exit(1)
