"""Durable, session-independent watcher for the Agent-S training ->
checkpoint-selection -> held-out-evaluation pipeline. Launched detached
(setsid/nohup/disown) so it survives this Claude conversation ending.
Polls training progress, and on natural completion, runs the already-
approved remaining pipeline stages in order, writing durable status after
every step so a fresh session can resume from exactly where this one left
off. Never retries a failed stage with changed settings -- it stops and
records the failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path("/home/hp5/tell")
OUT = REPO / "results/lora_training/agent_s_v1"
TRAIN_LOG = OUT / "training.log"
STATUS_PATH = OUT / "pipeline_status.json"
HANDOFF_PATH = OUT / "NEXT_SESSION_HANDOFF.md"
WATCHER_LOG = OUT / "durable_watcher.log"
PYTHON = "/home/hp5/tell/.venv/bin/python"
POLL_SECONDS = 60


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(WATCHER_LOG, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], stage: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    log(f"STAGE START: {stage} :: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:scripts"
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=timeout, env=env)
    (OUT / f"stage_log_{stage}.stdout.txt").write_text(proc.stdout)
    (OUT / f"stage_log_{stage}.stderr.txt").write_text(proc.stderr[-20000:])
    log(f"STAGE END: {stage} :: returncode={proc.returncode}")
    return proc


def write_status(**fields) -> None:
    status = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    STATUS_PATH.write_text(json.dumps(status, indent=1, default=str))
    write_handoff(status)


def write_handoff(status: dict) -> None:
    lines = [
        "# Next-session handoff", "",
        f"Last updated: {status['updated_at']}", "",
        f"## Current stage: `{status.get('stage')}`", "",
        f"{status.get('summary', '')}", "",
        "## Completed actions", "",
    ]
    for a in status.get("completed_actions", []):
        lines.append(f"- {a}")
    lines += ["", "## Pending steps", ""]
    for p in status.get("pending_steps", []):
        lines.append(f"- {p}")
    lines += ["", "## Key artifact paths", ""]
    for p in status.get("artifact_paths", []):
        lines.append(f"- `{p}`")
    lines += ["", "## Hashes", ""]
    for k, v in status.get("hashes", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Safe resume command(s)", "", "```bash"]
    for c in status.get("resume_commands", []):
        lines.append(c)
    lines += ["```", "", "## Failures / caveats", ""]
    for f in status.get("failures", []):
        lines.append(f"- {f}")
    lines += ["", "## Claims supported vs TO MEASURE", ""]
    for c in status.get("claims", []):
        lines.append(f"- {c}")
    HANDOFF_PATH.write_text("\n".join(lines) + "\n")


def parse_training_progress() -> dict:
    if not TRAIN_LOG.exists():
        return {}
    lines = TRAIN_LOG.read_text().splitlines()
    last_step_line = next((l for l in reversed(lines) if " step " in l and "loss=" in l), None)
    done = any("TRAINING COMPLETE" in l for l in lines)
    checkpoints = sorted(p.name for p in (OUT / "checkpoints").glob("*_epoch_end")) if (OUT / "checkpoints").exists() else []
    return {"last_step_line": last_step_line, "training_complete_logged": done, "epoch_boundary_checkpoints": checkpoints}


def wait_for_training_completion() -> None:
    while True:
        progress = parse_training_progress()
        checkpoints = progress.get("epoch_boundary_checkpoints", [])
        write_status(stage="training", summary=f"Waiting for training to complete. {len(checkpoints)}/3 epoch-boundary checkpoints so far.",
                    completed_actions=[f"epoch-boundary checkpoint saved: {c}" for c in checkpoints],
                    pending_steps=["wait for training.log to report TRAINING COMPLETE and all 3 epoch-boundary checkpoints"],
                    artifact_paths=[str(TRAIN_LOG), str(OUT / "checkpoints")], hashes={}, resume_commands=[
                        f"tail -f {TRAIN_LOG}",
                        f"PYTHONPATH=src:scripts {PYTHON} scripts/agent_s_v1/train.py --resume  # only if the process died before completion",
                    ], failures=[], claims=["Training in progress -- TO MEASURE: final adapter quality"], last_training_line=progress.get("last_step_line"))
        if progress.get("training_complete_logged") and len(checkpoints) >= 3:
            log("training complete and all 3 epoch-boundary checkpoints present")
            return
        time.sleep(POLL_SECONDS)


def stage_checkpoint_selection() -> bool:
    proc = run([PYTHON, "scripts/agent_s_v1/select_checkpoint.py", "--run"], "checkpoint_selection", timeout=6 * 3600)
    ok = proc.returncode == 0 and (OUT / "frozen_adapter" / "FROZEN.sha256").exists()
    write_status(stage="checkpoint_selection", summary="Ran validation-only checkpoint selection over all 3 epoch-boundary checkpoints." if ok else "Checkpoint selection FAILED.",
                completed_actions=["training complete", "checkpoint selection run" if ok else "checkpoint selection attempted (failed)"],
                pending_steps=["agent-1 equivalence spot-check"] if ok else ["fix checkpoint selection failure -- see stage_log_checkpoint_selection.stderr.txt; do not retry with changed settings automatically"],
                artifact_paths=[str(OUT / "checkpoint_selection_report.json"), str(OUT / "frozen_adapter")],
                hashes={"frozen_adapter": (OUT / "frozen_adapter" / "FROZEN.sha256").read_text() if ok else "N/A"},
                resume_commands=[f"PYTHONPATH=src:scripts {PYTHON} scripts/agent_s_v1/select_checkpoint.py --run"],
                failures=[] if ok else [proc.stderr[-2000:]], claims=["Checkpoint selection uses only the frozen 279-record validation contract"])
    return ok


def stage_agent1_spotcheck() -> bool:
    proc = run([PYTHON, "scripts/agent_s_v1/spot_check_agent1_equivalence.py"], "agent1_spotcheck", timeout=1800)
    passed = proc.returncode == 0
    write_status(stage="agent1_spotcheck", summary=f"Agent-1 equivalence spot-check {'passed' if passed else 'FAILED -- audit amended to REGENERATION_REQUIRED'}.",
                completed_actions=["checkpoint selection complete", "agent-1 spot-check run"],
                pending_steps=["held-out test evaluation", "delayed-memory OOD evaluation", "lexical-challenge evaluation"],
                artifact_paths=[str(OUT / "agent1_equivalence_spotcheck.json"), str(OUT / "agent1_reuse_audit.json")], hashes={},
                resume_commands=[f"PYTHONPATH=src:scripts {PYTHON} scripts/agent_s_v1/spot_check_agent1_equivalence.py"],
                failures=[] if passed else ["spot-check failed; three_arm_eval will regenerate all Agent-1 outputs instead of reusing"],
                claims=["Agent-1 reuse decision is now empirically verified, not just structurally argued"])
    return True  # not fatal either way -- three_arm_eval.py reads the (possibly amended) audit itself


def stage_eval(partition: str) -> bool:
    proc = run([PYTHON, "scripts/agent_s_v1/three_arm_eval.py", "--run", "--partitions", partition], f"eval_{partition}", timeout=6 * 3600)
    ok = proc.returncode == 0 and (OUT / "three_arm_eval" / f"summary_{partition}.json").exists()
    write_status(stage=f"eval_{partition}", summary=f"Three-arm evaluation on {partition}: {'complete' if ok else 'FAILED'}.",
                completed_actions=[f"{partition} evaluation run" if ok else f"{partition} evaluation attempted (failed)"],
                pending_steps=[], artifact_paths=[str(OUT / "three_arm_eval" / f"summary_{partition}.json")], hashes={},
                resume_commands=[f"PYTHONPATH=src:scripts {PYTHON} scripts/agent_s_v1/three_arm_eval.py --run --partitions {partition}"],
                failures=[] if ok else [proc.stderr[-2000:]], claims=[])
    return ok


RELEASE_FILE = OUT / "RELEASE_HELD_STAGES"


def selection_already_done() -> bool:
    return (OUT / "checkpoint_selection_report.json").exists() and (OUT / "frozen_adapter" / "FROZEN.sha256").exists()


def wait_for_release() -> None:
    """The user asked that the Agent-1 spot-check and the 1,400-sample
    held-out Agent-S/three-arm collection NOT start until they say so.
    Blocks (polling, durable across sessions) until RELEASE_FILE exists."""
    while not RELEASE_FILE.exists():
        write_status(
            stage="HELD_after_validation_selection",
            summary=("Training and validation-only checkpoint selection are complete. The Agent-1 equivalence spot-check and the "
                    "held-out evaluation (1,400 samples: test, delayed-memory OOD, lexical challenge) are ON HOLD by user instruction "
                    f"and will not start until the file {RELEASE_FILE} exists."),
            completed_actions=["training", "checkpoint selection (validation only)"],
            pending_steps=["WAIT for user notification", f"release with: touch {RELEASE_FILE}",
                           "then: agent-1 spot-check -> eval test -> eval delayed_memory_ood -> eval lexical_challenge"],
            artifact_paths=[str(OUT / "checkpoint_selection_report.json"), str(OUT / "frozen_adapter")],
            hashes={"frozen_adapter": (OUT / "frozen_adapter" / "FROZEN.sha256").read_text() if (OUT / "frozen_adapter" / "FROZEN.sha256").exists() else "N/A"},
            resume_commands=[f"touch {RELEASE_FILE}"], failures=[],
            claims=["Adapter chosen on validation only; test/OOD/lexical untouched -- TO MEASURE held-out performance"])
        time.sleep(POLL_SECONDS)
    log("release file found; continuing with held-out stages")


def main() -> None:
    try:
        wait_for_training_completion()

        if selection_already_done():
            log("checkpoint selection already complete (report + frozen adapter exist); not re-running")
        elif not stage_checkpoint_selection():
            log("STOPPING: checkpoint selection failed")
            return

        wait_for_release()  # USER HOLD: nothing after validation selection runs until released
        stage_agent1_spotcheck()  # informational; never blocks the pipeline

        for partition in ("test", "delayed_memory_ood", "lexical_challenge"):
            if not stage_eval(partition):
                log(f"STOPPING: {partition} evaluation failed")
                return

        write_status(stage="complete", summary="Full pipeline complete: training, checkpoint selection, spot-check, and all three held-out partitions evaluated.",
                    completed_actions=["training", "checkpoint_selection", "agent1_spotcheck", "eval_test", "eval_delayed_memory_ood", "eval_lexical_challenge"],
                    pending_steps=["human review of results/lora_training/agent_s_v1/three_arm_eval/summary_*.json"],
                    artifact_paths=[str(OUT / "three_arm_eval")], hashes={"frozen_adapter": (OUT / "frozen_adapter" / "FROZEN.sha256").read_text()},
                    resume_commands=[], failures=[],
                    claims=["Three-arm held-out results now exist for test/delayed-memory-OOD/lexical-challenge",
                            "TO MEASURE: whether these results should change the demo routing recommendation"])
        log("PIPELINE COMPLETE")
    except Exception:
        tb = traceback.format_exc()
        log(f"WATCHER CRASHED: {tb}")
        write_status(stage="failed", summary="The durable watcher itself crashed unexpectedly.", completed_actions=[], pending_steps=["manual investigation required"],
                    artifact_paths=[str(WATCHER_LOG)], hashes={}, resume_commands=[], failures=[tb[-3000:]], claims=[])


if __name__ == "__main__":
    main()
