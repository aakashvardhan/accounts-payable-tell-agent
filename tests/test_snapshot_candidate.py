"""CPU-only tests for the epoch-candidate snapshot tool. No model loaded,
no GPU, no mutation of any real training checkpoint -- all fixtures are
temp directories."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, "/home/hp5/tell/scripts")
from agent_s_v1.snapshot_candidate import SnapshotError, reject_mutable_alias, snapshot, verify_epoch_complete_marker, verify_files_present, verify_not_still_being_written  # noqa: E402


def _make_checkpoint(tmp_path: Path, *, epoch_end: bool = True, position_in_epoch: int = 0, include_files=True, name="step_000100_epoch_end") -> Path:
    d = tmp_path / name
    d.mkdir()
    if include_files:
        (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B"}))
        (d / "adapter_model.safetensors").write_bytes(b"fake-weights")
    (d / "trainer_state.json").write_text(json.dumps({"epoch": 1, "global_step": 100, "examples_seen": 1655, "position_in_epoch": position_in_epoch}))
    return d


def test_reject_mutable_alias_path():
    with pytest.raises(SnapshotError):
        reject_mutable_alias(Path("/models/agent_s/latest"))


def test_accept_named_immutable_path(tmp_path):
    reject_mutable_alias(tmp_path / "step_000100_epoch_end")  # must not raise


def test_refuse_checkpoint_missing_epoch_marker(tmp_path):
    d = _make_checkpoint(tmp_path, position_in_epoch=42, name="step_000042")
    with pytest.raises(SnapshotError):
        verify_epoch_complete_marker(d)


def test_accept_checkpoint_with_epoch_marker(tmp_path):
    d = _make_checkpoint(tmp_path)
    state = verify_epoch_complete_marker(d)
    assert state["position_in_epoch"] == 0


def test_refuse_checkpoint_missing_trainer_state(tmp_path):
    d = tmp_path / "step_000100_epoch_end"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    with pytest.raises(SnapshotError):
        verify_epoch_complete_marker(d)


def test_refuse_incomplete_adapter_files(tmp_path):
    d = _make_checkpoint(tmp_path, include_files=False)
    with pytest.raises(SnapshotError):
        verify_files_present(d)


def test_refuse_still_being_written_checkpoint(tmp_path):
    d = _make_checkpoint(tmp_path)
    # simulate an active writer appending to the weights file mid-check
    import threading

    def mutate():
        time.sleep(0.5)
        (d / "adapter_model.safetensors").write_bytes(b"fake-weights-plus-more-bytes")

    t = threading.Thread(target=mutate)
    t.start()
    with pytest.raises(SnapshotError):
        verify_not_still_being_written(d)
    t.join()


def test_accept_stable_checkpoint(tmp_path):
    d = _make_checkpoint(tmp_path)
    verify_not_still_being_written(d)  # must not raise (nothing mutates it)


def test_full_snapshot_of_valid_checkpoint(tmp_path, monkeypatch):
    import agent_s_v1.snapshot_candidate as mod

    d = _make_checkpoint(tmp_path)
    candidates_dir = tmp_path / "candidates"
    monkeypatch.setattr(mod, "CANDIDATES_DIR", candidates_dir)
    monkeypatch.setattr(mod, "STABILITY_CHECK_DELAY_SECONDS", 0.1)
    out = snapshot(str(d), "epoch1_candidate")
    assert out == candidates_dir / "epoch1_candidate"
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()
    assert (out / "provenance.json").exists()
    assert (out / "CANDIDATE.sha256").exists()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["global_step"] == 100
    assert prov["adapter_never_loaded_during_snapshot"] is True
    # source checkpoint must be untouched (never mutated)
    assert (d / "adapter_model.safetensors").read_bytes() == b"fake-weights"


def test_refuse_overwrite_existing_candidate(tmp_path, monkeypatch):
    import agent_s_v1.snapshot_candidate as mod

    d = _make_checkpoint(tmp_path)
    candidates_dir = tmp_path / "candidates"
    monkeypatch.setattr(mod, "CANDIDATES_DIR", candidates_dir)
    monkeypatch.setattr(mod, "STABILITY_CHECK_DELAY_SECONDS", 0.1)
    snapshot(str(d), "dup_candidate")
    with pytest.raises(SnapshotError):
        snapshot(str(d), "dup_candidate")
