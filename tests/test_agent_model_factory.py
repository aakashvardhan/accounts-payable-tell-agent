"""CPU-only tests for the adapter configuration + lazy loader factory.
Run with CUDA_VISIBLE_DEVICES="" (also enforced structurally: this file
never imports torch, transformers, or peft, and the factory under test
performs no import-time model/CUDA work by construction)."""
from __future__ import annotations

import json
import time

import pytest

from tell.safety.agent_model_factory import AgentModelFactory, FAIL_CLOSED_MARKER, is_checkpoint_still_being_written
from tell.safety.agent_s_config import ADAPTER_PATH_ENV_VAR, AdapterPathError, AgentSRuntimeConfig, OPERATIONAL_THRESHOLD, SCIENTIFIC_THRESHOLD

REAL_SMOKE_ADAPTER = "/home/hp5/tell/results/lora_training/smoke_v1/adapter"


class _FakeRuntime:
    """Deterministic stand-in for AdapterAwareRuntime -- no model, no
    tensors, no torch import at all."""

    def __init__(self):
        self.adapter_info = None
        self.attached_path = None

    def attach_agent_s_adapter(self, path):
        self.adapter_info = {"path": str(path)}
        self.attached_path = str(path)
        return self.adapter_info

    def render_chat_prompt(self, messages, *, enable_thinking=False):
        return "PROMPT"

    def tokenize(self, text):
        return {"input_ids": [[1, 2, 3]]}


def _config(adapter_path: str = REAL_SMOKE_ADAPTER, **overrides) -> AgentSRuntimeConfig:
    return AgentSRuntimeConfig(agent_s_adapter_path=adapter_path, **overrides)


# --- thresholds ----------------------------------------------------------------------


def test_operational_threshold_value_unchanged():
    assert OPERATIONAL_THRESHOLD == 0.5134634443863925
    assert SCIENTIFIC_THRESHOLD == 0.1708046793937683


# --- adapter path validation -----------------------------------------------------------


def test_config_rejects_latest_alias_path():
    c = _config(adapter_path="/models/agent_s/latest")
    with pytest.raises(AdapterPathError):
        c.reject_mutable_or_incomplete_path()


def test_config_accepts_real_frozen_adapter_path():
    c = _config()
    c.validate_adapter_directory_complete()  # must not raise
    c.check_adapter_base_model_compatibility()  # must not raise


def test_config_rejects_missing_adapter_directory():
    c = _config(adapter_path="/tmp/does/not/exist/adapter")
    with pytest.raises(AdapterPathError):
        c.validate_adapter_directory_complete()


def test_config_rejects_incomplete_adapter_directory(tmp_path):
    d = tmp_path / "incomplete_adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")  # missing adapter_model.safetensors
    c = _config(adapter_path=str(d))
    with pytest.raises(AdapterPathError):
        c.validate_adapter_directory_complete()


def test_config_rejects_base_model_mismatch(tmp_path):
    d = tmp_path / "wrong_base_adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "some/other/model"}))
    (d / "adapter_model.safetensors").write_bytes(b"fake")
    c = _config(adapter_path=str(d))
    c.validate_adapter_directory_complete()  # structurally complete
    with pytest.raises(AdapterPathError):
        c.check_adapter_base_model_compatibility()


def test_env_var_overrides_adapter_path(monkeypatch, tmp_path):
    d = tmp_path / "env_adapter"
    d.mkdir()
    monkeypatch.setenv(ADAPTER_PATH_ENV_VAR, str(d))
    c = AgentSRuntimeConfig.from_env_or_default(default_adapter_path="/should/not/be/used")
    assert c.agent_s_adapter_path == str(d)


def test_no_env_var_uses_default(monkeypatch):
    monkeypatch.delenv(ADAPTER_PATH_ENV_VAR, raising=False)
    c = AgentSRuntimeConfig.from_env_or_default(default_adapter_path=REAL_SMOKE_ADAPTER)
    assert c.agent_s_adapter_path == REAL_SMOKE_ADAPTER


def test_still_being_written_detection(tmp_path):
    d = tmp_path / "hot_checkpoint"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"partial")
    assert is_checkpoint_still_being_written(d, stability_window=5.0) is True
    old = time.time() - 3600
    import os

    os.utime(d / "adapter_model.safetensors", (old, old))
    assert is_checkpoint_still_being_written(d, stability_window=5.0) is False


# --- lazy loader: no import-time model/CUDA work ----------------------------------------


def test_factory_construction_does_no_io():
    """Constructing AgentModelFactory (even with a nonexistent adapter
    path) must not touch the filesystem or raise -- validation only
    happens when explicitly requested."""
    c = _config(adapter_path="/definitely/not/a/real/path")
    factory = AgentModelFactory(c, runtime_builder=_FakeRuntime)
    assert factory is not None  # constructing alone must not raise


def test_get_agent_1_does_not_require_adapter():
    c = _config(adapter_path="/definitely/not/a/real/path")
    factory = AgentModelFactory(c, runtime_builder=_FakeRuntime)
    model = factory.get_agent_1()
    assert isinstance(model, _FakeRuntime)


def test_get_agent_s_with_missing_adapter_raises_no_silent_fallback():
    """The critical no-silent-fallback guarantee: a missing/incomplete
    Agent-S adapter must raise, never quietly hand back an Agent-1-
    equivalent model."""
    c = _config(adapter_path="/definitely/not/a/real/path")
    factory = AgentModelFactory(c, runtime_builder=_FakeRuntime)
    with pytest.raises(AdapterPathError):
        factory.get_agent_s()


def test_get_agent_s_with_real_frozen_adapter_succeeds():
    c = _config()
    factory = AgentModelFactory(c, runtime_builder=_FakeRuntime)
    model = factory.get_agent_s()
    assert model.attached_path == REAL_SMOKE_ADAPTER


def test_administratively_disabled_adapter_raises_even_if_files_exist():
    c = _config(adapter_enabled=False)
    factory = AgentModelFactory(c, runtime_builder=_FakeRuntime)
    with pytest.raises(AdapterPathError):
        factory.get_agent_s()


def test_get_agent_s_on_still_writing_checkpoint_fails_closed(tmp_path):
    d = tmp_path / "writing_now"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B"}))
    (d / "adapter_model.safetensors").write_bytes(b"partial")  # fresh mtime
    c = _config(adapter_path=str(d))
    factory = AgentModelFactory(c, runtime_builder=_FakeRuntime)
    with pytest.raises(AdapterPathError):
        factory.get_agent_s()
