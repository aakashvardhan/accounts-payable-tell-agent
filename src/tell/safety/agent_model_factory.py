"""Lazy adapter-aware model factory. Importing this module does nothing --
no model load, no CUDA call, no filesystem write. `AgentModelFactory` only
touches the filesystem (small JSON reads) when explicitly asked to
validate, and only loads a model/adapter when `.get_agent_1()` /
`.get_agent_s()` is actually called.

No silent fallback: if the operational router selects Agent S and the
adapter is missing, incomplete, or still being written by an active
training process, `.get_agent_s()` raises -- it never quietly returns an
Agent-1-equivalent model instead. A high Tell score with no usable Agent-S
adapter is a fail-closed condition for the caller to handle explicitly
(see `FAIL_CLOSED_MARKER`), not a reason to continue as Agent 1.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from tell.safety.agent_s_config import AdapterPathError, AgentSRuntimeConfig

FAIL_CLOSED_MARKER = "fail_closed_no_usable_agent_s_adapter"
STILL_WRITING_STABILITY_WINDOW_SECONDS = 5.0


class ModelHandle(Protocol):
    """What `.get_agent_1()`/`.get_agent_s()` return -- narrow on purpose
    (render/tokenize/generate only), so a caller cannot reach into a
    lower-level API this factory doesn't intend to expose."""

    def render_chat_prompt(self, messages: list[dict], *, enable_thinking: bool = False) -> str: ...
    def tokenize(self, text: str): ...


def is_checkpoint_still_being_written(adapter_dir: Path, *, stability_window: float = STILL_WRITING_STABILITY_WINDOW_SECONDS) -> bool:
    """True if any file under `adapter_dir` was modified more recently
    than `stability_window` seconds ago -- a live filesystem poll, kept
    separate from `AgentSRuntimeConfig` (which does no time-dependent I/O)
    so the config's validation stays deterministic and testable."""
    if not adapter_dir.exists():
        return False
    now = time.time()
    for f in adapter_dir.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) < stability_window:
            return True
    return False


class AgentModelFactory:
    """Constructing this does no I/O beyond storing the config. Every
    method that touches the filesystem or GPU is called explicitly."""

    def __init__(self, config: AgentSRuntimeConfig, *, runtime_builder=None) -> None:
        """`runtime_builder`, if given, is a zero-arg callable returning an
        object with `AdapterAwareRuntime`'s interface -- the injection seam
        unit tests use instead of constructing a real GPU-backed runtime."""
        self._config = config
        self._runtime_builder = runtime_builder
        self._runtime = None  # lazily constructed on first real use

    def validate_agent_s_adapter_ready(self) -> None:
        """Raises AdapterPathError if the adapter cannot be safely used
        right now -- missing, incomplete, mid-write, or base-model-
        mismatched. Never returns a boolean silently ignored by a caller
        that forgets to check it."""
        self._config.validate_adapter_directory_complete()
        self._config.check_adapter_base_model_compatibility()
        adapter_dir = Path(self._config.agent_s_adapter_path)
        if is_checkpoint_still_being_written(adapter_dir):
            raise AdapterPathError(f"adapter directory {adapter_dir} has files modified within the last "
                                   f"{STILL_WRITING_STABILITY_WINDOW_SECONDS}s; refusing to load a possibly-mid-write checkpoint")

    def _build_runtime(self):
        if self._runtime is None:
            if self._runtime_builder is not None:
                self._runtime = self._runtime_builder()
            else:  # pragma: no cover - real GPU path, exercised only in the later GPU phase
                from tell.safety.adapter_runtime import AdapterAwareRuntime

                self._runtime = AdapterAwareRuntime()
                self._runtime.load()
        return self._runtime

    def get_agent_1(self) -> ModelHandle:
        return self._build_runtime()

    def get_agent_s(self) -> ModelHandle:
        if not self._config.adapter_enabled:
            raise AdapterPathError("Agent-S adapter is administratively disabled (adapter_enabled=False) -- refusing to load it")
        self.validate_agent_s_adapter_ready()  # raises; never caught-and-ignored here
        runtime = self._build_runtime()
        if getattr(runtime, "adapter_info", None) is None:
            runtime.attach_agent_s_adapter(self._config.agent_s_adapter_path)
        return runtime


__all__ = ["AgentModelFactory", "FAIL_CLOSED_MARKER", "is_checkpoint_still_being_written"]
