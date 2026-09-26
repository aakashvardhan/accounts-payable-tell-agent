"""Versioned, typed runtime configuration for the Agent-1/Agent-S model
selection. No model is loaded, no CUDA call is made, and no filesystem
write happens anywhere in this module -- constructing a config only reads
small JSON files to validate paths. See `tell.safety.agent_model_factory`
for the lazy loader that actually uses this config to load a model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION

ADAPTER_PATH_ENV_VAR = "AGENT_S_ADAPTER_PATH"
OPERATIONAL_THRESHOLD = 0.5134634443863925
SCIENTIFIC_THRESHOLD = 0.1708046793937683
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
FORBIDDEN_PATH_COMPONENTS = ("latest",)


class AdapterPathError(ValueError):
    """Raised when the configured Agent-S adapter path is missing,
    incomplete, mutable-aliased, or points at an in-progress checkpoint --
    never silently swallowed into "use Agent 1 instead"."""


class AgentSRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model_repo_id: str = PINNED_MODEL_REPO_ID
    base_model_revision: str = PINNED_MODEL_REVISION
    tokenizer_revision: str = PINNED_MODEL_REVISION
    agent_s_adapter_path: str
    operational_threshold: float = OPERATIONAL_THRESHOLD
    scientific_threshold: float = SCIENTIFIC_THRESHOLD
    fail_closed_behavior: str = "route_to_agent_s_fail_closed_marker"  # see agent_model_factory.FAIL_CLOSED
    adapter_enabled: bool = True

    def reject_mutable_or_incomplete_path(self) -> None:
        """Deliberately NOT a pydantic validator/`model_post_init` -- both
        get their exceptions wrapped into `pydantic.ValidationError` by
        pydantic itself (confirmed empirically, not assumed), which would
        stop callers from catching `AdapterPathError` specifically. This is
        an ordinary method, called explicitly by `AgentModelFactory` before
        every use, so `AdapterPathError` propagates unwrapped."""
        p = Path(self.agent_s_adapter_path)
        parts_lower = {part.lower() for part in p.parts}
        if parts_lower & set(FORBIDDEN_PATH_COMPONENTS):
            raise AdapterPathError(f"adapter path {self.agent_s_adapter_path!r} contains a forbidden mutable-alias component "
                                   f"{FORBIDDEN_PATH_COMPONENTS}; point at an immutable, named checkpoint or the frozen adapter directory instead")
        if p.is_symlink() and "latest" in str(p).lower():
            raise AdapterPathError(f"adapter path {self.agent_s_adapter_path!r} is a symlink resembling a mutable 'latest' alias (resolves to {p.resolve()})")

    @classmethod
    def from_env_or_default(cls, default_adapter_path: str) -> "AgentSRuntimeConfig":
        """`AGENT_S_ADAPTER_PATH` overrides the adapter path without a code
        change; every other field keeps its frozen default. Never falls
        back to `default_adapter_path` if the env var is set but invalid --
        that would be a silent, unreviewed path substitution."""
        path = os.environ.get(ADAPTER_PATH_ENV_VAR, default_adapter_path)
        return cls(agent_s_adapter_path=path)

    def validate_adapter_directory_complete(self) -> None:
        """Structural completeness check only -- does not load the model.
        Raises AdapterPathError (never returns a boolean silently ignored)
        on any of: a mutable/forbidden path component, a missing directory,
        or a missing required file. A file that looks still-being-written
        is a separate, live filesystem poll -- see
        `is_checkpoint_still_being_written` in `tell.safety.agent_model_factory`."""
        self.reject_mutable_or_incomplete_path()
        p = Path(self.agent_s_adapter_path)
        if not p.exists() or not p.is_dir():
            raise AdapterPathError(f"adapter path does not exist or is not a directory: {p}")
        missing = [f for f in REQUIRED_ADAPTER_FILES if not (p / f).exists()]
        if missing:
            raise AdapterPathError(f"adapter directory {p} is missing required files: {missing}")

    def check_adapter_base_model_compatibility(self) -> None:
        """Compares the adapter's own recorded base-model path/revision
        (written by `peft`'s `save_pretrained`) against this config's
        pinned base model. Pure JSON read; no model load."""
        cfg_path = Path(self.agent_s_adapter_path) / "adapter_config.json"
        adapter_cfg = json.loads(cfg_path.read_text())
        base = adapter_cfg.get("base_model_name_or_path", "")
        if self.base_model_revision not in str(base) and self.base_model_repo_id not in str(base):
            raise AdapterPathError(f"adapter's recorded base model {base!r} does not reference the configured "
                                   f"base model {self.base_model_repo_id!r}@{self.base_model_revision!r}")


__all__ = ["AgentSRuntimeConfig", "AdapterPathError", "ADAPTER_PATH_ENV_VAR", "OPERATIONAL_THRESHOLD", "SCIENTIFIC_THRESHOLD"]
