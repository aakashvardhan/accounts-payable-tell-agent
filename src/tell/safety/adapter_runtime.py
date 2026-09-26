"""One adapter-aware Qwen3-8B runtime, shared by Agent 1 and Agent S.

Rather than loading two separate 8B-parameter models (Agent 1's base model
and Agent S's LoRA-adapted model), this wraps a single base-model load with
a PEFT adapter attached and toggles it on/off per generation call via
`peft.PeftModel.disable_adapter()`. Agent 1's output is therefore always
byte-identical to what an unwrapped `QwenLocalRuntime` would produce (same
weights, same forward pass) -- attaching an adapter and disabling it is not
the same as never having loaded one only in the trivial sense of extra
(inert) LoRA parameters existing in memory; it changes nothing about the
frozen base weights or the computation Agent 1's path executes.

This module never trains anything and never mutates the adapter's weights;
`load_agent_s_adapter` only ever reads a frozen adapter directory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH, QwenLocalRuntime


@dataclass(frozen=True)
class AdapterInfo:
    adapter_dir: str
    adapter_config_sha256: str
    adapter_weights_sha256: str


class AdapterAwareRuntime:
    """Wraps one `QwenLocalRuntime`-loaded base model with an optional
    frozen Agent-S adapter attached. `.generate_as_agent_1(...)` and
    `.generate_as_agent_s(...)` are the only two entry points; neither
    exposes the underlying `PeftModel` for arbitrary mutation."""

    def __init__(self, snapshot_path: Path = PINNED_SNAPSHOT_PATH) -> None:
        self._runtime = QwenLocalRuntime(snapshot_path)
        self._peft_model: PeftModel | None = None
        self._adapter_info: AdapterInfo | None = None

    @property
    def tokenizer(self):
        return self._runtime.tokenizer

    @property
    def is_loaded(self) -> bool:
        return self._runtime.is_loaded

    @property
    def adapter_info(self) -> AdapterInfo | None:
        return self._adapter_info

    def load(self):
        return self._runtime.load()

    def attach_agent_s_adapter(self, adapter_dir: str | Path) -> AdapterInfo:
        """Attaches a frozen, read-only adapter. Raises if the base model
        isn't loaded yet, or if an adapter is already attached (call
        `.detach_adapter()` first -- this module never silently replaces
        one adapter with another)."""
        if not self._runtime.is_loaded:
            raise RuntimeError("base model must be loaded before attaching an adapter")
        if self._peft_model is not None:
            raise RuntimeError("an adapter is already attached; call detach_adapter() first")
        adapter_dir = Path(adapter_dir)
        info = _hash_adapter(adapter_dir)
        self._peft_model = PeftModel.from_pretrained(self._runtime.model, str(adapter_dir), is_trainable=False)
        self._peft_model.eval()
        self._adapter_info = info
        return info

    def detach_adapter(self) -> None:
        self._peft_model = None
        self._adapter_info = None

    @contextmanager
    def agent_1_model_context(self):
        """Public accessor to the base-model-only forward pass (e.g. for
        capturing Tell activations on the unadapted weights). If an
        adapter is attached, its effect is disabled for the duration of
        this context (PEFT's documented `disable_adapter` context
        manager) so Agent 1's computation is unaffected by Agent S's
        adapter ever having been loaded into the same process."""
        if self._peft_model is None:
            yield self._runtime.model
        else:
            with self._peft_model.disable_adapter():
                yield self._peft_model

    def _as_agent_s_model(self):
        if self._peft_model is None:
            raise RuntimeError("no Agent-S adapter attached; call attach_agent_s_adapter() first")
        return self._peft_model

    def render_chat_prompt(self, messages: list[dict], *, enable_thinking: bool = False) -> str:
        return self._runtime.render_chat_prompt(messages, enable_thinking=enable_thinking)

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        return self._runtime.tokenize(text)

    def generate_as_agent_1(self, inputs: dict[str, torch.Tensor], *, max_new_tokens: int) -> torch.Tensor:
        with self.agent_1_model_context() as model, torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, return_dict_in_generate=False)

    def generate_as_agent_s(self, inputs: dict[str, torch.Tensor], *, max_new_tokens: int) -> torch.Tensor:
        model = self._as_agent_s_model()
        with torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, return_dict_in_generate=False)


def _sha256_file(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_adapter(adapter_dir: Path) -> AdapterInfo:
    cfg = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapter_model.safetensors"
    if not cfg.exists() or not weights.exists():
        raise RuntimeError(f"adapter directory missing required files: {adapter_dir}")
    return AdapterInfo(adapter_dir=str(adapter_dir), adapter_config_sha256=_sha256_file(cfg), adapter_weights_sha256=_sha256_file(weights))


__all__ = ["AdapterAwareRuntime", "AdapterInfo", "PINNED_MODEL_REPO_ID", "PINNED_MODEL_REVISION"]
