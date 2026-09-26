"""Explicit-lifetime local Qwen3-8B runtime for Tell's model-decision
experiments.

Loads only the pinned local snapshot already on disk, in BF16 on
`cuda:0`, with SDPA attention, `local_files_only=True`, and
`trust_remote_code` never set (so Transformers uses its own mainline
`Qwen3ForCausalLM`, not repo-supplied code). Never downloads anything and
never calls an external endpoint -- `from_pretrained` is always given the
local snapshot directory, never the `Qwen/Qwen3-8B` repo id.

Nothing loads at import time or on construction: `QwenLocalRuntime()` just
records configuration. Call `.load()` explicitly to actually load the
tokenizer and model, and `.unload()` to release them deterministically
(delete references, `gc.collect()`, `torch.cuda.empty_cache()`). This
keeps unit-test collection and ordinary imports of this module free of any
GPU/model dependency -- see tests/test_decision.py and
tests/test_prompts.py, which import from `tell.agent` without ever
constructing or loading a runtime.

One `QwenLocalRuntime` instance owns at most one loaded model; it holds no
class-level or module-level mutable state, so multiple instances (or
repeated load/unload cycles within a test) do not interfere with each
other.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

PINNED_MODEL_REPO_ID = "Qwen/Qwen3-8B"
PINNED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# TELL_MODEL_SNAPSHOT overrides; otherwise the standard Hugging Face hub cache (HF_HUB_CACHE / HF_HOME / ~/.cache).
_HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
PINNED_SNAPSHOT_PATH = Path(
    os.environ.get("TELL_MODEL_SNAPSHOT")
    or _HF_HUB_CACHE / "models--Qwen--Qwen3-8B" / "snapshots" / PINNED_MODEL_REVISION
)


@dataclass(frozen=True)
class ModelLoadResult:
    elapsed_seconds: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    device: str
    dtype: str
    attn_implementation: str | None
    num_parameters: int
    model_repo_id: str
    model_revision: str
    model_class: str
    tokenizer_class: str


def _require_local_snapshot(path: Path) -> Path:
    """Refuses anything that is not an existing local snapshot directory
    -- in particular, a bare repo id such as "Qwen/Qwen3-8B" is rejected
    rather than silently triggering a network fetch."""
    if not path.is_dir():
        raise ValueError(
            f"Not a local model snapshot directory: {path!r}. This runtime "
            "never downloads a model; pass an existing local snapshot path."
        )
    if not (path / "config.json").exists():
        raise ValueError(f"No config.json found under {path!r}; does not look like a model snapshot.")
    return path


class QwenLocalRuntime:
    """Explicit-lifetime wrapper around one local Qwen3-8B model + tokenizer."""

    def __init__(self, snapshot_path: Path = PINNED_SNAPSHOT_PATH) -> None:
        self._snapshot_path = Path(snapshot_path)
        self._model: Qwen3ForCausalLM | None = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Qwen3ForCausalLM:
        if self._model is None:
            raise RuntimeError("Model not loaded; call .load() first.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded; call .load() first.")
        return self._tokenizer

    def load(self) -> ModelLoadResult:
        if self._model is not None:
            raise RuntimeError("Model already loaded; call .unload() first.")

        snapshot_path = _require_local_snapshot(self._snapshot_path)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Refusing to silently fall back to CPU.")

        self._tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        self._model = Qwen3ForCausalLM.from_pretrained(
            str(snapshot_path),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self._model.eval()
        elapsed = time.perf_counter() - start

        return ModelLoadResult(
            elapsed_seconds=elapsed,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
            device=str(next(self._model.parameters()).device),
            dtype=str(next(self._model.parameters()).dtype),
            attn_implementation=getattr(self._model.config, "_attn_implementation", None),
            num_parameters=sum(p.numel() for p in self._model.parameters()),
            model_repo_id=PINNED_MODEL_REPO_ID,
            model_revision=PINNED_MODEL_REVISION,
            model_class=type(self._model).__name__,
            tokenizer_class=type(self._tokenizer).__name__,
        )

    def render_chat_prompt(self, messages: list[dict], *, enable_thinking: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def unload(self) -> None:
        """Deterministic cleanup: drop references, collect, empty the CUDA
        cache. Safe to call even if nothing was loaded."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
