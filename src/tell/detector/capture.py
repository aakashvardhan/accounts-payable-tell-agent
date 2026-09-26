"""Reusable activation-capture boundary for Tell's model-decision
experiments.

Captures the model's hidden-state representation immediately before it
generates its next action -- at the final non-padding prompt token, at a
fixed set of hidden-state indices -- and persists only the *selected*
vectors (float32, no full raw prompt, no complete-sequence tensors) to a
SafeTensors file plus a companion metadata JSON.

--------------------------------------------------------------------------
Hidden-state indexing semantics for Qwen3
--------------------------------------------------------------------------
Verified directly against the installed transformers==4.57.6 Qwen3
implementation (`transformers/models/qwen3/modeling_qwen3.py`, and the
`check_model_inputs` decorator in `transformers/utils/generic.py`), not
assumed:

- ``hidden_states[0]``      -- **embedding output.** The raw output of
  ``embed_tokens`` (`inputs_embeds`), captured as the input to the first
  decoder layer, before any transformer layer has processed it. Not
  normalized.

- ``hidden_states[1..35]`` -- **intermediate transformer-layer
  representations.** ``hidden_states[n]`` is the residual-stream output
  of ``Qwen3DecoderLayer`` number ``n`` (1-indexed): residual + attention
  + MLP. These are the raw residual stream at that depth -- they are
  *not* passed through the model-level final RMSNorm, so they are not
  the representation the LM head would directly consume at that depth.

- ``hidden_states[36]``     -- **final normalized model representation,**
  not simply "decoder layer 36's raw output". ``Qwen3Model.forward`` is
  decorated with ``@check_model_inputs``, whose default
  ``tie_last_hidden_states=True`` overwrites ``hidden_states[-1]`` with
  ``last_hidden_state`` -- decoder layer 36's output passed through the
  model-level ``self.norm`` (RMSNorm). That is exactly the representation
  ``Qwen3ForCausalLM`` feeds to ``lm_head`` to produce logits. So index 36
  is qualitatively different from indices 1..35: it is the final,
  normalized representation, not an intermediate, unnormalized one.

This module does not decide layer/token selection for probe training --
that remains a later empirical step (see Tell_Project_Knowledge.md,
"Probe training"). It captures a fixed candidate set,
``DEFAULT_HIDDEN_STATE_INDICES = (0, 9, 18, 27, 36)``, so later analysis
can compare across depth using a consistent artifact shape.

--------------------------------------------------------------------------
What never goes in here
--------------------------------------------------------------------------
No attack label, ``evaluation_only`` field, or expected outcome is ever
read or written by this module. Later evaluation code joins labels to a
captured sample by ``scenario_id`` / ``decision_point`` -- this module has
no import of, and no code path to, ``tell.evaluation.scenario``'s
``EvaluationOnly``/``full_evaluation_view``.

The full prompt text is never written into the SafeTensors file's own
metadata header (SafeTensors metadata must be a flat string->string map
in any case) -- only a SHA-256 hash of it. The companion JSON metadata
file also stores only the hash, never the prompt text itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

DEFAULT_HIDDEN_STATE_INDICES: tuple[int, ...] = (0, 9, 18, 27, 36)

HIDDEN_STATE_INDEX_SEMANTICS: dict[int, str] = {
    0: "embedding_output",
    9: "intermediate_layer_residual_stream",
    18: "intermediate_layer_residual_stream",
    27: "intermediate_layer_residual_stream",
    36: "final_normalized_model_representation",
}


def _semantic_role(index: int) -> str:
    return HIDDEN_STATE_INDEX_SEMANTICS.get(index, "intermediate_layer_residual_stream")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_input_ids(input_ids: torch.Tensor) -> str:
    """Hashes the input-id sequence deterministically, independent of the
    tensor's original device/dtype (int64 on CPU is the canonical form)."""
    ids = input_ids.detach().to("cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class CaptureRequest:
    """Everything about the *context* of a capture that isn't produced by
    running the model itself. `source_ids` records which agent-visible
    observations fed the prompt (e.g. invoice docid, vendor id, email
    message id) -- operational provenance, not an evaluation label.

    The five fields after `hidden_state_indices` are optional and used
    only by the autonomous loop (tell.agent.loop), which captures at
    every turn rather than once: `run_id` and `turn_number` identify
    which turn of which run this is; `previous_action` is the model's own
    already-produced prior action (agent-visible, not an evaluation
    label); `most_recent_observation_source_type`/`_id` name which tool
    result (if any) was most recently appended to context before this
    capture. All default to `None` and are omitted from persisted
    metadata when unset, so the single-shot clean/attack-pilot scripts
    are unaffected.
    """

    scenario_id: str
    decision_point: str
    source_ids: tuple[str, ...]
    model_repo_id: str
    model_revision: str
    tokenizer_class: str
    model_class: str
    prompt_text: str
    hidden_state_indices: tuple[int, ...] = DEFAULT_HIDDEN_STATE_INDICES
    run_id: str | None = None
    turn_number: int | None = None
    previous_action: dict | None = None
    most_recent_observation_source_type: str | None = None
    most_recent_observation_source_id: str | None = None


@dataclass(frozen=True)
class CapturedActivation:
    hidden_state_index: int
    semantic_role: str
    shape: tuple[int, ...]
    original_dtype: str
    stored_dtype: str
    device: str
    finite: bool
    l2_norm: float


@dataclass(frozen=True)
class CaptureResult:
    activations: tuple[CapturedActivation, ...]
    selected_token_index: int
    selected_token_id: int
    sequence_length: int
    prompt_sha256: str
    input_ids_sha256: str
    elapsed_seconds: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int


def capture_predecision_activations(
    model,
    inputs: dict[str, torch.Tensor],
    request: CaptureRequest,
) -> tuple[CaptureResult, dict[int, torch.Tensor]]:
    """Runs one forward pass (`output_hidden_states=True, use_cache=False`)
    over an already-tokenized `inputs` dict (on `model.device`) and
    extracts the final non-padding prompt token's representation at each
    of `request.hidden_state_indices`.

    Returns a disk-safe `CaptureResult` (no tensors, trivially JSON-able)
    plus a separate dict of the actual float32 CPU vectors
    (`hidden_state_index -> 1D tensor of length hidden_size`) for the
    caller to persist via `save_activation_artifact`.

    Batch size must be 1 (this experiment never batches scenarios); the
    final non-padding token is read from `attention_mask` when present,
    else from the raw sequence length.
    """
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError(f"capture_predecision_activations requires batch size 1, got {input_ids.shape[0]}")

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        last_idx = int(attention_mask[0].sum().item()) - 1
    else:
        last_idx = input_ids.shape[1] - 1

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    elapsed = time.perf_counter() - start

    hidden_states = outputs.hidden_states
    n_available = len(hidden_states)

    vectors: dict[int, torch.Tensor] = {}
    activations: list[CapturedActivation] = []
    for idx in request.hidden_state_indices:
        if idx >= n_available:
            raise ValueError(f"Requested hidden_state index {idx} but only {n_available} are available")
        tensor = hidden_states[idx]
        selected = tensor[0, last_idx, :]
        vec_f32 = selected.detach().to("cpu", dtype=torch.float32).clone()
        is_finite = bool(torch.isfinite(vec_f32).all().item())
        l2_norm = float(torch.linalg.vector_norm(vec_f32).item())
        vectors[idx] = vec_f32
        activations.append(
            CapturedActivation(
                hidden_state_index=idx,
                semantic_role=_semantic_role(idx),
                shape=tuple(tensor.shape),
                original_dtype=str(tensor.dtype),
                stored_dtype="torch.float32",
                device=str(tensor.device),
                finite=is_finite,
                l2_norm=l2_norm,
            )
        )

    selected_token_id = int(input_ids[0, last_idx].item())
    result = CaptureResult(
        activations=tuple(activations),
        selected_token_index=last_idx,
        selected_token_id=selected_token_id,
        sequence_length=int(input_ids.shape[1]),
        prompt_sha256=sha256_text(request.prompt_text),
        input_ids_sha256=sha256_input_ids(input_ids),
        elapsed_seconds=elapsed,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
    )
    del outputs, hidden_states
    return result, vectors


def save_activation_artifact(
    *,
    vectors: dict[int, torch.Tensor],
    capture_result: CaptureResult,
    request: CaptureRequest,
    safetensors_path: Path,
    metadata_path: Path,
) -> None:
    """Writes the float32 vectors to a SafeTensors file (one tensor per
    requested index, named ``hidden_state_{idx}``) and all non-tensor
    metadata to a separate JSON file.

    The SafeTensors file's own metadata header carries only a minimal
    cross-reference (scenario_id, decision_point, prompt_sha256,
    input_ids_sha256) -- never the prompt text, and never an attack label
    or evaluation-only field. The full metadata (shapes, norms, timing,
    memory, source ids, semantics table) lives only in `metadata_path`.
    """
    tensors = {f"hidden_state_{idx}": vec.contiguous() for idx, vec in vectors.items()}
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(safetensors_path),
        metadata={
            "scenario_id": request.scenario_id,
            "decision_point": request.decision_point,
            "prompt_sha256": capture_result.prompt_sha256,
            "input_ids_sha256": capture_result.input_ids_sha256,
        },
    )

    metadata = {
        "scenario_id": request.scenario_id,
        "decision_point": request.decision_point,
        "source_ids": list(request.source_ids),
        "model_repo_id": request.model_repo_id,
        "model_revision": request.model_revision,
        "tokenizer_class": request.tokenizer_class,
        "model_class": request.model_class,
        "prompt_sha256": capture_result.prompt_sha256,
        "input_ids_sha256": capture_result.input_ids_sha256,
        "sequence_length": capture_result.sequence_length,
        "selected_token_index": capture_result.selected_token_index,
        "selected_token_id": capture_result.selected_token_id,
        "hidden_state_indices": list(request.hidden_state_indices),
        "hidden_state_index_semantics": {str(k): v for k, v in HIDDEN_STATE_INDEX_SEMANTICS.items()},
        "activations": [
            {
                "hidden_state_index": a.hidden_state_index,
                "semantic_role": a.semantic_role,
                "shape": list(a.shape),
                "original_dtype": a.original_dtype,
                "stored_dtype": a.stored_dtype,
                "device": a.device,
                "finite": a.finite,
                "l2_norm": a.l2_norm,
            }
            for a in capture_result.activations
        ],
        "timing": {"elapsed_seconds": capture_result.elapsed_seconds},
        "memory": {
            "peak_memory_allocated_bytes": capture_result.peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": capture_result.peak_memory_reserved_bytes,
        },
        "safetensors_file": safetensors_path.name,
    }
    if request.run_id is not None:
        metadata["run_id"] = request.run_id
    if request.turn_number is not None:
        metadata["turn_number"] = request.turn_number
    if request.previous_action is not None:
        metadata["previous_action"] = request.previous_action
    if request.most_recent_observation_source_type is not None:
        metadata["most_recent_observation_source_type"] = request.most_recent_observation_source_type
    if request.most_recent_observation_source_id is not None:
        metadata["most_recent_observation_source_id"] = request.most_recent_observation_source_id

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def load_activation_artifact(safetensors_path: Path) -> dict[int, torch.Tensor]:
    """Reloads the persisted float32 vectors, keyed by hidden-state index."""
    result: dict[int, torch.Tensor] = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            idx = int(key.removeprefix("hidden_state_"))
            result[idx] = f.get_tensor(key)
    return result
