"""Tests for tell.detector.capture: the SafeTensors/JSON persistence
boundary. No GPU and no model load -- these exercise the pure
persistence and hashing logic with small synthetic tensors, never
`capture_predecision_activations` itself (which requires a real loaded
model and is exercised only by scripts/run_clean_model_decision.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tell.detector.capture import (
    DEFAULT_HIDDEN_STATE_INDICES,
    CaptureRequest,
    CaptureResult,
    CapturedActivation,
    load_activation_artifact,
    save_activation_artifact,
    sha256_input_ids,
    sha256_text,
)

FORBIDDEN_EVALUATION_MARKERS = [
    "is_attacked", "attack_surface", "expected_safe_action",
    "expected_beneficiary_account_id", "expected_amount",
    "expected_currency", "expected_outcome", "evaluation_labels", "probe_label",
]


def _build_fake_artifact(tmp_path: Path) -> tuple[Path, Path, dict[int, torch.Tensor]]:
    request = CaptureRequest(
        scenario_id="clean_04d531ca_v1",
        decision_point="pre_payment_decision",
        source_ids=("04d531ca811f448a91c6ff4e", "SIM-VENDOR-MDSPHARMA-0001", "SIM-MSG-04D531CA-0001"),
        model_repo_id="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        tokenizer_class="Qwen2TokenizerFast",
        model_class="Qwen3ForCausalLM",
        prompt_text="fake prompt text, never persisted verbatim",
    )
    generator = torch.Generator().manual_seed(0)
    vectors = {idx: torch.randn(32, generator=generator, dtype=torch.float32) for idx in DEFAULT_HIDDEN_STATE_INDICES}
    activations = tuple(
        CapturedActivation(
            hidden_state_index=idx,
            semantic_role="embedding_output" if idx == 0 else "intermediate_layer_residual_stream",
            shape=(1, 21, 4096),
            original_dtype="torch.bfloat16",
            stored_dtype="torch.float32",
            device="cuda:0",
            finite=True,
            l2_norm=float(torch.linalg.vector_norm(vectors[idx]).item()),
        )
        for idx in DEFAULT_HIDDEN_STATE_INDICES
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    capture_result = CaptureResult(
        activations=activations,
        selected_token_index=6,
        selected_token_id=7,
        sequence_length=7,
        prompt_sha256=sha256_text(request.prompt_text),
        input_ids_sha256=sha256_input_ids(input_ids),
        elapsed_seconds=0.05,
        peak_memory_allocated_bytes=16_000_000_000,
        peak_memory_reserved_bytes=16_100_000_000,
    )

    safetensors_path = tmp_path / "activations.safetensors"
    metadata_path = tmp_path / "activations_metadata.json"
    save_activation_artifact(
        vectors=vectors,
        capture_result=capture_result,
        request=request,
        safetensors_path=safetensors_path,
        metadata_path=metadata_path,
    )
    return safetensors_path, metadata_path, vectors


def test_safetensors_artifact_contains_only_requested_indices(tmp_path: Path):
    safetensors_path, _metadata_path, vectors = _build_fake_artifact(tmp_path)
    reloaded = load_activation_artifact(safetensors_path)
    assert set(reloaded.keys()) == set(DEFAULT_HIDDEN_STATE_INDICES)
    for idx, vec in vectors.items():
        assert torch.equal(reloaded[idx], vec)


def test_stored_vectors_are_float32_and_finite(tmp_path: Path):
    safetensors_path, _metadata_path, _vectors = _build_fake_artifact(tmp_path)
    reloaded = load_activation_artifact(safetensors_path)
    for vec in reloaded.values():
        assert vec.dtype == torch.float32
        assert torch.isfinite(vec).all()


def test_metadata_contains_no_attack_label_or_expected_outcome(tmp_path: Path):
    _safetensors_path, metadata_path, _vectors = _build_fake_artifact(tmp_path)
    dumped = metadata_path.read_text()
    for marker in FORBIDDEN_EVALUATION_MARKERS:
        assert marker not in dumped, f"capture metadata leaked {marker}"


def test_metadata_does_not_contain_full_prompt_text(tmp_path: Path):
    _safetensors_path, metadata_path, _vectors = _build_fake_artifact(tmp_path)
    dumped = metadata_path.read_text()
    assert "fake prompt text, never persisted verbatim" not in dumped


def test_safetensors_header_does_not_contain_full_prompt_text(tmp_path: Path):
    from safetensors import safe_open

    safetensors_path, _metadata_path, _vectors = _build_fake_artifact(tmp_path)
    with safe_open(str(safetensors_path), framework="pt") as f:
        header_metadata = f.metadata()
    assert "fake prompt text, never persisted verbatim" not in json.dumps(header_metadata)


def test_metadata_records_expected_fields(tmp_path: Path):
    _safetensors_path, metadata_path, _vectors = _build_fake_artifact(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["scenario_id"] == "clean_04d531ca_v1"
    assert metadata["decision_point"] == "pre_payment_decision"
    assert metadata["model_repo_id"] == "Qwen/Qwen3-8B"
    assert metadata["hidden_state_indices"] == list(DEFAULT_HIDDEN_STATE_INDICES)
    assert set(metadata["hidden_state_index_semantics"].keys()) == {str(i) for i in DEFAULT_HIDDEN_STATE_INDICES}
    assert metadata["hidden_state_index_semantics"]["0"] == "embedding_output"
    assert metadata["hidden_state_index_semantics"]["36"] == "final_normalized_model_representation"
    assert len(metadata["activations"]) == len(DEFAULT_HIDDEN_STATE_INDICES)
    assert metadata["selected_token_index"] == 6
    assert metadata["selected_token_id"] == 7
    assert metadata["sequence_length"] == 7


def test_prompt_hash_is_stable():
    assert sha256_text("hello world") == sha256_text("hello world")
    assert sha256_text("hello world") != sha256_text("hello world!")


def test_input_id_hash_is_stable_and_device_independent():
    ids_cpu = torch.tensor([[1, 2, 3, 4]])
    ids_cpu_2 = torch.tensor([[1, 2, 3, 4]])
    assert sha256_input_ids(ids_cpu) == sha256_input_ids(ids_cpu_2)
    different = torch.tensor([[1, 2, 3, 5]])
    assert sha256_input_ids(ids_cpu) != sha256_input_ids(different)
