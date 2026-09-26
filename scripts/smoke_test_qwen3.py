"""Bounded local smoke test for Qwen/Qwen3-8B on this GB10 machine.

Proves, and only proves: PyTorch runs on CUDA here, Qwen3-8B loads
locally in BF16 from an already-downloaded snapshot, it can generate a
short non-thinking response, a separate forward pass exposes all 37
hidden-state tensors, and a handful of selected last-token activations
can be inspected (summary statistics only -- no full vectors persisted).

Does NOT connect to Tell's tools, gate, ledger, scenarios, or payment
workflow, and does not implement the agent loop, probe, LoRA, memory
system, API, or UI. Does not use any hosted inference endpoint -- the
model is loaded with local_files_only=True from a path already on disk.

Usage:
    python scripts/smoke_test_qwen3.py <local_snapshot_path>

The argument must be an existing local directory (the Hugging Face
snapshot directory downloaded ahead of time, e.g. via snapshot_download).
A bare repo id such as "Qwen/Qwen3-8B" is refused -- this script never
triggers a network model fetch itself.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

PROMPT = "Reply with exactly: TELL_LOCAL_MODEL_OK"
MARKER = "TELL_LOCAL_MODEL_OK"
SELECTED_LAYERS = [0, 9, 18, 27, 36]
LAYER_DESCRIPTIONS = {
    0: "embedding output",
    9: "transformer layer 9 output",
    18: "transformer layer 18 output",
    27: "transformer layer 27 output",
    36: "transformer layer 36 output (final layer)",
}


def _mem_stats() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _system_memory_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def load_snapshot_path(arg: str) -> Path:
    path = Path(arg)
    if "/" in arg and not path.exists():
        raise SystemExit(
            f"Refusing remote model id or nonexistent path: {arg!r}. "
            "This script requires an existing local snapshot directory "
            "(local_files_only=True) -- it never downloads a model itself."
        )
    if not path.is_dir():
        raise SystemExit(f"Not a local directory: {arg!r}")
    if not (path / "config.json").exists():
        raise SystemExit(f"No config.json found under {arg!r}; does not look like a model snapshot.")
    return path


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_path", help="Local Hugging Face snapshot directory for Qwen/Qwen3-8B")
    parser.add_argument("--output-json", default=None, help="Optional path to write the structured result JSON")
    args = parser.parse_args()

    snapshot_path = load_snapshot_path(args.snapshot_path)
    result: dict = {"snapshot_path": str(snapshot_path)}

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to silently fall back to CPU.")

    torch.cuda.reset_peak_memory_stats()

    # ---- tokenizer + model load ----
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)

    load_start = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(
        str(snapshot_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    load_elapsed = time.perf_counter() - load_start
    load_mem = _mem_stats()

    result["model_load"] = {
        "elapsed_seconds": load_elapsed,
        "peak_memory_during_load": load_mem,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    print(f"[load] {load_elapsed:.2f}s, peak allocated {load_mem['max_allocated_bytes'] / 1e9:.2f} GB")

    # ---- prompt rendering (shared by generation and hidden-state passes) ----
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    prompt_token_count = int(inputs["input_ids"].shape[1])
    result["prompt"] = {"text": PROMPT, "rendered_token_count": prompt_token_count}

    # ---- generation pass ----
    torch.cuda.reset_peak_memory_stats()
    gen_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
            use_cache=True,
        )
    gen_elapsed = time.perf_counter() - gen_start
    gen_mem = _mem_stats()

    new_tokens = generated[0][prompt_token_count:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    marker_present = MARKER in decoded

    result["generation"] = {
        "generated_token_count": int(new_tokens.shape[0]),
        "decoded_response": decoded,
        "elapsed_seconds": gen_elapsed,
        "marker_present": marker_present,
        "peak_memory": gen_mem,
    }
    print(f"[generate] {gen_elapsed:.2f}s -> {decoded!r} (marker_present={marker_present})")

    del generated, new_tokens

    # ---- hidden-state pass (separate forward pass, no generation) ----
    torch.cuda.reset_peak_memory_stats()
    hs_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hs_elapsed = time.perf_counter() - hs_start
    hs_mem = _mem_stats()

    hidden_states = outputs.hidden_states
    n_hidden_states = len(hidden_states)
    hidden_size = model.config.hidden_size
    seq_len = inputs["input_ids"].shape[1]

    shape_check_ok = all(tuple(h.shape) == (1, seq_len, hidden_size) for h in hidden_states)
    dtype_check_ok = all(h.dtype == torch.bfloat16 for h in hidden_states)
    device_check_ok = all(h.is_cuda for h in hidden_states)
    finite_check_ok = all(bool(torch.isfinite(h).all().item()) for h in hidden_states)

    result["hidden_states"] = {
        "count": n_hidden_states,
        "expected_count": 37,
        "count_ok": n_hidden_states == 37,
        "hidden_size": hidden_size,
        "sequence_length": seq_len,
        "shape_check_ok": shape_check_ok,
        "dtype_check_ok": dtype_check_ok,
        "device_check_ok": device_check_ok,
        "finite_check_ok": finite_check_ok,
        "elapsed_seconds": hs_elapsed,
        "peak_memory": hs_mem,
        "note": (
            "hidden_states[0] is the embedding output; hidden_states[n] is "
            "the output of transformer layer n (1-indexed), so "
            "hidden_states[36] is the final transformer layer's output."
        ),
    }
    print(f"[hidden_states] {hs_elapsed:.2f}s, count={n_hidden_states}, shape_ok={shape_check_ok}")

    # ---- selected last-token activations: summary stats only, no full vectors ----
    selected = []
    for idx in SELECTED_LAYERS:
        tensor = hidden_states[idx]
        last_token_vec = tensor[0, -1, :]
        l2_norm = float(torch.linalg.vector_norm(last_token_vec.float()).item())
        is_finite = bool(torch.isfinite(last_token_vec).all().item())
        selected.append(
            {
                "hidden_state_index": idx,
                "description": LAYER_DESCRIPTIONS[idx],
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "last_token_l2_norm": l2_norm,
                "finite": is_finite,
            }
        )
    result["selected_activations"] = selected
    for s in selected:
        print(f"  layer {s['hidden_state_index']:>2} ({s['description']}): "
              f"shape={s['shape']} l2_norm={s['last_token_l2_norm']:.3f} finite={s['finite']}")

    # ---- cleanup ----
    del outputs, hidden_states, inputs, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    cleanup_mem = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    sys_mem_after = _system_memory_available_bytes()
    result["cleanup"] = {
        "cuda_memory_after": cleanup_mem,
        "system_memory_available_bytes_after": sys_mem_after,
    }
    print(f"[cleanup] cuda allocated={cleanup_mem['allocated_bytes']/1e9:.3f} GB, "
          f"reserved={cleanup_mem['reserved_bytes']/1e9:.3f} GB")

    result["acceptance_checks"] = {
        "generation_marker_present": marker_present,
        "hidden_state_count_is_37": n_hidden_states == 37,
        "hidden_state_shapes_ok": shape_check_ok,
        "hidden_state_dtype_ok": dtype_check_ok,
        "hidden_state_device_ok": device_check_ok,
        "hidden_state_values_finite": finite_check_ok,
    }
    result["verdict"] = "PASS" if all(result["acceptance_checks"].values()) else "FAIL"

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.output_json}")

    return result


if __name__ == "__main__":
    r = main()
    print()
    print("VERDICT:", r["verdict"])
    sys.exit(0 if r["verdict"] == "PASS" else 1)
