"""LoRA smoke-training run (Part 4). Loads Qwen3-8B locally in BF16,
wraps it with a PEFT LoRA adapter over the attention projection modules,
and runs ~20 optimizer steps over a fixed 32-example subset of the LoRA
corpus v1 train split. This is a smoke test, not full training -- see
this repo's task instructions and configs/lora/smoke_v1_config.json.

Verifies and reports, before training: total/trainable parameter counts
and percentage, that every trainable parameter belongs to the adapter,
that the base model is frozen, and that one collated batch's non-masked
labels decode to exactly the expected gold action. Aborts cleanly on a
NaN/Inf loss or on any non-adapter parameter becoming trainable.

Never touches results/probe_dataset, results/probe_training, the ledger,
the gate, or SQLite. Never overwrites the base-model snapshot cache --
the adapter is saved only under results/lora_training/smoke_v1/adapter/.
"""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3ForCausalLM

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH
from tell.lora_dataset.masking import build_masked_example, decode_non_masked_labels, render_gold_completion_text

CONFIG_PATH = Path("/home/hp5/tell/configs/lora/smoke_v1_config.json")
CORPUS_DIR = Path("/home/hp5/tell/results/lora_dataset/v1")
RENDERED_SAMPLES_PATH = CORPUS_DIR / "rendered_samples.jsonl"
SMOKE_SUBSET_PATH = Path("/home/hp5/tell/results/lora_training/smoke_v1/smoke_subset_manifest.json")

OUTPUT_DIR = Path("/home/hp5/tell/results/lora_training/smoke_v1")
ADAPTER_DIR = OUTPUT_DIR / "adapter"
PRETRAIN_CHECKS_PATH = OUTPUT_DIR / "pretrain_checks.json"
TRAINING_METRICS_PATH = OUTPUT_DIR / "training_metrics.jsonl"
ENVIRONMENT_REPORT_PATH = OUTPUT_DIR / "environment_report.json"


def _load_rendered_by_id() -> dict[str, dict]:
    out = {}
    for line in RENDERED_SAMPLES_PATH.open():
        r = json.loads(line)
        out[r["sample_id"]] = r
    return out


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    lora_cfg = config["lora_config"]
    train_cfg = config["training_config"]

    smoke_subset = json.loads(SMOKE_SUBSET_PATH.read_text())
    sample_ids = smoke_subset["sample_ids"]
    if len(sample_ids) != 32:
        raise RuntimeError(f"Expected 32 smoke sample_ids, found {len(sample_ids)}")

    rendered_by_id = _load_rendered_by_id()
    examples = [rendered_by_id[sid] for sid in sample_ids]

    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to silently fall back to CPU.")

    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)

    print("[load] loading base model (bf16, cuda:0, local_files_only=True) ...")
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(
        str(PINNED_SNAPSHOT_PATH),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    load_elapsed = time.perf_counter() - load_start
    print(f"[load] base model loaded in {load_elapsed:.2f}s")

    base_total_params = sum(p.numel() for p in model.parameters())
    base_trainable_before_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    non_adapter_trainable = [n for n in trainable_param_names if "lora_" not in n]
    frozen_non_adapter_ok = len(non_adapter_trainable) == 0
    trainable_pct = 100.0 * trainable_params / total_params

    print(f"[params] total={total_params:,} trainable={trainable_params:,} ({trainable_pct:.4f}%)")
    print(f"[params] every trainable parameter belongs to the adapter: {frozen_non_adapter_ok}")
    if not frozen_non_adapter_ok:
        raise RuntimeError(f"Non-adapter trainable parameters found (aborting): {non_adapter_trainable[:10]}")

    # ---- Verify one collated batch's masking before training ----
    max_seq_len = train_cfg["max_seq_len"]
    first = examples[0]
    masked0 = build_masked_example(
        tokenizer,
        sample_id=first["sample_id"],
        messages=first["messages"],
        gold_action_dict=first["gold_action_dict"],
        max_seq_len=max_seq_len,
    )
    if masked0.rejected:
        raise RuntimeError(f"First smoke example unexpectedly rejected: {masked0.reject_reason}")
    decoded_completion = decode_non_masked_labels(tokenizer, masked0.labels)
    expected_completion = render_gold_completion_text(first["gold_action_dict"]) + tokenizer.eos_token
    batch_check_ok = decoded_completion == expected_completion

    pretrain_checks = {
        "base_total_params_before_lora_wrap": base_total_params,
        "base_trainable_params_before_lora_wrap": base_trainable_before_freeze,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": trainable_pct,
        "every_trainable_param_is_adapter": frozen_non_adapter_ok,
        "non_adapter_trainable_param_names": non_adapter_trainable,
        "base_model_frozen": frozen_non_adapter_ok,
        "n_lora_target_modules_matched": len(trainable_param_names),
        "sample_batch_check": {
            "sample_id": first["sample_id"],
            "prompt_token_count": masked0.prompt_token_count,
            "completion_token_count": masked0.completion_token_count,
            "total_token_count": masked0.total_token_count,
            "prefix_verified": masked0.prefix_verified,
            "non_masked_labels_decode_to_expected_gold_action": batch_check_ok,
            "decoded_completion": decoded_completion,
            "expected_completion": expected_completion,
        },
        "model_load_elapsed_seconds": load_elapsed,
        "model_load_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRETRAIN_CHECKS_PATH.write_text(json.dumps(pretrain_checks, indent=2))
    print(f"Wrote {PRETRAIN_CHECKS_PATH}")
    if not batch_check_ok:
        raise RuntimeError("Batch masking check FAILED: non-masked labels do not decode to the expected gold action. Aborting before training.")

    # ---- Build all 32 masked examples once (train-mode, no padding needed since batch_size=1) ----
    masked_examples = []
    for r in examples:
        m = build_masked_example(tokenizer, sample_id=r["sample_id"], messages=r["messages"], gold_action_dict=r["gold_action_dict"], max_seq_len=max_seq_len)
        if m.rejected:
            raise RuntimeError(f"{r['sample_id']}: unexpectedly rejected during smoke training ({m.reject_reason})")
        masked_examples.append(m)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg["learning_rate"])

    n_steps = train_cfg["n_optimizer_steps"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    device = torch.device("cuda:0")

    training_metrics = []
    aborted = False
    abort_reason = None
    micro_batch_counter = 0

    print(f"[train] starting {n_steps} optimizer steps x {grad_accum} grad-accum micro-batches over {len(masked_examples)} examples")
    for step in range(1, n_steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for _ in range(grad_accum):
            ex = masked_examples[micro_batch_counter % len(masked_examples)]
            micro_batch_counter += 1

            input_ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
            labels = torch.tensor([ex.labels], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                aborted = True
                abort_reason = f"non-finite loss ({loss.item()}) at step {step}, sample {ex.sample_id}"
                break
            (loss / grad_accum).backward()
            step_losses.append(loss.item())

        if aborted:
            break

        current_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if any("lora_" not in n for n in current_trainable):
            aborted = True
            abort_reason = f"non-adapter parameter became trainable at step {step}"
            break

        optimizer.step()
        step_elapsed = time.perf_counter() - step_start

        mem_allocated = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
        mean_loss = sum(step_losses) / len(step_losses)
        record = {
            "step": step,
            "loss": mean_loss,
            "learning_rate": train_cfg["learning_rate"],
            "gpu_allocated_bytes": mem_allocated,
            "gpu_reserved_bytes": mem_reserved,
            "elapsed_seconds": step_elapsed,
        }
        training_metrics.append(record)
        print(f"[train] step {step}/{n_steps} loss={mean_loss:.4f} elapsed={step_elapsed:.2f}s gpu_alloc={mem_allocated/1e9:.2f}GB")

    with TRAINING_METRICS_PATH.open("w") as f:
        for r in training_metrics:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {TRAINING_METRICS_PATH}")

    if aborted:
        (OUTPUT_DIR / "abort_report.json").write_text(json.dumps({"aborted": True, "reason": abort_reason, "completed_steps": len(training_metrics)}, indent=2))
        raise RuntimeError(f"Smoke training aborted: {abort_reason}")

    # ---- Save adapter only (never touches the base-model snapshot cache) ----
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"[save] adapter saved to {ADAPTER_DIR}")

    peak_mem = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}
    environment_report = {
        "cuda_available": True,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_memory_allocated_bytes": peak_mem["allocated_bytes"],
        "peak_memory_reserved_bytes": peak_mem["reserved_bytes"],
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "n_steps_completed": len(training_metrics),
        "final_loss": training_metrics[-1]["loss"] if training_metrics else None,
        "first_loss": training_metrics[0]["loss"] if training_metrics else None,
        "adapter_saved_to": str(ADAPTER_DIR),
        "base_model_snapshot_untouched": True,
    }
    ENVIRONMENT_REPORT_PATH.write_text(json.dumps(environment_report, indent=2))
    print(f"Wrote {ENVIRONMENT_REPORT_PATH}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
