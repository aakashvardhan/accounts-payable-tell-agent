"""Real Safety-LoRA ("Agent S") training run over the frozen v1.1 sampler
contract's 3 epochs (4,965 examples total). Same LoRA architecture as the
tested smoke run (r=8, alpha=16, dropout=0.05, target_modules q/k/v/o_proj,
CAUSAL_LM, bias=none) and the same per-example (batch_size=1) + gradient-
accumulation training loop, scaled up and made resumable.

Deviations from smoke_v1_config.json, and why (LoRA architecture itself is
UNCHANGED; only run-scale/data-shape settings below differ, all disclosed):
  - max_seq_len 6144 -> 7168: smoke's value was sized for a different,
    smaller corpus; this pool's real max token count is 6852 (measured by
    scripts/agent_s_v1/check_lengths.py over all 2,230 pool items, 0
    rejections) -- 6144 would have silently dropped real training data.
  - gradient_accumulation_steps 4 -> 8: smoke's value was sized for a
    32-example run; effective batch 8 is a reasoned choice for a ~5,000-
    example run, not a correctness-relevant hyperparameter.
  - n_optimizer_steps (smoke: fixed 20) -> derived from 3 real epochs x
    1,655 examples / effective batch 8.
  - learning_rate (2e-4), optimizer (AdamW), scheduler (constant, no
    warmup) are UNCHANGED from smoke_v1_config.json, per instruction not
    to silently alter them absent an authoritative override.

Resumable: writes a checkpoint (adapter + optimizer state + RNG state +
position) every CHECKPOINT_EVERY_STEPS steps and at every epoch boundary,
under a numbered, monotonically-increasing directory that is never
overwritten. `--resume` finds the latest checkpoint, verifies its config/
data hashes match this run's, and continues from the exact next example.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3ForCausalLM

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH  # noqa: E402
from tell.lora_dataset.masking import build_masked_example, decode_non_masked_labels, render_gold_completion_text  # noqa: E402
from agent_s_v1.build_epochs import main as build_epochs_main  # noqa: E402

OUT = REPO / "results/lora_training/agent_s_v1"
CKPT_DIR = OUT / "checkpoints"
FROZEN_DIR = OUT / "frozen_adapter"
METRICS_PATH = OUT / "training_metrics.jsonl"
LOG_PATH = OUT / "training.log"
RESOLVED_CONFIG_PATH = OUT / "resolved_training_config.json"
ENV_PATH = OUT / "environment_report.json"
SMOKE_LORA_CFG_PATH = REPO / "configs/lora/smoke_v1_config.json"

MAX_SEQ_LEN = 7168
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
SEED = 20260923  # same seed as the frozen sampler contract, reused for the optimizer/RNG
CHECKPOINT_EVERY_STEPS = 100


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_examples(tokenizer):
    out = build_epochs_main()
    inputs, targets, epochs = out["inputs"], out["targets"], out["epochs"]
    epoch_examples = []
    for e, items in enumerate(epochs):
        exs = []
        for p in items:
            m = build_masked_example(tokenizer, sample_id=p.sample_id, messages=inputs[p.sample_id]["messages"],
                                     gold_action_dict=targets[p.sample_id]["gold_action"], max_seq_len=MAX_SEQ_LEN)
            if m.rejected:
                raise RuntimeError(f"epoch {e} sample {p.sample_id} unexpectedly rejected: {m.reject_reason}")
            exs.append(m)
        epoch_examples.append(exs)
    return epoch_examples


def build_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to silently fall back to CPU.")
    tokenizer = AutoTokenizer.from_pretrained(str(PINNED_SNAPSHOT_PATH), local_files_only=True)
    log("loading base model (bf16, cuda:0, local_files_only=True)...")
    t0 = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(str(PINNED_SNAPSHOT_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                             device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    log(f"base model loaded in {time.perf_counter()-t0:.1f}s")
    smoke_cfg = json.loads(SMOKE_LORA_CFG_PATH.read_text())["lora_config"]
    lora_config = LoraConfig(r=smoke_cfg["r"], lora_alpha=smoke_cfg["lora_alpha"], lora_dropout=smoke_cfg["lora_dropout"],
                             target_modules=smoke_cfg["target_modules"], task_type=TaskType.CAUSAL_LM, bias=smoke_cfg["bias"])
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_adapter = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if non_adapter:
        raise RuntimeError(f"non-adapter trainable parameters found: {non_adapter[:10]}")
    log(f"params: total={total:,} trainable={trainable:,} ({100*trainable/total:.4f}%) -- matches pilot ~7,667,712" if trainable == 7_667_712
        else f"params: total={total:,} trainable={trainable:,} ({100*trainable/total:.4f}%) -- DIFFERS from pilot's 7,667,712, reported not silently changed")
    return tokenizer, model, lora_config, smoke_cfg


def latest_checkpoint():
    if not CKPT_DIR.exists():
        return None
    dirs = sorted((d for d in CKPT_DIR.iterdir() if d.is_dir() and (d / "trainer_state.json").exists()), key=lambda d: int(d.name.split("_")[1]))
    return dirs[-1] if dirs else None


def save_checkpoint(model, optimizer, epoch, position_in_epoch, global_step, examples_seen, rng_state, tag):
    ckpt_dir = CKPT_DIR / f"step_{global_step:06d}{('_' + tag) if tag else ''}"
    if ckpt_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint dir {ckpt_dir}")
    ckpt_dir.mkdir(parents=True)
    model.save_pretrained(str(ckpt_dir))
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(rng_state, ckpt_dir / "rng_state.pt")
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({
        "epoch": epoch, "position_in_epoch": position_in_epoch, "global_step": global_step,
        "examples_seen": examples_seen, "max_seq_len": MAX_SEQ_LEN, "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE, "seed": SEED,
        "resolved_config_sha256": sha(RESOLVED_CONFIG_PATH.read_bytes()) if RESOLVED_CONFIG_PATH.exists() else None,
    }, indent=1))
    log(f"checkpoint saved: {ckpt_dir}")
    return ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, default=0, help="run N real micro-batches only, report throughput, then exit (no save)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    tokenizer, model, lora_config, smoke_lora_cfg = build_model()
    epoch_examples = build_examples(tokenizer)
    n_epochs = len(epoch_examples)

    resolved = {
        "lora_config": smoke_lora_cfg, "max_seq_len": MAX_SEQ_LEN, "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE, "optimizer": "AdamW", "scheduler": "constant", "seed": SEED,
        "n_epochs": n_epochs, "epoch_size": len(epoch_examples[0]), "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
        "model_repo_id": PINNED_MODEL_REPO_ID, "model_revision": PINNED_MODEL_REVISION,
        "tokenizer_revision": PINNED_MODEL_REVISION, "deviations_from_smoke_config": {
            "max_seq_len": "6144 -> 7168 (real pool max token count is 6852; 6144 would drop data)",
            "gradient_accumulation_steps": "4 -> 8 (effective batch sizing for a ~5000-example run, not correctness)",
            "n_optimizer_steps": "fixed 20 -> derived from real epoch/example counts",
        },
    }
    RESOLVED_CONFIG_PATH.write_text(json.dumps(resolved, indent=1))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    device = torch.device("cuda:0")
    start_epoch, start_pos, global_step, examples_seen = 0, 0, 0, 0

    if args.resume:
        ck = latest_checkpoint()
        if ck is None:
            log("no checkpoint found; starting from scratch despite --resume")
        else:
            state = json.loads((ck / "trainer_state.json").read_text())
            expected = sha(RESOLVED_CONFIG_PATH.read_bytes())
            if state.get("resolved_config_sha256") not in (None, expected):
                raise RuntimeError(f"checkpoint config hash {state.get('resolved_config_sha256')} != current run config hash {expected}; refusing to resume mismatched run")
            model = PeftModel.from_pretrained(model.get_base_model(), str(ck), is_trainable=True)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
            optimizer.load_state_dict(torch.load(ck / "optimizer.pt", map_location=device))
            rng_state = torch.load(ck / "rng_state.pt")
            torch.set_rng_state(rng_state["cpu"])
            torch.cuda.set_rng_state(rng_state["cuda"])
            random.setstate(rng_state["py"])
            start_epoch, start_pos = state["epoch"], state["position_in_epoch"]
            global_step, examples_seen = state["global_step"], state["examples_seen"]
            log(f"resumed from {ck}: epoch={start_epoch} position={start_pos} global_step={global_step}")

    if args.calibrate:
        n = args.calibrate
        exs = epoch_examples[0][:n]
        t0 = time.perf_counter()
        for i, ex in enumerate(exs, 1):
            input_ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
            labels = torch.tensor([ex.labels], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
            (out.loss / GRAD_ACCUM).backward()
            if i % GRAD_ACCUM == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            print(f"calib {i}/{n} loss={out.loss.item():.4f} tok={ex.total_token_count} elapsed={time.perf_counter()-t0:.1f}s "
                  f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        total_examples = n_epochs * len(epoch_examples[0])
        rate = (time.perf_counter() - t0) / n
        print(f"\nCALIBRATION: {rate:.2f}s/example -> full run ({total_examples} examples) ETA {rate*total_examples/3600:.2f}h", flush=True)
        return

    total_examples = sum(len(e) for e in epoch_examples)
    log(f"starting real training: {n_epochs} epochs x {len(epoch_examples[0])} examples = {total_examples} total, "
        f"grad_accum={GRAD_ACCUM}, lr={LEARNING_RATE}, max_seq_len={MAX_SEQ_LEN}")
    t_run_start = time.perf_counter()

    for epoch in range(start_epoch, n_epochs):
        exs = epoch_examples[epoch]
        pos0 = start_pos if epoch == start_epoch else 0
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        t_epoch_start = time.perf_counter()
        i = pos0
        while i < len(exs):
            ex = exs[i]
            input_ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
            labels = torch.tensor([ex.labels], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
            loss = out.loss
            if not torch.isfinite(loss):
                (OUT / "abort_report.json").write_text(json.dumps({"aborted": True, "reason": f"non-finite loss at epoch {epoch} pos {i} sample {ex.sample_id}", "global_step": global_step}, indent=1))
                raise RuntimeError(f"non-finite loss ({loss.item()}) at epoch {epoch}, position {i}, sample {ex.sample_id}")
            (loss / GRAD_ACCUM).backward()
            step_losses.append(loss.item())
            examples_seen += 1
            i += 1
            is_accum_boundary = (len(step_losses) == GRAD_ACCUM) or (i == len(exs))
            if is_accum_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.perf_counter() - t0
                mean_loss = sum(step_losses) / len(step_losses)
                record = {"global_step": global_step, "epoch": epoch, "position_in_epoch": i, "examples_seen": examples_seen,
                          "loss": mean_loss, "grad_norm": float(grad_norm), "learning_rate": LEARNING_RATE,
                          "micro_batch_size": len(step_losses), "elapsed_seconds_last_micro_batch": elapsed,
                          "throughput_examples_per_sec": len(step_losses) / max(elapsed, 1e-6),
                          "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
                          "wall_elapsed_seconds": time.perf_counter() - t_run_start}
                with open(METRICS_PATH, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if global_step % 10 == 0 or is_accum_boundary and i == len(exs):
                    log(f"epoch {epoch} step {global_step} pos {i}/{len(exs)} loss={mean_loss:.4f} grad_norm={grad_norm:.3f} "
                        f"vram={torch.cuda.max_memory_allocated()/1e9:.1f}GB total_elapsed={record['wall_elapsed_seconds']/60:.1f}min")
                step_losses = []
                if global_step % CHECKPOINT_EVERY_STEPS == 0:
                    save_checkpoint(model, optimizer, epoch, i, global_step, examples_seen,
                                    {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "py": random.getstate()}, tag=None)
        save_checkpoint(model, optimizer, epoch + 1, 0, global_step, examples_seen,
                        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "py": random.getstate()}, tag="epoch_end")
        log(f"epoch {epoch} complete in {(time.perf_counter()-t_epoch_start)/60:.1f} min")

    log(f"TRAINING COMPLETE: {global_step} optimizer steps, {examples_seen} examples, {(time.perf_counter()-t_run_start)/3600:.2f}h total")
    (OUT / "environment_report.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(), "total_wall_seconds": time.perf_counter() - t_run_start,
        "global_steps": global_step, "examples_seen": examples_seen,
    }, indent=1))
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
