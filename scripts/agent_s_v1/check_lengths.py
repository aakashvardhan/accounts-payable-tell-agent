"""CPU-only: tokenize every sampled example once (no model) to pick a
max_seq_len that accepts the whole pool, and to sanity-check response
masking on one real example before training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

from tell.agent.local_model import PINNED_SNAPSHOT_PATH  # noqa: E402
from tell.lora_dataset.masking import build_masked_example, decode_non_masked_labels  # noqa: E402
from agent_s_v1.build_epochs import main as build_epochs_main  # noqa: E402


def main():
    out = build_epochs_main()
    tok = AutoTokenizer.from_pretrained(PINNED_SNAPSHOT_PATH, local_files_only=True)
    inputs, targets = out["inputs"], out["targets"]
    all_ids = sorted({p.sample_id for e in out["epochs"] for p in e})
    lengths = []
    rejects = []
    for sid in all_ids:
        m = build_masked_example(tok, sample_id=sid, messages=inputs[sid]["messages"], gold_action_dict=targets[sid]["gold_action"], max_seq_len=1 << 30)
        lengths.append(m.total_token_count)
        if not m.prefix_verified:
            rejects.append(sid)
    lengths.sort()
    n = len(lengths)
    print(f"n_unique_examples={n} min={lengths[0]} p50={lengths[n//2]} p95={lengths[int(n*0.95)]} p99={lengths[int(n*0.99)]} max={lengths[-1]}")
    print("prefix_verification_failures:", len(rejects), rejects[:5])

    # decode + mask sanity check on one real example
    sid = all_ids[0]
    m = build_masked_example(tok, sample_id=sid, messages=inputs[sid]["messages"], gold_action_dict=targets[sid]["gold_action"], max_seq_len=1 << 30)
    decoded_completion = decode_non_masked_labels(tok, m.labels)
    gold = json.dumps(targets[sid]["gold_action"], separators=(",", ":"))
    print("sample_id:", sid)
    print("prompt_token_count:", m.prompt_token_count, "completion_token_count:", m.completion_token_count, "total:", m.total_token_count)
    print("decoded non-masked labels == gold completion + eos:", decoded_completion == gold + tok.eos_token)
    print("decoded:", repr(decoded_completion))
    print("num masked (IGNORE_INDEX) positions == prompt_token_count:", m.labels[: m.prompt_token_count] == [-100] * m.prompt_token_count)


if __name__ == "__main__":
    main()
