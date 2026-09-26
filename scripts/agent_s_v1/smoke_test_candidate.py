"""GPU smoke test for a snapshotted candidate adapter (see
snapshot_candidate.py). NOT executed as part of the CPU-safe work --
prepared for the later GPU phase only. Runs the fixed, frozen
`epoch1_smoke_set_v1.json` sample set (10 training-pool records, no held-
out test/lexical/delayed-memory-OOD sample) through the candidate to
confirm it loads and produces well-formed, parseable output. This is a
viability check, not checkpoint selection -- it does not compare against
other candidates or compute the selection formula.

Usage (GPU phase only):
    .venv/bin/python scripts/agent_s_v1/smoke_test_candidate.py \\
        --candidate results/lora_training/agent_s_v1/candidates/epoch1_candidate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SMOKE_SET_PATH = REPO / "results/lora_training/agent_s_v1/epoch1_smoke_set_v1.json"
CORPUS = REPO / "results/enterprise_corpus/v2_2"
SUP_DIR = REPO / "results/enterprise_corpus/v2_2_training_supplement"
MAX_NEW_TOKENS = 256


def main(candidate_dir: str) -> dict:
    import torch  # noqa: F401
    from enterprise_v2_2.contract import parse_action_v22
    from tell.safety.adapter_runtime import AdapterAwareRuntime

    smoke = json.loads(SMOKE_SET_PATH.read_text())
    ids = set(smoke["sample_ids"])
    labels = {r["sample_id"]: r for r in (json.loads(l) for l in open(SUP_DIR / "memory_supplement_v1_labels.jsonl")) if r["sample_id"] in ids}
    inputs = {r["sample_id"]: r for r in (json.loads(l) for l in open(SUP_DIR / "memory_supplement_v1_inputs.jsonl")) if r["sample_id"] in ids}
    assert set(labels) == ids, f"missing sample_ids in corpus: {ids - set(labels)}"

    runtime = AdapterAwareRuntime()
    runtime.load()
    info = runtime.attach_agent_s_adapter(candidate_dir)

    results = []
    for sid in sorted(ids):
        messages = inputs[sid]["messages"]
        chat_text = runtime.render_chat_prompt(messages, enable_thinking=False)
        tok = runtime.tokenize(chat_text)
        plen = int(tok["input_ids"].shape[1])
        t0 = time.perf_counter()
        gen = runtime.generate_as_agent_s(tok, max_new_tokens=MAX_NEW_TOKENS)
        dt = time.perf_counter() - t0
        raw = runtime.tokenizer.decode(gen[0][plen:], skip_special_tokens=True)
        parsed = parse_action_v22(raw, labels[sid]["contract"])
        results.append({"sample_id": sid, "parse_outcome": parsed.outcome.value,
                        "action_type": parsed.action.model_dump(mode="json").get("action") if parsed.action else None,
                        "generation_seconds": dt, "raw_output": raw[:300]})

    report = {"candidate_dir": candidate_dir, "adapter_hash": info.adapter_weights_sha256,
             "n_samples": len(results), "n_parse_valid": sum(1 for r in results if r["parse_outcome"] == "valid"),
             "results": results}
    out_path = Path(candidate_dir) / "smoke_test_report.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    main(args.candidate)
