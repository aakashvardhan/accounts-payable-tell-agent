"""GPU spot-check (run once, after the final adapter is frozen): confirms
`AdapterAwareRuntime.generate_as_agent_1` (base model with the frozen
adapter attached-but-disabled via PEFT's `disable_adapter()`) reproduces
the exact raw output already recorded by the probe milestone's plain
(never-PEFT-wrapped) generation, for a small fixed sample. This is the
one empirically-unverified structural assumption flagged in
agent1_reuse_audit.json.

On failure, amends (never silently overwrites) agent1_reuse_audit.json:
preserves the original verdict under `superseded_original_verdict`, sets
`verdict` to REGENERATION_REQUIRED, and records the exact mismatching
samples -- an auditable correction, same pattern as
checkpoint_selection_rule_v1_amendment_1.md.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PROBE_GEN_DIR = REPO / "results/activations/probe_enterprise_v1/generation"
AUDIT_PATH = REPO / "results/lora_training/agent_s_v1/agent1_reuse_audit.json"
OUT_PATH = REPO / "results/lora_training/agent_s_v1/agent1_equivalence_spotcheck.json"
FROZEN_ADAPTER = REPO / "results/lora_training/agent_s_v1/frozen_adapter"
N_SPOT_CHECK = 10
MAX_NEW_TOKENS = 256


def main() -> dict:
    import torch  # noqa: F401
    from tell.safety.adapter_runtime import AdapterAwareRuntime

    recorded = {}
    for f in sorted(glob.glob(str(PROBE_GEN_DIR / "shard_*.jsonl")))[:2]:
        for l in open(f):
            r = json.loads(l)
            recorded[r["sample_id"]] = r
    sample_ids = sorted(recorded)[:N_SPOT_CHECK]

    inputs_by_pop = {}
    for pop in ("probe_v2_2", "probe_v2_2_lexical_challenge", "probe_v2_2_delayed_memory_ood"):
        for l in open(REPO / f"results/enterprise_corpus/v2_2/{pop}_inputs.jsonl"):
            r = json.loads(l)
            inputs_by_pop[r["sample_id"]] = r

    runtime = AdapterAwareRuntime()
    runtime.load()
    info = runtime.attach_agent_s_adapter(FROZEN_ADAPTER)

    mismatches = []
    checked = []
    for sid in sample_ids:
        if sid not in inputs_by_pop:
            continue
        messages = inputs_by_pop[sid]["messages"]
        chat_text = runtime.render_chat_prompt(messages, enable_thinking=False)
        tok = runtime.tokenize(chat_text)
        plen = int(tok["input_ids"].shape[1])
        gen = runtime.generate_as_agent_1(tok, max_new_tokens=MAX_NEW_TOKENS)
        raw = runtime.tokenizer.decode(gen[0][plen:], skip_special_tokens=True)
        expected = recorded[sid]["raw_output"]
        match = raw == expected
        checked.append(sid)
        if not match:
            mismatches.append({"sample_id": sid, "expected": expected[:300], "got": raw[:300]})

    passed = len(mismatches) == 0 and len(checked) > 0
    result = {"n_checked": len(checked), "n_mismatches": len(mismatches), "passed": passed,
             "adapter_weights_sha256": info.adapter_weights_sha256, "mismatches": mismatches}
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))

    if not passed:
        audit = json.loads(AUDIT_PATH.read_text())
        audit["superseded_original_verdict"] = audit["verdict"]
        audit["verdict"] = "REGENERATION_REQUIRED"
        audit["amendment_reason"] = f"Agent-1 equivalence spot-check failed: {len(mismatches)}/{len(checked)} samples produced different output " \
                                    "through the PEFT disable_adapter() path than the originally-recorded plain-model generation."
        audit["spot_check_result"] = result
        AUDIT_PATH.write_text(json.dumps(audit, indent=1))
        print("AMENDED agent1_reuse_audit.json: verdict -> REGENERATION_REQUIRED")
    return result


if __name__ == "__main__":
    sys.exit(0 if main()["passed"] else 1)
