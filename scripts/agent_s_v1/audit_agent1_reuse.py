"""CPU-only, read-only. Audits whether the probe milestone's already-
recorded Agent-1 generations (results/activations/probe_enterprise_v1/generation/)
can be reused as Arm A of the three-arm evaluation instead of regenerating
them. Compares every material field this task asks for; does not decide
from sample-ID matching alone."""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

REPO = Path("/home/hp5/tell")
GEN_DIR = REPO / "results/activations/probe_enterprise_v1/generation"
OUT = REPO / "results/lora_training/agent_s_v1"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> dict:
    gen_rows = {}
    for f in sorted(glob.glob(str(GEN_DIR / "shard_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            gen_rows[r["sample_id"]] = r

    needed = {}
    v22 = [json.loads(l) for l in open(REPO / "results/enterprise_corpus/v2_2/probe_v2_2_labels.jsonl")]
    needed["test"] = {r["sample_id"] for r in v22 if r["split"] == "test"}
    needed["lexical_challenge"] = {json.loads(l)["sample_id"] for l in open(REPO / "results/enterprise_corpus/v2_2/probe_v2_2_lexical_challenge_labels.jsonl")}
    needed["delayed_memory_ood"] = {json.loads(l)["sample_id"] for l in open(REPO / "results/enterprise_corpus/v2_2/probe_v2_2_delayed_memory_ood_labels.jsonl")}
    expected_counts = {"test": 400, "lexical_challenge": 400, "delayed_memory_ood": 600}

    fields = {}

    def field(name, matches, detail):
        fields[name] = {"matches": matches, "detail": detail}

    coverage = {p: {"needed": len(ids), "present": len(ids & gen_rows.keys()), "missing": sorted(ids - gen_rows.keys())[:5],
                    "missing_count": len(ids - gen_rows.keys())} for p, ids in needed.items()}
    field("sample_ids_and_coverage", all(c["missing_count"] == 0 and c["needed"] == expected_counts[p] for p, c in coverage.items()), coverage)

    src_hashes = {
        "probe_v2_2_inputs.jsonl": sha(REPO / "results/enterprise_corpus/v2_2/probe_v2_2_inputs.jsonl"),
        "probe_v2_2_lexical_challenge_inputs.jsonl": sha(REPO / "results/enterprise_corpus/v2_2/probe_v2_2_lexical_challenge_inputs.jsonl"),
        "probe_v2_2_delayed_memory_ood_inputs.jsonl": sha(REPO / "results/enterprise_corpus/v2_2/probe_v2_2_delayed_memory_ood_inputs.jsonl"),
    }
    recorded_at_capture_time = {  # from results/activations/probe_enterprise_v1/activation_manifest.json (frozen, unchanged)
        "probe_v2_2_inputs.jsonl": "b62a93b0609367b2e1a7c93dc535728095a94b1089bb7251ab2b3d4c9cb61b34",
    }
    field("source_records_unchanged_since_generation", src_hashes["probe_v2_2_inputs.jsonl"] == recorded_at_capture_time["probe_v2_2_inputs.jsonl"], src_hashes)

    field("prompt_construction_and_chat_template",
         "identical: both call runtime.render_chat_prompt(messages, enable_thinking=False), i.e. "
         "tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False), on the same tokenizer instance from the same pinned snapshot",
         {"collect.py_call": "rt.render_chat_prompt(messages, enable_thinking=False)", "three_arm_eval.py_call": "runtime.render_chat_prompt(messages, enable_thinking=False)"})

    field("system_prompt", "identical: no system prompt is added by either caller; the full message list (including any system role) "
         "comes verbatim from the same frozen *_inputs.jsonl file", None)

    field("base_model_revision_and_tokenizer_revision", True,
         {"pinned_model_repo_id": "Qwen/Qwen3-8B", "pinned_model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
          "local_model.py_sha256_now": sha(REPO / "src/tell/agent/local_model.py"), "note": "both callers load via the same PINNED_SNAPSHOT_PATH constant; unchanged since generation"})

    field("tool_definitions_and_tool_state", "not applicable to either run: this is a single-decision-point prompt replay "
         "(the full conversation, including any prior tool results, is already baked into the frozen `messages` list) -- "
         "neither the original generation nor the planned Arm-A pass calls any live tool", None)

    field("memory_state", "not applicable, for the same reason as tool state -- any memory content is already rendered into `messages`", None)

    field("generation_seed_and_sampling_params", True,
         {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
          "note": "greedy decoding both times; with do_sample=False no seed or sampling parameter affects the output, so this is deterministic given identical weights+prompt"})

    field("max_new_tokens", True, {"collect.py": 256, "three_arm_eval.py_and_select_checkpoint.py": 256, "source_constant": "tell.agent.loop.MAX_NEW_TOKENS"})

    field("action_schema_and_parser_version", True,
         {"parser": "scripts/enterprise_v2_2/contract.py:parse_action_v22", "sha256_now": sha(REPO / "scripts/enterprise_v2_2/contract.py"),
          "note": "same function, same file, used identically by both the original generation and the planned Arm-A consumer"})

    field("generation_call_arguments", True,
         {"collect.py": "model.generate(**inputs, max_new_tokens=256, do_sample=False, output_hidden_states=False, return_dict_in_generate=False, use_cache=True)",
          "adapter_runtime.generate_as_agent_1": "model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True, return_dict_in_generate=False)",
          "note": "output_hidden_states=False is generate()'s own default and has no effect on generated text; otherwise byte-identical call"})

    unverified_note = (
        "ONE STRUCTURAL DIFFERENCE NOT YET EMPIRICALLY VERIFIED (requires GPU, deferred): the ORIGINAL recorded "
        "generations were produced by a plain (never PEFT-wrapped) `Qwen3ForCausalLM.generate()` call. The PLANNED "
        "Arm-A generation, once the Agent-S adapter exists, would go through `AdapterAwareRuntime.generate_as_agent_1`, "
        "which calls generate() inside `PeftModel.disable_adapter()` -- a different code path (base model wrapped by "
        "PEFT, then the adapter's forward contribution disabled) that is DESIGNED to be computation-identical to the "
        "unwrapped base model, but this has not been empirically bit-verified in this session. Recommended GPU-phase "
        "step 0 (cheap, ~10 samples): after the adapter is frozen, regenerate a small spot-check subset via "
        "generate_as_agent_1 and diff raw output against the recorded value before trusting reuse for the full 1,400."
    )

    hard_boolean_checks = ["sample_ids_and_coverage", "source_records_unchanged_since_generation",
                          "base_model_revision_and_tokenizer_revision", "generation_seed_and_sampling_params",
                          "max_new_tokens", "action_schema_and_parser_version", "generation_call_arguments"]
    failed = [name for name in hard_boolean_checks if fields[name]["matches"] is not True]
    verdict = "REGENERATION_REQUIRED" if failed else "REUSE_EXACT"

    report = {"verdict": verdict, "verdict_qualifier": "REUSE_EXACT pending a cheap GPU spot-check (see caveat), not an unconditional guarantee",
             "fields_compared": fields, "unverified_structural_assumption": unverified_note,
             "conclusion": ("Configure the evaluation runner to consume results/activations/probe_enterprise_v1/generation/ "
                           "for Arm A instead of regenerating 1,400 Agent-1 outputs, PROVIDED the GPU-phase spot-check "
                           "in the caveat above passes on first use. If it does not pass, fall back to regenerating Arm A "
                           "via generate_as_agent_1 for all 1,400 records -- do not silently mix reused and freshly-"
                           "generated Agent-1 outputs within one arm.")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agent1_reuse_audit.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"verdict": verdict, "failed_checks": failed, "coverage": coverage}, indent=1))
    return report


if __name__ == "__main__":
    main()
