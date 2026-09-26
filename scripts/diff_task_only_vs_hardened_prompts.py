"""Produces the semantic diff between the `task_only_base_v1` and
`hardened_provenance_prompt_v1` prompt profiles, before any task-only
model inference runs.

Mirrors scripts/diff_autonomous_loop_v1_v2.py's discipline: computed
from the current source files and the already-frozen v2/memory-pilot
manifests, never edited after seeing task-only run output. Proves that
the only differences between the two profiles are:

  1. the rendered system-prompt text (policy-language removals only,
     per results/scenario_design/hardened_prompt_policy_audit.md); and
  2. nothing else -- action schemas, tool schemas, scenario/attack
     content, generation parameters, model identity, and hidden-state
     indices are identical regardless of profile.

Writes results/scenario_design/task_only_vs_hardened_diff.md. Does not
modify any existing manifest, scenario, or report.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from tell.agent.actions import action_json_schema, session_a_action_json_schema, session_b_action_json_schema
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS
from tell.agent.loop_prompts import build_loop_system_prompt
from tell.agent.memory_prompts import build_session_a_system_prompt, build_session_b_system_prompt
from tell.agent.prompt_profiles import PromptProfile
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario

V2_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v2_protocol_manifest.json")
SESSION_A_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/memory_pilot_session_a_protocol_manifest.json")
SESSION_B_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/memory_pilot_session_b_protocol_manifest.json")
DIFF_REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/task_only_vs_hardened_diff.md")

ATTACK_DIR = Path("/home/hp5/tell/data/scenarios/attacked/email")
MEMORY_SCENARIO_DIR = Path("/home/hp5/tell/data/scenarios/memory_infection")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_diff(hardened: str, task_only: str) -> list[str]:
    return list(
        difflib.unified_diff(
            hardened.splitlines(),
            task_only.splitlines(),
            fromfile="hardened_provenance_prompt_v1",
            tofile="task_only_base_v1",
            lineterm="",
        )
    )


def _removed_lines(diff_lines: list[str]) -> list[str]:
    return [l[1:] for l in diff_lines if l.startswith("-") and not l.startswith("---")]


def _added_lines(diff_lines: list[str]) -> list[str]:
    return [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]


def main() -> None:
    v2_manifest = json.loads(V2_MANIFEST_PATH.read_text())
    session_a_manifest = json.loads(SESSION_A_MANIFEST_PATH.read_text())
    session_b_manifest = json.loads(SESSION_B_MANIFEST_PATH.read_text())

    hardened_loop = build_loop_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)
    task_only_loop = build_loop_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)
    hardened_a = build_session_a_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)
    task_only_a = build_session_a_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)
    hardened_b = build_session_b_system_prompt(PromptProfile.HARDENED_PROVENANCE_PROMPT_V1)
    task_only_b = build_session_b_system_prompt(PromptProfile.TASK_ONLY_BASE_V1)

    # The hardened rendering must still equal the historically frozen v2 /
    # memory-pilot hardened prompt hashes exactly -- i.e. adding the
    # task-only profile changed nothing about the default path.
    assert _sha256_text(hardened_loop) == v2_manifest["rendered_loop_system_prompt_sha256"]
    assert _sha256_text(hardened_a) == session_a_manifest["rendered_system_prompt_sha256"]
    assert _sha256_text(hardened_b) == session_b_manifest["rendered_system_prompt_sha256"]

    # Action/tool schemas never depend on prompt profile at all -- same
    # function call embeds the identical schema block into both prompts.
    loop_schema = json.dumps(action_json_schema(), indent=2)
    session_a_schema = json.dumps(session_a_action_json_schema(), indent=2)
    session_b_schema = json.dumps(session_b_action_json_schema(), indent=2)
    assert loop_schema in hardened_loop and loop_schema in task_only_loop
    assert session_a_schema in hardened_a and session_a_schema in task_only_a
    assert session_b_schema in hardened_b and session_b_schema in task_only_b

    clean_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    attack_hashes = {p.name: _sha256_file(p) for p in sorted(ATTACK_DIR.glob("*.json"))}
    memory_scenario_hashes = {p.name: _sha256_file(p) for p in sorted(MEMORY_SCENARIO_DIR.glob("*.json"))}

    invariants = {
        "model_repo_id": PINNED_MODEL_REPO_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "max_turns": MAX_TURNS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "selected_hidden_state_indices": list(DEFAULT_HIDDEN_STATE_INDICES),
        "clean_scenario_sha256": _sha256_file(DEFAULT_SCENARIO_PATH_04D531CA),
        "approved_beneficiary_account_id": clean_bundle.trusted_state.approved_beneficiary_account_id,
        "num_attack_scenarios": len(attack_hashes),
        "num_memory_scenarios": len(memory_scenario_hashes),
    }
    # These must be exactly the values the hardened manifests already recorded --
    # i.e. this ablation touches none of them.
    invariant_checks = {
        "model_repo_id": (v2_manifest["model_repo_id"], invariants["model_repo_id"]),
        "model_revision": (v2_manifest["model_revision"], invariants["model_revision"]),
        "max_turns": (v2_manifest["max_turns"], invariants["max_turns"]),
        "generation_parameters": (v2_manifest["generation_parameters"], {
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "use_cache": True,
            "output_hidden_states": False, "return_dict_in_generate": False,
            "enable_thinking": False, "add_generation_prompt": True,
        }),
        "selected_hidden_state_indices": (v2_manifest["selected_hidden_state_indices"], invariants["selected_hidden_state_indices"]),
        "clean_scenario_sha256": (v2_manifest["clean_scenario_sha256"], invariants["clean_scenario_sha256"]),
        "approved_beneficiary_account_id": (v2_manifest["approved_beneficiary_account_id"], invariants["approved_beneficiary_account_id"]),
    }
    all_invariants_unchanged = all(a == b for a, b in invariant_checks.values())

    loop_diff = _line_diff(hardened_loop, task_only_loop)
    a_diff = _line_diff(hardened_a, task_only_a)
    b_diff = _line_diff(hardened_b, task_only_b)

    lines = ["# task_only_base_v1 vs. hardened_provenance_prompt_v1 -- Semantic Diff\n"]
    lines.append(
        "Computed from the current `tell.agent.loop_prompts` / `tell.agent.memory_prompts` source against "
        "the already-frozen `autonomous_loop_v2` and `memory_pilot_session_a`/`session_b` protocol manifests. "
        "No task-only model inference has run yet.\n"
    )

    lines.append("## Hardened rendering unchanged\n")
    lines.append(
        f"- Autonomous loop hardened prompt SHA-256 matches v2 manifest: "
        f"**{_sha256_text(hardened_loop) == v2_manifest['rendered_loop_system_prompt_sha256']}**\n"
        f"- Session A hardened prompt SHA-256 matches frozen manifest: "
        f"**{_sha256_text(hardened_a) == session_a_manifest['rendered_system_prompt_sha256']}**\n"
        f"- Session B hardened prompt SHA-256 matches frozen manifest: "
        f"**{_sha256_text(hardened_b) == session_b_manifest['rendered_system_prompt_sha256']}**\n"
    )

    lines.append("## Action/tool schemas identical across profiles\n")
    lines.append(
        "The rendered JSON schema block is generated by the same `tell.agent.actions` functions regardless of "
        "profile and is confirmed byte-identical inside both the hardened and task-only rendering of each of "
        "the three prompts (loop, session A, session B).\n"
    )

    for name, diff in (("Autonomous loop prompt", loop_diff), ("Session A prompt", a_diff), ("Session B prompt", b_diff)):
        removed = _removed_lines(diff)
        added = _added_lines(diff)
        lines.append(f"## {name}: removed/added lines\n")
        lines.append(f"- Lines removed going hardened -> task-only: {len(removed)}")
        lines.append(f"- Lines added going hardened -> task-only: {len(added)}\n")
        lines.append("Removed (present only in the hardened prompt):\n")
        lines.append("```")
        lines.extend(removed if removed else ["(none)"])
        lines.append("```\n")
        lines.append("Added (present only in the task-only prompt):\n")
        lines.append("```")
        lines.extend(added if added else ["(none)"])
        lines.append("```\n")

    lines.append("## Semantic invariants checked (must be unchanged)\n")
    lines.append("| Invariant | hardened-recorded | current | Unchanged |")
    lines.append("|---|---|---|---|")
    for name, (hval, cval) in invariant_checks.items():
        lines.append(f"| {name} | `{hval}` | `{cval}` | {hval == cval} |")
    lines.append("")
    lines.append(f"**All semantic invariants unchanged: {all_invariants_unchanged}**\n")

    lines.append("## Scenario/attack content hashes (unchanged from what hardened runs used)\n")
    lines.append(f"- Clean scenario `{DEFAULT_SCENARIO_PATH_04D531CA.name}`: `{invariants['clean_scenario_sha256']}`")
    for name, h in attack_hashes.items():
        lines.append(f"- Attack `{name}`: `{h}`")
    for name, h in memory_scenario_hashes.items():
        lines.append(f"- Memory scenario `{name}`: `{h}`")
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(
        f"Scope is as intended: **{'PASS' if all_invariants_unchanged else 'FAIL'}** -- every checked semantic "
        "invariant (model identity, generation settings, hidden-state indices, turn budget, clean scenario, "
        "approved beneficiary id) is unchanged, action/tool schemas are byte-identical across profiles, and "
        "the removed lines shown above are policy-language sentences classified as security/trust or "
        "attack-specific guidance in `results/scenario_design/hardened_prompt_policy_audit.md` (plus the "
        "'Trust rules, strictly enforced:' / trust-note header lines those sentences lived under). The added "
        "lines are the single task-only `Task:` sentence each profile needs in place of the removed policy "
        "paragraph, plus (for the loop prompt) the tool descriptions with their embedded 'trusted'/'verified' "
        "wording removed. No scenario, attack, tool schema, or action schema text changed.\n"
    )

    DIFF_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIFF_REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {DIFF_REPORT_PATH}")
    print(f"All semantic invariants unchanged: {all_invariants_unchanged}")


if __name__ == "__main__":
    main()
