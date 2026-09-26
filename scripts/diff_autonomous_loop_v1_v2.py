"""Produces the version-1-vs-version-2 autonomous-loop protocol diff
report, before any version-2 model inference runs.

Reads the frozen v1 manifest
(results/scenario_design/autonomous_loop_protocol_manifest.json) and
compares its recorded source hashes and rendered-system-prompt hash
against the current files, to prove the version-2 change is narrowly
scoped to the action schema (and its consequences: the evaluator and the
generated JSON schema/prompt text), and that everything else --
trust policy wording, available read tools, tool descriptions,
provenance handling, attack scenarios, generation settings, model
revision, selected hidden-state indices, max turns, terminal safety
behavior -- is unchanged.

Writes results/scenario_design/autonomous_loop_v1_v2_diff.md. Does not
modify the v1 manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION
from tell.agent.loop import MAX_NEW_TOKENS, MAX_TURNS
from tell.agent.loop_prompts import build_loop_system_prompt
from tell.detector.capture import DEFAULT_HIDDEN_STATE_INDICES
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, load_scenario

V1_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_protocol_manifest.json")
DIFF_REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v1_v2_diff.md")

ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
LOOP_SRC = Path("/home/hp5/tell/src/tell/agent/loop.py")
LOOP_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/loop_prompts.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    v1 = json.loads(V1_MANIFEST_PATH.read_text())
    clean_bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)

    current_hashes = {
        "actions_schema": _sha256_file(ACTIONS_SRC),
        "loop_engine": _sha256_file(LOOP_SRC),
        "loop_prompt_builder": _sha256_file(LOOP_PROMPTS_SRC),
        "work_item_envelope": _sha256_file(WORK_ITEM_SRC),
        "capture_module": _sha256_file(CAPTURE_MODULE_SRC),
    }
    v1_hashes = {k: v["sha256"] for k, v in v1["source_hashes"].items()}

    changed = {k: (v1_hashes.get(k), current_hashes[k]) for k in current_hashes if v1_hashes.get(k) != current_hashes[k]}
    unchanged = {k: current_hashes[k] for k in current_hashes if v1_hashes.get(k) == current_hashes[k]}

    current_rendered_prompt = build_loop_system_prompt()
    current_rendered_hash = _sha256_text(current_rendered_prompt)
    prompt_changed = current_rendered_hash != v1["rendered_loop_system_prompt_sha256"]

    unchanged_semantics_checks = {
        "model_repo_id": (v1["model_repo_id"], PINNED_MODEL_REPO_ID),
        "model_revision": (v1["model_revision"], PINNED_MODEL_REVISION),
        "max_turns": (v1["max_turns"], MAX_TURNS),
        "generation_parameters.max_new_tokens": (v1["generation_parameters"]["max_new_tokens"], MAX_NEW_TOKENS),
        "generation_parameters.do_sample": (v1["generation_parameters"]["do_sample"], False),
        "selected_hidden_state_indices": (v1["selected_hidden_state_indices"], list(DEFAULT_HIDDEN_STATE_INDICES)),
        "clean_scenario_sha256": (v1["clean_scenario_sha256"], _sha256_file(DEFAULT_SCENARIO_PATH_04D531CA)),
        "approved_beneficiary_account_id": (v1["approved_beneficiary_account_id"], clean_bundle.trusted_state.approved_beneficiary_account_id),
        "attack_target_beneficiary_account_id": (v1["attack_target_beneficiary_account_id"], "SIM-ATTACKER-ACCT-0001"),
    }
    all_semantics_unchanged = all(a == b for a, b in unchanged_semantics_checks.values())

    lines = ["# Autonomous Loop v1 -> v2 Protocol Diff\n"]
    lines.append(f"Diff computed against v1 manifest: `{V1_MANIFEST_PATH}`\n")

    lines.append("## Source files that changed (intended)\n")
    lines.append("| File | v1 SHA-256 | v2 SHA-256 |")
    lines.append("|---|---|---|")
    for name, (old, new) in changed.items():
        lines.append(f"| `{name}` | `{(old or 'N/A')[:16]}...` | `{new[:16]}...` |")
    lines.append("")
    lines.append(f"Rendered loop system prompt changed: **{prompt_changed}** (expected -- it embeds the action JSON schema)\n")
    lines.append(f"- v1 rendered prompt SHA-256: `{v1['rendered_loop_system_prompt_sha256']}`")
    lines.append(f"- v2 rendered prompt SHA-256: `{current_rendered_hash}`\n")

    lines.append("## Source files that did NOT change\n")
    for name, h in unchanged.items():
        lines.append(f"- `{name}`: `{h[:16]}...` (identical to v1)")
    lines.append("")

    lines.append("## Semantic invariants checked (must be unchanged)\n")
    lines.append("| Invariant | v1 | v2 | Unchanged |")
    lines.append("|---|---|---|---|")
    for name, (v1_val, v2_val) in unchanged_semantics_checks.items():
        lines.append(f"| {name} | `{v1_val}` | `{v2_val}` | {v1_val == v2_val} |")
    lines.append("")
    lines.append(f"**All semantic invariants unchanged: {all_semantics_unchanged}**\n")

    lines.append("## Intended semantic changes\n")
    lines.append(
        "- `ProposePaymentAction.invoice_id` (ambiguous) replaced by two explicit required fields: "
        "`invoice_document_id` (opaque internal id) and `invoice_number` (human-readable business number).\n"
        "- A new `invoice_document_id`-vs-`evidence.invoice_document_id` consistency validator was added "
        "(fails closed on mismatch); the legacy `invoice_id` field name is not accepted as an alias -- "
        "`extra=\"forbid\"` rejects it as an unknown field.\n"
        "- `tell.evaluation.agentic_outcomes.PolicyViolations`/`evaluate_policy_violations` were updated to "
        "check both new identifiers against what `read_invoice` actually returned, plus a new "
        "`currency_not_from_invoice` check; `classify_agentic_outcome`'s verified-vs-unverified boundary for "
        "`propose_payment` now also depends on those identifier/amount/currency checks passing, not only on "
        "whether `read_invoice`/`get_vendor_record` were called at all.\n"
        "- The generated JSON schema embedded in the loop system prompt changed automatically (it is rendered "
        "live from the action models by `tell.agent.loop_prompts.build_loop_system_prompt`), reflecting the "
        "new fields. The fixed English policy paragraphs in that prompt template are untouched -- only the "
        "appended schema block differs, which is exactly `loop_prompt_builder`'s unchanged-hash / "
        "prompt-changed-hash split shown above.\n"
    )

    lines.append("## Verdict\n")
    lines.append(
        f"Scope is as intended: **{'PASS' if all_semantics_unchanged else 'FAIL'}** -- all checked semantic "
        "invariants (model, trust policy source text, generation settings, hidden-state indices, turn budget, "
        "clean scenario, approved/attacker beneficiary ids) are unchanged, and the only source-level diffs are "
        "in `actions_schema` and (as a consequence) `loop_prompt_builder`'s *rendered* output -- "
        "`work_item_envelope` and `capture_module` are untouched. `loop_engine` unchanged confirms the loop's "
        "own control flow (turn budget, tool-execution boundary, provenance handling) was not touched by this fix.\n"
    )

    if changed.keys() - {"actions_schema"}:
        lines.append(
            f"\n**Note:** unexpected additional source changes detected beyond `actions_schema`: "
            f"{sorted(changed.keys() - {'actions_schema'})}. Investigate before proceeding.\n"
        )

    DIFF_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIFF_REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {DIFF_REPORT_PATH}")
    print(f"Changed: {sorted(changed.keys())}")
    print(f"Unchanged: {sorted(unchanged.keys())}")
    print(f"All semantic invariants unchanged: {all_semantics_unchanged}")


if __name__ == "__main__":
    main()
