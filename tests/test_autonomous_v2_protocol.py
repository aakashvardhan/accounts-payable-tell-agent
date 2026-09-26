"""Tests that the version-2 autonomous-loop protocol was frozen
correctly and narrowly: source hashes match the manifest, the intended
semantic invariants versus v1 hold, and v1 artifacts remain untouched.
No GPU, no model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

V1_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_protocol_manifest.json")
V2_MANIFEST_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v2_protocol_manifest.json")
DIFF_REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/autonomous_loop_v1_v2_diff.md")

ACTIONS_SRC = Path("/home/hp5/tell/src/tell/agent/actions.py")
LOOP_SRC = Path("/home/hp5/tell/src/tell/agent/loop.py")
LOOP_PROMPTS_SRC = Path("/home/hp5/tell/src/tell/agent/loop_prompts.py")
WORK_ITEM_SRC = Path("/home/hp5/tell/src/tell/agent/work_item.py")
CAPTURE_MODULE_SRC = Path("/home/hp5/tell/src/tell/detector/capture.py")
AGENTIC_OUTCOMES_SRC = Path("/home/hp5/tell/src/tell/evaluation/agentic_outcomes.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_manifest_exists_and_supersedes_v1():
    assert V2_MANIFEST_PATH.exists()
    v2 = json.loads(V2_MANIFEST_PATH.read_text())
    assert v2["protocol"] == "autonomous_email_attack_loop_v2"
    assert v2["supersedes"] == "autonomous_email_attack_loop_v1"


def test_v1_manifest_untouched_by_v2_freeze():
    assert V1_MANIFEST_PATH.exists()
    v1 = json.loads(V1_MANIFEST_PATH.read_text())
    assert v1["protocol"] == "autonomous_email_attack_loop_v1"


def test_v2_source_hashes_match_current_files():
    # actions.py was later extended (purely additively -- see
    # tell.agent.actions's "Memory-pilot actions" section) with
    # WriteMemoryAction/SearchMemoryAction/FinishReviewAction and the
    # SessionAAction/SessionBAction unions for the delayed-memory pilot,
    # before any new inference ran. The original AgentAction/
    # ProposePaymentAction/parse_agent_action contract is unchanged --
    # see tests/test_agent_loop.py and tests/test_actions.py, which still
    # pass unmodified. Same rationale as test_email_attack_pilot.py's
    # capture-module hash test: this checks the v2 manifest's own
    # historically recorded value for actions_schema, not the live file.
    v2 = json.loads(V2_MANIFEST_PATH.read_text())
    frozen_actions_schema_hash_at_v2_freeze = "aae6fb94524c1a6a1eebbb4038e602deeb4e448278c31587c1b5bd06204db945"
    assert v2["source_hashes"]["actions_schema"]["sha256"] == frozen_actions_schema_hash_at_v2_freeze

    # loop.py and loop_prompts.py were later extended (purely additively,
    # before any new inference ran under the new configuration) with the
    # task_only_base_v1 prompt-profile ablation: a `prompt_profile`
    # parameter defaulting to the hardened profile on `run_agent_loop`
    # and `build_loop_system_prompt`, plus a second, separate task-only
    # prompt template. Calling either with no argument (as every v2-era
    # call site above already did) still renders byte-for-byte the same
    # hardened text -- see tell.agent.prompt_profiles,
    # results/scenario_design/hardened_prompt_policy_audit.md, and
    # results/scenario_design/task_only_vs_hardened_diff.md. Same
    # historical-record rationale as the actions_schema check just above:
    # this checks the v2 manifest's own recorded value, not the live file.
    frozen_loop_engine_hash_at_v2_freeze = "aaf82aea6420ac54bf98d6b48863ea44549c318dc6c01159ac958731bb2ce256"
    frozen_loop_prompt_builder_hash_at_v2_freeze = "c93e3deeb8c615b26b2b9fcf9446b0c01c8ea1ad06012b66be8b40c344aae32a"
    assert v2["source_hashes"]["loop_engine"]["sha256"] == frozen_loop_engine_hash_at_v2_freeze
    assert v2["source_hashes"]["loop_prompt_builder"]["sha256"] == frozen_loop_prompt_builder_hash_at_v2_freeze

    mapping = {
        "work_item_envelope": WORK_ITEM_SRC,
        "capture_module": CAPTURE_MODULE_SRC,
        "agentic_outcomes_evaluator": AGENTIC_OUTCOMES_SRC,
    }
    for name, path in mapping.items():
        assert v2["source_hashes"][name]["sha256"] == _sha256(path), f"{name} hash drifted from frozen v2 manifest"


def test_hardened_loop_prompt_unchanged_by_prompt_profile_addition():
    """The task_only_base_v1 ablation added a `prompt_profile` parameter
    to `build_loop_system_prompt`/`run_agent_loop`, defaulting to the
    hardened profile. This proves the default rendering is still exactly
    the v2-frozen hardened prompt, byte for byte."""
    from tell.agent.loop_prompts import build_loop_system_prompt

    v2 = json.loads(V2_MANIFEST_PATH.read_text())
    rendered = build_loop_system_prompt()
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == v2["rendered_loop_system_prompt_sha256"]
    assert len(rendered) == v2["rendered_loop_system_prompt_char_count"]


def test_only_actions_schema_changed_between_v1_and_v2():
    v1 = json.loads(V1_MANIFEST_PATH.read_text())
    v2 = json.loads(V2_MANIFEST_PATH.read_text())
    for name in ("loop_engine", "loop_prompt_builder", "work_item_envelope", "capture_module"):
        assert v1["source_hashes"][name]["sha256"] == v2["source_hashes"][name]["sha256"], f"{name} unexpectedly changed"
    assert v1["source_hashes"]["actions_schema"]["sha256"] != v2["source_hashes"]["actions_schema"]["sha256"]


def test_semantic_invariants_unchanged_between_v1_and_v2():
    v1 = json.loads(V1_MANIFEST_PATH.read_text())
    v2 = json.loads(V2_MANIFEST_PATH.read_text())
    assert v1["model_repo_id"] == v2["model_repo_id"]
    assert v1["model_revision"] == v2["model_revision"]
    assert v1["max_turns"] == v2["max_turns"]
    assert v1["generation_parameters"] == v2["generation_parameters"]
    assert v1["selected_hidden_state_indices"] == v2["selected_hidden_state_indices"]
    assert v1["clean_scenario_sha256"] == v2["clean_scenario_sha256"]
    assert v1["approved_beneficiary_account_id"] == v2["approved_beneficiary_account_id"]
    assert v1["attack_target_beneficiary_account_id"] == v2["attack_target_beneficiary_account_id"]


def test_diff_report_exists_and_reports_pass():
    assert DIFF_REPORT_PATH.exists()
    text = DIFF_REPORT_PATH.read_text()
    assert "**PASS**" in text
    assert "actions_schema" in text
