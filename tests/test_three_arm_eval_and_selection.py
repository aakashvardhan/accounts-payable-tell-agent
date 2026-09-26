"""CPU-only tests for: resumable evaluation generation, duplicate/conflict
detection, the Agent-1 reuse audit, and the corrected checkpoint-selection
formula's always-block resistance. No model, no GPU, no torch import at
module scope (three_arm_eval.py's heavy imports are inside run_generation(),
never at module level)."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/hp5/tell/scripts")

import agent_s_v1.select_checkpoint as sel  # noqa: E402
import agent_s_v1.three_arm_eval as tae  # noqa: E402


# --- resumable generation --------------------------------------------------------------


def _row(sid):
    return {"sample_id": sid, "contract": "session_b_processing", "class": "clean"}


def test_resumable_generation_skips_already_completed(tmp_path):
    out_path = tmp_path / "gen.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    meta = {"hash": "abc123"}
    calls = []

    def gen_fn(r):
        calls.append(r["sample_id"])
        return {"raw_output": "x", "action": {"action": "fail_closed"}, "action_type": "fail_closed", "parse_outcome": "valid", "generated_tokens": 3, "generation_seconds": 0.1}

    rows = [_row("a"), _row("b")]
    tae.generate_resumable("agent_1", rows, gen_fn, meta, out_path, fail_path)
    assert calls == ["a", "b"]

    calls.clear()
    tae.generate_resumable("agent_1", rows, gen_fn, meta, out_path, fail_path)
    assert calls == [], "already-completed rows must not be regenerated"


def test_resumable_generation_appends_only_missing_rows(tmp_path):
    out_path = tmp_path / "gen.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    meta = {"hash": "abc123"}

    def gen_fn(r):
        return {"raw_output": "x", "action": {"action": "fail_closed"}, "action_type": "fail_closed", "parse_outcome": "valid", "generated_tokens": 3, "generation_seconds": 0.1}

    tae.generate_resumable("agent_1", [_row("a")], gen_fn, meta, out_path, fail_path)
    tae.generate_resumable("agent_1", [_row("a"), _row("b")], gen_fn, meta, out_path, fail_path)
    completed = tae._load_completed(out_path)
    assert set(completed) == {"a", "b"}


def test_resumable_generation_detects_conflicting_run_config(tmp_path):
    out_path = tmp_path / "gen.jsonl"
    fail_path = tmp_path / "fail.jsonl"

    def gen_fn(r):
        return {"raw_output": "x", "action": {"action": "fail_closed"}, "action_type": "fail_closed", "parse_outcome": "valid", "generated_tokens": 3, "generation_seconds": 0.1}

    tae.generate_resumable("agent_1", [_row("a")], gen_fn, {"hash": "config-1"}, out_path, fail_path)
    try:
        tae.generate_resumable("agent_1", [_row("a")], gen_fn, {"hash": "config-2-DIFFERENT"}, out_path, fail_path)
        raise AssertionError("expected a SystemExit for conflicting run configuration")
    except SystemExit:
        pass


def test_resumable_generation_saves_failures_separately(tmp_path):
    out_path = tmp_path / "gen.jsonl"
    fail_path = tmp_path / "fail.jsonl"

    def gen_fn(r):
        if r["sample_id"] == "bad":
            raise RuntimeError("boom")
        return {"raw_output": "x", "action": {"action": "fail_closed"}, "action_type": "fail_closed", "parse_outcome": "valid", "generated_tokens": 3, "generation_seconds": 0.1}

    tae.generate_resumable("agent_1", [_row("good"), _row("bad")], gen_fn, {"hash": "h"}, out_path, fail_path)
    completed = tae._load_completed(out_path)
    assert set(completed) == {"good"}
    failures = [json.loads(l) for l in open(fail_path)]
    assert failures[0]["sample_id"] == "bad" and failures[0]["exception_type"] == "RuntimeError"


def test_resumable_generation_validates_row_before_marking_complete(tmp_path):
    out_path = tmp_path / "gen.jsonl"
    fail_path = tmp_path / "fail.jsonl"

    def gen_fn(r):
        return {"raw_output": None}  # missing required fields

    tae.generate_resumable("agent_1", [_row("a")], gen_fn, {"hash": "h"}, out_path, fail_path)
    assert tae._load_completed(out_path) == {}
    failures = [json.loads(l) for l in open(fail_path)]
    assert failures and failures[0]["sample_id"] == "a"


# --- Agent-1 reuse audit -----------------------------------------------------------------


def test_agent1_reuse_audit_reports_reuse_exact_on_real_repo_state():
    """Real, non-mocked check against the actual repo files -- confirms
    the already-produced audit.json still reflects REUSE_EXACT (or, if the
    corpus changed since, fails loudly rather than silently trusting a
    stale file)."""
    import agent_s_v1.audit_agent1_reuse as audit_mod

    report = audit_mod.main()
    assert report["verdict"] == "REUSE_EXACT"
    for p in ("test", "lexical_challenge", "delayed_memory_ood"):
        assert report["fields_compared"]["sample_ids_and_coverage"]["detail"][p]["missing_count"] == 0


# --- selection formula: always-block cannot win -----------------------------------------


def test_harmonic_mean_is_zero_if_either_side_is_zero():
    assert sel.harmonic_mean(1.0, 0.0) == 0.0
    assert sel.harmonic_mean(0.0, 1.0) == 0.0
    assert sel.harmonic_mean(None, 0.9) == 0.0


def test_harmonic_mean_rewards_joint_success_over_perfect_containment_alone():
    always_block = sel.harmonic_mean(1.0, 0.0)  # never pays anyone -> perfect containment, zero completion
    balanced = sel.harmonic_mean(0.8, 0.8)  # imperfect but does both
    assert balanced > always_block
    assert always_block == 0.0


def test_summarize_and_rank_never_select_an_always_refuse_checkpoint_over_a_balanced_one():
    def mk_row(sid, cls, gold, generated, beneficiary_ok):
        act = {"action": generated}
        if generated == "propose_payment":
            act["beneficiary_account_id"] = "APPROVED" if beneficiary_ok else "WRONG"
        return {"sample_id": sid, "class": cls, "gold_action_type": gold,
               "generated_action_type": generated, "parse_outcome": "valid", "generated_tokens": 10,
               "unauthorized_payment": generated == "propose_payment" and not beneficiary_ok,
               "correct_clean_payment": generated == "propose_payment" and beneficiary_ok,
               "matches_gold": generated == gold}

    always_block_rows = (
        [mk_row(f"m{i}", "attacked", "submit_evidence_report", "submit_evidence_report", True) for i in range(5)]
        + [mk_row(f"c{i}", "clean", "propose_payment", "submit_evidence_report", False) for i in range(5)]  # never pays clean either
    )
    balanced_rows = (
        [mk_row(f"m{i}", "attacked", "submit_evidence_report", "submit_evidence_report", True) for i in range(5)]
        + [mk_row(f"c{i}", "clean", "propose_payment", "propose_payment", True) for i in range(5)]  # completes clean payments correctly
    )
    always_block_summary = sel.summarize(always_block_rows)
    balanced_summary = sel.summarize(balanced_rows)
    assert always_block_summary["clean_canonical_beneficiary_completion_rate"] == 0.0
    assert always_block_summary["primary_objective_harmonic_mean"] == 0.0
    assert balanced_summary["primary_objective_harmonic_mean"] > 0.0

    results = {"step_000100_epoch_end": {"summary": always_block_summary}, "step_000200_epoch_end": {"summary": balanced_summary}}
    ranked = sel.rank(results)
    assert ranked[0] == "step_000200_epoch_end", "the balanced checkpoint must rank above the always-block one"
