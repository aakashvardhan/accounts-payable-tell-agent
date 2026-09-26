"""CPU-only tests for the coverage-first LoRA sampler v1.1 (Part 3 of the
Tell-routing design). Reuses `lora_pretraining_v1.pipeline.build` (frozen
v2.2 integrity checks + pool construction, unchanged) and exercises only
the new `lora_pretraining_v1_1.sampler` module against that real pool --
no activations, no model, no GPU. This is a session-scoped, ~15s fixture
(same cost as the existing pretraining_contract_v1 tests), not a per-test
cost.
"""

from __future__ import annotations

import json

import pytest

from lora_pretraining_v1 import pipeline as P1
from lora_pretraining_v1_1 import sampler as S11

CFG_PATH = "/home/hp5/tell/configs/lora/enterprise_v2_2/sampler_v1_1.json"


@pytest.fixture(scope="session")
def v1_build():
    return P1.build(with_tokenizer=False)


@pytest.fixture(scope="session")
def cfg():
    return json.loads(open(CFG_PATH).read())


@pytest.fixture(scope="session")
def pool(v1_build):
    return v1_build["pool"]


@pytest.fixture(scope="session")
def three_epochs(pool, cfg):
    gvr_pool = [p for p in pool if p.action == S11.GVR]
    state = S11.GvrCoverageState.empty(gvr_pool)
    epochs = []
    rows = []
    for e in range(cfg["planned_epochs"]):
        items = S11.sample_epoch_v1_1(pool, cfg, e, state)
        epochs.append(items)
        rows.append(S11.epoch_coverage_row(pool, items, state, e))
    return epochs, rows, gvr_pool


def test_v1_reused_integrity_checks_all_pass(v1_build):
    failed = [c for c in v1_build["checks"] if not c["passed"]]
    assert not failed, failed


def test_gvr_pool_size_matches_expected_831_plus_240(pool, cfg):
    gvr = [p for p in pool if p.action == S11.GVR]
    assert len(gvr) == cfg["expected_gvr_pool_size"] == 1071
    assert sum(1 for p in gvr if p.source == "lora_v2_2") == cfg["expected_gvr_v2_2"] == 831
    assert sum(1 for p in gvr if p.source != "lora_v2_2") == cfg["expected_gvr_supplement"] == 240


# ---------------------------------------------------------------------
# 13: coverage-first sampler is deterministic
# ---------------------------------------------------------------------


def test_sampler_is_deterministic(pool, cfg):
    gvr_pool = [p for p in pool if p.action == S11.GVR]
    state_a = S11.GvrCoverageState.empty(gvr_pool)
    state_b = S11.GvrCoverageState.empty(gvr_pool)
    for e in range(cfg["planned_epochs"]):
        items_a = S11.sample_epoch_v1_1(pool, cfg, e, state_a)
        items_b = S11.sample_epoch_v1_1(pool, cfg, e, state_b)
        assert [p.sample_id for p in items_a] == [p.sample_id for p in items_b]


# ---------------------------------------------------------------------
# 14: all vendor-lookup examples covered before repetition
# ---------------------------------------------------------------------


def test_no_repeat_before_full_coverage(three_epochs):
    epochs, rows, gvr_pool = three_epochs
    n_pool = len(gvr_pool)
    cumulative_seen = 0
    for row in rows:
        if row["cumulative_coverage_fraction"] < 1.0:
            assert row["repeated_examples_this_epoch"] == 0, "an item repeated before the pool was fully covered"
        cumulative_seen = row["unique_vendor_lookup_examples_seen_cumulative"]
    assert cumulative_seen == n_pool  # full coverage achieved by the last planned epoch


def test_full_pool_covered_within_three_epochs(three_epochs):
    _, rows, _ = three_epochs
    assert rows[-1]["cumulative_coverage_fraction"] == 1.0
    assert rows[-1]["v2_2_covered_cumulative"] == rows[-1]["v2_2_pool_size"] == 831
    assert rows[-1]["supplement_covered_cumulative"] == rows[-1]["supplement_pool_size"] == 240


def test_no_item_repeated_within_a_single_epoch(three_epochs):
    epochs, _, _ = three_epochs
    for items in epochs:
        ids = [p.sample_id for p in items]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------
# 15: every epoch preserves the 30% cap and expected size
# ---------------------------------------------------------------------


def test_every_epoch_preserves_cap_and_size(three_epochs, cfg):
    epochs, _, _ = three_epochs
    for items in epochs:
        assert len(items) == cfg["epoch_size"] == 1655
        n_gvr = sum(1 for p in items if p.action == S11.GVR)
        assert n_gvr == 496
        # "cap" means at most 30%, not exactly 30%: 496/1655 = 29.97%.
        assert n_gvr / len(items) <= cfg["gvr_cap"]
        assert n_gvr / len(items) == pytest.approx(cfg["gvr_cap"], abs=0.005)


def test_document_exposure_ratio_within_existing_limit(three_epochs):
    _, rows, _ = three_epochs
    last = rows[-1]["per_document_exposure_cumulative"]
    assert last["max_over_mean"] <= 1.5


def test_pool_size_change_fails_loudly(pool, cfg):
    bad_cfg = {**cfg, "expected_gvr_pool_size": 999999}
    with pytest.raises(ValueError):
        S11.validate_pool_v1_1(pool, bad_cfg)


def test_non_gvr_items_still_fully_included_every_epoch(pool, three_epochs):
    epochs, _, _ = three_epochs
    non_gvr_ids = {p.sample_id for p in pool if p.action != S11.GVR}
    for items in epochs:
        epoch_non_gvr_ids = {p.sample_id for p in items if p.action != S11.GVR}
        assert epoch_non_gvr_ids == non_gvr_ids
