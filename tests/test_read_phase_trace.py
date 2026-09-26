"""Tests for the read-phase demonstration trace (run_clean_read_phase.py)."""

from __future__ import annotations

import pytest

from run_clean_read_phase import PERMITTED_VARIABLE_KEYS, TRACE_PATH, build_trace

from tell.evaluation.scenario import (
    DEFAULT_SCENARIO_PATH,
    DEFAULT_SCENARIO_PATH_04D531CA,
    ScenarioBundle,
    load_scenario,
)

FORBIDDEN_EVALUATION_MARKERS = [
    "is_attacked", "attack_surface", "expected_safe_action",
    "expected_beneficiary_account_id", "expected_amount",
    "expected_currency", "expected_outcome", "evaluation_labels", "probe_label",
]


@pytest.fixture(scope="module")
def bundle() -> ScenarioBundle:
    return load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)


def _strip_permitted_keys(records: list[dict]) -> list[dict]:
    stripped = []
    for record in records:
        stripped.append({k: v for k, v in record.items() if k not in PERMITTED_VARIABLE_KEYS})
    return stripped


def test_trace_is_deterministic_apart_from_permitted_runtime_metadata(bundle: ScenarioBundle):
    trace_a = build_trace(bundle)
    trace_b = build_trace(bundle)
    assert trace_a == trace_b  # build_trace itself stamps no wall-clock fields at all
    assert _strip_permitted_keys(trace_a) == _strip_permitted_keys(trace_b)


def test_trace_records_exactly_three_tool_calls_in_order(bundle: ScenarioBundle):
    trace = build_trace(bundle)
    tool_calls = [r for r in trace if r["record_type"] == "tool_call"]
    assert [r["tool_name"] for r in tool_calls] == ["read_email", "read_invoice", "get_vendor_record"]
    assert [r["step"] for r in tool_calls] == [1, 2, 3]


def test_trace_records_no_payment_attempted(bundle: ScenarioBundle):
    trace = build_trace(bundle)
    footer = trace[-1]
    assert footer["record_type"] == "run_footer"
    assert footer["payment_attempted"] is False
    assert footer["payment_tool_called"] is False


def test_trace_contains_no_evaluation_only_markers(bundle: ScenarioBundle):
    trace = build_trace(bundle)
    import json

    dumped = json.dumps(trace)
    for marker in FORBIDDEN_EVALUATION_MARKERS:
        assert marker not in dumped, f"trace leaked {marker}"


def test_trace_contains_operational_provenance_per_tool_call(bundle: ScenarioBundle):
    trace = build_trace(bundle)
    for record in trace:
        if record["record_type"] != "tool_call":
            continue
        provenance = record["result"]["provenance"]
        assert provenance is not None
        assert "source_type" in provenance
        assert "source_id" in provenance
        assert "provenance" in provenance
        assert "recorded_at" in provenance
        assert "trust_boundary" in provenance


def test_written_trace_file_matches_build_trace_apart_from_timestamps(bundle: ScenarioBundle):
    import json

    assert TRACE_PATH.exists(), "run scripts/run_clean_read_phase.py before running this test"
    with TRACE_PATH.open() as f:
        on_disk = [json.loads(line) for line in f]
    expected = build_trace(bundle)
    assert _strip_permitted_keys(on_disk) == _strip_permitted_keys(expected)


def test_existing_002f9b82_scenario_still_loads():
    """Cross-check that extending the schema for this scenario did not
    break the original clean_002f9b82_v1 scenario."""
    old_bundle = load_scenario(DEFAULT_SCENARIO_PATH)
    assert old_bundle.scenario_id == "clean_002f9b82_v1"
    assert old_bundle.untrusted_inputs.invoice_document.operational_provenance is None
    assert old_bundle.trusted_state.operational_provenance is None
