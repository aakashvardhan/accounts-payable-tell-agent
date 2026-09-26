"""CPU-only tests for tell.registry.prior_document_registry_v1_1 (Part 2
of the runtime-integration milestone). Reads only already-existing,
already-frozen historical files; writes nothing back to any of them."""

from __future__ import annotations

from pathlib import Path

import pytest

from tell.registry.prior_document_registry import resolve_prior_docids
from tell.registry.prior_document_registry_v1_1 import (
    REGISTRY_VERSION_V1_1,
    MissingRegisteredSourceError,
    build_registry_v1_1,
    detect_unregistered_eligible_sources,
)

REPO = "/home/hp5/tell"


def test_v1_1_registry_supersets_v1_and_covers_768_documents():
    from tell.registry.prior_document_registry import build_registry_from_frozen_manifests

    v1_reg = build_registry_from_frozen_manifests(REPO)
    v1_1_reg, report = build_registry_v1_1(REPO)
    v1_docids = resolve_prior_docids(v1_reg)
    v1_1_docids = resolve_prior_docids(v1_1_reg)
    assert v1_docids <= v1_1_docids  # strictly a superset
    assert len(v1_1_docids) == 768
    assert report.unique_documents_total == 768
    assert report.unique_documents_from_new_sources == 92


def test_v1_registry_unchanged_by_importing_v1_1():
    from tell.registry.prior_document_registry import REGISTRY_VERSION, build_registry_from_frozen_manifests

    reg = build_registry_from_frozen_manifests(REPO)
    assert reg.registry_version == REGISTRY_VERSION == "prior_document_registry_v1"
    assert all(e.registry_version == "prior_document_registry_v1" for e in reg.entries)


def test_v1_1_entries_tagged_with_v1_1_version():
    reg, _ = build_registry_v1_1(REPO)
    assert reg.registry_version == REGISTRY_VERSION_V1_1
    assert all(e.registry_version == REGISTRY_VERSION_V1_1 for e in reg.entries)


def test_source_counts_match_verified_inventory():
    _, report = build_registry_v1_1(REPO)
    assert report.source_counts == {
        "legacy_scenario_fixture": 2,
        "probe_dataset_v1_selection": 20,
        "lora_dataset_v1_selection": 60,
        "dataset_inspection_pilot_cohort": 6,
        "dataset_inspection_clean_candidate": 5,
        "dataset_inspection_sample_rendering": 10,
    }


def test_probe_v1_1_is_reprocessing_not_a_new_source():
    _, report = build_registry_v1_1(REPO)
    assert report.reprocessed_not_new["probe_dataset_v1_1_new_documents"] == 0


# ---------------------------------------------------------------------
# Duplicate / conflict / missing / unregistered / ambiguous detection
# ---------------------------------------------------------------------


def test_overlapping_sources_are_deduplicated_first_use_wins_and_tracked():
    _, report = build_registry_v1_1(REPO)
    assert report.overlaps  # at least one overlap exists (verified: 11 total across sources)
    assert sum(report.overlaps.values()) > 0


def test_missing_registered_source_fails_loudly(tmp_path, monkeypatch):
    import tell.registry.prior_document_registry_v1_1 as mod

    monkeypatch.setattr(mod, "PILOT_COHORT_MANIFEST", "results/dataset_inspection/DOES_NOT_EXIST.json")
    with pytest.raises(MissingRegisteredSourceError):
        mod.build_registry_v1_1(REPO)


def test_detect_unregistered_eligible_sources_is_narrow_not_recursive():
    import inspect

    source = inspect.getsource(detect_unregistered_eligible_sources)
    assert "rglob" not in source
    assert "walk" not in source
    found = detect_unregistered_eligible_sources(Path(REPO))
    # Known, manually-reviewed narrative reports that add no new docids
    # (see prior_document_registry_v1_1's module docstring) -- this proves
    # the detector actually finds real candidates, not that it's a no-op.
    assert set(found) == {
        "results/dataset_inspection/official_docile_inspection.json",
        "results/dataset_inspection/official_docile_report.md",
        "results/dataset_inspection/pilot_cohort_review.md",
        "results/dataset_inspection/clean_candidate_review.md",
    }


def test_ambiguous_source_metadata_detected(tmp_path):
    import json

    from tell.registry.prior_document_registry_v1_1 import _docids_from_pilot_cohort_manifest, AmbiguousSourceMetadataError

    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps({"not-24-hex": {}}))
    with pytest.raises(AmbiguousSourceMetadataError):
        _docids_from_pilot_cohort_manifest(bad)


def test_generated_output_directories_are_not_scanned_as_sources():
    import tell.registry.prior_document_registry_v1_1 as mod

    allowlisted_paths = {
        *mod.SCENARIO_FIXTURE_FILES, mod.PROBE_DATASET_V1_SELECTION, mod.PROBE_DATASET_V1_1_SELECTION,
        mod.LORA_DATASET_V1_SELECTION, mod.PILOT_COHORT_MANIFEST, mod.CLEAN_CANDIDATE_IMAGES_DIR, mod.SAMPLE_IMAGES_DIR,
    }
    for generated_root in ("results/activations", "results/traces", "results/evaluation", "results/probe_training", "results/economics"):
        assert not any(p == generated_root or p.startswith(generated_root + "/") for p in allowlisted_paths)


def test_build_does_not_write_anything(tmp_path):
    import inspect

    import tell.registry.prior_document_registry_v1_1 as mod

    source = inspect.getsource(mod.build_registry_v1_1)
    assert "write_text" not in source
    assert ".write(" not in source
