"""CPU-only tests for tell.registry.prior_document_registry_v1_2_exclusions
(registry cleanup milestone, Part 4). Read-only against existing files;
writes nothing back anywhere."""

from __future__ import annotations

from tell.registry.prior_document_registry import resolve_prior_docids
from tell.registry.prior_document_registry_v1_1 import build_registry_v1_1, detect_unregistered_eligible_sources
from tell.registry.prior_document_registry_v1_2_exclusions import (
    REVIEWED_EXCLUSIONS,
    ReviewedExclusion,
    ReviewedExclusionReason,
    classify_findings,
)

REPO = "/home/hp5/tell"


def test_v1_1_registry_and_report_unchanged_by_importing_v1_2():
    from pathlib import Path

    reg, report = build_registry_v1_1(REPO)
    assert len(resolve_prior_docids(reg)) == 768
    assert set(report.unregistered_eligible_sources) == set(detect_unregistered_eligible_sources(Path(REPO)))


def test_reviewed_exclusions_are_fixed_at_zero_documents():
    for e in REVIEWED_EXCLUSIONS:
        assert e.contributes_documents == 0


def test_reviewed_exclusion_cannot_represent_a_nonzero_count():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReviewedExclusion(source_path="x", exclusion_reason=ReviewedExclusionReason.WHOLE_CORPUS_STATISTICS_NO_SELECTION, contributes_documents=1, notes="test")


def test_all_four_known_sources_are_matched_and_classified():
    result = classify_findings(REPO)
    assert len(result.reviewed_exclusions) == 4
    matched_paths = {e.source_path for e in result.reviewed_exclusions}
    assert matched_paths == {
        "results/dataset_inspection/official_docile_inspection.json",
        "results/dataset_inspection/official_docile_report.md",
        "results/dataset_inspection/pilot_cohort_review.md",
        "results/dataset_inspection/clean_candidate_review.md",
    }


def test_final_unexplained_count_is_zero():
    result = classify_findings(REPO)
    assert result.unexplained_eligible_sources == ()
    assert result.is_fully_explained is True


def test_classification_distinguishes_three_buckets():
    reg, migration_report = build_registry_v1_1(REPO)
    result = classify_findings(REPO)
    registered_document_count = len(resolve_prior_docids(reg))
    reviewed_exclusion_count = len(result.reviewed_exclusions)
    unexplained_count = len(result.unexplained_eligible_sources)
    # Three genuinely disjoint counters: no overlap between a registered
    # DOCUMENT count and a reviewed SOURCE-FILE count is expected or
    # meaningful (different units), but all three must be independently
    # reportable and the unexplained one must be zero.
    assert registered_document_count == 768
    assert reviewed_exclusion_count == 4
    assert unexplained_count == 0
