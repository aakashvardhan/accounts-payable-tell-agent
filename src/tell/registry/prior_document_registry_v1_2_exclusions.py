"""Prior-document registry v1.2: classifies v1.1's four manually-reviewed
"unregistered eligible sources" as explicit, typed, reviewed exclusions
instead of leaving them as an unexplained detector finding.

Additive only: `tell.registry.prior_document_registry` (v1, 676 documents)
and `tell.registry.prior_document_registry_v1_1` (v1.1, 768 documents) are
both imported unchanged and produce the identical document sets they
always did -- this module adds NO new document to either registry. It
adds a second, distinct kind of record, `ReviewedExclusion`, for a source
`detect_unregistered_eligible_sources` (v1.1) flags for human review that
a human then confirmed contributes zero documents. A `ReviewedExclusion`
is never a `PriorDocumentRegistryEntry` and is never fed to
`resolve_prior_docids` -- it would be meaningless there (it names a
source, not a document id).

Why this exists (the instruction this corrects)
--------------------------------------------------------------------------
v1.1's migration report listed the four files
`detect_unregistered_eligible_sources` found as free text in
"remaining_gaps," with a manual-review conclusion in prose. That is not a
structured, versioned record: a future automated check has no typed way
to confirm "this exact source was reviewed, by this process, and
contributes zero documents" versus "this finding was never looked at."
v1.2 fixes that without touching v1.1's own report or registry (both
remain byte-identical) by defining `REVIEWED_EXCLUSIONS` as first-class,
typed, versioned data, and `classify_findings` as the function that turns
v1.1's raw finding list into three disjoint buckets: registered document
sources (irrelevant here -- those are already in the registry, not
findings), reviewed exclusions (zero documents, explained), and
genuinely unexplained sources (the count `classify_findings` must drive
to zero, and does, given the current repository state).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tell.registry.prior_document_registry_v1_1 import detect_unregistered_eligible_sources

REGISTRY_VERSION_V1_2 = "prior_document_registry_v1_2_exclusions"


class ReviewedExclusionReason(str, Enum):
    WHOLE_CORPUS_STATISTICS_NO_SELECTION = "whole_corpus_statistics_no_document_selection"
    NARRATIVE_REVIEW_OF_ALREADY_REGISTERED_SOURCE = "narrative_review_of_already_registered_source"


class ReviewedExclusion(BaseModel):
    """One source path a human reviewed and confirmed contributes zero
    documents. `contributes_documents` is fixed at 0 by the type itself
    (not merely a convention) -- a source that turned out to contribute
    even one real document could never be represented here; it would need
    to become a `PriorDocumentRegistryEntry` in a future registry version
    instead."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = Field(min_length=1)
    exclusion_reason: ReviewedExclusionReason
    contributes_documents: int = Field(default=0, ge=0, le=0)
    review_version: str = REGISTRY_VERSION_V1_2
    notes: str = Field(min_length=1)


REVIEWED_EXCLUSIONS: tuple[ReviewedExclusion, ...] = (
    ReviewedExclusion(
        source_path="results/dataset_inspection/official_docile_inspection.json",
        exclusion_reason=ReviewedExclusionReason.WHOLE_CORPUS_STATISTICS_NO_SELECTION,
        notes="Whole-DocILE-corpus directory/file-count statistics (produced by scripts/inspect_official_docile.py); no specific document is selected or referenced by id.",
    ),
    ReviewedExclusion(
        source_path="results/dataset_inspection/official_docile_report.md",
        exclusion_reason=ReviewedExclusionReason.WHOLE_CORPUS_STATISTICS_NO_SELECTION,
        notes="Prose companion to official_docile_inspection.json; explicitly states 'no Tell components, scenarios, or models were implemented' -- inspection-only, no document selection.",
    ),
    ReviewedExclusion(
        source_path="results/dataset_inspection/pilot_cohort_review.md",
        exclusion_reason=ReviewedExclusionReason.NARRATIVE_REVIEW_OF_ALREADY_REGISTERED_SOURCE,
        notes="Field-by-field narrative review of the same 6 documents already registered via the dataset_inspection_pilot_cohort source (results/dataset_inspection/pilot_cohort_manifest.json); introduces no docid beyond that set.",
    ),
    ReviewedExclusion(
        source_path="results/dataset_inspection/clean_candidate_review.md",
        exclusion_reason=ReviewedExclusionReason.NARRATIVE_REVIEW_OF_ALREADY_REGISTERED_SOURCE,
        notes="Narrative description of the clean_candidate_scores.json 744-candidate ranking pass; the 5 documents actually reviewed/rendered are already registered via dataset_inspection_clean_candidate (results/dataset_inspection/clean_candidate_images/).",
    ),
)


class ClassificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reviewed_exclusions: tuple[ReviewedExclusion, ...]
    unexplained_eligible_sources: tuple[str, ...]

    @property
    def is_fully_explained(self) -> bool:
        return len(self.unexplained_eligible_sources) == 0


def classify_findings(repo_root: str | Path) -> ClassificationResult:
    """Re-runs v1.1's narrow (non-recursive) detector and classifies every
    finding as a reviewed exclusion or, if newly-found and not in
    `REVIEWED_EXCLUSIONS`, an unexplained eligible source. Read-only."""
    found = detect_unregistered_eligible_sources(Path(repo_root))
    reviewed_paths = {e.source_path for e in REVIEWED_EXCLUSIONS}
    matched = tuple(e for e in REVIEWED_EXCLUSIONS if e.source_path in found)
    unexplained = tuple(sorted(set(found) - reviewed_paths))
    return ClassificationResult(reviewed_exclusions=matched, unexplained_eligible_sources=unexplained)
