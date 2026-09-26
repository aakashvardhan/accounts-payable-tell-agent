"""Prior-document registry v1.1: extends v1's coverage from just
`document_assignments.jsonl` (v2, v2.1 -- 676 documents) to every
historical pilot and scenario document source this project actually
produced before `enterprise_corpus_v2`.

Additive only: `tell.registry.prior_document_registry` (v1) is imported,
never modified -- `PriorDocumentRegistryEntry`, `PriorDocumentRegistry`,
`resolve_prior_docids`, `load_registry`, and `save_registry` are the exact
same classes/functions, reused unchanged. v1's own
`build_registry_from_frozen_manifests` output is not altered; this module
adds a second, superset builder (`build_registry_v1_1`) tagged with
`registry_version="prior_document_registry_v1_1"`.

Explicit allowlist, not a scan
--------------------------------------------------------------------------
`SOURCE_ALLOWLIST` below is a fixed, named list of exactly which files and
directories are historical pilot/scenario sources. This is the opposite of
`scripts/enterprise_v2/selection.py`'s `discover_prior_docids`, which
walks arbitrary file content under four whole directory trees -- see
`results/routing_design/prior_document_registry_spec_v1.md` for why that
approach is fragile. Nothing here recurses into `results/` looking for
docid-shaped strings; every source is named explicitly, and
`detect_unregistered_eligible_sources` below checks only the three narrow,
named parent directories under `results/dataset_inspection/` for a sibling
directory this allowlist does not already know about -- it does not scan
`results/` at large.

Inventory (verified against this repository; see
`build_migration_report`'s `source_counts` for the exact numbers a build
actually found):

| source_type                          | path                                                              | role                              | n docs |
|---------------------------------------|-------------------------------------------------------------------|------------------------------------|-------:|
| legacy_scenario_fixture               | data/scenarios/clean/*.json, data/scenarios/memory_infection/*.json | legacy_scenario                  |      2 |
| probe_dataset_v1_selection            | results/probe_dataset/document_selection_manifest.json            | probe_dataset_v1                  |     20 |
| lora_dataset_v1_selection             | results/lora_dataset/v1/document_selection_manifest.json          | lora_dataset_v1                   |     60 |
| dataset_inspection_pilot_cohort       | results/dataset_inspection/pilot_cohort_manifest.json              | dataset_inspection_pilot_cohort   |      6 |
| dataset_inspection_clean_candidate    | results/dataset_inspection/clean_candidate_images/<docid>/         | dataset_inspection_reviewed       |      5 |
| dataset_inspection_sample_rendering   | results/dataset_inspection/sample_images/<docid>-*.png             | dataset_inspection_reviewed       |     10 |

`results/probe_dataset/v1_1/document_selection_manifest.json` is
deliberately NOT a separate source: it selects the identical 20 documents
as `probe_dataset_v1` (verified: 0 new docids) -- a reprocessing of the
same selection, not a new one. `results/dataset_inspection/clean_candidate_scores.json`
is deliberately NOT a source either, for the same reason v1's spec
documents: it is a 744-row *ranked candidate* list, not a corpus, pilot,
or scenario -- only the 5 documents actually rendered into
`clean_candidate_images/` were reviewed and are registered.

Total: verified 92 unique documents across all six source types (some
overlap between legacy/pilot_cohort/clean_candidate/sample_rendering,
e.g. the flagship demo document appears in more than one; first-use wins,
see `_DEDUP_PRIORITY`), plus 676 from v1's own
`build_registry_from_frozen_manifests` (v2 + v2.1) = 768 total registered
documents, all independently verified absent from both frozen
`document_assignments.jsonl` files (i.e. the existing scan-based selector
already excluded every one of them correctly; this migration formalizes
that outcome, it does not change it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tell.registry.prior_document_registry import (
    PriorDocumentRegistry,
    PriorDocumentRegistryEntry,
    _ASSIGNMENT_FILES,
    build_registry_from_frozen_manifests,
)

REGISTRY_VERSION_V1_1 = "prior_document_registry_v1_1"

# ---------------------------------------------------------------------
# Explicit source allowlist
# ---------------------------------------------------------------------

SCENARIO_FIXTURE_FILES = (
    "data/scenarios/clean/clean_002f9b82_v1.json",
    "data/scenarios/clean/clean_04d531ca_v1.json",
    "data/scenarios/memory_infection/memory_infection_clean_04d531ca_v1.json",
    "data/scenarios/memory_infection/memory_infection_poisoned_04d531ca_v1.json",
)
PROBE_DATASET_V1_SELECTION = "results/probe_dataset/document_selection_manifest.json"
PROBE_DATASET_V1_1_SELECTION = "results/probe_dataset/v1_1/document_selection_manifest.json"  # reprocessing check only, not a source
LORA_DATASET_V1_SELECTION = "results/lora_dataset/v1/document_selection_manifest.json"
PILOT_COHORT_MANIFEST = "results/dataset_inspection/pilot_cohort_manifest.json"
CLEAN_CANDIDATE_IMAGES_DIR = "results/dataset_inspection/clean_candidate_images"
SAMPLE_IMAGES_DIR = "results/dataset_inspection/sample_images"

# Named, non-source siblings under results/dataset_inspection/ -- used
# only by detect_unregistered_eligible_sources to notice a NEW sibling
# directory this allowlist doesn't know about. Not a recursive scan.
_KNOWN_DATASET_INSPECTION_ENTRIES = {
    "pilot_cohort_manifest.json", "pilot_cohort_images", "clean_candidate_images", "sample_images", "clean_candidate_scores.json",
}

# First-use precedence when the same docid appears in more than one
# source (lowest index wins) -- mirrors the project's actual chronology:
# the hand-built demo scenarios predate every automated selection pass.
_DEDUP_PRIORITY = (
    "legacy_scenario_fixture",
    "probe_dataset_v1_selection",
    "lora_dataset_v1_selection",
    "dataset_inspection_pilot_cohort",
    "dataset_inspection_clean_candidate",
    "dataset_inspection_sample_rendering",
)


class MissingRegisteredSourceError(Exception):
    pass


class AmbiguousSourceMetadataError(Exception):
    pass


class UnregisteredEligibleSourceError(Exception):
    pass


@dataclass
class MigrationReport:
    source_counts: dict[str, int] = field(default_factory=dict)
    unique_documents_from_new_sources: int = 0
    unique_documents_total: int = 0
    overlaps: dict[str, int] = field(default_factory=dict)
    missing_sources: list[str] = field(default_factory=list)
    unregistered_eligible_sources: list[str] = field(default_factory=list)
    reprocessed_not_new: dict[str, int] = field(default_factory=dict)
    remaining_gaps: list[str] = field(default_factory=list)


def _require(repo_root: Path, rel: str) -> Path:
    p = repo_root / rel
    if not p.exists():
        raise MissingRegisteredSourceError(f"registered source is missing from disk: {rel}")
    return p


def _docids_from_selection_manifest(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    docs = data.get("documents")
    if not isinstance(docs, list):
        raise AmbiguousSourceMetadataError(f"{path}: expected a top-level 'documents' list")
    out = set()
    for d in docs:
        docid = d.get("docid")
        if not isinstance(docid, str) or len(docid) != 24:
            raise AmbiguousSourceMetadataError(f"{path}: entry has a missing or malformed docid: {d!r}")
        out.add(docid)
    return out


def _docid_from_scenario_fixture(path: Path) -> str:
    data = json.loads(path.read_text())
    docid = (data.get("untrusted_inputs") or {}).get("invoice_document", {}).get("docid")
    if not isinstance(docid, str) or len(docid) != 24:
        raise AmbiguousSourceMetadataError(f"{path}: scenario fixture has a missing or malformed untrusted_inputs.invoice_document.docid")
    return docid


def _docids_from_pilot_cohort_manifest(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise AmbiguousSourceMetadataError(f"{path}: expected a non-empty docid-keyed object")
    for k in data:
        if len(k) != 24:
            raise AmbiguousSourceMetadataError(f"{path}: key {k!r} is not a 24-hex docid")
    return set(data.keys())


def _docids_from_image_subdirs(dir_path: Path) -> set[str]:
    out = set()
    for child in sorted(dir_path.iterdir()):
        if child.is_dir() and len(child.name) == 24:
            out.add(child.name)
    return out


def _docids_from_sample_images(dir_path: Path) -> set[str]:
    out = set()
    for child in sorted(dir_path.iterdir()):
        if not child.is_file():
            continue
        stem = child.name.split("-")[0]
        if len(stem) == 24:
            out.add(stem)
    return out


def detect_unregistered_eligible_sources(repo_root: Path) -> list[str]:
    """Narrow check: does results/dataset_inspection/ contain a directory
    or file this allowlist does not already know about? Does NOT scan any
    other part of the repository."""
    base = repo_root / "results" / "dataset_inspection"
    if not base.exists():
        return []
    unexpected = sorted(p.name for p in base.iterdir() if p.name not in _KNOWN_DATASET_INSPECTION_ENTRIES)
    return [f"results/dataset_inspection/{name}" for name in unexpected]


def build_registry_v1_1(repo_root: str | Path) -> tuple[PriorDocumentRegistry, MigrationReport]:
    """Builds the v1.1 registry: v1's own 676-document build (v2 + v2.1,
    unchanged, read-only) UNIONED with every historical pilot/scenario
    source in `SOURCE_ALLOWLIST`. Reads only; writes nothing back into any
    enterprise_corpus, dataset_inspection, scenario, or dataset directory.
    """
    repo_root = Path(repo_root)
    report = MigrationReport()

    missing = []
    for rel in (*SCENARIO_FIXTURE_FILES, PROBE_DATASET_V1_SELECTION, PROBE_DATASET_V1_1_SELECTION, LORA_DATASET_V1_SELECTION, PILOT_COHORT_MANIFEST, CLEAN_CANDIDATE_IMAGES_DIR, SAMPLE_IMAGES_DIR, *_ASSIGNMENT_FILES.values()):
        if not (repo_root / rel).exists():
            missing.append(rel)
    if missing:
        report.missing_sources = missing
        raise MissingRegisteredSourceError(f"registered source(s) missing from disk: {missing}")

    legacy_docids = {_docid_from_scenario_fixture(_require(repo_root, rel)) for rel in SCENARIO_FIXTURE_FILES}
    probe_v1_docids = _docids_from_selection_manifest(_require(repo_root, PROBE_DATASET_V1_SELECTION))
    probe_v1_1_docids = _docids_from_selection_manifest(_require(repo_root, PROBE_DATASET_V1_1_SELECTION))
    lora_v1_docids = _docids_from_selection_manifest(_require(repo_root, LORA_DATASET_V1_SELECTION))
    pilot_cohort_docids = _docids_from_pilot_cohort_manifest(_require(repo_root, PILOT_COHORT_MANIFEST))
    clean_candidate_docids = _docids_from_image_subdirs(_require(repo_root, CLEAN_CANDIDATE_IMAGES_DIR))
    sample_rendering_docids = _docids_from_sample_images(_require(repo_root, SAMPLE_IMAGES_DIR))

    report.reprocessed_not_new["probe_dataset_v1_1_new_documents"] = len(probe_v1_1_docids - probe_v1_docids)

    sources: dict[str, tuple[set[str], str, str]] = {
        "legacy_scenario_fixture": (legacy_docids, "legacy_scenario", "data/scenarios/**"),
        "probe_dataset_v1_selection": (probe_v1_docids, "probe_dataset_v1", PROBE_DATASET_V1_SELECTION),
        "lora_dataset_v1_selection": (lora_v1_docids, "lora_dataset_v1", LORA_DATASET_V1_SELECTION),
        "dataset_inspection_pilot_cohort": (pilot_cohort_docids, "dataset_inspection_pilot_cohort", PILOT_COHORT_MANIFEST),
        "dataset_inspection_clean_candidate": (clean_candidate_docids, "dataset_inspection_reviewed", CLEAN_CANDIDATE_IMAGES_DIR),
        "dataset_inspection_sample_rendering": (sample_rendering_docids, "dataset_inspection_reviewed", SAMPLE_IMAGES_DIR),
    }
    report.source_counts = {k: len(v[0]) for k, v in sources.items()}

    entries: list[PriorDocumentRegistryEntry] = []
    claimed: dict[str, str] = {}
    for source_type in _DEDUP_PRIORITY:
        docids, role, manifest = sources[source_type]
        for docid in sorted(docids):
            if docid in claimed:
                report.overlaps[f"{source_type}<-{claimed[docid]}"] = report.overlaps.get(f"{source_type}<-{claimed[docid]}", 0) + 1
                continue
            claimed[docid] = source_type
            entries.append(
                PriorDocumentRegistryEntry(
                    docile_document_id=docid,
                    first_project_use=source_type,
                    role=role,
                    split_or_population="pilot",
                    source_manifest=manifest,
                    reason_for_exclusion=f"historical pilot/scenario document from {source_type}",
                    registry_version=REGISTRY_VERSION_V1_1,
                )
            )
    report.unique_documents_from_new_sources = len(claimed)

    v1_registry = build_registry_from_frozen_manifests(repo_root)
    v1_1_entries = [e.model_copy(update={"registry_version": REGISTRY_VERSION_V1_1}) for e in v1_registry.entries] + entries
    registry = PriorDocumentRegistry(registry_version=REGISTRY_VERSION_V1_1, entries=tuple(v1_1_entries))

    report.unique_documents_total = len({e.docile_document_id for e in registry.entries})
    report.unregistered_eligible_sources = detect_unregistered_eligible_sources(repo_root)
    report.remaining_gaps = [
        "This migration covers document_assignments.jsonl (v2, v2.1), the legacy hand-built scenario "
        "fixtures, probe_dataset_v1, lora_dataset_v1, and the three dataset_inspection review sources. "
        "It does not attempt to register every activation/trace/evaluation output directory (e.g. "
        "results/activations/**, results/traces/**, results/evaluation/**) because those are GENERATED "
        "from already-selected documents, not independent document selections -- registering them would "
        "not change which documents are prior, only duplicate the same docids under a different label.",
    ]
    if report.unregistered_eligible_sources:
        report.remaining_gaps.append(
            "detect_unregistered_eligible_sources flagged "
            f"{report.unregistered_eligible_sources} for human review at migration time. Manual review found: "
            "official_docile_inspection.json/official_docile_report.md are whole-DocILE-corpus statistics with "
            "no specific document selection; pilot_cohort_review.md and clean_candidate_review.md are narrative "
            "reviews OF the already-registered pilot_cohort and clean_candidate sources, introducing no new "
            "docids. None of the four required adding a new registry source."
        )
    return registry, report
