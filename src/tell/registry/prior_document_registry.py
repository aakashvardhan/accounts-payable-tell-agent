"""Explicit prior-document registry (Part 11 of the Tell-routing design;
see results/routing_design/prior_document_registry_spec_v1.md for the full
migration write-up).

Today, `scripts/enterprise_v2/selection.py`'s frozen `discover_prior_docids`
decides which DocILE documents a future corpus build must exclude by
RECURSIVELY SCANNING results/, data/scenarios/, configs/, and demo/ for any
24-hex-character substring, in filenames and in the text content of
`.json`/`.jsonl`/`.md`/`.txt`/`.py`/`.csv` files. That scan is exactly the
selector that produced the frozen v2/v2.1/v2.2 corpora and MUST NOT be
changed -- see this task's non-negotiable constraints and
tests/test_enterprise_corpus_v2.py::test_every_prior_document_excluded,
which pins its exact behavior. It is also fragile in a way this task hit
directly while producing its own artifacts: any new text file placed
anywhere under those four roots that happens to mention a real docid
substring (even by coincidence, e.g. inside an unrelated hash) silently
changes what the NEXT corpus build excludes, with no explicit record of
why a given document was excluded or by which project.

This module defines the alternative for FUTURE corpus builders (v3 and
later): an explicit, versioned registry of exactly which documents were
already used, by whom, for what role, recorded once, and consumed by
simple set membership instead of a filesystem crawl. It does not replace
`discover_prior_docids` (that stays exactly as it is) and it produces no
change to any frozen v2/v2.1/v2.2 output -- `build_registry_from_frozen_manifests`
below only READS the already-frozen `document_assignments.jsonl` files
that v2 and v2.1 already publish; it writes nothing back to those corpora.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

REGISTRY_VERSION = "prior_document_registry_v1"


class PriorDocumentRegistryEntry(BaseModel):
    """One row: one DocILE document, its first project use, and why a
    future builder must exclude it. `reason_for_exclusion` is free text for
    human readability but is NEVER parsed by `resolve_prior_docids` --
    exclusion is decided purely by an entry existing for that docid, so a
    registry can never accidentally under- or over-exclude based on
    wording, unlike the current regex scan's dependence on incidental
    string content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    docile_document_id: str = Field(min_length=24, max_length=24, pattern=r"^[0-9a-f]{24}$")
    first_project_use: str = Field(min_length=1, description="e.g. 'enterprise_corpus_v2', 'enterprise_corpus_v2_1'")
    role: str = Field(min_length=1, description="e.g. population name such as 'probe_v2' or 'lora_v2'")
    split_or_population: str = Field(min_length=1, description="e.g. 'train', 'validation', 'test', 'benchmark'")
    source_manifest: str = Field(min_length=1, description="repo-relative path to the manifest/file this entry was derived from")
    reason_for_exclusion: str = Field(min_length=1)
    registry_version: str = REGISTRY_VERSION


class DuplicateRegistryEntryError(Exception):
    pass


class PriorDocumentRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_version: str = REGISTRY_VERSION
    entries: tuple[PriorDocumentRegistryEntry, ...]

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        seen: dict[str, PriorDocumentRegistryEntry] = {}
        for e in self.entries:
            prior = seen.get(e.docile_document_id)
            if prior is not None and (prior.role, prior.split_or_population, prior.first_project_use) != (e.role, e.split_or_population, e.first_project_use):
                raise DuplicateRegistryEntryError(f"docid {e.docile_document_id} has conflicting registry entries: {prior} vs {e}")
            seen[e.docile_document_id] = e


def resolve_prior_docids(registry: PriorDocumentRegistry) -> frozenset[str]:
    """The one function a future selection module needs: pure set
    membership, no filesystem access, no text scanning. Deterministic --
    calling it twice on the same registry returns the identical result."""
    return frozenset(e.docile_document_id for e in registry.entries)


def load_registry(path: str | Path) -> PriorDocumentRegistry:
    data = json.loads(Path(path).read_text())
    return PriorDocumentRegistry.model_validate(data)


def save_registry(registry: PriorDocumentRegistry, path: str | Path) -> None:
    Path(path).write_text(json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=False) + "\n")


# ---------------------------------------------------------------------
# Reference migration: build a v1 registry from the two frozen corpora's
# OWN already-published document_assignments.jsonl files. Read-only --
# writes nothing back into results/enterprise_corpus/**. This demonstrates
# the registry format is buildable from real frozen data; it is not a
# claim that it reproduces discover_prior_docids' full exclusion set
# (which also scans data/scenarios, configs, demo, and dataset-inspection
# image directories -- see the design doc's migration-completeness note).
# ---------------------------------------------------------------------

_ASSIGNMENT_FILES = {
    "enterprise_corpus_v2": "results/enterprise_corpus/v2/document_assignments.jsonl",
    "enterprise_corpus_v2_1": "results/enterprise_corpus/v2_1/document_assignments.jsonl",
}


def build_registry_from_frozen_manifests(repo_root: str | Path) -> PriorDocumentRegistry:
    repo_root = Path(repo_root)
    entries: list[PriorDocumentRegistryEntry] = []
    seen: set[str] = set()
    for project, rel in _ASSIGNMENT_FILES.items():
        path = repo_root / rel
        for line in path.read_text().splitlines():
            row = json.loads(line)
            docid = row["docid"]
            if docid in seen:
                continue  # first project use wins; v2.1 deliberately overlaps some v2 documents
            seen.add(docid)
            entries.append(
                PriorDocumentRegistryEntry(
                    docile_document_id=docid,
                    first_project_use=project,
                    role=row["population"],
                    split_or_population=row["split"],
                    source_manifest=rel,
                    reason_for_exclusion=f"assigned to population={row['population']!r} split={row['split']!r} in {project}",
                )
            )
    return PriorDocumentRegistry(entries=tuple(entries))
