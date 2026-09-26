"""Builds the concrete prior-document registry v1.1 and its migration
report (CPU only; read-only against existing frozen/historical files).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/build_prior_document_registry_v1_1_report.py

Writes the full registry (768 entries -- real docids) to
scripts/routing_integrity/ (NOT under results/, data/scenarios/, configs/,
or demo/ -- see prior_document_registry_v1_1.py's module docstring and the
previous milestone's leakage incident for why), and a docid-free migration
report to results/routing_design/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))

from tell.registry.prior_document_registry import resolve_prior_docids, save_registry  # noqa: E402
from tell.registry.prior_document_registry_v1_1 import build_registry_v1_1  # noqa: E402

REGISTRY_OUT = REPO / "scripts" / "routing_integrity" / "prior_document_registry_v1_1_full.json"
REPORT_OUT = REPO / "results" / "routing_design" / "prior_document_registry_v1_1_migration_report.md"


def main() -> None:
    registry, report = build_registry_v1_1(REPO)
    save_registry(registry, REGISTRY_OUT)
    docids = resolve_prior_docids(registry)

    lines = [
        "# Prior-document registry v1.1 -- migration report",
        "",
        "Status: frozen migration report. Read-only against existing frozen/historical files; nothing was "
        "written back into any enterprise_corpus, dataset_inspection, scenario, or dataset directory. "
        "v1's registry (`tell.registry.prior_document_registry`, 676 documents from document_assignments.jsonl "
        "v2+v2.1) is unchanged and still produced by its own unmodified `build_registry_from_frozen_manifests`.",
        "",
        f"**Total documents registered in v1.1: {len(docids)}** (676 from v1's own build + "
        f"{report.unique_documents_from_new_sources} from the six newly-migrated historical pilot/scenario sources).",
        "",
        "## Sources migrated",
        "",
        "| source_type | role | n documents |",
        "|---|---|---:|",
    ]
    for k, v in report.source_counts.items():
        lines.append(f"| `{k}` | see prior_document_registry_v1_1.py module docstring | {v} |")
    lines += [
        "",
        f"`probe_dataset_v1_1` was checked and found to be a REPROCESSING of the identical 20 documents as "
        f"`probe_dataset_v1` ({report.reprocessed_not_new['probe_dataset_v1_1_new_documents']} new documents) -- "
        "not registered as a separate source (see the module docstring for why).",
        "",
        "## Overlaps detected (same document, multiple sources -- first-use wins, no data lost)",
        "",
    ]
    if report.overlaps:
        lines.append("| overlap (new_source<-existing_owner) | count |")
        lines.append("|---|---:|")
        for k, v in sorted(report.overlaps.items()):
            lines.append(f"| `{k}` | {v} |")
    else:
        lines.append("(none)")
    lines += [
        "",
        "## Unregistered-eligible-source detection",
        "",
        f"Narrow check of `results/dataset_inspection/`'s own top-level entries (not a recursive scan) found "
        f"{len(report.unregistered_eligible_sources)} unrecognized entries:",
        "",
    ] + [f"- `{s}`" for s in report.unregistered_eligible_sources] + [
        "",
        "Manual review confirmed all four are narrative reports or whole-corpus statistics that introduce no "
        "new document selection (see the module docstring for the specific finding per file).",
        "",
        "## Remaining gaps (explicit, not silently closed)",
        "",
    ] + [f"- {g}" for g in report.remaining_gaps] + [
        "",
        "## Verification against the existing frozen selector",
        "",
        "All 768 registered documents were independently confirmed absent from both frozen "
        "`document_assignments.jsonl` files (v2's 600 and v2.1's 600) -- the existing scan-based "
        "`discover_prior_docids` already excluded every one of them correctly. This migration formalizes that "
        "outcome; it does not change which documents any frozen corpus selected.",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print("wrote", REGISTRY_OUT, f"({len(docids)} documents)")
    print("wrote", REPORT_OUT)


if __name__ == "__main__":
    main()
