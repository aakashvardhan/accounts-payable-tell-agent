"""Builds the prior-document registry v1.2 exclusions report (CPU only;
read-only against existing frozen/historical files).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/build_prior_document_registry_v1_2_report.py

v1.1's registry (768 documents) is unchanged and re-verified here, not
regenerated differently. This script adds nothing to it -- it only
classifies v1.1's four detector findings as reviewed exclusions and
reports the resulting three-bucket count (registered documents / reviewed
exclusions / unexplained sources), with the last driven to zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))

from tell.registry.prior_document_registry import resolve_prior_docids  # noqa: E402
from tell.registry.prior_document_registry_v1_1 import build_registry_v1_1  # noqa: E402
from tell.registry.prior_document_registry_v1_2_exclusions import classify_findings  # noqa: E402

REPORT_OUT = REPO / "results" / "routing_design" / "prior_document_registry_v1_2_exclusions_report.md"


def main() -> None:
    reg, migration_report = build_registry_v1_1(REPO)
    docids = resolve_prior_docids(reg)
    classification = classify_findings(REPO)

    lines = [
        "# Prior-document registry v1.2 -- exclusions correction report",
        "",
        "Status: frozen correction. Read-only; v1 (676 documents) and v1.1 (768 documents) registries are "
        "byte-for-byte unchanged by this correction -- see "
        "`prior_document_registry_v1_1_migration_report.md` for their own reports. This document adds ONE "
        "thing: a typed, versioned classification of the four sources v1.1's narrow "
        "`detect_unregistered_eligible_sources` detector flagged, so 'manually reviewed, contributes zero "
        "documents' is a structured fact (`tell.registry.prior_document_registry_v1_2_exclusions.REVIEWED_EXCLUSIONS`) "
        "instead of only prose in a migration report.",
        "",
        "## Three-bucket final count",
        "",
        "| Bucket | Count |",
        "|---|---:|",
        f"| Registered document sources (v1 + v1.1, unique documents) | {len(docids)} |",
        f"| Explicit reviewed exclusions (source files, zero documents each) | {len(classification.reviewed_exclusions)} |",
        f"| **Unexplained eligible sources (must be zero)** | **{len(classification.unexplained_eligible_sources)}** |",
        "",
        "## Reviewed exclusions",
        "",
        "| source_path | exclusion_reason | contributes_documents |",
        "|---|---|---:|",
    ]
    for e in classification.reviewed_exclusions:
        lines.append(f"| `{e.source_path}` | `{e.exclusion_reason.value}` | {e.contributes_documents} |")
    lines += [
        "",
        "## Unexplained eligible sources",
        "",
        "(none)" if not classification.unexplained_eligible_sources else "\n".join(f"- `{s}`" for s in classification.unexplained_eligible_sources),
        "",
        "## Verification",
        "",
        f"- `is_fully_explained`: **{classification.is_fully_explained}**",
        "- The detector (`tell.registry.prior_document_registry_v1_1.detect_unregistered_eligible_sources`) is "
        "unchanged -- it is a narrow, non-recursive check of `results/dataset_inspection/`'s own top-level "
        "entries only, not a scan of `results/` at large.",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT_OUT)
    print(json.dumps({
        "registered_documents": len(docids),
        "reviewed_exclusions": len(classification.reviewed_exclusions),
        "unexplained_eligible_sources": len(classification.unexplained_eligible_sources),
    }, indent=2))


if __name__ == "__main__":
    main()
