"""Section 7's narrowly scoped, versioned invoice-text observation for
the probe corpus: a bounded block of untrusted document text, distinct
from the structured KILE fields `read_invoice` already exposes.

This is new, additive structure -- it does not modify
`tell.agent.tools.ReadInvoiceResult`, `tell.evaluation.scenario
.DocileInvoiceRecord`, or any historical v1 tool output/trace. It exists
only inside this corpus's own observation-message construction
(`tell.probe_dataset.sample_builder`).

Every block is tagged with:
  - `block_id`: stable, corpus-local identifier (not a filesystem path).
  - `text`: bounded (<= MAX_BLOCK_CHARS) text.
  - `page`: page number if applicable (all selected documents are
    single-page, so always 0 here -- see the document-selection report's
    documented page-diversity limitation).
  - `source_type`: `docile_ocr` (copied verbatim, bounded, from the
    official annotation's own field/line-item text -- never OCR'd from
    the raw image, since this project does not run OCR; "docile_ocr" is
    the label DocILE's own dataset uses for this text, not a claim this
    code re-derives it) or `synthetic_overlay` (Tell-authored).
  - `provenance`: `docile_annotation` or `synthetic_controlled` (the
    same two-value vocabulary `tell.evaluation.scenario.ProvenanceSource`
    already uses elsewhere in this codebase).
  - `trust_boundary`: always `untrusted` -- every text block here lives
    under the untrusted invoice/tool-result surface, exactly like
    `tell.evaluation.scenario.DocileInvoiceRecord`'s own untrusted
    classification.

No filesystem path and no evaluation-only field (clean/attack label,
expected outcome) is ever included on a block or in its rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MAX_BLOCK_CHARS = 600


class TextBlockSourceType(str, Enum):
    DOCILE_OCR = "docile_ocr"
    SYNTHETIC_OVERLAY = "synthetic_overlay"


@dataclass(frozen=True)
class InvoiceTextBlock:
    block_id: str
    text: str
    page: int
    source_type: str
    provenance: str
    trust_boundary: str

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "text": self.text,
            "page": self.page,
            "source_type": self.source_type,
            "provenance": self.provenance,
            "trust_boundary": self.trust_boundary,
        }


def build_docile_ocr_block(*, block_id: str, vendor_address: str | None, line_item_descriptions: list[str]) -> InvoiceTextBlock:
    """A bounded, real-text block built only from already-extracted
    DocILE annotation text (vendor address plus line-item descriptions,
    the same fields `DocumentFacts` already copied verbatim) -- never
    reads a PDF or image, and never invents text. Truncated to
    MAX_BLOCK_CHARS with an explicit ellipsis marker so a probe never
    sees a silently truncated, unlabeled cut."""
    parts = []
    if vendor_address:
        parts.append(vendor_address.replace("\n", ", "))
    parts.extend(d.replace("\n", " ") for d in line_item_descriptions if d)
    text = " | ".join(parts) if parts else "(no additional document text available)"
    if len(text) > MAX_BLOCK_CHARS:
        text = text[: MAX_BLOCK_CHARS - 15] + " [...truncated]"
    return InvoiceTextBlock(
        block_id=block_id,
        text=text,
        page=0,
        source_type=TextBlockSourceType.DOCILE_OCR.value,
        provenance="docile_annotation",
        trust_boundary="untrusted",
    )


def build_synthetic_overlay_block(*, block_id: str, text: str) -> InvoiceTextBlock:
    if len(text) > MAX_BLOCK_CHARS:
        raise ValueError(f"synthetic overlay block text exceeds MAX_BLOCK_CHARS ({len(text)} > {MAX_BLOCK_CHARS}) for {block_id}")
    return InvoiceTextBlock(
        block_id=block_id,
        text=text,
        page=0,
        source_type=TextBlockSourceType.SYNTHETIC_OVERLAY.value,
        provenance="synthetic_controlled",
        trust_boundary="untrusted",
    )


__all__ = ["MAX_BLOCK_CHARS", "TextBlockSourceType", "InvoiceTextBlock", "build_docile_ocr_block", "build_synthetic_overlay_block"]
