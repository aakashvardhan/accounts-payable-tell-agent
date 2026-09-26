"""Typed data model for Tell's minimal provenance-aware memory
subsystem, used by the delayed-memory-poisoning pilot
(scripts/run_memory_pilot.py).

--------------------------------------------------------------------------
The trust boundary this module exists to enforce
--------------------------------------------------------------------------
The model may propose a memory's `content` and `memory_kind` (see
`tell.agent.actions.WriteMemoryAction`). Trusted application code --
never the model -- derives every other field on `MemoryRecord`:
`memory_id`, `created_at`, `status`, `content_sha256`, and all four
`origin_*` fields. This is the same boundary `tell.agent.decision` and
`tell.agent.actions` already enforce for payment fields (no tool ever
returns the company's own paying account, so the model is never asked
for it) applied to memory: no action the model can take ever lets it
declare its own content trusted, so `origin_trust_boundary` can only
ever be assigned by the code that actually knows where the content came
from (see `tell.memory.store.MemoryStore.append_memory`, which accepts
no model output directly -- only already-application-derived keyword
arguments).

A memory whose content originated in an email is `origin_trust_boundary
= TrustBoundary.UNTRUSTED` forever, even after it is stored locally and
even if the model's own `memory_kind`/`content` framing makes it read as
authoritative. Storage is not verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tell.evaluation.scenario import ProvenanceSource, SourceType, TrustBoundary


class MemoryStatus(str, Enum):
    """Minimum lifecycle statuses. A memory is written ACTIVE; only
    trusted application/gate code (not exercised by the model in this
    pilot) may later move it to QUARANTINED or SUPERSEDED."""

    ACTIVE = "active"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


class MemoryKind(str, Enum):
    """Model-proposed content category. Purely descriptive -- it never
    affects trust or lifecycle bookkeeping, both of which stay
    application-derived regardless of what kind the model claims."""

    OPERATIONAL_NOTE = "operational_note"
    PAYMENT_INSTRUCTION_CLAIM = "payment_instruction_claim"
    OTHER = "other"


@dataclass(frozen=True)
class MemoryRecord:
    """One persisted memory. Every field except `content` and
    `memory_kind` is application-derived -- see this module's docstring.
    """

    memory_id: str
    vendor_id: str
    memory_kind: MemoryKind
    content: str
    status: MemoryStatus
    created_at: str
    content_sha256: str
    origin_source_type: SourceType
    origin_source_id: str
    origin_provenance: ProvenanceSource
    origin_trust_boundary: TrustBoundary
    parent_memory_id: str | None = None


@dataclass(frozen=True)
class MemoryAuditEvent:
    event_id: int
    event_type: str
    payload: dict
    created_at: str
