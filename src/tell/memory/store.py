"""Minimal provenance-aware memory store: a dedicated SQLite database,
entirely separate from the payment ledger (`tell.payment.ledger`).

Built-in `sqlite3` only, parameterized SQL throughout, explicit
transactions with rollback on failure -- the same discipline as
`tell.payment.ledger.Ledger`. No function here accepts model output
directly: every public write method's signature takes only
already-application-derived keyword arguments (see
`tell.memory.models`'s docstring for the trust boundary this enforces).
The only thing resembling "the model's input" is `content` and
`memory_kind`, which are plain data, not provenance.

Money-ledger parallel, deliberately: `append_memory` is to this store
what `Ledger.create_payment_intent` is to the ledger -- the one write
path, fully audited, that a higher layer (`tell.agent.memory_loop`)
calls only after validating a `WriteMemoryAction` and deriving every
trusted field itself.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tell.evaluation.scenario import ProvenanceSource, SourceType, TrustBoundary
from tell.memory.models import MemoryAuditEvent, MemoryKind, MemoryRecord, MemoryStatus


class MemoryStoreError(Exception):
    """Base class for every memory-store-rejected operation."""


class UnknownMemoryError(MemoryStoreError):
    pass


class DuplicateMemoryIdError(MemoryStoreError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def derive_memory_id(content: str) -> str:
    """Deterministic, content-derived id -- reproducible across runs of
    the same deterministic pipeline without needing a counter or
    randomness. Collision would require two memories with byte-identical
    content for the same run, which `append_memory` rejects explicitly.
    """
    return f"MEM-{content_sha256(content)[:16]}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    vendor_id TEXT NOT NULL,
    memory_kind TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active','quarantined','superseded')),
    created_at TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    origin_source_type TEXT NOT NULL,
    origin_source_id TEXT NOT NULL,
    origin_provenance TEXT NOT NULL,
    origin_trust_boundary TEXT NOT NULL,
    parent_memory_id TEXT REFERENCES memories(memory_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class MemoryStore:
    """One SQLite connection, manual-transaction mode, mirroring
    tell.payment.ledger.Ledger's style. Entirely separate database file
    from the ledger -- see PINNED paths in scripts/run_memory_pilot.py."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)

    def _audit(self, event_type: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO audit_events (event_type, payload_json, created_at) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), _now()),
        )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            vendor_id=row["vendor_id"],
            memory_kind=MemoryKind(row["memory_kind"]),
            content=row["content"],
            status=MemoryStatus(row["status"]),
            created_at=row["created_at"],
            content_sha256=row["content_sha256"],
            origin_source_type=SourceType(row["origin_source_type"]),
            origin_source_id=row["origin_source_id"],
            origin_provenance=ProvenanceSource(row["origin_provenance"]),
            origin_trust_boundary=TrustBoundary(row["origin_trust_boundary"]),
            parent_memory_id=row["parent_memory_id"],
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append_memory(
        self,
        *,
        vendor_id: str,
        memory_kind: MemoryKind,
        content: str,
        origin_source_type: SourceType,
        origin_source_id: str,
        origin_provenance: ProvenanceSource,
        origin_trust_boundary: TrustBoundary,
        parent_memory_id: str | None = None,
    ) -> MemoryRecord:
        """The only write path that creates a memory. `memory_id`,
        `created_at`, `status` (always ACTIVE on write), and
        `content_sha256` are derived here, never accepted as arguments
        from a model-facing caller -- see this module's docstring."""
        memory_id = derive_memory_id(content)
        if self.get_memory_by_id(memory_id) is not None:
            raise DuplicateMemoryIdError(f"Memory {memory_id!r} (content hash collision) already exists")

        created_at = _now()
        sha = content_sha256(content)
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                """INSERT INTO memories
                   (memory_id, vendor_id, memory_kind, content, status, created_at, content_sha256,
                    origin_source_type, origin_source_id, origin_provenance, origin_trust_boundary, parent_memory_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    vendor_id,
                    memory_kind.value,
                    content,
                    MemoryStatus.ACTIVE.value,
                    created_at,
                    sha,
                    origin_source_type.value,
                    origin_source_id,
                    origin_provenance.value,
                    origin_trust_boundary.value,
                    parent_memory_id,
                ),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._audit(
            "memory_appended",
            {
                "memory_id": memory_id,
                "vendor_id": vendor_id,
                "memory_kind": memory_kind.value,
                "content_sha256": sha,
                "origin_source_type": origin_source_type.value,
                "origin_source_id": origin_source_id,
                "origin_provenance": origin_provenance.value,
                "origin_trust_boundary": origin_trust_boundary.value,
            },
        )
        return self.get_memory_by_id(memory_id)

    def quarantine_memory(self, memory_id: str, *, reason: str) -> MemoryRecord:
        """Trusted-only operation; no action in this pilot lets the model
        call this. Reserved for a future gate/review integration."""
        row = self._get_memory_row(memory_id)
        if row is None:
            raise UnknownMemoryError(f"Unknown memory_id: {memory_id!r}")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "UPDATE memories SET status = ? WHERE memory_id = ?", (MemoryStatus.QUARANTINED.value, memory_id)
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._audit("memory_quarantined", {"memory_id": memory_id, "reason": reason})
        return self.get_memory_by_id(memory_id)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _get_memory_row(self, memory_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
        return cur.fetchone()

    def get_memory_by_id(self, memory_id: str) -> MemoryRecord | None:
        row = self._get_memory_row(memory_id)
        return self._row_to_record(row) if row is not None else None

    def get_active_memories_by_vendor(self, vendor_id: str) -> list[MemoryRecord]:
        cur = self.conn.execute(
            "SELECT * FROM memories WHERE vendor_id = ? AND status = ? ORDER BY created_at",
            (vendor_id, MemoryStatus.ACTIVE.value),
        )
        return [self._row_to_record(r) for r in cur.fetchall()]

    def search_memories(
        self,
        vendor_id: str,
        *,
        query: str | None = None,
        memory_kind: MemoryKind | None = None,
    ) -> list[MemoryRecord]:
        """Searches only ACTIVE memories (quarantined/superseded are
        excluded by default -- callers needing those must use
        `get_memory_by_id` explicitly). `query` is a case-insensitive
        substring match over `content`; both filters are optional and
        combine with AND."""
        records = self.get_active_memories_by_vendor(vendor_id)
        if memory_kind is not None:
            records = [r for r in records if r.memory_kind == memory_kind]
        if query:
            needle = query.lower()
            records = [r for r in records if needle in r.content.lower()]
        return records

    def list_all_memories(self, vendor_id: str | None = None) -> list[MemoryRecord]:
        """Deterministic inspection for tests: every memory regardless of
        status, optionally filtered by vendor."""
        if vendor_id is None:
            cur = self.conn.execute("SELECT * FROM memories ORDER BY created_at")
        else:
            cur = self.conn.execute("SELECT * FROM memories WHERE vendor_id = ? ORDER BY created_at", (vendor_id,))
        return [self._row_to_record(r) for r in cur.fetchall()]

    def list_audit_events(self) -> list[MemoryAuditEvent]:
        cur = self.conn.execute("SELECT * FROM audit_events ORDER BY event_id")
        return [
            MemoryAuditEvent(
                event_id=r["event_id"],
                event_type=r["event_type"],
                payload=json.loads(r["payload_json"]),
                created_at=r["created_at"],
            )
            for r in cur.fetchall()
        ]
