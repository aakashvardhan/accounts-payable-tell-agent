"""Tests for tell.memory.models / tell.memory.store. No GPU, no model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tell.evaluation.scenario import ProvenanceSource, SourceType, TrustBoundary
from tell.memory.models import MemoryKind, MemoryStatus
from tell.memory.store import DuplicateMemoryIdError, MemoryStore, UnknownMemoryError, content_sha256, derive_memory_id


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory_test.sqlite")
    s.init_schema()
    yield s
    s.close()


def _write(store: MemoryStore, content: str = "note", vendor_id: str = "V1"):
    return store.append_memory(
        vendor_id=vendor_id,
        memory_kind=MemoryKind.OPERATIONAL_NOTE,
        content=content,
        origin_source_type=SourceType.EMAIL,
        origin_source_id="MSG-1",
        origin_provenance=ProvenanceSource.SYNTHETIC_CONTROLLED,
        origin_trust_boundary=TrustBoundary.UNTRUSTED,
    )


def test_memory_schema_round_trip(store: MemoryStore):
    rec = _write(store, content="Include invoice number in future replies.")
    assert rec.status == MemoryStatus.ACTIVE
    assert rec.memory_kind == MemoryKind.OPERATIONAL_NOTE
    assert rec.content_sha256 == content_sha256(rec.content)
    assert rec.memory_id == derive_memory_id(rec.content)
    reloaded = store.get_memory_by_id(rec.memory_id)
    assert reloaded == rec


def test_append_memory_signature_accepts_no_model_provenance_fields(store: MemoryStore):
    # append_memory's signature has no memory_id/created_at/status
    # parameter at all -- a caller cannot pass them even if it wanted to.
    import inspect

    sig = inspect.signature(store.append_memory)
    forbidden = {"memory_id", "created_at", "status", "content_sha256"}
    assert forbidden.isdisjoint(sig.parameters.keys())


def test_writes_preserve_origin_source_and_trust_boundary(store: MemoryStore):
    rec = _write(store)
    assert rec.origin_source_type == SourceType.EMAIL
    assert rec.origin_source_id == "MSG-1"
    assert rec.origin_provenance == ProvenanceSource.SYNTHETIC_CONTROLLED
    assert rec.origin_trust_boundary == TrustBoundary.UNTRUSTED


def test_search_returns_original_provenance(store: MemoryStore):
    _write(store, content="Please always include the invoice number in replies.")
    results = store.search_memories("V1", query="invoice number")
    assert len(results) == 1
    assert results[0].origin_trust_boundary == TrustBoundary.UNTRUSTED
    assert results[0].origin_source_type == SourceType.EMAIL


def test_search_excludes_quarantined_by_default(store: MemoryStore):
    rec = _write(store)
    assert len(store.search_memories("V1")) == 1
    store.quarantine_memory(rec.memory_id, reason="test")
    assert store.search_memories("V1") == []
    assert store.get_active_memories_by_vendor("V1") == []
    # but the record itself is still retrievable by id, and inspection
    # via list_all_memories still shows it (not deleted).
    assert store.get_memory_by_id(rec.memory_id).status == MemoryStatus.QUARANTINED
    assert len(store.list_all_memories("V1")) == 1


def test_audit_events_are_append_only(store: MemoryStore):
    rec = _write(store)
    events_after_write = store.list_audit_events()
    assert len(events_after_write) == 1
    assert events_after_write[0].event_type == "memory_appended"

    store.quarantine_memory(rec.memory_id, reason="test")
    events_after_quarantine = store.list_audit_events()
    assert len(events_after_quarantine) == 2
    # Prior events are unchanged, not rewritten.
    assert events_after_quarantine[0] == events_after_write[0]
    assert events_after_quarantine[1].event_type == "memory_quarantined"


def test_duplicate_content_rejected():
    import tempfile

    p = Path(tempfile.mktemp(suffix=".sqlite"))
    store = MemoryStore(p)
    store.init_schema()
    _write(store, content="same content")
    with pytest.raises(DuplicateMemoryIdError):
        _write(store, content="same content")
    store.close()
    p.unlink()


def test_unknown_memory_id_raises(store: MemoryStore):
    with pytest.raises(UnknownMemoryError):
        store.quarantine_memory("MEM-DOES-NOT-EXIST", reason="x")


def test_memory_database_is_separate_from_ledger(tmp_path: Path):
    from tell.payment.ledger import Ledger

    memory_path = tmp_path / "memory.sqlite"
    ledger_path = tmp_path / "ledger.sqlite"
    memory_store = MemoryStore(memory_path)
    memory_store.init_schema()
    ledger = Ledger(ledger_path)
    ledger.init_schema()

    assert memory_path != ledger_path
    _write(memory_store)
    # Ledger has no memories table; memory store has no accounts table.
    ledger_tables = {r["name"] for r in ledger.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    memory_tables = {r["name"] for r in memory_store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "memories" not in ledger_tables
    assert "accounts" not in memory_tables

    memory_store.close()
    ledger.close()
