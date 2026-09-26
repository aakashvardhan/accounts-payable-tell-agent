"""CPU-only tests for tell.registry.prior_document_registry (Part 11 of
the Tell-routing design). Reads only the already-frozen
document_assignments.jsonl files (read-only); writes nothing back into any
enterprise_corpus directory."""

from __future__ import annotations

import json

import pytest

from tell.registry.prior_document_registry import (
    DuplicateRegistryEntryError,
    PriorDocumentRegistry,
    PriorDocumentRegistryEntry,
    build_registry_from_frozen_manifests,
    load_registry,
    resolve_prior_docids,
    save_registry,
)

REPO = "/home/hp5/tell"


def _entry(docid="a" * 24, **kw) -> PriorDocumentRegistryEntry:
    fields = dict(docile_document_id=docid, first_project_use="enterprise_corpus_v2", role="probe_v2", split_or_population="train", source_manifest="results/x.jsonl", reason_for_exclusion="test")
    fields.update(kw)
    return PriorDocumentRegistryEntry(**fields)


def test_entry_rejects_non_24_hex_docid():
    with pytest.raises(Exception):
        _entry(docid="not-a-docid")


def test_registry_rejects_conflicting_duplicate_entry():
    a = _entry(role="probe_v2")
    b = _entry(role="lora_v2")  # same docid, different role -- a real conflict
    with pytest.raises(DuplicateRegistryEntryError):
        PriorDocumentRegistry(entries=(a, b))


def test_registry_allows_identical_duplicate_entry():
    a = _entry()
    b = _entry()
    reg = PriorDocumentRegistry(entries=(a, b))
    assert len(resolve_prior_docids(reg)) == 1


# ---------------------------------------------------------------------
# 27: explicit document registry is deterministic
# ---------------------------------------------------------------------


def test_resolve_is_deterministic():
    reg = PriorDocumentRegistry(entries=(_entry(docid="a" * 24), _entry(docid="b" * 24, role="lora_v2")))
    a = resolve_prior_docids(reg)
    b = resolve_prior_docids(reg)
    assert a == b == frozenset({"a" * 24, "b" * 24})


def test_registry_round_trips_through_json(tmp_path):
    reg = PriorDocumentRegistry(entries=(_entry(docid="a" * 24), _entry(docid="b" * 24, role="lora_v2")))
    path = tmp_path / "registry.json"
    save_registry(reg, path)
    loaded = load_registry(path)
    assert resolve_prior_docids(loaded) == resolve_prior_docids(reg)
    assert json.loads(path.read_text())["registry_version"] == "prior_document_registry_v1"


def test_future_selection_code_can_consume_registry_deterministically(tmp_path):
    """Simulates what a future v3 selector would do: load the registry,
    intersect with a candidate train-docid pool, and get back an
    exclusion set -- deterministic, no filesystem scan."""
    reg = PriorDocumentRegistry(entries=(_entry(docid="a" * 24), _entry(docid="b" * 24, role="lora_v2")))
    path = tmp_path / "registry.json"
    save_registry(reg, path)

    candidate_pool = {"a" * 24, "c" * 24, "d" * 24}
    loaded = load_registry(path)
    excluded = resolve_prior_docids(loaded) & candidate_pool
    eligible = candidate_pool - excluded
    assert excluded == {"a" * 24}
    assert eligible == {"c" * 24, "d" * 24}


# ---------------------------------------------------------------------
# Reference migration: read-only against the real frozen v2/v2.1 outputs
# ---------------------------------------------------------------------


def test_reference_migration_reads_only_and_matches_expected_counts():
    reg = build_registry_from_frozen_manifests(REPO)
    docids = resolve_prior_docids(reg)
    assert len(docids) == 676  # 600 from v2 + 76 new-to-v2.1 (v2.1 overlaps 524 of its 600 with v2)
    n_v2 = sum(1 for e in reg.entries if e.first_project_use == "enterprise_corpus_v2")
    n_v2_1 = sum(1 for e in reg.entries if e.first_project_use == "enterprise_corpus_v2_1")
    assert n_v2 == 600
    assert n_v2_1 == 76
    assert all(len(d) == 24 for d in docids)


def test_reference_migration_does_not_write_anything(tmp_path, monkeypatch):
    import tell.registry.prior_document_registry as mod

    # Sanity: the function signature and source contain no file-write call.
    import inspect

    source = inspect.getsource(mod.build_registry_from_frozen_manifests)
    assert "write_text" not in source
    assert "open(" not in source or "'w'" not in source
