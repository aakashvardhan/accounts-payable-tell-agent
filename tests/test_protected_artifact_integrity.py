"""CPU-only test proving every frozen corpus/contract artifact this task
touched-adjacent code for is still byte-identical to its recorded sha256.
Independent of the before/after snapshot workflow in
scripts/hash_protected_artifacts_routing_v1.py (item 28 of the required
tests) -- this can be run any time, not just as part of that script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")


def _load_hash_module():
    spec = importlib.util.spec_from_file_location("hash_protected_artifacts_routing_v1", REPO / "scripts" / "hash_protected_artifacts_routing_v1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_all_frozen_manifests_remain_byte_identical():
    mod = _load_hash_module()
    result = mod.frozen_manifest_check()
    assert set(result) == {"v2", "v2_1", "v2_2", "v2_2_training_supplement", "pretraining_contract_v1"}
    for version, info in result.items():
        assert info["byte_identical"], f"{version}: mismatched files {info['mismatched']}"
        assert info["n_files_checked"] > 0


def test_docile_dataset_directory_present_and_not_individually_enumerated_under_results():
    # Regression test for the leakage bug this task hit and fixed: the
    # hashing script must never enumerate data/docile file-by-file into an
    # output under results/, data/scenarios/, configs/, or demo/ (those are
    # exactly the roots scripts/enterprise_v2/selection.py's
    # discover_prior_docids scans).
    mod = _load_hash_module()
    assert "data/docile" not in mod.EXTRA_ROOTS
    docile_info = mod._docile_aggregate_hash()
    assert docile_info["present"] is True
    assert docile_info["n_files"] > 15000


def test_hash_script_output_locations_avoid_prior_docid_scan_roots():
    mod = _load_hash_module()
    scanned_roots = ("results", "data/scenarios", "configs", "demo")
    for path in (mod.BEFORE_PATH, mod.AFTER_PATH, mod.REPORT_PATH):
        rel = str(path.relative_to(REPO))
        assert not any(rel == r or rel.startswith(r + "/") for r in scanned_roots), f"{rel} sits under a docid-scanned root"
