"""Protected-artifact integrity check for the CPU-only runtime-integration
milestone (trusted lookup tools, registry v1.1, orchestrator wiring).

Everything produced by the PREVIOUS milestone (enterprise corpus
v2/v2.1/v2.2, the v2.2 training supplement, pretraining_contract_v1, the
v1.1 sampler/contract outputs, and every file under results/routing_design/,
configs/routing/, scripts/routing_integrity/, and the previous milestone's
src/tell/{safety,memory,routing,registry}/* modules and tests) is now
PROTECTED. Only the specific new files this milestone adds (listed in
ALLOWED_NEW_CODE) may appear; only the specific pre-existing files this
milestone narrowly edits (ALLOWED_MODIFIED_FILES) may change.

Run `before` once before any edit and `after` once everything is
complete. `after` also re-checks every file listed in the frozen
v2/v2.1/v2.2/v2.2-supplement/pretraining-contract-v1 manifests against the
sha256 recorded in the manifest itself -- unchanged from the previous
milestone's script, reused here read-only (not imported, to keep this
script self-contained and independently freezable).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
# Deliberately NOT under results/, data/scenarios/, configs/, or demo/ --
# see the previous milestone's scripts/hash_protected_artifacts_routing_v1.py
# and results/routing_design/prior_document_registry_spec_v1.md for why a
# raw per-file path listing must never live where
# scripts/enterprise_v2/selection.py's discover_prior_docids scans it.
SNAP_DIR = REPO_ROOT / "scripts" / "routing_integrity"
BEFORE_PATH = SNAP_DIR / "runtime_integration_hashes_before.json"
AFTER_PATH = SNAP_DIR / "runtime_integration_hashes_after.json"
REPORT_PATH = SNAP_DIR / "runtime_integration_integrity.json"
SUMMARY_PATH = REPO_ROOT / "results" / "routing_design" / "runtime_integration_protected_artifact_integrity_summary.json"

ALLOWED_NEW_CODE = (
    "src/tell/agent/trusted_lookups.py",
    "src/tell/agent/routing_orchestrator.py",
    "src/tell/registry/prior_document_registry_v1_1.py",
    "tests/test_trusted_lookups.py",
    "tests/test_prior_document_registry_v1_1.py",
    "tests/test_routing_orchestrator.py",
    "scripts/hash_protected_artifacts_runtime_integration_v1.py",
    "scripts/build_prior_document_registry_v1_1_report.py",
    "scripts/routing_integrity/runtime_integration_hashes_before.json",
    "scripts/routing_integrity/runtime_integration_hashes_after.json",
    "scripts/routing_integrity/runtime_integration_integrity.json",
    "scripts/routing_integrity/prior_document_registry_v1_1_full.json",
    "results/routing_design/tell_runtime_integration_v1.md",
    "results/routing_design/prior_document_registry_v1_1_migration_report.md",
    "results/routing_design/runtime_integration_protected_artifact_integrity_summary.json",
)
# Narrow, additive edits to pre-existing files only. No other change to
# these files is permitted by this milestone.
ALLOWED_MODIFIED_FILES: tuple[str, ...] = (
    "src/tell/evaluation/scenario.py",  # two new SourceType enum members (INVOICE_PAYMENT_HISTORY, DISPUTE_CASE); no existing member changed
)

EXTRA_ROOTS = ("data/scenarios", "configs", "src", "scripts", "tests")
FROZEN_CORPUS_MANIFESTS = {
    "v2": "results/enterprise_corpus/v2/enterprise_corpus_v2_manifest.json",
    "v2_1": "results/enterprise_corpus/v2_1/enterprise_corpus_v2_1_manifest.json",
    "v2_2": "results/enterprise_corpus/v2_2/enterprise_corpus_v2_2_manifest.json",
    "v2_2_training_supplement": "results/enterprise_corpus/v2_2_training_supplement/memory_supplement_v1_manifest.json",
    "pretraining_contract_v1": "results/lora_training/pretraining_contract_v1/pretraining_contract_v1_manifest.json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_roots() -> list[Path]:
    roots = [p for p in sorted((REPO_ROOT / "results").iterdir()) if p.is_dir()]
    roots += [REPO_ROOT / r for r in EXTRA_ROOTS]
    return roots


def _docile_aggregate_hash() -> dict:
    root = REPO_ROOT / "data" / "docile"
    if not root.exists():
        return {"present": False}
    files = sorted(p for p in root.rglob("*") if p.is_file())
    listing = "".join(f"{p.relative_to(REPO_ROOT)}:{_sha256(p)}\n" for p in files)
    return {"present": True, "n_files": len(files), "aggregate_sha256": hashlib.sha256(listing.encode()).hexdigest()}


def snapshot(roots: list[str] | None = None) -> dict[str, str]:
    root_paths = [REPO_ROOT / r for r in roots] if roots else protected_roots()
    out: dict[str, str] = {}
    for root in root_paths:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
                continue
            rel = str(p.relative_to(REPO_ROOT))
            if rel.startswith("data/docile/"):
                continue  # verified separately as a single aggregate hash -- see this module's docstring / the previous milestone's leakage incident
            if any(rel == c or rel.startswith(c) for c in ALLOWED_NEW_CODE):
                continue
            out[rel] = _sha256(p)
    for p in sorted(REPO_ROOT.rglob("*.sqlite")):
        if ".venv" in p.parts or ".venv-inspect" in p.parts:
            continue
        out[str(p.relative_to(REPO_ROOT))] = _sha256(p)
    return out


def _manifest_file_entries(manifest: dict) -> dict[str, str]:
    found: dict[str, str] = {}

    def walk(o, key=None):
        if isinstance(o, dict):
            if isinstance(o.get("sha256"), str) and len(o["sha256"]) == 64:
                path = o.get("path") or key
                if isinstance(path, str) and "/" in path:
                    found[path] = o["sha256"]
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)

    walk(manifest)
    return found


def frozen_manifest_check() -> dict:
    out = {}
    for version, rel in FROZEN_CORPUS_MANIFESTS.items():
        entries = _manifest_file_entries(json.loads((REPO_ROOT / rel).read_text()))
        mismatched = sorted(p for p, h in entries.items() if not (REPO_ROOT / p).exists() or _sha256(REPO_ROOT / p) != h)
        out[version] = {"manifest": rel, "manifest_sha256": _sha256(REPO_ROOT / rel), "n_files_checked": len(entries), "mismatched": mismatched, "byte_identical": not mismatched and bool(entries)}
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    if mode not in ("before", "after"):
        raise SystemExit("usage: hash_protected_artifacts_runtime_integration_v1.py [before|after]")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if mode == "before":
        if BEFORE_PATH.exists():
            raise SystemExit(f"{BEFORE_PATH} already exists -- refusing to overwrite the pre-work snapshot.")
        roots = [str(p.relative_to(REPO_ROOT)) for p in protected_roots()]
        snap = snapshot()
        docile = _docile_aggregate_hash()
        BEFORE_PATH.write_text(json.dumps({"protected_roots": roots, "n_files": len(snap), "hashes": snap, "docile_aggregate": docile}, indent=1, sort_keys=True))
        print(f"[before] hashed {len(snap)} protected files + docile aggregate ({docile.get('n_files')} files) -> {BEFORE_PATH}")
        return

    before_doc = json.loads(BEFORE_PATH.read_text())
    before = before_doc["hashes"]
    after = snapshot(before_doc["protected_roots"])
    docile_before = before_doc.get("docile_aggregate", {})
    docile_after = _docile_aggregate_hash()
    AFTER_PATH.write_text(json.dumps({"protected_roots": before_doc["protected_roots"], "n_files": len(after), "hashes": after, "docile_aggregate": docile_after}, indent=1, sort_keys=True))

    missing = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    added = sorted(set(after) - set(before))
    allowed_changed = set(ALLOWED_MODIFIED_FILES)
    unexpected_changed = [c for c in changed if c not in allowed_changed]
    unexpected_added = [a for a in added if not any(a == c or a.startswith(c) for c in ALLOWED_NEW_CODE)]
    manifests = frozen_manifest_check()
    docile_ok = docile_before == docile_after
    report = {
        "protected_roots": before_doc["protected_roots"],
        "n_files_before": len(before),
        "n_files_after": len(after),
        "n_missing": len(missing),
        "n_changed": len(changed),
        "n_added": len(added),
        "missing_files": missing,
        "changed_files": changed,
        "unexpected_changed_files": unexpected_changed,
        "added_files": added,
        "unexpected_added_files": unexpected_added,
        "frozen_manifest_sha256_check": manifests,
        "docile_dataset_unchanged": docile_ok,
        "integrity_ok": not missing and not unexpected_changed and not unexpected_added and docile_ok and all(m["byte_identical"] for m in manifests.values()),
        "before_snapshot": str(BEFORE_PATH.relative_to(REPO_ROOT)),
        "after_snapshot": str(AFTER_PATH.relative_to(REPO_ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    summary = {k: report[k] for k in ("n_files_before", "n_files_after", "n_missing", "n_changed", "n_added", "unexpected_changed_files", "unexpected_added_files", "docile_dataset_unchanged", "integrity_ok", "before_snapshot", "after_snapshot")}
    summary["added_files"] = added
    summary["frozen_manifest_sha256_check"] = {v: {k: m[k] for k in ("manifest", "manifest_sha256", "n_files_checked", "byte_identical")} | {"n_mismatched": len(m["mismatched"])} for v, m in manifests.items()}
    summary["full_report"] = str(REPORT_PATH.relative_to(REPO_ROOT))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[after] integrity_ok={report['integrity_ok']} missing={len(missing)} changed={len(changed)} (unexpected={len(unexpected_changed)}) added={len(added)} (unexpected={len(unexpected_added)})")
    for v, m in manifests.items():
        print(f"  {v}: manifest files checked={m['n_files_checked']} byte_identical={m['byte_identical']}")


if __name__ == "__main__":
    main()
