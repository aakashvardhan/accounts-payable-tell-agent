"""Protected-artifact integrity check for the Tell-routing design task
(pretraining-contract v1.1 sampler correction + payment-validation /
memory / CUDA-guard / routing-design additions).

Everything that existed before this task started is PROTECTED, including
the outputs of the *previous* task (enterprise corpus v2/v2.1/v2.2, the
v2.2 training supplement, and pretraining_contract_v1). Only the roots
listed in TASK_OUTPUT_ROOTS are this task's own output, and only the
files/dirs listed in ALLOWED_NEW_CODE may be added outside those roots
(narrowly-scoped source corrections and new tests this task adds).

Run `before` once before any output is generated and `after` once
everything is complete. `after` also re-checks every file listed in the
frozen v2 / v2.1 / v2.2 / v2.2-supplement / pretraining-contract-v1
manifests against the sha256 recorded in the manifest itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
OUT_DIR = REPO_ROOT / "results" / "routing_design"
# The raw, full-path snapshots deliberately do NOT live under results/,
# data/scenarios/, configs/, or demo/: scripts/enterprise_v2/selection.py's
# frozen discover_prior_docids() scans exactly those four roots for any
# 24-hex-char substring in file content to decide which DocILE documents
# are "already used". A raw path listing that happens to enumerate an
# existing docid-bearing path elsewhere in the repo (e.g. a
# dataset_inspection image directory) would inject an extra reference into
# that scan and change results/enterprise_corpus/v2/prior_document_exclusions.json
# on the next regeneration -- breaking the frozen corpus's byte-identical
# regeneration test even though no protected file's *content* actually
# changed. scripts/ is not one of the four scanned roots, so the full
# per-file listings are written there; only the docid-free summary
# (counts, booleans, short new/changed-file lists) is published under
# results/routing_design/, this task's own output root.
SNAP_DIR = REPO_ROOT / "scripts" / "routing_integrity"
BEFORE_PATH = SNAP_DIR / "protected_artifact_hashes_before.json"
AFTER_PATH = SNAP_DIR / "protected_artifact_hashes_after.json"
REPORT_PATH = SNAP_DIR / "protected_artifact_integrity.json"
SUMMARY_PATH = OUT_DIR / "protected_artifact_integrity_summary.json"

TASK_OUTPUT_ROOTS = (
    "results/lora_training/pretraining_contract_v1_1",
    "results/routing_design",
    "configs/routing",
)
ALLOWED_NEW_CODE = (
    "scripts/lora_pretraining_v1_1/",
    "scripts/build_lora_pretraining_contract_v1_1.py",
    "scripts/hash_protected_artifacts_routing_v1.py",
    "scripts/routing_integrity/",
    "configs/lora/enterprise_v2_2/sampler_v1_1.json",
    "configs/lora/enterprise_v2_2/pretraining_contract_v1_1.json",
    "src/tell/safety/payment_validation.py",
    "src/tell/safety/alarm.py",
    "src/tell/safety/resolution.py",
    "src/tell/memory/next_step.py",
    "src/tell/routing/",
    "src/tell/registry/",
    "tests/test_payment_validation.py",
    "tests/test_memory_next_step.py",
    "tests/test_lora_pretraining_contract_v1_1.py",
    "tests/test_routing_metrics.py",
    "tests/test_prior_document_registry.py",
    "tests/test_alarm_states.py",
    "tests/test_cuda_guards.py",
    "tests/test_protected_artifact_integrity.py",
    "scripts/build_routing_metrics_example_v1.py",
    "scripts/build_memory_next_step_measurement_v1.py",
    "tests/conftest.py",
)
# Narrow, line-level CUDA-availability-guard fixes only (Part 4). No other
# change to these pre-existing files is permitted by this task.
ALLOWED_MODIFIED_FILES: tuple[str, ...] = (
    "src/tell/agent/loop.py",
    "src/tell/agent/conditional_retrieval.py",
    "src/tell/agent/memory_loop.py",
)

EXTRA_ROOTS = ("data/scenarios", "configs", "src", "scripts", "tests")
# data/docile (the raw, static, third-party DocILE dataset -- ~17k files)
# is deliberately NOT enumerated file-by-file here. Writing that many
# per-file paths (which are literally "<24-hex-docid>.json") into a JSON
# artifact under results/routing_design would itself be read by
# scripts/enterprise_v2/selection.py's discover_prior_docids scan (it
# walks all of "results" for 24-hex-char substrings in file content) and
# would wrongly mark nearly the entire DocILE corpus as "already used",
# emptying the eligible pool for any regeneration test. data/docile is
# input data this task never writes to; it is instead verified by a
# single aggregate directory hash (see `_docile_aggregate_hash`).
DOCILE_ROOT = "data/docile"


def _docile_aggregate_hash() -> dict:
    root = REPO_ROOT / DOCILE_ROOT
    if not root.exists():
        return {"present": False}
    files = sorted(p for p in root.rglob("*") if p.is_file())
    listing = "".join(f"{p.relative_to(REPO_ROOT)}:{_sha256(p)}\n" for p in files)
    return {"present": True, "n_files": len(files), "aggregate_sha256": hashlib.sha256(listing.encode()).hexdigest()}
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


def _is_task_output(rel: str) -> bool:
    return any(rel == r or rel.startswith(r + "/") for r in TASK_OUTPUT_ROOTS)


def protected_roots() -> list[Path]:
    roots = [p for p in sorted((REPO_ROOT / "results").iterdir()) if p.is_dir()]
    roots += [REPO_ROOT / r for r in EXTRA_ROOTS]
    return roots


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
            if _is_task_output(rel) or any(rel == c or rel.startswith(c) for c in ALLOWED_NEW_CODE):
                continue  # this task's own outputs / code are not protected
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
        raise SystemExit("usage: hash_protected_artifacts_routing_v1.py [before|after]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

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
    corpus_versions = {v: sorted(k for k in before if k.startswith(f"results/enterprise_corpus/{v.split('_training_supplement')[0]}/")) for v in ("v2", "v2_1", "v2_2")}
    per_version = {
        v: {"n_files": len(files), "n_changed": sum(1 for f in files if f in changed), "n_missing": sum(1 for f in files if f in missing),
            "tree_sha256": hashlib.sha256("".join(f"{f}:{after.get(f)}\n" for f in files).encode()).hexdigest()}
        for v, files in corpus_versions.items()
    }
    manifests = frozen_manifest_check()
    docile_ok = docile_before == docile_after
    report = {
        "protected_roots": before_doc["protected_roots"],
        "task_output_roots": list(TASK_OUTPUT_ROOTS),
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
        "enterprise_corpus_versions": per_version,
        "frozen_manifest_sha256_check": manifests,
        "docile_dataset_unchanged": docile_ok,
        "docile_before": docile_before,
        "docile_after": docile_after,
        "integrity_ok": not missing and not unexpected_changed and not unexpected_added and docile_ok and all(m["byte_identical"] for m in manifests.values()),
        "before_snapshot": str(BEFORE_PATH.relative_to(REPO_ROOT)),
        "after_snapshot": str(AFTER_PATH.relative_to(REPO_ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    summary = {k: report[k] for k in ("task_output_roots", "n_files_before", "n_files_after", "n_missing", "n_changed", "n_added", "unexpected_changed_files", "unexpected_added_files", "enterprise_corpus_versions", "docile_dataset_unchanged", "docile_before", "docile_after", "integrity_ok", "before_snapshot", "after_snapshot")}
    summary["added_files"] = added
    summary["frozen_manifest_sha256_check"] = {v: {k: m[k] for k in ("manifest", "manifest_sha256", "n_files_checked", "byte_identical")} | {"n_mismatched": len(m["mismatched"])} for v, m in manifests.items()}
    summary["full_report"] = str(REPORT_PATH.relative_to(REPO_ROOT))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[after] integrity_ok={report['integrity_ok']} missing={len(missing)} changed={len(changed)} (unexpected={len(unexpected_changed)}) added={len(added)} (unexpected={len(unexpected_added)})")
    for v, m in manifests.items():
        print(f"  {v}: manifest files checked={m['n_files_checked']} byte_identical={m['byte_identical']}")


if __name__ == "__main__":
    main()
