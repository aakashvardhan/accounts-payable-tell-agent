"""Protected-artifact integrity check for the LoRA pre-training contract v1
(v2.2 training supplement + action-stratified sampler).

Enterprise corpus v2, v2.1, and v2.2 (and every other pre-existing file in
the protected roots) are PROTECTED: nothing may be missing or changed.
Only these roots are this task's own output:

  - results/lora_training/pretraining_contract_v1
  - results/enterprise_corpus/v2_2_training_supplement
  - configs/lora/enterprise_v2_2

New files are additionally allowed only at this task's versioned code
paths (see ALLOWED_NEW_CODE). Run `before` once before any output is
generated and `after` once everything is complete. `after` also re-checks
every file listed in the frozen v2 / v2.1 / v2.2 manifests against the
sha256 recorded in the manifest itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
OUT_DIR = REPO_ROOT / "results" / "lora_training" / "pretraining_contract_v1"
# Full snapshots name DocILE files by docid, so they live under the
# supplement root: the frozen v2 selection code excludes documents named
# anywhere under results/ except results/enterprise_corpus/** (and a few
# other roots), and results/lora_training/** must stay docid-free or the
# frozen corpora would no longer regenerate byte-identically.
SNAP_DIR = REPO_ROOT / "results" / "enterprise_corpus" / "v2_2_training_supplement" / "integrity"
BEFORE_PATH = SNAP_DIR / "protected_artifact_hashes_before.json"
AFTER_PATH = SNAP_DIR / "protected_artifact_hashes_after.json"
REPORT_PATH = SNAP_DIR / "protected_artifact_integrity.json"
SUMMARY_PATH = OUT_DIR / "protected_artifact_integrity_summary.json"

TASK_OUTPUT_ROOTS = (
    "results/lora_training/pretraining_contract_v1",
    "results/enterprise_corpus/v2_2_training_supplement",
    "configs/lora/enterprise_v2_2",
)
ALLOWED_NEW_CODE = (
    "scripts/lora_pretraining_v1/",
    "scripts/build_lora_pretraining_contract_v1.py",
    "scripts/hash_protected_artifacts_lora_pretraining_v1.py",
    "tests/test_lora_pretraining_contract_v1.py",
)
EXTRA_ROOTS = ("data/scenarios", "data/docile", "configs", "src", "scripts", "tests")
FROZEN_CORPUS_MANIFESTS = {
    "v2": "results/enterprise_corpus/v2/enterprise_corpus_v2_manifest.json",
    "v2_1": "results/enterprise_corpus/v2_1/enterprise_corpus_v2_1_manifest.json",
    "v2_2": "results/enterprise_corpus/v2_2/enterprise_corpus_v2_2_manifest.json",
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
    """Every {path, sha256} pair recorded anywhere inside a frozen manifest."""
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
        raise SystemExit("usage: hash_protected_artifacts_lora_pretraining_v1.py [before|after]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "before":
        if BEFORE_PATH.exists():
            raise SystemExit(f"{BEFORE_PATH} already exists -- refusing to overwrite the pre-work snapshot.")
        roots = [str(p.relative_to(REPO_ROOT)) for p in protected_roots()]
        snap = snapshot()
        BEFORE_PATH.write_text(json.dumps({"protected_roots": roots, "n_files": len(snap), "hashes": snap}, indent=1, sort_keys=True))
        print(f"[before] hashed {len(snap)} protected files -> {BEFORE_PATH}")
        return

    before_doc = json.loads(BEFORE_PATH.read_text())
    before = before_doc["hashes"]
    after = snapshot(before_doc["protected_roots"])
    AFTER_PATH.write_text(json.dumps({"protected_roots": before_doc["protected_roots"], "n_files": len(after), "hashes": after}, indent=1, sort_keys=True))

    missing = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    added = sorted(set(after) - set(before))
    unexpected_added = [a for a in added if not any(a == c or a.startswith(c) for c in ALLOWED_NEW_CODE)]
    corpus_versions = {v: sorted(k for k in before if k.startswith(f"results/enterprise_corpus/{v}/")) for v in FROZEN_CORPUS_MANIFESTS}
    per_version = {
        v: {"n_files": len(files), "n_changed": sum(1 for f in files if f in changed), "n_missing": sum(1 for f in files if f in missing),
            "tree_sha256": hashlib.sha256("".join(f"{f}:{after.get(f)}\n" for f in files).encode()).hexdigest()}
        for v, files in corpus_versions.items()
    }
    manifests = frozen_manifest_check()
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
        "added_files": added,
        "unexpected_added_files": unexpected_added,
        "enterprise_corpus_versions": per_version,
        "frozen_manifest_sha256_check": manifests,
        "integrity_ok": not missing and not changed and not unexpected_added and all(m["byte_identical"] for m in manifests.values()),
        "before_snapshot": str(BEFORE_PATH.relative_to(REPO_ROOT)),
        "after_snapshot": str(AFTER_PATH.relative_to(REPO_ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    # docid-free summary (counts, booleans, manifest checks only)
    summary = {k: report[k] for k in ("task_output_roots", "n_files_before", "n_files_after", "n_missing", "n_changed", "n_added", "unexpected_added_files", "enterprise_corpus_versions", "integrity_ok", "before_snapshot", "after_snapshot")}
    summary["added_files"] = added
    summary["frozen_manifest_sha256_check"] = {v: {k: m[k] for k in ("manifest", "manifest_sha256", "n_files_checked", "byte_identical")} | {"n_mismatched": len(m["mismatched"])} for v, m in manifests.items()}
    summary["full_report"] = str(REPORT_PATH.relative_to(REPO_ROOT))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[after] integrity_ok={report['integrity_ok']} missing={len(missing)} changed={len(changed)} added={len(added)} unexpected_added={len(unexpected_added)}")
    for v, m in manifests.items():
        print(f"  {v}: manifest files checked={m['n_files_checked']} byte_identical={m['byte_identical']}")


if __name__ == "__main__":
    main()
