"""Protected-artifact integrity check for the enterprise-corpus-v2.2 resolution-policy correction. All v2 and v2.1
outputs, configs, and code are PROTECTED here; v2 outputs
(results/enterprise_corpus/v2, results/enterprise_benchmark/v1,
results/economics/enterprise_v1, configs/enterprise_corpus/*.json) are
PROTECTED here; only the v2_1 / v1_1 roots are this task's output.

New, versioned script -- `scripts/hash_protected_artifacts.py` (the LoRA
smoke-training version) is left untouched because its output location is
itself a protected pilot directory.

Run `before` once, before any enterprise-v2 file is generated, and
`after` once all generation/validation is complete. `after` diffs the two
snapshots and writes `protected_artifact_integrity.json`.

Protected scope (every pre-existing file, recursively):
  - every pre-existing subdirectory of results/ (probe corpus v1/v1.1,
    probe pilot_v1, LoRA corpus v1, LoRA smoke_v1, autonomous-loop v1/v2,
    prompt-policy ablations, memory pilots, conditional-retrieval,
    reporting, runtime, traces, scenario design, dataset inspection,
    activations) -- enumerated at `before` time, so the new
    enterprise_corpus / enterprise_benchmark / economics output roots are
    NOT protected (they are this task's own output);
  - data/scenarios, data/docile (official DocILE source files);
  - configs/lora, configs/probe (trained adapter/probe configs + weights);
  - every pre-existing file under src/, scripts/, tests/ (so a silently
    edited utility would be detected);
  - every *.sqlite file anywhere under the repository.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")
OUT_DIR = REPO_ROOT / "results" / "enterprise_corpus" / "v2_2"
BEFORE_PATH = OUT_DIR / "protected_artifact_hashes_before.json"
AFTER_PATH = OUT_DIR / "protected_artifact_hashes_after.json"
REPORT_PATH = OUT_DIR / "protected_artifact_integrity.json"

# Output roots created by this task -- never part of the protected set.
TASK_OUTPUT_ROOTS = (
    "results/enterprise_corpus/v2_2",
    "results/enterprise_benchmark/v1_2",
    "configs/enterprise_corpus/v2_2",
)
_UNUSED_V2_ROOTS = (
    "results/enterprise_corpus",
    "results/enterprise_benchmark",
    "results/economics",
    "configs/enterprise_corpus",
)

EXTRA_ROOTS = ("data/scenarios", "data/docile", "configs", "src", "scripts", "tests")


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
    roots = [p for p in roots if not _is_task_output(str(p.relative_to(REPO_ROOT)))]
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
            if _is_task_output(rel):
                continue
            out[rel] = _sha256(p)
    for p in sorted(REPO_ROOT.rglob("*.sqlite")):
        if ".venv" in p.parts or ".venv-inspect" in p.parts:
            continue
        out[str(p.relative_to(REPO_ROOT))] = _sha256(p)
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    if mode not in ("before", "after"):
        raise SystemExit("usage: hash_protected_artifacts_enterprise_v2.py [before|after]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # New files are expected only under scripts/ and tests/ (this task's
    # generation scripts and CPU tests). Anything else added inside a
    # protected root is a violation.
    unexpected_added = [a for a in added if not (a.startswith("scripts/") or a.startswith("tests/"))]
    report = {
        "protected_roots": before_doc["protected_roots"],
        "n_files_before": len(before),
        "n_files_after": len(after),
        "n_missing": len(missing),
        "n_changed": len(changed),
        "n_added": len(added),
        "missing_files": missing,
        "changed_files": changed,
        "added_files": added,
        "unexpected_added_files": unexpected_added,
        "integrity_ok": not missing and not changed and not unexpected_added,
        "before_snapshot": str(BEFORE_PATH.relative_to(REPO_ROOT)),
        "after_snapshot": str(AFTER_PATH.relative_to(REPO_ROOT)),
        "note": "Added files under scripts/ and tests/ are this task's new, versioned generation scripts and CPU tests; "
        "no pre-existing file may be missing or changed.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[after] integrity_ok={report['integrity_ok']} missing={len(missing)} changed={len(changed)} added={len(added)} unexpected_added={len(unexpected_added)}")


if __name__ == "__main__":
    main()
