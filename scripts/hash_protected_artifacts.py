"""Hashes every file under the directories this LoRA-corpus/smoke-training
task must never modify, before any change is made, and again afterward to
prove nothing changed. Run with `mode=before` first, then `mode=after`
once all work is done; the two manifests are then diff-checked.

Protected roots: results/probe_dataset, results/probe_training,
data/scenarios (existing scenarios), results/activations (existing
activation experiments), results/evaluation (existing evaluation
reports), every *.sqlite file anywhere under results/ or data/, and
results/runtime (existing model runtime reports). results/scenario_design
and results/dataset_inspection and results/traces are included too, since
they hold frozen protocol manifests and dataset-inspection artifacts this
task must not touch either.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hp5/tell")

PROTECTED_ROOTS = [
    REPO_ROOT / "results" / "probe_dataset",
    REPO_ROOT / "results" / "probe_training",
    REPO_ROOT / "results" / "activations",
    REPO_ROOT / "results" / "evaluation",
    REPO_ROOT / "results" / "runtime",
    REPO_ROOT / "results" / "scenario_design",
    REPO_ROOT / "results" / "dataset_inspection",
    REPO_ROOT / "results" / "traces",
    REPO_ROOT / "data" / "scenarios",
]

OUTPUT_DIR = REPO_ROOT / "results" / "lora_training" / "smoke_v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(REPO_ROOT))] = _sha256_file(p)
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    if mode not in ("before", "after"):
        raise SystemExit("usage: hash_protected_artifacts.py [before|after]")

    manifest: dict[str, str] = {}
    for root in PROTECTED_ROOTS:
        manifest.update(_hash_tree(root))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"protected_artifact_hashes_{mode}.json"
    out_path.write_text(json.dumps({"n_files": len(manifest), "hashes": manifest}, indent=2, sort_keys=True))
    print(f"[{mode}] hashed {len(manifest)} files under {len(PROTECTED_ROOTS)} protected roots -> {out_path}")

    if mode == "after":
        before_path = OUTPUT_DIR / "protected_artifact_hashes_before.json"
        if not before_path.exists():
            raise RuntimeError(f"{before_path} does not exist -- run with mode=before first.")
        before = json.loads(before_path.read_text())["hashes"]
        after = manifest

        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        changed = sorted(k for k in (set(before) & set(after)) if before[k] != after[k])

        report = {
            "n_files_before": len(before),
            "n_files_after": len(after),
            "n_missing": len(missing),
            "n_added": len(added),
            "n_changed": len(changed),
            "missing_files": missing,
            "added_files": added,
            "changed_files": changed,
            "integrity_ok": not missing and not changed,
            "note": "added_files (new protected-tree files with no 'before' counterpart) do not by themselves indicate a violation "
            "unless the task description forbids adding files there too; missing_files and changed_files always indicate a violation.",
        }
        report_path = OUTPUT_DIR / "protected_artifact_integrity_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[after] integrity_ok={report['integrity_ok']} (missing={len(missing)}, changed={len(changed)}, added={len(added)})")
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
