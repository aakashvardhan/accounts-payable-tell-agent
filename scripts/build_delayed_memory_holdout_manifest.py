"""Section 10: a holdout *pointer* manifest referencing (never copying)
the existing `retrieval_post_memory` activations from the conditional-
memory-retrieval experiment -- reserved for a later zero-shot test of
whatever probe is eventually trained on this corpus. Nothing here is
used for probe fitting, layer selection, hyperparameter selection, or
threshold calibration; this script does not even look at the run's
exposure label beyond confirming the referenced files exist and are
loadable.

References (does not copy):
  - task-only clean-memory `retrieval_post_memory` activation
  - task-only poisoned-memory `retrieval_post_memory` activation
  - hardened clean-memory and poisoned-memory `retrieval_post_memory`
    activations, for descriptive comparison only

Writes results/probe_dataset/delayed_memory_holdout_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors import safe_open

RESULTS_JSON_PATH = Path("/home/hp5/tell/results/evaluation/conditional_memory_retrieval_v1/conditional_retrieval_results.json")
OUTPUT_PATH = Path("/home/hp5/tell/results/probe_dataset/delayed_memory_holdout_manifest.json")

HOLDOUT_TAG = "zero_shot_delayed_memory_holdout"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_safetensors(path: Path) -> dict:
    """Confirms the file exists and is loadable, and reports which
    hidden-state indices it contains -- does not read or interpret any
    label."""
    with safe_open(str(path), framework="pt") as f:
        keys = sorted(f.keys())
        shapes = {k: list(f.get_slice(k).get_shape()) for k in keys}
    return {"keys": keys, "shapes": shapes}


def main() -> None:
    if not RESULTS_JSON_PATH.exists():
        raise RuntimeError(f"{RESULTS_JSON_PATH} does not exist -- run scripts/run_conditional_memory_retrieval.py first.")
    results = json.loads(RESULTS_JSON_PATH.read_text())

    entries = []
    for section, profiles in (("clean_control", results["clean_control"]), ("poisoned_retrieval", results["poisoned_retrieval"])):
        for profile, ev in profiles.items():
            if ev is None:
                continue
            if not ev.get("retrieval_post_memory_reached"):
                continue
            path = Path(ev["retrieval_post_memory_safetensors_path"])
            metadata_path = path.with_name(path.stem + "_metadata.json")
            if not path.exists() or not metadata_path.exists():
                raise RuntimeError(f"Referenced activation artifact missing: {path}")
            tensor_info = _validate_safetensors(path)
            entries.append(
                {
                    "condition_id": ev["condition_id"],
                    "prompt_profile": profile,
                    "memory_condition": "poisoned" if section == "poisoned_retrieval" else "clean",
                    "decision_point": "retrieval_post_memory",
                    "safetensors_path": str(path),
                    "safetensors_sha256": _sha256_file(path),
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": _sha256_file(metadata_path),
                    "tensor_keys": tensor_info["keys"],
                    "tensor_shapes": tensor_info["shapes"],
                    "holdout_tag": HOLDOUT_TAG,
                }
            )

    manifest = {
        "holdout_tag": HOLDOUT_TAG,
        "source_results_path": str(RESULTS_JSON_PATH),
        "source_results_sha256": _sha256_file(RESULTS_JSON_PATH),
        "entries": entries,
        "usage_restrictions": [
            "Must not be used for probe fitting.",
            "Must not be used for layer selection.",
            "Must not be used for hyperparameter selection.",
            "Must not be used for threshold calibration.",
            "May only be used for a later, explicit zero-shot evaluation of an already-fixed probe.",
        ],
        "note": (
            "This manifest only confirms the referenced safetensors/metadata files exist and are loadable "
            "(tensor keys and shapes recorded above); it does not inspect or record any exposure/attack label "
            "for these runs beyond the condition_id and memory_condition already public in the conditional-"
            "retrieval report."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {OUTPUT_PATH} ({len(entries)} holdout entries)")


if __name__ == "__main__":
    main()
