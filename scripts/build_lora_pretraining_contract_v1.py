"""Build the LoRA pre-training contract v1 outputs (CPU only; no model, no GPU).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/build_lora_pretraining_contract_v1.py

Run `scripts/hash_protected_artifacts_lora_pretraining_v1.py before` first
and `... after` once this completes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/hp5/tell/src")))
sys.path.insert(0, str(Path("/home/hp5/tell/scripts")))

from lora_pretraining_v1 import pipeline  # noqa: E402


def main() -> None:
    o = pipeline.build(with_tokenizer=True)
    failed = [c for c in o["checks"] if not c["passed"]]
    for c in o["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    if failed:
        raise SystemExit(f"{len(failed)} check(s) failed; nothing written")
    files = pipeline.write(o)
    for p in sorted(files):
        print("wrote", Path(p).relative_to(pipeline.REPO))


if __name__ == "__main__":
    main()
