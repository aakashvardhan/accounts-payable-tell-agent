"""Measures Part 2's required "at least 25% of history-required scenarios
have a different next action than their matched no-history negative" bar,
using tell.memory.next_step against a small paired synthetic fixture set
(no real enterprise data; the point is to prove the mapping's coverage,
mirrored by tests/test_memory_next_step.py).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/build_memory_next_step_measurement_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hp5/tell")
sys.path.insert(0, str(REPO / "src"))

from tell.memory.next_step import (  # noqa: E402
    LinkedHistoryStatus,
    RequiredNextStep,
    RetrievedMemorySignal,
    decide_next_step,
)

OUT_PATH = REPO / "results" / "routing_design" / "memory_next_step_measurement_v1.json"

# Two matched scenarios per signal: history-required (linked_history
# pending, a real retrieved signal) vs. the matched no-history negative
# (linked_history NONE -- the same workflow, but nothing ties it to prior
# history). This mirrors Part 2's "matched no-history negative" language.
SIGNALS = list(RetrievedMemorySignal)


def main() -> None:
    rows = []
    for signal in SIGNALS:
        history_required = decide_next_step(LinkedHistoryStatus.PENDING, signal)
        no_history_negative = decide_next_step(LinkedHistoryStatus.NONE, signal)
        differs = history_required.next_step != no_history_negative.next_step
        rows.append({
            "signal": signal.value,
            "history_required_next_step": history_required.next_step.value,
            "no_history_negative_next_step": no_history_negative.next_step.value,
            "differs": differs,
        })
    n_differ = sum(1 for r in rows if r["differs"])
    pct = n_differ / len(rows)
    report = {
        "status": "measured against tell.memory.next_step's full signal table (synthetic paired scenarios, no enterprise data)",
        "n_history_required_scenarios": len(rows),
        "n_differing_from_matched_no_history_negative": n_differ,
        "measured_percentage": round(pct, 4),
        "meets_25_percent_bar": pct >= 0.25,
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
