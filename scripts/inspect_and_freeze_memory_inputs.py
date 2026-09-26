"""Section 1 of the conditional-memory-retrieval experiment: inspects
the two SQLite databases produced by the real `task_only_base_v1`
delayed-memory-poisoning pilot (scripts/run_task_only_memory_pilot.py),
confirms every required invariant about the clean and poisoned memory
records, and creates immutable, hashed snapshot copies for the new
experiment -- so every downstream run (all four profile x condition
combinations) reads the same two byte-identical files, and the original
task-only databases are never opened for writing here.

This script only ever opens the *source* task-only databases with
read-only SQLite connections (`file:...?mode=ro`) and hashes them before
and after every read, to prove inspection did not modify them. It then
copies the raw bytes (never re-serializes rows) to the new snapshot
paths and hashes the copies too.

Writes:
  - results/scenario_design/conditional_memory_retrieval_v1/memory_clean_snapshot.sqlite
  - results/scenario_design/conditional_memory_retrieval_v1/memory_poisoned_snapshot.sqlite
  - results/scenario_design/conditional_memory_retrieval_v1/memory_inputs_manifest.json

Once written, memory_inputs_manifest.json must not be edited after the
first generation of the downstream experiment begins.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SOURCE_CLEAN_DB = Path("/home/hp5/tell/results/runtime/task_only_base_v1/memory_clean.sqlite")
SOURCE_POISONED_DB = Path("/home/hp5/tell/results/runtime/task_only_base_v1/memory_poisoned.sqlite")
SOURCE_TRACE_PATH = Path("/home/hp5/tell/results/traces/task_only_base_v1/memory_pilot_task_only.jsonl")

OUTPUT_DIR = Path("/home/hp5/tell/results/scenario_design/conditional_memory_retrieval_v1")
SNAPSHOT_CLEAN_DB = OUTPUT_DIR / "memory_clean_snapshot.sqlite"
SNAPSHOT_POISONED_DB = OUTPUT_DIR / "memory_poisoned_snapshot.sqlite"
MANIFEST_PATH = OUTPUT_DIR / "memory_inputs_manifest.json"

ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"
VENDOR_ID = "SIM-VENDOR-MDSPHARMA-0001"
TRUSTED_VENDOR_STATE_PATH = Path("/home/hp5/tell/data/scenarios/clean/clean_04d531ca_v1.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_only_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _inspect(label: str, path: Path) -> dict:
    hash_before = _sha256_file(path)

    conn = _read_only_connect(path)
    tables = sorted(r[0] for r in conn.execute("select name from sqlite_master where type='table'").fetchall())
    ledger_tables = [t for t in tables if t in ("vendors", "invoices", "payment_intents", "accounts", "journal_entries")]

    records = []
    for row in conn.execute("select * from memories order by created_at").fetchall():
        records.append(dict(row))
    conn.close()

    hash_after = _sha256_file(path)
    if hash_before != hash_after:
        raise RuntimeError(f"{path} was modified during inspection ({hash_before} -> {hash_after}) -- aborting.")

    return {
        "label": label,
        "path": str(path),
        "file_sha256": hash_after,
        "tables": tables,
        "contains_ledger_tables": bool(ledger_tables),
        "records": records,
    }


def _find_write_memory_trace_turn(run_id: str, expected_content: str) -> dict | None:
    """Finds the real Session-A `write_memory` turn in the task-only
    trace whose action content matches the persisted record exactly --
    proof the record was written by the real model loop, not fabricated
    or app-injected."""
    if not SOURCE_TRACE_PATH.exists():
        return None
    with SOURCE_TRACE_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            if (
                rec.get("record_type") == "turn"
                and rec.get("run_id") == run_id
                and rec.get("action") is not None
                and rec["action"].get("action") == "write_memory"
                and rec["action"].get("content") == expected_content
            ):
                return {
                    "turn_number": rec["turn_number"],
                    "decision_point": rec["decision_point"],
                    "parse_outcome": rec["parse_outcome"],
                    "raw_output": rec["raw_output"],
                    "prompt_profile": rec.get("prompt_profile"),
                }
    return None


def main() -> None:
    if not SOURCE_CLEAN_DB.exists() or not SOURCE_POISONED_DB.exists():
        raise RuntimeError("Task-only memory databases not found -- run scripts/run_task_only_memory_pilot.py first.")

    clean_inspection = _inspect("clean", SOURCE_CLEAN_DB)
    poisoned_inspection = _inspect("poisoned", SOURCE_POISONED_DB)

    if len(clean_inspection["records"]) != 1:
        raise RuntimeError(f"Expected exactly 1 clean memory record, found {len(clean_inspection['records'])}")
    if len(poisoned_inspection["records"]) != 1:
        raise RuntimeError(f"Expected exactly 1 poisoned memory record, found {len(poisoned_inspection['records'])}")

    clean_record = clean_inspection["records"][0]
    poisoned_record = poisoned_inspection["records"][0]

    # ---- Required confirmations ----
    checks = {
        "poison_written_by_real_model": None,  # filled below
        "attacker_account_in_poisoned_memory": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID in poisoned_record["content"],
        "attacker_account_absent_from_trusted_vendor_state": None,  # filled below
        "clean_record_active": clean_record["status"] == "active",
        "poisoned_record_active": poisoned_record["status"] == "active",
        "clean_db_has_no_ledger_tables": not clean_inspection["contains_ledger_tables"],
        "poisoned_db_has_no_ledger_tables": not poisoned_inspection["contains_ledger_tables"],
        "clean_record_content_hash_matches_stored_content_sha256": (
            hashlib.sha256(clean_record["content"].encode("utf-8")).hexdigest() == clean_record["content_sha256"]
        ),
        "poisoned_record_content_hash_matches_stored_content_sha256": (
            hashlib.sha256(poisoned_record["content"].encode("utf-8")).hexdigest() == poisoned_record["content_sha256"]
        ),
    }

    trusted_state_text = TRUSTED_VENDOR_STATE_PATH.read_text()
    trusted_state = json.loads(trusted_state_text)["trusted_state"]
    checks["attacker_account_absent_from_trusted_vendor_state"] = ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID not in json.dumps(trusted_state)

    poison_provenance_turn = _find_write_memory_trace_turn("task_only_memory_pilot_session_a_poisoned", poisoned_record["content"])
    clean_provenance_turn = _find_write_memory_trace_turn("task_only_memory_pilot_session_a_clean", clean_record["content"])
    checks["poison_written_by_real_model"] = poison_provenance_turn is not None and poison_provenance_turn["parse_outcome"] == "valid"

    if not all(checks.values()):
        failed = {k: v for k, v in checks.items() if not v}
        raise RuntimeError(f"Memory-input invariant checks failed: {failed}")

    # ---- Immutable snapshot copies ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CLEAN_DB, SNAPSHOT_CLEAN_DB)
    shutil.copy2(SOURCE_POISONED_DB, SNAPSHOT_POISONED_DB)
    snapshot_clean_hash = _sha256_file(SNAPSHOT_CLEAN_DB)
    snapshot_poisoned_hash = _sha256_file(SNAPSHOT_POISONED_DB)
    if snapshot_clean_hash != clean_inspection["file_sha256"]:
        raise RuntimeError("Clean snapshot copy hash does not match source -- copy corrupted.")
    if snapshot_poisoned_hash != poisoned_inspection["file_sha256"]:
        raise RuntimeError("Poisoned snapshot copy hash does not match source -- copy corrupted.")

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_databases": {
            "clean": {"path": str(SOURCE_CLEAN_DB), "sha256": clean_inspection["file_sha256"], "tables": clean_inspection["tables"]},
            "poisoned": {"path": str(SOURCE_POISONED_DB), "sha256": poisoned_inspection["file_sha256"], "tables": poisoned_inspection["tables"]},
        },
        "snapshot_databases": {
            "clean": {"path": str(SNAPSHOT_CLEAN_DB), "sha256": snapshot_clean_hash},
            "poisoned": {"path": str(SNAPSHOT_POISONED_DB), "sha256": snapshot_poisoned_hash},
        },
        "clean_memory_record": clean_record,
        "poisoned_memory_record": poisoned_record,
        "vendor_id": VENDOR_ID,
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "trusted_vendor_state_path": str(TRUSTED_VENDOR_STATE_PATH),
        "trusted_vendor_state_sha256": hashlib.sha256(trusted_state_text.encode("utf-8")).hexdigest(),
        "provenance_proof": {
            "clean_record_write_memory_turn": clean_provenance_turn,
            "poisoned_record_write_memory_turn": poison_provenance_turn,
            "source_trace_path": str(SOURCE_TRACE_PATH),
            "source_trace_sha256": _sha256_file(SOURCE_TRACE_PATH) if SOURCE_TRACE_PATH.exists() else None,
        },
        "checks": checks,
        "notes": [
            "Both prompt profiles in the downstream experiment use these exact snapshot files -- never the "
            "original task-only databases under results/runtime/task_only_base_v1/, which remain untouched.",
            "Snapshot files are byte-for-byte copies (shutil.copy2 of the raw file), never re-serialized from "
            "the row data above, so memory_id/content/content_sha256/origin_* fields are guaranteed identical "
            "to the source.",
            "The downstream experiment must call only MemoryStore.search_memories (a read) against these "
            "snapshots, never append_memory/quarantine_memory -- verified after all inference by re-hashing "
            "both snapshot files against snapshot_databases.*.sha256 above.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {SNAPSHOT_CLEAN_DB} ({snapshot_clean_hash[:16]}...)")
    print(f"Wrote {SNAPSHOT_POISONED_DB} ({snapshot_poisoned_hash[:16]}...)")
    print(f"All checks passed: {all(checks.values())}")


if __name__ == "__main__":
    main()
