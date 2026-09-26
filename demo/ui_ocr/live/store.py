"""Shared persistence for the live runtime (stdlib only, so the torch-free web server can import it).

One SQLite file (`<runtime>/jobs.sqlite`) is shared by the web server and the live worker (WAL, busy timeout):
  jobs          -- owned by intake.py (upload/extraction); the worker only advances `status`, `progress`, `latest_event`, `updated_at`
  runs          -- one row per agent execution (run id is generated at upload)
  run_events    -- ordered, persisted runtime events. The UI displays ONLY records that exist here (or intake events in jobs.events)
  runtime_info  -- identifiers the worker actually loaded (model revision, probe hash, adapter hash, threshold) + heartbeat
Nothing here derives an outcome: it stores what the runtime reported.
"""
import json
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

# ---- event contract (types the runtime can emit; the UI adapter also maps intake events onto these)
EVENT_TYPES = (
    "intake_received", "intake_validated", "extraction_started", "extraction_progress", "extraction_completed",
    "job_queued", "job_claimed", "job_requeued",
    "agent_turn_started", "context_prepared", "tool_call_requested", "tool_call_completed", "untrusted_content_entered_context",
    "activation_captured", "probe_scored", "route_selected", "model_generation_completed",
    "action_proposed", "action_parse_failed", "trusted_lookup_completed",
    "gate_evaluated", "memory_written", "memory_quarantined", "clarification_created", "evidence_report_created",
    "payment_intent_created", "ledger_posted", "job_completed", "job_failed",
    "reviewer_decision", "case_escalated",
)
EVENT_STATUSES = ("ok", "warning", "blocked", "failed", "info")
PROVENANCE = ("intake", "extraction", "runtime", "model", "probe", "trusted_lookup", "gate", "ledger", "outbox", "review_store", "application", "memory", "reviewer")
RUN_ID_RE = re.compile(r"^run-[0-9a-f]{32}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, execution_id TEXT, state TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  outcome TEXT, error TEXT, worker_id TEXT, attempt INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS run_events(
  event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts TEXT NOT NULL, type TEXT NOT NULL, title TEXT NOT NULL,
  payload TEXT NOT NULL, provenance TEXT NOT NULL, status TEXT NOT NULL, duration_ms REAL, UNIQUE(run_id, seq));
CREATE INDEX IF NOT EXISTS run_events_run ON run_events(run_id, seq);
CREATE TABLE IF NOT EXISTS runtime_info(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_queue(
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL, decision TEXT NOT NULL CHECK(decision IN ('approve','reject','clarify')), reviewer_id TEXT NOT NULL,
  note TEXT, created_at TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('pending','claimed','done')), claimed_at TEXT, done_at TEXT, result TEXT);
CREATE INDEX IF NOT EXISTS review_queue_job ON review_queue(job_id, created_at);
"""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def connect(path):
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def ensure_schema(conn):
    conn.executescript(SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if cols and "run_id" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN run_id TEXT")
    if cols and "tell_secured" not in cols:        # 1 = TellSecured ON (probe + Agent S + validator/gate), 0 = undefended Agent-1 baseline
        conn.execute("ALTER TABLE jobs ADD COLUMN tell_secured INTEGER NOT NULL DEFAULT 1")


def new_run_id():
    return "run-" + uuid.uuid4().hex


class RunStore:
    """Thread-safe writer/reader of runs + run_events. One instance per connection."""

    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.RLock()

    # -- runs
    def ensure_run(self, run_id, job_id):
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO runs(run_id, job_id, state) VALUES(?,?,?)", (run_id, job_id, "queued"))

    def start_execution(self, run_id, worker_id):
        with self.lock:
            attempt = self.conn.execute("SELECT attempt FROM runs WHERE run_id=?", (run_id,)).fetchone()["attempt"] + 1
            exec_id = "exec-" + uuid.uuid4().hex[:16]
            self.conn.execute("UPDATE runs SET execution_id=?, state='running', started_at=?, finished_at=NULL, worker_id=?, attempt=?, error=NULL WHERE run_id=?",
                              (exec_id, now(), worker_id, attempt, run_id))
            return exec_id, attempt

    def finish_run(self, run_id, state, outcome=None, error=None):
        with self.lock:
            self.conn.execute("UPDATE runs SET state=?, finished_at=?, outcome=?, error=? WHERE run_id=?",
                              (state, now(), json.dumps(outcome) if outcome is not None else None, error, run_id))

    def get_run(self, run_id):
        r = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["outcome"] = json.loads(d["outcome"]) if d["outcome"] else None
        return d

    # -- events
    def emit(self, run_id, type_, title, payload=None, *, provenance="runtime", status="ok", duration_ms=None):
        if type_ not in EVENT_TYPES:
            raise ValueError(f"unknown event type {type_!r}")
        if provenance not in PROVENANCE or status not in EVENT_STATUSES:
            raise ValueError("bad provenance/status")
        with self.lock:
            seq = self.conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM run_events WHERE run_id=?", (run_id,)).fetchone()["n"]
            ev = {"event_id": "ev-" + uuid.uuid4().hex[:20], "run_id": run_id, "seq": seq, "ts": now(), "type": type_, "title": title,
                  "payload": payload or {}, "provenance": provenance, "status": status, "duration_ms": None if duration_ms is None else round(float(duration_ms), 1)}
            self.conn.execute("INSERT INTO run_events(event_id,run_id,seq,ts,type,title,payload,provenance,status,duration_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (ev["event_id"], run_id, seq, ev["ts"], type_, title, json.dumps(ev["payload"], default=str), provenance, status, ev["duration_ms"]))
            return ev

    def list_events(self, run_id, after_seq=-1):
        rows = self.conn.execute("SELECT * FROM run_events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after_seq)).fetchall()
        return [dict(r, payload=json.loads(r["payload"])) for r in rows]

    # -- runtime identifiers / heartbeat
    def set_info(self, key, value):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO runtime_info(key,value,updated_at) VALUES(?,?,?)", (key, json.dumps(value, default=str), now()))

    def get_info(self, key):
        r = self.conn.execute("SELECT value, updated_at FROM runtime_info WHERE key=?", (key,)).fetchone()
        return (json.loads(r["value"]), r["updated_at"]) if r else (None, None)

    def all_info(self):
        return {r["key"]: {"value": json.loads(r["value"]), "updated_at": r["updated_at"]} for r in self.conn.execute("SELECT * FROM runtime_info").fetchall()}


def heartbeat_age_s(updated_at):
    if not updated_at:
        return None
    try:
        t = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return round((datetime.now(timezone.utc) - t).total_seconds(), 1)
