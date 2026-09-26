#!/usr/bin/env python3
"""Live runtime worker. Run with the project's Python (torch/peft/transformers), e.g.

.venv/bin/python demo/ui_ocr/live/worker.py --runtime-dir demo/ui_ocr/runtime [--runtime-dir demo/ui_ocr/runtime_public]

One process, one copy of Qwen3-8B + the frozen LoRA + the frozen probe; GPU inference is serialised (one job at a time). It serves any
number of isolated runtime directories (each with its own jobs.sqlite, ledger, outbox and review store), round-robin.

The worker never touches training data, the frozen adapter/probe files (read-only, hash-verified), datasets or evaluation artifacts.
"""
import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from live import store as S  # noqa: E402

STOP = False


def _stop(*_):
    global STOP
    STOP = True


class Lane:
    """One runtime directory: DB, run store, trusted context."""

    def __init__(self, runtime_dir):
        from live import trusted as T
        self.dir = Path(runtime_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.conn = S.connect(self.dir / "jobs.sqlite")
        # the server owns the jobs/outbox tables; make sure they exist before we look at them
        self.conn.executescript("""
CREATE TABLE IF NOT EXISTS jobs(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, sha256 TEXT, display_name TEXT NOT NULL, stored_name TEXT,
  size INTEGER, source TEXT NOT NULL, mime TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, status TEXT NOT NULL,
  progress INTEGER NOT NULL, latest_event TEXT, events TEXT NOT NULL, extraction TEXT, error TEXT, duplicate_of TEXT, claimed INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY, job_id TEXT NOT NULL, created_at TEXT NOT NULL, send_status TEXT NOT NULL, record TEXT NOT NULL);
""")
        S.ensure_schema(self.conn)
        self.store = S.RunStore(self.conn)
        self.ctx = T.LiveContext(self.dir)

    def claim(self):
        r = self.conn.execute("""UPDATE jobs SET status='PROCESSING', progress=60, latest_event='Live agent run started', updated_at=?
                                 WHERE id=(SELECT id FROM jobs WHERE status='READY_FOR_PROCESSING' ORDER BY seq LIMIT 1) RETURNING id""", (S.now(),)).fetchone()
        return r["id"] if r else None

    def recover_stale(self):
        rows = self.conn.execute("SELECT id, run_id FROM jobs WHERE status='PROCESSING'").fetchall()
        for r in rows:
            run_id = r["run_id"]
            if run_id:
                self.store.ensure_run(run_id, r["id"])
                self.store.emit(run_id, "job_requeued", "Job re-queued after a worker restart (payments are idempotent)", {"previous_status": "PROCESSING"}, provenance="runtime", status="warning")
            self.conn.execute("UPDATE jobs SET status='READY_FOR_PROCESSING', progress=50, latest_event='Re-queued after worker restart', updated_at=? WHERE id=?", (S.now(), r["id"]))
        return len(rows)

    def claim_review(self):
        r = self.conn.execute("""UPDATE review_queue SET state='claimed', claimed_at=?
                                 WHERE id=(SELECT id FROM review_queue WHERE state='pending' ORDER BY created_at LIMIT 1) RETURNING *""", (S.now(),)).fetchone()
        return dict(r) if r else None

    def recover_reviews(self):
        return self.conn.execute("UPDATE review_queue SET state='pending', claimed_at=NULL WHERE state='claimed'").rowcount

    def handle_reset(self):
        info, _ = self.store.get_info("reset_requested")
        if info:
            self.ctx.reset()
            self.store.set_info("reset_requested", None)
            self.store.set_info("reset_completed", {"at": S.now()})

    def job(self, job_id):
        r = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        d = dict(r)
        d["extraction"] = json.loads(d["extraction"]) if d["extraction"] else {}
        return d


def process(lane, ports, job_id, worker_id, fail_tool=None):
    from live.agent import LiveAgentRun, RunResult
    job = lane.job(job_id)
    run_id = job.get("run_id") or S.new_run_id()
    if not job.get("run_id"):
        lane.conn.execute("UPDATE jobs SET run_id=? WHERE id=?", (run_id, job_id))
    lane.store.ensure_run(run_id, job_id)
    exec_id, attempt = lane.store.start_execution(run_id, worker_id)
    lane.store.emit(run_id, "job_claimed", f"Live worker claimed the job (attempt {attempt})", {"worker_id": worker_id, "execution_id": exec_id, "attempt": attempt}, provenance="runtime")
    try:
        tools = None
        if fail_tool:
            from live.testing import FaultyTools
            from live.tools import LiveTools
            tools = FaultyTools(LiveTools(job, run_id, lane.ctx), fail_tool)
        result = LiveAgentRun(job=job, run_id=run_id, store=lane.store, ctx=lane.ctx, model=ports, tell=ports, worker_id=worker_id, tools=tools, tell_secured=bool(job.get("tell_secured", 1))).run()
    except Exception as exc:                       # never let one invoice stop the queue
        tb = traceback.format_exc()[-1200:]
        lane.store.emit(run_id, "job_failed", f"Run failed: {type(exc).__name__}", {"error": {"type": type(exc).__name__, "message": str(exc)[:300]}}, provenance="runtime", status="failed")
        result = RunResult("FAILED", f"run failed: {type(exc).__name__}", {"status": "FAILED", "reason": "exception", "traceback_tail": tb}, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    lane.store.finish_run(run_id, "failed" if result.job_status == "FAILED" else "completed", result.outcome, result.error)
    lane.conn.execute("UPDATE jobs SET status=?, progress=100, latest_event=?, error=?, updated_at=? WHERE id=?",
                      (result.job_status, result.latest_event, result.error, S.now(), job_id))
    return result


def process_review(lane, ports, req, worker_id):
    """Apply one human reviewer decision (see LiveAgentRun.apply_reviewer_decision). Payments stay idempotent if this is retried."""
    from live.agent import LiveAgentRun, RunResult
    job = lane.job(req["job_id"])
    run_id = job.get("run_id")
    try:
        run = LiveAgentRun(job=job, run_id=run_id, store=lane.store, ctx=lane.ctx, model=ports, tell=ports, worker_id=worker_id,
                           tell_secured=bool(job.get("tell_secured", 1)))
        result = run.apply_reviewer_decision(req["decision"], req["reviewer_id"], req.get("note"))
    except Exception as exc:
        lane.store.emit(run_id, "job_failed", f"Reviewer decision failed: {type(exc).__name__}", {"error": {"type": type(exc).__name__, "message": str(exc)[:300]}},
                        provenance="runtime", status="failed")
        result = RunResult("NEEDS_DOCUMENT_REVIEW", f"reviewer decision failed: {type(exc).__name__}", {"status": "NEEDS_DOCUMENT_REVIEW"}, error=type(exc).__name__)
    t = S.now()
    lane.conn.execute("UPDATE jobs SET status=?, progress=100, latest_event=?, error=?, updated_at=? WHERE id=?", (result.job_status, result.latest_event, result.error, t, req["job_id"]))
    lane.conn.execute("UPDATE review_queue SET state='done', done_at=?, result=? WHERE id=?",
                      (t, json.dumps({"status": result.job_status, "message": result.latest_event}), req["id"]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-dir", action="append", required=True)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--no-model", action="store_true", help="(tests only) do not load the model")
    ap.add_argument("--test-double", action="store_true", help="(tests only) use scripted stand-ins for the LLM and probe; identifiers publish test_double=true")
    ap.add_argument("--test-score", type=float, default=None, help="(tests only, requires --test-double) constant probe score")
    ap.add_argument("--test-fail-tool", default=None, help="(tests only, requires --test-double) make one tool raise a controlled fault")
    ap.add_argument("--activation-artifacts", action="store_true", help="persist each captured activation vector under <runtime>/live/activations")
    args = ap.parse_args()
    if (args.test_fail_tool or args.test_score is not None) and not args.test_double:
        ap.error("--test-fail-tool/--test-score require --test-double")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    worker_id = f"worker-{os.getpid()}"
    lanes = [Lane(d) for d in args.runtime_dir]
    ports = None
    if args.test_double:
        from live.testing import ScriptedPorts
        ports = ScriptedPorts(score_fn=(lambda _t: args.test_score)) if args.test_score is not None else ScriptedPorts()
        ports.load_seconds = 0
        print(f"[{worker_id}] TEST DOUBLE in use (no model, no probe)", flush=True)
    elif not args.no_model:
        from live.model import RealPorts
        print(f"[{worker_id}] loading model + frozen adapter + frozen probe ...", flush=True)
        ports = RealPorts(activation_root=(lanes[0].dir / "live" / "activations") if args.activation_artifacts else None)
        print(f"[{worker_id}] loaded in {ports.load_seconds}s: {json.dumps({k: ports.identifiers()[k] for k in ('model_revision', 'adapter_weights_sha256', 'probe_weights_sha256', 'layer')})}", flush=True)
    for lane in lanes:
        lane.recover_reviews()
        n = lane.recover_stale()
        if ports:
            from live.agent import OPERATIONAL_THRESHOLD
            lane.store.set_info("identifiers", {**ports.identifiers(), "operational_threshold": OPERATIONAL_THRESHOLD})
        lane.store.set_info("mode", "live")
        lane.store.set_info("worker", {"worker_id": worker_id, "pid": os.getpid(), "state": "idle", "started_at": S.now(), "requeued": n})
    print(f"[{worker_id}] serving {[str(l.dir) for l in lanes]}", flush=True)
    last_hb = 0.0
    while not STOP:
        did = False
        for lane in lanes:
            lane.handle_reset()
            rq = lane.claim_review()                 # human decisions first: they close loops that are already waiting
            if rq:
                did = True
                res = process_review(lane, ports, rq, worker_id)
                print(f"[{worker_id}] review {rq['decision']} {rq['job_id'][:8]} -> {res.job_status}", flush=True)
            jid = lane.claim()
            if jid:
                did = True
                lane.store.set_info("worker", {"worker_id": worker_id, "pid": os.getpid(), "state": "processing", "current_job": jid})
                res = process(lane, ports, jid, worker_id, args.test_fail_tool)
                print(f"[{worker_id}] {jid[:8]} -> {res.job_status}: {res.latest_event}", flush=True)
                lane.store.set_info("worker", {"worker_id": worker_id, "pid": os.getpid(), "state": "idle"})
            if STOP:
                break
        if time.time() - last_hb > 2.0:
            for lane in lanes:
                cur, _ = lane.store.get_info("worker")
                lane.store.set_info("worker", {**(cur or {}), "worker_id": worker_id, "pid": os.getpid()})
            last_hb = time.time()
        if not did:
            time.sleep(args.poll)
    for lane in lanes:
        lane.store.set_info("worker", {"worker_id": worker_id, "pid": os.getpid(), "state": "stopped"})
        lane.ctx.close()
    print(f"[{worker_id}] stopped", flush=True)


if __name__ == "__main__":
    main()
