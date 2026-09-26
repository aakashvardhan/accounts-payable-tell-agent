"""Server-side replay adapter (stdlib). Merges the two PERSISTED event stores into one ordered list in the runtime event contract:

  * intake events  (jobs.events JSON, written by intake.py during upload/extraction)      -> mapped onto the contract's extraction types
  * run events     (run_events table, written by the live worker as the agent executes)   -> passed through unchanged

The adapter is a pure read-time mapping: it never creates, orders by heuristic, or fabricates an event. Every returned event carries
`source` naming the store it came from, so the UI can only ever show what exists in the backend.
"""
INTAKE_TYPES = {
    "upload": ("intake_received", "intake"), "validation": ("intake_validated", "intake"), "duplicate": ("intake_validated", "intake"),
    "embedded_text": ("extraction_progress", "extraction"), "ocr_queue": ("extraction_progress", "extraction"), "ocr_render": ("extraction_progress", "extraction"),
    "ocr_page": ("extraction_progress", "extraction"), "ocr_summary": ("extraction_progress", "extraction"), "extraction": ("extraction_completed", "extraction"),
    "required_field_check": ("extraction_progress", "extraction"), "vendor_lookup": ("extraction_progress", "extraction"), "agent_action": ("extraction_progress", "extraction"),
    "email_prepared": ("extraction_progress", "extraction"), "status": ("job_queued", "intake"), "error": ("job_failed", "intake"),
}


def unify_intake(job_id, run_id, events):
    out, started = [], False
    for e in events:
        kind = e.get("kind", "status")
        type_, prov = INTAKE_TYPES.get(kind, ("extraction_progress", "extraction"))
        if kind == "embedded_text" and not started:
            type_, started = "extraction_started", True
        status = "failed" if kind == "error" else ("blocked" if kind == "duplicate" else "ok")
        payload = dict(e.get("data") or {})
        payload["stage"] = e.get("stage")
        out.append({"event_id": f"intake-{job_id[:8]}-{e['seq']}", "run_id": run_id, "ts": e["at"], "type": type_, "title": e["message"], "payload": payload,
                    "provenance": prov, "status": status, "duration_ms": None, "source": "jobs.events"})
    return out


def build_replay(job_id, run_id, intake_events, run_events, run):
    """-> {run_id, events:[... + sequence], run}. `sequence` is the position in the merged, persisted order."""
    events = unify_intake(job_id, run_id, intake_events) + [dict(e, source="run_events") for e in run_events]
    for i, e in enumerate(events):
        e["sequence"] = i
    return {"run_id": run_id, "events": events, "run": run}
