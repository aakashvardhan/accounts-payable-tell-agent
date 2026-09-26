"""Local invoice intake for the Tell demo: PDF validation, embedded-text extraction, job registry, inbox scan.

CPU-only, stdlib only (plus the system `pdftotext` binary from poppler for text extraction).
No model, no CUDA, no network, no OCR. Nothing here scores, routes, validates or pays an invoice:
the terminal state of a new invoice in this slice is READY_FOR_PROCESSING (or a failure/attention state).

Sources are pluggable: `InvoiceSource.discover()` yields (display_name, bytes) candidates; `LocalFolderInvoiceSource`
reads a dedicated demo inbox directory (never a real mailbox) and never deletes or modifies source files.
"""
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import clarification as K
import extraction as X
import ocr as O

HERE = Path(__file__).resolve().parent
ACTIVE = ("UPLOADING", "VALIDATING", "EXTRACTING_EMBEDDED_TEXT", "OCR_QUEUED", "OCR_RUNNING", "OCR_COMPLETE", "EXTRACTION_COMPLETE")
TERMINAL = ("READY_FOR_PROCESSING", "AWAITING_VENDOR_CLARIFICATION", "NEEDS_DOCUMENT_REVIEW", "UNREADABLE_DOCUMENT", "DUPLICATE", "FAILED")
STAGES = ACTIVE + TERMINAL
PROGRESS = {"UPLOADING": 5, "VALIDATING": 15, "EXTRACTING_EMBEDDED_TEXT": 30, "OCR_QUEUED": 40, "OCR_RUNNING": 55, "OCR_COMPLETE": 82,
            "EXTRACTION_COMPLETE": 92, **{t: 100 for t in TERMINAL}}
LEGACY_STATUS = {"NEEDS_OCR": "NEEDS_DOCUMENT_REVIEW", "NEEDS_REVIEW": "NEEDS_DOCUMENT_REVIEW", "EXTRACTING": "FAILED", "READY_FOR_PROCESSING": "READY_FOR_PROCESSING"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
UPLOAD_STALE_S = 120

# Intake event schema (every timeline entry validates against this)
EVENT_SCHEMA = "tell.intake_event/1.0"
EVENT_KINDS = ("upload", "validation", "duplicate", "embedded_text", "ocr_queue", "ocr_render", "ocr_page", "ocr_summary", "extraction",
               "required_field_check", "vendor_lookup", "agent_action", "email_prepared", "status", "error")
EVENT_ACTORS = ("application", "ocr_engine", "simulated_agent", "trusted_vendor_record")
EVENT_PROVENANCE = ("APPLICATION", "EMBEDDED_TEXT", "OCR", "TRUSTED_VENDOR_RECORD", "SIMULATED_AGENT_ACTION")


class EventSchemaError(ValueError):
    pass


def validate_event(e, seq=None):
    """Raise EventSchemaError unless `e` is a well-formed tell.intake_event/1.0 entry."""
    def need(c, m):
        if not c:
            raise EventSchemaError(m)
    need(isinstance(e, dict), "event must be an object")
    need(e.get("schema") == EVENT_SCHEMA, "schema")
    need(isinstance(e.get("seq"), int) and (seq is None or e["seq"] == seq), "seq")
    need(isinstance(e.get("at"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", e["at"]), "at")
    need(e.get("kind") in EVENT_KINDS, "kind")
    need(e.get("stage") in STAGES, "stage")
    need(e.get("actor") in EVENT_ACTORS, "actor")
    need(e.get("provenance") in EVENT_PROVENANCE, "provenance")
    need(isinstance(e.get("message"), str) and e["message"], "message")
    need(e.get("data") is None or isinstance(e["data"], dict), "data")
    if e["kind"] == "agent_action":
        need(e["provenance"] == "SIMULATED_AGENT_ACTION" and e["actor"] == "simulated_agent", "agent actions must be labelled SIMULATED_AGENT_ACTION")
    return e


class IntakeError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class IntakeConfig:
    root: Path                              # runtime directory (uploads/, jobs.sqlite, inbox/, ocr_tmp/)
    inbox_dir: Path = None
    max_bytes: int = 10 * 1024 * 1024       # per file
    max_files: int = 20                     # per batch
    max_queue: int = 200                    # total jobs kept
    max_pages: int = 50                     # embedded-text pages read
    max_text_chars: int = 20000
    extract_timeout: float = 15.0
    stage_delay: float = 0.7                # visible pacing between stages (0 in tests)
    workers: int = 3
    pdftotext: tuple = ("pdftotext",)
    vendor_master: Path = HERE / "data" / "vendor_master_demo.json"   # trusted, application-controlled (synthetic fixture)
    tesseract: str = None                   # explicit engine path (else discovered); tests point this at a fake engine
    tessdata: str = None
    ocr: object = None                      # ocr.OcrLimits

    def __post_init__(self):
        self.root = Path(self.root)
        self.inbox_dir = Path(self.inbox_dir) if self.inbox_dir else self.root / "inbox"
        self.vendor_master = Path(self.vendor_master)
        self.ocr = self.ocr or O.OcrLimits()

    def limits(self):
        return {"max_bytes": self.max_bytes, "max_files_per_batch": self.max_files, "max_queue": self.max_queue,
                "max_ocr_pages": self.ocr.max_pages, "ocr_dpi": self.ocr.dpi, "extract_timeout_s": self.extract_timeout, "mime": "application/pdf"}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- sanitizing / validation
def safe_display_name(name):
    """Display-only name: basename, no separators/controls/markup characters, bounded. Never used as a path."""
    name = str(name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w .()\-]", "_", name, flags=re.UNICODE).strip(" .")
    name = re.sub(r"_{2,}", "_", name)
    return (name[:80] or "invoice.pdf")


def sanitize_text(text, limit):
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n"))
    text = "".join(ch for ch in text if ch in "\n\t" or (ord(ch) >= 32 and ord(ch) != 127 and unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co")))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text[:limit], len(text) > limit)


def validate_pdf_bytes(data, max_bytes):
    """Returns (ok, reason, warnings). Strict header check at offset 0; no parsing, nothing executed."""
    if len(data) == 0:
        return False, "empty file", []
    if len(data) > max_bytes:
        return False, f"file exceeds {max_bytes // (1024 * 1024)} MB limit", []
    if not data.startswith(b"%PDF-"):
        return False, "not a PDF (missing %PDF- signature)", []
    warnings = []
    if b"%%EOF" not in data[-2048:]:
        warnings.append("PDF is missing an end-of-file marker; it may be truncated")
    return True, "", warnings


# ---------------------------------------------------------------- embedded text
def extract_embedded(path, cfg):
    """Run poppler pdftotext. Returns dict(status=OK|ENCRYPTED|FAILED, raw=str with form-feed page breaks, error)."""
    cmd = list(cfg.pdftotext) + ["-layout", "-enc", "UTF-8", "-q", "-l", str(cfg.max_pages), str(path), "-"]
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=cfg.extract_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            return {"status": "FAILED", "error": f"extraction timed out after {cfg.extract_timeout:g}s", "code": "TIMEOUT"}
    except FileNotFoundError:
        return {"status": "FAILED", "error": "PDF text extractor (pdftotext) is not installed", "code": "TOOL_MISSING"}
    err = err.decode("utf-8", "replace")
    if proc.returncode != 0:
        if "password" in err.lower() or "encrypt" in err.lower():
            return {"status": "ENCRYPTED", "error": "PDF is encrypted; its content cannot be read", "raw": ""}
        return {"status": "FAILED", "error": "PDF could not be parsed (malformed or unsupported)", "code": "PDF_UNPARSEABLE"}
    return {"status": "OK", "raw": out.decode("utf-8", "replace"), "error": None}


# ---------------------------------------------------------------- sources
class InvoiceSource:
    name = "abstract"

    def discover(self):  # -> iterator of (display_name, size, reader) ; reader() -> bytes
        raise NotImplementedError


class LocalFolderInvoiceSource(InvoiceSource):
    """Dedicated demo inbox directory (top level, regular *.pdf files only, no symlinks). Read-only."""
    name = "local_inbox"

    def __init__(self, directory, max_bytes):
        self.dir, self.max_bytes = Path(directory), max_bytes

    def discover(self):
        try:
            entries = sorted(os.scandir(self.dir), key=lambda e: e.name)
        except FileNotFoundError:
            return
        for e in entries:
            if not e.name.lower().endswith(".pdf") or e.is_symlink() or not e.is_file(follow_symlinks=False):
                continue
            size = e.stat(follow_symlinks=False).st_size
            yield e.name, size, (lambda p=e.path, n=size: self._read(p, n))

    def _read(self, path, size):
        if size > self.max_bytes:
            return None
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as f:
            return f.read(self.max_bytes + 1)


# ---------------------------------------------------------------- registry + service
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, sha256 TEXT, display_name TEXT NOT NULL, stored_name TEXT,
  size INTEGER, source TEXT NOT NULL, mime TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, status TEXT NOT NULL,
  progress INTEGER NOT NULL, latest_event TEXT, events TEXT NOT NULL, extraction TEXT, error TEXT, duplicate_of TEXT, claimed INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS jobs_sha ON jobs(sha256);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL, created_at TEXT NOT NULL, send_status TEXT NOT NULL, record TEXT NOT NULL);
"""


class IntakeService:
    def __init__(self, cfg: IntakeConfig, source: InvoiceSource = None, agent=None):
        self.cfg = cfg
        cfg.root.mkdir(parents=True, exist_ok=True)
        self.uploads = cfg.root / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.ocr_tmp = cfg.root / "ocr_tmp"
        self.ocr_tmp.mkdir(exist_ok=True)
        self._clean_ocr_tmp()
        cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.source = source or LocalFolderInvoiceSource(cfg.inbox_dir, cfg.max_bytes)
        self.agent = agent or K.simulated_agent_action   # deterministic replay generator; a live agent could replace it later
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(cfg.root / "jobs.sqlite"), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="intake")
        self.scans = {}
        self.closed = False
        with self.lock:
            for old, new in LEGACY_STATUS.items():
                if old != new:
                    self.db.execute("UPDATE jobs SET status=? WHERE status=?", (new, old))
            marks = ",".join("?" * len(ACTIVE))
            for r in self.db.execute(f"SELECT id FROM jobs WHERE status IN ({marks})", ACTIVE).fetchall():   # jobs a previous process left mid-flight
                self._event(r["id"], "error", "Interrupted by a server restart", status="FAILED", error="interrupted by restart")

    # -- housekeeping
    def _clean_ocr_tmp(self):
        for p in self.ocr_tmp.iterdir():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() and not p.is_symlink() else None

    # -- low-level event log
    def _job_events(self, job_id):
        r = self.db.execute("SELECT events FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(r["events"]) if r else None

    def _event(self, job_id, kind, message, *, status=None, actor="application", provenance="APPLICATION", data=None, progress=None, **cols):
        """Append a schema-valid event; optionally move the job to `status` and set columns (extraction is a JSON column)."""
        with self.lock:
            ev = self._job_events(job_id)
            if ev is None:
                return
            cur = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()["status"]
            st = status or cur
            e = validate_event({"schema": EVENT_SCHEMA, "seq": len(ev), "at": now(), "kind": kind, "stage": st, "actor": actor,
                                "provenance": provenance, "message": message, "data": data}, seq=len(ev))
            ev.append(e)
            sets = {"status": st, "progress": PROGRESS[st] if progress is None else progress, "latest_event": message,
                    "events": json.dumps(ev), "updated_at": now()}
            for k, v in cols.items():
                sets[k] = json.dumps(v) if k == "extraction" else v
            self.db.execute("UPDATE jobs SET " + ", ".join(f"{k}=?" for k in sets) + " WHERE id=?", (*sets.values(), job_id))

    def _patch_extraction(self, job_id, **patch):
        with self.lock:
            r = self.db.execute("SELECT extraction FROM jobs WHERE id=?", (job_id,)).fetchone()
            if r is None:
                return
            ex = json.loads(r["extraction"]) if r["extraction"] else {}
            ex.update(patch)
            self.db.execute("UPDATE jobs SET extraction=? WHERE id=?", (json.dumps(ex), job_id))

    def _new_job(self, display_name, source, size, mime, status="UPLOADING", message="Waiting for file content", error=None):
        jid = uuid.uuid4().hex
        t = now()
        with self.lock:
            total = self.db.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]
            if total >= self.cfg.max_queue:
                raise IntakeError(f"queue is full ({self.cfg.max_queue} jobs); reset the demo queue first", 429)
            ev = [validate_event({"schema": EVENT_SCHEMA, "seq": 0, "at": t, "kind": "upload" if status == "UPLOADING" else "error", "stage": status,
                                  "actor": "application", "provenance": "APPLICATION", "message": message, "data": None}, seq=0)]
            self.db.execute("INSERT INTO jobs(id,display_name,size,source,mime,created_at,updated_at,status,progress,latest_event,events,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (jid, safe_display_name(display_name), size, source, mime, t, t, status, PROGRESS[status], message, json.dumps(ev), error))
        return jid

    def _fail(self, jid, reason, stored=None, code=None):
        self._event(jid, "error", reason, status="FAILED", error=reason, data={"failure_code": code} if code else None)
        self._delete_file(stored)

    def _delete_file(self, stored_name):
        if stored_name and re.fullmatch(r"[0-9a-f]{32}\.pdf", stored_name):
            try:
                (self.uploads / stored_name).unlink()
            except FileNotFoundError:
                pass

    def _pause(self):
        if self.cfg.stage_delay:
            time.sleep(self.cfg.stage_delay)

    # -- queries
    def _row(self, r, detail=False):
        ex = json.loads(r["extraction"]) if r["extraction"] else None
        d = {"job_id": r["id"], "display_name": r["display_name"], "size": r["size"], "source": r["source"], "sha256": r["sha256"],
             "created_at": r["created_at"], "updated_at": r["updated_at"], "status": r["status"], "progress": r["progress"],
             "latest_event": r["latest_event"], "error": r["error"], "duplicate_of": r["duplicate_of"], "provenance": "uploaded_pdf"}
        f = (ex or {}).get("fields", {})
        d["summary"] = {k: (f[k]["value"] if k in f and f[k].get("found") else None) for k in ("invoice_number", "supplier_name", "amount", "currency")}
        d["payment"] = ({"hold": True, "eligible": False, "reason": {"AWAITING_VENDOR_CLARIFICATION": "waiting for vendor clarification",
                                                                  "NEEDS_DOCUMENT_REVIEW": "document needs human review",
                                                                  "UNREADABLE_DOCUMENT": "document is unreadable"}[r["status"]]}
                        if r["status"] in ("AWAITING_VENDOR_CLARIFICATION", "NEEDS_DOCUMENT_REVIEW", "UNREADABLE_DOCUMENT") else None)
        if detail:
            d["events"] = json.loads(r["events"]); d["extraction"] = ex
            ob = self.db.execute("SELECT record FROM outbox WHERE job_id=? ORDER BY created_at DESC LIMIT 1", (r["id"],)).fetchone()
            d["outbox"] = json.loads(ob["record"]) if ob else None
        return d

    def _expire_stale(self):
        cutoff = datetime.fromtimestamp(time.time() - UPLOAD_STALE_S, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in self.db.execute("SELECT id FROM jobs WHERE status='UPLOADING' AND updated_at < ?", (cutoff,)).fetchall():
            self._fail(r["id"], "upload was not received")

    def list_jobs(self):
        with self.lock:
            self._expire_stale()
            return [self._row(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY seq DESC").fetchall()]

    def get_job(self, jid):
        if not JOB_ID_RE.match(jid or ""):
            raise IntakeError("unknown job", 404)
        with self.lock:
            r = self.db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
            if not r:
                raise IntakeError("unknown job", 404)
            return self._row(r, detail=True)

    def status(self):
        jobs = self.list_jobs()
        c = {s: 0 for s in STAGES}
        for j in jobs:
            c[j["status"]] += 1
        with self.lock:
            last = self.db.execute("SELECT value FROM meta WHERE key='last_scan'").fetchone()
        engine = O.find_engine(self.cfg.tesseract)
        return {"enabled": True, "limits": self.cfg.limits(), "counts": c, "total": len(jobs),
                "waiting": c["UPLOADING"] + c["VALIDATING"],
                "processing": c["EXTRACTING_EMBEDDED_TEXT"] + c["OCR_QUEUED"] + c["OCR_RUNNING"] + c["OCR_COMPLETE"] + c["EXTRACTION_COMPLETE"],
                "ready": c["READY_FOR_PROCESSING"], "awaiting_vendor": c["AWAITING_VENDOR_CLARIFICATION"],
                "attention": c["NEEDS_DOCUMENT_REVIEW"] + c["UNREADABLE_DOCUMENT"], "failed": c["FAILED"], "duplicates": c["DUPLICATE"],
                "ocr_available": bool(engine), "last_scan": json.loads(last["value"]) if last else None,
                "inbox": "demo inbox (local folder)", "source": self.source.name}

    # -- batch upload
    def create_batch(self, manifest):
        if not isinstance(manifest, list) or not manifest:
            raise IntakeError("manifest must be a non-empty list of files")
        if len(manifest) > self.cfg.max_files:
            raise IntakeError(f"too many files: {len(manifest)} > {self.cfg.max_files} per batch", 413)
        out = []
        for m in manifest:
            if not isinstance(m, dict):
                raise IntakeError("invalid manifest entry")
            name, size, mime = m.get("name"), m.get("size"), (m.get("type") or "")
            if not isinstance(name, str) or not isinstance(size, int) or size < 0:
                raise IntakeError("manifest entries need a string name and integer size")
            lim = f"file exceeds {self.cfg.max_bytes // (1024 * 1024)} MB limit"
            if size > self.cfg.max_bytes:
                jid = self._new_job(name, "manual_upload", size, mime, "FAILED", lim, lim)
            elif mime != "application/pdf":
                jid = self._new_job(name, "manual_upload", size, mime, "FAILED", "unsupported MIME type (application/pdf required)", "unsupported MIME type (application/pdf required)")
            else:
                jid = self._new_job(name, "manual_upload", size, mime)
            out.append(self.get_job(jid))
        return out

    def receive_content(self, jid, content_type, data):
        job = self.get_job(jid)
        if job["status"] != "UPLOADING":
            raise IntakeError("job is not awaiting content", 409)
        if (content_type or "").split(";")[0].strip().lower() != "application/pdf":
            self._fail(jid, "unsupported MIME type (application/pdf required)")
            raise IntakeError("unsupported MIME type", 415)
        self._ingest(jid, data)
        return self.get_job(jid)

    def _ingest(self, jid, data):
        ok, reason, warnings = validate_pdf_bytes(data, self.cfg.max_bytes)
        sha = hashlib.sha256(data).hexdigest()
        stored = None
        with self.lock:
            self.db.execute("UPDATE jobs SET sha256=?, size=? WHERE id=?", (sha, len(data), jid))
        self._event(jid, "upload", f"PDF uploaded ({len(data) / 1024:.1f} KB, sha256 {sha[:12]}…)", data={"size": len(data), "sha256": sha})
        if ok:
            stored = uuid.uuid4().hex + ".pdf"
            tmp = self.uploads / (stored + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, self.uploads / stored)
            with self.lock:
                self.db.execute("UPDATE jobs SET stored_name=? WHERE id=?", (stored, jid))
        self.pool.submit(self._process, jid, ok, reason, warnings, stored)

    # -- the pipeline
    def _process(self, jid, ok, reason, warnings, stored):
        try:
            self._event(jid, "validation", "Checking PDF signature, size and type", status="VALIDATING")
            self._pause()
            if not ok:
                return self._fail(jid, reason, stored, "INVALID_PDF")
            with self.lock:   # duplicate check + claim is atomic so identical concurrent uploads yield one original
                r = self.db.execute("SELECT sha256 FROM jobs WHERE id=?", (jid,)).fetchone()
                dup = self.db.execute("SELECT id FROM jobs WHERE sha256=? AND id!=? AND claimed=1 AND status NOT IN ('FAILED','DUPLICATE')", (r["sha256"], jid)).fetchone()
                if dup:
                    self._event(jid, "duplicate", f"Identical content already in the queue (job {dup['id'][:8]})", status="DUPLICATE", duplicate_of=dup["id"])
                    self._delete_file(stored)
                    return
                self.db.execute("UPDATE jobs SET claimed=1 WHERE id=?", (jid,))
            self._event(jid, "validation", "PDF validated" + (" (" + "; ".join(warnings) + ")" if warnings else ""), data={"warnings": warnings})
            self._pause()
            self._event(jid, "embedded_text", "Extracting embedded text", status="EXTRACTING_EMBEDDED_TEXT", provenance="EMBEDDED_TEXT")
            emb = extract_embedded(self.uploads / stored, self.cfg)
            self._pause()
            if emb["status"] == "FAILED":
                return self._fail(jid, emb["error"], stored, emb.get("code"))
            if emb["status"] == "ENCRYPTED":
                self._patch_extraction(jid, fields={}, warnings=["encrypted PDF"], source=None, decision={"status_hint": "REVIEW", "reasons": [emb["error"]]})
                return self._event(jid, "status", emb["error"] + "; human document review required", status="NEEDS_DOCUMENT_REVIEW", error=emb["error"])
            raw = emb["raw"]
            q = X.embedded_quality(raw)
            self._patch_extraction(jid, embedded_quality=q)
            if q["ok"]:
                pages = X.split_embedded_pages(raw)
                source = "EMBEDDED_TEXT"
                self._event(jid, "embedded_text", f"Embedded text extracted ({q['chars']} characters, {len(pages)} page(s))", provenance="EMBEDDED_TEXT", data=q)
            else:
                self._event(jid, "embedded_text", f"No usable embedded text ({q['reason']}); OCR required", provenance="EMBEDDED_TEXT", data=q)
                pages = self._run_ocr(jid, self.uploads / stored, stored)
                if pages is None:
                    return
                source = "OCR"
            self._extract_and_decide(jid, pages, source, raw_pages=[{"page": p["page"], "text": p["text"]} for p in pages])
        except Exception as e:   # one bad file must never take down the worker or the batch
            self._fail(jid, f"internal error: {type(e).__name__}", stored, "INTERNAL")

    def _run_ocr(self, jid, pdf, stored):
        """Returns [{page, text, line_conf}] or None when the job reached a terminal state (FAILED / UNREADABLE_DOCUMENT)."""
        engine = O.find_engine(self.cfg.tesseract)
        self._event(jid, "ocr_queue", "OCR queued: the document has no usable embedded text", status="OCR_QUEUED", actor="ocr_engine", provenance="OCR")
        if not engine:
            self._fail(jid, "OCR is required but no local OCR engine (Tesseract) is installed", stored, "OCR_ENGINE_UNAVAILABLE")
            return None
        tmp = Path(tempfile.mkdtemp(prefix=f"ocr-{jid[:8]}-", dir=self.ocr_tmp))     # isolated per-job temp dir, always removed
        pages_meta = []
        try:
            limits = self.cfg.ocr

            def on_rendered(total, n, dpi):
                note = f" (only the first {n} of {total} pages are permitted)" if total > n else ""
                self._event(jid, "ocr_render", f"OCR running: rendered {n} page(s) at {dpi} DPI{note}", status="OCR_RUNNING", actor="ocr_engine", provenance="OCR",
                            data={"pages_total": total, "pages_rendered": n, "dpi": dpi})
                self._patch_extraction(jid, ocr={"state": "running", "page_count": total, "pages_processed": n, "dpi": dpi, "pages": []})

            def on_page(res, i, n):
                meta = {"page": res.page, "outcome": res.outcome, "words": res.words, "usable_words": res.usable_words,
                        "mean_confidence": res.mean_confidence, "duration_s": res.duration_s}
                pages_meta.append(meta)
                self._event(jid, "ocr_page", f"OCR page {res.page} ({i}/{n}): {res.usable_words} usable words, mean confidence {res.mean_confidence:.0%}",
                            status="OCR_RUNNING", actor="ocr_engine", provenance="OCR", data=meta, progress=55 + int(25 * i / n))
                cur = self.get_job(jid)["extraction"] or {}
                ocr_state = dict(cur.get("ocr") or {}); ocr_state["pages"] = list(pages_meta)
                self._patch_extraction(jid, ocr=ocr_state)

            try:
                res = O.run_ocr(pdf, tmp, limits, engine, self.cfg.tessdata, on_page=on_page, on_rendered=on_rendered)
            except O.OcrError as e:
                self._fail(jid, f"OCR failed: {e.message}", stored, "OCR_" + e.code)
                return None
            summary = {"engine": res.engine, "version": res.version, "dpi": res.dpi, "page_count": res.page_count, "pages_processed": len(res.pages),
                       "duration_s": res.duration_s, "usable_words": res.usable_words, "mean_confidence": res.mean_confidence,
                       "usable": res.usable, "state": "complete", "pages": pages_meta}
            raw_pages = [{"page": p.page, "text": p.text, "source": "OCR"} for p in res.pages]
            self._patch_extraction(jid, ocr=summary, raw_text=raw_pages)
            self._event(jid, "ocr_summary", f"OCR complete ({res.engine} {res.version}): {len(res.pages)} page(s), {res.usable_words} usable words, "
                        f"mean confidence {res.mean_confidence:.0%}", status="OCR_COMPLETE", actor="ocr_engine", provenance="OCR", data={k: v for k, v in summary.items() if k != "pages"})
            self._pause()
            if not res.usable:
                self._patch_extraction(jid, fields={}, warnings=["OCR produced insufficient usable text"],
                                       decision={"status_hint": "UNREADABLE", "reasons": [f"OCR ran on {len(res.pages)} page(s) and produced only {res.usable_words} usable words"]})
                self._event(jid, "status", f"Unreadable document: OCR ran but produced only {res.usable_words} usable word(s); no business content could be read",
                            status="UNREADABLE_DOCUMENT", actor="ocr_engine", provenance="OCR", error="unreadable document")
                self._delete_file(stored)
                return None
            return [{"page": p.page, "text": p.text, "line_conf": [l["confidence"] for l in p.lines]} for p in res.pages]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _extract_and_decide(self, jid, pages, source, raw_pages):
        fields, warnings = X.extract_fields(pages, source)
        vendor_hint = None
        if fields["currency"]["ambiguous"] and fields["supplier_name"]["found"]:     # a trusted vendor record may confirm a bare symbol; nothing else may
            peek = K.resolve_vendor(self.cfg.vendor_master, fields["supplier_name"]["value"])
            if peek["status"] != K.LOOKUP_FAILED and peek.get("canonical_currency"):
                vendor_hint = {"canonical_currency": peek["canonical_currency"]}
                fields, warnings = X.extract_fields(pages, source, vendor_hint)
        text_disp, truncated = sanitize_text("\n\n".join(p["text"] for p in raw_pages), self.cfg.max_text_chars)
        decision = X.assess(fields, source)
        found = [k for k in X.FIELD_KEYS if fields[k]["found"]]
        self._patch_extraction(jid, fields=fields, warnings=warnings, source=source, text=text_disp, text_truncated=truncated, decision=decision,
                               raw_text=[{"page": p["page"], "text": sanitize_text(p["text"], self.cfg.max_text_chars)[0], "source": source} for p in raw_pages])
        self._event(jid, "extraction", f"Fields extracted from {'embedded text' if source == 'EMBEDDED_TEXT' else 'OCR text'}: document type "
                    f"{fields['document_type']['value'] or 'unknown'}; found {', '.join(found) if found else 'nothing'}",
                    status="EXTRACTION_COMPLETE", provenance=source, data={"found": found, "warnings": warnings})
        self._pause()
        hint = decision["status_hint"]
        if hint == "REVIEW":
            return self._event(jid, "required_field_check", "Assessment: document needs human review — " + "; ".join(decision["reasons"]),
                               status="NEEDS_DOCUMENT_REVIEW", data=decision, error="; ".join(decision["reasons"]))
        if hint == "READY":
            self._event(jid, "required_field_check", "Required-field check: all required fields present", data=decision)
            return self._event(jid, "status", "Extraction complete: ready for agent processing", status="READY_FOR_PROCESSING")
        self._clarify(jid, fields, decision)

    def _clarify(self, jid, fields, decision):
        """Missing-field workflow. The application resolves the recipient and owns the template; the agent only names fields."""
        missing = decision["missing_required_fields"]
        result = K.missing_field_result(jid, missing)
        self._patch_extraction(jid, missing_field_result=result)
        self._event(jid, "required_field_check", f"Required-field check: {', '.join(missing)} missing — payment on hold", data=result)
        self._pause()
        lookup = K.resolve_vendor(self.cfg.vendor_master, fields["supplier_name"]["value"])
        public_lookup = {k: lookup[k] for k in ("status", "vendor_id", "vendor_name", "reason", "source", "label")}
        self._patch_extraction(jid, vendor_lookup=public_lookup)
        self._event(jid, "vendor_lookup", f"Trusted vendor lookup: {lookup['status']} — {lookup['reason']}", actor="trusted_vendor_record",
                    provenance="TRUSTED_VENDOR_RECORD", data=public_lookup)
        self._pause()
        if lookup["status"] != K.CONTACT_FOUND:
            return self._event(jid, "status", f"No clarification email drafted ({lookup['status']}); human document review required",
                               status="NEEDS_DOCUMENT_REVIEW", error=lookup["reason"])
        envelope = self.agent(result, lookup["vendor_id"])
        self._patch_extraction(jid, agent_action=envelope)
        self._event(jid, "agent_action", f"Clarification requested: {envelope['action'].get('action')} for {', '.join(envelope['action'].get('missing_fields', []))} "
                    f"({envelope['origin']})", actor="simulated_agent", provenance="SIMULATED_AGENT_ACTION", data=envelope)
        try:
            req_fields = K.validate_action(envelope["action"], invoice_id=jid, resolved_vendor_id=lookup["vendor_id"], missing_required=missing)
            record = K.build_outbox_record(invoice_id=jid, lookup=lookup, action_envelope=envelope, missing_fields=req_fields,
                                           invoice_number=fields["invoice_number"]["value"], amount=fields["amount"]["value"])
        except K.ClarificationError as e:
            return self._event(jid, "status", f"Clarification action rejected by the application ({e}); human document review required",
                               status="NEEDS_DOCUMENT_REVIEW", error=str(e))
        self._pause()
        oid = "out-" + uuid.uuid4().hex[:12]
        with self.lock:
            self.db.execute("INSERT INTO outbox(id,job_id,created_at,send_status,record) VALUES(?,?,?,?,?)", (oid, jid, record["created_at"], record["send_status"], json.dumps(dict(record, outbox_id=oid))))
        self._event(jid, "email_prepared", "Simulated clarification email prepared — NOT SENT", data={"outbox_id": oid, "send_status": record["send_status"],
                    "recipient_source": record["recipient_source"], "requested_fields": record["requested_fields"]})
        self._event(jid, "status", "Awaiting vendor clarification; payment remains on hold", status="AWAITING_VENDOR_CLARIFICATION")

    # -- scan
    def scan(self):
        with self.lock:
            for s in self.scans.values():
                if s["state"] == "running":
                    return dict(s)
            sid = uuid.uuid4().hex
            s = {"scan_id": sid, "state": "running", "started_at": now(), "completed_at": None, "discovered": 0, "imported": 0,
                 "duplicates": 0, "rejected": 0, "source": self.source.name, "message": "Scanning demo inbox"}
            self.scans[sid] = s
            snapshot = dict(s)
        threading.Thread(target=self._scan, args=(sid,), daemon=True, name="intake-scan").start()
        return snapshot

    def get_scan(self, sid):
        with self.lock:
            if sid not in self.scans:
                raise IntakeError("unknown scan", 404)
            return dict(self.scans[sid])

    def _scan(self, sid):
        s = self.scans[sid]
        try:
            self._pause()
            for name, size, reader in self.source.discover():
                with self.lock:
                    s["discovered"] += 1
                data = reader()
                if data is None:
                    if not self._seen_by_name_size(name, size):
                        self._reject(name, size, f"file exceeds {self.cfg.max_bytes // (1024 * 1024)} MB limit")
                        s["rejected"] += 1
                    else:
                        s["duplicates"] += 1
                    continue
                sha = hashlib.sha256(data).hexdigest()
                with self.lock:
                    seen = self.db.execute("SELECT 1 FROM jobs WHERE sha256=? AND (claimed=1 OR source=?) AND status!='DUPLICATE' LIMIT 1", (sha, self.source.name)).fetchone()
                if seen:
                    s["duplicates"] += 1
                    continue
                ok, reason, _ = validate_pdf_bytes(data, self.cfg.max_bytes)
                jid = self._new_job(name, self.source.name, len(data), "application/pdf", "UPLOADING", "Discovered in demo inbox")
                self._ingest(jid, data)
                if ok:
                    s["imported"] += 1
                else:
                    s["rejected"] += 1
            s["message"] = "No new invoices found" if s["imported"] == 0 and s["rejected"] == 0 else f"{s['imported']} imported"
        except IntakeError as e:
            s["message"] = str(e); s["error"] = str(e)
        except Exception as e:
            s["message"] = f"scan failed: {type(e).__name__}"; s["error"] = s["message"]
        finally:
            s["completed_at"] = now()
            with self.lock:
                if not self.closed:
                    self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_scan',?)", (json.dumps({k: s[k] for k in ("scan_id", "completed_at", "discovered", "imported", "duplicates", "rejected", "message")}),))
                s["state"] = "complete"

    def _seen_by_name_size(self, name, size):
        with self.lock:
            return self.db.execute("SELECT 1 FROM jobs WHERE source=? AND display_name=? AND size=? AND status='FAILED'", (self.source.name, safe_display_name(name), size)).fetchone() is not None

    def _reject(self, name, size, reason):
        self._new_job(name, self.source.name, size, "application/pdf", "FAILED", reason, reason)

    # -- lifecycle
    def wait_idle(self, timeout=10.0, include_uploading=True):
        """Block until no job is in flight and no scan is running (UPLOADING jobs await the client, so close() ignores them)."""
        active = ACTIVE if include_uploading else tuple(a for a in ACTIVE if a != "UPLOADING")
        marks = ",".join("?" * len(active))
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                busy = self.db.execute(f"SELECT COUNT(*) c FROM jobs WHERE status IN ({marks})", active).fetchone()["c"]
                scanning = any(s["state"] == "running" for s in self.scans.values())
            if not busy and not scanning:
                return True
            time.sleep(0.02)
        return False

    def reset(self):
        """Delete queue state, outbox and temporary uploads. The inbox folder and its files are never touched."""
        with self.lock:
            self.db.execute("DELETE FROM jobs"); self.db.execute("DELETE FROM outbox"); self.db.execute("DELETE FROM meta")
            self.db.execute("DELETE FROM sqlite_sequence WHERE name='jobs'")
            self.scans.clear()
            removed = 0
            for p in self.uploads.iterdir():
                if p.is_file() and not p.is_symlink():
                    p.unlink(); removed += 1
            self._clean_ocr_tmp()
        return {"reset": True, "uploads_removed": removed}

    def close(self):
        self.wait_idle(5, include_uploading=False)
        with self.lock:
            self.closed = True
            self.db.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
