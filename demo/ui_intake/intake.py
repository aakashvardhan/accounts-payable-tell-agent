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
import signal
import sqlite3
import subprocess
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("UPLOADING", "VALIDATING", "EXTRACTING", "READY_FOR_PROCESSING", "NEEDS_OCR", "NEEDS_REVIEW", "DUPLICATE", "FAILED")
ACTIVE = ("UPLOADING", "VALIDATING", "EXTRACTING")
TERMINAL = tuple(s for s in STAGES if s not in ACTIVE)
PROGRESS = {"UPLOADING": 10, "VALIDATING": 35, "EXTRACTING": 65, "READY_FOR_PROCESSING": 100, "NEEDS_OCR": 100,
            "NEEDS_REVIEW": 100, "DUPLICATE": 100, "FAILED": 100}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CURRENCIES = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "JPY", "CZK", "PLN", "SEK", "NOK", "DKK", "INR", "CNY"}
SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
UPLOAD_STALE_S = 120


class IntakeError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class IntakeConfig:
    root: Path                              # runtime directory (uploads/, jobs.sqlite, inbox/)
    inbox_dir: Path = None
    max_bytes: int = 10 * 1024 * 1024       # per file
    max_files: int = 20                     # per batch
    max_queue: int = 200                    # total jobs kept
    max_pages: int = 50
    max_text_chars: int = 20000
    extract_timeout: float = 15.0
    stage_delay: float = 0.7                # visible pacing between stages (0 in tests)
    workers: int = 3
    pdftotext: tuple = ("pdftotext",)

    def __post_init__(self):
        self.root = Path(self.root)
        self.inbox_dir = Path(self.inbox_dir) if self.inbox_dir else self.root / "inbox"

    def limits(self):
        return {"max_bytes": self.max_bytes, "max_files_per_batch": self.max_files, "max_queue": self.max_queue,
                "max_pages": self.max_pages, "extract_timeout_s": self.extract_timeout, "mime": "application/pdf"}


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


# ---------------------------------------------------------------- extraction
_LABEL_TAIL = r"[ \t]*[:#\-]?[ \t]*"
PATTERNS = {
    "invoice_number": re.compile(r"(?im)^[ \t]*(?:invoice[ \t]*(?:no\.?|number|num\.?|#)|inv[ \t]*(?:no\.?|#))" + _LABEL_TAIL + r"([A-Za-z0-9][A-Za-z0-9\-_/]{1,40})[ \t]*$"),
    "supplier_name": re.compile(r"(?im)^[ \t]*(?:supplier|vendor|seller|billed[ \t]+from|bill[ \t]+from|from)[ \t]*[:\-][ \t]*(\S.{1,78}?)[ \t]*$"),
    "invoice_date": re.compile(r"(?im)^[ \t]*(?:invoice[ \t]+date|issue[ \t]+date|date[ \t]+of[ \t]+issue|date)[ \t]*[:\-][ \t]*(\S.{2,30}?)[ \t]*$"),
    "due_date": re.compile(r"(?im)^[ \t]*(?:due[ \t]+date|payment[ \t]+due|due)[ \t]*[:\-][ \t]*(\S.{2,30}?)[ \t]*$"),
    "beneficiary_name": re.compile(r"(?im)^[ \t]*(?:beneficiary|account[ \t]+name|account[ \t]+holder|payee)[ \t]*[:\-][ \t]*(\S.{1,78}?)[ \t]*$"),
    "beneficiary_account": re.compile(r"(?im)^[ \t]*(?:iban|bank[ \t]+account|beneficiary[ \t]+account|account[ \t]*(?:no\.?|number|#))[ \t]*[:\-]?[ \t]*([A-Za-z0-9][A-Za-z0-9 \-]{5,38}?)[ \t]*$"),
}
AMOUNT_PATTERNS = (
    re.compile(r"(?im)^[ \t]*(?:amount[ \t]+due|balance[ \t]+due|total[ \t]+due|grand[ \t]+total)[ \t]*[:\-]?[ \t]*(.+?)[ \t]*$"),
    re.compile(r"(?im)^[ \t]*(?:total(?:[ \t]+amount)?)[ \t]*[:\-]?[ \t]*(.+?)[ \t]*$"),
)
AMOUNT_RE = re.compile(r"^(?:(?P<c1>[A-Z]{3}|[$€£])[ \t]*)?(?P<n>\d[\d.,' ]*\d|\d)(?:[ \t]*(?P<c2>[A-Z]{3}|[$€£]))?$")
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_amount(raw):
    m = AMOUNT_RE.match(raw.strip())
    if not m:
        return None
    n = m.group("n").replace("'", "").replace(" ", "")
    cur = m.group("c1") or m.group("c2")
    cur = SYMBOLS.get(cur, cur)
    if cur is not None and cur not in CURRENCIES:
        cur = None
    dec = None
    if "," in n and "." in n:
        dec = "," if n.rfind(",") > n.rfind(".") else "."
    elif "," in n or "." in n:
        sep = "," if "," in n else "."
        tail = n.split(sep)[-1]
        if n.count(sep) == 1 and len(tail) in (1, 2):
            dec = sep
        elif n.count(sep) > 1 or len(tail) == 3:
            dec = None   # thousands separators only
        else:
            return None
    if dec:
        whole, frac = n.rsplit(dec, 1)
        whole = re.sub(r"[.,]", "", whole)
    else:
        whole, frac = re.sub(r"[.,]", "", n), "00"
    frac = (frac + "00")[:2]
    return {"amount": f"{int(whole)}.{frac}", "currency": cur}


def normalize_date(raw):
    s = raw.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31 else None
    m = re.match(r"^(\d{1,2})[ \-]([A-Za-z]{3,9})[ \-,]+(\d{4})$", s) or None
    if m and m.group(2)[:3].lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(2)[:3].lower()]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"^([A-Za-z]{3,9})\.? (\d{1,2}),? (\d{4})$", s)
    if m and m.group(1)[:3].lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"
    return None


def _found(value, evidence, **extra):
    d = {"found": True, "value": value, "source": "uploaded_pdf", "evidence": evidence[:120]}
    d.update(extra)
    return d


def _missing():
    return {"found": False, "value": None, "source": None, "evidence": None}


def extract_fields(text):
    """Deterministic label-based extraction over embedded text. Anything not present stays missing (never inferred)."""
    fields, warnings = {}, []
    for key in ("invoice_number", "supplier_name", "invoice_date", "due_date", "beneficiary_name", "beneficiary_account"):
        m = PATTERNS[key].search(text)
        if not m:
            fields[key] = _missing(); continue
        val = m.group(1).strip()
        if key in ("invoice_date", "due_date"):
            iso = normalize_date(val)
            fields[key] = _found(val, m.group(0).strip(), normalized=iso)
            if iso is None:
                warnings.append(f"{key.replace('_', ' ')} '{val}' has an ambiguous or unsupported format; kept as written")
        else:
            fields[key] = _found(val, m.group(0).strip())
    amount = _missing(); currency = _missing()
    for pat in AMOUNT_PATTERNS:
        for m in pat.finditer(text):
            parsed = parse_amount(m.group(1))
            if parsed:
                amount = _found(parsed["amount"], m.group(0).strip(), raw=m.group(1).strip())
                if parsed["currency"]:
                    currency = _found(parsed["currency"], m.group(0).strip())
                break
        if amount["found"]:
            break
    fields["amount"], fields["currency"] = amount, currency
    if amount["found"] and not currency["found"]:
        warnings.append("amount found but no currency was stated in the document")
    return fields, warnings


REQUIRED = ("invoice_number", "supplier_name", "amount", "currency")


def extract_pdf(path, cfg):
    """Run poppler pdftotext on a stored PDF. Returns dict(status, fields, warnings, text, text_truncated, error)."""
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
            return {"status": "FAILED", "error": f"extraction timed out after {cfg.extract_timeout:g}s"}
    except FileNotFoundError:
        return {"status": "FAILED", "error": "PDF text extractor (pdftotext) is not installed"}
    err = err.decode("utf-8", "replace")
    if proc.returncode != 0:
        if "password" in err.lower() or "encrypt" in err.lower():
            return {"status": "NEEDS_REVIEW", "error": "PDF is encrypted; text cannot be read", "fields": {}, "warnings": ["encrypted PDF"], "text": "", "text_truncated": False}
        return {"status": "FAILED", "error": "PDF could not be parsed (malformed or unsupported)"}
    text, truncated = sanitize_text(out.decode("utf-8", "replace"), cfg.max_text_chars)
    warnings = ["extracted text was truncated for display"] if truncated else []
    if len(re.sub(r"\s", "", text)) < 20:
        return {"status": "NEEDS_OCR", "error": None, "fields": {}, "warnings": warnings + ["no usable embedded text; OCR is not available in this build"], "text": text, "text_truncated": truncated}
    fields, fw = extract_fields(text)
    warnings += fw
    missing = [k for k in REQUIRED if not fields[k]["found"]]
    if missing:
        warnings.append("missing required fields: " + ", ".join(missing))
    return {"status": "NEEDS_REVIEW" if missing else "READY_FOR_PROCESSING", "error": None, "fields": fields, "warnings": warnings,
            "text": text, "text_truncated": truncated}


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
"""
SUMMARY_COLS = "id, sha256, display_name, size, source, created_at, updated_at, status, progress, latest_event, error, duplicate_of, extraction"


class IntakeService:
    def __init__(self, cfg: IntakeConfig, source: InvoiceSource = None):
        self.cfg = cfg
        cfg.root.mkdir(parents=True, exist_ok=True)
        self.uploads = cfg.root / "uploads"
        self.uploads.mkdir(exist_ok=True)
        cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.source = source or LocalFolderInvoiceSource(cfg.inbox_dir, cfg.max_bytes)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(cfg.root / "jobs.sqlite"), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="intake")
        self.scans = {}
        self.closed = False
        with self.lock:   # jobs left mid-flight by a previous process can never finish
            for r in self.db.execute("SELECT id FROM jobs WHERE status IN ('UPLOADING','VALIDATING','EXTRACTING')").fetchall():
                self._set(r["id"], "FAILED", "Interrupted by a server restart", error="interrupted by restart")

    # -- low-level
    def _events(self, job_id):
        r = self.db.execute("SELECT events FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(r["events"]) if r else []

    def _set(self, job_id, status, message, **cols):
        with self.lock:
            ev = self._events(job_id)
            if not ev and self.db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone() is None:
                return
            ev.append({"at": now(), "stage": status, "message": message})
            sets = {"status": status, "progress": PROGRESS[status], "latest_event": message, "events": json.dumps(ev), "updated_at": now()}
            for k, v in cols.items():
                sets[k] = json.dumps(v) if k == "extraction" else v
            self.db.execute("UPDATE jobs SET " + ", ".join(f"{k}=?" for k in sets) + " WHERE id=?", (*sets.values(), job_id))

    def _new_job(self, display_name, source, size, mime, status="UPLOADING", message="Waiting for file content", error=None):
        jid = uuid.uuid4().hex
        t = now()
        with self.lock:
            total = self.db.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]
            if total >= self.cfg.max_queue:
                raise IntakeError(f"queue is full ({self.cfg.max_queue} jobs); reset the demo queue first", 429)
            ev = [{"at": t, "stage": status, "message": message}]
            self.db.execute("INSERT INTO jobs(id,display_name,size,source,mime,created_at,updated_at,status,progress,latest_event,events,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (jid, safe_display_name(display_name), size, source, mime, t, t, status, PROGRESS[status], message, json.dumps(ev), error))
        return jid

    def _fail(self, jid, reason, stored=None):
        self._set(jid, "FAILED", reason, error=reason)
        if stored:
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
        if ex:
            f = ex.get("fields", {})
            d["summary"] = {k: (f[k]["value"] if k in f and f[k]["found"] else None) for k in ("invoice_number", "supplier_name", "amount", "currency")}
        else:
            d["summary"] = {k: None for k in ("invoice_number", "supplier_name", "amount", "currency")}
        if detail:
            d["events"] = json.loads(r["events"]); d["extraction"] = ex
        return d

    def _expire_stale(self):
        cutoff = datetime.fromtimestamp(time.time() - UPLOAD_STALE_S, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in self.db.execute("SELECT id FROM jobs WHERE status='UPLOADING' AND updated_at < ?", (cutoff,)).fetchall():
            self._fail(r["id"], "upload was not received")

    def list_jobs(self):
        with self.lock:
            self._expire_stale()
            rows = self.db.execute("SELECT * FROM jobs ORDER BY seq DESC").fetchall()
            return [self._row(r) for r in rows]

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
        return {"enabled": True, "limits": self.cfg.limits(), "counts": c, "total": len(jobs),
                "waiting": c["UPLOADING"] + c["VALIDATING"], "extracting": c["EXTRACTING"], "ready": c["READY_FOR_PROCESSING"],
                "attention": c["NEEDS_OCR"] + c["NEEDS_REVIEW"], "failed": c["FAILED"], "duplicates": c["DUPLICATE"],
                "last_scan": json.loads(last["value"]) if last else None, "inbox": "demo inbox (local folder)", "source": self.source.name}

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
            if size > self.cfg.max_bytes:
                jid = self._new_job(name, "manual_upload", size, mime, "FAILED", f"file exceeds {self.cfg.max_bytes // (1024 * 1024)} MB limit", f"file exceeds {self.cfg.max_bytes // (1024 * 1024)} MB limit")
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

    # -- shared pipeline entry: hash, store, then process asynchronously
    def _ingest(self, jid, data):
        ok, reason, warnings = validate_pdf_bytes(data, self.cfg.max_bytes)
        sha = hashlib.sha256(data).hexdigest()
        stored = None
        with self.lock:
            self.db.execute("UPDATE jobs SET sha256=?, size=? WHERE id=?", (sha, len(data), jid))
        if ok:
            stored = uuid.uuid4().hex + ".pdf"
            tmp = self.uploads / (stored + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, self.uploads / stored)
            with self.lock:
                self.db.execute("UPDATE jobs SET stored_name=? WHERE id=?", (stored, jid))
        self.pool.submit(self._process, jid, ok, reason, warnings, stored)

    def _process(self, jid, ok, reason, warnings, stored):
        try:
            self._set(jid, "VALIDATING", "Checking PDF signature, size and type")
            self._pause()
            if not ok:
                return self._fail(jid, reason, stored)
            with self.lock:   # duplicate check + claim is atomic so identical concurrent uploads yield one original
                r = self.db.execute("SELECT sha256 FROM jobs WHERE id=?", (jid,)).fetchone()
                dup = self.db.execute("SELECT id FROM jobs WHERE sha256=? AND id!=? AND claimed=1 AND status NOT IN ('FAILED','DUPLICATE')", (r["sha256"], jid)).fetchone()
                if dup:
                    self._set(jid, "DUPLICATE", f"Identical content already in the queue (job {dup['id'][:8]})", duplicate_of=dup["id"])
                    self._delete_file(stored)
                    return
                self.db.execute("UPDATE jobs SET claimed=1 WHERE id=?", (jid,))
            self._set(jid, "EXTRACTING", "Extracting embedded text" + (" (" + "; ".join(warnings) + ")" if warnings else ""))
            self._pause()
            res = extract_pdf(self.uploads / stored, self.cfg)
            status = res["status"]
            if status == "FAILED":
                return self._fail(jid, res["error"], stored)
            res["warnings"] = warnings + res.get("warnings", [])
            msg = {"READY_FOR_PROCESSING": "Extraction complete: ready for agent processing",
                   "NEEDS_OCR": "No embedded text found: OCR is required",
                   "NEEDS_REVIEW": res.get("error") or "Extraction complete: fields need review"}[status]
            self._set(jid, status, msg, extraction={k: res[k] for k in ("fields", "warnings", "text", "text_truncated")}, error=res.get("error"))
        except Exception as e:   # one bad file must never take down the worker or the batch
            self._fail(jid, f"internal error: {type(e).__name__}", stored)

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
            snapshot = dict(s)   # taken before the worker can advance it: callers always see the scan as started
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
                if data is None:   # oversize: never read
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
                time.sleep(0)
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
                s["state"] = "complete"   # last: wait_idle() and pollers see a scan as finished only after its summary is persisted

    def _seen_by_name_size(self, name, size):
        with self.lock:
            return self.db.execute("SELECT 1 FROM jobs WHERE source=? AND display_name=? AND size=? AND status='FAILED'", (self.source.name, safe_display_name(name), size)).fetchone() is not None

    def _reject(self, name, size, reason):
        self._new_job(name, self.source.name, size, "application/pdf", "FAILED", reason, reason)

    # -- lifecycle
    def wait_idle(self, timeout=10.0, include_uploading=True):
        """Block until no job is active and no scan is running (UPLOADING jobs await the client, so close() ignores them)."""
        active = "('UPLOADING','VALIDATING','EXTRACTING')" if include_uploading else "('VALIDATING','EXTRACTING')"
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                busy = self.db.execute(f"SELECT COUNT(*) c FROM jobs WHERE status IN {active}").fetchone()["c"]
                scanning = any(s["state"] == "running" for s in self.scans.values())
            if not busy and not scanning:
                return True
            time.sleep(0.02)
        return False

    def reset(self):
        """Delete queue state and temporary uploads. The inbox folder and its files are never touched."""
        with self.lock:
            self.db.execute("DELETE FROM jobs")
            self.db.execute("DELETE FROM meta")
            self.db.execute("DELETE FROM sqlite_sequence WHERE name='jobs'")
            self.scans.clear()
            removed = 0
            for p in self.uploads.iterdir():
                if p.is_file() and not p.is_symlink():
                    p.unlink(); removed += 1
        return {"reset": True, "uploads_removed": removed}

    def close(self):
        self.wait_idle(5, include_uploading=False)
        with self.lock:
            self.closed = True
            self.db.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
