"""Local CPU OCR adapter: Poppler (`pdfinfo`, `pdftoppm`) renders pages, Tesseract reads them.

Narrowly scoped and fail-closed:
* every subprocess is started from a list (never a shell), with no filename interpolation, in a scrubbed environment;
* each runs under `prlimit` (address-space + CPU) *and* a wall-clock timeout that kills the whole process group;
* rendering is bounded by page count, page size and pixel size; rendered images live only in a per-job temp dir that the
  caller removes;
* an engine that is missing, times out, exits non-zero or prints malformed output yields an explicit technical failure
  (`OcrError`) -- never an empty "successful" result.
No cloud service, no network, no model server. Embedded PDF content is never executed (poppler only rasterises).
"""
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DPI = 300
MAX_PIXELS_SIDE = 5000            # cap on either rendered dimension
MAX_PAGE_INCHES = 40.0            # refuse absurd page boxes outright
MIN_USABLE_WORDS = 8              # OCR "usable content" floor (words with real text and confidence)
MIN_WORD_CONF = 30.0
ENGINE_SEARCH = ("/usr/bin/tesseract", "/usr/local/bin/tesseract")
SAFE_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "OMP_THREAD_LIMIT": "1"}


class OcrError(Exception):
    """Technical OCR failure. `code` is machine-readable; the job then ends FAILED (fail closed)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class OcrLimits:
    max_pages: int = 10
    dpi: int = DEFAULT_DPI
    render_timeout: float = 60.0          # per PDF
    ocr_timeout: float = 90.0             # per page
    total_timeout: float = 240.0          # whole OCR stage
    mem_bytes: int = 3 * 1024 ** 3        # RLIMIT_AS for each child
    cpu_seconds: int = 120                # RLIMIT_CPU for each child


@dataclass
class PageResult:
    page: int
    outcome: str                          # OK | EMPTY | ERROR
    words: int = 0
    usable_words: int = 0
    mean_confidence: float = 0.0          # 0..1 over usable words
    duration_s: float = 0.0
    text: str = ""
    lines: list = field(default_factory=list)   # [{"text", "confidence"(0..1)}]
    error: str = None


@dataclass
class OcrResult:
    engine: str
    version: str
    dpi: int
    page_count: int
    pages: list
    duration_s: float
    usable: bool
    usable_words: int
    mean_confidence: float


# ---------------------------------------------------------------- process helpers
def run_limited(cmd, timeout, limits=None, cwd=None, extra_env=None):
    """Run `cmd` (a list) under prlimit with a hard wall-clock timeout. Returns (returncode, stdout_bytes, stderr_bytes)."""
    limits = limits or OcrLimits()
    wrapped = list(cmd)
    if shutil.which("prlimit", path=SAFE_ENV["PATH"]):
        wrapped = ["prlimit", f"--as={limits.mem_bytes}", f"--cpu={limits.cpu_seconds}", "--"] + wrapped
    env = dict(SAFE_ENV)
    env.update(extra_env or {})
    try:
        proc = subprocess.Popen(wrapped, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, cwd=cwd, start_new_session=True)
    except FileNotFoundError:
        raise OcrError("TOOL_MISSING", f"required tool not found: {cmd[0]}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise OcrError("TIMEOUT", f"{Path(cmd[0]).name} timed out after {timeout:g}s")
    return proc.returncode, out, err


# ---------------------------------------------------------------- engine discovery
def find_engine(explicit=None):
    """Return the Tesseract executable path or None. `explicit` (or TELL_TESSERACT) lets tests and a user-local install point at a binary."""
    cand = explicit or os.environ.get("TELL_TESSERACT")
    if cand:
        return cand if os.access(cand, os.X_OK) else None
    for p in ENGINE_SEARCH:
        if os.access(p, os.X_OK):
            return p
    return shutil.which("tesseract", path=SAFE_ENV["PATH"])


def engine_info(path, tessdata=None):
    """(name, version) for the engine, or raise OcrError. Version comes from the engine itself, never assumed."""
    env = {"TESSDATA_PREFIX": tessdata} if tessdata else None
    rc, out, err = run_limited([path, "--version"], 15, extra_env=env)
    text = (out + err).decode("utf-8", "replace")
    m = re.search(r"tesseract\s+v?(\d[\w.\-]*)", text, re.I)
    if rc != 0 or not m:
        raise OcrError("ENGINE_UNUSABLE", "OCR engine did not report a version")
    return "tesseract", m.group(1)


# ---------------------------------------------------------------- rendering
def pdf_page_info(pdf, limits):
    rc, out, err = run_limited(["pdfinfo", "-box", str(pdf)], 20, limits)
    text = out.decode("utf-8", "replace")
    if rc != 0:
        raise OcrError("PDF_UNREADABLE", "PDF could not be inspected")
    m = re.search(r"^Pages:\s+(\d+)", text, re.M)
    if not m:
        raise OcrError("PDF_UNREADABLE", "PDF page count unavailable")
    sz = re.search(r"^Page size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", text, re.M)
    w, h = (float(sz.group(1)), float(sz.group(2))) if sz else (612.0, 792.0)
    return int(m.group(1)), w, h


def render_pages(pdf, outdir, limits):
    """Rasterise up to `limits.max_pages` pages to PNG in `outdir`. Returns (page_count_total, [(page_no, png_path)], dpi_used)."""
    total, w_pt, h_pt = pdf_page_info(pdf, limits)
    if total < 1:
        raise OcrError("PDF_UNREADABLE", "PDF has no pages")
    if max(w_pt, h_pt) / 72.0 > MAX_PAGE_INCHES:
        raise OcrError("PAGE_TOO_LARGE", "page dimensions exceed the OCR safety limit")
    dpi = max(72, min(limits.dpi, int(MAX_PIXELS_SIDE / (max(w_pt, h_pt) / 72.0))))
    last = min(total, limits.max_pages)
    rc, out, err = run_limited(["pdftoppm", "-png", "-gray", "-r", str(dpi), "-f", "1", "-l", str(last), str(pdf), str(Path(outdir) / "page")],
                               limits.render_timeout, limits, cwd=str(outdir))
    if rc != 0:
        raise OcrError("RENDER_FAILED", "PDF page rendering failed")
    pages = []
    for p in sorted(Path(outdir).glob("page-*.png")):
        m = re.fullmatch(r"page-(\d+)\.png", p.name)
        if m:
            pages.append((int(m.group(1)), p))
    if not pages:
        raise OcrError("RENDER_FAILED", "no page images were produced")
    return total, pages, dpi


# ---------------------------------------------------------------- OCR
TSV_HEADER = ["level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"]


def parse_tsv(raw):
    """Parse Tesseract TSV into (lines, words). Any structural surprise raises OcrError(MALFORMED_OUTPUT)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise OcrError("MALFORMED_OUTPUT", "OCR output was not valid UTF-8")
    rows = text.replace("\r\n", "\n").split("\n")
    if not rows or rows[0].split("\t") != TSV_HEADER:
        raise OcrError("MALFORMED_OUTPUT", "OCR output did not have the expected TSV header")
    lines, order, words = {}, [], 0
    for row in rows[1:]:
        if not row.strip("\n"):
            continue
        cols = row.split("\t")
        if len(cols) < 12:
            cols += [""] * (12 - len(cols))
        if len(cols) != 12:
            raise OcrError("MALFORMED_OUTPUT", "OCR TSV row had an unexpected column count")
        try:
            level = int(cols[0]); conf = float(cols[10])
            key = (int(cols[2]), int(cols[3]), int(cols[4])); left = int(cols[6])
        except ValueError:
            raise OcrError("MALFORMED_OUTPUT", "OCR TSV row had non-numeric structural fields")
        if level != 5 or conf < 0:
            continue
        word = cols[11].strip()
        if not word:
            continue
        if key not in lines:
            lines[key] = []; order.append(key)
        lines[key].append((left, word, conf)); words += 1
    out = []
    for key in order:
        ws = sorted(lines[key], key=lambda t: t[0])
        good = [c for _, w, c in ws if c >= MIN_WORD_CONF and len(re.sub(r"\W", "", w)) >= 1]
        out.append({"text": " ".join(w for _, w, _ in ws), "confidence": round((sum(c for _, _, c in ws) / len(ws)) / 100.0, 3),
                    "words": len(ws), "usable_words": len(good)})
    return out, words


def ocr_page(engine, png, page_no, limits, tessdata=None, lang="eng"):
    """OCR one rendered page. Returns PageResult; raises OcrError for timeouts / exit failures / malformed output."""
    import time
    t0 = time.monotonic()
    env = {"TESSDATA_PREFIX": tessdata} if tessdata else None
    cmd = [engine, str(png), "stdout", "-l", lang, "--dpi", str(limits.dpi), "-c", "preserve_interword_spaces=1", "tsv"]
    rc, out, err = run_limited(cmd, limits.ocr_timeout, limits, extra_env=env)
    if rc != 0:
        raise OcrError("ENGINE_FAILED", f"OCR engine failed on page {page_no} (exit {rc})")
    lines, words = parse_tsv(out)
    usable = sum(l["usable_words"] for l in lines)
    confs = [l["confidence"] for l in lines if l["usable_words"]]
    text = "\n".join(l["text"] for l in lines)
    return PageResult(page=page_no, outcome="OK" if usable else "EMPTY", words=words, usable_words=usable,
                      mean_confidence=round(sum(confs) / len(confs), 3) if confs else 0.0,
                      duration_s=round(time.monotonic() - t0, 2), text=text,
                      lines=[{"text": l["text"], "confidence": l["confidence"]} for l in lines])


def run_ocr(pdf, workdir, limits, engine, tessdata=None, on_page=None, on_rendered=None):
    """Render + OCR every permitted page. `workdir` is the job's isolated temp dir (caller deletes it).
    Callbacks: on_rendered(total_pages, pages_rendered, dpi) then on_page(PageResult, index, count) after each page."""
    import time
    t0 = time.monotonic()
    name, version = engine_info(engine, tessdata)
    total, pages, dpi = render_pages(pdf, workdir, limits)
    if on_rendered:
        on_rendered(total, len(pages), dpi)
    results = []
    for i, (no, png) in enumerate(pages, 1):
        if time.monotonic() - t0 > limits.total_timeout:
            raise OcrError("TIMEOUT", "OCR exceeded the total time budget")
        res = ocr_page(engine, png, no, limits, tessdata)
        results.append(res)
        try:
            png.unlink()          # remove each page image as soon as it has been read
        except FileNotFoundError:
            pass
        if on_page:
            on_page(res, i, len(pages))
    usable = sum(r.usable_words for r in results)
    confs = [r.mean_confidence for r in results if r.usable_words]
    return OcrResult(engine=name, version=version, dpi=dpi, page_count=total, pages=results, duration_s=round(time.monotonic() - t0, 2),
                     usable=usable >= MIN_USABLE_WORDS, usable_words=usable,
                     mean_confidence=round(sum(confs) / len(confs), 3) if confs else 0.0)
