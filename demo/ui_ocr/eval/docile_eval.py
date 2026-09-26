#!/usr/bin/env python3
"""DEV-ONLY harness: measure the runtime OCR extractor against DocILE gold KILE fields. Never imported by the app or server.

Discipline:
* Tuning uses official-train documents that the project has NOT selected for its own probe/LoRA work (subset manifests excluded).
* The official validation split (none of it project-selected) is scored only for the final report.
* Gold labels are read here only; the runtime (extraction.py / intake.py / ocr.py) never sees them.

  python3 eval/docile_eval.py select            # write eval/splits.json (seeded)
  python3 eval/docile_eval.py ocr tune          # render + OCR the tuning docs into eval/cache/ (real Tesseract, cached)
  python3 eval/docile_eval.py score tune [-v]   # per-field accuracy + error samples
"""
import concurrent.futures as cf
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
sys.path.insert(0, str(APP))
import extraction as X  # noqa: E402
import ocr as O  # noqa: E402

DATA = Path("/home/hp5/tell/data/docile")
CACHE = HERE / "cache"
SPLITS = HERE / "splits.json"
PROJECT = Path("/tmp/claude-1001/-home-hp5/8c5a7bf6-64b4-4c69-9c45-805e21c0a97c/scratchpad/project_docs.json")
PSM = None                                    # None = engine default (auto page segmentation)


# ---------------------------------------------------------------- selection
def select(n_tune=150, n_val=120, seed=20260925):
    proj = set(json.loads(PROJECT.read_text())) if PROJECT.exists() else set()
    train = sorted(set(json.load(open(DATA / "train.json"))) - proj)
    val = sorted(set(json.load(open(DATA / "val.json"))) - proj)
    rnd = random.Random(seed)
    out = {"seed": seed, "excluded_project_docs": len(proj), "tune": rnd.sample(train, n_tune), "val": rnd.sample(val, n_val)}
    SPLITS.write_text(json.dumps(out, indent=1))
    print(f"tune={len(out['tune'])} val={len(out['val'])} (excluded {len(proj)} project-selected docs)")


# ---------------------------------------------------------------- OCR cache
def ocr_doc(doc_id, limits=None, psm=None):
    """Render (kept for re-OCR experiments) + OCR one document; cache text/lines/confidences as JSON."""
    psm = psm or (int(__import__("os").environ["EVAL_PSM"]) if __import__("os").environ.get("EVAL_PSM") else None)
    out = CACHE / "ocr2" / (doc_id + (f"-psm{psm}" if psm else "") + ".json")
    if out.exists():
        return json.loads(out.read_text())
    limits = limits or O.OcrLimits()
    png_dir = CACHE / "png" / doc_id
    if not png_dir.exists() or not list(png_dir.glob("page-*.png")):
        png_dir.mkdir(parents=True, exist_ok=True)
        O.render_pages(DATA / "pdfs" / f"{doc_id}.pdf", png_dir, limits)
    engine = O.find_engine()
    pages, t0 = [], time.time()
    for p in sorted(png_dir.glob("page-*.png"), key=lambda q: int(re.search(r"(\d+)", q.stem).group(1))):
        no = int(re.search(r"(\d+)", p.stem).group(1))
        r = O.ocr_page(engine, p, no, limits, psm=psm)
        pages.append({"page": r.page, "text": r.text, "line_conf": [l["confidence"] for l in r.lines], "line_boxes": [l["boxes"] for l in r.lines], "usable_words": r.usable_words})
    res = {"doc": doc_id, "pages": pages, "seconds": round(time.time() - t0, 2)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def run_ocr_split(name, workers=8, psm=None):
    ids = json.loads(SPLITS.read_text())[name]
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(lambda i: ocr_doc(i, psm=psm), ids))
    print(f"{len(done)} docs OCR'd in {time.time() - t0:.0f}s (cached under eval/cache)")


# ---------------------------------------------------------------- gold
def gold_of(doc_id):
    a = json.loads((DATA / "annotations" / f"{doc_id}.json").read_text())
    g = {}
    for f in a["field_extractions"]:
        g.setdefault(f["fieldtype"], []).append(f["text"])
    return g, a["metadata"]


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def to_ymd(s):
    """Parse a date string (gold or ours) to (y, m, d); MDY assumed for all-numeric forms. None if unparseable."""
    s = s.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    iso = X.normalize_date(s)
    if iso:
        y, mo, d = map(int, iso.split("-")); return y, mo, d
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3)); y += 2000 if y < 100 else 0
        return y, a, b
    return None


def same_name(a, b):
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    return SequenceMatcher(None, alnum(a), alnum(b)).ratio() >= 0.8


def amt(s):
    pa = X.parse_amount(s.strip()) if s else None
    return pa["amount"] if pa else None


PAYABLE = {"tax_invoice"}


def score_doc(doc_id, ocr, gold, meta):
    """-> {field: (status, detail)} for the fields the runtime extracts. status: correct|wrong|ambiguous|missing|absent(gold none)."""
    fields, warnings = X.extract_fields(ocr["pages"], "OCR")
    dec = X.assess(fields, "OCR")
    text = alnum(" ".join(p["text"] for p in ocr["pages"]))
    res = {}

    def cmp(name, ours_key, gold_vals, eq):
        f = fields[ours_key]
        if not gold_vals:
            res[name] = ("absent", f["value"] if f["found"] else None); return
        reach = any(alnum(g) in text for g in gold_vals)
        if f["found"]:
            ok = any(eq(f["value"], g) for g in gold_vals)
            res[name] = ("correct" if ok else "wrong", (f["value"], gold_vals[:2], reach))
        elif f["ambiguous"]:
            hit = any(any(eq(c, g) for g in gold_vals) for c in f["candidates"])
            res[name] = ("ambiguous", (f["candidates"][:4], gold_vals[:2], hit, reach))
        else:
            res[name] = ("missing", (gold_vals[:2], reach))

    cmp("supplier_name", "supplier_name", gold.get("vendor_name", []), same_name)
    cmp("invoice_number", "invoice_number", gold.get("document_id", []), lambda a, b: alnum(a) == alnum(b))
    cmp("invoice_date", "invoice_date", gold.get("date_issue", []), lambda a, b: to_ymd(a) is not None and to_ymd(a) == to_ymd(b))
    cmp("due_date", "due_date", gold.get("date_due", []), lambda a, b: to_ymd(a) is not None and to_ymd(a) == to_ymd(b))
    due = gold.get("amount_due", []) or gold.get("amount_total_gross", [])
    cmp("amount", "amount", due, lambda a, b: amt(b) is not None and a == amt(b))
    cmp("beneficiary_account", "beneficiary_account", gold.get("iban", []) + gold.get("account_num", []), lambda a, b: alnum(a) == alnum(b))
    # currency: gold is usually a symbol; policy says a bare symbol is ambiguous, so score "symbol/code seen" separately from "resolved code"
    gc = gold.get("currency_code_amount_due", [])
    cf_ = fields["currency"]
    if not gc:
        res["currency"] = ("absent", cf_["value"])
    elif cf_["found"]:
        res["currency"] = ("correct" if alnum(cf_["value"]) == alnum(gc[0]) or (gc[0] == "$" and cf_["value"] == "USD") else "wrong", (cf_["value"], gc[:1]))
    elif cf_["ambiguous"]:
        res["currency"] = ("ambiguous", (cf_["candidates"][:3], gc[:1]))
    else:
        res["currency"] = ("missing", (gc[:1],))
    payable = meta.get("document_type") in PAYABLE
    ours = fields["document_type"]
    fam_payable = ours["found"] and ours["value"] == "invoice"
    res["is_payable_invoice"] = ("correct" if fam_payable == payable else "wrong", (ours["value"], meta.get("document_type"), ours["ambiguous"]))
    return res, fields, dec, warnings


def score(name, verbose=False, show=int(__import__("os").environ.get("SHOW", 6)), only=None):
    only = only or __import__("os").environ.get("ONLY")
    ids = json.loads(SPLITS.read_text())[name]
    if only:
        ids = [i for i in ids if (gold_of(i)[1].get("document_type") == only) == (not only.startswith("!"))] if not only.startswith("!") else [i for i in ids if gold_of(i)[1].get("document_type") != only[1:]]
    table, samples = {}, {}
    status_of_doc = {"READY": 0, "CLARIFY": 0, "REVIEW": 0}
    for i in ids:
        ocr = ocr_doc(i)
        gold, meta = gold_of(i)
        res, fields, dec, w = score_doc(i, ocr, gold, meta)
        status_of_doc[dec["status_hint"]] += 1
        for k, (st, det) in res.items():
            table.setdefault(k, {}).setdefault(st, 0); table[k][st] += 1
            samples.setdefault((k, st), []).append((i, det))
    print(f"\n== {name}{(' [only '+only+']') if only else ''}: {len(ids)} docs  (decision mix: {status_of_doc})")
    print(f"{'field':22} {'gold':>5} {'correct':>8} {'wrong':>6} {'ambig':>6} {'missing':>8} {'spurious':>9} | {'recall':>7} {'precision':>9}")
    for k, t in table.items():
        c, w_, a, m, ab = (t.get(x, 0) for x in ("correct", "wrong", "ambiguous", "missing", "absent"))
        gold_n = c + w_ + a + m
        if k == "is_payable_invoice":
            print(f"{k:22} {len(ids):5d} {c:8d} {w_:6d}  (payable-vs-not classification accuracy {c / len(ids):.0%})"); continue
        spur = sum(1 for _, d in samples.get((k, "absent"), []) if d)
        prec = f"{c / (c + w_):.0%}" if c + w_ else "n/a"
        print(f"{k:22} {gold_n:5d} {c:8d} {w_:6d} {a:6d} {m:8d} {spur:9d} | {c / gold_n if gold_n else 0:7.0%} {prec:>9}")
    if verbose:
        for k in table:
            for st in ("wrong", "missing", "ambiguous"):
                for i, det in samples.get((k, st), [])[:show]:
                    print(f"  [{k}/{st}] {i[:8]} {det}")
    return table


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select()
    elif cmd == "ocr":
        run_ocr_split(sys.argv[2])
    elif cmd == "score":
        score(sys.argv[2], verbose="-v" in sys.argv)
    elif cmd == "extra":                    # python3 eval/docile_eval.py extra tune2 200   (fresh train docs, disjoint from every earlier split)
        sp = json.loads(SPLITS.read_text()); proj = set(json.loads(PROJECT.read_text())) if PROJECT.exists() else set()
        used = set(sp["tune"]) | set(sp["val"]) | {i for k, v in sp.items() if isinstance(v, list) for i in v}
        pool = sorted(set(json.load(open(DATA / "train.json"))) - proj - used)
        sp[sys.argv[2]] = random.Random(sp["seed"] + 1).sample(pool, int(sys.argv[3])); SPLITS.write_text(json.dumps(sp, indent=1)); print(sys.argv[2], len(sp[sys.argv[2]]))
    elif cmd == "psm":                      # python3 eval/docile_eval.py psm tune 4
        PSM_MODE = int(sys.argv[3]); run_ocr_split(sys.argv[2], psm=PSM_MODE)
