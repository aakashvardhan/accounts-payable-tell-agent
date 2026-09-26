"""Typed invoice-field extraction with explicit provenance, over embedded text or OCR text. Pure functions, stdlib only.

Every field is a dict:
  {found, value, source: EMBEDDED_TEXT|OCR, page, confidence(0..1), evidence_text, normalization_applied, ambiguous, candidates}
Nothing is invented: absent fields stay absent; conflicting evidence marks a field *ambiguous* (value None, candidates listed).
Currency is never inferred from supplier, location, language, filename or a generic symbol: a bare symbol ($, EUR sign, ...)
stays ambiguous unless an independent trusted vendor record's canonical currency is one of that symbol's candidates.
The runtime never reads any dataset gold labels; extraction sees only the text it is given.
"""
import re
import unicodedata

SOURCES = ("EMBEDDED_TEXT", "OCR")
REQUIRED = ("invoice_number", "supplier_name", "amount", "currency")
CLARIFIABLE = ("invoice_number", "amount", "currency")      # a vendor can supply these; supplier identity they cannot
FIELD_KEYS = ("document_type", "supplier_name", "invoice_number", "invoice_date", "due_date", "amount", "currency",
              "beneficiary_name", "beneficiary_account")
ISO_CURRENCIES = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD", "JPY", "CNY", "CZK", "PLN", "SEK", "NOK", "DKK", "INR", "MXN", "SGD", "HKD"}
SYMBOL_CANDIDATES = {"$": ["USD", "CAD", "AUD", "NZD", "MXN", "SGD", "HKD"], "€": ["EUR"], "£": ["GBP"], "¥": ["JPY", "CNY"]}
LOW_CONFIDENCE = 0.60

_TAIL = r"[ \t]*[:#\-]?[ \t]*"
LINE_PATTERNS = {
    "invoice_number": re.compile(r"^\s*(?:invoice\s*(?:no\.?|number|num\.?|#)|inv\s*(?:no\.?|#))" + _TAIL + r"([A-Za-z0-9][A-Za-z0-9\-_/]{1,40})\s*$", re.I),
    "supplier_name": re.compile(r"^\s*(?:supplier|vendor|seller|billed\s+from|bill\s+from|from)\s*[:\-]\s*(\S.{1,78}?)\s*$", re.I),
    "invoice_date": re.compile(r"^\s*(?:invoice\s+date|issue\s+date|date\s+of\s+issue|date)\s*[:\-]\s*(\S.{2,30}?)\s*$", re.I),
    "due_date": re.compile(r"^\s*(?:due\s+date|payment\s+due|due)\s*[:\-]\s*(\S.{2,30}?)\s*$", re.I),
    "beneficiary_name": re.compile(r"^\s*(?:beneficiary|account\s+name|account\s+holder|payee)\s*[:\-]\s*(\S.{1,78}?)\s*$", re.I),
    "beneficiary_account": re.compile(r"^\s*(?:iban|bank\s+account|beneficiary\s+account|account\s*(?:no\.?|number|#))\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9 \-]{5,38}?)\s*$", re.I),
}
STRONG_AMOUNT = re.compile(r"^\s*(?:amount\s+due|balance\s+due|total\s+due|grand\s+total)\s*[:\-]?\s*(.+?)\s*$", re.I)
WEAK_AMOUNT = re.compile(r"^\s*(?:total(?:\s+amount)?)\s*[:\-]?\s*(.+?)\s*$", re.I)
CURRENCY_LINE = re.compile(r"^\s*currency\s*[:\-]\s*([A-Z]{3})\s*$", re.I)
AMOUNT_RE = re.compile(r"^(?:(?P<c1>[A-Za-z]{3}|[$€£¥])\s*)?(?P<n>\d[\d.,' ]*\d|\d)(?:\s*(?P<c2>[A-Za-z]{3}|[$€£¥]))?$")
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
TYPE_MARKERS = (("credit_note", r"credit\s+note"), ("purchase_order", r"purchase\s+order"), ("estimate", r"estimate|quotation|quote|proforma"),
                ("contract", r"contract|agreement"), ("receipt", r"receipt"), ("invoice", r"invoice|tax\s+invoice"))


# ---------------------------------------------------------------- primitives
def _empty(source=None):
    return {"found": False, "value": None, "source": None, "page": None, "confidence": None, "evidence_text": None,
            "normalization_applied": None, "ambiguous": False, "candidates": []}


def _norm_key(v):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(v)).casefold()).strip()


def parse_amount(raw):
    """-> {amount:'1234.56', token:'EUR'|'$'|None, normalization:str|None} or None."""
    m = AMOUNT_RE.match(raw.strip())
    if not m:
        return None
    n = m.group("n").replace("'", "").replace(" ", "")
    tok = m.group("c1") or m.group("c2")
    if tok and tok not in SYMBOL_CANDIDATES:
        tok = tok.upper()
        if tok not in ISO_CURRENCIES:
            return None      # a stray 3-letter word is not a currency: reject the whole amount
    dec, note = None, []
    if "," in n and "." in n:
        dec = "," if n.rfind(",") > n.rfind(".") else "."
    elif "," in n or "." in n:
        sep = "," if "," in n else "."
        tail = n.split(sep)[-1]
        if n.count(sep) == 1 and len(tail) in (1, 2):
            dec = sep
        elif n.count(sep) > 1 or len(tail) == 3:
            dec = None
        else:
            return None
    if dec:
        whole, frac = n.rsplit(dec, 1)
        stripped = re.sub(r"[.,]", "", whole)
        if stripped != whole:
            note.append("thousands separators removed")
        if dec == ",":
            note.append("decimal comma converted to '.'")
    else:
        whole, frac = re.sub(r"[.,]", "", n), "00"
        if whole != n:
            note.append("thousands separators removed")
        stripped = whole
    frac = (frac + "00")[:2]
    return {"amount": f"{int(stripped)}.{frac}", "token": tok, "normalization": "; ".join(note) or None}


def normalize_date(raw):
    s = raw.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31 else None
    m = re.match(r"^(\d{1,2})[ \-]([A-Za-z]{3,9})[ \-,]+(\d{4})$", s)
    if m and m.group(2)[:3].lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(2)[:3].lower()]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"^([A-Za-z]{3,9})\.? (\d{1,2}),? (\d{4})$", s)
    if m and m.group(1)[:3].lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"
    return None


def iban_valid(v):
    s = re.sub(r"\s", "", v).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return None            # not IBAN-shaped: a plain account number, no checksum applies
    r = (s[4:] + s[:4])
    return int("".join(str(int(c, 36)) for c in r)) % 97 == 1


# ---------------------------------------------------------------- source normalisation
def _lines(pages):
    """Flatten [{page, text, lines?}] into [(page, text, confidence)]. Embedded text is an exact text layer (confidence 1.0)."""
    out = []
    for p in pages:
        raw = p["text"].split("\n")
        confs = p.get("line_conf")
        for i, t in enumerate(raw):
            if t.strip():
                out.append((p["page"], t, (confs[i] if confs and i < len(confs) else 1.0)))
    return out


def split_embedded_pages(text):
    """pdftotext separates pages with form feeds (sanitize_text turns them into blank lines, so callers pass the raw string)."""
    parts = [x for x in text.split("\f")]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [{"page": i + 1, "text": t} for i, t in enumerate(parts)] or [{"page": 1, "text": text}]


# ---------------------------------------------------------------- classification
def classify_document(lines):
    """Heading-first classification. Returns dict(value, ambiguous, candidates, page, confidence, evidence_text)."""
    heads = [(p, t, c) for p, t, c in lines[:8] if len(t.strip()) <= 48]
    def hits(items):
        found = {}
        for p, t, c in items:
            for name, pat in TYPE_MARKERS:
                if re.search(rf"\b(?:{pat})\b", t, re.I):
                    found.setdefault(name, (p, t.strip(), c))
        return found
    h = hits(heads)
    body = hits(lines)
    cand = h or body
    if "invoice" in body and not h:
        cand = {"invoice": body["invoice"]}
        others = {k: v for k, v in body.items() if k != "invoice"}
        if others:
            cand = dict(body)                    # invoice label + other type words in body, no heading: ambiguous
    if not cand:
        return {"value": "unknown", "ambiguous": False, "candidates": [], "page": None, "confidence": None, "evidence_text": None}
    if len(cand) > 1:
        # a heading "credit note" or "proforma invoice" legitimately co-occurs with the word invoice; the more specific type wins
        specific = [k for k in cand if k != "invoice"]
        if h and len(specific) == 1 and "invoice" in cand:
            k = specific[0]; p, t, c = cand[k]
            return {"value": k, "ambiguous": False, "candidates": [], "page": p, "confidence": c, "evidence_text": t}
        return {"value": None, "ambiguous": True, "candidates": sorted(cand), "page": None, "confidence": None, "evidence_text": None}
    k = next(iter(cand)); p, t, c = cand[k]
    return {"value": k, "ambiguous": False, "candidates": [], "page": p, "confidence": c, "evidence_text": t}


# ---------------------------------------------------------------- extraction
def _mk(value, source, page, conf, evidence, norm=None):
    return {"found": True, "value": value, "source": source, "page": page, "confidence": round(conf, 3), "evidence_text": evidence.strip()[:160],
            "normalization_applied": norm, "ambiguous": False, "candidates": []}


def extract_fields(pages, source, vendor=None):
    """pages: [{page, text, line_conf?}]. vendor: optional independent trusted record dict with `canonical_currency` (used ONLY
    to confirm a currency symbol). Returns (fields, warnings)."""
    assert source in SOURCES
    lines = _lines(pages)
    fields, warnings = {k: _empty() for k in FIELD_KEYS}, []
    used = []

    # document type
    dt = classify_document(lines)
    if dt["ambiguous"]:
        fields["document_type"] = dict(_empty(), ambiguous=True, candidates=dt["candidates"])
        warnings.append("document type is ambiguous: " + " vs ".join(dt["candidates"]))
    elif dt["value"] != "unknown":
        fields["document_type"] = _mk(dt["value"], source, dt["page"], dt["confidence"], dt["evidence_text"])
        used.append((dt["page"], dt["evidence_text"]))

    # single-valued labelled fields
    for key, pat in LINE_PATTERNS.items():
        cands = []
        for pg, text, conf in lines:
            m = pat.match(text)
            if m:
                cands.append((m.group(1).strip(), pg, conf, text))
        if not cands:
            continue
        distinct = {}
        for v, pg, conf, ev in cands:
            distinct.setdefault(_norm_key(v), (v, pg, conf, ev))
        if len(distinct) > 1:
            fields[key] = dict(_empty(), ambiguous=True, candidates=[d[0] for d in distinct.values()])
            warnings.append(f"{key.replace('_', ' ')} has conflicting values: " + " | ".join(d[0] for d in distinct.values()))
            continue
        v, pg, conf, ev = next(iter(distinct.values()))
        norm = None
        if key in ("invoice_date", "due_date"):
            iso = normalize_date(v)
            if iso and iso != v:
                norm = f"date converted to ISO 8601 from '{v}'"
            fields[key] = _mk(iso or v, source, pg, conf, ev, norm)
            if iso is None:
                warnings.append(f"{key.replace('_', ' ')} '{v}' has an ambiguous or unsupported format; kept as written")
        else:
            fields[key] = _mk(v, source, pg, conf, ev)
            if key == "beneficiary_account":
                ok = iban_valid(v)
                if ok is False:
                    warnings.append("beneficiary account looks like an IBAN but fails its checksum (possible OCR error)")
                    fields[key]["normalization_applied"] = "IBAN checksum FAILED"
                elif ok:
                    fields[key]["normalization_applied"] = "IBAN checksum valid"
        used.append((pg, ev))

    # amount: labelled candidates, strong labels outrank a bare 'total'
    strong, weak = [], []
    for pg, text, conf in lines:
        for pat, bucket in ((STRONG_AMOUNT, strong), (WEAK_AMOUNT, weak)):
            m = pat.match(text)
            if m:
                pa = parse_amount(m.group(1))
                if pa:
                    bucket.append((pa, pg, conf, text))
                break
    chosen = strong or weak
    if chosen:
        distinct = {}
        for pa, pg, conf, ev in chosen:
            distinct.setdefault(pa["amount"], (pa, pg, conf, ev))
        if len(distinct) > 1:
            fields["amount"] = dict(_empty(), ambiguous=True, candidates=sorted(distinct))
            warnings.append("amount has conflicting values: " + " | ".join(sorted(distinct)))
        else:
            pa, pg, conf, ev = next(iter(distinct.values()))
            fields["amount"] = _mk(pa["amount"], source, pg, conf, ev, pa["normalization"])
            used.append((pg, ev))
            other = sorted({p["amount"] for p, *_ in weak} - {pa["amount"]}) if strong else []
            if other:
                warnings.append("a 'total' of " + ", ".join(other) + f" differs from the amount due {pa['amount']}; the amount due was used")
            fields["currency"] = _currency(chosen, fields["amount"], lines, source, vendor, warnings, distinct)
            if fields["currency"]["found"]:
                used.append((fields["currency"]["page"], fields["currency"]["evidence_text"]))
    if not fields["currency"]["found"] and not fields["currency"]["ambiguous"]:
        # an explicit "Currency: XXX" line is independent evidence even without an amount line
        codes = {}
        for pg, text, conf in lines:
            m = CURRENCY_LINE.match(text)
            if m and m.group(1).upper() in ISO_CURRENCIES:
                codes.setdefault(m.group(1).upper(), (pg, conf, text))
        if len(codes) == 1:
            code, (pg, conf, ev) = next(iter(codes.items()))
            fields["currency"] = _mk(code, source, pg, conf, ev)
            used.append((pg, ev))
        elif len(codes) > 1:
            fields["currency"] = dict(_empty(), ambiguous=True, candidates=sorted(codes))
            warnings.append("conflicting currency codes: " + ", ".join(sorted(codes)))

    seen, excerpt = set(), []
    for pg, ev in used:
        if ev and (pg, ev) not in seen:
            seen.add((pg, ev)); excerpt.append(ev.strip())
    if excerpt:
        fields["relevant_source_text"] = {"found": True, "value": "\n".join(excerpt[:14]), "source": source, "page": None, "confidence": None,
                                          "evidence_text": None, "normalization_applied": None, "ambiguous": False, "candidates": []}
    else:
        fields["relevant_source_text"] = _empty()
    return fields, warnings


def _currency(chosen, amount_field, lines, source, vendor, warnings, distinct):
    """Resolve currency from the tokens attached to the chosen amount lines + an explicit Currency: line. Never infers."""
    codes, symbols = {}, {}
    for pa, pg, conf, ev in chosen:
        tok = pa["token"]
        if tok in SYMBOL_CANDIDATES:
            symbols.setdefault(tok, (pg, conf, ev))
        elif tok:
            codes.setdefault(tok, (pg, conf, ev))
    for pg, text, conf in lines:
        m = CURRENCY_LINE.match(text)
        if m and m.group(1).upper() in ISO_CURRENCIES:
            codes.setdefault(m.group(1).upper(), (pg, conf, text))
    if len(codes) > 1:
        warnings.append("conflicting currency codes: " + ", ".join(sorted(codes)))
        return dict(_empty(), ambiguous=True, candidates=sorted(codes))
    if len(codes) == 1:
        code, (pg, conf, ev) = next(iter(codes.items()))
        return _mk(code, source, pg, conf, ev)
    if symbols:
        sym, (pg, conf, ev) = next(iter(symbols.items()))
        cands = SYMBOL_CANDIDATES[sym]
        canon = (vendor or {}).get("canonical_currency")
        if canon and canon in cands:
            f = _mk(canon, source, pg, conf, ev, f"symbol '{sym}' resolved to {canon} by the trusted vendor record (canonical currency)")
            f["confirmed_by"] = "TRUSTED_VENDOR_RECORD"
            return f
        warnings.append(f"currency symbol '{sym}' is ambiguous ({', '.join(cands)}) and no trusted vendor record confirms a currency")
        return dict(_empty(), ambiguous=True, candidates=list(cands), page=pg, evidence_text=ev.strip()[:160], source=source)
    return _empty()


# ---------------------------------------------------------------- text quality + assessment
def embedded_quality(text):
    """Is the embedded text layer usable? Returns dict(ok, chars, words, reason)."""
    stripped = re.sub(r"\s", "", text)
    words = re.findall(r"[A-Za-zÀ-ɏ]{2,}", text)
    printable = sum(1 for ch in stripped if ch.isprintable())
    ratio = printable / len(stripped) if stripped else 0.0
    ok = len(stripped) >= 20 and len(words) >= 3 and ratio >= 0.85
    return {"ok": ok, "chars": len(stripped), "words": len(words), "printable_ratio": round(ratio, 3),
            "reason": None if ok else ("no embedded text" if not stripped else "embedded text too sparse or garbled")}


def assess(fields, source):
    """Required-field assessment -> decision dict. status_hint in READY | CLARIFY | REVIEW."""
    dtype = fields["document_type"]
    d = {"document_type": dtype["value"], "missing_required_fields": [], "ambiguous_fields": [], "low_confidence_fields": [],
         "reasons": [], "status_hint": "READY", "payable_schema": True}
    if dtype["ambiguous"]:
        d["status_hint"], d["payable_schema"] = "REVIEW", False
        d["reasons"].append("document type is ambiguous: " + " vs ".join(dtype["candidates"]))
        return d
    if dtype["found"] and dtype["value"] != "invoice":
        d["status_hint"], d["payable_schema"] = "REVIEW", False
        d["reasons"].append(f"document type is '{dtype['value']}', not a payable invoice")
        return d
    for k in REQUIRED:
        f = fields[k]
        if f["ambiguous"]:
            d["ambiguous_fields"].append(k)
        elif not f["found"]:
            d["missing_required_fields"].append(k)
        elif source == "OCR" and (f["confidence"] or 0) < LOW_CONFIDENCE:
            d["low_confidence_fields"].append(k)
    if d["ambiguous_fields"]:
        d["reasons"].append("ambiguous required field(s): " + ", ".join(d["ambiguous_fields"]))
    if d["low_confidence_fields"]:
        d["reasons"].append("low OCR confidence for: " + ", ".join(d["low_confidence_fields"]))
    if d["ambiguous_fields"] or d["low_confidence_fields"]:
        d["status_hint"] = "REVIEW"
        return d
    if "supplier_name" in d["missing_required_fields"]:
        d["status_hint"] = "REVIEW"
        d["reasons"].append("supplier could not be identified, so no trusted vendor contact can be resolved")
        return d
    if d["missing_required_fields"]:
        d["status_hint"] = "CLARIFY"
        d["reasons"].append("missing required field(s): " + ", ".join(d["missing_required_fields"]))
    return d
