"""Typed invoice-field extraction with explicit provenance, over embedded text or OCR text. Pure functions, stdlib only.

Every field is a dict:
  {found, value, source: EMBEDDED_TEXT|OCR, page, confidence(0..1), evidence_text, normalization_applied, ambiguous, candidates}
Nothing is invented: absent fields stay absent; conflicting evidence marks a field *ambiguous* (value None, candidates listed).
Currency is never inferred from supplier, location, language, filename or a generic symbol: a bare symbol ($, EUR sign, ...)
stays ambiguous unless an independent trusted vendor record's canonical currency is one of that symbol's candidates.
The runtime never reads any dataset gold labels; extraction sees only the text it is given.

Method: every line is scanned for *labels* (Invoice No., Invoice Date, Amount Due, ...). Each label yields typed *candidates*
(an ID is not a date, an amount is not a phone number) taken from the same line, or -- for header/value tables -- from the line
below at the same horizontal position when word geometry is available. Candidates are scored (label strength, OCR confidence,
how directly the value follows the label); a field is filled only when one candidate clearly wins, otherwise it is ambiguous.
"""
import re
import unicodedata

SOURCES = ("EMBEDDED_TEXT", "OCR")
REQUIRED = ("invoice_number", "supplier_name", "amount", "currency")
CLARIFIABLE = ("invoice_number", "amount", "currency")      # a vendor can supply these; supplier identity they cannot
FIELD_KEYS = ("document_type", "supplier_name", "invoice_number", "invoice_date", "due_date", "amount", "currency",
              "beneficiary_name", "beneficiary_account")
ISO_CURRENCIES = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD", "JPY", "CNY", "CZK", "PLN", "SEK", "NOK", "DKK", "INR", "MXN", "SGD", "HKD", "BRL", "ZAR"}
# currency symbols -> the ISO codes they can denote. A symbol with exactly one candidate is unambiguous and resolves on its own;
# a symbol with several (bare '$', '¥') stays ambiguous unless a trusted vendor record or an explicit ISO code on the same amount settles it.
SYMBOL_CANDIDATES = {"$": ["USD", "CAD", "AUD", "NZD", "MXN", "SGD", "HKD"], "€": ["EUR"], "£": ["GBP"], "¥": ["JPY", "CNY"], "₹": ["INR"],
                     "US$": ["USD"], "USD$": ["USD"], "CA$": ["CAD"], "C$": ["CAD"], "AU$": ["AUD"], "A$": ["AUD"], "NZ$": ["NZD"], "SG$": ["SGD"], "S$": ["SGD"],
                     "HK$": ["HKD"], "MX$": ["MXN"], "R$": ["BRL"], "zł": ["PLN"], "Kč": ["CZK"]}
_SYM_RX = "|".join(re.escape(s) for s in sorted(SYMBOL_CANDIDATES, key=len, reverse=True))
_CODE_RX = "|".join(sorted(ISO_CURRENCIES))
LOW_CONFIDENCE = 0.60
MARGIN = 1.5                     # a candidate must beat the runner-up (different value) by this score to be chosen

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


# ---------------------------------------------------------------- typed token finders
DATE_RES = (
    re.compile(r"(?<![\d/])(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?![\d/])"),
    re.compile(r"(?<![\d/])(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?![\d/])"),
    re.compile(r"\b(" + _MON + r")\.?[ \-]+(\d{1,2})(?:st|nd|rd|th)?,?[ \-]+(\d{4})\b", re.I),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[ \-]+(" + _MON + r")\.?,?[ \-]+(\d{4})\b", re.I),
)
_NUM_PLAIN = r"\d{1,3}(?:,\d{3})+(?:[.,]\d{2})?|\d{1,3}(?:\.\d{3})+,\d{2}|\d+[.,]\d{2}|\d+"
_NUM_SPACED = r"\d{1,3}(?:[ \u00a0]\d{3})+[.,]\d{2}"      # '1 200,00' -- only accepted when a currency code/symbol is attached
_MONEY = (r"(?:(?<![A-Za-z])(?P<pcc>" + _CODE_RX + r")\s?)?(?:(?<![A-Za-z])(?P<sym>" + _SYM_RX + r"))?\s?(?P<neg>-)?\s?(?P<num>{num})(?![\d])"
          r"(?:\s?(?:(?P<cc>" + _CODE_RX + r")(?![A-Za-z])|(?P<ssym>" + _SYM_RX + r")))?")
MONEY_RE = re.compile(_MONEY.replace("{num}", _NUM_SPACED + "|" + _NUM_PLAIN))
MONEY_RE_PLAIN = re.compile(_MONEY.replace("{num}", _NUM_PLAIN))
ID_RE = re.compile(r"(?<![\w\-/])(?=[A-Za-z0-9\-_/]*\d)[A-Za-z0-9][A-Za-z0-9\-_/]{0,28}(?![\w\-/])")
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")


def _date_tuple(m, idx):
    g = m.groups()
    if idx == 0:
        return int(g[0]), int(g[1]), int(g[2]), False
    if idx == 1:
        a, b, y = int(g[0]), int(g[1]), int(g[2]); y += 2000 if y < 100 else 0
        if a > 12:
            return y, b, a, False           # DD/MM
        return y, a, b, (b <= 12 and a != b)   # MM/DD (US default); ambiguous when day and month could swap
    if idx == 2:
        return int(g[2]), MONTHS[g[0][:3].lower()], int(g[1]), False
    return int(g[2]), MONTHS[g[1][:3].lower()], int(g[0]), False


def find_dates(text):
    """[{raw, iso, ambiguous, start, end}] for every date-like expression; month names and ISO are never ambiguous."""
    out, taken = [], []
    for idx, rx in enumerate(DATE_RES):
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            y, mo, d, amb = _date_tuple(m, idx)
            if not (1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
                continue
            taken.append((m.start(), m.end()))
            out.append({"raw": m.group(0), "iso": f"{y:04d}-{mo:02d}-{d:02d}", "ambiguous": amb, "start": m.start(), "end": m.end()})
    return sorted(out, key=lambda x: x["start"])


def _amount_string(num, negative=False):
    n = re.sub(r"[ \u00a0]", "", num)
    if "," in n and "." in n:
        dec = "," if n.rfind(",") > n.rfind(".") else "."
    elif "," in n or "." in n:
        sep = "," if "," in n else "."
        tail = n.split(sep)[-1]
        dec = sep if (n.count(sep) == 1 and len(tail) in (1, 2)) else None
    else:
        dec = None
    if dec:
        whole, frac = n.rsplit(dec, 1)
        whole = re.sub(r"[.,]", "", whole)
    else:
        whole, frac = re.sub(r"[.,]", "", n), "00"
    return f"{'-' if negative else ''}{int(whole)}.{(frac + '00')[:2]}"


def find_money(text, dates=None):
    """[{amount, token, raw, start, end}] -- values that look like money (decimals, thousands separators or a symbol). Dates are excluded.
    OCR often splits decimals ('1050. 00'); that is repaired first, keeping character offsets stable."""
    text = re.sub(r"(?<=\d)([.,]) (?=\d{2}(?!\d))", r"\1", text)      # '1050. 00' -> '1050.00' (offsets shift by the removed space only after the match)
    dates = dates if dates is not None else find_dates(text)
    out = []
    matches = []
    for m in MONEY_RE.finditer(text):
        if re.search(r"\s", m.group("num")) and not (m.group("pcc") or m.group("sym") or m.group("cc") or m.group("ssym")):
            matches += list(MONEY_RE_PLAIN.finditer(text, m.start(), m.end()))     # space-grouping without a currency marker: likely two columns
        else:
            matches.append(m)
    for m in matches:
        start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
        if any(d["start"] <= start < d["end"] for d in dates):
            continue
        num, sym = m.group("num"), m.group("sym") or m.group("ssym")
        if not (re.search(r"[.,]\d{2}$", num) or "," in num or (sym and len(num) >= 2)):
            continue
        if PHONE_RE.match(text[start:m.end() + 8]):
            continue
        out.append({"amount": _amount_string(num, bool(m.group("neg"))), "token": m.group("pcc") or m.group("cc") or sym, "raw": m.group(0).strip(), "start": start, "end": m.end()})
    return out


def parse_amount(raw):
    """-> {amount:'1234.56', token:'EUR'|'$'|None, normalization:str|None} or None (the whole string must be one amount)."""
    s = raw.strip()
    m = re.fullmatch(r"(?:(?P<c1>" + _SYM_RX + r"|[A-Za-z]{3})\s*)?-?\s*(?P<n>\d[\d.,' \u00a0]*\d|\d)(?:\s*(?P<c2>" + _SYM_RX + r"|[A-Za-z]{3}))?", s)
    if not m:
        return None
    n = m.group("n").replace("'", "").replace(" ", "").replace("\u00a0", "")
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
        stripped = whole
        if whole != n:
            note.append("thousands separators removed")
    return {"amount": f"{int(stripped)}.{(frac + '00')[:2]}", "token": tok, "normalization": "; ".join(note) or None}


def normalize_date(raw):
    """ISO date when `raw` is exactly one unambiguous date expression, else None."""
    s = raw.strip()
    ds = find_dates(s)
    if len(ds) == 1 and ds[0]["start"] == 0 and ds[0]["end"] == len(s) and not ds[0]["ambiguous"]:
        return ds[0]["iso"]
    return None


def find_ids(text, min_len=2):
    out = []
    dates, money = find_dates(text), find_money(text)
    phones = [(p.start(), p.end()) for p in PHONE_RE.finditer(text)]
    for m in ID_RE.finditer(text):
        tok = m.group(0).strip("-_/")
        if len(tok) < min_len or not re.search(r"\d", tok):
            continue
        if any(d["start"] <= m.start() < d["end"] for d in dates) or any(a < m.end() and m.start() < b for a, b in phones):
            continue
        if re.fullmatch(r"\d{5}(?:-\d{4})?", tok) and re.search(r"\b[A-Z]{2}\.?\s*$", text[:m.start()]):
            continue                  # a ZIP code after a state abbreviation is an address, not an identifier
        if re.fullmatch(r"\d{1,3}(?:,\d{3})*\.\d{2}", tok) or any(mm["start"] <= m.start() and m.end() <= mm["end"] and re.search(r"[.,]", mm["raw"]) for mm in money):
            continue
        out.append({"value": tok, "start": m.start(), "end": m.end()})
    return out


def iban_valid(v):
    s = re.sub(r"\s", "", v).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return None            # not IBAN-shaped: a plain account number, no checksum applies
    r = (s[4:] + s[:4])
    return int("".join(str(int(c, 36)) for c in r)) % 97 == 1


def _norm_key(v):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(v)).casefold()).strip()


# ---------------------------------------------------------------- lines and geometry
class Line:
    __slots__ = ("page", "idx", "text", "conf", "boxes")

    def __init__(self, page, idx, text, conf, boxes):
        self.page, self.idx, self.text, self.conf, self.boxes = page, idx, text, conf, boxes


def build_lines(pages):
    """[{page, text, line_conf?, line_boxes?}] -> [Line]. Embedded text is an exact text layer (confidence 1.0)."""
    out = []
    for p in pages:
        raw = p["text"].split("\n")
        confs, boxes = p.get("line_conf"), p.get("line_boxes")
        for i, t in enumerate(raw):
            t = t.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
            if t.strip():
                out.append(Line(p["page"], len(out), t, (confs[i] if confs and i < len(confs) else 1.0), (boxes[i] if boxes and i < len(boxes) else None)))
    return out


def _char_to_x(line, start, end):
    """x-range of the words covering line.text[start:end] (Tesseract joins words with single spaces); None without geometry."""
    if not line.boxes:
        return None
    pos, x0, x1 = 0, None, None
    for w, l, r, _ in line.boxes:
        ws, we = pos, pos + len(w)
        if we > start and ws < end:
            x0 = l if x0 is None else min(x0, l); x1 = r if x1 is None else max(x1, r)
        pos = we + 1
    return (x0, x1) if x0 is not None else None


def split_embedded_pages(text):
    """pdftotext separates pages with form feeds (callers pass the raw string, before sanitising)."""
    parts = text.split("\f")
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [{"page": i + 1, "text": t} for i, t in enumerate(parts)] or [{"page": 1, "text": text}]


# ---------------------------------------------------------------- label specs: (regex, weight) -- weight = how reliably the label names the field
LABELS = {
    "invoice_number": [
        (r"\b(?:tax\s+)?inv(?!entory)\w{0,6}\.?\s*(?:(?:no|number|num|nbr|id)\b\.?|n[oe0*&\u00b0\u00ba]{1,2}\.?|#)\s*[:#.\-]?", 10),
        (r"\binvoice\b\s*[:#\-(]*\s*#?(?=\s*[A-Za-z0-9][A-Za-z0-9\-/_]{0,20}\d)", 6),     # 'INVOICE CDS-2609-1188', 'Invoice SFG-SEP26-779' 
        (r"\b(?:document|bill|statement|debit\s+memo|credit\s+memo)\s*(?:no\.?|number|#)\s*[:#.\-]?", 5),
        (r"\b(?:contract|order|reference|ref)\s*(?:no\.?|number|#)\s*[:#.\-]?", 3),
    ],
    "invoice_date": [
        (r"\binv\w{0,6}\s+da[rtl]e\b\s*[:.\-]?", 10),
        (r"\b(?:date\s+(?:of\s+)?(?:invoice|issue|issued)|issue\s+date|billing\s+date|bill\s+date|invoice\s+dated)\b\s*[:.\-]?", 9),
        (r"^[\W_]*da[rtl]e\b(?!\s+(?:entered|due|of\s+birth|modified))\s*[:.\-]?", 4),
    ],
    "due_date": [
        (r"\b(?:due\s+date|payment\s+due(?:\s+date)?|date\s+due|due\s+on|pay(?:ment)?\s+by|due\s+by)\b\s*[:.\-]?", 10),
        (r"\bdue\b\s*[:.\-]?(?=\s*\d|\s*" + _MON + r")", 6),
    ],
    "amount": [
        (r"\b(?:(?:amount|amt\.?)\s+due|balance\s+due|net\s+due|net\s+amount\s+due|please\s+pay|pay\s+this\s+amount|amount\s+payable|due\s+this\s+invoice|balance\s+forward\s+due|pay\s+this\s+invoice)\b\s*(?:\(usd\)|in\s+usd)?\s*[:.\-=]?", 10),
        (r"\b(?:total\s+amount\s+due|total\s+due|total\s+payable)\b\s*(?:\(usd\)|in\s+usd)?\s*[:.\-=]?", 9),
        (r"\b(?:invoice\s+total|total\s+invoice|grand\s+total|total\s+amount|total\s+charges|total\s+this\s+invoice|total\s+for\s+this\s+invoice|invoice\s+amount|total\s+invoice\s+amount)\b\s*[:.\-]?", 7),
        (r"\b(?:net|gross)\b(?!\s*(?:\d+\s*(?:days?|d)\b|terms|weight|of))\s*[:.\-]?", 6),
        (r"(?<![\w])(?<!sub)(?<!sub )(?<!sub-)(?<!vendor )(?<!tax )(?<!sales tax )(?<!page )(?<!line )(?<!item )(?<!weekly )total(?!\s*(?:spots|tax|vat|qty|quantity|hours|units|items|weight|pages|lines))\b\s*[:.\-]?", 4),
        (r"\bbalance\b\s*[:.\-]?", 3),
    ],
}
LABEL_RE = {k: [(re.compile(rx, re.I), w) for rx, w in v] for k, v in LABELS.items()}
VENDOR_LABELS = ((re.compile(r"\b(?:make\s+(?:all\s+)?(?:checks?|cheques?)\s+payable\s+to|remit(?:tance)?\s+to|pay\s+to|payable\s+to|please\s+remit\s+to|send\s+(?:payment|remittance)\s+to)\b\s*[:.\-]?", re.I), 8),
                 (re.compile(r"\b(?:supplier|vendor|seller|billed\s+from|bill\s+from|invoice\s+from|issued\s+by|from)\b(?:\s*:|\s*[\-\u2013](?!\w))", re.I), 7))     # 'Vendor:' / 'Vendor -', never 'vendor-master' 
NAME_STOP = re.compile(r"\b(?:invoice|date|page|bill\s*to|ship\s*to|sold\s*to|attn|attention|tel|telephone|fax|phone|www|http|account|number|no\.|total|amount|terms|description|quantity|qty|"
                       r"customer|statement|remit|payable|po\s*box|p\.o\.|suite|ste\.|floor|street|avenue|ave|blvd|road|drive|lane|federal|tax\s+id|balance|due|please|thank|email|e-mail|re:|to:|from:|copy|original)\b", re.I)
LETTERHEAD_MIN = 3.0            # minimum name strength for an un-labelled (letterhead) vendor
REQUIRE_CORROBORATION = False   # letterhead must be repeated elsewhere on the document (remit-to, footer, signature)
STATE_ZIP = re.compile(r",?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")


# ---------------------------------------------------------------- candidate generation
def _next_lines(lines, i, n=2):
    return [ln for ln in lines[i + 1:i + 1 + n] if ln.page == lines[i].page]


def _is_prose(text):
    """A sentence (many lowercase words) rather than a header cell or a short label."""
    return len(re.findall(r"\b[a-z]{3,}\b", text)) >= 5


def _label_matches(line, specs):
    """All label matches in a line as (start, end, weight); overlapping matches keep the stronger."""
    hits = []
    for rx, w in specs:
        for m in rx.finditer(line.text):
            hits.append((m.start(), m.end(), w))
    hits.sort(key=lambda h: (-h[2], h[0]))
    kept = []
    for s, e, w in hits:
        if not any(s < ke and ks < e for ks, ke, _ in kept):
            kept.append((s, e, w))
    return sorted(kept)


def _typed_tokens(field, text, min_len=2):
    if field in ("invoice_date", "due_date"):
        return [(d["start"], d["end"], d) for d in find_dates(text)]
    if field == "amount":
        return [(m["start"], m["end"], m) for m in find_money(text)]
    return [(m["start"], m["end"], m) for m in find_ids(text, min_len)]


def _candidates(field, lines):
    """All typed candidates for an id/date/amount field."""
    out = []
    specs = LABEL_RE[field]
    for li, line in enumerate(lines):
        hits = _label_matches(line, specs)
        for k, (s, e, w) in enumerate(hits):
            limit = hits[k + 1][0] if k + 1 < len(hits) else len(line.text)
            toks = [t for t in _typed_tokens(field, line.text, 1 if (field == "invoice_number" and w >= 10) else 2) if e <= t[0] < limit]
            method, chosen, src_line = "inline", None, line
            if toks:
                chosen = toks[0]
            elif _is_prose(line.text):
                continue                  # 'Please refer to our invoice number when remitting' is a sentence, not a column header
            else:
                lx = _char_to_x(line, s, e)
                for nl in _next_lines(lines, li, 2):
                    ntoks = _typed_tokens(field, nl.text)
                    if not ntoks:
                        continue
                    if lx and nl.boxes:
                        peers = [_char_to_x(line, hs, he) for hs, he, _ in hits]
                        best = None
                        for t in ntoks:
                            tx = _char_to_x(nl, t[0], t[1])
                            if not tx:
                                continue
                            tc = (tx[0] + tx[1]) / 2
                            dists = [min(abs((px[0] + px[1]) / 2 - tc), abs(px[0] - tx[0])) for px in peers if px]
                            mine = min(abs((lx[0] + lx[1]) / 2 - tc), abs(lx[0] - tx[0]))
                            # the token belongs to this label only if no other label of this field on the header line is closer
                            if mine <= 450 and mine <= min(dists) + 1e-9 and (dists.count(mine) <= 1):
                                best = (mine, t) if best is None or mine < best[0] else best
                        if best:
                            chosen, src_line, method = best[1], nl, "aligned"
                            break
                    elif not lx and ntoks[0][0] <= 2 and e >= len(line.text.rstrip()) - 1:
                        chosen, src_line, method = ntoks[0], nl, "below"      # plain text: value on the very next line, label at line end
                        break
                if chosen is None:
                    continue
            score = w + 2.0 * min(line.conf, src_line.conf) + {"inline": 2.0, "aligned": 1.0, "below": 0.5}[method]
            out.append({"tok": chosen[2], "score": score, "page": src_line.page, "line": src_line.idx, "evidence": src_line.text.strip(),
                        "conf": min(line.conf, src_line.conf), "label": line.text[s:e].strip(" :.-#"), "method": method, "weight": w})
    return out


RANGE_CUE = re.compile(r"\b(?:due|period|through|thru|to|from|start|end|entered|modified|expir\w*|effective|flight|service|ship|deliver\w*|order(?:ed)?|through)\b|\d\s*-\s*\d{1,2}/", re.I)


def _fallback_dates(lines):
    """Unlabelled date near the top of the first page: a line that is (almost) only a date, or sits beside the invoice number."""
    out = []
    for ln in lines:
        if ln.idx > 24 or ln.page != lines[0].page or RANGE_CUE.search(ln.text):
            continue
        ds = find_dates(ln.text)
        if len(ds) != 1:
            continue
        rest = (ln.text[:ds[0]["start"]] + " " + ln.text[ds[0]["end"]:]).strip()
        standalone = len(re.sub(r"[\W_]", "", rest)) <= 3
        beside_invoice = bool(re.search(r"\binv\w{0,6}\b", ln.text, re.I)) or bool(re.search(r"\bdate\b", ln.text, re.I))
        if standalone or beside_invoice:
            out.append({"tok": ds[0], "score": 3.0 + 2.0 * ln.conf + (1.0 if standalone else 0.0) + (2.0 if beside_invoice else 0.0) - 0.05 * ln.idx,
                        "page": ln.page, "line": ln.idx, "evidence": ln.text.strip(), "conf": ln.conf, "label": "unlabelled date", "method": "unlabelled", "weight": 3})
    return out


def _pick(cands, keyfn, margin=MARGIN):
    """Return (winner, conflict_list). Winner is None when the top candidates disagree within `margin`."""
    if not cands:
        return None, []
    ranked = sorted(cands, key=lambda c: -c["score"])
    top = ranked[0]
    rivals = [c for c in ranked[1:] if keyfn(c) != keyfn(top) and top["score"] - c["score"] < margin]
    if rivals:
        seen, conflict = set(), []
        for c in [top] + rivals:
            if keyfn(c) not in seen:
                seen.add(keyfn(c)); conflict.append(c)
        return None, conflict
    return top, []


def _mk(value, source, page, conf, evidence, norm=None, **extra):
    d = {"found": True, "value": value, "source": source, "page": page, "confidence": round(conf, 3), "evidence_text": evidence.strip()[:160],
         "normalization_applied": norm, "ambiguous": False, "candidates": []}
    d.update(extra)
    return d


def _empty():
    return {"found": False, "value": None, "source": None, "page": None, "confidence": None, "evidence_text": None,
            "normalization_applied": None, "ambiguous": False, "candidates": []}


def _ambiguous(cands, valfn, **extra):
    d = dict(_empty(), ambiguous=True, candidates=[valfn(c) for c in cands])
    d.update(extra)
    return d


# ---------------------------------------------------------------- document type (scored)
def classify_document(lines):
    """Score-based typing. Heading-like lines (short, near the top) carry the most weight.
    Returns dict(value, ambiguous, candidates, page, confidence, evidence_text)."""
    score, ev = {}, {}

    def add(kind, w, ln):
        score[kind] = score.get(kind, 0) + w
        ev.setdefault(kind, ln)
    for ln in lines:
        t = ln.text.strip()
        low = t.lower()
        head = ln.idx < 14 and len(t) <= 48
        if re.search(r"\bcredit\s+(?:note|memo)\b", low) and head:
            add("credit_note", 6, ln)
        if re.search(r"\bpurchase\s+ord(?:er|e)?\b", low) and (head or re.search(r"\bpurchase\s+order\s*(?:no|number|#)", low) or len(t) <= 60):
            add("purchase_order", 6 if head else 3, ln)
        if re.search(r"\border\s+(?:worksheet|printout|confirmation|form)\b|\bscheduling\s+order\b|\bmake\s*good\b|\binsertion\s+order\b", low):
            add("order", 5, ln)
        if re.search(r"thank\s+you\s+for\s+your\s+payment|\bpayment\s+received\b|\breceived\s+with\s+thanks\b", low):
            add("receipt", 5, ln)
        if head and re.search(r"\b(?:sales\s+order|order\s+confirmation|insertion\s+order|work\s+order)\b", low):
            add("order", 6, ln)
        if head and len(t) <= 30 and re.search(r"\b(?:estimate|quotation|quote|proforma|pro\s+forma)\b", low):
            add("estimate", 6, ln)
        if head and len(t) <= 40 and re.search(r"\b(?:contract|agreement)\b", low) and not re.search(r"contract\s*(?:no\b|number\b|#)", low):
            add("contract", 5, ln)
        if head and len(t) <= 30 and re.search(r"\breceipt\b", low):
            add("receipt", 6, ln)
        if re.search(r"\b(?:tax\s+)?invoice\b", low):
            if head and len(t) <= 34 and not re.search(r"invoice\s*(?:no|number|#|date)", low):
                add("invoice", 6, ln)              # a standalone INVOICE heading
            elif re.search(r"\binvoice\s*(?:no\.?|number|#|date|total|amount)", low):
                add("invoice", 2, ln)              # invoice-labelled field
            else:
                add("invoice", 1, ln)
        if re.search(r"\b(?:amount|balance|total)\s+due\b|\bplease\s+pay\b|\bremit(?:tance)?\b|\bbill(?:ed)?\s+to\b", low):
            add("invoice", 1, ln)
        if re.search(r"contract\s*(?:no\b|number\b|#)|\border\s*(?:no\b\.?|number\b|#)", low) and re.search(r"\bstart\s+date\b|\bend\s+date\b", low):
            add("order", 3, ln)
    if not score:
        return {"value": "unknown", "ambiguous": False, "candidates": [], "page": None, "confidence": None, "evidence_text": None}
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    (k1, s1) = ranked[0]
    rival = [k for k, s in ranked[1:] if s >= s1 - 2 and s >= 3]
    if s1 < 3:                     # incidental mentions ('...with invoice', an Invoices@ address) are not evidence of a document type
        return {"value": "unknown", "ambiguous": False, "candidates": [], "page": None, "confidence": None, "evidence_text": None}
    if rival:
        return {"value": None, "ambiguous": True, "candidates": [k1] + rival, "page": None, "confidence": None, "evidence_text": None}
    ln = ev[k1]
    return {"value": k1, "ambiguous": False, "candidates": [], "page": ln.page, "confidence": ln.conf, "evidence_text": ln.text.strip()}


# ---------------------------------------------------------------- vendor
def _clean_name(t):
    t = re.sub(r"^[^A-Za-z0-9(]+", "", t.strip())
    t = re.sub(r"\s{2,}.*$", "", t)
    t = re.sub(r"[\s,;:|_\-—]+$", "", t)
    return t.strip()


def _name_ok(t):
    if not (3 <= len(t) <= 70):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 3 or letters / len(t) < 0.6 or NAME_STOP.search(t) or STATE_ZIP.search(t) or "@" in t:
        return False
    if len(re.findall(r"\b[A-Za-z]\b", t)) > 2 or PHONE_RE.search(t):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'&.\-]*", t)
    return len(words) >= 1 and sum(len(w) >= 3 for w in words) >= 1


ENTITY_WORDS = re.compile(r"\b(?:inc|incorporated|llc|l\.l\.c|llp|l\.l\.p|ltd|limited|co|company|corp|corporation|group|university|college|institute|associates|partners|broadcasting|"
                          r"communications|media|radio|television|network|services|systems|solutions|industries|enterprises|foundation|laboratories|labs|technologies|studios|trust|bank|"
                          r"holdings|consulting|design|printing|publishing|agency|advertising|productions|entertainment|cable|wireless|marketing|research|management|gmbh|ag|sa|plc)\b\.?", re.I)
CALLSIGN = re.compile(r"^[KW][A-Z]{2,3}(?:\s*[-\u2013]?\s*(?:FM|AM|TV|DT|HD))?(?:\s*/\s*(?:FM|AM|TV))*$")
HEADING_WORDS = re.compile(r"\b(?:order|orders|printout|scheduling|schedule|makegood|advisory|form|member|signed|sales\s+office|program\s+logs?|client|agency\s+job|product|estimate|proposal|summary|report|"
                           r"confirmation|agreement|receipt|voucher|remittance|address|correspondence|billing|customer|instructions?|notes?|comments?|message|expenses?|services\s+rendered)\b", re.I)
CUSTOMER_CUE = re.compile(r"\b(?:bill(?:ed)?\s*to|ship(?:ped)?\s*to|sold\s*to|customer|client|advertiser|agency|attn|attention|c/o|deliver\s*to|invoice\s*to|to\s*:|prepared\s+for|for\s*:)", re.I)
FUNCTION_WORDS = {"the", "from", "and", "will", "be", "to", "all", "you", "your", "our", "for", "with", "this", "that", "are", "is", "on", "by", "at", "or", "as", "in"}


def _name_strength(t, occurrences):
    """How much a bare line looks like the issuing company's name. 0 means: do not use."""
    words = re.findall(r"[A-Za-z][A-Za-z'&.\-]*", t)
    if not words or HEADING_WORDS.search(t):
        return 0.0
    low = [w.lower() for w in words]
    if sum(w in FUNCTION_WORDS for w in low) / len(low) > 0.34 and not ENTITY_WORDS.search(t):
        return 0.0
    strength = 0.0
    if CALLSIGN.match(t.strip()):
        strength += 3.0
    if ENTITY_WORDS.search(t):
        strength += 3.0
    if len(words) >= 2 and (t.isupper() or t.istitle()):
        strength += 1.5
    strength += 2.0 * min(occurrences, 2)
    if len(words) == 1 and not CALLSIGN.match(t.strip()) and occurrences < 1:
        return 0.0
    return strength


def _vendor_candidates(lines, allow_station=False):
    out = []
    for li, ln in enumerate(lines):
        for rx, w in VENDOR_LABELS:
            m = rx.search(ln.text)
            if not m:
                continue
            rest = _clean_name(ln.text[m.end():])
            src = ln
            if not _name_ok(rest) or re.search(r"\b(?:will|shall|must|should)\b", rest, re.I):
                nxt = _next_lines(lines, li, 1)
                rest, src = (_clean_name(nxt[0].text), nxt[0]) if nxt else ("", ln)
            if _name_ok(rest) and not re.search(r"\b(?:will|shall|must|should)\b", rest, re.I) and not HEADING_WORDS.search(rest):
                out.append({"value": rest, "score": w + 2.0 * min(ln.conf, src.conf), "page": src.page, "evidence": src.text.strip(),
                            "conf": min(ln.conf, src.conf), "method": "label:" + m.group(0).strip(" :.-").lower()})
            break
    # a call sign after / under a 'Station' label (broadcast orders and invoices)
    for li, ln in enumerate(lines if allow_station else []):
        m = re.search(r"\bstation\b\s*[:\-]?", ln.text, re.I)
        if not m or _is_prose(ln.text):
            continue
        sign = re.search(r"\b([KW][A-Z]{2,3}(?:\s*[-\u2013]\s*(?:FM|AM|TV|DT))?)\b", ln.text[m.end():])
        src = ln
        if not sign:
            lx = _char_to_x(ln, m.start(), m.end())
            for nl in _next_lines(lines, li, 1):
                best = None
                for sg in re.finditer(r"\b([KW][A-Z]{2,3}(?:\s*[-\u2013]\s*(?:FM|AM|TV|DT))?)\b", nl.text):
                    tx = _char_to_x(nl, sg.start(), sg.end())
                    if lx and tx:
                        dist = abs((lx[0] + lx[1]) / 2 - (tx[0] + tx[1]) / 2)
                        best = (dist, sg) if dist <= 450 and (best is None or dist < best[0]) else best
                if best:
                    sign, src = best[1], nl
        if sign:
            name = re.sub(r"\s+", "", sign.group(1)).replace("\u2013", "-")
            out.append({"value": name, "score": 2.5, "page": src.page, "evidence": src.text.strip(),
                        "conf": min(ln.conf, src.conf), "method": "label:station"})
            break
    top = [ln for ln in lines if ln.page == lines[0].page][:10] if lines else []
    blob = [re.sub(r"[^a-z0-9]", "", x.text.lower()) for x in lines]
    zone = set()              # lines that belong to the customer / agency / recipient block are never the issuing vendor
    for k, x in enumerate(lines):
        if CUSTOMER_CUE.search(x.text):
            zone.update(range(k, k + 4))
    for ln in top:
        t = _clean_name(ln.text)
        if not _name_ok(t) or ln.conf < 0.6 or ln.idx in zone or re.search(r"\bc/o\b", ln.text, re.I) or re.search(r"\d{5}", t) or "#" in t:
            continue
        key = re.sub(r"[^a-z0-9]", "", t.lower())
        occ = sum(1 for k, b in enumerate(blob) if k != ln.idx and key and key in b)
        strength = _name_strength(t, occ)
        if strength >= LETTERHEAD_MIN and (occ >= 1 or not REQUIRE_CORROBORATION or CALLSIGN.match(t.strip())):
            out.append({"value": t, "score": 2.0 + strength + 2.0 * ln.conf - 0.15 * ln.idx, "page": ln.page, "evidence": ln.text.strip(), "conf": ln.conf, "method": "letterhead"})
    return out


def _same_name(a, b):
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return bool(ta and tb and (ta <= tb or tb <= ta))


# ---------------------------------------------------------------- extraction
def extract_fields(pages, source, vendor=None):
    """pages: [{page, text, line_conf?, line_boxes?}]. vendor: optional independent trusted record dict with `canonical_currency`
    (used ONLY to confirm a currency symbol). Returns (fields, warnings)."""
    assert source in SOURCES
    lines = build_lines(pages)
    fields, warnings = {k: _empty() for k in FIELD_KEYS}, []
    used = []

    dt = classify_document(lines)
    if dt["ambiguous"]:
        fields["document_type"] = dict(_empty(), ambiguous=True, candidates=dt["candidates"])
        warnings.append("document type is ambiguous: " + " vs ".join(dt["candidates"]))
    elif dt["value"] != "unknown":
        fields["document_type"] = _mk(dt["value"], source, dt["page"], dt["confidence"], dt["evidence_text"])
        used.append((dt["page"], dt["evidence_text"]))

    # ---- reference number
    win, conflict = _pick(_candidates("invoice_number", lines), lambda c: _norm_key(c["tok"]["value"]))
    if win:
        fields["invoice_number"] = _mk(win["tok"]["value"], source, win["page"], win["conf"], win["evidence"],
                                       None if win["weight"] >= 5 else f"reference number taken from '{win['label']}'")
        used.append((win["page"], win["evidence"]))
    elif conflict:
        fields["invoice_number"] = _ambiguous(conflict, lambda c: c["tok"]["value"])
        warnings.append("invoice number has conflicting values: " + " | ".join(c["tok"]["value"] for c in conflict))

    # ---- dates
    for key in ("invoice_date", "due_date"):
        cands = _candidates(key, lines)
        if key == "invoice_date" and not cands:
            cands = _fallback_dates(lines)
        win, conflict = _pick(cands, lambda c: c["tok"]["iso"])
        if win:
            tok = win["tok"]
            if tok["ambiguous"]:
                fields[key] = _mk(tok["raw"], source, win["page"], win["conf"], win["evidence"])
                warnings.append(f"{key.replace('_', ' ')} '{tok['raw']}' has an ambiguous format (day/month order); kept as written")
            else:
                fields[key] = _mk(tok["iso"], source, win["page"], win["conf"], win["evidence"],
                                  f"date converted to ISO 8601 from '{tok['raw']}'" if tok["iso"] != tok["raw"] else None)
            used.append((win["page"], win["evidence"]))
        elif conflict:
            fields[key] = _ambiguous(conflict, lambda c: c["tok"]["iso"])
            warnings.append(f"{key.replace('_', ' ')} has conflicting values: " + " | ".join(c["tok"]["iso"] for c in conflict))

    # ---- amount: the highest label tier that yields candidates decides; same-tier disagreement stays ambiguous
    amts = _candidates("amount", lines)
    if amts:
        tier = max(c["weight"] for c in amts)
        top = [c for c in amts if c["weight"] == tier]
        distinct = {}
        for c in top:
            c["amt"] = c["tok"]["amount"]
            distinct.setdefault(c["amt"], c)
        if len(distinct) == 1:
            best = next(iter(distinct.values()))
            pa = parse_amount(best["tok"]["raw"].lstrip("-"))
            fields["amount"] = _mk(best["amt"], source, best["page"], best["conf"], best["evidence"], (pa or {}).get("normalization"))
            used.append((best["page"], best["evidence"]))
            lower = sorted({c["tok"]["amount"] for c in amts if c["weight"] < tier} - {best["amt"]})
            if lower and tier >= 9 and len(lower) <= 3:
                warnings.append("other totals on the document (" + ", ".join(lower) + ") differ from the amount due " + best["amt"] + "; the amount due was used")
        else:
            fields["amount"] = _ambiguous(list(distinct.values()), lambda c: c["amt"])
            warnings.append("amount has conflicting values: " + " | ".join(sorted(distinct)))
        fields["currency"] = _currency(top, lines, source, vendor, warnings)
        if fields["currency"]["found"]:
            used.append((fields["currency"]["page"], fields["currency"]["evidence_text"]))
    if not fields["currency"]["found"] and not fields["currency"]["ambiguous"]:
        code = _explicit_currency(lines)
        if code and code[0] != "CONFLICT":
            fields["currency"] = _mk(code[0], source, code[1].page, code[1].conf, code[1].text)
            used.append((code[1].page, code[1].text))
        elif code:
            fields["currency"] = dict(_empty(), ambiguous=True, candidates=code[2])
            warnings.append("conflicting currency codes: " + ", ".join(code[2]))

    # ---- supplier / vendor
    vend = _vendor_candidates(lines, allow_station=(dt["value"] != "invoice"))
    if vend:
        ranked = sorted(vend, key=lambda c: -c["score"])
        best = ranked[0]
        rivals = [c for c in ranked[1:] if not _same_name(c["value"], best["value"]) and best["score"] - c["score"] < MARGIN]
        if rivals:
            fields["supplier_name"] = dict(_empty(), ambiguous=True, candidates=[best["value"]] + [c["value"] for c in rivals][:3])
            warnings.append("supplier name has conflicting candidates: " + " | ".join(fields["supplier_name"]["candidates"]))
        else:
            if best["method"] == "letterhead":
                note = "taken from the letterhead (first prominent line)"
            elif "from" in best["method"] or best["method"].split(":", 1)[-1] in ("supplier", "vendor", "seller"):
                note = None
            else:
                note = f"taken from the '{best['method'].split(':', 1)[-1]}' block"
            fields["supplier_name"] = _mk(best["value"], source, best["page"], best["conf"], best["evidence"], note,
                                          basis="letterhead" if best["method"] == "letterhead" else "label")
            used.append((best["page"], best["evidence"]))

    # ---- beneficiary (labelled lines only; rare in this corpus)
    for key, rx in (("beneficiary_name", re.compile(r"^\s*(?:beneficiary(?:\s+name)?|account\s+name|account\s+holder|payee)\s*[:\-]\s*(\S(?:(?!\s{3}).){1,78}?)\s*(?:\s{3,}\S.*)?$", re.I)),
                    ("beneficiary_account", re.compile(r"^\s*(?:iban|bank\s+account|beneficiary\s+account|(?:electronic\s+|approved\s+)?beneficiary\s+id|account\s*(?:no\.?|number|#))\s*[:\-]?\s*((?=\S*\d)[A-Za-z0-9][A-Za-z0-9_\-]*(?: [A-Za-z0-9_\-]+)*)\s*(?:\s{3,}\S.*)?$", re.I))):
        vals = {}
        for ln in lines:
            m = rx.match(ln.text)
            if m:
                vals.setdefault(_norm_key(m.group(1)), (m.group(1).strip(), ln))
        if len(vals) == 1:
            v, ln = next(iter(vals.values()))
            fields[key] = _mk(v, source, ln.page, ln.conf, ln.text)
            if key == "beneficiary_account":
                ok = iban_valid(v)
                if ok is False:
                    warnings.append("beneficiary account looks like an IBAN but fails its checksum (possible OCR error)")
                    fields[key]["normalization_applied"] = "IBAN checksum FAILED"
                elif ok:
                    fields[key]["normalization_applied"] = "IBAN checksum valid"
            used.append((ln.page, ln.text))
        elif len(vals) > 1:
            fields[key] = dict(_empty(), ambiguous=True, candidates=[v[0] for v in vals.values()])
            warnings.append(f"{key.replace('_', ' ')} has conflicting values")

    seen, excerpt = set(), []
    for pg, ev in used:
        if ev and (pg, ev) not in seen:
            seen.add((pg, ev)); excerpt.append(ev.strip())
    fields["relevant_source_text"] = ({"found": True, "value": "\n".join(excerpt[:14]), "source": source, "page": None, "confidence": None, "evidence_text": None,
                                       "normalization_applied": None, "ambiguous": False, "candidates": []} if excerpt else _empty())
    return fields, warnings


CURRENCY_STATEMENT = re.compile(r"(?:currency|all\s+(?:amounts?|prices?)\s+(?:are\s+)?(?:in|shown\s+in|stated\s+in))\s*[:\-]?\s*\(?\b(" + "|".join(sorted(ISO_CURRENCIES)) + r")\b", re.I)


def _explicit_currency(lines):
    """An explicit statement ('Currency: CHF', 'All amounts in USD'). Returns (code, Line) | ('CONFLICT', None, codes) | None."""
    codes = {}
    for ln in lines:
        m = CURRENCY_STATEMENT.search(ln.text)
        if m:
            codes.setdefault(m.group(1).upper(), ln)
    if len(codes) == 1:
        c, ln = next(iter(codes.items()))
        return c, ln
    if len(codes) > 1:
        return "CONFLICT", None, sorted(codes)
    return None


def _currency(top, lines, source, vendor, warnings):
    """Currency from the tokens attached to the chosen amount lines, or an explicit statement. Never inferred."""
    codes, symbols = {}, {}
    for c in top:
        tok = c["tok"]["token"]
        if tok in SYMBOL_CANDIDATES and len(SYMBOL_CANDIDATES[tok]) == 1:     # e.g. '€', '£', 'US$': the symbol itself names one currency
            codes.setdefault(SYMBOL_CANDIDATES[tok][0], dict(c, sym_note=f"symbol '{tok}' denotes {SYMBOL_CANDIDATES[tok][0]}"))
        elif tok in SYMBOL_CANDIDATES:
            symbols.setdefault(tok, c)
        elif tok in ISO_CURRENCIES:
            codes.setdefault(tok, c)
    stmt = _explicit_currency(lines)
    if stmt and stmt[0] != "CONFLICT":
        codes.setdefault(stmt[0], {"page": stmt[1].page, "conf": stmt[1].conf, "evidence": stmt[1].text})
    if len(codes) > 1:
        warnings.append("conflicting currency codes: " + ", ".join(sorted(codes)))
        return dict(_empty(), ambiguous=True, candidates=sorted(codes))
    if len(codes) == 1:
        code, c = next(iter(codes.items()))
        return _mk(code, source, c["page"], c["conf"], c["evidence"], c.get("sym_note"))
    if symbols:
        sym, c = next(iter(symbols.items()))
        cands = SYMBOL_CANDIDATES[sym]
        canon = (vendor or {}).get("canonical_currency")
        if canon and canon in cands:
            f = _mk(canon, source, c["page"], c["conf"], c["evidence"], f"symbol '{sym}' resolved to {canon} by the trusted vendor record (canonical currency)")
            f["confirmed_by"] = "TRUSTED_VENDOR_RECORD"
            return f
        same = {}      # the same amount written elsewhere with an explicit ISO code ('USD 11,575.00') settles the symbol
        amt = c["tok"]["amount"]
        for ln in lines:
            for mo in find_money(ln.text):
                if mo["amount"] == amt and mo["token"] in cands:
                    same.setdefault(mo["token"], ln)
        if len(same) == 1:
            code, ln = next(iter(same.items()))
            f = _mk(code, source, ln.page, ln.conf, ln.text, f"symbol '{sym}' resolved to {code}: the same amount is stated with the code {code} in the document")
            f["confirmed_by"] = "DOCUMENT_ISO_CODE_SAME_AMOUNT"
            return f
        warnings.append(f"currency symbol '{sym}' is ambiguous ({', '.join(cands)}) and no trusted vendor record confirms a currency")
        return dict(_empty(), ambiguous=True, candidates=list(cands), page=c["page"], evidence_text=c["evidence"][:160], source=source)
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
    if not dtype["found"]:         # fail closed: an unrecognised document is never assumed to be a payable invoice
        d["status_hint"], d["payable_schema"] = "REVIEW", False
        d["reasons"].append("document type could not be determined")
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
    if d["missing_required_fields"] and fields["supplier_name"].get("basis") == "letterhead":
        d["status_hint"] = "REVIEW"      # never contact a vendor on the strength of a letterhead guess
        d["reasons"].append("supplier was inferred from the letterhead, so the vendor must be confirmed before requesting clarification; missing: " + ", ".join(d["missing_required_fields"]))
        return d
    if d["missing_required_fields"]:
        d["status_hint"] = "CLARIFY"
        d["reasons"].append("missing required field(s): " + ", ".join(d["missing_required_fields"]))
    return d
