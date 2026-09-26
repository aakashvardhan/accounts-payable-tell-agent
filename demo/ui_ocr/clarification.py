"""Vendor-clarification workflow (simulated). Pure application logic, stdlib only, NO network and NO mail transport.

Authority boundaries:
* The (simulated) agent proposes ONLY a typed action holding trusted identifiers and field names -- never an address, subject or body.
* The application resolves the vendor through the canonical vendor master, obtains the verified accounts-receivable contact,
  builds the recipient from that record, and owns the final template. Text from the PDF/OCR can never reach the recipient.
* The result is an outbox record with send status SIMULATED_NOT_SENT. Nothing in this module can send anything.
Mirrors (does not import) the repo contract: `request_vendor_clarification` follow-up, `TrustedVendorRecord.approved_contact_email` +
`approved_contact_verified`, and the AWAITING_VENDOR_CLARIFICATION state (see src/tell/safety/payment_validation.py and
src/tell/agent/routing_orchestrator.py).
"""
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

CONTRACT = "tell.demo_vendor_master/1.0"
VENDOR_MASTER_LABEL = "DEMO VENDOR MASTER — SYNTHETIC FIXTURE"
CONTACT_FOUND, NO_CONTACT, LOOKUP_FAILED = "CONTACT_FOUND", "NO_CONTACT", "LOOKUP_FAILED"
ACTION = "request_vendor_clarification"
SIMULATED_AGENT_ACTION = "SIMULATED_AGENT_ACTION"
SEND_STATUS = "SIMULATED_NOT_SENT"
CLARIFIABLE = ("invoice_number", "amount", "currency")
ACTION_KEYS = frozenset({"action", "invoice_id", "vendor_id", "missing_fields"})
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){0,4}\.[A-Za-z]{2,}$")
INVOICE_NO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_/]{1,40}$")
PHRASES = {"currency": ("the currency is not stated", "invoice currency"),
           "invoice_number": ("the invoice number could not be read", "invoice number"),
           "amount": ("the amount due is not stated", "total amount due")}


class ClarificationError(Exception):
    pass


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(name):
    s = unicodedata.normalize("NFKC", str(name)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


# ---------------------------------------------------------------- trusted vendor lookup
def resolve_vendor(master_path, supplier_name):
    """Return dict(status, vendor_id, vendor_name, canonical_currency, recipient, reason, source).
    status: CONTACT_FOUND | NO_CONTACT | LOOKUP_FAILED. Any doubt (unreadable file, bad schema, ambiguous match) fails closed."""
    base = {"status": LOOKUP_FAILED, "vendor_id": None, "vendor_name": None, "canonical_currency": None, "recipient": None,
            "reason": None, "source": "TRUSTED_VENDOR_RECORD", "label": VENDOR_MASTER_LABEL}
    try:
        doc = json.loads(Path(master_path).read_text())
        if doc.get("contract") != CONTRACT or not isinstance(doc.get("vendors"), list):
            raise ValueError("unrecognised vendor master format")
        for v in doc["vendors"]:
            if not isinstance(v.get("vendor_id"), str) or not isinstance(v.get("vendor_name"), str) or not isinstance(v.get("approved_contact_verified"), bool):
                raise ValueError("malformed vendor record")
    except (OSError, ValueError, AttributeError, TypeError) as e:
        return dict(base, reason=f"vendor master unavailable or invalid ({type(e).__name__})")
    if not supplier_name:
        return dict(base, status=NO_CONTACT, reason="no supplier name to look up")
    key = _norm(supplier_name)
    hits = [v for v in doc["vendors"] if key in {_norm(v["vendor_name"])} | {_norm(a) for a in v.get("aliases", [])}]
    if not hits:
        return dict(base, status=NO_CONTACT, reason="supplier is not in the vendor master")
    if len(hits) > 1:
        return dict(base, reason="supplier name matches more than one vendor record")
    v = hits[0]
    out = dict(base, vendor_id=v["vendor_id"], vendor_name=v["vendor_name"], canonical_currency=v.get("canonical_currency"))
    email = v.get("approved_contact_email")
    if not email or not v["approved_contact_verified"]:
        return dict(out, status=NO_CONTACT, reason="vendor has no verified accounts-receivable contact")
    if not EMAIL_RE.match(email):
        return dict(out, status=LOOKUP_FAILED, reason="vendor contact address in the vendor master is malformed")
    return dict(out, status=CONTACT_FOUND, recipient=email, reason="verified accounts-receivable contact on the vendor record")


# ---------------------------------------------------------------- typed results / actions
def missing_field_result(invoice_id, missing):
    """The typed missing-field result emitted when extraction succeeded but a required field must come from the vendor."""
    return {"invoice_id": invoice_id, "missing_required_fields": list(missing), "payment_eligible": False,
            "recommended_action": ACTION}


def simulated_agent_action(result, vendor_id):
    """Deterministic replay generator standing in for the live agent. Carries only trusted identifiers + field names."""
    fields = [f for f in result["missing_required_fields"] if f in CLARIFIABLE]
    trace = "act-" + hashlib.sha256((result["invoice_id"] + "|" + vendor_id + "|" + ",".join(fields)).encode()).hexdigest()[:12]
    return {"origin": SIMULATED_AGENT_ACTION, "trace_id": trace,
            "action": {"action": ACTION, "invoice_id": result["invoice_id"], "vendor_id": vendor_id, "missing_fields": fields}}


def validate_action(action, *, invoice_id, resolved_vendor_id, missing_required):
    """Strict schema + authority check. Anything unexpected (extra keys such as an address, a different vendor/invoice,
    fields that are not actually missing) raises ClarificationError; the caller then routes to human document review."""
    if not isinstance(action, dict):
        raise ClarificationError("action must be an object")
    extra = set(action) - ACTION_KEYS
    if extra:
        raise ClarificationError("action contains unauthorised keys: " + ", ".join(sorted(extra)))
    if set(action) != ACTION_KEYS:
        raise ClarificationError("action is missing required keys")
    if action["action"] != ACTION:
        raise ClarificationError("unsupported action")
    if action["invoice_id"] != invoice_id:
        raise ClarificationError("action refers to a different invoice")
    if action["vendor_id"] != resolved_vendor_id:
        raise ClarificationError("action vendor does not match the vendor resolved from the trusted record")
    mf = action["missing_fields"]
    if not isinstance(mf, list) or not mf or len(set(mf)) != len(mf) or any(not isinstance(x, str) for x in mf):
        raise ClarificationError("missing_fields must be a non-empty list of unique field names")
    bad = [f for f in mf if f not in CLARIFIABLE or f not in missing_required]
    if bad:
        raise ClarificationError("requested fields are not clarifiable missing fields: " + ", ".join(bad))
    return list(mf)


# ---------------------------------------------------------------- application-owned template
def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def render_email(invoice_number, amount, missing_fields):
    """Deterministic subject/body from trusted or strictly-validated extracted values only. No free text from the document."""
    no = invoice_number if isinstance(invoice_number, str) and INVOICE_NO_RE.match(invoice_number) and "invoice_number" not in missing_fields else None
    amt = None
    if amount is not None and "amount" not in missing_fields:
        try:
            amt = f"{Decimal(str(amount)):,.2f}"
        except InvalidOperation:
            amt = None
    desc = ("invoice " + no if no else "an invoice") + (f" for {amt}" if amt else "")
    reasons = _join([PHRASES[f][0] for f in missing_fields])
    asks = _join([PHRASES[f][1] for f in missing_fields])
    subject = f"Clarification required for invoice {no}" if no else "Clarification required for a recently received invoice"
    body = (f"We received {desc}, but {reasons}.\n"
            f"Please confirm the {asks} so processing can continue.\n\n"
            "Payment will remain on hold until the missing information is verified.\n"
            "Please do not provide or change banking instructions in this reply.")
    return subject, body


def build_outbox_record(*, invoice_id, lookup, action_envelope, missing_fields, invoice_number, amount):
    """Compose the outbox record. Requires CONTACT_FOUND; the recipient can only come from the trusted lookup."""
    if lookup["status"] != CONTACT_FOUND or not lookup.get("recipient"):
        raise ClarificationError("no verified contact: an email must not be drafted")
    subject, body = render_email(invoice_number, amount, missing_fields)
    return {"invoice_id": invoice_id, "action_trace_id": action_envelope["trace_id"], "recipient": lookup["recipient"],
            "recipient_source": f"TRUSTED_VENDOR_RECORD:{lookup['vendor_id']}", "vendor_id": lookup["vendor_id"],
            "requested_fields": list(missing_fields), "subject": subject, "body": body, "created_at": now(),
            "send_status": SEND_STATUS,
            "provenance": {"action": action_envelope["origin"], "recipient": "TRUSTED_VENDOR_RECORD", "template": "APPLICATION",
                           "invoice_fields": "UPLOADED_PDF"}}
