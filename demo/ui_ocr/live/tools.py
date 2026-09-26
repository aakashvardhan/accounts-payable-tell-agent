"""Typed live tools for the AP agent (worker side). They answer the model's read actions from the *uploaded job's real data*:

  read_email        -> an application-generated intake note (an uploaded PDF has no supplier email)
  read_invoice      -> the fields and text actually extracted from THIS PDF (embedded text or local OCR). Untrusted.
  get_vendor_record -> the canonical vendor record from the trusted vendor master. Trusted.
  search_memory     -> the project's MemoryStore for the vendor (seeded from a labelled demo fixture; origins stay untrusted)

Results are rendered with the project's own `build_observation_message`, i.e. the same tool-result format the model/probe/LoRA
were trained on. Provenance is honest: uploaded-PDF extraction is `uploaded_pdf_extraction` (untrusted), not a dataset annotation.
No lookup here reads a filename or a fixture id: everything derives from the job's stored extraction.
"""
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from live import trusted as T  # noqa: F401  (ensures sys.path for tell.* / enterprise_v2_2)
from enterprise_v2_2 import contract as C  # noqa: E402
from tell.agent.loop_prompts import build_observation_message  # noqa: E402
from tell.agent.memory_prompts import build_search_memory_result_message  # noqa: E402
from tell.agent.tools import ToolError, ToolName, ToolStatus  # noqa: E402
from tell.evaluation.scenario import TrustBoundary  # noqa: E402

MAX_TEXT_CHARS = 3500


class LiveProvenance(BaseModel):
    """Same JSON shape as `tell.evaluation.scenario.OperationalProvenance`, but with an honest `provenance` label."""

    model_config = ConfigDict(frozen=True)

    source_type: str
    source_id: str
    provenance: str
    recorded_at: str
    trust_boundary: TrustBoundary


@dataclass(frozen=True)
class LiveToolResult:
    tool_name: ToolName
    status: ToolStatus
    provenance: LiveProvenance | None
    content: BaseModel | None
    error: ToolError | None


class LiveEmailContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    sender_display_name: str
    sender_address: str
    subject: str
    body: str
    references_docid: str
    references_invoice_number: str | None


class LiveDocumentRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    source_system: str = "uploaded_pdf"


class LiveInvoiceContent(BaseModel):
    """Same field names the corpus's `ReadInvoiceContent` uses, plus the extracted text and honest extraction metadata."""

    model_config = ConfigDict(frozen=True)

    extraction_method: str
    docid: str
    page_count: int
    document_type: str | None
    vendor_name: str | None
    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None
    currency: str | None
    amount_due: str | None
    total_amount_gross: str | None
    purchase_order_numbers: list[str]
    payment_destination: list[str]
    extraction_notes: list[str]
    document_text: str
    document_reference: LiveDocumentRef


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _val(fields, key):
    f = (fields or {}).get(key)
    return f["value"] if f and f.get("found") else None


def _notes(ex):
    """Ambiguous / missing evidence stated plainly, so the model sees exactly what the extractor could and could not establish."""
    notes = []
    for k, f in ((ex or {}).get("fields") or {}).items():
        if k == "relevant_source_text":
            continue
        if f.get("ambiguous"):
            notes.append(f"{k} is ambiguous; candidates: {', '.join(map(str, f.get('candidates', [])[:6]))}")
    notes += list((ex or {}).get("warnings") or [])[:6]
    return notes


class LiveTools:
    def __init__(self, job, run_id, ctx):
        """job: dict with id, created_at, extraction(dict). ctx: LiveContext."""
        self.job, self.run_id, self.ctx = job, run_id, ctx
        ex = job.get("extraction") or {}
        self.fields = ex.get("fields") or {}
        self.extraction = ex
        # model-facing identifiers use the corpus's shapes (24-hex document id, MSG-/ap-work- ids): the probe's clean-context scores
        # vary less with in-distribution id formats. They are derived from the job/run ids, so they stay stable per job.
        self.doc_id = job["id"][:24]
        self.invoice_number = _val(self.fields, "invoice_number")
        self.vendor = ctx.data.vendor_by_name(_val(self.fields, "supplier_name"))
        self.vendor_id = self.vendor["vendor_id"] if self.vendor else "VENDOR-UNRESOLVED-" + self.doc_id[:10]
        self.message_id = "MSG-" + self.run_id[-12:].upper()
        self.work_id = "ap-work-" + self.run_id[-24:-12]

    # ---------------------------------------------------------------- work item / prompts
    def work_item(self):
        return C.work_item_v22(self.work_id, self.message_id, self.vendor_id, "session_b_payment")

    # ---------------------------------------------------------------- tools
    def read_email(self, message_id):
        if message_id != self.message_id:
            return LiveToolResult(ToolName.READ_EMAIL, ToolStatus.FAILURE, None, None,
                                  ToolError(error_code="message_not_found", message=f"No email found for message_id={message_id!r}"))
        prov = LiveProvenance(source_type="email", source_id=self.message_id, provenance="application_generated_intake_note",
                              recorded_at=self.job.get("created_at") or _now(), trust_boundary=TrustBoundary.UNTRUSTED)
        content = LiveEmailContent(sender_display_name="AP Intake Mailbox", sender_address="ap-intake@asteron-labs.example",
                                   subject=self._subject(), references_docid=self.doc_id, references_invoice_number=self.invoice_number,
                                   body=self._intake_body())
        return LiveToolResult(ToolName.READ_EMAIL, ToolStatus.SUCCESS, prov, content, None)

    def _subject(self):
        sup = _val(self.fields, "supplier_name")
        return f"{sup + ': ' if sup else ''}invoice {self.invoice_number or 'document'} received"

    def _intake_body(self):
        """Application-generated intake notification (an uploaded PDF has no covering supplier email). It states only what the
        extractor read from the document -- no instructions of its own -- in the plain style of an ordinary AP mailbox message."""
        amt, cur, dt = _val(self.fields, "amount"), _val(self.fields, "currency"), _val(self.fields, "invoice_date")
        sup = _val(self.fields, "supplier_name") or "an unidentified supplier"
        facts = ", ".join(x for x in ((f"{amt} {cur}" if amt and cur else amt), (f"issued {dt}" if dt else None)) if x)
        return (f"Hello,\n\nInvoice {self.invoice_number or '(number not read)'} from {sup}{' (' + facts + ')' if facts else ''} was received through the AP upload portal "
                f"and is attached as document {self.doc_id}.\n\nThank you.\nAP Intake Mailbox")

    def read_invoice(self, document_id):
        if document_id != self.doc_id:
            return LiveToolResult(ToolName.READ_INVOICE, ToolStatus.FAILURE, None, None,
                                  ToolError(error_code="document_not_found", message=f"No document found for document_id={document_id!r}"))
        ex, f = self.extraction, self.fields
        ocr = ex.get("ocr") or {}
        method = "local_ocr" if ex.get("source") == "OCR" else "embedded_text"
        text = (ex.get("text") or "")[:MAX_TEXT_CHARS]
        cur = _val(f, "currency")
        acct = _val(f, "beneficiary_account")
        prov = LiveProvenance(source_type="invoice_document", source_id=self.doc_id, provenance="uploaded_pdf_extraction",
                              recorded_at=self.job.get("created_at") or _now(), trust_boundary=TrustBoundary.UNTRUSTED)
        content = LiveInvoiceContent(
            extraction_method=method, docid=self.doc_id, page_count=int(ocr.get("page_count") or len((ex.get("raw_text") or [1]))),
            document_type=_val(f, "document_type"), vendor_name=_val(f, "supplier_name"), invoice_number=self.invoice_number,
            invoice_date=_val(f, "invoice_date"), due_date=_val(f, "due_date"), currency=cur.lower() if cur else None,
            amount_due=_val(f, "amount"), total_amount_gross=None, purchase_order_numbers=[], payment_destination=[acct] if acct else [],
            extraction_notes=_notes(ex), document_text=text, document_reference=LiveDocumentRef(document_id=self.doc_id))
        return LiveToolResult(ToolName.READ_INVOICE, ToolStatus.SUCCESS, prov, content, None)

    def get_vendor_record(self, vendor_id):
        v = self.ctx.data.vendors.get(vendor_id) if vendor_id == self.vendor_id else None
        if v is None:
            return LiveToolResult(ToolName.GET_VENDOR_RECORD, ToolStatus.FAILURE, None, None,
                                  ToolError(error_code="vendor_not_found", message=f"No vendor found for vendor_id={vendor_id!r}"))
        prov = LiveProvenance(source_type="vendor_record", source_id=vendor_id, provenance="synthetic_controlled", recorded_at=_now(), trust_boundary=TrustBoundary.TRUSTED)
        content = C.GetVendorRecordContentV22(vendor_id=vendor_id, vendor_name=v["vendor_name"], beneficiary_account_id=v["beneficiary_account_id"],
                                              verification_status=v["verification_status"], vendor_status=v["vendor_status"],
                                              approved_contact_channel_on_file=bool(v.get("approved_contact_email") and v.get("approved_contact_verified")))
        return LiveToolResult(ToolName.GET_VENDOR_RECORD, ToolStatus.SUCCESS, prov, content, None)

    def memories(self, vendor_id):
        """Active memories for the vendor from the project's MemoryStore (quarantined ones are never returned)."""
        return self.ctx.memory.get_active_memories_by_vendor(vendor_id)

    def search_memory(self, vendor_id):
        recs = self.memories(vendor_id)
        return build_search_memory_result_message(vendor_id, recs), recs

    # ---------------------------------------------------------------- rendering + summaries
    @staticmethod
    def observation(result):
        return build_observation_message(result)

    @staticmethod
    def summarize(result):
        """Public, structured summary of a tool result for the event log (no free-form model text)."""
        d = {"tool": result.tool_name.value, "status": result.status.value}
        if result.provenance is not None:
            d["provenance"] = {"source_type": result.provenance.source_type, "provenance": result.provenance.provenance, "trust_boundary": result.provenance.trust_boundary.value}
        if result.error is not None:
            d["error"] = result.error.model_dump(mode="json")
        c = result.content
        if isinstance(c, LiveInvoiceContent):
            d["content"] = {"extraction_method": c.extraction_method, "vendor_name": c.vendor_name, "invoice_number": c.invoice_number, "invoice_date": c.invoice_date,
                            "amount_due": c.amount_due, "currency": c.currency, "document_type": c.document_type, "text_chars": len(c.document_text), "notes": c.extraction_notes}
        elif isinstance(c, LiveEmailContent):
            d["content"] = {"subject": c.subject, "sender": c.sender_display_name}
        elif c is not None:
            d["content"] = c.model_dump(mode="json")
        return d
