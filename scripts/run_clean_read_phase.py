"""Deterministic read-phase demonstration for clean_04d531ca_v1.

Loads the scenario, calls read_email -> read_invoice -> get_vendor_record
in order, and writes a structured JSONL trace. No payment tool exists yet
and none is called -- this script demonstrates the read-only phase only.

`build_trace()` is a pure function of the scenario (no wall-clock, no
randomness) so it is safe to call twice and diff for determinism testing.
The only non-deterministic values in the written file are the explicitly
permitted `run_started_at` / `run_completed_at` fields on the header/footer
records, which `main()` stamps with the real wall-clock time; every
tool-call record is fully deterministic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tell.agent.tools import (
    GetVendorRecordArgs,
    ReadEmailArgs,
    ReadInvoiceArgs,
    get_vendor_record,
    read_email,
    read_invoice,
)
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, ScenarioBundle, load_scenario

TRACE_PATH = Path("/home/hp5/tell/results/traces/clean_04d531ca_read_phase.jsonl")

# Keys present only on the header/footer records that are allowed to vary
# between runs. Every other field in the trace must be identical run to run.
PERMITTED_VARIABLE_KEYS = {"run_started_at", "run_completed_at"}


def build_trace(bundle: ScenarioBundle) -> list[dict]:
    """Build the full ordered trace deterministically from the scenario
    alone. Contains only tool_name/status/provenance/content/error --
    never evaluation_only."""
    records: list[dict] = []

    records.append(
        {
            "record_type": "run_header",
            "scenario_id": bundle.scenario_id,
            "created_from_docid": bundle.created_from_docid,
        }
    )

    # Step 1: read_email. The agent is handed the message_id by the
    # environment (e.g. an inbox listing), mirrored here by reading it
    # straight off the scenario's untrusted email record.
    message_id = bundle.untrusted_inputs.supplier_email.message_id
    email_args = ReadEmailArgs(message_id=message_id)
    email_result = read_email(bundle, email_args)
    records.append(
        {
            "record_type": "tool_call",
            "step": 1,
            "tool_name": email_result.tool_name.value,
            "request": email_args.model_dump(mode="json"),
            "result": email_result.model_dump(mode="json"),
        }
    )

    # Step 2: read_invoice, using the docid the email itself references.
    document_id = bundle.untrusted_inputs.invoice_document.docid
    invoice_args = ReadInvoiceArgs(document_id=document_id)
    invoice_result = read_invoice(bundle, invoice_args)
    records.append(
        {
            "record_type": "tool_call",
            "step": 2,
            "tool_name": invoice_result.tool_name.value,
            "request": invoice_args.model_dump(mode="json"),
            "result": invoice_result.model_dump(mode="json"),
        }
    )

    # Step 3: get_vendor_record, using the canonical vendor id the
    # application's vendor directory associates with this invoice's
    # printed vendor name (that lookup step itself is not implemented yet;
    # here the id is read directly from trusted_state to drive the demo).
    vendor_id = bundle.trusted_state.canonical_vendor_id
    vendor_args = GetVendorRecordArgs(vendor_id=vendor_id)
    vendor_result = get_vendor_record(bundle, vendor_args)
    records.append(
        {
            "record_type": "tool_call",
            "step": 3,
            "tool_name": vendor_result.tool_name.value,
            "request": vendor_args.model_dump(mode="json"),
            "result": vendor_result.model_dump(mode="json"),
        }
    )

    records.append(
        {
            "record_type": "run_footer",
            "steps_completed": 3,
            "payment_tool_called": False,
            "payment_attempted": False,
            "note": "Read-only demonstration. No pay_invoice tool exists yet; none was called.",
        }
    )

    return records


def main() -> None:
    bundle = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    records = build_trace(bundle)

    now = datetime.now(timezone.utc).isoformat()
    records[0]["run_started_at"] = now
    records[-1]["run_completed_at"] = now

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    email_result = records[1]["result"]
    invoice_result = records[2]["result"]
    vendor_result = records[3]["result"]

    print("=== Tell clean read-phase demonstration ===")
    print(f"Scenario: {bundle.scenario_id} (DocILE document {bundle.created_from_docid})")
    print()
    print(
        f"Supplier email received: from {email_result['content']['sender_address']!r}, "
        f"subject {email_result['content']['subject']!r}"
    )
    print(
        f"Invoice identified: docid={invoice_result['content']['docid']}, "
        f"vendor={invoice_result['content']['vendor_name']!r}, "
        f"invoice_number={invoice_result['content']['invoice_number']!r}"
    )
    print(
        f"Payable amount found: {invoice_result['content']['amount_due']} "
        f"{invoice_result['content']['currency']} "
        f"(extraction_method={invoice_result['content']['extraction_method']})"
    )
    print(
        f"Canonical vendor verified: {vendor_result['content']['vendor_name']!r} "
        f"({vendor_result['content']['vendor_id']}), "
        f"status={vendor_result['content']['verification_status']}"
    )
    print(f"Approved simulated beneficiary retrieved: {vendor_result['content']['beneficiary_account_id']}")
    print()
    print("No payment attempted -- read-only demonstration only.")
    print(f"Trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
