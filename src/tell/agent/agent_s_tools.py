"""Agent-S-only trusted-lookup tool wiring.

Per `results/routing_design/agent_tool_access_matrix_v1.md`, the two
trusted-lookup tools (`check_trusted_invoice_payment_history`,
`check_trusted_dispute_case_status`) are read-only application/human-
reviewer tools; this module is what lets Agent S (and only Agent S) call
them, using the unmodified `tell.agent.trusted_lookups` tool functions.
Agent 1 has no reference to anything in this module.

`ScenarioMetadataProvider` (below) is an EVALUATION-ONLY fake provider, not
a production implementation -- this milestone explicitly excludes the
ledger backend. It derives its answers from a `ScenarioBundle`'s own
already-trusted fields (never from `untrusted_inputs`), and for the 8
pilot fixtures in `data/scenarios/` (which model no duplicate-payment or
dispute history at all) it always answers `NO_RECORD` -- an authoritative,
honest "this trusted store has nothing," not a fabricated finding.
"""
from __future__ import annotations

from tell.agent.trusted_lookups import (
    DisputeCaseLookupOutcome,
    DisputeCaseProvider,
    DisputeCaseQuery,
    InvoicePaymentHistoryLookupOutcome,
    InvoicePaymentHistoryProvider,
    InvoicePaymentHistoryQuery,
    check_trusted_dispute_case_status,
    check_trusted_invoice_payment_history,
)
from tell.evaluation.scenario import ScenarioBundle


class ScenarioMetadataProvider(InvoicePaymentHistoryProvider, DisputeCaseProvider):
    """Evaluation-only fake, injected explicitly (never a module-level
    default) -- same dependency-injection boundary the existing tests use.
    Always returns NO_RECORD for the pilot ScenarioBundle fixtures, which
    do not model this dimension; a real deployment injects a ledger/case-
    database-backed provider instead, which is out of scope here."""

    def lookup(self, query) -> InvoicePaymentHistoryLookupOutcome | DisputeCaseLookupOutcome:  # type: ignore[override]
        if isinstance(query, InvoicePaymentHistoryQuery):
            return InvoicePaymentHistoryLookupOutcome(record=None)
        if isinstance(query, DisputeCaseQuery):
            return DisputeCaseLookupOutcome(record=None)
        raise TypeError(f"unsupported query type: {type(query)}")


def agent_s_check_payment_history(bundle: ScenarioBundle, provider: InvoicePaymentHistoryProvider):
    query = InvoicePaymentHistoryQuery(invoice_document_id=bundle.created_from_docid, vendor_id=bundle.trusted_state.canonical_vendor_id)
    return check_trusted_invoice_payment_history(provider, query)


def agent_s_check_dispute_case(bundle: ScenarioBundle, provider: DisputeCaseProvider):
    query = DisputeCaseQuery(invoice_document_id=bundle.created_from_docid, vendor_id=bundle.trusted_state.canonical_vendor_id)
    return check_trusted_dispute_case_status(provider, query)


__all__ = ["ScenarioMetadataProvider", "agent_s_check_payment_history", "agent_s_check_dispute_case"]
