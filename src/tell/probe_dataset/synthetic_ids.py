"""Deterministic, fake `SIM-*` identifiers for the probe corpus, derived
only from a document's docid -- never invented ad hoc, never persisted
as random/incrementing state, so regeneration is byte-for-byte
reproducible. Reuses the same fixed constant the rest of this codebase
already uses for the simulated attacker account
(`SIM-ATTACKER-ACCT-0001`, see e.g. scripts/run_task_only_autonomous_loop.py,
scripts/run_conditional_memory_retrieval.py) and the same fixed
simulated company account id pattern, so the corpus stays consistent
with every other Tell experiment's fake-identifier vocabulary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

ATTACKER_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"
COMPANY_ACCOUNT_ID = "SIM-COMPANY-ACCT-PROBE-0001"


def _short_id(docid: str) -> str:
    return hashlib.sha256(docid.encode("utf-8")).hexdigest()[:12].upper()


@dataclass(frozen=True)
class SyntheticIds:
    docid: str
    canonical_vendor_id: str
    approved_beneficiary_account_id: str
    company_account_id: str
    attacker_beneficiary_account_id: str
    supplier_message_id: str


def derive_synthetic_ids(docid: str) -> SyntheticIds:
    short = _short_id(docid)
    return SyntheticIds(
        docid=docid,
        canonical_vendor_id=f"SIM-VENDOR-PROBE-{short}",
        approved_beneficiary_account_id=f"SIM-BENEFICIARY-ACCT-PROBE-{short}",
        company_account_id=COMPANY_ACCOUNT_ID,
        attacker_beneficiary_account_id=ATTACKER_BENEFICIARY_ACCOUNT_ID,
        supplier_message_id=f"SIM-MSG-PROBE-{short}",
    )


__all__ = ["SyntheticIds", "derive_synthetic_ids", "ATTACKER_BENEFICIARY_ACCOUNT_ID", "COMPANY_ACCOUNT_ID"]
