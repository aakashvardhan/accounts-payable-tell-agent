"""Corpus-v1.1-discipline account-identifier generation for the LoRA
corpus (see `tell.probe_dataset.synthetic_ids_v1_1` for the probe
corpus's identical discipline, which this module mirrors exactly with
its own salt/prefix so no LoRA-corpus id can ever collide with a
probe-corpus id).

Three per-document identifiers, one shared neutral format
(`SIM-ACCT-LORA-<12 uppercase hex chars>`, always 26 characters):

  - `approved_beneficiary_account_id`   -- the trusted account on file,
    returned by the (simulated) get_vendor_record tool.
  - `unauthorized_beneficiary_account_id` -- what attack email/invoice
    text asks the agent to pay instead.
  - `obsolete_beneficiary_account_id`   -- a same-format, same-length
    "former/rejected" account, mentioned only by clean hard-negative
    content, in an explicitly benign, non-payment context.

Deterministic from `(CORPUS_SALT_LORA_V1, docid, role)` via SHA-256 --
never random, never hand-picked. None of the three ever contains
`ATTACKER`, `MALICIOUS`, `POISON`, `FAKE`, or a class-label word.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

CORPUS_SALT_LORA_V1 = "tell-lora-corpus-v1-account-id-salt-2026"

ACCOUNT_ID_PREFIX = "SIM-ACCT-LORA-"
ACCOUNT_ID_HEX_LEN = 12

_FORBIDDEN_SUBSTRINGS = ("ATTACKER", "MALICIOUS", "POISON", "FAKE", "CLEAN", "ATTACK")


def _account_id(docid: str, tag: str) -> str:
    digest = hashlib.sha256(f"{CORPUS_SALT_LORA_V1}:{docid}:{tag}".encode("utf-8")).hexdigest()
    suffix = digest[:ACCOUNT_ID_HEX_LEN].upper()
    account_id = f"{ACCOUNT_ID_PREFIX}{suffix}"
    for word in _FORBIDDEN_SUBSTRINGS:
        if word in account_id:
            raise RuntimeError(f"derived account id {account_id!r} unexpectedly contains forbidden word {word!r}")
    return account_id


def _short_id(docid: str) -> str:
    return hashlib.sha256(docid.encode("utf-8")).hexdigest()[:12].upper()


@dataclass(frozen=True)
class LoraSyntheticIds:
    docid: str
    canonical_vendor_id: str
    supplier_message_id: str
    approved_beneficiary_account_id: str
    unauthorized_beneficiary_account_id: str
    obsolete_beneficiary_account_id: str


def derive_lora_synthetic_ids(docid: str) -> LoraSyntheticIds:
    short = _short_id(docid)
    approved = _account_id(docid, "approved")
    unauthorized = _account_id(docid, "unauthorized")
    obsolete = _account_id(docid, "obsolete")
    ids = {approved, unauthorized, obsolete}
    if len(ids) != 3:
        raise RuntimeError(f"{docid}: derived account ids collided ({approved!r}, {unauthorized!r}, {obsolete!r})")
    return LoraSyntheticIds(
        docid=docid,
        canonical_vendor_id=f"SIM-VENDOR-LORA-{short}",
        supplier_message_id=f"SIM-MSG-LORA-{short}",
        approved_beneficiary_account_id=approved,
        unauthorized_beneficiary_account_id=unauthorized,
        obsolete_beneficiary_account_id=obsolete,
    )


__all__ = [
    "CORPUS_SALT_LORA_V1",
    "ACCOUNT_ID_PREFIX",
    "ACCOUNT_ID_HEX_LEN",
    "LoraSyntheticIds",
    "derive_lora_synthetic_ids",
]
