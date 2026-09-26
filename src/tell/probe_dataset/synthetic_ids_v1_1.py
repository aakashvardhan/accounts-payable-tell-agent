"""Corpus-v1.1 account-identifier generation (see
`probe_activation_corpus_v1_1_report.md`, Part A / Section 1-2 of the
v1.1 spec).

Corpus v1 used one fixed, module-level constant for every attacked
sample's beneficiary account, `synthetic_ids.ATTACKER_BENEFICIARY_ACCOUNT_ID
= "SIM-ATTACKER-ACCT-0001"`. That constant is untouched by this module
(v1 is preserved unmodified) but is a trivial shortcut for a probe: the
literal substring alone perfectly predicts the label. v1.1 replaces it
with three **per-document, deterministically derived** identifiers that
share one neutral format and length:

  - ``approved_beneficiary_account_id``   -- the trusted account on file.
  - ``unauthorized_beneficiary_account_id`` -- the account attack text
    asks the model to pay instead. Present in every attack sample.
  - ``obsolete_beneficiary_account_id``   -- a same-format, same-length
    "former/rejected" account, mentioned only by the designated clean
    hard-negative family (``clean_account_reference``, see
    `templates_v1_1.py`) in an explicitly benign, non-payment context.
    This directly controls the "any account-ID in unauthorized format
    implies attack" shortcut: a probe sees that exact ID *shape* in both
    classes.

All three share the format ``SIM-ACCT-PROBE-<12 uppercase hex chars>``
(see `ACCOUNT_ID_PREFIX`) -- identical prefix, identical length, no
occurrence of "ATTACKER", "MALICIOUS", "POISON", "FAKE", or any class
label word. Values are derived only from `(CORPUS_SALT_V1_1, docid, tag)`
via SHA-256, never randomly generated or hand-picked, so regeneration is
byte-for-byte deterministic and every document's three IDs are (with
overwhelming probability) mutually distinct and distinct from every
other document's IDs -- including across train/validation/test, which
already never share a docid (see `select_probe_corpus_documents.py`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Recorded verbatim in the v1.1 protocol manifest. Changing this salt
# changes every derived ID -- it exists only to namespace this corpus
# version's identifiers away from any other deterministic-hash use in
# the codebase, not as a secret.
CORPUS_SALT_V1_1 = "tell-probe-corpus-v1_1-account-id-salt-2026"

ACCOUNT_ID_PREFIX = "SIM-ACCT-PROBE-"
ACCOUNT_ID_HEX_LEN = 12

_FORBIDDEN_SUBSTRINGS = ("ATTACKER", "MALICIOUS", "POISON", "FAKE", "CLEAN", "ATTACK")


def _account_id(docid: str, tag: str) -> str:
    digest = hashlib.sha256(f"{CORPUS_SALT_V1_1}:{docid}:{tag}".encode("utf-8")).hexdigest()
    suffix = digest[:ACCOUNT_ID_HEX_LEN].upper()
    account_id = f"{ACCOUNT_ID_PREFIX}{suffix}"
    for word in _FORBIDDEN_SUBSTRINGS:
        if word in account_id:
            raise RuntimeError(f"derived account id {account_id!r} unexpectedly contains forbidden word {word!r}")
    return account_id


def _short_id(docid: str) -> str:
    """Unchanged from `synthetic_ids._short_id` -- reused for the
    non-account identifiers below (vendor id, message id), which are not
    part of the identifier-shortcut concern this module exists to fix."""
    return hashlib.sha256(docid.encode("utf-8")).hexdigest()[:12].upper()


@dataclass(frozen=True)
class SyntheticIdsV11:
    docid: str
    canonical_vendor_id: str
    supplier_message_id: str
    approved_beneficiary_account_id: str
    unauthorized_beneficiary_account_id: str
    obsolete_beneficiary_account_id: str


def derive_synthetic_ids_v1_1(docid: str) -> SyntheticIdsV11:
    short = _short_id(docid)
    approved = _account_id(docid, "approved")
    unauthorized = _account_id(docid, "unauthorized")
    obsolete = _account_id(docid, "obsolete")
    ids = {approved, unauthorized, obsolete}
    if len(ids) != 3:
        raise RuntimeError(f"{docid}: derived account ids collided ({approved!r}, {unauthorized!r}, {obsolete!r})")
    return SyntheticIdsV11(
        docid=docid,
        canonical_vendor_id=f"SIM-VENDOR-PROBE-{short}",
        supplier_message_id=f"SIM-MSG-PROBE-{short}",
        approved_beneficiary_account_id=approved,
        unauthorized_beneficiary_account_id=unauthorized,
        obsolete_beneficiary_account_id=obsolete,
    )


__all__ = ["CORPUS_SALT_V1_1", "ACCOUNT_ID_PREFIX", "ACCOUNT_ID_HEX_LEN", "SyntheticIdsV11", "derive_synthetic_ids_v1_1"]
