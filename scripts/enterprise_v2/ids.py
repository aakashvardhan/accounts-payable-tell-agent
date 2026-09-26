"""Neutral, deterministic per-document identifiers for enterprise corpus
v2 and enterprise benchmark v1 -- the corpus-v1.1 account-ID discipline
(`tell.probe_dataset.synthetic_ids_v1_1`) extended to every role and every
population:

  - ONE format for every account role and every class:
    ``SIM-ACCT-<12 uppercase hex>`` (always 21 characters). No prefix,
    suffix, or length differs between approved, unauthorized, obsolete,
    or rejected accounts, or between populations.
  - Values are SHA-256 derived from (salt, population, docid, role), so
    they are per-document (never a fixed attacker id), reproducible, and
    cannot cross splits because documents never cross splits.
  - No id contains a class word (ATTACK, CLEAN, POISON, FAKE, MALICIOUS).

Vendor/message/run identifiers use the same salt-and-hash approach and
are shared by every sample of one document regardless of class, so no
identifier encodes a label.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

ID_SALT = "tell-enterprise-v2-neutral-id-salt-2026"
ACCOUNT_PREFIX = "SIM-ACCT-"
HEX_LEN = 12
FORBIDDEN_SUBSTRINGS = ("ATTACK", "CLEAN", "POISON", "FAKE", "MALICIOUS", "BENIGN", "PROBE", "LORA", "BENCH", "TRAIN", "TEST")


def _hex(*parts: str) -> str:
    return hashlib.sha256(":".join((ID_SALT,) + parts).encode("utf-8")).hexdigest()[:HEX_LEN].upper()


def account_id(population: str, docid: str, role: str) -> str:
    out = f"{ACCOUNT_PREFIX}{_hex(population, docid, 'acct', role)}"
    for w in FORBIDDEN_SUBSTRINGS:
        if w in out:
            raise RuntimeError(f"derived id {out!r} contains forbidden substring {w!r}")
    return out


@dataclass(frozen=True)
class DocIds:
    docid: str
    run_id: str
    vendor_id: str
    message_id: str
    prior_message_id: str
    approved: str
    obsolete: str
    rejected: str

    def unauthorized(self, pair_index: int) -> str:
        return account_id(self.population_tag, self.docid, f"unauthorized:{pair_index}")

    population_tag: str = ""


def derive_doc_ids(population: str, docid: str) -> DocIds:
    ids = DocIds(
        docid=docid,
        run_id=f"ap-work-{_hex(population, docid, 'run').lower()}",
        vendor_id=f"SIM-VENDOR-{_hex(population, docid, 'vendor')}",
        message_id=f"SIM-MSG-{_hex(population, docid, 'msg')}",
        prior_message_id=f"SIM-MSG-{_hex(population, docid, 'prior-msg')}",
        approved=account_id(population, docid, "approved"),
        obsolete=account_id(population, docid, "obsolete"),
        rejected=account_id(population, docid, "rejected"),
        population_tag=population,
    )
    if len({ids.approved, ids.obsolete, ids.rejected}) != 3:
        raise RuntimeError(f"{docid}: account id collision")
    return ids


__all__ = ["ACCOUNT_PREFIX", "ID_SALT", "DocIds", "derive_doc_ids", "account_id"]
