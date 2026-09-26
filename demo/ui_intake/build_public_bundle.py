#!/usr/bin/env python3
"""Build data/traces_public_v1.json: a PUBLIC-DEMO copy of traces_v1.json with synthetic identifiers.

Preserved: Tell scores, thresholds, zones, routing, alarm transitions, action names, statuses, event structure.
Replaced (deterministically): sample/run/doc ids, invoice numbers, beneficiary/vendor ids, amounts (and the ledger
figures derived from them), private artifact paths/hashes, audit ids/hash chain, case ids.  Fields whose value became
synthetic are re-tagged `fixture`.  Never overwrites data/traces_v1.json.  CPU-only, stdlib only.
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_contract as C  # noqa: E402

SRC, OUT = HERE / "data/traces_v1.json", HERE / "data/traces_public_v1.json"
NOTE = " PUBLIC DEMO: identifiers and amounts are synthetic replacements; scores and routing are from the replay."


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def money(minor, cur):
    return f"{cur.upper()} {minor / 100:,.2f}"


def sanitize(t, idx):
    t = copy.deepcopy(t)
    real_run, real_sample = t["run_id"], t["sample_id"]
    n = f"{idx + 1:02d}"
    cur = t["invoice"]["currency"]["v"]
    old_amt = t["invoice"]["amount_minor_units"]["v"]
    new_amt = (10000 + int(h(real_run)[:6], 16) % 900000) // 25 * 25
    m = {}  # exact-string replacements
    m[real_sample] = f"demo-{n}"; m[real_run] = f"run-demo-{n}"
    old_inv = t["invoice"]["invoice_number"]["v"]; new_inv = f"INV-DEMO-{2000 + idx * 7}"
    m[str(old_inv)] = new_inv
    acct = t["invoice"]["beneficiary_account_id"]["v"]; m[acct] = f"ACCT-DEMO-{n}"
    vid = t["trusted_vendor_record"]["vendor_id"]["v"]; m[vid] = f"VENDOR-DEMO-{n}"
    txt = json.dumps(t)
    for did in set(re.findall(r'"invoice_document_id": "([0-9a-f]{24})"', txt)) | set(re.findall(r"\b[0-9a-f]{24}\b", txt)):
        m.setdefault(did, "doc-demo-" + h(did)[:8])
    for s in set(re.findall(r"SIM-(?:ACCT|VENDOR|MSG)-[0-9A-F]{12}", txt)):
        m.setdefault(s, s.split("-")[1] + "-DEMO-" + h(s)[:6].upper())
    m[money(old_amt, cur)] = money(new_amt, cur)
    m[f"{old_amt / 100:,.2f}"] = f"{new_amt / 100:,.2f}"
    txt = json.dumps(t)
    for k in sorted(m, key=len, reverse=True):
        txt = txt.replace(k, m[k])
    t = json.loads(txt)
    # amounts (ints) and ledger arithmetic
    t["invoice"]["amount_minor_units"] = {"v": new_amt, "p": "fixture"}
    for e in t["events"]:
        a = e.get("action")
        if a and "amount_minor_units" in a.get("args", {}):
            a["args"]["amount_minor_units"] = new_amt
        lg = e.get("ledger")
        if lg:
            if lg["entry"]:
                lg["entry"]["amount_minor_units"] = new_amt
                lg["after"]["operating_balance_minor"] = lg["before"]["operating_balance_minor"] - new_amt
    for f in ("invoice_number", "beneficiary_account_id"):
        t["invoice"][f]["p"] = "fixture"
    t["trusted_vendor_record"]["vendor_id"]["p"] = "fixture"; t["trusted_vendor_record"]["approved_account_id"]["p"] = "fixture"
    t["invoice"]["invoice_number"]["v"] = new_inv
    t["sample_id"] = f"demo-{n}"
    t["hashes"].pop("records_source", None); t["hashes"].pop("records_source_sha256", None)
    t["thresholds"]["note"] += NOTE
    t["public_demo"] = True
    # regenerate audit ids / hash chain
    prev = "0" * 64
    for e in t["events"]:
        e["audit_id"] = "AUD-" + h(t["run_id"] + str(e["seq"]))[:12].upper(); e.pop("audit_hash", None)
        prev = e["audit_hash"] = C.chain_hash(prev, e)
    rv = t.get("review")
    if rv:
        rep = rv["evidence_report"]["v"]; rep["case"] = "CASE-" + h(t["run_id"])[:8].upper()
        rv["audit_ids"]["v"] = [e["audit_id"] for e in t["events"]]
        rv["artifact_hashes"].pop("records_source_sha256", None)
        rv["artifact_hashes"]["evidence_report_sha256"]["v"] = h(C.canon(rep))
        rv["artifact_hashes"]["audit_chain_head_sha256"]["v"] = prev
    return t


def main():
    src = json.loads(SRC.read_text())
    b = copy.deepcopy(src)
    b["traces"] = [sanitize(t, i) for i, t in enumerate(src["traces"])]
    b["source"] = {"note": "PUBLIC DEMO bundle: synthetic identifiers; Tell scores/routing from a TRAINING-SPLIT EPOCH-0 PREVIEW replay, "
                           "not held-out performance."}
    b["public_demo"] = True
    C.validate_bundle(b)
    OUT.write_text(json.dumps(b, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(b['traces'])} traces)")


if __name__ == "__main__":
    main()
