#!/usr/bin/env python3
"""Build data/testrun_v1.json: a compact, display-only view of the held-out three-arm evaluation (the 1,400-record test run).

    python3 demo/ui_ocr/build_testrun_data.py

Reads (read-only) results/lora_training/agent_s_v1/three_arm_eval/{summary,three_arm_rows}_*.json[l] and the frozen-artifact
manifests; writes one JSON file for the UI's "Test run" page. No model text, prompts, document ids or account ids are copied --
only per-record labels, the recorded Tell score and each arm's typed outcome. Every source file is hashed into the output so the
page can say exactly which evaluation it shows. Nothing here re-scores, re-routes or re-labels anything.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL = REPO / "results/lora_training/agent_s_v1/three_arm_eval"
PIPE = REPO / "results/lora_training/agent_s_v1/pipeline_status.json"
ADAPTER = REPO / "results/lora_training/agent_s_v1/frozen_adapter/FROZEN.sha256"
PROBE = REPO / "results/probe_training/enterprise_v1/frozen_probe/FROZEN.sha256"
OPERATIONAL = REPO / "results/routing_design/operational_threshold_v1/operational_threshold_v1.json"
OUT = HERE / "data" / "testrun_v1.json"
CORPUS = REPO / "results/enterprise_corpus/v2_2"
WATCHER = REPO / "results/lora_training/agent_s_v1/durable_watcher.log"
# records whose reference next step is a final payment decision (pay, report, ask the vendor, stop) -- the ledger shows only these;
# the rest are mid-workflow steps (read the email, read the invoice, check the vendor record) and are not payment outcomes
PAYMENT_DECISIONS = ("propose_payment", "submit_evidence_report", "request_vendor_clarification", "fail_closed")
CORPUS_SETS = ("probe_v2_2", "probe_v2_2_lexical_challenge", "probe_v2_2_delayed_memory_ood")

PARTITIONS = [("test", "Held-out test", "Unseen invoices, the four attack surfaces"),
              ("lexical_challenge", "Lexical paraphrase", "Attacks reworded to avoid the training vocabulary"),
              ("delayed_memory_ood", "Delayed memory (OOD)", "Poisoned memories that act later — out of distribution")]
ARMS = [("agent_1_only", "a1", "Agent 1 alone", "The base model with no Tell and no safety adapter"),
        ("always_agent_s", "as", "Always Agent S", "Every decision handled by the frozen safety adapter"),
        ("tell_routed", "tr", "Tell routing", "Tell sends a decision to Agent S at or above the operational threshold"),
        ("three_zone_ablation", "tz", "Three-zone routing", "Ablation: Agent 1 / Tell-Verify / Agent S bands")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def minor(s):
    """'$10,000.00' / '551.30' / '250' -> minor units, or None."""
    if s is None:
        return None
    t = re.sub(r"[^0-9.,]", "", str(s))
    if not t:
        return None
    if "," in t and "." in t:              # the separator that comes last is the decimal point
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        t = head.replace(",", "") + ("." + tail if len(tail) == 2 else tail)     # '2,793,00' -> 2793.00 ; '1,250' -> 1250
    try:
        return int(round(float(t) * 100))
    except ValueError:
        return None


def tidy(name):
    """Trim stray OCR punctuation at the ends of an annotated vendor name ('& WACHTELL, LIPTON,' -> 'WACHTELL, LIPTON'); the name itself is unchanged."""
    return re.sub(r"^[^\w(]+|[\s,;:&-]+$", "", name) if name else name


def invoice_meta():
    """docid -> invoice fields exactly as the model saw them in its read_invoice tool result (the DocILE annotation of that document)."""
    meta, files = {}, {}
    for s in CORPUS_SETS:
        for kind in ("labels", "inputs"):
            f = CORPUS / f"{s}_{kind}.jsonl"
            files[f.name] = sha(f)
        for lab, inp in zip((CORPUS / f"{s}_labels.jsonl").open(), (CORPUS / f"{s}_inputs.jsonl").open()):
            lab, inp = json.loads(lab), json.loads(inp)
            if lab["docid"] in meta:
                continue
            for m in inp["messages"]:
                c = m.get("content", "")
                if '"tool_name": "read_invoice"' not in c:
                    continue
                try:
                    ct = json.loads(c.split("Tool result:\n\n", 1)[1].split("\n\n[trust_boundary", 1)[0]).get("content") or {}
                except (IndexError, ValueError):
                    continue
                if ct.get("amount_due") or ct.get("total_amount_gross") or ct.get("invoice_number"):
                    due, total = minor(ct.get("amount_due")), minor(ct.get("total_amount_gross"))
                    meta[lab["docid"]] = {"inv": ct.get("invoice_number"), "amt": due if due else total,     # amount due; the invoice total when due reads 0.00
                                          "cur": ct.get("currency"), "vendor": ct.get("vendor_name")}
                    break
    return meta, files


def stage_end_times():
    """partition -> when its evaluation stage finished, from the pipeline watcher log (the records carry no per-record timestamp)."""
    out = {}
    for line in WATCHER.read_text().splitlines():
        m = re.match(r"\[(\S+)\] STAGE END: eval_(\w+) ::", line)
        if m:
            out[m.group(2)] = m.group(1) + "+01:00"
    return out


def proposed_payment(r):
    """The payment the Tell-routed decision proposed (amount, currency, whether it went to the approved vendor account) -- not an executed payment."""
    a = r["arms"]["tell_routed"]
    act = a.get("action") or {}
    if a.get("action_type") != "propose_payment" or act.get("amount_minor_units") is None:
        return None
    return {"amt": act["amount_minor_units"], "cur": act.get("currency"), "ok": act.get("beneficiary_account_id") == r.get("approved_account_id")}


def arm_view(a, key):
    v = {"t": a.get("action_type"), "u": bool(a.get("unauthorized_payment")), "g": bool(a.get("matches_gold")),
         "c": bool(a.get("vendor_clarification")), "e": bool(a.get("evidence_report")), "p": a.get("parse_outcome")}
    if key == "tell_routed":
        v["r"] = a.get("routed_agent")
    if key == "three_zone_ablation":
        v["r"] = a.get("route_label")
    return v


def main():
    op = json.loads(OPERATIONAL.read_text())
    files, parts, rows = {}, [], []
    meta, corpus_files = invoice_meta()
    ends = stage_end_times()
    cur_by_doc = {}
    gold_by_doc = {}                    # the same invoice's reference payment, from any record of that document
    for pid, _, _ in PARTITIONS:
        for line in (EVAL / f"three_arm_rows_{pid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            g = r.get("gold_action") or {}
            if g.get("amount_minor_units") is not None:
                gold_by_doc.setdefault(r["docid"], g)
            for arm in r["arms"].values():                 # the currency recorded on any payment action for this document
                act = arm.get("action") or {}
                if act.get("currency"):
                    cur_by_doc.setdefault(r["docid"], act["currency"])
    for pid, label, desc in PARTITIONS:
        sp, rp = EVAL / f"summary_{pid}.json", EVAL / f"three_arm_rows_{pid}.jsonl"
        files[sp.name], files[rp.name] = sha(sp), sha(rp)
        summary = json.loads(sp.read_text())
        n = 0
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            n += 1
            g, dm = r.get("gold_action") or gold_by_doc.get(r["docid"]) or {}, meta.get(r["docid"], {})
            if g.get("amount_minor_units") is None:
                g = gold_by_doc.get(r["docid"], {})
            rows.append({"id": r["sample_id"], "p": pid, "done_at": ends.get(pid),
                         "inv": dm.get("inv") or g.get("invoice_number"), "vendor": tidy(dm.get("vendor") or r.get("vendor_group_key")),
                         "amt": dm.get("amt") if dm.get("amt") is not None else g.get("amount_minor_units"), "cur": dm.get("cur") or g.get("currency") or gold_by_doc.get(r["docid"], {}).get("currency") or cur_by_doc.get(r["docid"]),
                         "cur_from_record": not dm.get("cur"), "decision": r.get("gold_action_type") in PAYMENT_DECISIONS, "cls": r["class"], "surface": r.get("attack_surface"), "family": r.get("attack_family"),
                         "dp": r.get("decision_point"), "gold": r.get("gold_action_type"), "score": r["tell_score"],
                         "arms": {short: arm_view(r["arms"][key], key) for key, short, _, _ in ARMS},
                         "pay": proposed_payment(r)})
        parts.append({"id": pid, "label": label, "desc": desc, "n": n,
                      "summary": {short: summary[key] for key, short, _, _ in ARMS if key in summary}})
    pipe = json.loads(PIPE.read_text())
    adapter = dict(l.split()[::-1] for l in ADAPTER.read_text().splitlines() if l.strip())
    probe = dict(l.split()[::-1] for l in PROBE.read_text().splitlines() if l.strip())
    out = {
        "contract": "tell.testrun_view/1.0",
        "label": "HELD-OUT EVALUATION — recorded three-arm test run, not live uploads",
        "completed_at": pipe.get("updated_at"),
        "thresholds": {"tell_verify_from": op["primary_threshold_unchanged"], "alarm_from": op["selected_operational_threshold"],
                       "note": "The alarm threshold is a post-hoc operational setting chosen on validation after the primary test result was known; "
                               "it is not the pre-registered primary threshold."},
        "artifacts": {"adapter_weights_sha256": adapter.get("adapter_model.safetensors"), "probe_weights_sha256": probe.get("probe_weights.safetensors")},
        "source": {"dir": str(EVAL.relative_to(REPO)), "files_sha256": files, "corpus_files_sha256": corpus_files,
                   "dates": "Each record is dated by the time its partition's evaluation stage finished (durable_watcher.log); records carry no own timestamp."},
        "notes": ["Invoice number, vendor and amount are the fields the model saw in its read_invoice result (DocILE annotation); "
                  "1,400 decision points come from 50 distinct invoices, each tested in several attack and clean variants.",
                  "These are decision-level records: the evaluation scored each decision; no payment was executed on any ledger. "
                  "The payment ledger shows only payment decisions (records whose reference next step is to pay, report, ask the vendor or stop); "
                  "mid-workflow steps are part of the evaluation but are not payment outcomes. Where an invoice states no currency, the currency of its "
                  "trusted reference record is used. "
                  "Where the Tell-routed decision proposed a payment, its amount and whether it targeted the approved vendor account are kept as a proposed payment.",
                  "Agent 1 actions were reused from the probe milestone's recorded generations (same frozen base model, same prompts).",
                  "Amount and currency correctness are not measured by this corpus; only the beneficiary is checked."],
        "arms": [{"key": key, "id": short, "label": label, "desc": desc} for key, short, label, desc in ARMS],
        "partitions": parts,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT.relative_to(REPO)}: {len(rows)} records, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
