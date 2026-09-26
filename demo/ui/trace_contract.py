"""Versioned trace contract for the Tell demo UI (tell.demo_trace/1.0).

The UI consumes only documents that pass ``validate_trace``. Every structured
event field carries a per-field provenance tag: ``real`` (captured epoch-0
replay) or ``fixture`` (deterministic SIMULATED DEMO FIXTURE). Values are never
merged across the two without a tag.  Stdlib only.
"""
import hashlib
import json

CONTRACT_VERSION = "tell.demo_trace/1.0"
FIXTURE_LABEL = "SIMULATED DEMO FIXTURE — NOT MEASURED MODEL PERFORMANCE"
MODES = ("REPLAY — REAL CAPTURE", "SIMULATED DEMO FIXTURE", "LIVE LOCAL INFERENCE")
PROV = ("real", "fixture")
STATUSES = ("PROCESSING", "FOLLOW_UP", "ESCALATED", "PAID", "BLOCKED")
ZONES = ("agent_1", "tell_verify", "agent_s")
ACTORS = ("intake", "agent_1", "tell", "router", "agent_s", "trusted_lookup",
          "validator", "gate", "ledger", "review_queue")
EVENT_TYPES = ("invoice_received", "tool_call", "tell_score", "routing", "agent_action",
               "trusted_lookup", "validator", "gate", "ledger", "evidence_submitted", "outcome")
# structured (non-narrative) event fields that must carry provenance when present
STRUCTURED = ("action", "tell", "routing", "lookups", "alarm", "validator", "gate",
              "ledger", "outcome", "summary", "latency_seconds", "tokens")
GATE = ("ALLOW", "BLOCK")
TERMINALS = ("PAID", "BLOCKED", "ESCALATED", "FOLLOW_UP")


class TraceError(ValueError):
    pass


def zone_for(score, lower, upper):
    """Three-zone routing; score < lower -> agent_1, [lower, upper) -> tell_verify, >= upper -> agent_s."""
    if not isinstance(score, (int, float)) or score != score or score in (float("inf"), float("-inf")) \
            or score < 0 or score > 1:
        raise TraceError("score must be a finite number in [0, 1]")
    if not (0 <= lower < upper <= 1):
        raise TraceError("thresholds must satisfy 0 <= lower < upper <= 1")
    return "agent_1" if score < lower else ("tell_verify" if score < upper else "agent_s")


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_hash(prev, event):
    body = {k: v for k, v in event.items() if k not in ("audit_hash",)}
    return hashlib.sha256((prev + canon(body)).encode()).hexdigest()


def _need(cond, msg):
    if not cond:
        raise TraceError(msg)


def _wrapped(d, key, where):
    _need(isinstance(d.get(key), dict) and "v" in d[key] and d[key].get("p") in PROV,
          f"{where}.{key} must be {{v, p}} with p in {PROV}")


def validate_trace(t):
    _need(isinstance(t, dict), "trace must be an object")
    _need(t.get("contract_version") == CONTRACT_VERSION, f"contract_version must be {CONTRACT_VERSION}")
    for k in ("run_id", "scenario_id", "title", "kind", "queue_status", "invoice", "thresholds",
              "hashes", "events", "side_effects_simulated"):
        _need(k in t, f"missing {k}")
    _need(t["kind"] in ("story", "background"), "kind")
    _need(t["queue_status"] in STATUSES, "queue_status")
    _need(t["side_effects_simulated"] is True, "side_effects_simulated must be true")
    for k in ("invoice_number", "vendor_name", "beneficiary_account_id", "amount_minor_units", "currency"):
        _wrapped(t["invoice"], k, "invoice")
    th = t["thresholds"]
    _wrapped(th, "lower", "thresholds"); _wrapped(th, "upper", "thresholds")
    _need(0 <= th["lower"]["v"] < th["upper"]["v"] <= 1, "thresholds order")
    for k in ("model_repo_id", "model_revision", "probe_weights_sha256", "adapter_weights_sha256"):
        _need(isinstance(t["hashes"].get(k), str) and t["hashes"][k], f"hashes.{k}")
    ev = t["events"]
    _need(isinstance(ev, list) and ev, "events must be a non-empty list")
    prev = "0" * 64
    for i, e in enumerate(ev):
        w = f"events[{i}]"
        _need(e.get("seq") == i, f"{w}.seq must equal index")
        _need(e.get("type") in EVENT_TYPES, f"{w}.type")
        _need(e.get("actor") in ACTORS, f"{w}.actor")
        _need(isinstance(e.get("title"), str) and e["title"], f"{w}.title")
        prov = e.get("prov")
        _need(isinstance(prov, dict), f"{w}.prov")
        for f in STRUCTURED:
            if e.get(f) is not None:
                _need(prov.get(f) in PROV, f"{w}.prov.{f} required (real|fixture)")
        for f, p in prov.items():
            _need(p in PROV and e.get(f) is not None, f"{w}.prov.{f} has no matching field")
        if e.get("tell"):
            tl = e["tell"]
            _need(tl["lower"] == th["lower"]["v"] and tl["upper"] == th["upper"]["v"],
                  f"{w}.tell thresholds must equal trace thresholds")
            _need(tl["zone"] == zone_for(tl["score"], tl["lower"], tl["upper"]), f"{w}.tell.zone mismatch")
        if e.get("routing"):
            _need(e["routing"].get("to") in ZONES, f"{w}.routing.to")
        if e.get("gate"):
            _need(e["gate"].get("decision") in GATE, f"{w}.gate.decision")
        if e.get("ledger"):
            _need(all(k in e["ledger"] for k in ("before", "after")), f"{w}.ledger before/after")
        if e.get("outcome") is not None:
            _need(e["outcome"] in TERMINALS, f"{w}.outcome")
        h = chain_hash(prev, e)
        _need(e.get("audit_hash") == h, f"{w}.audit_hash chain broken")
        prev = h
    if t["kind"] == "story":
        _need(ev[-1].get("outcome") is not None, "story must end with a terminal outcome")
    if t["queue_status"] == "ESCALATED":
        _need(isinstance(t.get("review"), dict), "ESCALATED trace requires review")
        for k in ("triggering_evidence", "attack_surface", "recommended_action", "evidence_report",
                  "audit_ids", "artifact_hashes"):
            _need(k in t["review"], f"review.{k}")
    return t


def validate_bundle(b):
    _need(b.get("contract_version") == CONTRACT_VERSION, "bundle contract_version")
    _need(b.get("fixture_label") == FIXTURE_LABEL, "fixture_label must be the exact required string")
    ids = set()
    for t in b["traces"]:
        validate_trace(t)
        _need(t["run_id"] not in ids, "duplicate run_id"); ids.add(t["run_id"])
    _need({"clean_payment", "escalated_attack"} <= {t["scenario_id"] for t in b["traces"]},
          "both guided stories required")
    return b
