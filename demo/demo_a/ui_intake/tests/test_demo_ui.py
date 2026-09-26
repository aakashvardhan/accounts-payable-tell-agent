"""Focused CPU-only tests for the Tell demo UI backend/contract. Run: python3 -m unittest discover -s tests -v"""
import copy
import http.client
import json
import re
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_traces  # noqa: E402
import server  # noqa: E402
import trace_contract as C  # noqa: E402

import os
BUNDLE_PATH = Path(os.environ.get("TELL_TRACE_BUNDLE", ROOT / "data/traces_v1.json"))
PUBLIC = bool(os.environ.get("TELL_TRACE_BUNDLE"))
BUNDLE = json.loads(BUNDLE_PATH.read_text())
BY = {t["scenario_id"]: t for t in BUNDLE["traces"]}


def rechain(t):
    prev = "0" * 64
    for e in t["events"]:
        e.pop("audit_hash", None); prev = e["audit_hash"] = C.chain_hash(prev, e)
    return t


class TraceSchema(unittest.TestCase):
    def test_bundle_valid(self):
        C.validate_bundle(BUNDLE)
        self.assertEqual({t["kind"] for t in BUNDLE["traces"]}, {"story", "background"})

    def test_rejects_missing_provenance(self):
        t = copy.deepcopy(BY["clean_payment"])
        g = next(e for e in t["events"] if e.get("gate")); del g["prov"]["gate"]; rechain(t)
        with self.assertRaisesRegex(C.TraceError, "prov.gate"):
            C.validate_trace(t)

    def test_rejects_broken_audit_chain(self):
        t = copy.deepcopy(BY["clean_payment"]); t["events"][2]["title"] = "tampered"
        with self.assertRaisesRegex(C.TraceError, "chain"):
            C.validate_trace(t)

    def test_rejects_zone_mismatch_and_threshold_drift(self):
        t = copy.deepcopy(BY["escalated_attack"]); e = next(e for e in t["events"] if e.get("tell")); e["tell"]["zone"] = "agent_1"; rechain(t)
        with self.assertRaisesRegex(C.TraceError, "zone"):
            C.validate_trace(t)
        t = copy.deepcopy(BY["escalated_attack"]); next(e for e in t["events"] if e.get("tell"))["tell"]["upper"] = 0.9; rechain(t)
        with self.assertRaisesRegex(C.TraceError, "thresholds"):
            C.validate_trace(t)

    def test_rejects_non_simulated_side_effects_and_bad_label(self):
        t = copy.deepcopy(BY["clean_payment"]); t["side_effects_simulated"] = False
        with self.assertRaises(C.TraceError):
            C.validate_trace(t)
        b = copy.deepcopy(BUNDLE); b["fixture_label"] = "simulated"
        with self.assertRaises(C.TraceError):
            C.validate_bundle(b)

    def test_escalated_requires_review_block(self):
        t = copy.deepcopy(BY["escalated_attack"]); del t["review"]
        with self.assertRaises(C.TraceError):
            C.validate_trace(t)

    def test_zone_for(self):
        lo, hi = 0.17, 0.51
        self.assertEqual([C.zone_for(s, lo, hi) for s in (0, .1699, .17, .5099, .51, 1)],
                         ["agent_1", "agent_1", "tell_verify", "tell_verify", "agent_s", "agent_s"])
        for bad in (float("nan"), -0.1, 1.1):
            with self.assertRaises(C.TraceError):
                C.zone_for(bad, lo, hi)


class Provenance(unittest.TestCase):
    def test_exact_fixture_label(self):
        self.assertEqual(BUNDLE["fixture_label"], "SIMULATED DEMO FIXTURE — NOT MEASURED MODEL PERFORMANCE")

    def test_unmeasured_outcomes_are_fixture(self):
        for name in ("clean_payment", "escalated_attack"):
            for e in BY[name]["events"]:
                for f in ("gate", "ledger", "outcome"):
                    if e.get(f) is not None:
                        self.assertEqual(e["prov"][f], "fixture", (name, f))
        # clean-payment validator approval is fixture and preserves the differing captured outcome
        v = next(e for e in BY["clean_payment"]["events"] if e.get("validator"))
        self.assertEqual(v["prov"]["validator"], "fixture"); self.assertEqual(v["validator"]["outcome"], "APPROVED")
        self.assertEqual(v["validator"]["captured_outcome"], "unobserved_identifier")

    def test_measured_fields_are_real(self):
        for name in ("clean_payment", "escalated_attack"):
            t = BY[name]; tell = next(e for e in t["events"] if e.get("tell"))
            self.assertEqual(tell["prov"]["tell"], "real")
            self.assertEqual(t["thresholds"]["lower"]["p"], "real"); self.assertEqual(t["thresholds"]["upper"]["p"], "real")
            self.assertEqual(t["invoice"]["beneficiary_account_id"]["p"], "fixture" if PUBLIC else "real")
            self.assertEqual(t["invoice"]["vendor_name"]["p"], "fixture")
        rt = next(e for e in BY["escalated_attack"]["events"] if e["type"] == "routing")
        self.assertEqual(rt["routing"]["to"], "agent_s"); self.assertEqual(rt["prov"]["routing"], "real")
        self.assertEqual(BY["clean_payment"]["events"][-1]["outcome"], "PAID")
        self.assertEqual(BY["escalated_attack"]["events"][-1]["outcome"], "ESCALATED")

    @unittest.skipIf(PUBLIC, "public bundle has no private sample ids")
    def test_scores_match_captured_records(self):
        recs = {json.loads(l)["sample_id"]: json.loads(l) for l in build_traces.REC.read_text().splitlines()}
        for t in BUNDLE["traces"]:
            e = next(e for e in t["events"] if e.get("tell"))
            self.assertEqual(e["tell"]["score"], recs[t["sample_id"]]["tell_activation_risk_score"])

    def test_no_side_effect_ever_real_and_all_simulated(self):
        self.assertTrue(all(t["side_effects_simulated"] for t in BUNDLE["traces"]))

    def test_hashes_present_and_review_hashes(self):
        h = BY["escalated_attack"]["hashes"]
        self.assertRegex(h["probe_weights_sha256"], r"^[0-9a-f]{64}$"); self.assertRegex(h["adapter_weights_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("NOT_HELD_OUT_PERFORMANCE", h["adapter_label"])
        for k, v in BY["escalated_attack"]["review"]["artifact_hashes"].items():
            self.assertRegex(v["v"], r"^[0-9a-f]{64}$", k)

    @unittest.skipIf(PUBLIC, "private-bundle determinism check")
    def test_build_is_deterministic_and_matches_committed_file(self):
        out = server.HERE / "data" / "traces_v1.json"; before = out.read_bytes()
        build_traces.main(); self.assertEqual(before, out.read_bytes())


class Ledger(unittest.TestCase):
    def test_clean_posts_and_attack_does_not(self):
        c = next(e for e in BY["clean_payment"]["events"] if e.get("ledger"))["ledger"]
        self.assertEqual(c["before"]["operating_balance_minor"] - c["after"]["operating_balance_minor"], BY["clean_payment"]["invoice"]["amount_minor_units"]["v"])
        self.assertEqual(c["after"]["invoice_status"], "paid"); self.assertEqual(c["entry"]["rail"], "LOCAL_SQLITE_SIMULATION")
        a = next(e for e in BY["escalated_attack"]["events"] if e.get("ledger"))["ledger"]
        self.assertEqual(a["before"]["operating_balance_minor"], a["after"]["operating_balance_minor"]); self.assertIsNone(a["entry"])

    def test_escalation_has_gate_block_and_no_paid(self):
        g = next(e for e in BY["escalated_attack"]["events"] if e.get("gate"))["gate"]
        self.assertEqual(g["decision"], "BLOCK")
        self.assertFalse(any(e.get("outcome") == "PAID" for e in BY["escalated_attack"]["events"]))


class Escalation(unittest.TestCase):
    def test_review_evidence_complete(self):
        esc = [t for t in BUNDLE["traces"] if t["queue_status"] == "ESCALATED"]
        self.assertGreaterEqual(len(esc), 2)
        for t in esc:
            r = t["review"]
            for k in ("triggering_evidence", "attack_surface", "recommended_action", "evidence_report", "audit_ids", "artifact_hashes"):
                self.assertIn(k, r)
            self.assertEqual(r["audit_ids"]["v"], [e["audit_id"] for e in t["events"]])
            self.assertEqual(r["artifact_hashes"]["audit_chain_head_sha256"]["v"], t["events"][-1]["audit_hash"])
            self.assertFalse(r["evidence_report"]["v"]["executed"])


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.Handler.bundle_bytes = json.dumps(BUNDLE).encode()
        cls.srv = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler); cls.srv.n_traces = len(BUNDLE["traces"])
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def req(self, method, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5); c.request(method, path); r = c.getresponse(); b = r.read(); c.close(); return r, b

    def test_healthz(self):
        r, b = self.req("GET", "/healthz"); d = json.loads(b)
        self.assertEqual(r.status, 200); self.assertEqual(d["status"], "ok"); self.assertFalse(d["live_inference"]); self.assertEqual(d["traces"], len(BUNDLE["traces"]))

    def test_pages_and_api(self):
        r, b = self.req("GET", "/"); self.assertEqual(r.status, 200); self.assertIn(b"<title>", b)
        r, b = self.req("GET", "/api/traces"); C.validate_bundle(json.loads(b))
        for p in ("/static/app.js", "/static/tell_core.js", "/static/styles.css"):
            self.assertEqual(self.req("GET", p)[0].status, 200)

    def test_security_headers_readonly_and_traversal(self):
        r, _ = self.req("GET", "/"); csp = r.getheader("Content-Security-Policy")
        self.assertIn("default-src 'self'", csp); self.assertEqual(r.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(self.req("POST", "/api/traces")[0].status, 405)
        self.assertEqual(self.req("GET", "/static/../server.py")[0].status, 404)
        self.assertEqual(self.req("GET", "/static/%2e%2e/server.py")[0].status, 404)
        self.assertEqual(self.req("GET", "/data/traces_v1.json")[0].status, 404)

    def test_bind_policy(self):
        with self.assertRaises(SystemExit):
            server.bind(8080, False)
        self.assertNotIn(8080, [p for p in range(8079, 8100) if p not in server.RESERVED])
        self.assertEqual(server.PREFERRED, 8083)
        self.assertIn("0.0.0.0", (ROOT / "server.py").read_text())

    def test_bind_falls_back_when_occupied(self):
        import socket
        s = socket.socket(); s.bind(("0.0.0.0", 0)); s.listen(); p = s.getsockname()[1]
        try:
            srv, skipped = server.bind(p, True)
            try:
                self.assertEqual(skipped[0][0], p); self.assertNotEqual(srv.server_address[1], p); self.assertEqual(srv.server_address[0], "0.0.0.0")
            finally:
                srv.server_close()
        finally:
            s.close()


class NoExternalDeps(unittest.TestCase):
    def test_static_assets_reference_only_local_resources(self):
        pat = re.compile(r"(https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}", re.I)
        for f in (ROOT / "static").iterdir():
            txt = f.read_text()
            for m in pat.finditer(txt):
                # the only permitted URL-like string is the SVG namespace identifier (not fetched)
                self.assertIn(m.group(0), ("http://www.w3.org",), f"{f.name}: {m.group(0)}")
            self.assertNotRegex(txt, r"@import|<link[^>]+href=\"https?:|<script[^>]+src=\"https?:|src=\"//", f.name)
            self.assertNotRegex(txt, r"fonts\.(googleapis|gstatic)|cdn\.|unpkg|jsdelivr|cdnjs", f.name)

    def test_no_inline_script_or_style_needed(self):
        html = (ROOT / "static/index.html").read_text()
        self.assertNotRegex(html, r"<script(?![^>]*src=)[^>]*>"); self.assertNotIn(" style=", html)

    def test_server_imports_stdlib_only(self):
        for f in ("server.py", "trace_contract.py", "build_traces.py"):
            mods = set(re.findall(r"^(?:import|from)\s+([a-zA-Z_]+)", (ROOT / f).read_text(), re.M))
            self.assertFalse(mods & {"torch", "transformers", "peft", "requests", "fastapi", "streamlit", "flask", "numpy", "urllib3"}, (f, mods))
            self.assertNotIn("cuda", (ROOT / f).read_text().lower().replace("no cuda", ""))

    def test_no_approve_or_clear_control_in_ui(self):
        js = (ROOT / "static/app.js").read_text().lower()
        self.assertNotRegex(js, r"approve payment|clear alarm|release payment|pay now")


if __name__ == "__main__":
    unittest.main()
