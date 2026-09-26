"""Leak checks for data/traces_public_v1.json against the private replay bundle. Run: python3 -m unittest tests.test_public_bundle"""
import json, re, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trace_contract as C  # noqa: E402

PRIV = (ROOT / "data/traces_v1.json").read_text()
PUB = (ROOT / "data/traces_public_v1.json").read_text()


class PublicBundle(unittest.TestCase):
    def test_valid_and_flagged(self):
        b = json.loads(PUB); C.validate_bundle(b); self.assertTrue(b["public_demo"])

    def test_no_private_identifiers(self):
        priv = json.loads(PRIV)
        ids = set(re.findall(r"SIM-[A-Z]+-[0-9A-F]{12}", PRIV)) | set(re.findall(r"\b[0-9a-f]{24}\b", PRIV)) | set(re.findall(r"(?:lv22|lms1)-[0-9a-f]{16}", PRIV))
        for t in priv["traces"]:
            ids |= {str(t["invoice"]["invoice_number"]["v"]), str(t["invoice"]["amount_minor_units"]["v"])} | {e["audit_id"] for e in t["events"]}
        self.assertGreater(len(ids), 100)
        self.assertEqual([i for i in ids if i in PUB], [])

    def test_no_paths_secrets_or_private_hashes(self):
        for bad in ("/home/", "results/", "records.jsonl", "hp5", "records_source", "secret", "api_key", "apikey", "password", "bearer", "access_token", "authorization"):
            self.assertNotIn(bad, PUB, bad)
        self.assertNotIn(json.loads(PRIV)["source"]["records_sha256"], PUB)

    def test_no_emails(self):
        self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", PUB))

    def test_scores_routing_preserved_and_provenance_kept(self):
        a, b = json.loads(PRIV)["traces"], json.loads(PUB)["traces"]
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual([(e["tell"], e["routing"] if "routing" in e else None, e.get("alarm")) for e in x["events"] if e.get("tell")],
                             [(e["tell"], e["routing"] if "routing" in e else None, e.get("alarm")) for e in y["events"] if e.get("tell")])
            self.assertEqual([e.get("gate") for e in x["events"]], [e.get("gate") for e in y["events"]])
            self.assertEqual([e["prov"] for e in x["events"]], [e["prov"] for e in y["events"]])
            self.assertEqual(x["queue_status"], y["queue_status"])
            self.assertEqual(x["thresholds"]["lower"], y["thresholds"]["lower"])
        self.assertEqual(json.loads(PUB)["fixture_label"], C.FIXTURE_LABEL)

    def test_replaced_fields_retagged_fixture(self):
        for t in json.loads(PUB)["traces"]:
            for f in ("invoice_number", "beneficiary_account_id", "amount_minor_units"):
                self.assertEqual(t["invoice"][f]["p"], "fixture")
            self.assertIn("PUBLIC DEMO", t["thresholds"]["note"])


if __name__ == "__main__":
    unittest.main()
