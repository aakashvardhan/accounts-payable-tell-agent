"""The Test run page shows the recorded held-out evaluation. Its view file must be a faithful, text-free projection of the evaluation
outputs: every count re-derived from the rows equals the evaluation's own summary, and no model text or ids leak into it.
Run: python3 -m unittest tests.test_testrun_view -v"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "data" / "testrun_v1.json"
EVAL = ROOT.parents[1] / "results/lora_training/agent_s_v1/three_arm_eval"


@unittest.skipUnless(VIEW.exists() and EVAL.exists(), "test-run view or evaluation outputs not present")
class TestRunView(unittest.TestCase):
    def setUp(self):
        self.v = json.loads(VIEW.read_text())

    def test_counts_match_the_evaluation_summaries(self):
        self.assertEqual(len(self.v["rows"]), 1400)
        for p in self.v["partitions"]:
            summary = json.loads((EVAL / f"summary_{p['id']}.json").read_text())
            rows = [r for r in self.v["rows"] if r["p"] == p["id"]]
            att, cln = [r for r in rows if r["cls"] == "attacked"], [r for r in rows if r["cls"] != "attacked"]
            for a in self.v["arms"]:
                s = summary[a["key"]]
                self.assertEqual(sum(r["arms"][a["id"]]["u"] for r in att), s["malicious"]["unauthorized_payment_count"], (p["id"], a["id"]))
                self.assertEqual(sum(r["arms"][a["id"]]["g"] for r in att), s["malicious"]["matches_gold_count"], (p["id"], a["id"]))
                self.assertEqual(sum(r["arms"][a["id"]]["g"] for r in cln), s["clean"]["matches_gold_count"], (p["id"], a["id"]))
                self.assertEqual(len(att) - sum(r["arms"][a["id"]]["u"] for r in att), s["malicious_contained_count"], (p["id"], a["id"]))

    def test_proposed_payments_come_from_tell_routed_payment_actions_only(self):
        for part in ("test", "lexical_challenge", "delayed_memory_ood"):
            rows = {json.loads(l)["sample_id"]: json.loads(l) for l in (EVAL / f"three_arm_rows_{part}.jsonl").read_text().splitlines()}
            for r in (x for x in self.v["rows"] if x["p"] == part):
                src = rows[r["id"]]["arms"]["tell_routed"]
                if src.get("action_type") == "propose_payment":
                    self.assertEqual(r["pay"]["amt"], src["action"]["amount_minor_units"])
                    self.assertEqual(r["pay"]["ok"], not src["unauthorized_payment"] or r["pay"]["ok"] is False)
                else:
                    self.assertIsNone(r["pay"])

    def test_source_hashes_are_recorded_and_no_text_or_ids_leak(self):
        self.assertEqual(len(self.v["source"]["files_sha256"]), 6)
        blob = json.dumps(self.v)
        for bad in ("raw_output", "SIM-ACCT", "docid", "approved_account", "/home/", "messages"):
            self.assertNotIn(bad, blob, bad)
        self.assertIn("post-hoc", self.v["thresholds"]["note"])


if __name__ == "__main__":
    unittest.main()
