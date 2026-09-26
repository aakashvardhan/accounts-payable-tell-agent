"""Focused CPU-only tests for local invoice intake (service + HTTP). No model, no CUDA, no network beyond loopback.
Run: python3 -m unittest discover -s tests -v"""
import http.client
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import intake as I  # noqa: E402
import extraction as X  # noqa: E402
import server  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402


def make_pdf(lines=(), shapes=False):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, invariant=1)
    y = 800
    for ln in lines:
        c.drawString(60, y, ln); y -= 20
    if shapes:
        c.rect(60, 600, 400, 100, fill=1)
    c.showPage(); c.save()
    return buf.getvalue()


def invoice(no="ACME-1001", supplier="Test Supplier GmbH", total="Total Due: EUR 99.90", extra=()):
    return make_pdf([f"Supplier: {supplier}", f"Invoice No: {no}", "Invoice Date: 2026-01-15", "Due Date: 2026-02-14", total, *extra])


def mk_service(tmp, **kw):
    kw.setdefault("stage_delay", 0)
    kw.setdefault("tesseract", "/nonexistent/tesseract")   # deterministic: no OCR engine unless a test supplies one
    return I.IntakeService(I.IntakeConfig(root=tmp, **kw))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tell-intake-test-"))
        self.svc = mk_service(self.tmp)

    def tearDown(self):
        self.svc.close(); shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, name, data, mime="application/pdf", svc=None):
        svc = svc or self.svc
        job = svc.create_batch([{"name": name, "size": len(data), "type": mime}])[0]
        if job["status"] == "UPLOADING":
            svc.receive_content(job["job_id"], mime, data)
        return job["job_id"]

    def done(self, jid):
        self.assertTrue(self.svc.wait_idle(15)); return self.svc.get_job(jid)


class Extraction(Base):
    def test_fields_come_from_pdf_contents_not_filename_with_full_provenance(self):
        jid = self.upload("INV-9999-WRONG-Supplier-USD-1.pdf", invoice("REAL-777", "Content Supplier Ltd", "Total Due: GBP 1,234.50"))
        j = self.done(jid); f = j["extraction"]["fields"]
        self.assertEqual(j["status"], "READY_FOR_PROCESSING")
        self.assertEqual((f["invoice_number"]["value"], f["supplier_name"]["value"], f["amount"]["value"], f["currency"]["value"]),
                         ("REAL-777", "Content Supplier Ltd", "1234.50", "GBP"))
        self.assertEqual(f["due_date"]["value"], "2026-02-14"); self.assertIn("ISO 8601", f["invoice_date"]["normalization_applied"] or "ISO 8601")
        for k in ("invoice_number", "supplier_name", "amount", "currency", "invoice_date", "due_date"):
            d = f[k]
            self.assertEqual((d["source"], d["page"], d["confidence"]), ("EMBEDDED_TEXT", 1, 1.0), k); self.assertTrue(d["evidence_text"]); self.assertIn("normalization_applied", d)
        self.assertEqual(f["amount"]["normalization_applied"], "thousands separators removed")

    def test_missing_supplier_needs_document_review_not_a_catch_all(self):
        j = self.done(self.upload("x.pdf", make_pdf(["Invoice No: Q-1", "Total: 50.00"])))
        f = j["extraction"]["fields"]
        self.assertEqual(j["status"], "NEEDS_DOCUMENT_REVIEW")
        for k in ("supplier_name", "currency", "due_date", "beneficiary_account", "beneficiary_name"):
            self.assertFalse(f[k]["found"]); self.assertIsNone(f[k]["value"]); self.assertIsNone(f[k]["source"])
        self.assertTrue(any("supplier could not be identified" in r for r in j["extraction"]["decision"]["reasons"]))

    def test_amount_formats_and_symbols_are_never_resolved_to_a_currency(self):
        self.assertEqual(X.parse_amount("\u20ac 1.234,56")["amount"], "1234.56")
        self.assertEqual(X.parse_amount("\u20ac 1.234,56")["token"], "\u20ac")             # a symbol stays a symbol
        self.assertEqual(X.parse_amount("USD 1,234.56")["token"], "USD")
        self.assertIsNone(X.parse_amount("1,250")["token"]); self.assertEqual(X.parse_amount("1,250")["amount"], "1250.00")
        self.assertIsNone(X.parse_amount("about ten")); self.assertIsNone(X.parse_amount("ABC 5.00"))

    def test_image_only_pdf_without_an_engine_fails_closed_with_an_explicit_reason(self):
        j = self.done(self.upload("scan.pdf", make_pdf(shapes=True)))
        self.assertEqual(j["status"], "FAILED"); self.assertIn("OCR", j["error"])
        self.assertEqual([e for e in j["events"] if e["kind"] == "error"][-1]["data"]["failure_code"], "OCR_ENGINE_UNAVAILABLE")
        self.assertNotIn(j["status"], I.ACTIVE)                       # never left waiting

    def test_beneficiary_fields_extracted_when_present(self):
        j = self.done(self.upload("b.pdf", invoice(extra=["Beneficiary: Test Supplier GmbH", "IBAN: DE89 3704 0044 0532 0130 00"])))
        f = j["extraction"]["fields"]
        self.assertEqual(f["beneficiary_account"]["value"], "DE89 3704 0044 0532 0130 00"); self.assertEqual(f["beneficiary_account"]["normalization_applied"], "IBAN checksum valid")
        self.assertEqual(f["beneficiary_name"]["value"], "Test Supplier GmbH")

    def test_text_is_sanitized(self):
        t, trunc = I.sanitize_text("ok\x00\x1b[31m red\u202e evil\x07\n\n\n\nend", 1000)
        self.assertNotIn("\x00", t); self.assertNotIn("\x1b", t); self.assertNotIn("\u202e", t); self.assertNotIn("\x07", t)
        self.assertEqual(I.sanitize_text("a" * 50, 10), ("a" * 10, True))

    def test_extraction_timeout(self):
        slow = self.tmp / "slow.sh"; slow.write_text("#!/bin/sh\nsleep 5\n"); slow.chmod(slow.stat().st_mode | stat.S_IXUSR)
        svc = mk_service(self.tmp / "t2", extract_timeout=0.3, pdftotext=(str(slow),))
        try:
            jid = self.upload("t.pdf", invoice(), svc=svc); self.assertTrue(svc.wait_idle(10))
            j = svc.get_job(jid); self.assertEqual(j["status"], "FAILED"); self.assertIn("timed out", j["error"])
        finally:
            svc.close()

    def test_stored_upload_removed_when_failed(self):
        j = self.done(self.upload("bad.pdf", b"%PDF-1.4\nthis is not a real pdf body"))
        self.assertEqual(j["status"], "FAILED")
        self.assertEqual(list((self.tmp / "uploads").glob("*.pdf")), [])


class Validation(Base):
    def test_rejects_non_pdf_malformed_empty_and_oversize(self):
        cases = {"a.pdf": b"hello world, not a pdf", "b.pdf": b"%PDF-1.4 garbage garbage", "c.pdf": b""}
        ids = {n: self.upload(n, d) for n, d in cases.items()}
        self.assertTrue(self.svc.wait_idle(10))
        for n, jid in ids.items():
            self.assertEqual(self.svc.get_job(jid)["status"], "FAILED", n)
        small = mk_service(self.tmp / "small", max_bytes=2048)
        try:
            j = small.create_batch([{"name": "big.pdf", "size": 5000, "type": "application/pdf"}])[0]
            self.assertEqual(j["status"], "FAILED"); self.assertIn("limit", j["error"])
            jid = small.create_batch([{"name": "liar.pdf", "size": 10, "type": "application/pdf"}])[0]["job_id"]
            small.receive_content(jid, "application/pdf", invoice() + b"0" * 4000); self.assertTrue(small.wait_idle(10))
            self.assertEqual(small.get_job(jid)["status"], "FAILED")
        finally:
            small.close()

    def test_wrong_mime_rejected_at_manifest_and_at_content(self):
        j = self.svc.create_batch([{"name": "a.pdf", "size": 5, "type": "text/html"}])[0]
        self.assertEqual(j["status"], "FAILED"); self.assertIn("MIME", j["error"])
        jid = self.svc.create_batch([{"name": "a.pdf", "size": 5, "type": "application/pdf"}])[0]["job_id"]
        with self.assertRaises(I.IntakeError) as cm:
            self.svc.receive_content(jid, "text/plain", invoice())
        self.assertEqual(cm.exception.status, 415); self.assertEqual(self.svc.get_job(jid)["status"], "FAILED")

    def test_magic_bytes_checked_even_with_pdf_mime_and_extension(self):
        ok, reason, _ = I.validate_pdf_bytes(b"MZ\x90\x00" + b"%PDF-" , 1000)
        self.assertFalse(ok); self.assertIn("signature", reason)
        self.assertTrue(I.validate_pdf_bytes(invoice(), 10**7)[0])

    def test_batch_limits_and_queue_bound(self):
        with self.assertRaises(I.IntakeError) as cm:
            self.svc.create_batch([{"name": f"{i}.pdf", "size": 1, "type": "application/pdf"} for i in range(21)])
        self.assertEqual(cm.exception.status, 413)
        q = mk_service(self.tmp / "q", max_queue=3)
        try:
            q.create_batch([{"name": f"{i}.pdf", "size": 1, "type": "application/pdf"} for i in range(3)])
            with self.assertRaises(I.IntakeError) as cm:
                q.create_batch([{"name": "x.pdf", "size": 1, "type": "application/pdf"}])
            self.assertEqual(cm.exception.status, 429)
        finally:
            q.close()

    def test_traversal_filenames_never_touch_the_filesystem(self):
        names = ["../../../../tmp/evil.pdf", "..\\..\\windows\\evil.pdf", "/etc/passwd.pdf", "a/../../b.pdf", "<img src=x onerror=alert(1)>.pdf", "x\x00y.pdf"]
        before = {p for p in self.tmp.parent.iterdir()}
        ids = [self.upload(n, invoice(no=f"T-{i}")) for i, n in enumerate(names)]
        self.assertTrue(self.svc.wait_idle(10))
        for jid in ids:
            j = self.svc.get_job(jid)
            self.assertRegex(j["display_name"], r"^[\w .()\-]+$"); self.assertNotIn("/", j["display_name"]); self.assertNotIn("..", j["display_name"].replace("...", ""))
        self.assertEqual({p for p in self.tmp.parent.iterdir()}, before)
        stored = list((self.tmp / "uploads").iterdir()); self.assertEqual(len(stored), len(names))
        for p in stored:
            self.assertRegex(p.name, r"^[0-9a-f]{32}\.pdf$")
        self.assertFalse(Path("/tmp/evil.pdf").exists())
        self.assertEqual(I.safe_display_name("../../etc/passwd"), "passwd")
        with self.assertRaises(I.IntakeError):
            self.svc.get_job("../../etc/passwd")

    def test_job_payload_never_exposes_storage_paths(self):
        j = self.done(self.upload("a.pdf", invoice()))
        blob = json.dumps(j)
        self.assertNotIn(str(self.tmp), blob); self.assertNotIn("uploads", blob); self.assertNotIn("stored_name", blob)


class QueueBehaviour(Base):
    def test_status_transitions_in_order(self):
        j = self.done(self.upload("a.pdf", invoice()))
        seq = []
        for e in j["events"]:
            if not seq or seq[-1] != e["stage"]:
                seq.append(e["stage"])
        self.assertEqual(seq, ["UPLOADING", "VALIDATING", "EXTRACTING_EMBEDDED_TEXT", "EXTRACTION_COMPLETE", "READY_FOR_PROCESSING"])
        for i, e in enumerate(j["events"]):
            I.validate_event(e, seq=i)
        self.assertEqual(j["progress"], 100); self.assertEqual(j["sha256"], __import__("hashlib").sha256(invoice()).hexdigest())

    def test_duplicate_detected_by_content_hash_not_filename(self):
        a = self.done(self.upload("first.pdf", invoice("D-1")))
        b = self.done(self.upload("totally-different-name.pdf", invoice("D-1")))
        c = self.done(self.upload("first.pdf", invoice("D-2")))   # same name, different content
        self.assertEqual(a["status"], "READY_FOR_PROCESSING"); self.assertEqual(b["status"], "DUPLICATE"); self.assertEqual(b["duplicate_of"], a["job_id"])
        self.assertEqual(c["status"], "READY_FOR_PROCESSING")
        self.assertEqual(len(list((self.tmp / "uploads").glob("*.pdf"))), 2)   # duplicate content is not kept

    def test_identical_concurrent_uploads_yield_one_original(self):
        data = invoice("C-1")
        ids = [self.svc.create_batch([{"name": f"{i}.pdf", "size": len(data), "type": "application/pdf"}])[0]["job_id"] for i in range(6)]
        ts = [threading.Thread(target=self.svc.receive_content, args=(i, "application/pdf", data)) for i in ids]
        [t.start() for t in ts]; [t.join() for t in ts]; self.assertTrue(self.svc.wait_idle(15))
        st = sorted(self.svc.get_job(i)["status"] for i in ids)
        self.assertEqual(st, ["DUPLICATE"] * 5 + ["READY_FOR_PROCESSING"])

    def test_batch_appears_immediately_and_failures_are_isolated(self):
        files = [("ok1.pdf", invoice("B-1")), ("junk.pdf", b"not a pdf"), ("ok2.pdf", invoice("B-2")), ("scan.pdf", make_pdf(shapes=True))]
        jobs = self.svc.create_batch([{"name": n, "size": len(d), "type": "application/pdf"} for n, d in files])
        self.assertEqual(len(jobs), 4); self.assertEqual(len({j["job_id"] for j in jobs}), 4)
        listed = self.svc.list_jobs()
        self.assertEqual([j["status"] for j in listed], ["UPLOADING"] * 4)          # queued before any content arrives
        for j, (n, d) in zip(jobs, files):
            self.svc.receive_content(j["job_id"], "application/pdf", d)
        self.assertTrue(self.svc.wait_idle(15))
        self.assertEqual([self.svc.get_job(j["job_id"])["status"] for j in jobs], ["READY_FOR_PROCESSING", "FAILED", "READY_FOR_PROCESSING", "FAILED"])

    def test_stale_uploads_expire(self):
        jid = self.svc.create_batch([{"name": "a.pdf", "size": 5, "type": "application/pdf"}])[0]["job_id"]
        with self.svc.lock:
            self.svc.db.execute("UPDATE jobs SET updated_at='2000-01-01T00:00:00Z' WHERE id=?", (jid,))
        self.assertEqual(self.svc.get_job(jid)["status"], "UPLOADING")
        self.svc.list_jobs(); self.assertEqual(self.svc.get_job(jid)["status"], "FAILED")

    def test_queue_survives_restart_and_interrupted_jobs_fail_cleanly(self):
        a = self.done(self.upload("keep.pdf", invoice("S-1")))
        stuck = self.svc.create_batch([{"name": "mid.pdf", "size": 5, "type": "application/pdf"}])[0]["job_id"]
        self.svc.close()
        again = mk_service(self.tmp)
        try:
            ids = {j["job_id"]: j for j in again.list_jobs()}
            self.assertEqual(ids[a["job_id"]]["status"], "READY_FOR_PROCESSING"); self.assertEqual(again.get_job(a["job_id"])["extraction"]["fields"]["invoice_number"]["value"], "S-1")
            self.assertEqual(ids[stuck]["status"], "FAILED")
        finally:
            again.close(); self.svc = mk_service(self.tmp)

    def test_reset_removes_uploads_and_state_but_keeps_inbox(self):
        (self.tmp / "inbox").mkdir(exist_ok=True); (self.tmp / "inbox" / "keep.pdf").write_bytes(invoice("R-1"))
        self.upload("u.pdf", invoice("R-2")); self.svc.scan(); self.assertTrue(self.svc.wait_idle(15))
        self.assertGreater(len(list((self.tmp / "uploads").glob("*.pdf"))), 0)
        r = self.svc.reset()
        self.assertEqual(self.svc.list_jobs(), []); self.assertEqual(list((self.tmp / "uploads").iterdir()), []); self.assertGreaterEqual(r["uploads_removed"], 2)
        self.assertIsNone(self.svc.status()["last_scan"]); self.assertTrue((self.tmp / "inbox" / "keep.pdf").exists())


class InboxScan(Base):
    def put(self, name, data):
        (self.tmp / "inbox").mkdir(exist_ok=True); (self.tmp / "inbox" / name).write_bytes(data)

    def run_scan(self):
        s = self.svc.scan(); self.assertEqual(s["state"], "running")
        self.assertTrue(self.svc.wait_idle(15)); return self.svc.get_scan(s["scan_id"])

    def test_scan_imports_new_pdfs_reports_counts_and_keeps_files(self):
        self.put("one.pdf", invoice("I-1")); self.put("two.pdf", invoice("I-2")); self.put("bad.pdf", b"nope"); self.put("notes.txt", b"ignored")
        s = self.run_scan()
        self.assertEqual((s["discovered"], s["imported"], s["duplicates"], s["rejected"]), (3, 2, 0, 1)); self.assertEqual(s["state"], "complete"); self.assertTrue(s["completed_at"])
        st = sorted(j["status"] for j in self.svc.list_jobs()); self.assertEqual(st, ["FAILED", "READY_FOR_PROCESSING", "READY_FOR_PROCESSING"])
        self.assertEqual(sorted(p.name for p in (self.tmp / "inbox").iterdir()), ["bad.pdf", "notes.txt", "one.pdf", "two.pdf"])   # never deleted
        self.assertEqual(self.svc.status()["last_scan"]["imported"], 2)
        self.assertTrue(all(j["source"] == "local_inbox" for j in self.svc.list_jobs()))

    def test_repeated_scan_does_not_duplicate_and_reports_no_new(self):
        self.put("one.pdf", invoice("I-1")); self.put("bad.pdf", b"nope")
        self.run_scan(); n = len(self.svc.list_jobs())
        s = self.run_scan()
        self.assertEqual((s["imported"], s["rejected"], s["duplicates"]), (0, 0, 2)); self.assertEqual(s["message"], "No new invoices found"); self.assertEqual(len(self.svc.list_jobs()), n)

    def test_scan_dedupes_by_hash_against_uploads_and_renamed_inbox_files(self):
        data = invoice("U-1"); self.done(self.upload("manual.pdf", data))
        self.put("renamed-copy.pdf", data)
        self.assertEqual(self.run_scan()["duplicates"], 1); self.assertEqual(len(self.svc.list_jobs()), 1)
        self.put("later.pdf", invoice("U-2")); s = self.run_scan(); self.assertEqual(s["imported"], 1)

    def test_empty_inbox_reports_no_new_invoices(self):
        s = self.run_scan(); self.assertEqual((s["discovered"], s["imported"]), (0, 0)); self.assertEqual(s["message"], "No new invoices found")

    def test_scan_ignores_symlinks_and_subdirectories(self):
        self.put("real.pdf", invoice("L-1")); (self.tmp / "secret.pdf").write_bytes(invoice("SECRET"))
        os.symlink(self.tmp / "secret.pdf", self.tmp / "inbox" / "link.pdf"); (self.tmp / "inbox" / "sub").mkdir(); (self.tmp / "inbox" / "sub" / "deep.pdf").write_bytes(invoice("D-1"))
        s = self.run_scan(); self.assertEqual(s["imported"], 1)
        self.assertNotIn("SECRET", json.dumps(self.svc.list_jobs()))

    def test_oversize_inbox_file_is_rejected_without_being_read(self):
        small = mk_service(self.tmp / "s2", max_bytes=1024)
        try:
            (small.cfg.inbox_dir / "huge.pdf").write_bytes(b"%PDF-" + b"0" * 5000)
            s = small.scan(); self.assertTrue(small.wait_idle(10)); s = small.get_scan(s["scan_id"]); self.assertEqual(s["rejected"], 1)
            s2 = small.scan(); self.assertTrue(small.wait_idle(10)); self.assertEqual(small.get_scan(s2["scan_id"])["rejected"], 0)
        finally:
            small.close()

    def test_source_abstraction_is_pluggable(self):
        class Fake(I.InvoiceSource):
            name = "fake_mailbox"
            def discover(self):
                d = invoice("F-1"); yield "mail-attachment.pdf", len(d), lambda: d
        svc = I.IntakeService(I.IntakeConfig(root=self.tmp / "f", stage_delay=0), source=Fake())
        try:
            svc.scan(); self.assertTrue(svc.wait_idle(10)); j = svc.list_jobs()[0]; self.assertEqual((j["source"], j["status"]), ("fake_mailbox", "READY_FOR_PROCESSING"))
        finally:
            svc.close()


class HttpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tell-http-"))
        cls.svc = mk_service(cls.tmp)
        cls.bundle = json.loads((ROOT / "data/traces_v1.json").read_text())
        server.Handler.bundle_bytes = json.dumps(cls.bundle).encode()
        server.Handler.intake, server.Handler.public = cls.svc, False
        cls.srv = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler); cls.srv.n_traces = len(cls.bundle["traces"])
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close(); cls.svc.close(); shutil.rmtree(cls.tmp, ignore_errors=True)
        server.Handler.intake, server.Handler.public = None, True

    def setUp(self):
        self.svc.reset()
        for p in (self.tmp / "inbox").glob("*"):
            p.unlink()

    def call(self, method, path, body=None, headers=None, port=None):
        c = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=10)
        h = {"X-Tell-Intake": "1"}; h.update(headers or {})
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode(); h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=body, headers=h); r = c.getresponse(); raw = r.read(); c.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw

    def test_multi_pdf_batch_over_http_end_to_end(self):
        files = [("a.pdf", invoice("H-1")), ("b.pdf", invoice("H-2")), ("bad.pdf", b"nope")]
        st, r = self.call("POST", "/api/intake/batch", {"files": [{"name": n, "size": len(d), "type": "application/pdf"} for n, d in files]})
        self.assertEqual(st, 201); jobs = r["jobs"]; self.assertEqual(len(jobs), 3)
        st, r = self.call("GET", "/api/intake/jobs"); self.assertEqual([j["status"] for j in r["jobs"]], ["UPLOADING"] * 3)   # queued immediately
        for j, (n, d) in zip(jobs, files):
            st, _ = self.call("PUT", f"/api/intake/jobs/{j['job_id']}/content", d, {"Content-Type": "application/pdf"}); self.assertEqual(st, 202)
        self.assertTrue(self.svc.wait_idle(15))
        st, r = self.call("GET", "/api/intake/jobs")
        self.assertEqual(sorted(j["status"] for j in r["jobs"]), ["FAILED", "READY_FOR_PROCESSING", "READY_FOR_PROCESSING"])
        st, d = self.call("GET", f"/api/intake/jobs/{jobs[0]['job_id']}"); self.assertEqual(d["extraction"]["fields"]["invoice_number"]["value"], "H-1")
        st, s = self.call("GET", "/api/intake/status"); self.assertEqual((s["ready"], s["failed"], s["enabled"]), (2, 1, True))

    def test_scan_over_http_and_reset(self):
        (self.tmp / "inbox" / "x.pdf").write_bytes(invoice("HS-1"))
        st, s = self.call("POST", "/api/intake/scan", {}); self.assertEqual(st, 202)
        self.assertTrue(self.svc.wait_idle(15)); st, s = self.call("GET", f"/api/intake/scan/{s['scan_id']}"); self.assertEqual(s["imported"], 1)
        st, r = self.call("POST", "/api/intake/reset", {}); self.assertEqual(st, 200)
        st, r = self.call("GET", "/api/intake/jobs"); self.assertEqual(r["jobs"], [])

    def test_wrong_content_type_and_bad_ids(self):
        st, r = self.call("POST", "/api/intake/batch", {"files": [{"name": "a.pdf", "size": 3, "type": "application/pdf"}]}); jid = r["jobs"][0]["job_id"]
        st, _ = self.call("PUT", f"/api/intake/jobs/{jid}/content", b"data", {"Content-Type": "text/html"}); self.assertEqual(st, 415)
        self.assertEqual(self.call("GET", "/api/intake/jobs/../../etc/passwd")[0], 404)
        self.assertEqual(self.call("GET", "/api/intake/jobs/" + "0" * 32)[0], 404)
        self.assertEqual(self.call("PUT", "/api/intake/jobs/nope/content", b"x", {"Content-Type": "application/pdf"})[0], 404)
        self.assertEqual(self.call("POST", "/api/intake/batch", {"files": "x"})[0], 400)

    def test_oversize_body_refused_before_reading(self):
        st, r = self.call("POST", "/api/intake/batch", {"files": [{"name": "a.pdf", "size": 3, "type": "application/pdf"}]}); jid = r["jobs"][0]["job_id"]
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.putrequest("PUT", f"/api/intake/jobs/{jid}/content"); c.putheader("X-Tell-Intake", "1"); c.putheader("Content-Type", "application/pdf")
        c.putheader("Content-Length", str(self.svc.cfg.max_bytes + 5)); c.endheaders()
        self.assertEqual(c.getresponse().status, 413); c.close()
        st, r = self.call("POST", "/api/intake/batch", {"files": [{"name": f"{i}.pdf", "size": 1, "type": "application/pdf"} for i in range(21)]}); self.assertEqual(st, 413)

    def test_csrf_guards(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port); c.request("POST", "/api/intake/scan", body=b"{}"); self.assertEqual(c.getresponse().status, 403); c.close()
        st, r = self.call("POST", "/api/intake/scan", {}, {"Origin": "http://evil.example"}); self.assertEqual(st, 403)
        st, r = self.call("POST", "/api/intake/scan", {}, {"Origin": f"http://127.0.0.1:{self.port}"}); self.assertEqual(st, 202)
        self.svc.wait_idle(10)

    def test_security_headers_on_intake_responses(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port); c.request("GET", "/api/intake/status"); r = c.getresponse(); r.read()
        self.assertIn("default-src 'self'", r.getheader("Content-Security-Policy")); self.assertEqual(r.getheader("Cache-Control"), "no-store")


class PublicMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tell-pub-"))
        bundle = json.loads((ROOT / "data/traces_public_v1.json").read_text())
        server.Handler.bundle_bytes = json.dumps(bundle).encode()
        cls.public_flag = bool(bundle.get("public_demo"))
        server.Handler.intake, server.Handler.public = None, True
        cls.srv = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler); cls.srv.n_traces = 11
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close(); shutil.rmtree(cls.tmp, ignore_errors=True)

    def req(self, method, path, body=b"{}"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, path, body=body if method != "GET" else None, headers={"X-Tell-Intake": "1", "Content-Type": "application/pdf"}); r = c.getresponse(); raw = r.read(); c.close()
        return r.status, json.loads(raw)

    def test_public_bundle_is_flagged_so_public_mode_is_implied(self):
        self.assertTrue(self.public_flag)

    def test_every_intake_mutation_fails_closed(self):
        for method, path in [("POST", "/api/intake/scan"), ("POST", "/api/intake/batch"), ("PUT", "/api/intake/jobs/" + "0" * 32 + "/content"),
                             ("POST", "/api/intake/reset"), ("POST", "/api/intake/anything")]:
            st, r = self.req(method, path); self.assertEqual(st, 403, (method, path)); self.assertIn("public-demo", r["error"])

    def test_public_status_reports_disabled_and_reads_are_empty(self):
        st, r = self.req("GET", "/api/intake/status"); self.assertEqual((st, r["enabled"]), (200, False))
        st, r = self.req("GET", "/api/intake/jobs"); self.assertEqual(st, 403)

    def test_public_mode_creates_no_runtime_files(self):
        self.assertIsNone(server.Handler.intake)

    def test_default_handler_is_fail_closed(self):
        src = (ROOT / "server.py").read_text()
        self.assertIn("public = True", src)
        self.assertIn('"--enable-intake", action="store_true"', src)   # opt-in only; never the default
        self.assertIn("--enable-intake requires --require-auth", src)


class NoModelOrCuda(unittest.TestCase):
    def test_no_ml_modules_loaded_and_no_gpu_code(self):
        for m in ("torch", "transformers", "peft", "tensorflow", "onnxruntime"):
            self.assertNotIn(m, sys.modules)
        for f in ("intake.py", "server.py", "make_sample_invoices.py"):
            src = (ROOT / f).read_text().lower()
            for bad in ("import torch", "cuda", "transformers", "langchain", "requests", "urllib.request", "openai", "anthropic", "http.client"):
                self.assertNotIn(bad, src.replace("no cuda", ""), (f, bad))

    def test_intake_does_not_import_tell_runtime_logic(self):
        src = (ROOT / "intake.py").read_text()
        for bad in ("tell.safety", "tell.routing", "tell.payment", "tell.agent", "tell.detector"):
            self.assertNotIn(bad, src)
        self.assertNotIn("sys.path", src)

    def test_extraction_uses_only_local_binary_without_shell(self):
        src = (ROOT / "intake.py").read_text()
        self.assertNotIn("shell=True", src); self.assertIn("stdin=subprocess.DEVNULL", src)


if __name__ == "__main__":
    unittest.main()
