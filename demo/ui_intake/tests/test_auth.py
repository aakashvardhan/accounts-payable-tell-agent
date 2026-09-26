"""Focused CPU-only tests for passcode authentication on the public intake server.
Servers are started as real subprocesses with the production flags on ephemeral ports. Secrets are throwaway test values.
Run: python3 -m unittest discover -s tests -v"""
import http.client
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import auth as A  # noqa: E402
from test_intake import invoice, make_pdf  # noqa: E402

PUBLIC = ROOT / "data" / "traces_public_v1.json"
PRIVATE = ROOT / "data" / "traces_v1.json"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def mk_auth(**kw):
    clock = kw.pop("clock", Clock())
    return A.Auth("TEST-PASS-CODE-0000", b"k" * 48, clock=clock, **kw), clock


class AuthUnit(unittest.TestCase):
    def test_login_and_cookie_shape(self):
        a, _ = mk_auth()
        c = a.login("TEST-PASS-CODE-0000", "1.2.3.4")
        self.assertTrue(a.check(c)); self.assertRegex(c, r"^[\w-]{40,}\.[0-9a-f]{64}$")
        ck = a.set_cookie(c)
        for attr in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/", "Max-Age=3600"):
            self.assertIn(attr, ck)
        self.assertTrue(ck.startswith("__Host-tell_session=")); self.assertNotIn("Domain", ck)
        self.assertNotIn("TEST-PASS-CODE", c)
        self.assertEqual(A.Auth.cookie_from("a=b; __Host-tell_session=" + c + "; z=1"), c)

    def test_wrong_passcode_rejected_and_types_safe(self):
        a, _ = mk_auth(fail_limit=1000, global_fail_limit=1000)
        for bad in ("", "nope", "TEST-PASS-CODE-0001", "TEST-PASS-CODE-0000 ", None, 123, "x" * 10000):
            with self.assertRaises(A.AuthError) as cm:
                a.login(bad, "c")
            self.assertEqual(cm.exception.status, 401)

    def test_tampered_forged_and_malformed_cookies(self):
        a, _ = mk_auth(); c = a.login("TEST-PASS-CODE-0000", "c"); tok, sig = c.split(".")
        for bad in (None, "", "abc", tok, tok + "." + "0" * 64, "x." + sig, tok + "." + sig[:-1] + ("0" if sig[-1] != "0" else "1"), sig + "." + tok):
            self.assertFalse(a.check(bad), bad)
        other = A.Auth("TEST-PASS-CODE-0000", b"z" * 48)     # different signing secret cannot validate
        self.assertFalse(other.check(c))

    def test_expiry_and_logout_invalidate_server_side(self):
        a, clock = mk_auth(ttl=120); c = a.login("TEST-PASS-CODE-0000", "c")
        clock.t += 119; self.assertTrue(a.check(c))
        clock.t += 2; self.assertFalse(a.check(c))
        c2 = a.login("TEST-PASS-CODE-0000", "c"); self.assertTrue(a.check(c2)); a.logout(c2); self.assertFalse(a.check(c2))   # valid signature, revoked session
        for bad_ttl in (0, 59, 86401):
            with self.assertRaises(A.AuthConfigError):
                A.Auth("p" * 20, b"k" * 40, ttl=bad_ttl)

    def test_rate_limit_per_client_then_recovery(self):
        a, clock = mk_auth(fail_limit=3, lockout=60)
        for _ in range(3):
            with self.assertRaises(A.AuthError):
                a.login("bad", "9.9.9.9")
        with self.assertRaises(A.AuthError) as cm:
            a.login("TEST-PASS-CODE-0000", "9.9.9.9")     # correct passcode is refused while locked out
        self.assertEqual(cm.exception.status, 429); self.assertGreaterEqual(cm.exception.retry_after, 1)
        self.assertTrue(a.check(a.login("TEST-PASS-CODE-0000", "8.8.8.8")))   # other clients unaffected
        clock.t += 61; self.assertTrue(a.check(a.login("TEST-PASS-CODE-0000", "9.9.9.9")))

    def test_global_lockout_defeats_rotating_client_addresses(self):
        a, clock = mk_auth(fail_limit=100, global_fail_limit=6, lockout=60)
        for i in range(6):
            with self.assertRaises(A.AuthError):
                a.login("bad", f"10.0.0.{i}")
        with self.assertRaises(A.AuthError) as cm:
            a.login("TEST-PASS-CODE-0000", "10.0.0.99")
        self.assertEqual(cm.exception.status, 429)

    def test_session_table_is_bounded(self):
        a, _ = mk_auth(max_sessions=5)
        cs = [a.login("TEST-PASS-CODE-0000", "c") for _ in range(9)]
        self.assertLessEqual(len(a._sessions), 5); self.assertTrue(a.check(cs[-1]))

    def test_client_key_trusts_forwarded_header_only_from_loopback(self):
        self.assertEqual(A.client_key("127.0.0.1", "203.0.113.7"), "203.0.113.7")
        self.assertEqual(A.client_key("198.51.100.4", "203.0.113.7"), "198.51.100.4")
        self.assertEqual(A.client_key("127.0.0.1", "not-an-ip"), "127.0.0.1")

    def test_constant_time_comparisons_are_used(self):
        src = (ROOT / "auth.py").read_text()
        self.assertGreaterEqual(src.count("hmac.compare_digest"), 2)   # passcode digest + cookie signature
        self.assertNotIn("== candidate", src); self.assertNotIn("== passcode", src)


class Secrets(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tell-sec-")); self.d = self.tmp / "secrets"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generation_permissions_independence_and_no_overwrite(self):
        made = A.write_secrets(self.d)
        self.assertEqual(sorted(made), ["demo_passcode", "session_secret"])
        self.assertEqual(stat.S_IMODE(self.d.stat().st_mode), 0o700)
        for n in made:
            self.assertEqual(stat.S_IMODE((self.d / n).stat().st_mode), 0o600)
        pc, key = A.load_secrets(self.d)
        self.assertRegex(pc, r"^[A-HJ-NP-Z2-9]{4}(-[A-HJ-NP-Z2-9]{4}){3}$"); self.assertGreaterEqual(len(key), 64); self.assertNotEqual(pc.encode(), key)
        self.assertEqual(A.write_secrets(self.d), [])                 # existing secrets are never overwritten silently
        self.assertEqual(A.load_secrets(self.d)[0], pc)
        A.write_secrets(self.d, rotate=True); self.assertNotEqual(A.load_secrets(self.d)[0], pc)
        self.assertNotEqual(A.generate_passcode(), A.generate_passcode())

    def test_loader_refuses_weak_or_unsafe_setups(self):
        A.write_secrets(self.d)
        def refuses(msg):
            with self.assertRaises(A.AuthConfigError, msg=msg):
                A.load_secrets(self.d)
        os.chmod(self.d, 0o755); refuses("dir perms"); os.chmod(self.d, 0o700)
        os.chmod(self.d / "demo_passcode", 0o644); refuses("file perms"); os.chmod(self.d / "demo_passcode", 0o600)
        (self.d / "demo_passcode").write_text("short\n"); refuses("short passcode")
        (self.d / "demo_passcode").write_text("SAME-VALUE-" * 5 + "\n"); (self.d / "session_secret").write_text("SAME-VALUE-" * 5 + "\n"); refuses("identical")
        (self.d / "session_secret").unlink(); refuses("missing")
        with self.assertRaises(A.AuthConfigError):
            A.load_secrets(self.tmp / "nonexistent")
        link = self.tmp / "link"; os.symlink(self.d, link)
        with self.assertRaises(A.AuthConfigError):
            A.load_secrets(link)

    def test_generator_script_never_prints_secrets(self):
        env = dict(os.environ)
        # run against a private copy so the real runtime_public/secrets is never touched
        work = self.tmp / "copy"; work.mkdir()
        for f in ("auth.py", "make_public_secrets.py"):
            shutil.copy(ROOT / f, work / f)
        out = subprocess.run([sys.executable, str(work / "make_public_secrets.py")], capture_output=True, text=True, timeout=20, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        pc = (work / "runtime_public" / "secrets" / "demo_passcode").read_text().strip()
        key = (work / "runtime_public" / "secrets" / "session_secret").read_text().strip()
        self.assertNotIn(pc, out.stdout + out.stderr); self.assertNotIn(key, out.stdout + out.stderr)


def server_cmd(rt, port=0, extra=(), data=PUBLIC, info="test-auth.json"):
    return [sys.executable, str(ROOT / "server.py"), "--host", "127.0.0.1", "--port", str(port), "--no-auto-port", "--data", str(data),
            "--info-file", info, "--stage-delay", "0", *extra]


class StartupPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tell-start-")); self.rt = self.tmp / "runtime_public"; A.write_secrets(self.rt / "secrets")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def refuse(self, args, needle, data=PUBLIC):
        p = subprocess.run(server_cmd(self.rt, extra=args, data=data), capture_output=True, text=True, timeout=20, cwd=str(ROOT))
        self.assertNotEqual(p.returncode, 0, args); self.assertIn(needle, p.stdout + p.stderr)
        self.assertNotIn("listening", p.stdout)
        return p

    def test_enable_intake_without_auth_is_refused(self):
        self.refuse(["--enable-intake", "--runtime-dir", str(self.rt)], "requires --require-auth")

    def test_auth_with_missing_or_weak_secrets_is_refused(self):
        self.refuse(["--enable-intake", "--require-auth", "--runtime-dir", str(self.tmp / "runtime_public_empty")], "refusing to start")
        os.chmod(self.rt / "secrets" / "session_secret", 0o644)
        self.refuse(["--enable-intake", "--require-auth", "--runtime-dir", str(self.rt)], "owner-only")
        os.chmod(self.rt / "secrets" / "session_secret", 0o600); (self.rt / "secrets" / "demo_passcode").write_text("tiny\n")
        self.refuse(["--enable-intake", "--require-auth", "--runtime-dir", str(self.rt)], "at least")

    def test_authenticated_intake_requires_sanitized_bundle_and_isolated_runtime(self):
        self.refuse(["--enable-intake", "--require-auth", "--runtime-dir", str(self.rt)], "sanitized public bundle", data=PRIVATE)
        self.refuse(["--enable-intake", "--require-auth", "--runtime-dir", str(ROOT / "runtime"), "--secrets-dir", str(self.rt / "secrets")], "runtime_public")
        other = self.tmp / "somewhere_else"; other.mkdir()
        self.refuse(["--enable-intake", "--require-auth", "--runtime-dir", str(other), "--secrets-dir", str(self.rt / "secrets")], "runtime_public")

    def test_session_ttl_out_of_range_is_refused(self):
        self.refuse(["--require-auth", "--runtime-dir", str(self.rt), "--session-ttl", "5"], "session lifetime")

    def test_defaults_never_enable_unauthenticated_intake(self):
        src = (ROOT / "server.py").read_text()
        self.assertIn("Handler.public = True            # intake stays disabled unless explicitly opted in", src)
        self.assertRegex(src, r"elif args\.public_demo or is_public_bundle or args\.require_auth:")


class LiveServer:
    def __init__(self, session_ttl=120, extra=()):
        self.tmp = Path(tempfile.mkdtemp(prefix="tell-live-")); self.rt = self.tmp / "runtime_public"
        A.write_secrets(self.rt / "secrets")
        self.passcode, key = A.load_secrets(self.rt / "secrets"); self.key = key.decode()
        (self.rt / "inbox").mkdir(parents=True); (self.rt / "inbox" / "inbox-sample.pdf").write_bytes(invoice("INBOX-1", "Inbox Supplier Ltd", "Total Due: USD 10.00"))
        self.info = f"test-auth-{os.getpid()}-{int(time.time() * 1000)}.json"
        self.log = open(self.tmp / "server.log", "w+")
        self.proc = subprocess.Popen(server_cmd(self.rt, extra=["--enable-intake", "--require-auth", "--runtime-dir", str(self.rt), "--session-ttl", str(session_ttl), *extra], info=self.info),
                                     stdout=self.log, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True)
        deadline = time.time() + 15; self.port = None
        while time.time() < deadline and self.port is None:
            m = re.search(r"listening on 127\.0\.0\.1:(\d+)", (self.tmp / "server.log").read_text())
            if m:
                self.port = int(m.group(1))
            elif self.proc.poll() is not None:
                raise RuntimeError("server exited: " + (self.tmp / "server.log").read_text())
            else:
                time.sleep(0.05)
        if self.port is None:
            raise RuntimeError("server did not start")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close(); (ROOT / "run" / self.info).unlink(missing_ok=True); shutil.rmtree(self.tmp, ignore_errors=True)

    def logtext(self):
        return (self.tmp / "server.log").read_text()

    def call(self, method, path, body=None, headers=None, cookie=None, raw=False):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        h = dict(headers or {})
        if cookie:
            h["Cookie"] = cookie
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode(); h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=body, headers=h); r = c.getresponse(); data = r.read(); hdrs = dict(r.getheaders()); c.close()
        if raw:
            return r.status, data, hdrs
        try:
            return r.status, json.loads(data), hdrs
        except ValueError:
            return r.status, data, hdrs

    def login(self, passcode=None, headers=None):
        h = {"X-Tell-Intake": "1"}; h.update(headers or {})
        st, body, hd = self.call("POST", "/auth/login", {"passcode": passcode or self.passcode}, h)
        cookie = hd.get("Set-Cookie", "").split(";")[0] if st == 200 else None
        return st, body, hd, cookie


class E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = LiveServer(); cls.lan_runtime_before = cls.snapshot_lan()

    @classmethod
    def tearDownClass(cls):
        cls.s.stop()

    @staticmethod
    def snapshot_lan():
        d = ROOT / "runtime"
        return sorted((p.name, p.stat().st_mtime_ns) for p in d.rglob("*") if p.is_file()) if d.exists() else []

    H = {"X-Tell-Intake": "1"}

    def authed(self):
        st, _, _, ck = self.s.login(); self.assertEqual(st, 200); return ck

    # -- unauthenticated surface
    def test_unauthenticated_requests_are_refused_everywhere_except_login_and_health(self):
        s = self.s
        for path in ("/api/traces", "/api/intake/status", "/api/intake/jobs", "/api/intake/jobs/" + "0" * 32, "/api/intake/scan/x", "/static/app.js", "/static/tell_core.js",
                     "/static/styles.css", "/static/index.html", "/data/traces_public_v1.json", "/server.py", "/../server.py", "/api/anything", "/index.html.bak", "/runtime_public/secrets/demo_passcode",
                     "/auth/session", "/auth/other.js"):
            st, body, _ = s.call("GET", path); self.assertEqual(st, 401, path)
            self.assertEqual(s.call("HEAD", path, raw=True)[0], 401, path)
        for m, path in (("POST", "/api/intake/scan"), ("POST", "/api/intake/batch"), ("POST", "/api/intake/reset"), ("PUT", "/api/intake/jobs/" + "0" * 32 + "/content"),
                        ("POST", "/api/intake/anything"), ("DELETE", "/api/intake/reset"), ("PATCH", "/x")):
            st, _, _ = s.call(m, path, b"{}", {**self.H, "Content-Type": "application/pdf"}); self.assertEqual(st, 401, (m, path))
        st, body, hd = s.call("GET", "/", raw=True); self.assertEqual((st, hd.get("Location")), (302, "/login")); self.assertNotIn(b"Operations dashboard", body)
        self.assertEqual(s.call("GET", "/index.html", raw=True)[0], 302)

    def test_only_login_assets_and_minimal_health_are_public(self):
        s = self.s
        st, body, _ = s.call("GET", "/healthz"); self.assertEqual((st, body), (200, {"status": "ok", "service": "tell-demo-ui"}))
        st, html, _ = s.call("GET", "/login", raw=True); self.assertEqual(st, 200); self.assertIn(b'type="password"', html)
        for p in ("/auth/login.js", "/auth/login.css"):
            st, b, _ = s.call("GET", p, raw=True); self.assertEqual(st, 200)
            self.assertNotIn(s.passcode.encode(), b); self.assertNotIn(s.key.encode(), b)
        self.assertNotIn(s.passcode.encode(), html)

    # -- login
    def test_wrong_passcode_and_csrf_guards(self):
        s = self.s
        st, body, hd, ck = s.login("WRONG-PASSCODE-XXXX"); self.assertEqual((st, ck), (401, None)); self.assertNotIn("Set-Cookie", hd)
        self.assertEqual(s.call("POST", "/auth/login", {"passcode": s.passcode})[0], 403)                                   # no CSRF header
        self.assertEqual(s.call("POST", "/auth/login", {"passcode": s.passcode}, {**self.H, "Origin": "http://evil.example"})[0], 403)   # cross-origin
        self.assertEqual(s.call("POST", "/auth/login", b"not json", self.H)[0], 400)
        self.assertEqual(s.call("POST", "/auth/login", {"passcode": ""}, self.H)[0], 400)
        self.assertEqual(s.call("POST", "/auth/login", {"passcode": 5}, self.H)[0], 400)

    def test_login_sets_secure_httponly_strict_cookie_and_unlocks_everything(self):
        s = self.s
        st, body, hd, ck = s.login(headers={"Origin": f"http://127.0.0.1:{s.port}"}); self.assertEqual(st, 200)
        sc = hd["Set-Cookie"]
        for attr in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/", "Max-Age=120"):
            self.assertIn(attr, sc)
        self.assertTrue(sc.startswith("__Host-tell_session=")); self.assertNotIn(s.passcode, sc); self.assertNotIn(s.key, sc)
        st, b, _ = s.call("GET", "/", cookie=ck, raw=True); self.assertEqual(st, 200); self.assertIn(b"<title>", b)
        st, tr, _ = s.call("GET", "/api/traces", cookie=ck); self.assertEqual((st, tr["public_demo"]), (200, True))
        st, status, _ = s.call("GET", "/api/intake/status", cookie=ck); self.assertEqual((st, status["enabled"], status["auth"]), (200, True, True))
        self.assertEqual(s.call("GET", "/login", cookie=ck, raw=True)[0], 302)                                              # already signed in

    def test_tampered_or_foreign_cookies_are_rejected(self):
        s = self.s; ck = self.authed(); name, _, val = ck.partition("=")
        tok, sig = val.split(".")
        for bad in (f"{name}={tok}.{'0' * 64}", f"{name}={'A' * len(tok)}.{sig}", f"{name}=garbage", "other=" + val, f"{name}={val}x"):
            self.assertEqual(s.call("GET", "/api/traces", cookie=bad)[0], 401, bad)

    def test_logout_revokes_the_session_server_side(self):
        s = self.s; ck = self.authed()
        self.assertEqual(s.call("GET", "/api/traces", cookie=ck)[0], 200)
        st, _, hd = s.call("POST", "/auth/logout", {}, self.H, cookie=ck); self.assertEqual(st, 200); self.assertIn("Max-Age=0", hd["Set-Cookie"])
        self.assertEqual(s.call("GET", "/api/traces", cookie=ck)[0], 401)                    # the old, correctly signed cookie no longer works

    # -- the actual purpose: intake through the authenticated public server
    def test_authenticated_multi_pdf_upload_extraction_detail_scan_and_reset(self):
        s = self.s; ck = self.authed(); hh = {**self.H, "Origin": f"http://127.0.0.1:{s.port}"}
        files = [("a.pdf", invoice("PUB-1", "Public Supplier A", "Total Due: EUR 120.00")), ("b.pdf", invoice("PUB-2", "Public Supplier B", "Total Due: USD 55.50")), ("scan.pdf", make_pdf(shapes=True)), ("x.txt", b"nope")]
        st, r, _ = s.call("POST", "/api/intake/batch", {"files": [{"name": n, "size": len(d), "type": "application/pdf"} for n, d in files]}, hh, cookie=ck); self.assertEqual(st, 201)
        self.assertEqual([j["status"] for j in r["jobs"]], ["UPLOADING"] * 4)
        for j, (n, d) in zip(r["jobs"], files):
            st, _, _ = s.call("PUT", f"/api/intake/jobs/{j['job_id']}/content", d, {**hh, "Content-Type": "application/pdf"}, cookie=ck); self.assertEqual(st, 202)
        deadline = time.time() + 15
        while time.time() < deadline:
            jobs = s.call("GET", "/api/intake/jobs", cookie=ck)[1]["jobs"]
            if not any(j["status"] in ("UPLOADING", "VALIDATING", "EXTRACTING") for j in jobs):
                break
            time.sleep(0.1)
        by = {j["display_name"]: j for j in jobs}
        self.assertEqual((by["a.pdf"]["status"], by["b.pdf"]["status"], by["scan.pdf"]["status"], by["x.txt"]["status"]), ("READY_FOR_PROCESSING", "READY_FOR_PROCESSING", "NEEDS_OCR", "FAILED"))
        d = s.call("GET", "/api/intake/jobs/" + by["a.pdf"]["job_id"], cookie=ck)[1]
        self.assertEqual(d["extraction"]["fields"]["invoice_number"]["value"], "PUB-1"); self.assertIn("Public Supplier A", d["extraction"]["text"])
        st, scan, _ = s.call("POST", "/api/intake/scan", {}, hh, cookie=ck); self.assertEqual(st, 202)
        for _ in range(100):
            sc = s.call("GET", "/api/intake/scan/" + scan["scan_id"], cookie=ck)[1]
            if sc["state"] == "complete":
                break
            time.sleep(0.1)
        self.assertEqual((sc["imported"], sc["rejected"]), (1, 0))
        self.assertEqual(s.call("POST", "/api/intake/reset", {}, hh, cookie=ck)[0], 200)
        self.assertEqual(s.call("GET", "/api/intake/jobs", cookie=ck)[1]["jobs"], [])
        self.assertTrue((s.rt / "inbox" / "inbox-sample.pdf").exists())                         # reset never touches the inbox
        self.assertEqual(list((s.rt / "uploads").glob("*")), [])

    def test_authenticated_mutations_still_need_csrf_header_and_same_origin(self):
        s = self.s; ck = self.authed()
        self.assertEqual(s.call("POST", "/api/intake/scan", {}, cookie=ck)[0], 403)
        self.assertEqual(s.call("POST", "/api/intake/reset", {}, {**self.H, "Origin": "http://evil.example"}, cookie=ck)[0], 403)

    def test_public_data_only_and_lan_preview_runtime_untouched(self):
        s = self.s; ck = self.authed()
        blob = s.call("GET", "/api/traces", cookie=ck, raw=True)[1].decode()
        priv = json.loads(PRIVATE.read_text()); self.assertNotIn(priv["source"]["records_sha256"], blob)
        for pat in (r"SIM-[A-Z]+-[0-9A-F]{12}", r"(?:lv22|lms1)-[0-9a-f]{16}", r"/home/", r"results/lora"):
            self.assertIsNone(re.search(pat, blob), pat)
        self.assertEqual(self.snapshot_lan(), self.lan_runtime_before)                         # the 8083 preview's runtime/ was not written

    def test_secrets_never_appear_in_logs_or_any_response(self):
        s = self.s; ck = self.authed()
        s.login("WRONG-PASSCODE-XXXX"); s.call("GET", "/api/traces", cookie=ck)
        log = s.logtext()
        for secret in (s.passcode, s.key, ck.split("=", 1)[1], ck.split("=", 1)[1].split(".")[0], "WRONG-PASSCODE-XXXX"):
            self.assertNotIn(secret, log)
        for p in ("/api/traces", "/api/intake/status", "/api/intake/jobs", "/", "/static/app.js", "/static/tell_core.js"):
            b = s.call("GET", p, cookie=ck, raw=True)[1]
            self.assertNotIn(s.passcode.encode(), b, p); self.assertNotIn(s.key.encode(), b, p)
        self.assertNotIn("Cookie", log)

    def test_security_headers_present_on_auth_responses(self):
        st, _, hd = self.s.call("GET", "/login", raw=True)
        self.assertIn("default-src 'self'", hd["Content-Security-Policy"]); self.assertEqual(hd["Cache-Control"], "no-store"); self.assertEqual(hd["X-Content-Type-Options"], "nosniff")


class LockoutOverHttp(unittest.TestCase):
    def test_repeated_wrong_passcodes_lock_the_client_out_even_for_the_right_one(self):
        s = LiveServer()
        try:
            for _ in range(5):
                self.assertEqual(s.login("WRONG-PASSCODE-XXXX")[0], 401)
            st, body, hd, _ = s.login(); self.assertEqual(st, 429); self.assertIn("Retry-After", hd)
            self.assertEqual(s.call("GET", "/api/traces")[0], 401)
        finally:
            s.stop()

    def test_expired_session_is_rejected_over_http(self):
        s = LiveServer(session_ttl=60)
        try:
            st, _, hd, ck = s.login(); self.assertEqual(st, 200); self.assertIn("Max-Age=60", hd["Set-Cookie"])
            self.assertEqual(s.call("GET", "/api/traces", cookie=ck)[0], 200)
        finally:
            s.stop()


class ReadOnlyAuthMode(unittest.TestCase):
    def test_require_auth_alone_keeps_intake_disabled(self):
        tmp = Path(tempfile.mkdtemp(prefix="tell-ro-")); rt = tmp / "runtime_public"; A.write_secrets(rt / "secrets")
        info = f"test-ro-{os.getpid()}.json"; log = open(tmp / "log", "w+")
        p = subprocess.Popen(server_cmd(rt, extra=["--require-auth", "--runtime-dir", str(rt)], info=info), stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True)
        try:
            port = None
            for _ in range(200):
                m = re.search(r"listening on 127\.0\.0\.1:(\d+)", (tmp / "log").read_text())
                if m:
                    port = int(m.group(1)); break
                time.sleep(0.05)
            self.assertIsNotNone(port)
            pc = A.load_secrets(rt / "secrets")[0]
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("POST", "/auth/login", body=json.dumps({"passcode": pc}), headers={"X-Tell-Intake": "1", "Content-Type": "application/json"}); r = c.getresponse(); r.read()
            ck = r.getheader("Set-Cookie").split(";")[0]; c.close()
            for m_, path in (("GET", "/api/intake/status"), ("POST", "/api/intake/scan")):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request(m_, path, body=b"{}" if m_ == "POST" else None, headers={"Cookie": ck, "X-Tell-Intake": "1"}); r = c.getresponse(); body = r.read(); c.close()
                self.assertIn(r.status, (200, 403)); self.assertNotEqual(json.loads(body).get("enabled"), True)
            self.assertFalse((rt / "uploads").exists())          # no intake service was created
        finally:
            p.terminate(); p.wait(10); log.close(); (ROOT / "run" / info).unlink(missing_ok=True); shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
