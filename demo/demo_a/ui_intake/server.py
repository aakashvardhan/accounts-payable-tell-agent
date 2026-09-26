#!/usr/bin/env python3
"""Tell demo UI server: stdlib only, CPU only, serves local static assets + the trace bundle.

Binds 0.0.0.0 (local-network reachable, NOT an internet tunnel). Preferred port 8081; if occupied it
picks the next free port, never 8080 (reserved for the future model/API service). GET/HEAD only.
"""
import argparse
import http.server
import json
import mimetypes
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
AUTH_STATIC = HERE / "auth_static"   # the only assets served before login
DATA = HERE / "data" / "traces_v1.json"
RUN = HERE / "run"
RESERVED = {8080}          # future model/API service: never bound
AVOID_AUTO = {8081, 8082}   # other running demo instances: skipped by automatic fallback, allowed when requested explicitly
PREFERRED = 8083
sys.path.insert(0, str(HERE))
import trace_contract as C  # noqa: E402
import intake as I  # noqa: E402
import auth as A  # noqa: E402

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
STARTED = time.time()


def lan_ip():
    """Active outbound IPv4 (no packet is sent by a UDP connect)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9)); return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def load_bundle(path=None):
    return C.validate_bundle(json.loads(Path(path or DATA).read_text()))


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "TellDemo/1"
    bundle_bytes = b""
    intake = None      # IntakeService (None in public-demo mode)
    public = True      # fail closed: intake mutations are refused unless explicitly enabled
    auth = None        # auth.Auth when --require-auth: every route except /login, /auth/login.*, /healthz needs a session
    JSON_CAP = 64 * 1024

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % a))

    def _send(self, code, body, ctype, head=False, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _route(self, head):
        path = unquote(urlparse(self.path).path)
        if path == "/healthz":
            body = json.dumps({"status": "ok", "service": "tell-demo-ui", "contract": C.CONTRACT_VERSION,
                               "traces": self.server.n_traces, "live_inference": False, "intake_enabled": not self.public,
                               "uptime_s": round(time.time() - STARTED, 1), "port": self.server.server_address[1]}).encode()
            return self._send(200, body, "application/json", head)
        if path.startswith("/api/intake"):
            return self._intake("GET", path, head)
        if path == "/api/traces":
            return self._send(200, self.bundle_bytes, "application/json", head)
        if path in ("/", "/index.html"):
            path = "/index.html"
        if path.startswith("/static/"):
            path = path[len("/static"):]
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._send(404, b'{"error":"not found"}', "application/json", head)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype, head)

    # ---- authentication gate (only active with --require-auth)
    def _origin_ok(self):
        origin = self.headers.get("Origin")
        return not origin or urlparse(origin).netloc == self.headers.get("Host")

    def _gate(self, method):
        """True -> continue to normal routing. False -> a response has already been sent."""
        if self.auth is None:
            return True
        path = unquote(urlparse(self.path).path)
        head = method == "HEAD"
        if method in ("GET", "HEAD"):
            if path == "/healthz":   # minimal: no counts, ports or mode
                self._send(200, b'{"status":"ok","service":"tell-demo-ui"}', "application/json", head); return False
            if path == "/login":
                if self.auth.check(A.Auth.cookie_from(self.headers.get("Cookie"))):
                    self._send(302, b"", "text/plain", head, {"Location": "/"}); return False
                self._send(200, (AUTH_STATIC / "login.html").read_bytes(), "text/html; charset=utf-8", head); return False
            if path in ("/auth/login.css", "/auth/login.js"):
                ctype = "text/css; charset=utf-8" if path.endswith(".css") else "application/javascript; charset=utf-8"
                self._send(200, (AUTH_STATIC / path.rsplit("/", 1)[1]).read_bytes(), ctype, head); return False
        if method == "POST" and path == "/auth/login":
            self._login(); return False
        if method == "POST" and path == "/auth/logout":
            self.auth.logout(A.Auth.cookie_from(self.headers.get("Cookie")))
            self._send(200, b'{"ok":true}', "application/json", extra={"Set-Cookie": A.Auth.clear_cookie()}); return False
        if self.auth.check(A.Auth.cookie_from(self.headers.get("Cookie"))):
            return True
        if method != "GET" and method != "HEAD":
            self.close_connection = True   # never read an unauthenticated body
        if method in ("GET", "HEAD") and path in ("/", "/index.html"):
            self._send(302, b"", "text/plain", head, {"Location": "/login"})
        else:
            self._send(401, b'{"error":"authentication required"}', "application/json", head)
        return False

    def _login(self):
        if self.headers.get("X-Tell-Intake") != "1" or not self._origin_ok():
            self.close_connection = True
            return self._json(403, {"error": "request refused"})
        raw = self._body(1024)
        if raw is None:
            return
        try:
            cand = json.loads(raw).get("passcode")
        except (ValueError, AttributeError):
            cand = None
        if not isinstance(cand, str) or not cand or len(cand) > 256:
            return self._json(400, {"error": "passcode required"})
        try:
            cookie = self.auth.login(cand, A.client_key(self.client_address[0], self.headers.get("CF-Connecting-IP")))
        except A.AuthError as e:
            return self._send(e.status, json.dumps({"error": str(e)}).encode(), "application/json",
                              extra={"Retry-After": str(e.retry_after)} if e.retry_after else None)
        self._send(200, b'{"ok":true}', "application/json", extra={"Set-Cookie": self.auth.set_cookie(cookie)})

    def do_GET(self):
        if self._gate("GET"):
            self._route(False)

    def do_HEAD(self):
        if self._gate("HEAD"):
            self._route(True)

    # ---- intake API (local mode only; every mutation fails closed in public-demo mode)
    def _json(self, code, obj, head=False):
        self._send(code, json.dumps(obj).encode(), "application/json", head)

    def _guard_mutation(self):
        if self.public or self.intake is None:
            self.close_connection = True
            self._json(403, {"error": "intake is disabled in public-demo mode"}); return False
        if self.headers.get("X-Tell-Intake") != "1":
            self.close_connection = True
            self._json(403, {"error": "missing X-Tell-Intake header"}); return False
        if not self._origin_ok():
            self.close_connection = True
            self._json(403, {"error": "cross-origin request refused"}); return False
        return True

    def _body(self, cap):
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._json(411, {"error": "Content-Length required"}); return None
        if n < 0 or n > cap:
            self.close_connection = True
            self._json(413, {"error": f"body exceeds {cap} bytes"}); return None
        data = b""
        while len(data) < n:
            chunk = self.rfile.read(min(65536, n - len(data)))
            if not chunk:
                break
            data += chunk
        return data

    def _intake(self, method, path, head=False):
        parts = [p for p in path.split("/") if p][2:]   # after api/intake
        if method == "GET" and parts == ["status"] and (self.public or self.intake is None):
            return self._json(200, {"enabled": False, "reason": "public-demo mode: intake is disabled"}, head)
        if method != "GET" and not self._guard_mutation():
            return
        if self.public or self.intake is None:
            return self._json(403, {"error": "intake is disabled in public-demo mode"}, head)
        svc = self.intake
        try:
            if method == "GET" and parts == ["status"]:
                st = svc.status(); st["auth"] = self.auth is not None
                return self._json(200, st, head)
            if method == "GET" and parts == ["jobs"]:
                return self._json(200, {"jobs": svc.list_jobs()}, head)
            if method == "GET" and len(parts) == 2 and parts[0] == "jobs":
                return self._json(200, svc.get_job(parts[1]), head)
            if method == "GET" and len(parts) == 2 and parts[0] == "scan":
                return self._json(200, svc.get_scan(parts[1]), head)
            if method == "POST" and parts == ["batch"]:
                raw = self._body(self.JSON_CAP)
                if raw is None:
                    return
                try:
                    manifest = json.loads(raw).get("files")
                except (ValueError, AttributeError):
                    raise I.IntakeError("body must be JSON {\"files\": [...]}")
                return self._json(201, {"jobs": svc.create_batch(manifest)})
            if method == "PUT" and len(parts) == 3 and parts[0] == "jobs" and parts[2] == "content":
                svc.get_job(parts[1])   # 404 before reading any body
                raw = self._body(svc.cfg.max_bytes + 1)
                if raw is None:
                    return
                return self._json(202, svc.receive_content(parts[1], self.headers.get("Content-Type", ""), raw))
            if method == "POST" and parts == ["scan"]:
                self._body(self.JSON_CAP)
                return self._json(202, svc.scan())
            if method == "POST" and parts == ["reset"]:
                self._body(self.JSON_CAP)
                return self._json(200, svc.reset())
        except I.IntakeError as e:
            return self._json(e.status, {"error": str(e)})
        return self._json(404, {"error": "not found"}, head)

    def do_POST(self):
        if not self._gate("POST"):
            return
        p = unquote(urlparse(self.path).path)
        if p.startswith("/api/intake"):
            return self._intake("POST", p)
        self._deny()

    def do_PUT(self):
        if not self._gate("PUT"):
            return
        p = unquote(urlparse(self.path).path)
        if p.startswith("/api/intake"):
            return self._intake("PUT", p)
        self._deny()

    def _deny(self):
        if not self._gate(self.command):
            return
        self.close_connection = True
        self._send(405, b'{"error":"read-only interface"}', "application/json", extra={"Allow": "GET, HEAD"})

    do_DELETE = do_PATCH = _deny


def bind(port, auto, host="0.0.0.0"):
    candidates = [port] if not auto else [p for p in range(port, port + 20) if p not in RESERVED | AVOID_AUTO]
    skipped = []
    for p in candidates:
        if p in RESERVED:
            raise SystemExit(f"port {p} is reserved for the future model/API service")
        try:
            srv = http.server.ThreadingHTTPServer((host, p), Handler)
            srv.daemon_threads = True
            return srv, skipped
        except OSError as e:
            skipped.append((p, str(e)))
    raise SystemExit("no free port: %r" % (skipped,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PREFERRED)
    ap.add_argument("--no-auto-port", action="store_true")
    ap.add_argument("--public-demo", action="store_true", help="force read-only mode (no intake); implied by a public_demo bundle")
    ap.add_argument("--enable-intake", action="store_true", help="explicit opt-in: enable upload/scan/reset on a sanitized public bundle; requires --require-auth")
    ap.add_argument("--require-auth", action="store_true", help="require a passcode session for every route except /login and /healthz")
    ap.add_argument("--secrets-dir", default=None, help="dir with demo_passcode + session_secret (default: <runtime-dir>/secrets)")
    ap.add_argument("--session-ttl", type=int, default=3600, help="session lifetime in seconds (60-86400)")
    ap.add_argument("--runtime-dir", default=str(HERE / "runtime"), help="intake job registry, uploads and demo inbox")
    ap.add_argument("--stage-delay", type=float, default=0.7, help="seconds of visible pacing between intake stages")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (use 127.0.0.1 behind a tunnel)")
    ap.add_argument("--data", default=str(DATA), help="trace bundle to serve")
    ap.add_argument("--info-file", default="server.json", help="runtime info file name under run/")
    args = ap.parse_args()
    bundle = load_bundle(args.data)
    is_public_bundle = bool(bundle.get("public_demo"))
    runtime = Path(args.runtime_dir).resolve()
    # ---- authentication / intake policy (fail closed; never a silent fallback to unauthenticated intake)
    if args.enable_intake and not args.require_auth:
        raise SystemExit("refusing to start: --enable-intake requires --require-auth")
    if args.require_auth:
        try:
            passcode, key = A.load_secrets(args.secrets_dir or runtime / "secrets")
            Handler.auth = A.Auth(passcode, key, ttl=args.session_ttl)
        except A.AuthConfigError as e:
            raise SystemExit(f"refusing to start: {e}")
        del passcode, key
    if args.enable_intake:
        if not is_public_bundle:
            raise SystemExit("refusing to start: authenticated intake must serve the sanitized public bundle (public_demo)")
        if runtime.name != "runtime_public" or runtime == (HERE / "runtime").resolve():
            raise SystemExit("refusing to start: authenticated public intake must use an isolated runtime_public/ directory")
    Handler.bundle_bytes = json.dumps(bundle).encode()
    if args.enable_intake:
        Handler.public = False
    elif args.public_demo or is_public_bundle or args.require_auth:
        Handler.public = True            # intake stays disabled unless explicitly opted in
    else:
        Handler.public = False           # legacy local/LAN preview behaviour (unchanged)
    Handler.intake = None if Handler.public else I.IntakeService(I.IntakeConfig(root=runtime, stage_delay=args.stage_delay))
    srv, skipped = bind(args.port, not args.no_auto_port, args.host)
    srv.n_traces = len(bundle["traces"])
    port = srv.server_address[1]
    ip = lan_ip()
    for p, why in skipped:
        print(f"port {p} unavailable ({why}); using next safe port", flush=True)
    info = {"pid": os.getpid(), "port": port, "bind": args.host, "data": str(Path(args.data).name), "lan_ip": ip,
            "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "port_fallback_reasons": [f"{p}: {w}" for p, w in skipped],
            "auth": "required" if Handler.auth else "none",
            "mode": ("public-demo (read-only)" if Handler.public else "local (intake enabled)") + (" + passcode auth" if Handler.auth else "")}
    RUN.mkdir(exist_ok=True)
    (RUN / args.info_file).write_text(json.dumps(info, indent=1))
    print(f"Tell demo UI listening on {args.host}:{port} (pid {os.getpid()})", flush=True)
    print(f"  Nano-local:   http://127.0.0.1:{port}", flush=True)
    print(f"  Same-network: http://{ip}:{port}" if ip else "  Same-network: (no active IPv4 found)", flush=True)
    print("  Authentication: " + (f"passcode required (session {args.session_ttl}s)" if Handler.auth else "none (local preview)"), flush=True)
    print("  Local-network URL only — not internet-public." if not Handler.auth else "  Intended to sit behind the existing tunnel; secrets are never logged.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
