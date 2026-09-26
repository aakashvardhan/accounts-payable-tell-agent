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
DATA = HERE / "data" / "traces_v1.json"
RUN = HERE / "run"
RESERVED = {8080}
PREFERRED = 8081
sys.path.insert(0, str(HERE))
import trace_contract as C  # noqa: E402

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
                               "traces": self.server.n_traces, "live_inference": False,
                               "uptime_s": round(time.time() - STARTED, 1), "port": self.server.server_address[1]}).encode()
            return self._send(200, body, "application/json", head)
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

    def do_GET(self):
        self._route(False)

    def do_HEAD(self):
        self._route(True)

    def _deny(self):
        self._send(405, b'{"error":"read-only interface"}', "application/json", extra={"Allow": "GET, HEAD"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _deny


def bind(port, auto, host="0.0.0.0"):
    candidates = [port] if not auto else [p for p in range(port, port + 20) if p not in RESERVED]
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
    ap.add_argument("--host", default="0.0.0.0", help="bind address (use 127.0.0.1 behind a tunnel)")
    ap.add_argument("--data", default=str(DATA), help="trace bundle to serve")
    ap.add_argument("--info-file", default="server.json", help="runtime info file name under run/")
    args = ap.parse_args()
    bundle = load_bundle(args.data)
    Handler.bundle_bytes = json.dumps(bundle).encode()
    srv, skipped = bind(args.port, not args.no_auto_port, args.host)
    srv.n_traces = len(bundle["traces"])
    port = srv.server_address[1]
    ip = lan_ip()
    for p, why in skipped:
        print(f"port {p} unavailable ({why}); using next safe port", flush=True)
    info = {"pid": os.getpid(), "port": port, "bind": args.host, "data": str(Path(args.data).name), "lan_ip": ip,
            "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "port_fallback_reasons": [f"{p}: {w}" for p, w in skipped]}
    RUN.mkdir(exist_ok=True)
    (RUN / args.info_file).write_text(json.dumps(info, indent=1))
    print(f"Tell demo UI listening on {args.host}:{port} (pid {os.getpid()})", flush=True)
    print(f"  Nano-local:   http://127.0.0.1:{port}", flush=True)
    print(f"  Same-network: http://{ip}:{port}" if ip else "  Same-network: (no active IPv4 found)", flush=True)
    print("  Local-network URL only — not internet-public.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
