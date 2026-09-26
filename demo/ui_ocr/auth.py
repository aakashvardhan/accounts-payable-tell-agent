"""Application-level authentication for the public demo server. Stdlib only, CPU only.

Model: one shared demo passcode + a separate session-signing secret, both read from owner-only files.
A login yields a random server-side session token; the cookie carries `token.HMAC(key, token)` so a forged or tampered
cookie is rejected before any lookup, and logout/expiry invalidate it server-side. Failed logins are rate limited per client
and globally. Secrets are never logged, never placed in URLs/HTML/JS, and never returned by any endpoint.
"""
import hashlib
import hmac
import ipaddress
import os
import secrets
import stat
import threading
import time
from pathlib import Path

COOKIE_NAME = "__Host-tell_session"     # __Host- prefix: browser enforces Secure, Path=/, no Domain
PASSCODE_FILE, KEY_FILE = "demo_passcode", "session_secret"
MIN_PASSCODE, MIN_KEY = 16, 32
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I: easy to read aloud and type


class AuthConfigError(Exception):
    pass


class AuthError(Exception):
    def __init__(self, message, status=401, retry_after=None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


def generate_passcode():
    """~100 bits: four groups of four characters from an unambiguous alphabet."""
    return "-".join("".join(secrets.choice(ALPHABET) for _ in range(4)) for _ in range(4))


def write_secrets(directory, rotate=False):
    """Create <directory>/ (0700) with two independent random secrets (0600). Refuses to overwrite unless rotate=True."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    made = []
    for name, value in ((PASSCODE_FILE, generate_passcode()), (KEY_FILE, secrets.token_hex(48))):
        p = d / name
        if p.exists() and not rotate:
            continue
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(value + "\n")
        os.chmod(p, 0o600)
        made.append(name)
    return made


def load_secrets(directory):
    """Validate and read (passcode, key_bytes). Any weakness -> AuthConfigError (the server then refuses to start)."""
    d = Path(directory)
    if d.is_symlink() or not d.is_dir():
        raise AuthConfigError(f"secrets directory missing or not a real directory: {d}")
    if stat.S_IMODE(d.stat().st_mode) & 0o077 or d.stat().st_uid != os.getuid():
        raise AuthConfigError("secrets directory must be owner-only (0700) and owned by the server user")
    vals = {}
    for name in (PASSCODE_FILE, KEY_FILE):
        p = d / name
        if p.is_symlink() or not p.is_file():
            raise AuthConfigError(f"secret file missing: {name}")
        if stat.S_IMODE(p.stat().st_mode) & 0o077:
            raise AuthConfigError(f"secret file {name} must be owner-only (0600)")
        vals[name] = p.read_text().strip()
    if len(vals[PASSCODE_FILE]) < MIN_PASSCODE:
        raise AuthConfigError(f"passcode must be at least {MIN_PASSCODE} characters")
    if len(vals[KEY_FILE]) < MIN_KEY:
        raise AuthConfigError(f"session secret must be at least {MIN_KEY} characters")
    if vals[PASSCODE_FILE] == vals[KEY_FILE]:
        raise AuthConfigError("passcode and session secret must be independent values")
    return vals[PASSCODE_FILE], vals[KEY_FILE].encode()


def client_key(peer, cf_header=None):
    """Rate-limit key. Behind the local tunnel the TCP peer is loopback, so trust CF-Connecting-IP only then."""
    try:
        if ipaddress.ip_address(peer).is_loopback and cf_header:
            return str(ipaddress.ip_address(cf_header.strip()))
    except ValueError:
        pass
    return peer


class Auth:
    def __init__(self, passcode, key, ttl=3600, clock=time.time, max_sessions=200,
                 fail_limit=5, window=300, lockout=300, global_fail_limit=40):
        if not (60 <= ttl <= 86400):
            raise AuthConfigError("session lifetime must be between 60 and 86400 seconds")
        self._key = key
        self._pc = hmac.new(key, passcode.encode(), hashlib.sha256).digest()   # compare digests: fixed length, constant time
        self.ttl, self.clock, self.max_sessions = ttl, clock, max_sessions
        self.fail_limit, self.window, self.lockout, self.global_fail_limit = fail_limit, window, lockout, global_fail_limit
        self._sessions = {}
        self._fails = {}        # client -> [timestamps]
        self._gfails = []
        self._locked = {}       # client|"*" -> until
        self._lock = threading.Lock()

    # -- helpers
    def _sig(self, token):
        return hmac.new(self._key, token.encode(), hashlib.sha256).hexdigest()

    def _prune(self, now):
        for t in [t for t, exp in self._sessions.items() if exp <= now]:
            del self._sessions[t]
        self._gfails = [t for t in self._gfails if now - t < self.window]
        for k in list(self._fails):
            self._fails[k] = [t for t in self._fails[k] if now - t < self.window]
            if not self._fails[k]:
                del self._fails[k]
        for k in [k for k, u in self._locked.items() if u <= now]:
            del self._locked[k]

    # -- login / session
    def login(self, candidate, client):
        now = self.clock()
        with self._lock:
            self._prune(now)
            for k in (client, "*"):
                if k in self._locked:
                    raise AuthError("too many failed attempts; try again later", 429, max(1, int(self._locked[k] - now)))
            ok = hmac.compare_digest(hmac.new(self._key, str(candidate).encode(), hashlib.sha256).digest(), self._pc)
            if not ok:
                self._fails.setdefault(client, []).append(now); self._gfails.append(now)
                if len(self._fails[client]) >= self.fail_limit:
                    self._locked[client] = now + self.lockout
                if len(self._gfails) >= self.global_fail_limit:
                    self._locked["*"] = now + self.lockout
                raise AuthError("invalid passcode", 401)
            self._fails.pop(client, None)
            if len(self._sessions) >= self.max_sessions:   # bounded memory: evict the oldest
                del self._sessions[min(self._sessions, key=self._sessions.get)]
            token = secrets.token_urlsafe(32)
            self._sessions[token] = now + self.ttl
            return f"{token}.{self._sig(token)}"

    def check(self, cookie_value):
        if not cookie_value or "." not in cookie_value:
            return False
        token, _, sig = cookie_value.partition(".")
        if not hmac.compare_digest(sig, self._sig(token)):
            return False
        now = self.clock()
        with self._lock:
            exp = self._sessions.get(token)
            if exp is None or exp <= now:
                self._sessions.pop(token, None)
                return False
            return True

    def logout(self, cookie_value):
        if cookie_value and "." in cookie_value:
            with self._lock:
                self._sessions.pop(cookie_value.partition(".")[0], None)

    def set_cookie(self, value):
        return f"{COOKIE_NAME}={value}; Path=/; Max-Age={self.ttl}; HttpOnly; Secure; SameSite=Strict"

    @staticmethod
    def clear_cookie():
        return f"{COOKIE_NAME}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict"

    @staticmethod
    def cookie_from(header):
        for part in (header or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE_NAME:
                return v
        return None
