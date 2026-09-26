#!/usr/bin/env python3
"""Create the public-demo secrets under runtime_public/secrets/ (dir 0700, files 0600). Never prints a secret.
  python3 make_public_secrets.py            # creates missing files only
  python3 make_public_secrets.py --rotate   # replaces both (invalidates every session)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auth

if __name__ == "__main__":
    d = Path(__file__).resolve().parent / "runtime_public" / "secrets"
    made = auth.write_secrets(d, rotate="--rotate" in sys.argv)
    auth.load_secrets(d)   # self-check: permissions, length, independence
    print(f"secrets ready in {d} ({'created: ' + ', '.join(made) if made else 'already present, unchanged'}); values not printed")
