"""Test-only stand-in for the Tesseract CLI. It really opens the rendered page image with Pillow (so a test proves the OCR stage
consumed a genuine 300-DPI rasterisation) and prints Tesseract-style TSV. It is NOT an OCR engine and is never used at runtime."""
import json
import os
import stat
import sys
from pathlib import Path

TEMPLATE = r'''#!{py}
import json, sys, time, re
CFG = json.loads({cfg!r})
if "--version" in sys.argv:
    print("tesseract 5.9.9-testdouble"); sys.exit(0)
img = sys.argv[1]
from PIL import Image
im = Image.open(img); w, h = im.size
m = re.search(r"page-(\d+)", img); page = int(m.group(1)) if m else 1
mode = CFG["mode"]
if mode == "fail": sys.stderr.write("boom"); sys.exit(2)
if mode == "sleep": time.sleep(60)
if mode == "malformed": print("this is not tsv"); sys.exit(0)
if mode == "badconf": print("\t".join(["level","page_num","block_num","par_num","line_num","word_num","left","top","width","height","conf","text"])); print("5\t1\t1\t1\t1\t1\t0\t0\t1\t1\tNaNx\tword"); sys.exit(0)
HEAD = ["level","page_num","block_num","par_num","line_num","word_num","left","top","width","height","conf","text"]
print("\t".join(HEAD))
lines = list(CFG.get("lines", []))
if CFG.get("emit_dims"): lines.append([["PAGEPX%dx%d" % (w, h), 96.0], ["P%d" % page, 96.0]])
for ln, words in enumerate(lines, 1):
    for wn, (text, conf) in enumerate(words, 1):
        print("\t".join(map(str, [5, 1, 1, 1, ln, wn, wn * 60, ln * 30, 50, 20, conf, text])))
'''


def make(dirpath, mode="ok", lines=(), emit_dims=True):
    p = Path(dirpath) / f"fake-tesseract-{mode}"
    p.write_text(TEMPLATE.format(py=sys.executable, cfg=json.dumps({"mode": mode, "lines": [list(map(list, l)) for l in lines], "emit_dims": emit_dims})))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def words(text, conf=95.0):
    return [[w, conf] for w in text.split()]
