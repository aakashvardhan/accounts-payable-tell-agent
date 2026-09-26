# Project-local Tesseract (no sudo, nothing system-wide)

Unpacked from the official Ubuntu 24.04 (noble, arm64) packages with `apt-get download` (apt verifies each download against the signed
repository index) and `dpkg -x` into `root/`. `SHA256SUMS` records the exact .deb files used.

    liblept5 1.82.0-3build4 · libtesseract5 5.3.4-1build5 · tesseract-ocr 5.3.4-1build5 · tesseract-ocr-eng 4.1.0-2 · tesseract-ocr-osd 4.1.0-2

`ocr.py` finds `root/usr/bin/tesseract` automatically and points it at its private libraries and language data
(LD_LIBRARY_PATH / TESSDATA_PREFIX are set only for that child process). A system install (`sudo apt-get install tesseract-ocr tesseract-ocr-eng`)
would be picked up first and this folder ignored. To recreate:

    mkdir -p debs root && cd debs && apt-get download liblept5 libtesseract5 tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
    for d in *.deb; do dpkg -x "$d" ../root; done
