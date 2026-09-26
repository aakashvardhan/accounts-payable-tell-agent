# Tell demo interface — Operations Dashboard Intake (isolated preview, port 8083)

An isolated copy of `demo/ui/` so the running review (8081) and public-preview (8082) servers, which read their static
files from `demo/ui/` on every request, are unaffected. CPU-only; no model, no CUDA, no network, no OCR.

    demo/ui_intake/start_demo.sh    # background on 0.0.0.0:8083 (pid -> run/server.pid, log -> run/server.log)
    demo/ui_intake/stop_demo.sh     # stops only the recorded pid after validating its command line
    python3 demo/ui_intake/make_sample_invoices.py   # (re)writes 3 fictional sample PDFs into runtime/inbox/

## What this slice adds
- Dashboard: story cards removed; action area (**Check for new invoices**, **Add invoices**, reset), intake counters
  (waiting / extracting / ready / need attention / failed / last checked), unified queue (uploads + replay rows).
- `intake.py`: PDF validation, embedded-text extraction (poppler `pdftotext`, list-form subprocess, timeout, no shell),
  label-based field parsing, SQLite job registry (`runtime/jobs.sqlite`), `InvoiceSource` + `LocalFolderInvoiceSource`.
- Server API (`/api/intake/*`): `status`, `jobs`, `jobs/<id>`, `batch`, `jobs/<id>/content`, `scan`, `scan/<id>`, `reset`.
  Mutations need `X-Tell-Intake: 1` and a same-origin `Origin`; all intake endpoints fail closed in public-demo mode
  (`--public-demo`, or any bundle flagged `public_demo`).
- Job detail page `#/job/<id>`; replay rows still open `#/run/<id>` with Start/Next/Play/Pause/Reset.

## Limits
PDF only (`application/pdf` + `%PDF-` at byte 0) · 10 MB per file · 20 files per batch · 200 jobs in queue ·
first 50 pages · 15 s extraction timeout · stale `UPLOADING` jobs expire after 120 s.

## Stages
UPLOADING → VALIDATING → EXTRACTING → READY_FOR_PROCESSING | NEEDS_OCR | NEEDS_REVIEW | DUPLICATE | FAILED.
READY needs invoice number, supplier, amount and currency all found in the PDF text; anything else found-but-incomplete is
NEEDS_REVIEW. Nothing is inferred: absent fields stay absent. No Tell scoring, routing, validation or payment happens here.

## Tests (CPU only)
    python3 -m unittest discover -s demo/ui_intake/tests -v
    node --test demo/ui_intake/tests/
