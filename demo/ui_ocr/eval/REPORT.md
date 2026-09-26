# OCR extraction tuning against DocILE (dev-only)

Harness: `eval/docile_eval.py` (never imported by the app). Real Tesseract 5.3.4 at 300 DPI, default page segmentation (psm 3 beat psm 4 and 6).
Gold: DocILE KILE labels + `metadata.document_type`, read **only** by the harness. The runtime never sees them (a test scans the runtime sources for it).

## Protocol
| split | docs | use |
|---|---|---|
| tune | 150 | official *train*, minus the 1,314 documents the project selected for its own probe/LoRA work; used to develop the extractor |
| tune2 | 200 | fresh *train* docs, disjoint from tune and from project-selected docs; used to check generalisation and find systematic errors |
| val | 120 | official *validation* split (none project-selected); scored for the final numbers; **no validation errors were inspected** |

Diagnostic: OCR already shows 87-99 % of gold values in its text, so the original 10-17 % recall was an extraction problem, not an OCR problem.

## Results (recall = correct / gold present; precision = correct / emitted)
Original extractor on `tune` (all types): supplier 0 %, invoice number 17 % (65 % precision), invoice date 12 %, amount 11 % (43 %), payable-vs-not 75 %, 150/150 documents to review.

| field | tune | tune2 (unseen) | **val (held out)** | val, payable invoices only |
|---|---|---|---|---|
| supplier | 41 % / 70 % | 39 % / 69 % | **47 % / 67 %** | 49 % / 76 % |
| invoice number | 55 % / 77 % | 55 % / 78 % | **53 % / 76 %** | 48 % / 73 % |
| invoice date | 60 % / 92 % | 60 % / 89 % | **49 % / 79 %** | 51 % / 78 % |
| due date | 36 % / 100 % | 52 % / 100 % | **33 % / 67 %** | 40 % / 67 % |
| amount | 48 % / 86 % | 47 % / 82 % | **47 % / 77 %** | 54 % / 82 % |
| currency (symbol -> code) | 0 % | 0 % | **0 %** | 0 % |

Currency is 0 % by policy: DocILE amounts carry a bare `$`, which is ambiguous (USD/CAD/AUD/...) unless a trusted vendor record confirms it, so the field is flagged ambiguous (~80 % of the time it is *seen* and reported as ambiguous with candidates), never guessed.

## Safety
Non-invoices (orders, purchase orders, receipts, ...) reaching READY/CLARIFY: **0 of 174** across tune/tune2/val, re-verified after the last change (8 of 54 on tune before the fix). An unknown or ambiguous document type now fails closed to human review.
A supplier inferred from the letterhead never triggers a clarification email (human review instead).

## Deliberate non-choices (measured)
* Choosing "last" or "largest" total when totals conflict: right 5 of 12 -> would replace honest ambiguity with 7 wrong answers. Kept ambiguous.
* A `Station` call-sign as vendor: helps order documents, hurts payable invoices (precision 59 %), so it applies only to documents not classified as invoices.
* Requiring letterhead corroboration: precision stayed ~70 % while recall fell. Not adopted; the safety rule above is used instead.
* Remaining misses are dominated by values that are not readable in the OCR text, wrong gold quirks (`N-1220856908` vs printed `IN-1220856908`), or layouts with no label.

Reproduce: `python3 eval/docile_eval.py select && python3 eval/docile_eval.py ocr tune && python3 eval/docile_eval.py score tune` (cache under `eval/cache/`, git-ignored).
