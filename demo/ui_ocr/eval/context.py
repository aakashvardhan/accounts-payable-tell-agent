"""Dev helper: show OCR lines around a gold value for docs where the extractor missed/erred."""
import sys, json, re
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import docile_eval as D
field, status, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
only = sys.argv[4] if len(sys.argv) > 4 else 'tax_invoice'
gkey = {'invoice_number': 'document_id', 'invoice_date': 'date_issue', 'amount': 'amount_due', 'supplier_name': 'vendor_name', 'due_date': 'date_due'}[field]
shown = 0
for i in sum((json.loads(D.SPLITS.read_text())[s] for s in __import__('os').environ.get('SPLIT','tune').split(',')),[]):
    g, m = D.gold_of(i)
    if m['document_type'] != only: continue
    o = D.ocr_doc(i); res, fields, dec, w = D.score_doc(i, o, g, m)
    if res[field][0] != status: continue
    gv = (g.get(gkey) or g.get('amount_total_gross') or [''])[0] if field == 'amount' else (g.get(gkey) or [''])[0]
    key = D.alnum(gv) if field != 'amount' else (D.amt(gv) or 'zz').replace('.', '')
    lines = [l for l in ' \n'.join(p['text'] for p in o['pages']).split('\n') if l.strip()]
    hit = [k for k, l in enumerate(lines) if (key and key in (D.alnum(l) if field != 'amount' else re.sub(r'[^0-9]', '', l)))]
    print(f"\n--- {i[:8]} gold[{gkey}]={gv[:50]!r} ours={fields[field]['value'] or fields[field]['candidates']}  reach={'YES' if hit else 'NO'}")
    for k in hit[:2]:
        for j in range(max(0, k - 1), min(len(lines), k + 2)): print(('  >> ' if j == k else '     ') + lines[j][:140])
    if not hit: print('     (value not visible in OCR)')
    shown += 1
    if shown >= n: break
