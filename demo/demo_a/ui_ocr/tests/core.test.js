// Focused CPU tests for static/tell_core.js and static/app.js rendering (stub DOM, no browser). Run: node --test tests/
const test = require('node:test'), assert = require('node:assert'), fs = require('fs'), path = require('path'), vm = require('vm');
const ROOT = path.join(__dirname, '..');
const T = require(path.join(ROOT, 'static/tell_core.js'));
const bundle = JSON.parse(fs.readFileSync(process.env.TELL_TRACE_BUNDLE || path.join(ROOT, 'data/traces_v1.json'), 'utf8'));
const clean = T.findTrace(bundle.traces, 'clean_payment'), attack = T.findTrace(bundle.traces, 'escalated_attack');

test('meter zones and thresholds (boundaries are half-open)', () => {
  const lo = attack.thresholds.lower.v, hi = attack.thresholds.upper.v;
  assert.ok(Math.abs(lo - 0.1708046793937683) < 1e-12 && Math.abs(hi - 0.5134634443863925) < 1e-12);
  assert.equal(T.zoneFor(0, lo, hi), 'agent_1');
  assert.equal(T.zoneFor(lo - 1e-9, lo, hi), 'agent_1');
  assert.equal(T.zoneFor(lo, lo, hi), 'tell_verify');
  assert.equal(T.zoneFor(hi - 1e-9, lo, hi), 'tell_verify');
  assert.equal(T.zoneFor(hi, lo, hi), 'agent_s');
  assert.equal(T.zoneFor(1, lo, hi), 'agent_s');
  assert.throws(() => T.zoneFor(NaN, lo, hi)); assert.throws(() => T.zoneFor(1.2, lo, hi)); assert.throws(() => T.zoneFor(0.5, hi, lo));
  const segs = T.zoneSegments(lo, hi);
  assert.deepEqual(segs.map(s => s.label), ['Agent 1', 'Tell-Verify', 'Agent S']);
  assert.equal(segs[0].to, segs[1].from); assert.equal(segs[1].to, segs[2].from);
});

test('every recorded tell reading matches its zone', () => {
  bundle.traces.forEach(t => t.events.filter(e => e.tell).forEach(e => assert.equal(e.tell.zone, T.zoneFor(e.tell.score, e.tell.lower, e.tell.upper))));
});

test('guided playback: start/next/play/pause/reset', () => {
  let cb, cleared = 0; const states = [];
  const p = T.createPlayer(clean, { setInterval: (f) => { cb = f; return 1; }, clearInterval: () => { cleared++; }, onChange: s => states.push(s) });
  assert.equal(p.state().started, false); assert.equal(p.state().status, 'PROCESSING');
  p.start(); assert.equal(p.index(), 0);
  p.next(); assert.equal(p.index(), 1);
  p.play(); assert.equal(p.state().playing, true);
  while (p.state().playing) cb();               // timer ticks until the end auto-pauses
  const end = p.state(); assert.equal(end.done, true); assert.equal(end.status, 'PAID'); assert.ok(cleared >= 1);
  p.reset(); const r = p.state(); assert.equal(r.started, false); assert.equal(r.log.length, 0); assert.equal(r.ledger, null); assert.equal(r.outcome, null);
  p.pause(); p.seek(3); assert.equal(p.index(), 3);
});

test('state reduces from events only: routing switch and outcomes', () => {
  const before = T.stateAt(attack, 3), after = T.stateAt(attack, 99);
  assert.equal(after.activeAgent, 'agent_s'); assert.equal(after.gate.decision, 'BLOCK'); assert.equal(after.status, 'ESCALATED');
  assert.equal(T.stateAt(clean, 99).gate.decision, 'ALLOW');
  assert.equal(T.stateAt(clean, -1).tell, null);
  assert.notEqual(before.status, 'ESCALATED');
});

test('ledger rows: clean changes, attack unchanged', () => {
  const c = T.ledgerRows(T.stateAt(clean, 99).ledger), a = T.ledgerRows(T.stateAt(attack, 99).ledger);
  assert.ok(c.find(r => r.key === 'operating_balance_minor').changed);
  assert.equal(c.find(r => r.key === 'operating_balance_minor').before - c.find(r => r.key === 'operating_balance_minor').after, clean.invoice.amount_minor_units.v);
  assert.ok(!a.find(r => r.key === 'operating_balance_minor').changed);
  assert.equal(T.ledgerRows(null).length, 0);
});

test('summary metrics derive from queue; escalations subset', () => {
  const s = T.summarize(bundle.traces, bundle.fleet.baseline);
  assert.equal(s.escalated, bundle.traces.filter(t => t.queue_status === 'ESCALATED').length);
  assert.equal(s.processing + s.follow_up + s.escalated + s.counts.PAID + s.counts.BLOCKED, bundle.traces.length);
  assert.equal(T.escalations(bundle.traces).length, s.escalated);
});

test('provenance: real vs fixture derived from field tags, never assumed', () => {
  const pa = T.provenanceOf(attack); assert.deepEqual(pa.modes, [T.MODE.REAL, T.MODE.FIXTURE]);
  assert.equal(T.provenanceOf({ events: [{ prov: { gate: 'fixture' } }] }).modes.join(), T.MODE.FIXTURE);
  assert.equal(T.provenanceOf({ events: [{ prov: { tell: 'real' } }] }).modes.join(), T.MODE.REAL);
  const g = attack.events.find(e => e.gate); assert.equal(T.eventEvidence(g), 'fixture');
  assert.equal(T.eventEvidence(attack.events.find(e => e.tell)), 'real');
});

// ---------- rendering with a stub DOM ----------
function makeDom() {
  class Node { constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this.style = {}; this._text = null; this.className = ''; this.id = ''; }
    appendChild(c) { this.children.push(c); return c; }
    insertBefore(c, ref) { const i = this.children.indexOf(ref); this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
    get firstChild() { return this.children[0] || null; }
    setAttribute(k, v) { this.attrs[k] = v; if (k === 'id') this.id = v; if (k === 'class') this.className = v; }
    addEventListener(t, f) { this.listeners[t] = f; }
    set textContent(v) { this.children = []; this._text = v === '' ? null : String(v); }
    get textContent() { return (this._text || '') + this.children.map(c => c.textContent).join(' '); }
    scrollIntoView() {}
    find(pred, out = []) { if (pred(this)) out.push(this); this.children.forEach(c => c.find && c.find(pred, out)); return out; } }
  class Text { constructor(t) { this._t = t; } get textContent() { return this._t; } }
  const els = {}; ['app', 'nav', 'modes', 'banner', 'foot'].forEach(id => { els[id] = new Node('div'); els[id].id = id; });
  const doc = { getElementById: id => els[id], createElement: t => new Node(t), createElementNS: (n, t) => new Node(t), createTextNode: t => new Text(t) };
  return { doc, els };
}
async function boot(hash, opts = {}) {
  const { doc, els } = makeDom(); const wl = {}; const calls = [], timers = [];
  const route = (url, init) => {
    calls.push({ url, method: (init && init.method) || 'GET', headers: (init && init.headers) || {}, body: init && init.body });
    if (url === '/api/traces') return JSON.parse(JSON.stringify(bundle));
    if (url === '/api/intake/status') return opts.status || { enabled: false };
    if (url === '/auth/logout') return { ok: true };
    if (url === '/api/intake/jobs') return { jobs: opts.jobs || [] };
    if (url.startsWith('/api/intake/jobs/')) return (opts.detail || {})[url.split('/').pop()] || { error: 'unknown job' };
    if (url === '/api/intake/scan') return opts.scan || { scan_id: 's1', state: 'running' };
    if (url.startsWith('/api/intake/scan/')) return opts.scanResult || { state: 'complete' };
    if (url === '/api/intake/batch') return opts.batch || { jobs: [] };
    return {};
  };
  const ctx = { document: doc, location: { hash, href: '' }, window: { addEventListener: (t, f) => { wl[t] = f; }, scrollTo() {} }, TellCore: T,
    fetch: (url, init) => Promise.resolve({ status: opts.unauthorized ? 401 : 200, json: () => Promise.resolve(opts.unauthorized ? { error: 'authentication required' } : route(url, init)) }),
    setInterval: (f, ms) => { timers.push({ f, ms }); return timers.length; }, clearInterval: () => {}, setTimeout: (f) => setTimeout(f, 0), Date, console };
  ctx.window.TellCore = T; ctx.self = ctx.window;
  vm.createContext(ctx); vm.runInContext(fs.readFileSync(path.join(ROOT, 'static/app.js'), 'utf8'), ctx);
  await new Promise(r => setTimeout(r, 20));
  const go = async (h) => { ctx.location.hash = h; wl.hashchange(); };
  return { els, ctx, go, calls, timers, wl };
}
const buttons = (n) => n.find(x => x.tag === 'button');

test('dashboard renders KPIs, statuses, story shortcuts, mode badges, fixture banner', async () => {
  const { els } = await boot('#/');
  const txt = els.app.textContent;
  ['Invoices processed', 'Currently processing', 'Awaiting follow-up', 'Escalated for review', 'Payments completed', 'Payments prevented'].forEach(k => assert.ok(txt.includes(k), k));
  ['PROCESSING', 'FOLLOW_UP', 'ESCALATED', 'PAID', 'BLOCKED'].forEach(k => assert.ok(txt.includes(k), k));
  assert.ok(!txt.includes('Clean payment') && !txt.includes('Escalated attack') && !txt.includes('STORY') && !txt.includes('guided walkthrough'));
  assert.ok(els.modes.textContent.includes('REPLAY — REAL CAPTURE') && els.modes.textContent.includes('SIMULATED DEMO FIXTURE'));
  assert.ok(els.modes.textContent.includes('LIVE LOCAL INFERENCE') && els.modes.textContent.includes('DISABLED'));
  assert.equal(els.banner.textContent, 'SIMULATED DEMO FIXTURE — NOT MEASURED MODEL PERFORMANCE');
  assert.equal(els.app.find(n => n.tag === 'tr' && n.attrs['data-run']).length, bundle.traces.length);
});

test('process selection: clicking a row opens that run; Escalated KPI opens review', async () => {
  const { els, ctx } = await boot('#/');
  const row = els.app.find(n => n.attrs['data-run'] === attack.run_id)[0]; row.listeners.click();
  assert.equal(ctx.location.hash, '#/run/' + attack.run_id);
  const kpi = buttons(els.app).find(b => b.textContent.includes('Escalated for review')); kpi.listeners.click();
  assert.equal(ctx.location.hash, '#/review');
});

test('process view: controls, meter, playback through the DOM, ledger, gate, outcome', async () => {
  const { els, ctx, go } = await boot('#/run/' + clean.run_id);
  const labels = buttons(els.app).map(b => b.textContent.trim());
  ['Back to dashboard', 'Start', 'Next', 'Play', 'Pause', 'Reset'].forEach(l => assert.ok(labels.some(x => x.includes(l)), l));
  assert.ok(els.app.textContent.includes('Awaiting Tell reading'));
  await go('#/run/' + clean.run_id + '@99');
  const t = els.app.textContent;
  assert.ok(t.includes('0.021') && t.includes('AGENT 1 ZONE') && t.includes('lower 0.1708') && t.includes('upper 0.5135'));
  assert.ok(t.includes('TELL-VERIFY') && t.includes('AGENT S'));
  assert.ok(t.includes('ALLOW') && t.includes('Payment completed (simulated)') && t.includes('LOCAL_SQLITE_SIMULATION'));
  const rows = els.app.find(n => n.id === 'timeline')[0].children; assert.equal(rows.length, clean.events.length);
  await go('#/run/' + clean.run_id + '@-1');
  assert.ok(els.app.textContent.includes('Press Start') && !els.app.textContent.includes('Payment completed'));
});

test('escalation run shows routing switch to Agent S and BLOCK, ledger unchanged', async () => {
  const { els, go } = await boot('#/');
  await go('#/run/' + attack.run_id + '@99');
  const t = els.app.textContent;
  assert.ok(t.includes('AGENT 1 → AGENT S') && t.includes('BLOCK') && t.includes('escalated to human review'));
  assert.ok(t.includes('held_for_review'));
  assert.ok(els.app.find(n => n.id === 'lane-as')[0].className.includes('active'));
});

test('human review renders all required evidence and offers no approve/clear control', async () => {
  const { els } = await boot('#/review/' + attack.run_id);
  const t = els.app.textContent, rv = attack.review;
  ['Triggering evidence', 'Attack surface', 'Tell score & routing', 'Actions attempted', 'Trusted lookups', 'Validator', 'Gate decision', 'Alarm state',
    'Captured evidence report', 'Recommended human action', 'audit ids', 'probe_weights_sha256', 'adapter_weights_sha256', 'evidence_report_sha256'
  ].concat(process.env.TELL_TRACE_BUNDLE ? [] : ['records_source_sha256']).forEach(k => assert.ok(t.includes(k), k));
  assert.ok(t.includes(rv.triggering_evidence.v) && t.includes(rv.artifact_hashes.evidence_report_sha256.v));
  assert.ok(rv.audit_ids.v.every(a => t.includes(a)));
  assert.equal(els.app.find(n => n.id && n.id.startsWith('case-')).length, T.escalations(bundle.traces).length);
  const labels = buttons(els.app).map(b => b.textContent.toLowerCase());
  assert.ok(!labels.some(l => /approve|clear|release|pay now|override/.test(l)), labels.join('|'));
});

// ---------- intake dashboard ----------
const JOB = (o) => Object.assign({ job_id: 'a'.repeat(32), display_name: 'inv.pdf', size: 2048, source: 'manual_upload', sha256: 'f'.repeat(64), created_at: '2026-09-25T10:00:00Z',
  updated_at: '2026-09-25T10:00:01Z', status: 'READY_FOR_PROCESSING', progress: 100, latest_event: 'Extraction complete: ready for agent processing', error: null, duplicate_of: null,
  summary: { invoice_number: 'X-1', supplier_name: 'Acme', amount: '10.00', currency: 'EUR' } }, o);
const STATUS = (o) => Object.assign({ enabled: true, limits: { max_bytes: 10485760, max_files_per_batch: 20, max_queue: 200 }, last_scan: null, counts: {} }, o);
const nodesBy = (root, pred) => root.find(pred);

test('dashboard: story boxes are removed from the DOM (not hidden) and source has no story-card markup', async () => {
  const { els } = await boot('#/', { status: STATUS() });
  assert.equal(nodesBy(els.app, n => /story/.test(n.className || '')).length, 0);
  const src = fs.readFileSync(path.join(ROOT, 'static/app.js'), 'utf8'), css = fs.readFileSync(path.join(ROOT, 'static/styles.css'), 'utf8');
  assert.ok(!/storyBtn|story-row|Open guided walkthrough|STORY \d/.test(src));
  assert.ok(!/\.story/.test(css.replace(/\/\*[\s\S]*?\*\//g, '')));
});

test('dashboard: replay invoice rows stay clickable and open a playable process view', async () => {
  const { els, ctx, go } = await boot('#/', { status: STATUS() });
  for (const t of [clean, attack]) {
    const row = nodesBy(els.app, n => n.attrs['data-run'] === t.run_id)[0]; assert.ok(row, t.scenario_id);
    row.listeners.click(); assert.equal(ctx.location.hash, '#/run/' + t.run_id);
    await go(ctx.location.hash);
    const labels = nodesBy(els.app, n => n.tag === 'button').map(b => b.textContent.trim());
    ['Start', 'Next', 'Play', 'Pause', 'Reset'].forEach(l => assert.ok(labels.includes(l), l));
    await go('#/'); 
  }
});

test('dashboard: primary action area, multi-PDF picker, counters and last-checked', async () => {
  const { els } = await boot('#/', { status: STATUS({ last_scan: { completed_at: '2026-09-25T09:00:00Z', imported: 2 } }), jobs: [
    JOB({ job_id: '1'.repeat(32), status: 'OCR_RUNNING', progress: 60, latest_event: 'OCR page 1 (1/2)' }), JOB({ job_id: '2'.repeat(32), status: 'UPLOADING', progress: 5 }),
    JOB({ job_id: '3'.repeat(32) }), JOB({ job_id: '4'.repeat(32), status: 'UNREADABLE_DOCUMENT' }), JOB({ job_id: '5'.repeat(32), status: 'NEEDS_DOCUMENT_REVIEW' }),
    JOB({ job_id: '6'.repeat(32), status: 'FAILED', error: 'bad' }), JOB({ job_id: '7'.repeat(32), status: 'AWAITING_VENDOR_CLARIFICATION' })] });
  const t = els.app.textContent;
  assert.ok(t.includes('Check for new invoices') && t.includes('Add invoices') && t.includes('Inbox last checked:') && !t.includes('last checked: never'));
  const input = nodesBy(els.app, n => n.tag === 'input')[0];
  assert.ok('multiple' in input.attrs && /application\/pdf/.test(input.attrs.accept) && input.attrs.type === 'file');
  const k = (name) => nodesBy(els.app, n => n.attrs['data-k'] === name)[0].textContent.trim();
  assert.ok(k('Waiting').startsWith('1')); assert.ok(k('Processing').startsWith('1')); assert.ok(k('Ready for agent processing').startsWith('1'));
  assert.ok(k('Awaiting vendor').startsWith('1')); assert.ok(k('Need attention').startsWith('2')); assert.ok(k('Failed intake').startsWith('1'));
  assert.equal(nodesBy(els.app, n => n.attrs['data-job']).length, 7);
  assert.ok(t.includes('OCR running') && t.includes('OCR page 1 (1/2)') && t.includes('Awaiting vendor') && t.includes('Unreadable document') && t.includes('Needs document review'));
  assert.equal(nodesBy(els.app, n => /fill/.test(n.className || '')).length, 2);   // progress bars only on active jobs
});

test('intake job rows navigate to the job detail view', async () => {
  const j = JOB({ job_id: '7'.repeat(32) });
  const { els, ctx } = await boot('#/', { status: STATUS(), jobs: [j] });
  nodesBy(els.app, n => n.attrs['data-job'] === j.job_id)[0].listeners.click(); assert.equal(ctx.location.hash, '#/job/' + j.job_id);
});

test('public mode: no intake controls or intake rows', async () => {
  const { els } = await boot('#/', { status: { enabled: false }, jobs: [JOB()] });
  const t = els.app.textContent;
  assert.ok(!t.includes('Check for new invoices') && !t.includes('Add invoices') && nodesBy(els.app, n => n.tag === 'input').length === 0 && nodesBy(els.app, n => n.attrs['data-job']).length === 0);
});

test('Check for new invoices: posts a scan with the CSRF header, shows in-progress, then the result counts', async () => {
  const done = { scan_id: 's1', state: 'complete', discovered: 4, imported: 2, duplicates: 1, rejected: 1, completed_at: '2026-09-25T10:00:00Z' };
  const { els, calls } = await boot('#/', { status: STATUS(), scanResult: done });
  nodesBy(els.app, n => n.id === 'btn-check')[0].listeners.click();
  assert.ok(nodesBy(els.app, n => n.id === 'scan-msg')[0].textContent.includes('Scanning demo inbox'));
  assert.ok(nodesBy(els.app, n => n.id === 'btn-check')[0].attrs.disabled !== undefined);
  await new Promise(r => setTimeout(r, 60));
  const post = calls.find(c => c.url === '/api/intake/scan' && c.method === 'POST'); assert.ok(post); assert.equal(post.headers['X-Tell-Intake'], '1');
  const msg = nodesBy(els.app, n => n.id === 'scan-msg')[0].textContent;
  assert.ok(msg.includes('2 invoices imported') && msg.includes('discovered 4') && msg.includes('duplicates skipped 1') && msg.includes('rejected 1') && msg.includes('completed'));
});

test('Check for new invoices: clear "No new invoices found" result', async () => {
  const { els } = await boot('#/', { status: STATUS(), scanResult: { scan_id: 's1', state: 'complete', discovered: 3, imported: 0, duplicates: 3, rejected: 0, completed_at: '2026-09-25T10:00:00Z' } });
  nodesBy(els.app, n => n.id === 'btn-check')[0].listeners.click(); await new Promise(r => setTimeout(r, 60));
  assert.ok(nodesBy(els.app, n => n.id === 'scan-msg')[0].textContent.includes('No new invoices found'));
});

test('Add invoices: manifest posted once for the whole batch, then each file uploaded independently', async () => {
  const files = [{ name: 'a.pdf', size: 10, type: 'application/pdf' }, { name: 'b.pdf', size: 20, type: 'application/pdf' }, { name: 'c.txt', size: 5, type: 'text/plain' }];
  const jobs = files.map((f, i) => JOB({ job_id: String(i + 1).repeat(32), display_name: f.name, status: i === 2 ? 'FAILED' : 'UPLOADING', progress: i === 2 ? 100 : 10, summary: {} }));
  const { els, calls } = await boot('#/', { status: STATUS(), batch: { jobs } });
  const input = nodesBy(els.app, n => n.tag === 'input')[0];
  input.listeners.change({ target: { files: files, value: 'x' } });
  await new Promise(r => setTimeout(r, 60));
  const batch = calls.filter(c => c.url === '/api/intake/batch'); assert.equal(batch.length, 1);
  assert.deepEqual(JSON.parse(batch[0].body).files.map(f => f.name), ['a.pdf', 'b.pdf', 'c.txt']);
  const puts = calls.filter(c => c.method === 'PUT'); assert.equal(puts.length, 2);   // failed job (wrong type) is never uploaded
  puts.forEach(p => { assert.match(p.url, /^\/api\/intake\/jobs\/[0-9]{32}\/content$/); assert.equal(p.headers['Content-Type'], 'application/pdf'); assert.equal(p.headers['X-Tell-Intake'], '1'); });
  assert.equal(nodesBy(els.app, n => n.attrs['data-job']).length, 3);   // all three visible in the queue immediately
});

const EV = (seq, kind, stage, message, extra) => Object.assign({ schema: 'tell.intake_event/1.0', seq, at: '2026-09-25T10:00:0' + seq + 'Z', kind, stage, actor: 'application', provenance: 'APPLICATION', message, data: null }, extra || {});
const FIELD = (v, o) => Object.assign({ found: true, value: v, source: 'EMBEDDED_TEXT', page: 1, confidence: 1, evidence_text: 'label: ' + v, normalization_applied: null, ambiguous: false, candidates: [] }, o || {});
const NOFIELD = { found: false, value: null, source: null, page: null, confidence: null, evidence_text: null, normalization_applied: null, ambiguous: false, candidates: [] };

test('job detail: fields with source/page/confidence, missing fields, warnings, sanitized text; hostile strings stay text', async () => {
  const id = '9'.repeat(32), evil = '<img src=x onerror=alert(1)>.pdf', text = '<script>alert(1)</script>\nTotal Due: EUR 5.00';
  const detail = JOB({ job_id: id, display_name: evil, status: 'NEEDS_DOCUMENT_REVIEW', error: 'ambiguous required field(s): currency', payment: { hold: true, eligible: false, reason: 'document needs human review' },
    events: [EV(0, 'upload', 'UPLOADING', 'PDF uploaded'), EV(1, 'extraction', 'EXTRACTION_COMPLETE', 'Fields extracted', { provenance: 'OCR' })],
    extraction: { source: 'OCR', fields: { document_type: FIELD('invoice', { source: 'OCR', confidence: 0.97 }), invoice_number: FIELD('Z-1', { source: 'OCR', confidence: 0.88, normalization_applied: 'thousands separators removed' }),
      supplier_name: NOFIELD, amount: FIELD('5.00', { source: 'OCR', page: 2, confidence: 0.91 }),
      currency: Object.assign({}, NOFIELD, { ambiguous: true, candidates: ['USD', 'CAD'], evidence_text: 'Total $ 5.00' }) },
      warnings: ["currency symbol '$' is ambiguous"], decision: { status_hint: 'REVIEW', reasons: ['ambiguous required field(s): currency'], missing_required_fields: [] },
      raw_text: [{ page: 1, text, source: 'OCR' }], text } });
  const { els } = await boot('#/job/' + id, { status: STATUS(), detail: { [id]: detail } });
  await new Promise(r => setTimeout(r, 30));
  const t = els.app.textContent;
  ['Filename', 'Uploaded', 'SHA-256', 'Intake source', 'Manual upload', 'Processing timeline', 'Extracted fields', 'Required-field decision', 'Extraction warnings', 'Raw OCR text (sanitized)', 'Next step'].forEach(k => assert.ok(t.includes(k), k));
  assert.ok(t.includes(evil) && t.includes('<script>alert(1)</script>'));                       // shown literally as text
  assert.equal(nodesBy(els.app, n => n.tag === 'img' || n.tag === 'script').length, 0);         // never turned into elements
  const row = (f) => nodesBy(els.app, n => n.attrs['data-field'] === f)[0].textContent;
  assert.ok(row('invoice_number').includes('Z-1') && row('invoice_number').includes('OCR') && row('invoice_number').includes('page 1') && row('invoice_number').includes('88% confidence') && row('invoice_number').includes('normalized: thousands'));
  assert.ok(row('amount').includes('page 2') && row('amount').includes('91% confidence'));
  assert.ok(row('supplier_name').includes('not found in document')); assert.ok(row('due_date').includes('not found in document'));
  assert.ok(row('currency').includes('ambiguous') && row('currency').includes('USD, CAD') && row('currency').includes('not resolved'));
  assert.ok(nodesBy(els.app, n => n.id === 'payment-hold')[0].textContent.includes('PAYMENT ON HOLD'));
  ['Tell score', 'Gate decision', 'Selected agent', 'Ledger'].forEach(k => assert.ok(!t.includes(k), 'must not show ' + k));
  assert.equal(nodesBy(els.app, n => n.id === 'email-preview').length, 0);
});

test('job detail: missing-currency invoice shows timeline, decision, vendor lookup, SIMULATED email and payment hold', async () => {
  const id = '8'.repeat(32);
  const outbox = { outbox_id: 'out-1', recipient: 'accounts-receivable@imaginary-freight.example', recipient_source: 'TRUSTED_VENDOR_RECORD:VND-IMAGINARY-FREIGHT', requested_fields: ['currency'],
    subject: 'Clarification required for invoice IFC-88213', body: 'We received invoice IFC-88213 for 1,250.00, but the currency is not stated.', send_status: 'SIMULATED_NOT_SENT',
    action_trace_id: 'act-490f14a957ee', created_at: '2026-09-25T10:00:05Z', provenance: { action: 'SIMULATED_AGENT_ACTION', recipient: 'TRUSTED_VENDOR_RECORD', template: 'APPLICATION' } };
  const detail = JOB({ job_id: id, status: 'AWAITING_VENDOR_CLARIFICATION', payment: { hold: true, eligible: false, reason: 'waiting for vendor clarification' }, outbox,
    events: [EV(0, 'upload', 'UPLOADING', 'PDF uploaded'), EV(1, 'embedded_text', 'EXTRACTING_EMBEDDED_TEXT', 'Embedded text extracted', { provenance: 'EMBEDDED_TEXT' }),
      EV(2, 'required_field_check', 'EXTRACTION_COMPLETE', 'Required-field check: currency missing — payment on hold'), EV(3, 'vendor_lookup', 'EXTRACTION_COMPLETE', 'Trusted vendor lookup: CONTACT_FOUND', { actor: 'trusted_vendor_record', provenance: 'TRUSTED_VENDOR_RECORD' }),
      EV(4, 'agent_action', 'EXTRACTION_COMPLETE', 'Clarification requested (SIMULATED_AGENT_ACTION)', { actor: 'simulated_agent', provenance: 'SIMULATED_AGENT_ACTION' }),
      EV(5, 'email_prepared', 'EXTRACTION_COMPLETE', 'Simulated clarification email prepared — NOT SENT'), EV(6, 'status', 'AWAITING_VENDOR_CLARIFICATION', 'Awaiting vendor clarification; payment remains on hold')],
    extraction: { source: 'EMBEDDED_TEXT', fields: { document_type: FIELD('invoice'), supplier_name: FIELD('Imaginary Freight Co'), invoice_number: FIELD('IFC-88213'), amount: FIELD('1250.00'), currency: NOFIELD },
      warnings: [], decision: { status_hint: 'CLARIFY', missing_required_fields: ['currency'], reasons: ['missing required field(s): currency'] },
      missing_field_result: { invoice_id: id, missing_required_fields: ['currency'], payment_eligible: false, recommended_action: 'request_vendor_clarification' },
      vendor_lookup: { status: 'CONTACT_FOUND', vendor_id: 'VND-IMAGINARY-FREIGHT', vendor_name: 'Imaginary Freight Co', reason: 'verified accounts-receivable contact on the vendor record' },
      raw_text: [{ page: 1, text: 'INVOICE', source: 'EMBEDDED_TEXT' }], text: 'INVOICE' } });
  const { els } = await boot('#/job/' + id, { status: STATUS(), detail: { [id]: detail } });
  await new Promise(r => setTimeout(r, 30));
  const t = els.app.textContent;
  const kinds = nodesBy(els.app, n => n.attrs['data-kind']).map(n => n.attrs['data-kind']);
  assert.deepEqual(kinds, ['upload', 'embedded_text', 'required_field_check', 'vendor_lookup', 'agent_action', 'email_prepared', 'status']);
  assert.ok(t.includes('Awaiting vendor') && t.includes('SIMULATED_AGENT_ACTION'));
  assert.equal(nodesBy(els.app, n => n.id === 'sim-badge')[0].textContent.trim(), 'SIMULATED — NOT SENT');
  const em = nodesBy(els.app, n => n.id === 'email-preview')[0].textContent;
  assert.ok(em.includes('accounts-receivable@imaginary-freight.example') && em.includes('TRUSTED_VENDOR_RECORD:VND-IMAGINARY-FREIGHT') && em.includes('Clarification required for invoice IFC-88213') && em.includes('SIMULATED_NOT_SENT'));
  assert.ok(nodesBy(els.app, n => n.id === 'email-body')[0].textContent.includes('the currency is not stated'));
  assert.ok(nodesBy(els.app, n => n.id === 'vendor-lookup')[0].textContent.includes('CONTACT_FOUND'));
  assert.ok(nodesBy(els.app, n => n.id === 'missing-field-result')[0].textContent.includes('"payment_eligible": false'));
  assert.ok(nodesBy(els.app, n => n.id === 'payment-hold')[0].textContent.includes('PAYMENT ON HOLD'));
  assert.ok(nodesBy(els.app, n => n.id === 'next-step')[0].textContent.includes('payment is on hold'));
  ['Tell score', 'Validator', 'Gate decision', 'Ledger', 'Payment completed'].forEach(k => assert.ok(!t.includes(k), 'must not show ' + k));
});

test('job detail: OCR progress is shown page by page with engine, DPI and per-page confidence', async () => {
  const id = '7'.repeat(32);
  const detail = JOB({ job_id: id, status: 'OCR_RUNNING', progress: 65, events: [EV(0, 'upload', 'UPLOADING', 'PDF uploaded'), EV(1, 'ocr_page', 'OCR_RUNNING', 'OCR page 1 (1/2): 120 usable words, mean confidence 91%', { actor: 'ocr_engine', provenance: 'OCR' })],
    extraction: { ocr: { state: 'running', engine: 'tesseract', version: '5.3.4', dpi: 300, page_count: 2, pages_processed: 2, pages: [{ page: 1, outcome: 'OK', words: 130, usable_words: 120, mean_confidence: 0.91, duration_s: 4.2 }] } } });
  const { els } = await boot('#/job/' + id, { status: STATUS(), detail: { [id]: detail } });
  await new Promise(r => setTimeout(r, 30));
  const ocr = nodesBy(els.app, n => n.id === 'ocr-card')[0].textContent;
  assert.ok(ocr.includes('tesseract 5.3.4') && ocr.includes('300 DPI') && ocr.includes('page 1 of 2') && ocr.includes('running'));
  const row = nodesBy(els.app, n => n.attrs['data-page'] === 1)[0].textContent; assert.ok(row.includes('OK') && row.includes('120') && row.includes('91%'));
  assert.ok(nodesBy(els.app, n => /fill/.test(n.className || '')).length >= 1);          // progress bar while active
});

test('job detail: unreadable scan and OCR-unavailable failure are distinct, explicit states', async () => {
  const a = 'a'.repeat(32), b = 'b'.repeat(32);
  const unreadable = JOB({ job_id: a, status: 'UNREADABLE_DOCUMENT', error: 'unreadable document', payment: { hold: true, eligible: false, reason: 'document is unreadable' },
    events: [EV(0, 'upload', 'UPLOADING', 'x'), EV(1, 'status', 'UNREADABLE_DOCUMENT', 'Unreadable document: OCR ran but produced only 0 usable word(s)')],
    extraction: { fields: {}, source: null, ocr: { state: 'complete', engine: 'tesseract', version: '5', dpi: 300, page_count: 1, pages_processed: 1, duration_s: 3, pages: [{ page: 1, outcome: 'EMPTY', words: 0, usable_words: 0, mean_confidence: 0, duration_s: 3 }] } } });
  const failed = JOB({ job_id: b, status: 'FAILED', error: 'OCR is required but no local OCR engine (Tesseract) is installed', events: [EV(0, 'upload', 'UPLOADING', 'x'), EV(1, 'error', 'FAILED', 'OCR is required but no local OCR engine (Tesseract) is installed')] });
  const { els, go } = await boot('#/job/' + a, { status: STATUS(), detail: { [a]: unreadable, [b]: failed } });
  await new Promise(r => setTimeout(r, 30));
  let t = els.app.textContent; assert.ok(t.includes('Unreadable document') && t.includes('EMPTY') && t.includes('OCR ran'));
  await go('#/job/' + b); await new Promise(r => setTimeout(r, 30));
  t = els.app.textContent; assert.ok(t.includes('Failed') && t.includes('no local OCR engine'));
});

test('job detail: ready state shows the next step and no scoring/routing/payment content', async () => {
  const id = '6'.repeat(32);
  const { els } = await boot('#/job/' + id, { status: STATUS(), detail: { [id]: JOB({ job_id: id, events: [EV(0, 'status', 'READY_FOR_PROCESSING', 'ok')], extraction: { fields: { document_type: FIELD('invoice') }, source: 'EMBEDDED_TEXT', warnings: [], text: 'hello', raw_text: [{ page: 1, text: 'hello', source: 'EMBEDDED_TEXT' }] } }) } });
  await new Promise(r => setTimeout(r, 30));
  assert.ok(nodesBy(els.app, n => n.id === 'next-step')[0].textContent.includes('Ready for agent processing'));
  assert.ok(!els.app.textContent.includes('Tell score') && nodesBy(els.app, n => n.id === 'payment-hold').length === 0 && nodesBy(els.app, n => n.id === 'email-preview').length === 0);
});

test('no unsafe DOM sinks anywhere in the UI source', () => {
  const src = fs.readFileSync(path.join(ROOT, 'static/app.js'), 'utf8');
  ['innerHTML', 'outerHTML', 'insertAdjacentHTML', 'document.write', 'eval(', 'new Function'].forEach(s => assert.ok(!src.includes(s), s));
});

test('intake helpers: counters and next-step wording', () => {
  const c = T.intakeCounts(['UPLOADING', 'VALIDATING', 'EXTRACTING_EMBEDDED_TEXT', 'OCR_QUEUED', 'OCR_RUNNING', 'OCR_COMPLETE', 'EXTRACTION_COMPLETE', 'READY_FOR_PROCESSING',
    'AWAITING_VENDOR_CLARIFICATION', 'NEEDS_DOCUMENT_REVIEW', 'UNREADABLE_DOCUMENT', 'FAILED', 'DUPLICATE'].map(s => ({ status: s })));
  assert.deepEqual([c.waiting, c.processing, c.ready, c.awaiting, c.attention, c.failed, c.duplicates, c.active], [2, 5, 1, 1, 2, 1, 1, 7]);
  assert.equal(T.INTAKE.length, 13); assert.ok(!T.INTAKE.includes('NEEDS_REVIEW') && !T.INTAKE.includes('NEEDS_OCR'));
  assert.equal(T.nextStep({ status: 'READY_FOR_PROCESSING' }), 'Ready for agent processing');
  assert.ok(T.nextStep({ status: 'FAILED', error: 'boom' }).includes('boom')); assert.ok(T.nextStep({ status: 'AWAITING_VENDOR_CLARIFICATION' }).includes('payment is on hold'));
  assert.ok(T.isActive('OCR_RUNNING') && !T.isActive('AWAITING_VENDOR_CLARIFICATION') && !T.isActive('UNREADABLE_DOCUMENT'));
});

test('job detail page shows the uploaded-PDF badge and no replay/fixture banner', async () => {
  const id = '5'.repeat(32);
  const { els } = await boot('#/job/' + id, { status: STATUS(), detail: { [id]: JOB({ job_id: id, events: [EV(0, 'status', 'READY_FOR_PROCESSING', 'ok')], extraction: { fields: {}, warnings: [], text: 'x' } }) } });
  await new Promise(r => setTimeout(r, 30));
  assert.ok(els.modes.textContent.includes('UPLOADED PDF') && !els.modes.textContent.includes('SIMULATED DEMO FIXTURE') && !els.modes.textContent.includes('REPLAY'));
  assert.ok(els.modes.textContent.includes('LIVE LOCAL INFERENCE')); assert.equal(els.banner.textContent, '');
});
test('auth: Sign out button only when the server reports auth; it posts logout and returns to the login page', async () => {
  const { els, ctx, calls } = await boot('#/', { status: STATUS({ auth: true }) });
  const btn = nodesBy(els.nav, n => n.id === 'btn-signout')[0]; assert.ok(btn && btn.textContent.includes('Sign out'));
  btn.listeners.click(); await new Promise(r => setTimeout(r, 30));
  assert.ok(calls.some(c => c.url === '/auth/logout' && c.method === 'POST' && c.headers['X-Tell-Intake'] === '1')); assert.equal(ctx.location.href, '/login');
  const plain = await boot('#/', { status: STATUS({ auth: false }) });
  assert.equal(nodesBy(plain.els.nav, n => n.id === 'btn-signout').length, 0);
});

test('auth: a 401 from the API sends the browser to /login and never renders data', async () => {
  const { els, ctx } = await boot('#/', { status: STATUS(), unauthorized: true });
  assert.equal(ctx.location.href, '/login');
  assert.ok(!els.app.textContent.includes('Operations dashboard'));
});

test('login page script contains no secrets, posts same-origin with the CSRF header, and clears the field', () => {
  const src = fs.readFileSync(path.join(ROOT, 'auth_static/login.js'), 'utf8');
  assert.ok(src.includes("'/auth/login'") && src.includes("'X-Tell-Intake'") && src.includes("input.value = ''") && !/innerHTML|eval\(|localStorage|sessionStorage/.test(src));
  const html = fs.readFileSync(path.join(ROOT, 'auth_static/login.html'), 'utf8');
  assert.ok(/type="password"/.test(html) && !/<script(?![^>]*src=)[^>]*>/.test(html) && !/ style=/.test(html));
});
