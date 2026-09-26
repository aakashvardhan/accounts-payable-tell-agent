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
    if (opts.extra) { const r = opts.extra(url); if (r !== undefined) return r; }
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
  assert.ok(t.includes('OCR running') && t.includes('OCR page 1 (1/2)') && t.includes('Awaiting clarification') && t.includes('Needs review'))   // required user-facing state names;
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
  assert.equal(T.INTAKE.length, 19); assert.ok(!T.INTAKE.includes('NEEDS_REVIEW') && !T.INTAKE.includes('NEEDS_OCR'));
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

// ---------- LIVE mode (backend-driven; no recorded traces for uploaded invoices) ----------
const IDS = { model_repo_id: 'Qwen/Qwen3-8B', model_revision: 'b968826d9c46dd6066d109eabc6255188de91218', dtype: 'bfloat16', device: 'cuda:0', adapter_weights_sha256: '1d6229b6' + 'a'.repeat(56),
  adapter_frozen_manifest_sha256: 'c'.repeat(64), probe_weights_sha256: '495649cf' + 'b'.repeat(56), layer: 27, scientific_threshold: 0.1708046793937683 };
const LSTATUS = (o) => STATUS(Object.assign({ mode: 'Live', total: 2, ready: 0, processing: 0, payment_completed: 1, attention: 1,
  runtime: { mode: 'Live', identifiers: IDS, worker: { online: true, state: 'idle' } } }, o));
const LJOB = (o) => JOB(Object.assign({ run_id: 'run-' + 'e'.repeat(32), status: 'PAYMENT_COMPLETED', display_status: 'Payment completed', active: false, extraction_status: 'Complete',
  processing_status: 'Complete', final_outcome: 'PAYMENT_COMPLETED', tell: { score: 0.031, above_threshold: false }, route: 'agent_1', tell_alert: 'Below threshold', mode: 'Live' }, o));
const LEV = (seq, type, o) => Object.assign({ event_id: 'ev' + seq, run_id: 'run-x', ts: '2026-09-25T10:00:0' + (seq % 10) + 'Z', sequence: seq, type, title: type, payload: {}, provenance: 'runtime', status: 'ok', duration_ms: null, source: 'run_events' }, o);

test('live dashboard: reads persisted jobs, shows backend identifiers, never loads recorded traces', async () => {
  const jobs = [LJOB({ job_id: '1'.repeat(32) }), LJOB({ job_id: '2'.repeat(32), status: 'PROCESSING', display_status: 'Processing', active: true, progress: 60, final_outcome: null, tell: null, tell_alert: 'Not measured', route: null })];
  const { els, calls, ctx } = await boot('#/', { status: LSTATUS(), jobs });
  const t = els.app.textContent;
  assert.ok(t.includes('LIVE') && t.includes('Invoice queue') && t.includes('Live runtime') && t.includes('Recorded examples'));
  assert.ok(t.includes('b968826d9c46') && t.includes('1d6229b6') && t.includes('495649cf') && t.includes('Not measured') && t.includes('0.031'));
  assert.ok(els.modes.textContent.includes('LIVE') && !els.modes.textContent.includes('SIMULATED DEMO FIXTURE') && !els.modes.textContent.includes('REPLAY'));
  assert.equal(els.banner.hidden, true);
  assert.ok(!calls.some(c => c.url === '/api/traces'), 'recorded bundle must not be fetched for the live dashboard');
  assert.equal(nodesBy(els.app, n => n.attrs['data-job']).length, 2);
  nodesBy(els.app, n => n.attrs['data-job'] === '1'.repeat(32))[0].listeners.click();
  assert.equal(ctx.location.hash, '#/job/' + '1'.repeat(32));
});

test('live dashboard: recorded examples are labelled and loaded only when expanded', async () => {
  const { els, calls } = await boot('#/', { status: LSTATUS(), jobs: [] });
  const det = nodesBy(els.app, n => n.id === 'recorded-examples')[0];
  assert.ok(det && det.textContent.includes('not used for uploaded invoices'));
  det.open = true; det.listeners.toggle(); await new Promise(r => setTimeout(r, 20));
  assert.ok(calls.some(c => c.url === '/api/traces'));
  assert.ok(det.textContent.includes('RECORDED'));
});

test('live replay: renders the backend events in sequence with real score, threshold, route, gate and ledger', async () => {
  const jid = '1'.repeat(32), events = [
    LEV(1, 'intake_received', { source: 'jobs.events' }), LEV(2, 'extraction_completed', { source: 'jobs.events' }), LEV(3, 'agent_turn_started', { payload: { turn: 1, agent: 'agent_1' } }),
    LEV(4, 'tool_call_completed', { payload: { tool: 'read_invoice', status: 'success' }, provenance: 'uploaded_pdf_extraction' }),
    LEV(5, 'activation_captured', { payload: { turn: 1, layer: 27 } }),
    LEV(6, 'probe_scored', { payload: { turn: 1, measured: true, score: 0.9312, operational_threshold: 0.5134634443863925, above_operational_threshold: true, layer: 27, probe_weights_sha256: IDS.probe_weights_sha256 } }),
    LEV(7, 'route_selected', { payload: { turn: 1, routed_agent: 'agent_s', model: 'frozen_safety_lora', reason: 'score above threshold', alarm_latched: true, adapter_sha256: IDS.adapter_weights_sha256 } }),
    LEV(8, 'action_proposed', { payload: { turn: 1, agent: 'agent_s', action: { action: 'submit_evidence_report' } } }),
    LEV(9, 'gate_evaluated', { payload: { gate_decision: 'block', gate_reason_code: 'alarm_unresolved', final_action: 'submit_evidence_report', executed: false, alarm_state_before: 'clear', alarm_state_after: 'alarmed' } }),
    LEV(10, 'job_completed', { title: 'Needs review', payload: { outcome: 'NEEDS_DOCUMENT_REVIEW' } })];
  const job = LJOB({ job_id: jid, status: 'NEEDS_DOCUMENT_REVIEW', display_status: 'Needs review' });
  const { els, calls } = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u === '/api/intake/jobs/' + jid + '/replay') return { job, run: { execution_id: 'exec-1', attempt: 1 }, runtime: { identifiers: IDS }, events };
    if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { source: 'EMBEDDED_TEXT', fields: {} }, sha256: 'f'.repeat(64) });
  } });
  const t = els.app.textContent;
  assert.ok(!calls.some(c => c.url === '/api/traces'));
  const seqs = nodesBy(els.app, n => n.attrs['data-seq']).map(n => Number(n.attrs['data-seq'])); assert.deepEqual(seqs, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.ok(t.includes('0.931') && t.includes('ALARM') && t.includes('0.5135') && t.includes('Agent S') && t.includes('BLOCK') && t.includes('alarm_unresolved') && t.includes('submit_evidence_report'));
  assert.ok(t.includes('run-' + 'e'.repeat(32)) && t.includes('exec-1') && !t.includes('Not measured'));
});

test('live replay: missing probe measurement is displayed as Not measured (an integration failure), never a placeholder score', async () => {
  const jid = '3'.repeat(32), job = LJOB({ job_id: jid, status: 'FAILED', display_status: 'Failed', tell: null });
  const events = [LEV(1, 'agent_turn_started'), LEV(2, 'job_failed', { status: 'failed', title: 'Run failed', payload: { error: { type: 'RuntimeError' } } })];
  const { els } = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u.endsWith('/replay')) return { job, run: {}, runtime: {}, events }; if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { fields: {} } }); } });
  const t = els.app.textContent; assert.ok(t.includes('Not measured') && t.includes('INTEGRATION FAILURE') && !nodesBy(els.app, n => n.id === 'tell-score').length);
});

test('live replay: malformed model output shows a parse failure that failed closed; tool failure is visible', async () => {
  const jid = '4'.repeat(32), job = LJOB({ job_id: jid, status: 'NEEDS_DOCUMENT_REVIEW', display_status: 'Needs review' });
  const events = [LEV(1, 'tool_call_completed', { status: 'failed', title: 'get_vendor_record failed', payload: { tool: 'get_vendor_record', status: 'failure', error: { error_code: 'vendor_not_found' } } }),
    LEV(2, 'action_parse_failed', { status: 'blocked', payload: { turn: 1, parse_outcome: 'parse_failed', error: 'not JSON', raw_output_excerpt: 'I will pay it' } })];
  const { els } = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u.endsWith('/replay')) return { job, run: {}, runtime: {}, events }; if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { fields: {} } }); } });
  const t = els.app.textContent; assert.ok(t.includes('failed closed') && t.includes('I will pay it') && t.includes('vendor_not_found'));
});

test('live mode uses no client-side simulated processing timers other than polling', async () => {
  const src = fs.readFileSync(path.join(ROOT, 'static/app.js'), 'utf8');
  assert.equal((src.match(/setTimeout\s*\(/g) || []).length, 1, 'the only setTimeout is the inbox-scan status poll');
  assert.ok(!/traces_v1|traces_public/.test(src));
});

test('TellSecured toggle: switch is shown in live mode, and uploads carry the chosen mode', async () => {
  const { els, calls } = await boot('#/', { status: LSTATUS(), jobs: [LJOB({ job_id: '1'.repeat(32), tell_secured: false, tell: null, display_status: 'Payment completed' })] });
  const sw = nodesBy(els.app, n => n.id === 'ts-switch')[0]; assert.ok(sw && els.app.textContent.includes('TellSecured ON'));
  assert.ok(els.app.textContent.includes('TellSecured OFF'), 'queue row shows the job mode');
  sw.listeners.click();
  assert.ok(nodesBy(els.app, n => n.id === 'ts-toggle')[0].textContent.includes('TellSecured OFF'));
  nodesBy(els.app, n => n.id === 'btn-check')[0].listeners.click(); await new Promise(r => setTimeout(r, 10));
  const scan = calls.find(c => c.url === '/api/intake/scan'); assert.equal(JSON.parse(scan.body).tell_secured, false);
});

test('live replay: a TellSecured OFF job shows the undefended baseline, not an integration failure', async () => {
  const jid = '5'.repeat(32), job = LJOB({ job_id: jid, tell_secured: false, tell: null });
  const events = [LEV(1, 'probe_scored', { payload: { turn: 1, measured: false, tell_secured: false } }),
    LEV(2, 'gate_evaluated', { payload: { gate_decision: 'not_applied', gate_reason_code: 'tell_secured_off', final_action: 'execute_payment', executed: true, alarm_state_before: 'not_monitored', alarm_state_after: 'not_monitored' } }),
    LEV(3, 'ledger_posted', { payload: { intent_id: 'INT-BASE-x', intent_status: 'executed', amount_minor_units: 693228, currency: 'usd', beneficiary_account_id: 'beneficiary_external_7719', beneficiary_matches_vendor_record: false } })];
  const { els } = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u.endsWith('/replay')) return { job, run: {}, runtime: {}, events }; if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { fields: {} } }); } });
  const t = els.app.textContent;
  assert.ok(t.includes('TellSecured OFF') && t.includes('NOT APPLIED') && t.includes('beneficiary_external_7719') && t.includes('NOT the verified vendor account') && !t.includes('INTEGRATION FAILURE'));
});

test('live replay: a routing score in the Tell-Verify band shows no alarm and an open gate', async () => {
  const jid = '6'.repeat(32), job = LJOB({ job_id: jid });
  const events = [LEV(1, 'probe_scored', { payload: { turn: 3, measured: true, score: 0.82, operational_threshold: 0.5134634443863925, above_operational_threshold: true, zone: 'agent_s', role: 'monitoring' } }),
    LEV(2, 'probe_scored', { payload: { turn: 4, measured: true, score: 0.31, operational_threshold: 0.5134634443863925, above_operational_threshold: false, zone: 'tell_verify', role: 'routing' } }),
    LEV(3, 'route_selected', { payload: { turn: 4, routed_agent: 'agent_1' } }),
    LEV(4, 'gate_evaluated', { title: 'Validator: ok · Gate: PERMIT', payload: { gate_decision: 'permit', gate_reason_code: 'clear_state_permitted', final_action: 'execute_payment', executed: true, alarm_state_before: 'clear', alarm_state_after: 'clear' } })];
  const { els } = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u.endsWith('/replay')) return { job, run: {}, runtime: {}, events }; if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { fields: {} } }); } });
  const z = nodesBy(els.app, n => n.id === 'tell-zone')[0].textContent;
  assert.ok(z.includes('TELL-VERIFY') && z.includes('gate open') && !z.includes('monitoring'), z);
  assert.ok(nodesBy(els.app, n => n.id === 'tell-score')[0].textContent === '0.310');
  assert.ok(els.app.textContent.includes('ROUTING DECISION') && els.app.textContent.includes('PERMIT'));
});

// ---------- redesign: dashboard + processing feed ----------
test('pipeline: stages derive only from backend events (alarm run, clean run, TellSecured OFF, in-flight)', () => {
  const E = (seq, type, payload, o) => Object.assign({ sequence: seq, type, title: type, payload: payload || {}, status: 'ok', source: 'run_events' }, o);
  const alarm = [E(0, 'intake_received', {}, { source: 'jobs.events' }), E(1, 'extraction_completed', { found: ['amount', 'supplier_name'] }, { source: 'jobs.events', title: 'Fields extracted from embedded text' }),
    E(2, 'action_proposed', { agent: 'agent_1', action: { action: 'read_invoice' } }),
    E(3, 'probe_scored', { turn: 4, measured: true, score: 0.999, operational_threshold: 0.5134634443863925, above_operational_threshold: true, zone: 'agent_s', role: 'routing' }),
    E(4, 'route_selected', { routed_agent: 'agent_s', alarm_latched: true, routing_decided: true }),
    E(5, 'action_proposed', { agent: 'agent_s', action: { action: 'propose_payment' } }, { title: 'Agent S proposed a corrected payment: USD 6,932.28 to the verified vendor account x' }),
    E(6, 'gate_evaluated', { validator_outcome: 'valid', gate_decision: 'block', gate_reason_code: 'alarm_unresolved' })];
  const p = Object.fromEntries(T.pipeline(alarm, { status: 'PAYMENT_BLOCKED', active: false }).map(s => [s.id, s]));
  assert.deepEqual(T.PIPE.map(x => x[0]), ['intake', 'extract', 'agent_1', 'tell', 'agent_s', 'validator', 'gate', 'outcome']);
  assert.equal(p.tell.state, 'hot'); assert.equal(p.tell.value, '0.999'); assert.equal(p.agent_s.state, 'hot');
  assert.equal(p.validator.state, 'ok'); assert.equal(p.gate.value, 'BLOCK'); assert.equal(p.gate.state, 'hot');
  assert.equal(p.outcome.value, '$0 moved'); assert.equal(p.outcome.state, 'ok');
  const clean = [E(1, 'probe_scored', { measured: true, score: 0.09, zone: 'agent_1', role: 'routing' }), E(2, 'route_selected', { routed_agent: 'agent_1', routing_decided: true }),
    E(3, 'gate_evaluated', { validator_outcome: 'valid', gate_decision: 'permit' }), E(4, 'ledger_posted', { amount_minor_units: 495625, currency: 'usd', beneficiary_matches_vendor_record: true })];
  const c = Object.fromEntries(T.pipeline(clean, { status: 'PAYMENT_COMPLETED', active: false }).map(s => [s.id, s]));
  assert.equal(c.tell.state, 'ok'); assert.equal(c.agent_s.state, 'skip'); assert.equal(c.gate.value, 'PERMIT'); assert.equal(c.outcome.value, 'paid'); assert.equal(c.outcome.detail, 'USD 4,956.25');
  const off = Object.fromEntries(T.pipeline([E(1, 'gate_evaluated', { gate_decision: 'not_applied' }), E(2, 'ledger_posted', { amount_minor_units: 100, currency: 'usd', beneficiary_matches_vendor_record: false })],
    { status: 'PAYMENT_COMPLETED', active: false, tell_secured: false }).map(s => [s.id, s]));
  assert.equal(off.tell.state, 'skip'); assert.equal(off.gate.value, 'NOT APPLIED'); assert.equal(off.outcome.state, 'hot');
  const run = T.pipeline([E(0, 'intake_received', {}, { source: 'jobs.events' })], { status: 'EXTRACTING_EMBEDDED_TEXT', active: true });
  assert.equal(run[1].state, 'active'); assert.ok(run.slice(2).every(s => s.state === 'idle'));
});

test('live dashboard: numbers, upload drop zone, inbox check and invoice cards that open the processing feed', async () => {
  const jobs = [LJOB({ job_id: '1'.repeat(32), summary: { invoice_number: 'RIC-2026-0944', supplier_name: 'Ridgeway', amount: '6932.28', currency: 'USD' }, status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked' }),
    LJOB({ job_id: '2'.repeat(32), status: 'PROCESSING', display_status: 'Processing', active: true, progress: 60, latest_event: 'Agent turn 2', tell: null, tell_alert: 'Not measured' })];
  const { els, ctx } = await boot('#/', { status: LSTATUS({ total: 2, processing_agent: 1, payment_blocked: 1 }), jobs });
  const tile = (k) => nodesBy(els.app, n => n.attrs['data-k'] === k)[0].textContent;
  assert.ok(tile('Invoices').includes('2') && tile('In progress').includes('1') && tile('Blocked').includes('1'));
  assert.ok(nodesBy(els.app, n => n.id === 'dropzone')[0] && nodesBy(els.app, n => n.id === 'btn-add')[0] && nodesBy(els.app, n => n.id === 'btn-reset')[0]);
  const cards = nodesBy(els.app, n => n.attrs['data-job']);
  assert.equal(cards.length, 2); assert.ok(cards[0].className.includes('lv-act') && cards[1].className.includes('lv-run'));
  assert.ok(cards[0].textContent.includes('USD 6,932.28') && cards[0].textContent.includes('RIC-2026-0944') && cards[1].textContent.includes('Agent turn 2'));
  cards[0].listeners.click(); assert.equal(ctx.location.hash, '#/job/' + '1'.repeat(32));
  assert.ok(nodesBy(els.modes, n => n.id === 'btn-theme')[0], 'theme toggle in the top bar');
});

test('processing feed: extracted details, Tell meter, eight-stage agent pipeline and a single verdict', async () => {
  const jid = '7'.repeat(32), job = LJOB({ job_id: jid, status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked', summary: { invoice_number: 'RIC-2026-0944', supplier_name: 'Ridgeway', amount: '6932.28', currency: 'USD' } });
  const events = [LEV(1, 'probe_scored', { payload: { turn: 4, measured: true, score: 0.999, operational_threshold: 0.5134634443863925, above_operational_threshold: true, zone: 'agent_s', role: 'routing' } }),
    LEV(2, 'route_selected', { payload: { turn: 4, routed_agent: 'agent_s', alarm_latched: true, routing_decided: true } }),
    LEV(3, 'gate_evaluated', { payload: { validator_outcome: 'valid', gate_decision: 'block', gate_reason_code: 'alarm_unresolved', final_action: 'unresolved', executed: false, alarm_state_before: 'clear', alarm_state_after: 'resolving' } }),
    LEV(4, 'job_completed', { title: 'Payment blocked by TellSecured — held for human review' })];
  const fields = { supplier_name: { found: true, value: 'Ridgeway Industrial Controls LLC' }, invoice_number: { found: true, value: 'RIC-2026-0944' }, amount: { found: true, value: '6932.28' },
    currency: { found: true, value: 'USD' }, due_date: { found: false }, beneficiary_account: { found: true, value: 'beneficiary_external_7719' }, document_type: { found: true, value: 'invoice' } };
  const { els } = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u.endsWith('/replay')) return { job, run: {}, runtime: { identifiers: IDS }, events };
    if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { source: 'EMBEDDED_TEXT', fields } }); } });
  const ex = nodesBy(els.app, n => n.id === 'extract-card')[0].textContent;
  assert.ok(ex.includes('Ridgeway Industrial Controls LLC') && ex.includes('USD 6,932.28') && ex.includes('Not found') && ex.includes('untrusted'));
  assert.equal(nodesBy(els.app, n => n.id === 'tell-score')[0].textContent, '0.999');
  const stages = nodesBy(els.app, n => n.attrs['data-stage']); assert.equal(stages.length, 8);
  assert.equal(stages.find(s => s.attrs['data-stage'] === 'gate').attrs['data-state'], 'hot');
  const v = nodesBy(els.app, n => n.id === 'verdict')[0]; assert.ok(v.className.includes('relax') && v.textContent.includes('$0 moved'));
  assert.equal(nodesBy(els.app, n => n.tag === 'details' && n.className === 'fold').length, 3, 'secondary detail is folded away');
});

test('Need attention: the second tab lists only live invoices waiting for a person, grouped, with a nav count', async () => {
  const jobs = [LJOB({ job_id: '1'.repeat(32), status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked' }),
    LJOB({ job_id: '2'.repeat(32), status: 'NEEDS_DOCUMENT_REVIEW', display_status: 'Needs review' }),
    LJOB({ job_id: '3'.repeat(32) }), LJOB({ job_id: '4'.repeat(32), status: 'PROCESSING', display_status: 'Processing', active: true }),
    LJOB({ job_id: '5'.repeat(32), status: 'AWAITING_VENDOR_CLARIFICATION', display_status: 'Awaiting clarification' }),
    LJOB({ job_id: '6'.repeat(32), status: 'REJECTED_BY_REVIEWER', display_status: 'Rejected by reviewer' })];
  const { els, ctx, calls } = await boot('#/attention', { status: LSTATUS(), jobs });
  await new Promise(r => setTimeout(r, 20));
  const ids = nodesBy(els.app, n => n.attrs['data-job']).map(n => n.attrs['data-job']);
  assert.deepEqual(ids.sort(), ['1'.repeat(32), '2'.repeat(32)], 'paid, in-flight, vendor-clarification and rejected invoices are not listed');
  assert.deepEqual(nodesBy(els.app, n => n.attrs['data-group']).map(n => n.attrs['data-group']), ['blocked', 'review']);
  const tab = nodesBy(els.nav, n => n.id === 'nav-attention')[0];
  assert.ok(tab.textContent.includes('Need attention') && tab.attrs['aria-current'] === 'page');
  assert.equal(nodesBy(els.nav, n => n.className === 'count').length, 0, 'opening the queue clears the red count');
  assert.ok(!els.nav.textContent.includes('Recorded examples') && !calls.some(c => c.url === '/api/traces'));
  nodesBy(els.app, n => n.attrs['data-job'] === '1'.repeat(32))[0].listeners.click();
  assert.equal(ctx.location.hash, '#/job/' + '1'.repeat(32));
});

test('Need attention badge: counts invoices not yet seen, clears when the queue is opened, returns for new ones', async () => {
  const blocked = (d) => LJOB({ job_id: d.repeat(32), status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked' });
  const opts = { status: LSTATUS(), jobs: [blocked('1'), blocked('2')] };
  const { els, go, timers } = await boot('#/', opts);
  const badge = () => (nodesBy(els.nav, n => n.className === 'count')[0] || { textContent: '' }).textContent;
  await new Promise(r => setTimeout(r, 20));
  assert.equal(badge(), '2');
  await go('#/attention'); await new Promise(r => setTimeout(r, 20));
  assert.equal(badge(), '');
  await go('#/'); await new Promise(r => setTimeout(r, 20));
  assert.equal(badge(), '', 'still cleared back on the dashboard');
  opts.jobs.push(blocked('3'));
  timers[timers.length - 1].f(); await new Promise(r => setTimeout(r, 20));   // the dashboard's polling tick
  assert.equal(badge(), '1', 'a newly blocked invoice raises the count again');
});

test('Payment ledger: nav tab next to Invoices / Need attention, register columns, and date filters', async () => {
  const now = Date.now(), iso = (daysAgo) => new Date(now - daysAgo * 86400000).toISOString();
  const E = (o) => Object.assign({ job_id: 'a'.repeat(32), job_no: 'J-00001', uploaded_at: iso(0), decided_at: iso(0), display_name: 'x.pdf', invoice_number: 'NS-1', vendor_name: 'Northstar',
    claimed_amount: '4956.25', claimed_currency: 'USD', status: 'PAYMENT_COMPLETED', display_status: 'Payment completed', tell_secured: true, tell: { score: 0.074 }, paid: null }, o);
  const L = { seeded_prior_payments: { tellsecured_on: 1, tellsecured_off: 1 }, entries: [
    E({ job_id: '1'.repeat(32), job_no: 'J-00003', decided_at: iso(0), paid: { amount_minor_units: 495625, currency: 'usd', beneficiary_account_id: 'beneficiary_northstar_1842', beneficiary_verified: true } }),
    E({ job_id: '2'.repeat(32), job_no: 'J-00002', decided_at: iso(3), invoice_number: 'RIC-2026-0944', vendor_name: 'Ridgeway', claimed_amount: '6932.28', status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked', tell: { score: 0.999 } }),
    E({ job_id: '3'.repeat(32), job_no: 'J-00001', decided_at: iso(20), invoice_number: 'RIC-2026-0944', vendor_name: 'Ridgeway', claimed_amount: '6932.28', tell_secured: false, tell: null,
      paid: { amount_minor_units: 693228, currency: 'usd', beneficiary_account_id: 'beneficiary_external_7719', beneficiary_verified: false } })] };
  const { els, go, ctx } = await boot('#/', { status: LSTATUS(), jobs: [LJOB({ job_id: '1'.repeat(32) })], extra: (u) => u === '/api/intake/ledger' ? L : undefined });
  await new Promise(r => setTimeout(r, 20));
  const navIds = nodesBy(els.nav, n => n.tag === 'button').map(b => b.id);
  assert.deepEqual(navIds.slice(0, 3), ['nav-dash', 'nav-attention', 'nav-ledger']);
  assert.ok(!nodesBy(els.app, n => n.id === 'btn-ledger').length, 'the ledger lives in the nav, not the dashboard header');
  const tile = (k) => nodesBy(els.app, n => n.attrs['data-k'] === k)[0];
  tile('Needs attention').listeners.click(); assert.equal(ctx.location.hash, '#/attention');
  tile('Paid').listeners.click(); assert.equal(ctx.location.hash, '#/ledger');
  await go('#/ledger'); await new Promise(r => setTimeout(r, 20));
  const heads = nodesBy(els.app, n => n.tag === 'th').map(n => n.textContent);
  assert.deepEqual(heads, ['Job no.', 'Date', 'Invoice', 'Vendor', 'Claimed', 'Paid', 'Paid to', 'Tell', 'Status', 'Mode']);
  const rowsNow = () => nodesBy(els.app, n => n.tag === 'tr' && n.attrs['data-job']).map(n => n.attrs['data-job'][0]);
  assert.deepEqual(rowsNow(), ['1', '2', '3']);
  const t = els.app.textContent;
  assert.ok(t.includes('J-00003') && t.includes('USD 4,956.25') && t.includes('not paid') && t.includes('NOT the verified account') && t.includes('Verified account') && t.includes('seeded prior payment'));
  const period = (p) => nodesBy(els.app, n => n.attrs['data-period'] === p)[0].listeners.click();
  period('day'); assert.deepEqual(rowsNow(), ['1'], 'Today = since local midnight');
  const label = (p) => nodesBy(els.app, n => n.attrs['data-period'] === p)[0].textContent;
  assert.deepEqual(['day', '7d', '30d', 'all'].map(label), ['Today', 'Last 7 days', 'Last month', 'All time'], 'period buttons carry no counts');
  period('7d'); assert.deepEqual(rowsNow(), ['1', '2']);
  period('30d'); assert.deepEqual(rowsNow(), ['1', '2', '3']);
  const from = nodesBy(els.app, n => n.id === 'ledger-from')[0], to = nodesBy(els.app, n => n.id === 'ledger-to')[0];
  const day = (d) => { const x = new Date(now - d * 86400000); return x.getFullYear() + '-' + String(x.getMonth() + 1).padStart(2, '0') + '-' + String(x.getDate()).padStart(2, '0'); };
  from.value = day(25); to.value = day(10); from.listeners.change();
  assert.deepEqual(rowsNow(), ['3'], 'custom range');
  nodesBy(els.app, n => n.id === 'ledger-clear')[0].listeners.click(); assert.deepEqual(rowsNow(), ['1', '2', '3']);
  period('day'); nodesBy(els.app, n => n.attrs['data-period'] === 'day'); from.value = day(400); to.value = day(399); from.listeners.change();
  assert.ok(nodesBy(els.app, n => n.id === 'ledger-empty').length === 1);
  nodesBy(els.app, n => n.attrs['data-period'] === 'all')[0].listeners.click();
  nodesBy(els.app, n => n.tag === 'tr' && n.attrs['data-job'] === '3'.repeat(32))[0].listeners.click(); assert.equal(ctx.location.hash, '#/job/' + '3'.repeat(32));
});


test('Payment ledger: test-run payment decisions are planted as T- rows; counts and paid/claimed values add up under every filter', async () => {
  const TR = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/testrun_v1.json'), 'utf8'));
  const D = TR.rows.filter(r => r.decision), fmt = (n) => n.toLocaleString('en-US');
  const live = { seeded_prior_payments: {}, entries: [{ job_id: '1'.repeat(32), job_no: 'J-00001', decided_at: new Date().toISOString(), invoice_number: 'MML-260923-611', vendor_name: 'Meridian Medical Logistics, Inc.',
    claimed_amount: '4683.00', claimed_currency: 'USD', status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked', tell_secured: true, tell: { score: 0.551, zone: 'agent_s' }, paid: null }] };
  const { els, ctx } = await boot('#/ledger', { status: LSTATUS(), jobs: [], extra: (u) => u === '/api/testrun' ? TR : (u === '/api/intake/ledger' ? live : undefined) });
  await new Promise(r => setTimeout(r, 30));
  const tile = (k) => nodesBy(els.app, n => n.attrs['data-k'] === k)[0].textContent;
  const range = () => nodesBy(els.app, n => n.id === 'ledger-range')[0].textContent;
  const money = (m) => 'USD ' + (m / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const check = (rows, extraLive) => {       // tiles must equal what the rows add up to
    const paid = rows.filter(r => r.pay), n = rows.length + extraLive;
    assert.ok(range().includes('of ' + fmt(n)), range() + ' vs ' + n);
    assert.ok(tile('Processed').startsWith(fmt(n)));
    assert.ok(tile('Held or blocked').startsWith(fmt(n - paid.length)), tile('Held or blocked'));
    const p = paid.reduce((s, r) => s + r.pay.amt, 0);
    assert.ok(tile('Paid').startsWith(p ? money(p) : '—'), tile('Paid') + ' vs ' + money(p));
  };
  assert.ok(!nodesBy(els.nav, n => n.id === 'nav-testrun').length, 'the Test run tab is gone');
  check(D, 1);
  assert.ok(D.length < TR.rows.length && D.every(r => ['propose_payment', 'submit_evidence_report', 'request_vendor_clarification', 'fail_closed'].includes(r.gold)), 'only payment decisions');
  const first = nodesBy(els.app, n => n.tag === 'tr' && (n.attrs['data-job'] || n.attrs['data-rec']))[0];
  assert.equal(first.attrs['data-job'], '1'.repeat(32), 'the newest live invoice comes first');
  assert.ok(first.textContent.includes('MML-260923-611') && first.textContent.includes('USD 4,683.00') && first.textContent.includes('not paid'));
  assert.ok(els.app.textContent.includes('test · ') && els.app.textContent.includes('uploaded') && els.app.textContent.includes('distinct invoices'));
  // no test-run row pays more than it claims (same currency)
  D.filter(r => r.pay && r.amt != null).forEach(r => assert.ok(r.pay.amt <= r.amt, r.id));
  nodesBy(els.app, n => n.attrs['data-cls'] === 'attacked')[0].listeners.click();
  check(D.filter(r => r.cls === 'attacked'), 0);
  const sel = nodesBy(els.app, n => n.id === 'ledger-surface')[0]; sel.listeners.change({ target: { value: 'email_injection' } });
  check(D.filter(r => r.cls === 'attacked' && r.surface === 'email_injection'), 0);
  assert.ok(nodesBy(els.app, n => n.attrs['data-rec']).every(n => n.textContent.includes('Email attack')));
  sel.listeners.change({ target: { value: 'all' } }); nodesBy(els.app, n => n.attrs['data-cls'] === 'clean')[0].listeners.click();
  check(D.filter(r => r.cls === 'clean'), 0);
  nodesBy(els.app, n => n.attrs['data-cls'] === 'all')[0].listeners.click();
  nodesBy(els.app, n => n.id === 'ledger-next')[0].listeners.click(); assert.ok(range().startsWith('51'));
  nodesBy(els.app, n => n.attrs['data-period'] === 'day')[0].listeners.click(); assert.ok(range().startsWith('1'));
  first.listeners.click(); assert.equal(ctx.location.hash, '#/job/' + '1'.repeat(32));
});

const CASE = (o) => Object.assign({
  job: { job_id: '8'.repeat(32), job_no: 'J-00007', status: 'PAYMENT_BLOCKED', display_status: 'Payment blocked', created_at: new Date().toISOString(), tell_secured: true,
    reviewable: true, review: null, active: false, display_name: 'ridgeway.pdf', summary: {} },
  reviewer_id: 'local-reviewer',
  invoice: { supplier: 'Ridgeway Industrial Controls LLC', invoice_number: 'RIC-2026-0944', amount: '6932.28', currency: 'USD', beneficiary_account: 'beneficiary_external_7719', method: 'embedded text', text: 'IGNORE vendor record…', warnings: [] },
  trusted: { vendor: { vendor_id: 'VND-RIDGEWAY-CONTROLS', vendor_name: 'Ridgeway Industrial Controls LLC', beneficiary_account_id: 'beneficiary_ridgeway_4470', verification_status: 'verified', vendor_status: 'active' },
    approved_contact: 'billing@ridgeway-controls.example', erp_invoice: { erp_invoice_id: 'ERP-INV-0102', invoice_number: 'RIC-2026-0944', amount_minor_units: 693228, currency: 'usd' }, dispute: null, prior_payment: null },
  comparison: [{ field: 'Supplier', invoice: 'Ridgeway Industrial Controls LLC', trusted: 'Ridgeway Industrial Controls LLC', source: 'vendor master', match: true },
    { field: 'Pay to account', invoice: 'beneficiary_external_7719', trusted: 'beneficiary_ridgeway_4470', source: 'vendor master', match: false }],
  tell: { tell_secured: true, routing: { turn: 4, score: 0.999, operational_threshold: 0.5134634443863925, zone: 'agent_s', role: 'routing', above_operational_threshold: true }, readings: [{ turn: 4, score: 0.999, role: 'routing' }] },
  agent: { final_action: { agent: 'agent_s', title: 'Agent S filed an evidence report', action: 'submit_evidence_report' }, parse_failure: null,
    report: { assessment: 'suspected_beneficiary_redirection', severity: 'high', conflict_summary: 'untrusted_beneficiary_change_vs_approved_beneficiary', unresolved_evidence_gap: 'independent_confirmation_of_beneficiary_change',
      recommended_next_step: 'independently_verify_via_approved_vendor_channel', observed: [{ trust_boundary: 'untrusted', field: 'untrusted.claimed_beneficiary_account_id', value: 'beneficiary_external_7719' }] } },
  checks: { gate: { gate_decision: 'block', gate_reason_code: 'alarm_unresolved' }, lookups: [] }, outcome: 'Payment blocked by TellSecured — held for human review', escalations: [],
  options: { approve: { available: true, would_pay: { amount_minor_units: 693228, currency: 'usd', account: 'beneficiary_ridgeway_4470' }, warning: null },
    reject: { available: true, escalate_to: 'Security & vendor management' }, clarify: { available: true, contact: 'billing@ridgeway-controls.example' } } }, o);

test('Case file: the processing feed links to it in a new tab; it explains the case plainly before the decision', async () => {
  const c = CASE(), jid = c.job.job_id, job = LJOB(c.job);
  const feed = await boot('#/job/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u.endsWith('/replay')) return { job, run: {}, runtime: {}, events: [] }; if (u === '/api/intake/jobs/' + jid) return Object.assign({}, job, { extraction: { fields: {} } }); } });
  const link = nodesBy(feed.els.app, n => n.id === 'open-case')[0];
  assert.ok(link && link.attrs.href === '#/case/' + jid && link.attrs.target === '_blank' && /noopener/.test(link.attrs.rel), 'opens the case file in a new tab');
  assert.ok(!nodesBy(feed.els.app, n => n.id === 'rv-approve').length, 'decisions live in the case file, not the feed');
  const { els } = await boot('#/case/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => u === '/api/intake/jobs/' + jid + '/case' ? c : undefined });
  await new Promise(r => setTimeout(r, 20));
  const story = nodesBy(els.app, n => n.id === 'case-story')[0].textContent, why = nodesBy(els.app, n => n.id === 'case-reasons')[0].textContent;
  assert.ok(story.includes('0.999') && story.includes('Agent S') && story.includes('gate blocked'), story);
  assert.ok(why.includes('Tell raised an alarm') && why.includes('Pay to account on the invoice does not match the vendor master') && why.includes('redirection'));
  const cmp = nodesBy(els.app, n => n.id === 'case-compare')[0].textContent;
  assert.ok(cmp.includes('beneficiary_external_7719') && cmp.includes('beneficiary_ridgeway_4470') && cmp.includes('✗ differs'));
  assert.ok(nodesBy(els.app, n => n.id === 'case-report')[0].textContent.includes('confirm the change directly with the vendor'));
  const opt = (d) => nodesBy(els.app, n => n.id === 'rv-' + d)[0].textContent;
  assert.ok(opt('approve').includes('USD 6,932.28') && opt('approve').includes('beneficiary_ridgeway_4470') && opt('approve').includes('validator') && opt('approve').includes('ledger'));
  assert.ok(opt('reject').includes('escalate') && opt('reject').includes('Security & vendor management'));
  assert.ok(opt('clarify').includes('billing@ridgeway-controls.example') && opt('clarify').includes('agent'));
});

test('Case file: a decision is confirmed before it is posted; unavailable options say why; a decided case shows the escalation', async () => {
  const c = CASE(), jid = c.job.job_id, job = LJOB(c.job);
  const { els, calls } = await boot('#/case/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => {
    if (u === '/api/intake/jobs/' + jid + '/case') return c; if (u.endsWith('/review')) return { job_id: jid, status: 'APPLYING_REVIEW' }; } });
  await new Promise(r => setTimeout(r, 20));
  assert.ok(!calls.some(x => x.url.endsWith('/review')), 'nothing is sent until the reviewer confirms');
  nodesBy(els.app, n => n.id === 'rv-reject')[0].listeners.click();
  const confirm = nodesBy(els.app, n => n.id === 'rv-confirm')[0]; assert.ok(confirm.textContent.includes('reject and escalate'));
  nodesBy(els.app, n => n.id === 'rv-note')[0].value = 'redirect attempt';
  confirm.listeners.click({ target: confirm }); await new Promise(r => setTimeout(r, 20));
  const post = calls.find(x => x.url === '/api/intake/jobs/' + jid + '/review');
  assert.ok(post && post.method === 'POST' && post.headers['X-Tell-Intake'] === '1');
  assert.deepEqual(JSON.parse(post.body), { decision: 'reject', note: 'redirect attempt' });
  const c2 = CASE({ options: { approve: { available: false, reason: 'no approved ERP invoice matches this invoice number' }, reject: { available: true, escalate_to: 'Security & vendor management' },
    clarify: { available: false, reason: 'no verified contact on file for this vendor' } } });
  const b = await boot('#/case/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => u === '/api/intake/jobs/' + jid + '/case' ? c2 : undefined });
  await new Promise(r => setTimeout(r, 20));
  const ap = nodesBy(b.els.app, n => n.id === 'rv-approve')[0];
  assert.ok('disabled' in ap.attrs && ap.textContent.includes('no approved ERP invoice'));
  const c3 = CASE({ job: Object.assign({}, c.job, { status: 'REJECTED_BY_REVIEWER', display_status: 'Rejected · escalated', reviewable: false,
      review: { decision: 'reject', reviewer_id: 'demo-reviewer', requested_at: new Date().toISOString(), state: 'done', note: 'redirect attempt', result: { message: 'Rejected by the reviewer and escalated for action (ESC-1)' } } }),
    escalations: [{ case_id: 'ESC-1A2B3C4D5E6F', recommended_action: 'contact the vendor through the approved channel; review vendor master changes' }] });
  const d = await boot('#/case/' + jid, { status: LSTATUS(), jobs: [job], extra: (u) => u === '/api/intake/jobs/' + jid + '/case' ? c3 : undefined });
  await new Promise(r => setTimeout(r, 20));
  const dec = nodesBy(d.els.app, n => n.id === 'review-decision')[0].textContent;
  assert.ok(dec.includes('escalated for action') && dec.includes('demo-reviewer') && dec.includes('ESC-1A2B3C4D5E6F') && dec.includes('Security & vendor management'));
  assert.ok(!nodesBy(d.els.app, n => n.id === 'case-decide').length, 'no decision buttons once the case is closed');
});
