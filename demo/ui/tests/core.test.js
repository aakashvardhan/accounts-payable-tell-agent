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
  class Node { constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this._text = null; this.className = ''; this.id = ''; }
    appendChild(c) { this.children.push(c); return c; }
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
async function boot(hash) {
  const { doc, els } = makeDom(); const wl = {};
  const ctx = { document: doc, location: { hash }, window: { addEventListener: (t, f) => { wl[t] = f; }, scrollTo() {} }, TellCore: T,
    fetch: () => Promise.resolve({ json: () => Promise.resolve(JSON.parse(JSON.stringify(bundle))) }), setInterval, clearInterval, console };
  ctx.window.TellCore = T; ctx.self = ctx.window;
  vm.createContext(ctx); vm.runInContext(fs.readFileSync(path.join(ROOT, 'static/app.js'), 'utf8'), ctx);
  await new Promise(r => setTimeout(r, 20));
  const go = async (h) => { ctx.location.hash = h; wl.hashchange(); };
  return { els, ctx, go };
}
const buttons = (n) => n.find(x => x.tag === 'button');

test('dashboard renders KPIs, statuses, story shortcuts, mode badges, fixture banner', async () => {
  const { els } = await boot('#/');
  const txt = els.app.textContent;
  ['Invoices processed', 'Currently processing', 'Awaiting follow-up', 'Escalated for review', 'Payments completed', 'Payments prevented'].forEach(k => assert.ok(txt.includes(k), k));
  ['PROCESSING', 'FOLLOW_UP', 'ESCALATED', 'PAID', 'BLOCKED'].forEach(k => assert.ok(txt.includes(k), k));
  assert.ok(txt.includes('Clean payment') && txt.includes('Escalated attack'));
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
