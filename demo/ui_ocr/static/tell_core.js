/* Pure UI logic for the Tell demo. No DOM, no network. UMD: window.TellCore / require(). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TellCore = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  var FIXTURE_LABEL = 'SIMULATED DEMO FIXTURE — NOT MEASURED MODEL PERFORMANCE';
  var MODE = { REAL: 'REPLAY — REAL CAPTURE', FIXTURE: 'SIMULATED DEMO FIXTURE', LIVE: 'LIVE LOCAL INFERENCE' };
  var STATUSES = ['PROCESSING', 'FOLLOW_UP', 'ESCALATED', 'PAID', 'BLOCKED'];

  function zoneFor(score, lower, upper) {
    if (typeof score !== 'number' || !isFinite(score) || score < 0 || score > 1) throw new Error('score out of range');
    if (!(lower >= 0 && lower < upper && upper <= 1)) throw new Error('bad thresholds');
    return score < lower ? 'agent_1' : (score < upper ? 'tell_verify' : 'agent_s');
  }
  function zoneSegments(lower, upper) {
    return [
      { zone: 'agent_1', label: 'Agent 1', from: 0, to: lower },
      { zone: 'tell_verify', label: 'Tell-Verify', from: lower, to: upper },
      { zone: 'agent_s', label: 'Agent S', from: upper, to: 1 }
    ];
  }
  /* Which evidence modes a trace contains, derived from field-level provenance (never assumed). */
  function provenanceOf(trace) {
    var real = 0, fixture = 0;
    function tally(p) { if (p === 'real') real++; else if (p === 'fixture') fixture++; }
    trace.events.forEach(function (e) { Object.keys(e.prov || {}).forEach(function (k) { tally(e.prov[k]); }); });
    ['invoice', 'trusted_vendor_record'].forEach(function (s) {
      Object.keys(trace[s] || {}).forEach(function (k) { tally(trace[s][k] && trace[s][k].p); });
    });
    var modes = [];
    if (real) modes.push(MODE.REAL);
    if (fixture) modes.push(MODE.FIXTURE);
    return { real: real, fixture: fixture, modes: modes };
  }
  function eventEvidence(e) {
    var v = {}; Object.keys(e.prov || {}).forEach(function (k) { v[e.prov[k]] = 1; });
    var ks = Object.keys(v);
    return ks.length === 2 ? 'mixed' : (ks[0] || 'fixture');
  }
  /* Fold events[0..idx] into the visual state. idx = -1 -> nothing has happened yet. */
  function stateAt(trace, idx) {
    var s = { step: idx, total: trace.events.length, tell: null, routing: null, activeAgent: null, alarm: null,
      lookups: null, validator: null, gate: null, ledger: null, outcome: null, status: 'PROCESSING',
      started: idx >= 0, done: false, log: [], prov: {} };
    var last = Math.min(idx, trace.events.length - 1);
    for (var i = 0; i <= last; i++) {
      var e = trace.events[i]; s.log.push(e);
      ['tell', 'routing', 'alarm', 'lookups', 'validator', 'gate', 'ledger', 'outcome'].forEach(function (k) {
        if (e[k] != null) { s[k] = e[k]; s.prov[k] = e.prov[k]; }
      });
      if (e.routing) s.activeAgent = e.routing.to;
      else if (e.tell && !s.activeAgent) s.activeAgent = 'agent_1';
    }
    if (s.outcome) s.status = s.outcome;
    else if (s.gate && s.gate.decision === 'BLOCK') s.status = 'BLOCKED';
    s.done = last === trace.events.length - 1 && idx >= 0;
    return s;
  }
  /* Guided playback. Timers are injected so it is testable without a clock. */
  function createPlayer(trace, opts) {
    opts = opts || {};
    var setT = opts.setInterval || setInterval, clrT = opts.clearInterval || clearInterval;
    var ms = opts.intervalMs || 1600, onChange = opts.onChange || function () {};
    var idx = -1, timer = null;
    function emit() { onChange(api.state()); }
    var api = {
      state: function () { var s = stateAt(trace, idx); s.playing = timer !== null; return s; },
      index: function () { return idx; },
      start: function () { if (idx < 0) idx = 0; emit(); },
      next: function () { if (idx < trace.events.length - 1) idx++; if (idx >= trace.events.length - 1) api.pause(); emit(); },
      play: function () {
        if (timer !== null) return;
        if (idx >= trace.events.length - 1) idx = -1;
        if (idx < 0) idx = 0;
        timer = setT(function () { api.next(); }, ms); emit();
      },
      pause: function () { if (timer !== null) { clrT(timer); timer = null; } emit(); },
      seek: function (i) { idx = Math.max(-1, Math.min(trace.events.length - 1, i)); emit(); },
      reset: function () { if (timer !== null) { clrT(timer); timer = null; } idx = -1; emit(); },
      dispose: function () { if (timer !== null) { clrT(timer); timer = null; } }
    };
    return api;
  }
  /* Dashboard counters: derived from the queue where possible; totals add the (fixture) baseline. */
  function summarize(traces, baseline) {
    var c = {}; STATUSES.forEach(function (k) { c[k] = 0; });
    traces.forEach(function (t) { c[t.queue_status]++; });
    return {
      processed: baseline.processed + c.PAID + c.BLOCKED + c.ESCALATED + c.FOLLOW_UP,
      processing: c.PROCESSING, follow_up: c.FOLLOW_UP, escalated: c.ESCALATED,
      paid: baseline.paid + c.PAID, blocked: baseline.blocked + c.BLOCKED, counts: c
    };
  }
  var INTAKE = ['UPLOADING', 'VALIDATING', 'EXTRACTING_EMBEDDED_TEXT', 'OCR_QUEUED', 'OCR_RUNNING', 'OCR_COMPLETE', 'EXTRACTION_COMPLETE', 'PROCESSING',
    'READY_FOR_PROCESSING', 'AWAITING_VENDOR_CLARIFICATION', 'NEEDS_DOCUMENT_REVIEW', 'UNREADABLE_DOCUMENT', 'DUPLICATE', 'FAILED',
    'BLOCKED_BY_DISPUTE', 'PAYMENT_COMPLETED', 'PAYMENT_BLOCKED', 'APPLYING_REVIEW', 'REJECTED_BY_REVIEWER'];
  var INTAKE_ACTIVE = INTAKE.slice(0, 8);
  var INTAKE_LABEL = { UPLOADING: 'Uploading', VALIDATING: 'Validating', EXTRACTING_EMBEDDED_TEXT: 'Extracting text', OCR_QUEUED: 'OCR queued',
    OCR_RUNNING: 'OCR running', OCR_COMPLETE: 'OCR complete', EXTRACTION_COMPLETE: 'Fields extracted', READY_FOR_PROCESSING: 'Ready for processing',
    AWAITING_VENDOR_CLARIFICATION: 'Awaiting clarification', NEEDS_DOCUMENT_REVIEW: 'Needs review', UNREADABLE_DOCUMENT: 'Needs review',
    DUPLICATE: 'Duplicate detected', FAILED: 'Failed', PROCESSING: 'Processing', BLOCKED_BY_DISPUTE: 'Blocked by dispute',
    PAYMENT_COMPLETED: 'Payment completed', PAYMENT_BLOCKED: 'Payment blocked', APPLYING_REVIEW: 'Applying reviewer decision',
    REJECTED_BY_REVIEWER: 'Rejected by reviewer' };
  /* Intake counters: waiting / processing (text + OCR) / ready / awaiting vendor / needs attention / failed. */
  function intakeCounts(jobs) {
    var c = {}; INTAKE.forEach(function (k) { c[k] = 0; });
    jobs.forEach(function (j) { if (c[j.status] !== undefined) c[j.status]++; });
    var active = 0; INTAKE_ACTIVE.forEach(function (k) { active += c[k]; });
    return { waiting: c.UPLOADING + c.VALIDATING, processing: c.EXTRACTING_EMBEDDED_TEXT + c.OCR_QUEUED + c.OCR_RUNNING + c.OCR_COMPLETE + c.EXTRACTION_COMPLETE,
      ready: c.READY_FOR_PROCESSING, awaiting: c.AWAITING_VENDOR_CLARIFICATION, attention: c.NEEDS_DOCUMENT_REVIEW + c.UNREADABLE_DOCUMENT,
      failed: c.FAILED, duplicates: c.DUPLICATE, active: active, counts: c };
  }
  function isActive(status) { return INTAKE_ACTIVE.indexOf(status) >= 0; }
  /* A job is active while the backend says so (live mode also keeps READY_FOR_PROCESSING active until the worker claims it). */
  function jobActive(job) { return typeof job.active === 'boolean' ? job.active : isActive(job.status); }
  function nextStep(job) {
    var m = { PROCESSING: 'The live agent is processing this invoice', PAYMENT_COMPLETED: 'Payment completed in the simulated ledger',
      PAYMENT_BLOCKED: 'Payment blocked by the validator or the deterministic gate', BLOCKED_BY_DISPUTE: 'Blocked \u2014 an open dispute case exists',
      READY_FOR_PROCESSING: 'Ready for agent processing',
      AWAITING_VENDOR_CLARIFICATION: 'Awaiting vendor clarification \u2014 a simulated email was prepared; payment is on hold',
      NEEDS_DOCUMENT_REVIEW: 'Needs document review \u2014 ' + (job.error || 'evidence is ambiguous or incomplete'),
      UNREADABLE_DOCUMENT: 'Unreadable document \u2014 OCR ran but found no usable business content',
      DUPLICATE: 'Duplicate \u2014 identical content is already in the queue', FAILED: 'Failed \u2014 ' + (job.error || 'see the timeline'),
      UPLOADING: 'Waiting for file content', VALIDATING: 'Validating file', EXTRACTING_EMBEDDED_TEXT: 'Extracting embedded text', OCR_QUEUED: 'OCR queued',
      OCR_RUNNING: 'Running local OCR', OCR_COMPLETE: 'OCR complete \u2014 extracting fields', EXTRACTION_COMPLETE: 'Fields extracted \u2014 checking required fields' };
    return m[job.status] || job.status;
  }
  /* Derive the replay sections from persisted backend events. Pure: every value comes from an event payload; absent evidence stays null. */
  function replaySections(events) {
    var s = { turns: [], measured: 0, route: null, action: null, parseFailure: null, gate: null, trustedEvidence: [], lookups: [], ledger: null, intent: null,
      clarification: null, review: null, untrustedEntered: false, result: null, failure: null, tools: [], claimed: null };
    (events || []).forEach(function (e) {
      var p = e.payload || {};
      switch (e.type) {
        case 'probe_scored': s.turns.push({ turn: p.turn, measured: !!p.measured, score: p.measured ? p.score : null, threshold: p.operational_threshold, above: p.above_operational_threshold, layer: p.layer, reason: p.reason, zone: p.zone, role: p.role });
          if (p.measured) s.measured++; break;
        case 'route_selected': s.route = { turn: p.turn, agent: p.routed_agent, model: p.model, reason: p.reason || p.basis, latched: !!p.alarm_latched, adapter: p.adapter_sha256 }; break;
        case 'action_proposed': if (p.discarded) break; s.action = { turn: p.turn, agent: p.agent, action: p.action, tokens: p.generated_tokens, ms: e.duration_ms, title: e.title, check: p.beneficiary_check }; break;
        case 'action_parse_failed': s.parseFailure = { turn: p.turn, agent: p.agent, outcome: p.parse_outcome, error: p.error, raw: p.raw_output_excerpt }; break;
        case 'tool_call_completed': s.tools.push({ tool: p.tool, status: p.status, trust: p.provenance && p.provenance.trust_boundary, ms: e.duration_ms });
          if (p.tool === 'get_vendor_record' && p.status === 'success') s.trustedEvidence.push({ tool: p.tool, content: p.content, provenance: p.provenance }); break;
        case 'trusted_lookup_completed': s.lookups.push({ tool: p.tool, status: p.status, record: p.record, error: p.error }); break;
        case 'untrusted_content_entered_context': s.untrustedEntered = true; break;
        case 'gate_evaluated': s.gate = p; s.gateTitle = e.title; break;
        case 'payment_intent_created': s.intent = p; break;
        case 'ledger_posted': s.ledger = p; break;
        case 'clarification_created': s.clarification = p; break;
        case 'evidence_report_created': s.review = p; break;
        case 'job_claimed': s.claimed = p; break;
        case 'reviewer_decision': s.reviewer = { decision: p.decision, reviewer: p.reviewer_id, note: p.note, title: e.title, ts: e.ts }; break;
        case 'job_completed': s.result = { title: e.title, status: p.status, final_action: p.final_action, routed: p.routed_agent, executed: p.executed }; break;
        case 'job_failed': s.failure = { title: e.title, payload: p }; break;
        default: break;
      }
    });
    return s;
  }
  function tellState(sections, terminal) {
    if (sections.measured > 0) {       // the routing reading decides the run; before it exists, show the latest monitoring reading
      var ms = sections.turns.filter(function (t) { return t.measured; });
      var r = ms.filter(function (t) { return t.role === 'routing'; }).pop() || ms[ms.length - 1];
      return { measured: true, score: r.score, threshold: r.threshold, above: r.above, zone: r.zone, routing: r.role === 'routing' };
    }
    return { measured: false, integrationFailure: !!terminal };
  }
  /* Live invoice -> design level (act = stopped/alarm, ready = needs a person, relax = done safely, run = in flight). */
  var LEVEL = { REJECTED_BY_REVIEWER: 'idle', PAYMENT_COMPLETED: 'relax', PAYMENT_BLOCKED: 'act', BLOCKED_BY_DISPUTE: 'act', FAILED: 'act', UNREADABLE_DOCUMENT: 'act',
    AWAITING_VENDOR_CLARIFICATION: 'ready', NEEDS_DOCUMENT_REVIEW: 'ready', DUPLICATE: 'ready' };
  /* Live invoices that are waiting for a person (none of these has been paid). */
  var ATTENTION = { PAYMENT_BLOCKED: 'blocked', BLOCKED_BY_DISPUTE: 'blocked', NEEDS_DOCUMENT_REVIEW: 'review',
    FAILED: 'failed', UNREADABLE_DOCUMENT: 'failed' };
  function attentionOf(job) { return jobActive(job) ? null : (ATTENTION[job.status] || null); }
  function levelOf(job) { return jobActive(job) ? 'run' : (LEVEL[job.status] || 'idle'); }

  /* Horizontal agent pipeline for one live run, derived ONLY from persisted backend events.
     Each stage: { id, label, state: idle|active|done|ok|warn|hot|skip, value, detail }. */
  var PIPE = [['intake', 'Intake'], ['extract', 'Extraction'], ['agent_1', 'Agent 1'], ['tell', 'Tell'], ['agent_s', 'Agent S'],
    ['validator', 'Validator'], ['gate', 'Gate'], ['outcome', 'Outcome']];
  function pipeline(events, job) {
    var ev = events || [], off = job && job.tell_secured === false, sec = replaySections(ev);
    var S = {}; PIPE.forEach(function (p) { S[p[0]] = { id: p[0], label: p[1], state: 'idle', value: '', detail: '' }; });
    var has = function (t) { return ev.some(function (e) { return e.type === t; }); };
    var intakeFail = ev.some(function (e) { return e.source === 'jobs.events' && e.status === 'failed'; });
    if (ev.some(function (e) { return e.source === 'jobs.events'; })) S.intake = Object.assign(S.intake, { state: 'done', value: 'PDF received' });
    if (has('extraction_completed')) {
      var ex = ev.filter(function (e) { return e.type === 'extraction_completed'; }).pop();
      S.extract = Object.assign(S.extract, { state: 'done', value: /OCR/.test(ex.title || '') ? 'local OCR' : 'embedded text',
        detail: ((ex.payload && ex.payload.found) || []).length + ' fields' });
    } else if (intakeFail) S.extract = Object.assign(S.extract, { state: 'hot', value: 'failed' });
    var acts = ev.filter(function (e) { return e.type === 'action_proposed' && !(e.payload && e.payload.discarded); });
    var a1 = acts.filter(function (e) { return e.payload.agent === 'agent_1'; }), as = acts.filter(function (e) { return e.payload.agent === 'agent_s'; });
    var READS = { read_email: 1, read_invoice: 1, get_vendor_record: 1, search_memory: 1 };
    if (a1.length) {
      var reads = a1.filter(function (e) { return READS[(e.payload.action || {}).action]; }).length, last = a1[a1.length - 1].payload.action || {};
      S.agent_1 = Object.assign(S.agent_1, { state: 'done', value: READS[last.action] ? 'reading' : String(last.action || '').replace(/_/g, ' '),
        detail: reads + ' read' + (reads === 1 ? '' : 's') });
      if (!READS[last.action]) S.agent_1.value = last.action === 'propose_payment' ? 'proposed payment' : S.agent_1.value;
    }
    var routeEvs = ev.filter(function (e) { return e.type === 'route_selected'; });
    var toS = routeEvs.filter(function (e) { return e.payload && e.payload.routed_agent === 'agent_s'; });
    var alarm = toS.some(function (e) { return e.payload.alarm_latched; });
    if (off) {
      S.tell = Object.assign(S.tell, { state: 'skip', value: 'off', detail: 'TellSecured OFF' });
      S.agent_s = Object.assign(S.agent_s, { state: 'skip', value: 'off' });
    } else {
      var st = tellState(sec, job && !jobActive(job));
      if (st.measured) {
        var zone = st.zone || (st.above ? 'agent_s' : 'agent_1');
        S.tell = Object.assign(S.tell, { state: !st.routing ? 'active' : (zone === 'agent_s' ? 'hot' : zone === 'tell_verify' ? 'warn' : 'ok'),
          value: st.score.toFixed(3), detail: !st.routing ? 'watching' : ({ agent_s: 'alarm', tell_verify: 'Tell-Verify', agent_1: 'clear' }[zone]) });
      } else if (job && !jobActive(job) && !has('action_proposed') && !has('probe_scored')) S.tell = Object.assign(S.tell, { state: 'idle', value: '—' });
      else if (job && !jobActive(job)) S.tell = Object.assign(S.tell, { state: 'hot', value: 'not measured' });
      if (toS.length || as.length) {
        var lastS = as.length ? as[as.length - 1] : null;
        S.agent_s = Object.assign(S.agent_s, { state: alarm ? 'hot' : 'warn', value: lastS ? (lastS.title || '').replace(/^Agent S\s*/, '').split(':')[0] || 'took over' : 'took over',
          detail: alarm ? 'alarm raised' : 'no alarm' });
      } else if (routeEvs.some(function (e) { return e.payload && e.payload.routing_decided; }) || (sec.gate && !toS.length))
        S.agent_s = Object.assign(S.agent_s, { state: 'skip', value: 'not needed' });
    }
    var g = sec.gate;
    if (g) {
      var vo = g.validator_outcome;
      if (g.gate_decision === 'not_applied') {
        S.validator = Object.assign(S.validator, { state: 'skip', value: 'not applied' });
        S.gate = Object.assign(S.gate, { state: 'hot', value: 'NOT APPLIED', detail: 'TellSecured OFF' });
      } else if (!g.gate_decision && !vo) {
        S.validator = Object.assign(S.validator, { state: 'skip', value: 'no payment' });
        S.gate = Object.assign(S.gate, { state: 'skip', value: 'no payment' });
      } else {
        S.validator = Object.assign(S.validator, { state: vo === 'valid' ? 'ok' : 'hot', value: vo === 'valid' ? 'passed' : String(vo || 'not run').replace(/_/g, ' ') });
        S.gate = Object.assign(S.gate, { state: g.gate_decision === 'permit' ? 'ok' : 'hot', value: String(g.gate_decision || 'not reached').toUpperCase(),
          detail: g.gate_reason_code === 'alarm_unresolved' ? 'alarm unresolved' : (g.gate_reason_code || '').replace(/_/g, ' ') });
      }
    }
    if (sec.ledger) {
      var bad = sec.ledger.beneficiary_matches_vendor_record === false;
      S.outcome = Object.assign(S.outcome, { state: bad ? 'hot' : 'ok', value: bad ? 'paid unverified' : 'paid', detail: money(sec.ledger.amount_minor_units, sec.ledger.currency) });
    } else if (sec.clarification) S.outcome = Object.assign(S.outcome, { state: 'warn', value: 'clarification', detail: 'simulated email' });
    else if (job && !jobActive(job)) {
      var so = { PAYMENT_BLOCKED: ['ok', '$0 moved'], BLOCKED_BY_DISPUTE: ['ok', '$0 moved'], NEEDS_DOCUMENT_REVIEW: ['warn', 'review'], DUPLICATE: ['warn', 'duplicate'],
        AWAITING_VENDOR_CLARIFICATION: ['warn', 'on hold'], REJECTED_BY_REVIEWER: ['ok', 'rejected'], FAILED: ['hot', 'failed'], UNREADABLE_DOCUMENT: ['hot', 'unreadable'], PAYMENT_COMPLETED: ['ok', 'paid'] }[job.status];
      if (so) S.outcome = Object.assign(S.outcome, { state: so[0], value: so[1], detail: sec.review ? 'evidence report' : '' });
    }
    var list = PIPE.map(function (p) { return S[p[0]]; });
    if (job && jobActive(job)) {                     // the next stage that has not happened yet is the one in progress
      var lastDone = -1; list.forEach(function (x, i) { if (x.state !== 'idle') lastDone = i; });
      var nxt = list[lastDone + 1];
      if (nxt && nxt.state === 'idle') { nxt.state = 'active'; nxt.value = nxt.value || 'working…'; }
    }
    return list;
  }
  function escalations(traces) { return traces.filter(function (t) { return t.queue_status === 'ESCALATED'; }); }
  function findTrace(traces, id) { return traces.filter(function (t) { return t.run_id === id || t.scenario_id === id; })[0] || null; }
  function money(minor, cur) {
    var v = (minor / 100).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return String(cur || '').toUpperCase() + ' ' + v;
  }
  /* Ledger view model: before/after rows + which rows changed. */
  function ledgerRows(l) {
    if (!l) return [];
    var keys = ['operating_balance_minor', 'invoice_status', 'journal_entries'];
    return keys.map(function (k) { return { key: k, before: l.before[k], after: l.after[k], changed: l.before[k] !== l.after[k] }; });
  }
  return { FIXTURE_LABEL: FIXTURE_LABEL, MODE: MODE, STATUSES: STATUSES, zoneFor: zoneFor, zoneSegments: zoneSegments,
    provenanceOf: provenanceOf, eventEvidence: eventEvidence, stateAt: stateAt, createPlayer: createPlayer,
    summarize: summarize, jobActive: jobActive, levelOf: levelOf, attentionOf: attentionOf, pipeline: pipeline, PIPE: PIPE, replaySections: replaySections, tellState: tellState, intakeCounts: intakeCounts, isActive: isActive, nextStep: nextStep, INTAKE: INTAKE, INTAKE_LABEL: INTAKE_LABEL, escalations: escalations, findTrace: findTrace, money: money, ledgerRows: ledgerRows };
}));
