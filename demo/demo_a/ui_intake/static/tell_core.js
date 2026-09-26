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
  var INTAKE = ['UPLOADING', 'VALIDATING', 'EXTRACTING', 'READY_FOR_PROCESSING', 'NEEDS_OCR', 'NEEDS_REVIEW', 'DUPLICATE', 'FAILED'];
  var INTAKE_ACTIVE = ['UPLOADING', 'VALIDATING', 'EXTRACTING'];
  var INTAKE_LABEL = { UPLOADING: 'Uploading', VALIDATING: 'Validating', EXTRACTING: 'Extracting', READY_FOR_PROCESSING: 'Ready for processing',
    NEEDS_OCR: 'Needs OCR', NEEDS_REVIEW: 'Needs review', DUPLICATE: 'Duplicate', FAILED: 'Failed' };
  /* Intake counters answering: waiting / extracting / ready / attention / failed. */
  function intakeCounts(jobs) {
    var c = {}; INTAKE.forEach(function (k) { c[k] = 0; });
    jobs.forEach(function (j) { if (c[j.status] !== undefined) c[j.status]++; });
    return { waiting: c.UPLOADING + c.VALIDATING, extracting: c.EXTRACTING, ready: c.READY_FOR_PROCESSING,
      attention: c.NEEDS_OCR + c.NEEDS_REVIEW, failed: c.FAILED, duplicates: c.DUPLICATE, active: c.UPLOADING + c.VALIDATING + c.EXTRACTING, counts: c };
  }
  function isActive(status) { return INTAKE_ACTIVE.indexOf(status) >= 0; }
  function nextStep(job) {
    var m = { READY_FOR_PROCESSING: 'Ready for agent processing', NEEDS_OCR: 'Needs OCR \u2014 no embedded text was found and OCR is not available',
      NEEDS_REVIEW: 'Needs review \u2014 required fields are missing or unreadable', DUPLICATE: 'Duplicate \u2014 identical content is already in the queue',
      FAILED: 'Failed intake \u2014 ' + (job.error || 'see the timeline'), UPLOADING: 'Waiting for file content', VALIDATING: 'Validating file', EXTRACTING: 'Extracting text' };
    return m[job.status] || job.status;
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
    summarize: summarize, intakeCounts: intakeCounts, isActive: isActive, nextStep: nextStep, INTAKE: INTAKE, INTAKE_LABEL: INTAKE_LABEL, escalations: escalations, findTrace: findTrace, money: money, ledgerRows: ledgerRows };
}));
