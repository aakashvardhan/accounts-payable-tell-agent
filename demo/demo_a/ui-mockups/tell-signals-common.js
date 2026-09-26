/* Shared view-model for the tell-signals-* mockups. Reads window.TELL_BUNDLE (tell.demo_trace/1.0)
   and reuses TellCore (../ui_ocr/static/tell_core.js) for zones, money and replay state. No DOM writes here. */
(function () {
  'use strict';
  var B = window.TELL_BUNDLE, C = window.TellCore;

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  /* Provenance tag: every value on screen says where it came from. */
  function prov(p) {
    var label = { real: 'real', fixture: 'fixture', measured: 'measured', calculated: 'calculated',
      assumption: 'assumption', quote: 'quote', gap: 'not measured', mixed: 'mixed' }[p] || p;
    return '<span class="prov prov-' + esc(p) + '" title="' + esc(PROV_HELP[p] || '') + '">' + esc(label) + '</span>';
  }
  var PROV_HELP = {
    real: 'Captured epoch-0 replay of the TRAINING split — not held-out performance',
    fixture: B.fixture_label,
    measured: 'Measured on the ZGX Nano (see source report)',
    calculated: 'Calculated from measured values + stated assumptions',
    assumption: 'Assumption — edit it, it is not measured',
    quote: 'Vendor quote',
    gap: 'Not measured yet',
    mixed: 'Mix of real replay and fixture fields'
  };

  var LEVEL = { ESCALATED: 'act', BLOCKED: 'act', FOLLOW_UP: 'ready', PAID: 'relax', PROCESSING: 'relax' };
  var LEVEL_LABEL = { act: 'Act now', ready: 'Ready to act', relax: 'Relax' };
  var SURFACE = { invoice_injection: 'Invoice', email_injection: 'Email', tool_result_forgery: 'Tool result',
    immediate_memory_poisoning: 'Memory', delayed_memory_poisoning: 'Memory' };
  var ZONE_LABEL = { agent_1: 'Agent 1', tell_verify: 'Tell Verify', agent_s: 'Agent S' };
  var STATUS_LABEL = { ESCALATED: 'Escalated', BLOCKED: 'Blocked', FOLLOW_UP: 'Follow-up', PAID: 'Paid (simulated)', PROCESSING: 'Processing' };

  function last(t, type) { var r = null; t.events.forEach(function (e) { if (e.type === type) r = e; }); return r; }
  function fx(n, d) { return Number(n).toFixed(d == null ? 3 : d); }

  /* Fixture UI copy for statuses that carry no review.recommended_action. */
  var ACTION = {
    BLOCKED: 'Gate held the payment. The memory note stays quarantined until a registered reviewer clears the alarm.',
    FOLLOW_UP: 'Clarification goes to the trusted approved contact on file only — the recipient is never chosen by the model.',
    PAID: 'Nothing to do: validator approved, gate allowed, simulated ledger posted to the vendor-record account.',
    PROCESSING: 'Nothing yet: Agent 1 continues; validator and gate still run before any ledger write.'
  };

  function signal(t) {
    var s = last(t, 'tell_score'), g = last(t, 'gate'), r = last(t, 'routing'), l = last(t, 'ledger'),
      o = last(t, 'outcome'), lk = last(t, 'trusted_lookup');
    var lo = t.thresholds.lower.v, hi = t.thresholds.upper.v, score = s.tell.score;
    var review = t.review || null, inv = t.invoice;
    var surface = inv.attack_surface ? inv.attack_surface.v : null;
    var tokens = 0, gen = 0;
    t.events.forEach(function (e) {
      if (e.tokens) tokens += e.tokens.generated || 0;
      if (e.latency_seconds && e.latency_seconds.generation) gen += e.latency_seconds.generation;
    });
    return {
      t: t, id: t.run_id, scenario: t.scenario_id, kind: t.kind, status: t.queue_status,
      level: LEVEL[t.queue_status], levelLabel: LEVEL_LABEL[LEVEL[t.queue_status]],
      title: t.title, vendor: inv.vendor_name.v,
      amount: C.money(inv.amount_minor_units.v, inv.currency.v),
      invoiceNo: inv.invoice_number ? inv.invoice_number.v : '',
      score: score, zone: C.zoneFor(score, lo, hi), lo: lo, hi: hi,
      capture: s.latency_seconds ? s.latency_seconds.tell_capture : null,
      tokens: tokens, generation: gen,
      alarm: r ? r.alarm : null, routeReason: r ? r.routing.reason : '',
      gate: g ? g.gate.decision : null, gateReason: g ? g.gate.reason : '',
      ledger: l ? l.ledger : null, lookups: lk ? lk.lookups : null,
      outcome: o ? o.outcome : null,
      surface: surface, surfaceLabel: surface ? (SURFACE[surface] || surface) : 'Routine',
      detail: review ? review.triggering_evidence.v : (o && o.summary) || inv.supplier_message_summary.v,
      action: review ? review.recommended_action.v : ACTION[t.queue_status],
      caseId: review ? review.evidence_report.v.case : null,
      tEnd: t.events[t.events.length - 1].t_offset_s,
      chainHead: t.events[t.events.length - 1].audit_hash,
      evidence: C.provenanceOf(t)
    };
  }

  var signals = B.traces.map(signal).sort(function (a, b) {
    var rank = { act: 0, ready: 1, relax: 2 };
    return rank[a.level] - rank[b.level] || b.score - a.score;
  });

  /* One ticker item per run — the most consequential thing that happened. */
  function tick(s) {
    if (s.gate === 'BLOCK') return { level: 'act', head: 'Gate BLOCK', tail: s.vendor + ' · ' + fx(s.score) };
    if (s.status === 'FOLLOW_UP') return { level: 'ready', head: 'Follow-up', tail: s.vendor };
    if (s.status === 'PAID') return { level: 'relax', head: 'Paid (sim)', tail: s.amount };
    if (s.zone === 'agent_s') return { level: 'act', head: 'Tell ' + fx(s.score) + ' → Agent S', tail: s.vendor };
    return { level: 'relax', head: 'Tell ' + fx(s.score) + ' → Agent 1', tail: s.vendor };
  }

  /* Pipeline stage each actor lights up during replay. */
  var STAGE = { intake: 'agent_1', agent_1: 'agent_1', tell: 'tell', router: 'router', agent_s: 'agent_s',
    trusted_lookup: 'agent_s', validator: 'validator', gate: 'gate', ledger: 'ledger', review_queue: 'review' };
  var STAGES = [
    { id: 'agent_1', label: 'Agent 1', sub: 'reads' },
    { id: 'tell', label: 'Tell', sub: 'probe' },
    { id: 'router', label: 'Router', sub: '3 zones' },
    { id: 'agent_s', label: 'Agent S', sub: 'verifies' },
    { id: 'validator', label: 'Validator', sub: 'typed' },
    { id: 'gate', label: 'Gate', sub: 'fail-closed' },
    { id: 'ledger', label: 'Ledger', sub: 'simulated' },
    { id: 'review', label: 'Review', sub: 'human' }
  ];

  /* Stage look after events[0..idx]: idle | done | block (gate BLOCK) | held (ledger unchanged); plus the current stage. */
  function stageStates(t, idx) {
    var st = {}; STAGES.forEach(function (s) { st[s.id] = 'idle'; });
    for (var i = 0; i <= idx && i < t.events.length; i++) {
      var e = t.events[i], id = STAGE[e.actor];
      st[id] = 'done';
      if (e.type === 'gate' && e.gate.decision === 'BLOCK') st.gate = 'block';
      if (e.type === 'ledger' && e.ledger.before.journal_entries === e.ledger.after.journal_entries) st.ledger = 'held';
    }
    return { st: st, current: idx >= 0 ? STAGE[t.events[Math.min(idx, t.events.length - 1)].actor] : null };
  }

  function ledgerLabel(k, v) {
    if (k === 'operating_balance_minor') return C.money(v, 'usd');
    return String(v).replace(/_/g, ' ');
  }

  /* Theme toggle shared by all three pages: cycles auto → light → dark. */
  function themeToggle(btn) {
    var order = ['auto', 'light', 'dark'], cur = 'auto';
    try { cur = localStorage.getItem('tell-theme') || 'auto'; } catch (e) { /* storage blocked */ }
    function apply() {
      if (cur === 'auto') document.documentElement.removeAttribute('data-theme');
      else document.documentElement.setAttribute('data-theme', cur);
      btn.textContent = 'Theme: ' + cur;
    }
    btn.addEventListener('click', function () {
      cur = order[(order.indexOf(cur) + 1) % 3];
      try { localStorage.setItem('tell-theme', cur); } catch (e) { /* storage blocked */ }
      apply();
    });
    apply();
  }

  window.TS = { B: B, C: C, F: window.TELL_FACTS, CHECK: window.TELL_CHECK, esc: esc, prov: prov, fx: fx,
    signals: signals, tick: tick, last: last, LEVEL_LABEL: LEVEL_LABEL, ZONE_LABEL: ZONE_LABEL,
    STATUS_LABEL: STATUS_LABEL, STAGE: STAGE, STAGES: STAGES, stageStates: stageStates, ledgerLabel: ledgerLabel, themeToggle: themeToggle,
    reduced: window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches };
})();
