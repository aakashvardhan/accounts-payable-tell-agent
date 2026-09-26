/* Tell demo UI. Renders strictly from /api/traces (contract tell.demo_trace/1.0); no data hardcoded here. */
(function () {
  'use strict';
  var T = window.TellCore, bundle = null, player = null, view = { name: 'dash' };
  var app = document.getElementById('app');

  function h(tag, attrs) {
    var el = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (k === 'class') el.className = v; else if (k === 'text') el.textContent = v;
      else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) el.setAttribute(k, v === true ? '' : v);
    });
    for (var i = 2; i < arguments.length; i++) add(el, arguments[i]);
    return el;
  }
  function add(el, c) {
    if (c == null || c === false) return;
    if (Array.isArray(c)) c.forEach(function (x) { add(el, x); });
    else el.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function s(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    for (var i = 2; i < arguments.length; i++) if (arguments[i]) el.appendChild(arguments[i]);
    return el;
  }
  function pv(p) { return p ? h('span', { class: 'pv ' + p, text: p, title: p === 'real' ? 'Captured epoch-0 replay data' : 'Deterministic simulated demo fixture' }) : null; }
  function chip(st) { return h('span', { class: 'chip ' + st, text: st }); }
  function zname(z) { return { agent_1: 'AGENT 1', tell_verify: 'TELL-VERIFY', agent_s: 'AGENT S' }[z] || z; }
  function fmt(v) { return v == null ? '—' : (typeof v === 'object' ? JSON.stringify(v) : String(v)); }

  /* ---------- chrome ---------- */
  function renderChrome(traces) {
    var pr = { real: 0, fixture: 0 };
    (traces || []).forEach(function (t) { var p = T.provenanceOf(t); pr.real += p.real; pr.fixture += p.fixture; });
    var modes = document.getElementById('modes'); modes.textContent = '';
    if (pr.real) modes.appendChild(h('span', { class: 'mode real', text: T.MODE.REAL }));
    if (pr.fixture) modes.appendChild(h('span', { class: 'mode fixture', text: T.MODE.FIXTURE }));
    modes.appendChild(h('span', { class: 'mode live', text: T.MODE.LIVE + ' — DISABLED', title: 'Live inference is not available in this version' }));
    document.getElementById('banner').textContent = T.FIXTURE_LABEL;
    var esc = bundle ? T.escalations(bundle.traces).length : 0;
    var nav = document.getElementById('nav'); nav.textContent = '';
    [['dash', 'Operations', '#/'], ['review', 'Human review', '#/review']].forEach(function (n) {
      var b = h('button', { 'aria-current': (view.name === n[0] || (view.name === 'run' && n[0] === 'dash')) ? 'page' : false,
        onclick: function () { location.hash = n[2]; } }, n[1], n[0] === 'review' ? h('span', { class: 'count', text: esc }) : null);
      nav.appendChild(b);
    });
    document.getElementById('foot').textContent =
      'Replay data: TRAINING-SPLIT EPOCH-0 PREVIEW (not held-out performance). Adapter: ' + (bundle ? bundle.traces[0].hashes.adapter_label : '') +
      '. All payments are local SQLite simulations; no external service is contacted. Served on the local network only.';
  }

  /* ---------- dashboard ---------- */
  function renderDash() {
    var sum = T.summarize(bundle.traces, bundle.fleet.baseline);
    var kp = [['Invoices processed', sum.processed, '', null], ['Currently processing', sum.processing, '', null],
      ['Awaiting follow-up', sum.follow_up, 'warn', null], ['Escalated for review', sum.escalated, 'bad', '#/review'],
      ['Payments completed', sum.paid, 'ok', null], ['Payments prevented / blocked', sum.blocked + sum.escalated, 'bad', null]];
    var kpis = h('div', { class: 'grid kpis' }, kp.map(function (k) {
      var inner = [h('div', { class: 'n', text: k[1].toLocaleString('en-US') }), h('div', { class: 'l' }, k[0], k[0].indexOf('processed') > -1 || k[0].indexOf('completed') > -1 ? pv('fixture') : null)];
      return k[3] ? h('button', { class: 'kpi ' + k[2], onclick: function () { location.hash = k[3]; }, title: 'Open the human-review queue' }, inner)
        : h('div', { class: 'kpi ' + k[2] }, inner);
    }));
    function storyBtn(sc, tag, blurb) {
      var t = T.findTrace(bundle.traces, sc);
      return h('button', { class: 'story', onclick: function () { location.hash = '#/run/' + t.run_id; } },
        h('span', { class: 'tag', text: tag }), h('h3', { text: t.title }), h('p', { text: blurb }), h('div', { class: 'go', text: 'Open guided walkthrough →' }));
    }
    var stories = h('div', { class: 'story-row' },
      storyBtn('clean_payment', 'STORY 1 · CLEAN PAYMENT', 'Tell stays low, the validator approves, the gate allows, and the simulated ledger posts the payment.'),
      storyBtn('escalated_attack', 'STORY 2 · ESCALATED ATTACK', 'Tell rises, routing switches to Agent S, trusted checks run, the gate prevents payment and evidence goes to a human.'));
    var agents = h('div', { class: 'agents' }, bundle.fleet.agents.map(function (a) {
      return h('div', { class: 'agent' }, h('span', { class: 'dot' }), h('div', {}, h('b', { text: a.name }), h('small', { text: a.role + ' · ' + a.detail })), h('span', { class: 'st', text: a.state }));
    }));
    var rows = bundle.traces.map(function (t) {
      var p = T.provenanceOf(t); var last = t.events.filter(function (e) { return e.tell; })[0];
      var open = function () { location.hash = '#/run/' + t.run_id; };
      return h('tr', { class: 'click', tabindex: 0, 'data-run': t.run_id, onclick: open, onkeydown: function (e) { if (e.key === 'Enter') open(); } },
        h('td', {}, h('b', { text: t.invoice.invoice_number.v }), h('div', { class: 'muted small', text: t.title })),
        h('td', { text: t.invoice.vendor_name.v }),
        h('td', { text: T.money(t.invoice.amount_minor_units.v, t.invoice.currency.v) }),
        h('td', { class: 'mono', text: last ? last.tell.score.toFixed(3) : '—' }),
        h('td', { class: 'small', text: last ? zname(last.tell.zone) : '' }),
        h('td', {}, chip(t.queue_status)));
    });
    var queue = h('div', { class: 'card' }, h('h2', {}, 'Invoice-processing queue', pv('fixture')),
      h('div', { class: 'muted small', text: 'Queue state is a demo fixture over real captured epoch-0 records. Select any row for its execution view.' }),
      h('table', {}, h('thead', {}, h('tr', {}, ['Invoice', 'Supplier', 'Amount', 'Tell score', 'Zone', 'Status'].map(function (x) { return h('th', { text: x }); }))), h('tbody', {}, rows)));
    app.textContent = '';
    app.appendChild(h('div', {}, h('div', { class: 'title-row' }, h('h1', { text: 'Operations dashboard' }),
      h('span', { class: 'muted', text: 'Multiple invoice-processing agents online · replay mode' })),
      kpis, h('div', { class: 'title-row' }), stories,
      h('div', { class: 'grid two' }, queue, h('div', { class: 'card' }, h('h2', {}, 'Agent status', pv('fixture')), agents))));
  }

  /* ---------- Tell meter ---------- */
  function meterSvg(t, st) {
    var lo = t.thresholds.lower.v, hi = t.thresholds.upper.v, W = 900, X0 = 20, X1 = 880, y = 40, H = 34;
    function x(v) { return X0 + (X1 - X0) * v; }
    var segs = T.zoneSegments(lo, hi), col = { agent_1: '#2f8a63', tell_verify: '#b98a2c', agent_s: '#b83a40' };
    var g = s('svg', { class: 'meter', viewBox: '0 0 ' + W + ' 150', role: 'img', 'aria-label': 'Tell score meter' });
    segs.forEach(function (z) {
      g.appendChild(s('rect', { x: x(z.from), y: y, width: x(z.to) - x(z.from), height: H, fill: col[z.zone], opacity: st.tell && st.tell.zone === z.zone ? 1 : .45, rx: 4 }));
      var lab = s('text', { x: (x(z.from) + x(z.to)) / 2, y: y + H / 2 + 5, 'text-anchor': 'middle', fill: '#fff', 'font-size': 14, 'font-weight': 700 });
      lab.textContent = z.label.toUpperCase(); if (x(z.to) - x(z.from) > 90) g.appendChild(lab);
    });
    [[lo, 'lower ' + lo.toFixed(4)], [hi, 'upper ' + hi.toFixed(4)]].forEach(function (th, i) {
      g.appendChild(s('line', { x1: x(th[0]), x2: x(th[0]), y1: y - 12, y2: y + H + 12, stroke: '#e8edf7', 'stroke-width': 2, 'stroke-dasharray': '4 3' }));
      var tx = s('text', { x: x(th[0]), y: y + H + 32, 'text-anchor': 'middle', fill: '#93a1bd', 'font-size': 14 }); tx.textContent = th[1]; g.appendChild(tx);
    });
    ['0', '1'].forEach(function (l, i) { var tx = s('text', { x: i ? X1 : X0, y: y + H + 32, 'text-anchor': i ? 'end' : 'start', fill: '#63708c', 'font-size': 13 }); tx.textContent = l; g.appendChild(tx); });
    if (st.tell) {
      var nd = s('g', { class: 'needle', transform: 'translate(' + x(st.tell.score) + ',0)' },
        s('path', { d: 'M0 ' + (y - 4) + ' l-10 -20 h20 z', fill: '#fff' }), s('line', { x1: 0, x2: 0, y1: y - 4, y2: y + H, stroke: '#fff', 'stroke-width': 3 }));
      g.appendChild(nd);
    }
    return g;
  }

  /* ---------- process view ---------- */
  function renderRun(id) {
    var t = T.findTrace(bundle.traces, id);
    if (!t) { app.textContent = ''; app.appendChild(h('p', { text: 'Unknown process.' })); return; }
    if (player) player.dispose();
    var host = h('div', {}); app.textContent = ''; app.appendChild(host);
    player = T.createPlayer(t, { intervalMs: 1700, onChange: function (st) { paint(host, t, st); } });
    paint(host, t, player.state());
  }
  function cmpRow(label, a, b, ap, bp) {
    var same = (b === undefined || a === '\u2014') ? null : String(a) === String(b);
    return h('tr', {}, h('td', { class: 'k', text: label }), h('td', {}, fmt(a), pv(ap)),
      h('td', {}, b === undefined ? h('span', { class: 'muted', text: 'n/a' }) : [fmt(b), pv(bp)]),
      h('td', {}, same === null ? '' : h('span', { class: same ? 'match' : 'mism', text: same ? '✓ match' : '✗ differs' })));
  }
  function paint(host, t, st) {
    var inv = t.invoice, tv = t.trusted_vendor_record, ev = st.log[st.log.length - 1];
    var prog = h('span', { class: 'prog', text: st.started ? 'Step ' + (st.step + 1) + ' / ' + st.total : 'Ready — ' + st.total + ' steps' });
    var ctl = h('div', { class: 'ctl' },
      h('button', { class: 'btn back', onclick: function () { location.hash = '#/'; } }, '← Back to dashboard'),
      h('button', { class: 'btn primary', id: 'b-start', disabled: st.started, onclick: function () { player.start(); } }, 'Start'),
      h('button', { class: 'btn', id: 'b-next', disabled: !st.started || st.done, onclick: function () { player.next(); } }, 'Next'),
      h('button', { class: 'btn', id: 'b-play', disabled: st.playing || st.done, onclick: function () { player.play(); } }, 'Play'),
      h('button', { class: 'btn', id: 'b-pause', disabled: !st.playing, onclick: function () { player.pause(); } }, 'Pause'),
      h('button', { class: 'btn', id: 'b-reset', disabled: !st.started, onclick: function () { player.reset(); } }, 'Reset'), prog);
    var prov = T.provenanceOf(t);

    var left = h('div', { class: 'grid' },
      h('div', { class: 'card' }, h('h2', { text: 'Invoice & supplier message' }),
        h('dl', { class: 'kv' },
          h('dt', { text: 'Invoice #' }), h('dd', {}, fmt(inv.invoice_number.v), pv(inv.invoice_number.p)),
          h('dt', { text: 'Supplier' }), h('dd', {}, inv.vendor_name.v, pv(inv.vendor_name.p)),
          h('dt', { text: 'Beneficiary' }), h('dd', { class: 'mono' }, inv.beneficiary_account_id.v, pv(inv.beneficiary_account_id.p)),
          h('dt', { text: 'Amount' }), h('dd', {}, T.money(inv.amount_minor_units.v, inv.currency.v), pv(inv.amount_minor_units.p)),
          h('dt', { text: 'Message' }), h('dd', {}, inv.supplier_message_summary.v, pv(inv.supplier_message_summary.p)),
          h('dt', { text: 'Surface' }), h('dd', {}, fmt(inv.attack_surface.v), pv(inv.attack_surface.p)))),
      h('div', { class: 'card' }, h('h2', { text: 'Trusted vendor-record comparison' }),
        h('table', { class: 'cmp' }, h('thead', {}, h('tr', {}, ['Field', 'Invoice / message', 'Trusted record', ''].map(function (x) { return h('th', { text: x }); }))),
          h('tbody', {}, cmpRow('Beneficiary', inv.beneficiary_account_id.v, tv.approved_account_id.v, inv.beneficiary_account_id.p, tv.approved_account_id.p),
            cmpRow('Supplier', inv.vendor_name.v, tv.name.v, inv.vendor_name.p, tv.name.p),
            cmpRow('Vendor state', '—', tv.verification_state.v, null, tv.verification_state.p))),
        h('div', { class: 'small muted', text: 'Approved contact on file: ' + fmt(tv.approved_contact_on_file.v) })),
      h('div', { class: 'card' }, h('h2', { text: 'Retrieved memory & provenance' }),
        t.memory.retrieved.v ? null : h('div', { class: 'obs-only' }, 'No memory retrieved in this run. Observations in context: ' + t.memory.observations_in_context.v.join(', '), pv('real')),
        t.memory.entries.v.length ? h('div', {}, t.memory.entries.v.map(function (m) {
          return h('div', { class: 'note' }, h('b', { text: m.id }), pv('fixture'), h('div', { text: m.summary }),
            h('div', { class: 'small', text: 'source: ' + m.source + ' · trust: ' + m.trust + ' · status: ' + m.status }));
        })) : null));

    var zone = st.tell ? st.tell.zone : null;
    var meter = h('div', { class: 'card meter-card' }, h('h2', { text: 'Tell meter' }),
      h('div', { class: 'meter-num' },
        h('span', { class: 'score ' + (zone ? 'z-' + zone : ''), id: 'score', text: st.tell ? st.tell.score.toFixed(3) : '—' }),
        h('span', { class: 'zone ' + (zone ? 'z-' + zone : 'muted'), text: zone ? zname(zone) + ' ZONE' : 'Awaiting Tell reading' }), st.tell ? pv(st.prov.tell) : null),
      meterSvg(t, st),
      h('div', { class: 'lanes' },
        h('div', { class: 'lane a1' + (st.activeAgent === 'agent_1' ? ' active' : ''), id: 'lane-a1' }, h('b', { text: 'Agent 1' }), h('small', { text: st.activeAgent === 'agent_1' ? 'Processing invoice' : (st.activeAgent === 'agent_s' ? 'Frozen — handed off' : 'Standing by') })),
        h('div', { class: 'arrow' + (st.activeAgent === 'agent_s' ? ' on' : ''), text: '→' }),
        h('div', { class: 'lane as' + (st.activeAgent === 'agent_s' ? ' active' : ''), id: 'lane-as' }, h('b', { text: 'Agent S' }), h('small', { text: st.activeAgent === 'agent_s' ? 'Investigating with trusted tools' : 'Not engaged' }))),
      st.routing ? h('div', { class: 'note', id: 'route-note' }, 'Routing: ', h('b', { text: st.routing.from === st.routing.to ? 'stays with ' + zname(st.routing.to) : zname(st.routing.from) + ' → ' + zname(st.routing.to) }), ' — ' + st.routing.reason, pv(st.prov.routing)) : null,
      st.alarm ? h('div', { class: 'small' }, 'Alarm: ', h('span', { class: 'alarm ' + st.alarm.after, text: st.alarm.before + ' → ' + st.alarm.after }), pv(st.prov.alarm)) : null,
      h('div', { class: 'small muted', text: t.thresholds.note }));

    var tl = h('ol', { class: 'tl', id: 'timeline' }, st.log.map(function (e, i) {
      var args = e.action ? e.action.name + (Object.keys(e.action.args || {}).length ? ' ' + JSON.stringify(e.action.args) : '') : null;
      return h('li', { class: i === st.log.length - 1 ? 'cur' : '', 'data-seq': e.seq }, h('time', { text: 'T+' + e.t_offset_s.toFixed(1) }),
        h('div', {}, h('span', { class: 'who', text: e.actor.replace('_', ' ') }), h('b', { text: e.title }),
          e.summary ? h('div', { class: 'small muted', text: e.summary }) : null, args ? h('div', { class: 'args', text: args }) : null,
          e.lookups ? h('div', { class: 'args', text: JSON.stringify(e.lookups) }) : null,
          Object.keys(e.prov).map(function (k) { return h('span', { class: 'pv ' + e.prov[k], text: k + ':' + e.prov[k] }); })));
    }));
    if (!st.log.length) tl = h('div', { class: 'empty', text: 'Press Start to begin the replay.' });
    var tlcol = h('div', { class: 'card tlcol' }, h('h2', { text: 'Action timeline (structured audit events)' }), tl);

    var v = st.validator, g = st.gate;
    var verdicts = h('div', { class: 'grid verdicts' },
      h('div', { class: 'card verdict' }, h('h2', { text: 'Validator' }),
        v ? h('div', { class: 'big ' + (v.outcome === 'APPROVED' ? 'ok' : 'warn'), text: v.outcome }) : h('div', { class: 'big idle', text: 'pending' }),
        v ? h('div', { class: 'small muted' }, 'Captured outcome: ' + fmt(v.captured_outcome), pv(st.prov.validator)) : null),
      h('div', { class: 'card verdict', id: 'gate' }, h('h2', { text: 'Deterministic gate' }),
        g ? h('div', { class: 'big ' + (g.decision === 'ALLOW' ? 'ok' : 'bad'), text: g.decision }) : h('div', { class: 'big idle', text: 'pending' }),
        g ? h('div', { class: 'small muted' }, g.reason, pv(st.prov.gate)) : null),
      h('div', { class: 'card verdict', id: 'ledger' }, h('h2', {}, 'Simulated ledger', pv(st.ledger ? st.prov.ledger : null)),
        st.ledger ? h('table', { class: 'led' }, h('thead', {}, h('tr', {}, ['', 'Before', 'After'].map(function (x) { return h('th', { text: x }); }))),
          h('tbody', {}, T.ledgerRows(st.ledger).map(function (r) {
            var isMoney = r.key.indexOf('balance') > -1, f = function (x) { return isMoney ? T.money(x, inv.currency.v) : fmt(x); };
            return h('tr', {}, h('td', { class: 'muted', text: r.key.replace(/_minor|_/g, ' ').trim() }), h('td', { text: f(r.before) }), h('td', { class: r.changed ? 'chg' : '', text: f(r.after) }));
          }))) : h('div', { class: 'big idle', text: 'not posted' }),
        st.ledger && st.ledger.entry ? h('div', { class: 'small hash', text: 'intent ' + st.ledger.entry.payment_intent_id + ' · ' + st.ledger.entry.rail }) : null),
      h('div', { class: 'card verdict' }, h('h2', { text: 'Hashes' }),
        h('div', { class: 'small hash' }, 'probe ' + t.hashes.probe_weights_sha256.slice(0, 16) + '…'), h('div', { class: 'small hash' }, 'adapter ' + t.hashes.adapter_weights_sha256.slice(0, 16) + '…'),
        h('div', { class: 'small muted', text: t.hashes.model_repo_id + ' @ ' + t.hashes.model_revision.slice(0, 8) }), pv('real'),
        h('div', { class: 'small muted', text: 'Side effects simulated: ' + (t.side_effects_simulated ? 'yes' : 'no') })));

    var out = h('div', { class: 'outcome', id: 'outcome' }, h('h2', { text: 'Terminal outcome' }),
      st.outcome ? [chip(st.outcome), h('span', { class: 'big', text: { PAID: 'Payment completed (simulated)', ESCALATED: 'Payment prevented — escalated to human review', BLOCKED: 'Payment blocked', FOLLOW_UP: 'Awaiting supplier follow-up' }[st.outcome] }), pv(st.prov.outcome)]
        : h('span', { class: 'muted', text: st.done ? 'Replay ends at last captured event (process still in progress).' : 'In progress…' }),
      st.outcome === 'ESCALATED' ? h('button', { class: 'btn', onclick: function () { location.hash = '#/review/' + t.run_id; } }, 'Open in human review →') : null);

    host.textContent = '';
    host.appendChild(h('div', {}, h('div', { class: 'title-row' }, h('h1', { text: t.title }), chip(st.status),
      prov.modes.map(function (m) { return h('span', { class: 'mode ' + (m === T.MODE.REAL ? 'real' : 'fixture'), text: m }); })), ctl,
      h('div', { class: 'grid proc' }, left, meter, tlcol), verdicts, out));
  }

  /* ---------- human review ---------- */
  function caseCard(t) {
    var r = t.review, ev = function (type) { return t.events.filter(function (e) { return e.type === type; }); };
    var tell = ev('tell_score')[0], look = ev('trusted_lookup')[0], val = ev('validator')[0], gate = ev('gate')[0];
    var acts = t.events.filter(function (e) { return e.action; });
    var al = t.events.filter(function (e) { return e.alarm; })[0];
    var row = function (k, v) { return [h('dt', { text: k }), h('dd', {}, v)]; };
    var hs = r.artifact_hashes;
    return h('div', { class: 'card', id: 'case-' + t.run_id },
      h('div', { class: 'title-row' }, h('h3', { text: 'Invoice ' + fmt(t.invoice.invoice_number.v) + ' · ' + t.invoice.vendor_name.v }), chip('ESCALATED'),
        h('span', { class: 'muted small', text: T.money(t.invoice.amount_minor_units.v, t.invoice.currency.v) })),
      h('div', { class: 'case' },
        h('dl', { class: 'kv' },
          row('Triggering evidence', [r.triggering_evidence.v, pv(r.triggering_evidence.p)]),
          row('Attack surface', [fmt(r.attack_surface.v) + (t.invoice.attack_family.v ? ' / ' + t.invoice.attack_family.v : ''), pv(r.attack_surface.p)]),
          row('Tell score & routing', [tell.tell.score.toFixed(3) + ' → ' + zname(tell.tell.agent === 'agent_s' ? 'agent_s' : tell.tell.zone) + ' (upper threshold ' + tell.tell.upper.toFixed(4) + ')', pv(tell.prov.tell)]),
          row('Actions attempted', acts.map(function (e) { return h('div', { class: 'small' }, e.actor + ': ' + e.action.name, pv(e.prov.action)); })),
          row('Trusted lookups', look ? [JSON.stringify(look.lookups), pv(look.prov.lookups)] : h('span', { class: 'empty', text: 'none captured' })),
          row('Validator', val ? [val.validator.outcome, pv(val.prov.validator), val.summary ? h('div', { class: 'small muted', text: val.summary }) : null] : '—'),
          row('Gate decision', gate ? [h('b', { text: gate.gate.decision }), ' — ' + gate.gate.reason, pv(gate.prov.gate)] : '—'),
          row('Alarm state', al ? [h('span', { class: 'alarm ' + al.alarm.after, text: al.alarm.before + ' → ' + al.alarm.after }), pv(al.prov.alarm),
            h('div', { class: 'small muted', text: 'Only a human reviewer can clear this alarm; this console has no clear or approve control.' })] : '—'),
          row('Recommended human action', [r.recommended_action.v, pv(r.recommended_action.p)])),
        h('div', {}, h('h2', {}, 'Captured evidence report', pv(r.evidence_report.p)), h('pre', { class: 'report', text: JSON.stringify(r.evidence_report.v, null, 2) }),
          h('h2', {}, 'Immutable audit identifiers & artifact hashes'),
          h('div', { class: 'small hash', text: 'audit ids: ' + r.audit_ids.v.join(' ') }), pv(r.audit_ids.p),
          Object.keys(hs).map(function (k) { return h('div', { class: 'small hash' }, k + ': ' + hs[k].v, pv(hs[k].p)); }))),
      h('div', { class: 'cta' }, h('button', { class: 'btn', onclick: function () { location.hash = '#/run/' + t.run_id; } }, 'View execution replay →')));
  }
  function renderReview(focus) {
    var list = T.escalations(bundle.traces);
    if (focus) list = list.slice().sort(function (a, b) { return (b.run_id === focus) - (a.run_id === focus); });
    app.textContent = '';
    app.appendChild(h('div', {}, h('div', { class: 'title-row' }, h('h1', { text: 'Human-review workspace' }), h('span', { class: 'muted', text: list.length + ' escalated · observation and review only' })),
      h('div', { class: 'note', text: 'Agent S cannot approve a payment or clear its own alarm. Decisions are made outside this console.' }),
      h('div', { class: 'review-list' }, list.map(caseCard))));
    if (focus) { var el = document.getElementById('case-' + focus); if (el && el.scrollIntoView) el.scrollIntoView(); }
  }

  /* ---------- router ---------- */
  function route() {
    if (!bundle) return;
    if (player) { player.dispose(); player = null; }
    var m = location.hash.match(/^#\/(run|review)(?:\/([^@]+))?(?:@(-?\d+))?$/);
    view = m ? { name: m[1], id: m[2], step: m[3] } : { name: 'dash' };
    renderChrome(view.name === 'run' ? [T.findTrace(bundle.traces, view.id)].filter(Boolean) : bundle.traces);
    if (view.name === 'run') { renderRun(view.id); if (view.step != null && player) player.seek(parseInt(view.step, 10)); } else if (view.name === 'review') renderReview(view.id); else renderDash();
    window.scrollTo(0, 0);
  }
  fetch('/api/traces').then(function (r) { return r.json(); }).then(function (b) {
    bundle = b; window.addEventListener('hashchange', route); route();
  }).catch(function (e) { app.textContent = 'Could not load trace contract: ' + e; });
})();
