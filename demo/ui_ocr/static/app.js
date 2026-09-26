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
    if (isLive() && view.name !== 'run' && view.name !== 'review') return renderLiveChrome();
    var pr = { real: 0, fixture: 0 };
    (traces || []).forEach(function (t) { var p = T.provenanceOf(t); pr.real += p.real; pr.fixture += p.fixture; });
    var modes = document.getElementById('modes'); modes.textContent = '';
    if (view.name === 'job') { /* no replay/fixture badges here */ } else if (pr.real) modes.appendChild(h('span', { class: 'mode real', text: T.MODE.REAL }));
    if (view.name !== 'job' && pr.fixture) modes.appendChild(h('span', { class: 'mode fixture', text: T.MODE.FIXTURE }));
    if (isLive()) modes.insertBefore(h('span', { class: 'mode recorded', text: 'RECORDED EXAMPLE \u2014 NOT A LIVE RUN' }), modes.firstChild);
    else modes.appendChild(h('span', { class: 'mode live', text: T.MODE.LIVE + ' \u2014 DISABLED', title: 'Live inference is not available in this version' }));
    modes.appendChild(themeButton());
    var banner = document.getElementById('banner');
    if (view.name === 'job') {   // uploaded-PDF data is neither replay capture nor fixture
      modes.insertBefore(h('span', { class: 'mode pdf', text: 'UPLOADED PDF \u2014 LOCAL CPU EXTRACTION' }), modes.firstChild);
      banner.textContent = ''; banner.hidden = true;
    } else { banner.hidden = false; banner.textContent = T.FIXTURE_LABEL; }
    var esc = bundle ? T.escalations(bundle.traces).length : 0;
    if (isLive()) { liveNav(); document.getElementById('foot').textContent = 'RECORDED EXAMPLES: historical replay fixtures (training-split epoch-0 preview), not produced by an uploaded invoice. Uploaded invoices always run Live.'; return; }
    var nav = document.getElementById('nav'); nav.textContent = '';
    [['dash', 'Operations', '#/'], ['review', 'Human review', '#/review']].forEach(function (n) {
      var b = h('button', { 'aria-current': (view.name === n[0] || (view.name === 'run' && n[0] === 'dash')) ? 'page' : false,
        onclick: function () { location.hash = n[2]; } }, n[1], n[0] === 'review' ? h('span', { class: 'count', text: esc }) : null);
      nav.appendChild(b);
    });
    if (intake.status && intake.status.auth) {
      nav.appendChild(h('button', { id: 'btn-signout', class: 'signout', onclick: function () {
        api('POST', '/auth/logout', {}).then(function () { location.href = '/login'; }, function () { location.href = '/login'; }); } }, 'Sign out'));
    }
    document.getElementById('foot').textContent =
      'Replay data: TRAINING-SPLIT EPOCH-0 PREVIEW (not held-out performance). Adapter: ' + (bundle ? bundle.traces[0].hashes.adapter_label : '') +
      '. All payments are local SQLite simulations; no external service is contacted. Served on the local network only.';
  }

  /* ---------- intake state / API ---------- */
  var intake = { status: null, jobs: [] }, scanState = { phase: 'idle', result: null }, pollTimer = null, jobTimer = null;
  function api(method, path, body, headers, raw) {
    var h2 = Object.assign({ 'X-Tell-Intake': '1' }, headers || {});
    if (body !== undefined && !raw) { h2['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
    return fetch(path, { method: method, headers: h2, body: body, credentials: 'same-origin' }).then(function (r) {
      if (r.status === 401) { location.href = '/login'; }   // session missing or expired
      return r.json().then(function (j) { j._status = r.status; return j; });
    });
  }
  function fmtShort(iso) {      // '25 Sep, 19:56' (year added when it is not the current year)
    if (!iso) return '\u2014'; var d = new Date(iso); if (isNaN(d)) return iso;
    var o = { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false };
    if (d.getFullYear() !== new Date().getFullYear()) o.year = 'numeric';
    return d.toLocaleString('en-GB', o);
  }
  function fmtTime(iso) { if (!iso) return '—'; var d = new Date(iso); return isNaN(d) ? iso : d.toLocaleString(); }
  function clock(iso) { var d = new Date(iso); return isNaN(d) ? String(iso) : d.toLocaleTimeString([], { hour12: false }); }
  function fmtSize(n) { return n == null ? '—' : (n >= 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB'); }
  function stopPolling() { if (pollTimer) clearInterval(pollTimer); if (jobTimer) clearInterval(jobTimer); pollTimer = jobTimer = null; }
  function refreshIntake() {
    return Promise.all([api('GET', '/api/intake/status'), api('GET', '/api/intake/jobs').catch(function () { return { jobs: [] }; })]).then(function (r) {
      var sig = JSON.stringify([r[0], r[1].jobs, scanState.phase]);
      intake.status = r[0]; intake.jobs = (r[0].enabled && r[1].jobs) || [];
      if ((view.name === 'dash' || view.name === 'attention') && sig !== intake.sig) {
        intake.sig = sig;
        if (view.name === 'attention' && isLive()) renderAttention(); else renderDash();
        if (isLive()) liveNav();
      }
      if (!intake.jobs.some(T.jobActive) && pollTimer && scanState.phase !== 'running') { clearInterval(pollTimer); pollTimer = setInterval(refreshIntake, 5000); }
    });
  }
  function startFastPoll() { if (pollTimer) clearInterval(pollTimer); pollTimer = setInterval(refreshIntake, 800); }
  /* TellSecured toggle: chosen per upload/scan and stored on each job by the backend. localStorage only remembers the switch position. */
  var tellSecured = (function () { try { return localStorage.getItem('tellSecured') !== 'off'; } catch (e) { return true; } })();
  function setTellSecured(on) { tellSecured = !!on; try { localStorage.setItem('tellSecured', on ? 'on' : 'off'); } catch (e) { /* storage unavailable */ } renderDash(); }
  function checkNew() {
    if (scanState.phase === 'running') return;
    scanState = { phase: 'running', result: null }; renderDash(); startFastPoll();
    api('POST', '/api/intake/scan', { tell_secured: tellSecured }).then(function (s) {
      var poll = function () { api('GET', '/api/intake/scan/' + s.scan_id).then(function (r) {
        if (r.state === 'running') { setTimeout(poll, 400); return; }
        scanState = { phase: 'done', result: r }; refreshIntake();
      }); };
      poll();
    }).catch(function (e) { scanState = { phase: 'done', result: { message: 'Scan failed: ' + e, error: true } }; renderDash(); });
  }
  function addInvoices(files) {
    files = Array.prototype.slice.call(files || []); if (!files.length) return;
    var manifest = files.map(function (f) { return { name: f.name, size: f.size, type: f.type || '' }; });
    api('POST', '/api/intake/batch', { files: manifest, tell_secured: tellSecured }).then(function (r) {
      if (!r.jobs) { scanState = { phase: 'done', result: { message: r.error || 'Batch rejected', error: true } }; renderDash(); return; }
      intake.jobs = r.jobs.concat(intake.jobs); renderDash(); startFastPoll();
      var queue = r.jobs.map(function (j, i) { return { job: j, file: files[i] }; }).filter(function (x) { return x.job.status === 'UPLOADING'; });
      var run = function () { var x = queue.shift(); if (!x) return; 
        api('PUT', '/api/intake/jobs/' + x.job.job_id + '/content', x.file, { 'Content-Type': x.file.type || 'application/octet-stream' }, true).then(run, run); };
      run(); run(); run();
    });
  }
  var fileInputSingleton = null;   // one persistent input so a re-render can never drop an open file dialog
  function fileInputEl() {
    if (!fileInputSingleton) fileInputSingleton = h('input', { type: 'file', id: 'file-input', accept: 'application/pdf,.pdf', multiple: true, class: 'sr-only', 'aria-label': 'Choose PDF invoices',
      onchange: function (e) { addInvoices(e.target.files); e.target.value = ''; } });
    return fileInputSingleton;
  }
  function resetDemo() { api('POST', '/api/intake/reset', {}).then(function () { scanState = { phase: 'idle', result: null }; refreshIntake(); }); }

  function buildActions() {
    var en = intake.status && intake.status.enabled;
    var actions = null, ic = null;
    if (en) {
      var fileInput = fileInputEl();
      var last = intake.status.last_scan, scanning = scanState.phase === 'running', res = scanState.result;
      var scanMsg;
      if (scanning) scanMsg = h('div', { class: 'scan running', id: 'scan-msg' }, h('span', { class: 'spinner' }), 'Scanning demo inbox…');
      else if (res && res.error && !res.scan_id) scanMsg = h('div', { class: 'scan bad', id: 'scan-msg', text: res.message });
      else if (res && res.scan_id) scanMsg = h('div', { class: 'scan done', id: 'scan-msg' },
        h('b', { text: res.discovered === 0 || (res.imported === 0 && res.rejected === 0) ? 'No new invoices found' : res.imported + ' invoice' + (res.imported === 1 ? '' : 's') + ' imported' }),
        ' · discovered ' + res.discovered + ' · imported ' + res.imported + ' · duplicates skipped ' + res.duplicates + ' · rejected ' + res.rejected + ' · completed ' + fmtTime(res.completed_at));
      else scanMsg = null;
      actions = h('div', { class: 'card actions' },
        h('div', { class: 'act-row' },
          h('button', { class: 'btn primary big', id: 'btn-check', disabled: scanning, onclick: checkNew }, scanning ? 'Checking…' : 'Check for new invoices'),
          h('button', { class: 'btn big', id: 'btn-add', onclick: function () { fileInput.click(); } }, 'Add invoices'), fileInput,
          h('span', { class: 'muted small last', id: 'last-checked', text: 'Inbox last checked: ' + (last ? fmtTime(last.completed_at) : 'never') }),
          h('button', { class: 'btn ghost', id: 'btn-reset', onclick: resetDemo, title: 'Clear the intake queue and temporary uploads (inbox files are kept)' }, 'Reset demo queue')),
        isLive() ? tsToggle() : null,
        scanMsg,
        h('div', { class: 'small muted', text: 'PDF only · up to ' + intake.status.limits.max_files_per_batch + ' files per batch · ' + Math.round(intake.status.limits.max_bytes / 1048576) + ' MB each · processed locally, nothing leaves this machine' }));
      var c = T.intakeCounts(intake.jobs);
      ic = h('div', { class: 'grid intake-kpis', id: 'intake-kpis' }, [['Waiting', c.waiting, ''], ['Processing', c.processing, ''], ['Ready for agent processing', c.ready, 'ok'],
        ['Awaiting vendor', c.awaiting, 'warn'], ['Need attention', c.attention, 'warn'], ['Failed intake', c.failed, 'bad']].map(function (k) {
        return h('div', { class: 'kpi small ' + k[2], 'data-k': k[0] }, h('div', { class: 'n', text: k[1] }), h('div', { class: 'l', text: k[0] })); }));
    }
    return { actions: actions, ic: ic };
  }

  /* ---------- dashboard ---------- */
  function renderDash() {
    if (isLive()) return renderLiveDash();
    var sum = T.summarize(bundle.traces, bundle.fleet.baseline), en = intake.status && intake.status.enabled;
    var kp = [['Invoices processed', sum.processed, '', null], ['Currently processing', sum.processing, '', null],
      ['Awaiting follow-up', sum.follow_up, 'warn', null], ['Escalated for review', sum.escalated, 'bad', '#/review'],
      ['Payments completed', sum.paid, 'ok', null], ['Payments prevented / blocked', sum.blocked + sum.escalated, 'bad', null]];
    var kpis = h('div', { class: 'grid kpis' }, kp.map(function (k) {
      var inner = [h('div', { class: 'n', text: k[1].toLocaleString('en-US') }), h('div', { class: 'l' }, k[0], k[0].indexOf('processed') > -1 || k[0].indexOf('completed') > -1 ? pv('fixture') : null)];
      return k[3] ? h('button', { class: 'kpi ' + k[2], onclick: function () { location.hash = k[3]; }, title: 'Open the human-review queue' }, inner)
        : h('div', { class: 'kpi ' + k[2] }, inner);
    }));
    var ab = buildActions(), actions = ab.actions, ic = ab.ic;
    var jobRows = (en ? intake.jobs : []).map(function (j) {
      var open = function () { location.hash = '#/job/' + j.job_id; }, s = j.summary || {};
      var bar = T.isActive(j.status) ? h('div', { class: 'bar' }, (function () { var f = h('div', { class: 'fill' }); f.style.width = j.progress + '%'; return f; })()) : null;
      return h('tr', { class: 'click', tabindex: 0, 'data-job': j.job_id, onclick: open, onkeydown: function (e) { if (e.key === 'Enter') open(); } },
        h('td', {}, h('b', { text: s.invoice_number || j.display_name }), h('div', { class: 'muted small', text: s.invoice_number ? j.display_name : 'Uploaded ' + fmtTime(j.created_at) })),
        h('td', { text: s.supplier_name || '—' }),
        h('td', { text: s.amount ? (s.currency ? s.currency + ' ' : '') + s.amount : '—' }),
        h('td', { class: 'small', text: j.source === 'local_inbox' ? 'Demo inbox' : 'Manual upload' }),
        h('td', {}, h('span', { class: 'chip ' + j.status, text: T.INTAKE_LABEL[j.status] }), bar, h('div', { class: 'muted small latest', text: j.latest_event || '' })),
        h('td', { class: 'mono', text: '—' }));
    });
    var replayRows = bundle.traces.map(function (t) {
      var last = t.events.filter(function (e) { return e.tell; })[0];
      var open = function () { location.hash = '#/run/' + t.run_id; };
      return h('tr', { class: 'click', tabindex: 0, 'data-run': t.run_id, onclick: open, onkeydown: function (e) { if (e.key === 'Enter') open(); } },
        h('td', {}, h('b', { text: t.invoice.invoice_number.v })),
        h('td', { text: t.invoice.vendor_name.v }),
        h('td', { text: T.money(t.invoice.amount_minor_units.v, t.invoice.currency.v) }),
        h('td', { class: 'small' }, 'Replay', pv('fixture')),
        h('td', {}, chip(t.queue_status)),
        h('td', { class: 'mono', text: last ? last.tell.score.toFixed(3) : '—' }));
    });
    var queue = h('div', { class: 'card', id: 'queue' }, h('h2', {}, 'Invoice-processing queue'),
      h('table', {}, h('thead', {}, h('tr', {}, ['Invoice', 'Supplier', 'Amount', 'Source', 'Status', 'Tell score'].map(function (x) { return h('th', { text: x }); }))), h('tbody', {}, jobRows.concat(replayRows))));
    var agents = h('div', { class: 'agents' }, bundle.fleet.agents.map(function (a) {
      return h('div', { class: 'agent' }, h('span', { class: 'dot' }), h('div', {}, h('b', { text: a.name }), h('small', { text: a.role + ' · ' + a.detail })), h('span', { class: 'st', text: a.state }));
    }));
    app.textContent = '';
    app.appendChild(h('div', {}, h('div', { class: 'title-row' }, h('h1', { text: 'Operations dashboard' }),
      h('span', { class: 'muted', text: 'Multiple invoice-processing agents online' })),
      kpis, actions, ic,
      h('div', { class: 'grid two' }, queue, h('div', { class: 'card' }, h('h2', {}, 'Agent status', pv('fixture')), agents))));
  }

  /* ---------- uploaded-invoice detail ---------- */
  var FIELD_LABELS = [['document_type', 'Document type'], ['supplier_name', 'Supplier / vendor'], ['invoice_number', 'Invoice / reference no.'], ['invoice_date', 'Invoice date'],
    ['due_date', 'Due date'], ['amount', 'Amount'], ['currency', 'Currency'], ['beneficiary_name', 'Beneficiary name'], ['beneficiary_account', 'Beneficiary account']];
  var SRC_LABEL = { EMBEDDED_TEXT: 'embedded text', OCR: 'OCR' };
  function renderJob(id) {
    if (isLive()) return renderLiveJob(id);
    stopPolling();
    var host = h('div', {}); app.textContent = ''; app.appendChild(host);
    var load = function () { api('GET', '/api/intake/jobs/' + id).then(function (j) {
      if (j.error && !j.job_id) { host.textContent = ''; host.appendChild(h('p', { text: 'Unknown invoice job.' })); if (jobTimer) clearInterval(jobTimer); return; }
      paintJob(host, j); if (!T.isActive(j.status) && jobTimer) { clearInterval(jobTimer); jobTimer = null; }
    }); };
    load(); jobTimer = setInterval(load, 900);
  }
  function fieldRow(f, d) {
    if (!d || (!d.found && !d.ambiguous)) return h('tr', { 'data-field': f[0] }, h('td', { class: 'k', text: f[1] }), h('td', { class: 'empty', colspan: 2, text: 'not found in document' }));
    if (d.ambiguous) return h('tr', { 'data-field': f[0], class: 'ambig' }, h('td', { class: 'k', text: f[1] }),
      h('td', {}, h('b', { class: 'warn-text', text: 'ambiguous' }), h('div', { class: 'muted small', text: 'candidates: ' + (d.candidates || []).join(', ') }),
        d.evidence_text ? h('div', { class: 'muted small', text: d.evidence_text }) : null), h('td', {}, h('span', { class: 'pv amb', text: 'not resolved' })));
    var conf = d.confidence == null ? '' : ' · ' + Math.round(d.confidence * 100) + '% confidence';
    return h('tr', { 'data-field': f[0] }, h('td', { class: 'k', text: f[1] }),
      h('td', {}, h('b', { text: d.value }), d.normalization_applied ? h('div', { class: 'muted small', text: 'normalized: ' + d.normalization_applied }) : null,
        d.evidence_text ? h('div', { class: 'muted small', text: '“' + d.evidence_text + '”' }) : null),
      h('td', {}, h('span', { class: 'pv pdf', text: SRC_LABEL[d.source] || d.source }), h('div', { class: 'muted small', text: (d.page ? 'page ' + d.page : '') + conf })));
  }
  function paintJob(host, j) {
    var ex = j.extraction || {}, fields = ex.fields || {}, ocr = ex.ocr, dec = ex.decision, ob = j.outbox, vl = ex.vendor_lookup, mfr = ex.missing_field_result;
    var meta = h('dl', { class: 'kv' }, h('dt', { text: 'Filename' }), h('dd', { text: j.display_name }), h('dt', { text: 'Uploaded' }), h('dd', { text: fmtTime(j.created_at) }),
      h('dt', { text: 'SHA-256' }), h('dd', { class: 'hash', text: j.sha256 || '—' }), h('dt', { text: 'Intake source' }), h('dd', { text: j.source === 'local_inbox' ? 'Demo inbox (local folder)' : 'Manual upload' }),
      h('dt', { text: 'Size' }), h('dd', { text: fmtSize(j.size) }), h('dt', { text: 'Status' }), h('dd', {}, h('span', { class: 'chip ' + j.status, text: T.INTAKE_LABEL[j.status] })),
      j.duplicate_of ? [h('dt', { text: 'Duplicate of' }), h('dd', {}, h('a', { href: '#/job/' + j.duplicate_of, text: 'original invoice job' }))] : null);
    var tl = h('ol', { class: 'tl', id: 'extraction-timeline' }, j.events.map(function (e) {
      return h('li', { 'data-seq': e.seq, 'data-kind': e.kind }, h('time', { text: clock(e.at) }), h('div', {}, h('span', { class: 'who', text: e.kind.replace(/_/g, ' ') }),
        h('b', { text: e.message }), e.provenance === 'SIMULATED_AGENT_ACTION' ? h('span', { class: 'pv sim', text: 'SIMULATED_AGENT_ACTION' }) : null));
    }));
    var hold = j.payment ? h('div', { class: 'hold', id: 'payment-hold' }, h('b', { text: 'PAYMENT ON HOLD' }), ' — ' + j.payment.reason + '. Not eligible for payment processing.') : null;
    var nextCard = h('div', { class: 'card', id: 'next-step' }, h('h2', { text: 'Next step' }), h('div', { class: 'big-line', text: T.nextStep(j) }), hold,
      T.isActive(j.status) ? h('div', { class: 'bar' }, (function () { var f = h('div', { class: 'fill' }); f.style.width = j.progress + '%'; return f; })()) : null);
    var right = [];
    if (fields.document_type || ex.source) {
      right.push(h('div', { class: 'card' }, h('h2', { text: 'Extracted fields' }),
        h('table', { class: 'cmp fields', id: 'fields' }, h('thead', {}, h('tr', {}, ['Field', 'Value & evidence', 'Source'].map(function (x) { return h('th', { text: x }); }))),
          h('tbody', {}, FIELD_LABELS.map(function (f) { return fieldRow(f, fields[f[0]]); })))));
    }
    if (dec && dec.status_hint) {
      right.push(h('div', { class: 'card', id: 'decision' }, h('h2', { text: 'Required-field decision' }),
        h('div', { class: 'big-line', text: { READY: 'All required fields present', CLARIFY: 'Missing: ' + (dec.missing_required_fields || []).join(', '), REVIEW: 'Needs document review', UNREADABLE: 'Unreadable document' }[dec.status_hint] || dec.status_hint }),
        (dec.reasons || []).length ? h('ul', {}, dec.reasons.map(function (r) { return h('li', { text: r }); })) : null,
        mfr ? h('pre', { class: 'report', id: 'missing-field-result', text: JSON.stringify(mfr, null, 2) }) : null));
    }
    if (vl) {
      right.push(h('div', { class: 'card', id: 'vendor-lookup' }, h('h2', {}, 'Trusted vendor lookup', h('span', { class: 'pv fixture', text: 'demo vendor master' })),
        h('dl', { class: 'kv' }, h('dt', { text: 'Result' }), h('dd', {}, h('b', { text: vl.status })), h('dt', { text: 'Vendor' }), h('dd', { text: vl.vendor_name ? vl.vendor_name + ' (' + vl.vendor_id + ')' : '—' }),
          h('dt', { text: 'Detail' }), h('dd', { text: vl.reason }))));
    }
    if (ob) {
      right.push(h('div', { class: 'card email', id: 'email-preview' }, h('div', { class: 'email-head' }, h('h2', { text: 'Clarification email (simulated)' }),
        h('span', { class: 'sim-badge', id: 'sim-badge', text: 'SIMULATED — NOT SENT' })),
        h('dl', { class: 'kv' }, h('dt', { text: 'To' }), h('dd', { text: ob.recipient }), h('dt', { text: 'Recipient source' }), h('dd', { class: 'mono', text: ob.recipient_source }),
          h('dt', { text: 'Subject' }), h('dd', { text: ob.subject }), h('dt', { text: 'Requested' }), h('dd', { text: ob.requested_fields.join(', ') }),
          h('dt', { text: 'Send status' }), h('dd', { class: 'mono', text: ob.send_status }), h('dt', { text: 'Agent action' }), h('dd', {}, h('span', { class: 'pv sim', text: ob.provenance.action }), ' ', h('span', { class: 'mono', text: ob.action_trace_id })),
          h('dt', { text: 'Prepared' }), h('dd', { text: fmtTime(ob.created_at) })),
        h('pre', { class: 'report mail', id: 'email-body', text: ob.body }),
        h('div', { class: 'small muted', text: 'The recipient and template are owned by the application. No email was sent; no network connection was made.' })));
    }
    if (ocr) {
      var rows = (ocr.pages || []).map(function (p) { return h('tr', { 'data-page': p.page }, h('td', { text: p.page }), h('td', { text: p.outcome }), h('td', { text: p.words }), h('td', { text: p.usable_words }),
        h('td', { text: Math.round(p.mean_confidence * 100) + '%' }), h('td', { text: p.duration_s + 's' })); });
      right.push(h('div', { class: 'card', id: 'ocr-card' }, h('h2', { text: 'OCR' }),
        h('div', { class: 'small muted', text: (ocr.engine ? ocr.engine + ' ' + ocr.version + ' · ' : '') + (ocr.dpi ? ocr.dpi + ' DPI · ' : '') + 'page ' + (ocr.pages || []).length + ' of ' + (ocr.pages_processed || ocr.page_count || '?') +
          (ocr.duration_s != null ? ' · ' + ocr.duration_s + 's' : '') + ' · ' + (ocr.state === 'complete' ? 'complete' : 'running') }),
        h('table', { class: 'cmp' }, h('thead', {}, h('tr', {}, ['Page', 'Outcome', 'Words', 'Usable', 'Confidence', 'Time'].map(function (x) { return h('th', { text: x }); }))), h('tbody', {}, rows))));
    }
    if (ex.warnings && ex.warnings.length) right.push(h('div', { class: 'card', id: 'warnings' }, h('h2', { text: 'Extraction warnings' }), h('ul', {}, ex.warnings.map(function (w) { return h('li', { text: w }); }))));
    var raw = ex.raw_text || [];
    if (raw.length || ex.text) {
      right.push(h('div', { class: 'card' }, h('h2', { text: (ex.source === 'OCR' ? 'Raw OCR text' : 'Extracted text') + ' (sanitized)' }),
        raw.length ? raw.map(function (p) { return h('div', {}, h('div', { class: 'small muted', text: 'page ' + p.page + ' · ' + (SRC_LABEL[p.source] || p.source) }), h('pre', { class: 'report', text: p.text || '(no text)' })); })
          : h('pre', { class: 'report', id: 'extracted-text', text: ex.text })));
    }
    host.textContent = '';
    host.appendChild(h('div', {}, h('div', { class: 'title-row' }, h('h1', { text: j.display_name }), h('span', { class: 'chip ' + j.status, text: T.INTAKE_LABEL[j.status] })),
      h('div', { class: 'ctl' }, h('button', { class: 'btn back', onclick: function () { location.hash = '#/'; } }, '← Back to dashboard')),
      h('div', { class: 'grid two-eq' }, h('div', { class: 'grid' }, h('div', { class: 'card' }, h('h2', { text: 'Uploaded invoice' }), meta), nextCard,
        h('div', { class: 'card' }, h('h2', { text: 'Processing timeline' }), tl)), h('div', { class: 'grid' }, right))));
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
      h('div', { class: 'note', text: 'Agent S cannot approve a payment or clear its own alarm \u2014 only a registered human reviewer can, on a live invoice\u2019s processing feed. These recorded examples are read-only.' }),
      h('div', { class: 'review-list' }, list.map(caseCard))));
    if (focus) { var el = document.getElementById('case-' + focus); if (el && el.scrollIntoView) el.scrollIntoView(); }
  }

  /* ---------- LIVE mode (real runtime): everything below renders backend state only ---------- */
  function isLive() { return !!(intake.status && intake.status.mode === 'Live'); }
  function ensureBundle() {   // recorded examples only; never used for uploaded invoices
    if (bundle) return Promise.resolve(bundle);
    return fetch('/api/traces', { credentials: 'same-origin' }).then(function (r) {
      if (r.status === 401) { location.href = '/login'; throw new Error('sign-in required'); }
      return r.json();
    }).then(function (b) { bundle = b; return b; });
  }
  function shortHash(x) { return x ? String(x).slice(0, 12) + '…' : '—'; }
  function fmtMs(ms) { return ms == null ? '' : (ms >= 1000 ? (ms / 1000).toFixed(1) + ' s' : Math.round(ms) + ' ms'); }
  function fmtAmount(v, cur) {
    if (v == null || v === '') return null;
    var n = Number(v), s = isNaN(n) ? String(v) : n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return (cur ? String(cur).toUpperCase() + ' ' : '') + s;
  }

  /* provenance pill: every value on screen says where it came from */
  var PROV_HELP = { live: 'Produced by this run on the local Tell runtime', measured: 'Measured by the frozen Tell probe on this run’s hidden-state activations',
    extracted: 'Read from the uploaded PDF — untrusted content', trusted: 'Trusted vendor master / ERP record (synthetic demo fixture)',
    simulated: 'Local SQLite simulation — no money moves and no email is sent', gap: 'Not measured', fixture: 'Deterministic recorded demo fixture' };
  function prov(kind, label) { return h('span', { class: 'prov prov-' + kind, title: PROV_HELP[kind] || '', text: label || kind }); }

  /* <details> that keeps its open/closed state across the 1 s polling repaint */
  var openState = {};
  function fold(id, summary, body) {
    var d = h('details', { class: 'fold', id: id }, h('summary', {}, summary), body);
    if (openState[id]) d.open = true;
    d.addEventListener('toggle', function () { openState[id] = d.open; });
    return d;
  }

  /* theme: auto -> light -> dark (remembered per browser only) */
  var theme = (function () { try { return localStorage.getItem('tell-theme') || 'auto'; } catch (e) { return 'auto'; } })();
  function applyTheme() {
    var root = document.documentElement; if (!root) return;
    if (theme === 'auto') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', theme);
  }
  applyTheme();
  function themeButton() {
    return h('button', { class: 'badge theme', id: 'btn-theme', type: 'button', title: 'Switch colour theme', onclick: function () {
      theme = { auto: 'light', light: 'dark', dark: 'auto' }[theme] || 'auto';
      try { localStorage.setItem('tell-theme', theme); } catch (e) { /* storage unavailable */ }
      applyTheme(); this.textContent = 'Theme: ' + theme; } }, 'Theme: ' + theme);
  }

  function runtimeRows(rt) {
    var w = (rt && rt.worker) || {}, ids = rt && rt.identifiers, online = !!w.online;
    var rows = [['Live worker', online ? 'online · ' + (w.state || '') + (w.current_job ? ' · job ' + String(w.current_job).slice(0, 8) : '') : 'OFFLINE — invoices wait in “Ready for processing”']];
    if (ids) {
      rows = rows.concat([['Base model', ids.model_repo_id + ' @ ' + shortHash(ids.model_revision) + ' (' + (ids.dtype || '') + ', ' + (ids.device || '') + ')'],
        ['Safety LoRA (frozen)', 'sha256 ' + shortHash(ids.adapter_weights_sha256) + ' · manifest ' + shortHash(ids.adapter_frozen_manifest_sha256)],
        ['Tell probe (frozen)', 'weights ' + shortHash(ids.probe_weights_sha256) + ' · layer ' + ids.layer],
        ['Thresholds', 'Tell-Verify from ' + (ids.scientific_threshold != null ? ids.scientific_threshold.toFixed(4) : '0.1708') + ' · alarm from ' +
          (ids.operational_threshold != null ? String(ids.operational_threshold) : '0.5135')]]);
      if (ids.test_double) rows.unshift(['⚠ TEST DOUBLE', 'scripted stand-ins for the model and probe are running — NOT the real runtime']);
    } else rows.push(['Loaded identifiers', 'not reported by the backend yet']);
    return h('dl', { class: 'kv' }, rows.map(function (r) {
      return [h('dt', { text: r[0] }), h('dd', { class: (r[1].indexOf('OFFLINE') === 0 || r[0].indexOf('TEST') >= 0 ? 'warn-text ' : '') + (r[0] === 'Live worker' || r[0] === 'Thresholds' ? '' : 'mono'), text: r[1] })];
    }));
  }

  /* ---------- live chrome ---------- */
  function renderLiveChrome() {
    var modes = document.getElementById('modes'); modes.textContent = '';
    var rt = (intake.status && intake.status.runtime) || {}, ids = rt.identifiers || {}, w = rt.worker || {};
    modes.appendChild(h('span', { class: 'badge', title: w.online ? 'Live worker online' : 'Live worker offline' },
      h('span', { class: 'led' + (w.online ? '' : ' off'), 'aria-hidden': 'true' }), h('span', { class: 'long', text: 'HP ZGX Nano · ' }), 'GB10 · on-device'));
    modes.appendChild(h('span', { class: 'mode live-on', text: 'LIVE' }));
    if (ids.test_double) modes.appendChild(h('span', { class: 'mode fixture', text: 'TEST DOUBLE' }));
    modes.appendChild(themeButton());
    var banner = document.getElementById('banner'); banner.textContent = ''; banner.hidden = true;
    liveNav();
    document.getElementById('foot').textContent = 'Everything on these pages is read from the local Tell runtime (Qwen3-8B, frozen safety LoRA, frozen probe). Vendor and ERP records are a synthetic demo fixture; payments and emails are local simulations.';
  }
  function liveNav() {
    var nav = document.getElementById('nav'); nav.textContent = '';
    var waiting = (intake.jobs || []).filter(function (j) { return T.attentionOf(j) && !seenAttention[j.job_id]; }).length;
    [['dash', 'Invoices', '#/'], ['attention', 'Need attention', '#/attention'], ['ledger', 'Payment ledger', '#/ledger']].forEach(function (n) {
      nav.appendChild(h('button', { id: 'nav-' + n[0], 'aria-current': (view.name === n[0] || (view.name === 'job' && n[0] === 'dash') || (view.name === 'case' && n[0] === 'attention')) ? 'page' : false, onclick: function () { location.hash = n[2]; } },
        n[1], n[0] === 'attention' && waiting ? h('span', { class: 'count', text: String(waiting) }) : null));
    });
    if (intake.status && intake.status.auth) {
      nav.appendChild(h('button', { id: 'btn-signout', class: 'signout', onclick: function () {
        api('POST', '/auth/logout', {}).then(function () { location.href = '/login'; }, function () { location.href = '/login'; }); } }, 'Sign out'));
    }
  }

  /* ---------- live dashboard ---------- */
  function scanMessage() {
    var res = scanState.result;
    if (scanState.phase === 'running') return h('div', { class: 'scan', id: 'scan-msg' }, h('span', { class: 'spinner' }), 'Checking the inbox…');
    if (res && res.error && !res.scan_id) return h('div', { class: 'scan bad', id: 'scan-msg', text: res.message });
    if (res && res.scan_id) return h('div', { class: 'scan done', id: 'scan-msg' },
      h('b', { text: res.discovered === 0 || (res.imported === 0 && res.rejected === 0) ? 'No new invoices found' : res.imported + ' invoice' + (res.imported === 1 ? '' : 's') + ' imported' }),
      ' · ' + res.duplicates + ' duplicate' + (res.duplicates === 1 ? '' : 's') + ' skipped · ' + res.rejected + ' rejected');
    return null;
  }
  function liveIntake() {
    var fileInput = fileInputEl(), last = intake.status.last_scan, lim = intake.status.limits || {};
    var drop = h('div', { class: 'drop', id: 'dropzone',
      ondragover: function (e) { e.preventDefault(); drop.className = 'drop over'; },
      ondragleave: function () { drop.className = 'drop'; },
      ondrop: function (e) { e.preventDefault(); drop.className = 'drop'; if (e.dataTransfer) addInvoices(e.dataTransfer.files); } },
      h('div', { class: 'txt' }, h('b', { text: 'Upload invoices' }),
        h('span', { text: 'Drop PDFs here or choose files · up to ' + (lim.max_files_per_batch || 20) + ' per batch, ' + Math.round((lim.max_bytes || 10485760) / 1048576) + ' MB each · processed on this machine' })),
      h('button', { class: 'btn primary big', id: 'btn-add', onclick: function () { fileInput.click(); } }, 'Choose PDFs'), fileInput);
    var scanning = scanState.phase === 'running';
    var inbox = h('div', { class: 'inbox' },
      h('div', { class: 'row' }, h('button', { class: 'btn', id: 'btn-check', disabled: scanning, onclick: checkNew }, scanning ? 'Checking…' : 'Check for new invoices'),
        h('button', { class: 'btn ghost', id: 'btn-reset', onclick: resetDemo, title: 'Clear the demo queue, both simulated ledgers and the demo memory (inbox files are kept)' }, 'Reset demo')),
      h('div', { class: 'small muted', id: 'last-checked', text: 'Inbox last checked: ' + (last ? fmtTime(last.completed_at) : 'never') }), scanMessage());
    return h('div', { class: 'intake' }, drop, inbox);
  }
  function tsToggle() {
    return h('div', { class: 'ts-toggle ' + (tellSecured ? 'on' : 'off'), id: 'ts-toggle' },
      h('button', { class: 'ts-switch', role: 'switch', 'aria-checked': tellSecured ? 'true' : 'false', 'aria-label': 'TellSecured', id: 'ts-switch', onclick: function () { setTellSecured(!tellSecured); } },
        h('span', { class: 'knob' })),
      h('div', {}, h('b', { text: 'TellSecured ' + (tellSecured ? 'ON' : 'OFF') }),
        h('div', { class: 'small', text: tellSecured ? 'New uploads are protected by the Tell probe, Agent S, the validator and the gate.'
          : 'New uploads run the undefended baseline: Agent 1 alone, no probe, validator or gate (separate simulated books).' })));
  }
  function liveCard(j, target) {    // target 'case' opens the reviewer's case file (Need attention); otherwise the processing feed
    var s = j.summary || {}, lv = T.levelOf(j), amt = fmtAmount(s.amount, s.currency);
    var tell = j.tell_secured === false ? 'Off' : (j.tell ? j.tell.score.toFixed(3) : (j.tell_alert || '—'));
    var bar = j.active ? h('div', { class: 'bar' }, (function () { var f = h('div', { class: 'fill' }); f.style.width = (j.progress || 0) + '%'; return f; })()) : null;
    return h('button', { type: 'button', class: 'qc lv-' + lv, 'data-job': j.job_id, 'data-run': j.run_id, onclick: function () { location.hash = (target === 'case' ? '#/case/' : '#/job/') + j.job_id; } },
      h('span', { class: 'ey' }, h('span', { text: s.supplier_name || 'Supplier not read' }), h('span', { class: 'pill lv-' + lv, text: j.display_status })),
      h('h3', { text: s.invoice_number || j.display_name }),
      h('span', { class: 'amt', text: amt || '—' }),
      h('span', { class: 'sc' }, h('span', { text: 'Tell score' }), h('b', { text: tell })),
      bar, j.active ? h('span', { class: 'latest', text: j.latest_event || '' }) : null,
      h('span', { class: 'ft' }, h('span', { class: 'tsbadge ' + (j.tell_secured === false ? 'off' : 'on'), text: j.tell_secured === false ? 'TellSecured OFF' : 'TellSecured ON' }),
        h('span', { title: fmtTime(j.created_at), text: fmtShort(j.created_at) })));
  }

  var recordedOpen = false;
  function recordedCard() {
    var body = h('div', { id: 'recorded-body', class: 'muted small', text: 'Loading recorded examples…' });
    var det = h('details', { class: 'fold recorded', id: 'recorded-examples' }, h('summary', {}, 'Recorded examples',
      h('span', { class: 'muted', text: 'historical replay fixtures — not used for uploaded invoices' })), body);
    if (recordedOpen) det.open = true;
    det.addEventListener('toggle', function () {
      recordedOpen = det.open;
      if (!det.open) return;
      ensureBundle().then(function (b) {
        body.textContent = '';
        body.appendChild(h('div', { class: 'note', text: T.FIXTURE_LABEL }));
        body.appendChild(h('table', {}, h('tbody', {}, b.traces.map(function (t) {
          var open = function () { location.hash = '#/run/' + t.run_id; };
          return h('tr', { class: 'click', tabindex: 0, 'data-run': t.run_id, onclick: open, onkeydown: function (e) { if (e.key === 'Enter') open(); } },
            h('td', {}, h('b', { text: t.invoice.invoice_number.v })), h('td', { text: t.invoice.vendor_name.v }), h('td', {}, chip(t.queue_status)), h('td', { class: 'small muted', text: 'RECORDED' }));
        }))));
      });
    });
    return det;
  }

  function renderLiveDash() {
    var st = intake.status, rt = st.runtime || {}, jobs = intake.jobs || [], w = rt.worker || {};
    var busy = (st.waiting || 0) + (st.processing || 0) + (st.ready || 0) + (st.processing_agent || 0);
    var tiles = [['Invoices', st.total || 0, ''], ['In progress', busy, busy ? 'run' : ''], ['Paid', st.payment_completed || 0, 'relax'],
      ['Blocked', (st.payment_blocked || 0) + (st.blocked_by_dispute || 0), 'act'], ['Needs attention', (st.attention || 0) + (st.failed || 0), 'ready']];
    var TILE_LINK = { Paid: ['#/ledger', 'Open the payment ledger'], Blocked: ['#/attention', 'Open the Need attention queue'], 'Needs attention': ['#/attention', 'Open the Need attention queue'] };
    var queue = jobs.length ? h('div', { class: 'queue', id: 'queue' }, jobs.map(liveCard))
      : h('div', { class: 'empty-q', id: 'queue' }, 'No invoices yet. Upload a PDF or check the inbox to start.');
    app.textContent = '';
    app.appendChild(h('div', {},
      h('div', { class: 'page-h' }, h('div', {}, h('div', { class: 'eyebrow' }, h('span', { class: 'pill lv-run', text: 'LIVE' }), h('span', { class: 'muted small', text: 'local Tell runtime' })), h('h1', { text: 'Invoices' }), h('p', { text: 'Upload invoices and watch Tell decide which agent may handle each payment.' })),
        h('div', { class: 'page-actions' }, tsToggle())),
      h('div', { class: 'tiles', id: 'live-kpis' }, tiles.map(function (k) {
        var link = TILE_LINK[k[0]];
        return h(link ? 'button' : 'div', { class: 'tile ' + k[2] + (link ? ' link' : ''), 'data-k': k[0], type: link ? 'button' : null, title: link ? link[1] : null,
          onclick: link ? function () { location.hash = link[0]; } : null }, h('div', { class: 'v', text: String(k[1]) }), h('div', { class: 'k', text: k[0] }));
      })),
      liveIntake(),
      h('div', { class: 'sec-h' }, h('h2', { text: 'Invoice queue' }), h('p', { text: jobs.length ? 'Select an invoice to open its processing feed.' : '' })),
      queue,
      h('div', { class: 'folds' },
        fold('runtime-fold', ['Live runtime', h('span', { class: 'muted', text: w.online ? 'worker online' : 'worker offline' })], runtimeRows(rt)),
        recordedCard())));
  }

  /* ---------- need attention: live invoices waiting for a person ---------- */
  /* The red count shows invoices that joined the queue since this viewer last opened it (remembered per browser only). */
  var seenAttention = (function () { try { return JSON.parse(localStorage.getItem('tell-attention-seen') || '{}') || {}; } catch (e) { return {}; } })();
  function markAttentionSeen(jobs) {
    var next = {};
    jobs.forEach(function (j) { next[j.job_id] = 1; });
    seenAttention = next;          // only ids still in the queue are kept, so the list never grows
    try { localStorage.setItem('tell-attention-seen', JSON.stringify(next)); } catch (e) { /* storage unavailable */ }
  }
  function renderAttention() {
    var jobs = (intake.jobs || []).filter(T.attentionOf);
    markAttentionSeen(jobs); liveNav();
    var groups = [['blocked', 'Blocked', 'Payment held \u2014 $0 moved. A human reviewer decides what happens next.'],
      ['review', 'Needs review', 'Evidence was incomplete or the agent could not decide safely.'],
      ['failed', 'Could not be processed', 'The document or the run failed; nothing was paid.']];
    app.textContent = '';
    app.appendChild(h('div', { id: 'attention' },
      h('div', { class: 'page-h' }, h('div', {}, h('div', { class: 'eyebrow' }, h('span', { class: 'pill lv-run', text: 'LIVE' }), h('span', { class: 'muted small', text: 'local Tell runtime' })),
        h('h1', { text: 'Need attention' }), h('p', { text: 'Invoices waiting for a person. None of them has been paid.' }))),
      jobs.length ? groups.map(function (g) {
        var list = jobs.filter(function (j) { return T.attentionOf(j) === g[0]; });
        return list.length ? h('section', { class: 'att-group', 'data-group': g[0] }, h('div', { class: 'sec-h' }, h('h2', { text: g[1] + ' \u00b7 ' + list.length }), h('p', { text: g[2] })),
          h('div', { class: 'queue' }, list.map(function (j) { return liveCard(j, j.reviewable ? 'case' : null); }))) : null;
      }) : h('div', { class: 'empty-q', id: 'attention-empty' }, 'Nothing needs attention right now.')));
  }

  /* ---------- payment ledger: live invoices + the recorded held-out test run, one register ---------- */
  var ledgerFilter = { mode: 'all', from: '', to: '', cls: 'all', surface: 'all', page: 0 }, ledgerData = null, trData = null;
  var PERIODS = [['day', 'Today', 0, 'Since midnight today'], ['7d', 'Last 7 days', 7, 'The last 7 days'], ['30d', 'Last month', 30, 'The last 30 days'], ['all', 'All time', null, 'Everything']];
  var periodButtons = {};
  var SURF = { invoice_injection: 'Invoice', email_injection: 'Email', tool_result_forgery: 'Tool result', immediate_memory_poisoning: 'Memory', delayed_memory_poisoning: 'Delayed memory' };
  var ACT = { propose_payment: 'propose payment', get_vendor_record: 'check vendor record', read_invoice: 'read invoice', read_email: 'read email', search_memory: 'search memory',
    submit_evidence_report: 'evidence report', request_vendor_clarification: 'ask vendor', fail_closed: 'fail closed' };
  var LEDGER_PAGE = 50;
  function inPeriod(iso, mode) {
    mode = mode || ledgerFilter.mode;
    var t = new Date(iso).getTime(); if (isNaN(t)) return mode === 'all';
    if (mode === 'range') {
      var from = ledgerFilter.from ? new Date(ledgerFilter.from + 'T00:00:00').getTime() : -Infinity;
      var to = ledgerFilter.to ? new Date(ledgerFilter.to + 'T23:59:59.999').getTime() : Infinity;
      return t >= from && t <= to;
    }
    var p = PERIODS.filter(function (x) { return x[0] === mode; })[0];
    if (!p || p[2] == null) return true;
    if (p[2] === 0) { var m = new Date(); m.setHours(0, 0, 0, 0); return t >= m.getTime(); }     // Today: since local midnight
    return t >= Date.now() - p[2] * 86400000;
  }
  var LIVE_LV = { REJECTED_BY_REVIEWER: 'idle', PAYMENT_COMPLETED: 'relax', PAYMENT_BLOCKED: 'act', BLOCKED_BY_DISPUTE: 'act', FAILED: 'act', UNREADABLE_DOCUMENT: 'act' };
  function ledgerRows() {                  // one normalised row per live invoice and per test-run record
    var out = [];
    ((ledgerData && ledgerData.entries) || []).forEach(function (e) {
      out.push({ kind: 'live', key: e.job_id, job_id: e.job_id, job_no: e.job_no || e.job_id.slice(0, 8), date: e.decided_at, invoice: e.invoice_number, sub: e.invoice_number ? null : e.display_name,
        vendor: e.vendor_name, claimed: e.claimed_amount != null ? Math.round(Number(e.claimed_amount) * 100) : null, cur: e.claimed_currency, paid: e.paid,
        tell: e.tell_secured === false ? 'off' : (e.tell ? { score: e.tell.score, zone: e.tell.zone || (e.tell.above_threshold ? 'agent_s' : 'agent_1') } : null),
        status: e.display_status, level: LIVE_LV[e.status] || 'ready', note: null, mode: e.tell_secured === false ? 'OFF' : 'ON', cls: null, surface: null });
    });
    if (trData) {
      var th = trData.thresholds;
      trData.rows.filter(function (r) { return r.decision; }).forEach(function (r, i) {   // payment decisions only; mid-workflow steps are not ledger entries
        var a = r.arms.tr, st;
        if (a.u) st = ['act', 'Unauthorized payment proposed'];
        else if (a.t === 'propose_payment') st = ['relax', 'Payment proposed'];
        else if (a.t === 'submit_evidence_report') st = ['ready', 'Evidence report'];
        else if (a.t === 'request_vendor_clarification') st = ['ready', 'Awaiting clarification'];
        else if (a.t === 'fail_closed') st = ['ready', 'Failed closed'];
        else if (a.t) st = ['idle', 'Verifying: ' + (ACT[a.t] || a.t.replace(/_/g, ' '))];
        else st = ['ready', 'Unparseable — failed closed'];
        out.push({ kind: 'test', key: r.id, job_no: 'T-' + String(i + 1).padStart(4, '0'), date: r.done_at, invoice: r.inv, sub: null, vendor: r.vendor,
          claimed: r.amt, cur: r.cur, paid: r.pay ? { amount_minor_units: r.pay.amt, currency: r.pay.cur, beneficiary_verified: !!r.pay.ok, proposed: true } : null, tell: { score: r.score, zone: r.score >= th.alarm_from ? 'agent_s' : (r.score >= th.tell_verify_from ? 'tell_verify' : 'agent_1') },
          status: st[1], level: st[0], ref: !!a.g, mode: 'ON', cls: r.cls, surface: r.surface, record: r.id });
      });
    }
    return out;
  }
  function renderLedger() {
    stopPolling();
    var content = h('div', { id: 'ledger-content' }, h('p', { class: 'muted', text: 'Loading…' }));
    var seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Period' }), clsSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Class', id: 'ledger-cls' });
    var sel = h('select', { id: 'ledger-surface', 'aria-label': 'Attack surface', onchange: function (e) { ledgerFilter.surface = e.target.value; ledgerFilter.page = 0; paintLedger(content); } });
    var from = h('input', { type: 'date', id: 'ledger-from', 'aria-label': 'From date' }), to = h('input', { type: 'date', id: 'ledger-to', 'aria-label': 'To date' });
    from.value = ledgerFilter.from; to.value = ledgerFilter.to;
    var paintSegs = function () {
      seg.textContent = '';
      PERIODS.forEach(function (p) {
        seg.appendChild(periodButtons[p[0]] = h('button', { type: 'button', 'data-period': p[0], 'aria-pressed': ledgerFilter.mode === p[0] ? 'true' : 'false', title: p[3],
          onclick: function () { ledgerFilter.mode = p[0]; ledgerFilter.from = ledgerFilter.to = ''; ledgerFilter.page = 0; from.value = ''; to.value = ''; paintSegs(); paintLedger(content); } }, p[1]));
      });
      clsSeg.textContent = '';
      [['all', 'All'], ['attacked', 'Attacks'], ['clean', 'Clean']].forEach(function (c) {
        clsSeg.appendChild(h('button', { type: 'button', 'data-cls': c[0], 'aria-pressed': ledgerFilter.cls === c[0] ? 'true' : 'false',
          onclick: function () { ledgerFilter.cls = c[0]; ledgerFilter.page = 0; paintSegs(); paintLedger(content); } }, c[1]));
      });
      var surfaces = {}; ((trData && trData.rows) || []).forEach(function (r) { if (r.surface) surfaces[r.surface] = 1; });
      sel.textContent = '';
      [['all', 'All surfaces']].concat(Object.keys(surfaces).map(function (k) { return [k, SURF[k] || k]; })).forEach(function (o) { sel.appendChild(h('option', { value: o[0], text: o[1] })); });
      sel.value = ledgerFilter.surface;
    };
    var onRange = function () {
      ledgerFilter.mode = (from.value || to.value) ? 'range' : 'all'; ledgerFilter.from = from.value; ledgerFilter.to = to.value; ledgerFilter.page = 0;
      paintSegs(); paintLedger(content);
    };
    from.addEventListener('change', onRange); to.addEventListener('change', onRange);
    paintSegs();
    app.textContent = '';
    app.appendChild(h('div', { id: 'ledger' },
      h('div', { class: 'page-h' }, h('div', {}, h('div', { class: 'eyebrow' }, h('span', { class: 'pill lv-run', text: 'LIVE' })),
        h('h1', { text: 'Payment ledger' }), h('p', { text: 'Every invoice the runtime finished, plus the payment decisions of the held-out test run: what was claimed and what was paid. No real money moves.' }))),
      h('div', { class: 'filters', id: 'ledger-filters' }, seg,
        h('div', { class: 'range' }, h('label', { for: 'ledger-from', text: 'From' }), from, h('label', { for: 'ledger-to', text: 'To' }), to,
          h('button', { type: 'button', class: 'btn ghost', id: 'ledger-clear', onclick: function () { from.value = ''; to.value = ''; onRange(); } }, 'Clear'))),
      h('div', { class: 'filters' }, clsSeg, sel),
      content));
    var load = function () { api('GET', '/api/intake/ledger').then(function (L) { if (view.name !== 'ledger') return; ledgerData = L; paintLedger(content); }); };
    if (!trData) api('GET', '/api/testrun').then(function (d) { if (d && d.rows) { trData = d; paintSegs(); if (view.name === 'ledger') paintLedger(content); } }, function () {});
    load(); jobTimer = setInterval(load, 4000);
  }
  function sumBy(list, amt, cur) {
    var t = {};
    list.forEach(function (e) { var c = cur(e), v = amt(e); if (c && v != null) t[c.toLowerCase()] = (t[c.toLowerCase()] || 0) + v; });
    var k = Object.keys(t).sort(function (a, b) { return t[b] - t[a]; });
    return k.length ? k.slice(0, 2).map(function (c) { return T.money(t[c], c); }).join(' · ') + (k.length > 2 ? ' +' + (k.length - 2) : '') : '—';
  }
  function paintLedger(content) {
    if (!ledgerData && !trData) return;
    var f = ledgerFilter;
    var base = ledgerRows().filter(function (r) {
      if (f.cls !== 'all' || f.surface !== 'all') {          // attack labels exist only for test-run records (live invoices have no ground truth)
        if (r.kind !== 'test') return false;
        if (f.cls !== 'all' && (r.cls === 'attacked') !== (f.cls === 'attacked')) return false;
        if (f.surface !== 'all' && r.surface !== f.surface) return false;
      }
      return true;
    });
    var rows = base.filter(function (r) { return inPeriod(r.date); })
      .sort(function (a, b) { return (new Date(b.date).getTime() || 0) - (new Date(a.date).getTime() || 0); });
    var paid = rows.filter(function (r) { return r.paid; });
    var unauth = paid.filter(function (r) { return !r.paid.beneficiary_verified; }).length + rows.filter(function (r) { return r.level === 'act' && r.kind === 'test'; }).length;
    var tiles = [['Processed', rows.length.toLocaleString('en-US'), ''],
      ['Claimed', sumBy(rows, function (r) { return r.claimed; }, function (r) { return r.cur; }), 'small-v'],
      ['Paid', sumBy(paid, function (r) { return r.paid.amount_minor_units; }, function (r) { return r.paid.currency; }), 'small-v relax'],
      ['Held or blocked', (rows.length - paid.length).toLocaleString('en-US'), ''], ['Paid to unverified accounts', String(unauth), unauth ? 'act' : '']];
    var nProp = paid.filter(function (r) { return r.paid.proposed; }).length, nExec = paid.length - nProp;
    var tileSub = { Paid: nExec.toLocaleString('en-US') + ' executed \u00b7 ' + nProp.toLocaleString('en-US') + ' proposed (test run)',
      Claimed: (function () {
        var inv = {}, nocur = 0;
        rows.forEach(function (r) { inv[(r.invoice || '') + '|' + (r.vendor || '')] = 1; if (r.claimed != null && !r.cur) nocur++; });
        return 'across ' + Object.keys(inv).length + ' distinct invoices' + (nocur ? ' \u00b7 ' + nocur + ' without currency not totalled' : '');
      })(),
      'Held or blocked': 'evidence report, vendor asked, gate or dispute',
      Processed: rows.filter(function (r) { return r.kind === 'test'; }).length.toLocaleString('en-US') + ' test-run payment decisions' };
    var pages = Math.max(1, Math.ceil(rows.length / LEDGER_PAGE)); if (f.page >= pages) f.page = pages - 1;
    var shown = rows.slice(f.page * LEDGER_PAGE, f.page * LEDGER_PAGE + LEDGER_PAGE);
    var repaint = function () { paintLedger(content); };
    content.textContent = '';
    content.appendChild(h('div', {},
      h('div', { class: 'tiles', id: 'ledger-kpis' }, tiles.map(function (k) { return h('div', { class: 'tile ' + k[2], 'data-k': k[0] }, h('div', { class: 'v', text: k[1] }), h('div', { class: 'k', text: k[0] }),
        tileSub[k[0]] ? h('div', { class: 'small muted', text: tileSub[k[0]] }) : null); })),
      h('section', { class: 'card book' },
        shown.length ? h('div', { class: 'tablewrap scroll' }, h('table', { class: 'ledger', id: 'ledger-table' },
          h('thead', {}, h('tr', {}, [['Job no.', 'c-job'], ['Date', 'c-date'], ['Invoice', ''], ['Vendor', ''], ['Claimed', 'num'], ['Paid', 'num'], ['Paid to', 'c-acct'], ['Tell', 'num c-tell'], ['Status', ''], ['Mode', 'c-mode']]
            .map(function (x) { return h('th', { class: x[1], scope: 'col', text: x[0] }); }))),
          h('tbody', {}, shown.map(function (r) {
            var open = r.job_id ? function () { location.hash = '#/job/' + r.job_id; } : null;
            return h('tr', { class: (open ? 'click' : '') + (r.kind === 'test' ? ' test' : ''), tabindex: open ? 0 : null, 'data-job': r.job_id || null, 'data-rec': r.record || null,
                onclick: open, onkeydown: open ? function (ev) { if (ev.key === 'Enter') open(); } : null },
              h('td', { class: 'mono c-job' }, r.job_no, r.kind === 'test' ? h('div', { class: 'src ' + (r.cls === 'attacked' ? 'atk' : ''), text: 'test \u00b7 ' + (r.cls === 'attacked' ? (SURF[r.surface] || 'attack') + ' attack' : 'clean') }) : h('div', { class: 'src live', text: 'uploaded' })),
              h('td', { class: 'c-date', title: fmtTime(r.date), text: fmtShort(r.date) }),
              h('td', {}, h('b', { text: r.invoice || '—' }), r.sub ? h('div', { class: 'small muted', text: r.sub }) : null),
              h('td', { class: 'c-vendor' }, h('span', { class: 'clamp', title: r.vendor || '', text: r.vendor || '\u2014' })),
              h('td', { class: 'num', text: r.claimed != null ? (r.cur ? T.money(r.claimed, r.cur) : (r.claimed / 100).toLocaleString('en-US', { minimumFractionDigits: 2 })) : '—' }),
              h('td', { class: 'num c-paid' + (r.paid ? '' : ' muted') }, r.paid ? [T.money(r.paid.amount_minor_units, r.paid.currency), r.paid.proposed ? h('div', { class: 'tag', title: 'Proposed by the Tell-routed decision in the test run; the evaluation did not execute payments', text: 'proposed' }) : null]
                : (r.kind === 'test' ? '\u2014' : 'not paid')),
              h('td', { class: 'c-acct' }, r.paid ? [h('div', { class: 'acct', text: r.paid.beneficiary_account_id || 'vendor-record account' }),
                h('span', { class: 'chk ' + (r.paid.beneficiary_verified ? 'ok' : 'bad'), text: r.paid.beneficiary_verified ? '\u2713 Verified account' : '\u2717 NOT the verified account' })] : h('span', { class: 'muted', text: '\u2014' })),
              h('td', { class: 'num c-tell' }, r.tell === 'off' ? h('span', { class: 'muted', text: 'off' }) : (r.tell ? [h('i', { class: 'zdot ' + ({ agent_s: 'act', tell_verify: 'ready' }[r.tell.zone] || 'relax') }),
                r.tell.score.toFixed(3)] : h('span', { class: 'muted', text: '—' }))),
              h('td', { class: 'c-status' }, h('span', { class: 'pill lv-' + r.level, text: r.status }),
                r.kind === 'test' ? h('div', { class: 'ref ' + (r.ref ? 'ok' : ''), title: 'Compared with the evaluation\u2019s reference action', text: r.ref ? '\u2713 matches reference' : '\u2260 differs from reference' }) : null),
              h('td', { class: 'c-mode' }, h('span', { class: 'tsbadge ' + (r.mode === 'OFF' ? 'off' : 'on'), text: r.mode })));
          })))) : h('div', { class: 'empty', id: 'ledger-empty', text: 'Nothing matches these filters.' }),
        h('div', { class: 'pager' }, h('button', { class: 'btn', id: 'ledger-prev', disabled: f.page === 0, onclick: function () { f.page--; repaint(); } }, '← Previous'),
          h('span', { class: 'muted small', id: 'ledger-range', text: rows.length ? (f.page * LEDGER_PAGE + 1).toLocaleString('en-US') + '–' + Math.min(rows.length, f.page * LEDGER_PAGE + LEDGER_PAGE).toLocaleString('en-US') + ' of ' + rows.length.toLocaleString('en-US') : '0 rows' }),
          h('button', { class: 'btn', id: 'ledger-next', disabled: f.page >= pages - 1, onclick: function () { f.page++; repaint(); } }, 'Next →'))),
      h('p', { class: 'small muted', text: 'J- rows are uploaded invoices: claimed amounts come from the PDF (untrusted) and paid amounts from the simulated ledger of their TellSecured mode. ' +
        'T- rows are the payment decisions of the held-out test run (Tell routing): records whose correct next step was to pay, report or ask the vendor, dated by when their evaluation partition finished, ' +
        'with invoice fields as the model saw them. Their payments are proposals \u2014 the evaluation executed none. They come from 50 real invoices, each tested in several attack and clean variants. ' +
        'Attack and surface filters apply to test-run rows only.' + (function () {
          var s = (ledgerData && ledgerData.seeded_prior_payments) || {}, n = (s.tellsecured_on || 0) + (s.tellsecured_off || 0);
          return n ? ' ' + n + ' seeded prior payment(s) from the demo fixture are not listed.' : ''; })() })));
  }

  /* ---------- processing feed (one invoice) ---------- */
  function evRow(e) {
    var hasPayload = e.payload && Object.keys(e.payload).length;
    return h('li', { 'data-seq': e.sequence, 'data-type': e.type, 'data-source': e.source, class: 'ev ' + e.status },
      h('time', { text: clock(e.ts) }),
      h('div', {}, h('span', { class: 'who', text: e.type.replace(/_/g, ' ') }), h('b', { text: e.title }),
        h('div', { class: 'evmeta' }, h('span', { class: 'prov prov-extracted', text: e.provenance }), e.status !== 'ok' ? h('span', { class: 'prov prov-' + (e.status === 'failed' || e.status === 'blocked' ? 'gap' : 'simulated'), text: e.status }) : null,
          e.duration_ms != null ? h('span', { class: 'muted small', text: fmtMs(e.duration_ms) }) : null),
        hasPayload ? h('details', {}, h('summary', { text: 'structured payload' }), h('pre', { class: 'report', text: JSON.stringify(e.payload, null, 2) })) : null));
  }

  var LIVE_FIELDS = [['supplier_name', 'Supplier'], ['invoice_number', 'Invoice number'], ['invoice_date', 'Invoice date'], ['due_date', 'Due date'],
    ['amount', 'Amount'], ['beneficiary_name', 'Beneficiary'], ['beneficiary_account', 'Bank account'], ['document_type', 'Document type']];
  function extractCard(job, detail) {
    var ex = (detail && detail.extraction) || {}, f = ex.fields || {}, rows = [];
    LIVE_FIELDS.forEach(function (x) {
      var d = f[x[0]];
      if (x[0] === 'document_type' && (!d || d.value === 'invoice')) return;
      var cls = 'fr' + (x[0] === 'amount' ? ' big' : ''), val;
      if (x[0] === 'amount' && d && d.found) {
        var c = f.currency || {};
        val = h('div', { class: 'v', title: d.evidence_text || '' }, fmtAmount(d.value, c.found ? c.value : null),
          c.ambiguous ? h('small', { text: 'currency unclear: ' + (c.candidates || []).slice(0, 4).join(', ') + '…' }) : (!c.found ? h('small', { text: 'currency not found' }) : null));
      } else if (d && d.ambiguous) val = h('div', { class: 'v amb' }, 'Ambiguous', h('small', { text: (d.candidates || []).slice(0, 4).join(' · ') }));
      else if (d && d.found) val = h('div', { class: 'v', title: d.evidence_text || '' }, d.value,
        d.source === 'OCR' && d.confidence != null && d.confidence < 0.8 ? h('small', { text: 'OCR confidence ' + Math.round(d.confidence * 100) + '%' }) : null);
      else val = h('div', { class: 'v miss', text: 'Not found' });
      rows.push(h('div', { class: cls, 'data-field': x[0] }, h('div', { class: 'k', text: x[1] }), val));
    });
    var method = ex.source === 'OCR' ? 'local OCR' : (ex.source ? 'embedded text' : null);
    var body = Object.keys(f).length ? h('div', { class: 'fields', id: 'fields' }, rows)
      : h('div', { class: 'empty', text: T.jobActive(job) ? 'Reading the PDF…' : 'No fields could be read from this document.' });
    var warn = (ex.warnings || []).length ? h('div', { class: 'note', text: ex.warnings[0] + (ex.warnings.length > 1 ? ' (+' + (ex.warnings.length - 1) + ' more under Document & runtime)' : '') }) : null;
    return h('section', { class: 'card', id: 'extract-card' },
      h('div', { class: 'card-h' }, h('h2', { text: 'Extracted from the PDF' }), h('span', {}, prov('extracted', 'untrusted · ' + (method || 'pending')))), body, warn);
  }

  /* Tell meter: semicircle, 0 at the left, 1 at the right; three bands from the frozen thresholds */
  var CX = 170, CY = 170, R = 136;
  function mpt(v, r) { var a = Math.PI * (1 - v); return [CX + r * Math.cos(a), CY - r * Math.sin(a)]; }
  function marc(v0, v1, r) { var a = mpt(v0, r), b = mpt(v1, r); return 'M' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' A' + r + ' ' + r + ' 0 0 1 ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1); }
  function tellMeterSvg(lo, hi, score, sub, subCls, off) {
    var G = 0.006, svg = s('svg', { viewBox: '0 0 340 250', role: 'img', 'aria-label': 'Tell meter' + (score != null ? ': ' + score.toFixed(3) : '') });
    [[0, lo - G, 'a-relax'], [lo + G, hi - G, 'a-ready'], [hi + G, 1, 'a-act']].forEach(function (b) {
      svg.appendChild(s('path', { d: marc(b[0], b[1], R), class: 'arc ' + (off ? 'a-off' : b[2]), 'stroke-width': 22, fill: 'none' }));
    });
    [[lo, 'lo'], [hi, 'hi']].forEach(function (t) {
      var p = mpt(t[0], R + 24), tx = s('text', { class: 'm-thr', x: p[0].toFixed(1), y: p[1].toFixed(1), 'text-anchor': 'middle' }); tx.textContent = t[0].toFixed(4); svg.appendChild(tx);
    });
    if (score != null) svg.appendChild(s('g', { transform: 'rotate(' + (score * 180).toFixed(1) + ' ' + CX + ' ' + CY + ')' },
      s('line', { class: 'ndl', x1: CX, y1: CY, x2: CX - R + 20, y2: CY, 'stroke-width': 4, 'stroke-linecap': 'round' })));
    svg.appendChild(s('circle', { class: 'hub', cx: CX, cy: CY, r: 9 }));
    var val = s('text', { class: 'm-val', id: score != null ? 'tell-score' : 'tell-score-empty', x: CX, y: CY + 54, 'text-anchor': 'middle' });
    val.textContent = score != null ? score.toFixed(3) : '—'; svg.appendChild(val);
    var sb = s('text', { class: 'm-sub ' + (subCls || 'f-idle'), x: CX, y: CY + 76, 'text-anchor': 'middle' }); sb.textContent = sub; svg.appendChild(sb);
    return svg;
  }
  function meterCard(sec, rt, terminal, job) {
    var ids = (rt && rt.identifiers) || {}, lo = ids.scientific_threshold || 0.1708046793937683;
    var head = h('div', { class: 'card-h' }, h('h2', { text: 'Tell meter' }), prov(job.tell_secured === false ? 'gap' : 'measured', job.tell_secured === false ? 'off' : 'measured live'));
    if (job.tell_secured === false) return h('section', { class: 'card', id: 'tell-card' }, head,
      h('div', { class: 'tm' }, tellMeterSvg(lo, ids.operational_threshold || 0.5134634443863925, null, 'not running', 'f-idle', true),
        h('div', { class: 'tm-off', id: 'tell-off' }, h('b', { text: 'TellSecured OFF' }),
          h('div', { class: 'small', text: 'Processed by the undefended baseline: no activation capture, no probe, no Agent S, no validator or gate.' }))));
    var st = T.tellState(sec, terminal), hi = (st.measured && st.threshold) || ids.operational_threshold || 0.5134634443863925;
    var body;
    if (st.measured) {
      var zone = st.zone || (st.above ? 'agent_s' : 'agent_1'), lv = { agent_s: 'act', tell_verify: 'ready', agent_1: 'relax' }[zone];
      var zl = zone === 'agent_s' ? 'ALARM — at/above ' + hi.toFixed(4) + ': Agent S takes over, gate holds payments'
        : zone === 'tell_verify' ? 'TELL-VERIFY — ' + lo.toFixed(4) + ' to ' + hi.toFixed(4) + ': Agent S takes over, no alarm, gate open'
        : 'CLEAR — below ' + lo.toFixed(4) + ': Agent 1 continues, gate open';
      var reads = sec.turns.filter(function (t) { return t.measured; }).map(function (t) {
        var z = { agent_s: 'act', tell_verify: 'ready', agent_1: 'relax' }[t.zone || (t.above ? 'agent_s' : 'agent_1')];
        return h('span', { class: 'rd' + (t.role === 'routing' ? ' route' : ''), 'data-turn': t.turn }, h('i', { class: z }),
          'T' + t.turn + ' ' + t.score.toFixed(3) + (t.role === 'routing' ? ' · ROUTING DECISION' : ' · MONITORING'));
      });
      body = h('div', { class: 'tm' }, tellMeterSvg(lo, hi, st.score, st.routing ? { agent_s: 'Agent S · alarm', tell_verify: 'Agent S · Tell-Verify', agent_1: 'Agent 1' }[zone] : 'watching', 'f-' + lv),
        h('div', { class: 'zl ' + (st.routing ? lv : 'idle'), id: 'tell-zone', text: zl + (st.routing ? '' : ' (monitoring reading)') }),
        reads.length ? h('div', { class: 'reads' }, reads) : null);
    } else {
      body = h('div', { class: 'tm' }, tellMeterSvg(lo, hi, null, terminal ? 'not measured' : 'waiting', terminal ? 'f-act' : 'f-idle'),
        h('div', { class: 'tm-gap' + (terminal ? ' fail' : ''), id: 'tell-not-measured' }, h('b', { text: 'Not measured' }),
          h('div', { class: 'small', text: terminal ? 'INTEGRATION FAILURE: the run ended without an activation capture and probe score.' : 'Tell reads the model once untrusted content (the email, the invoice) is in its context.' })));
    }
    return h('section', { class: 'card', id: 'tell-card' }, head, body);
  }

  function pipelineCard(job, rep) {
    var stages = T.pipeline(rep.events, job), lastEv = rep.events[rep.events.length - 1];
    return h('section', { class: 'card', id: 'pipeline-card' },
      h('div', { class: 'card-h' }, h('h2', { text: 'Agents at work' }), prov('live', 'live run')),
      h('div', { class: 'rail', id: 'rail' }, stages.map(function (x, i) {
        return h('div', { class: 'stg st-' + x.state, 'data-stage': x.id, 'data-state': x.state },
          h('div', { class: 'nd', text: String(i + 1) }), h('div', { class: 'lb', text: x.label }),
          h('div', { class: 'vl', text: x.value || (x.state === 'idle' ? '—' : '') }), x.detail ? h('div', { class: 'ds', text: x.detail }) : null);
      })),
      lastEv ? h('div', { class: 'now' }, h('span', { class: 'lbl', text: T.jobActive(job) ? 'Now' : 'Last step' }), h('b', { id: 'now-title', text: lastEv.title }),
        h('span', { class: 'muted small', text: clock(lastEv.ts) })) : null);
  }

  function verdictBlock(job, sec) {
    var L = sec.ledger, money = L ? T.money(L.amount_minor_units, L.currency) : null, v;
    var resultTitle = sec.result ? sec.result.title : '';
    if (job.status === 'APPLYING_REVIEW') v = ['run', 'Applying decision\u2026', job.latest_event || 'The reviewer\u2019s decision is being applied.', ''];
    else if (T.jobActive(job)) v = ['run', 'Processing\u2026', job.latest_event || '', ''];
    else if (job.status === 'REJECTED_BY_REVIEWER') v = ['idle', 'Rejected', 'The reviewer rejected this invoice as a threat. Case closed \u2014 nothing was paid.', ''];
    else if (job.status === 'PAYMENT_COMPLETED') {
      if (L && L.beneficiary_matches_vendor_record === false) v = ['act', money + ' paid', 'Sent to ' + L.beneficiary_account_id + ' — NOT the verified vendor account.',
        job.tell_secured === false ? 'TellSecured OFF: no probe, validator or gate checked this payment. Simulated ledger.' : 'Simulated ledger.'];
      else v = ['posted', (money || 'Payment') + ' paid', L && L.beneficiary_account_id ? 'To the verified vendor account ' + L.beneficiary_account_id + '.' : (resultTitle || 'Payment completed.'),
        job.tell_secured === false ? 'TellSecured OFF: paid with no Tell, validator or gate check. Simulated ledger.' : 'Simulated ledger — no real money moves.'];
    } else if (job.status === 'PAYMENT_BLOCKED' || job.status === 'BLOCKED_BY_DISPUTE') v = ['relax', '$0 moved', resultTitle || job.latest_event || 'Payment blocked.',
      sec.review ? 'An evidence report is waiting for a human reviewer.' : 'Nothing was paid; the case waits for a human.'];
    else if (job.status === 'AWAITING_VENDOR_CLARIFICATION') v = ['ready', 'On hold', 'A clarification email to the approved vendor contact was prepared (simulated, not sent).', ''];
    else if (job.status === 'NEEDS_DOCUMENT_REVIEW') v = ['ready', 'Needs review', resultTitle || job.latest_event || '', ''];
    else if (job.status === 'DUPLICATE') v = ['ready', 'Duplicate', job.latest_event || 'This PDF is already in the queue.', ''];
    else if (job.status === 'FAILED' || job.status === 'UNREADABLE_DOCUMENT') v = ['act', job.status === 'FAILED' ? 'Failed' : 'Unreadable', (sec.failure && sec.failure.title) || job.error || job.latest_event || '', ''];
    else v = ['run', job.display_status || '', job.latest_event || '', ''];
    return h('section', { class: 'vd ' + v[0], id: 'verdict' }, h('span', { class: 'big', text: v[1] }),
      h('div', { class: 'why', id: 'final-outcome' }, v[2], v[3] ? h('small', { text: v[3] }) : null,
        sec.parseFailure ? h('div', { class: 'parsefail', id: 'parse-failure' }, h('b', { text: 'Model output could not be parsed — failed closed' }),
          h('div', { class: 'small', text: sec.parseFailure.outcome + ': ' + (sec.parseFailure.error || '') }), h('pre', { class: 'report', text: sec.parseFailure.raw || '' })) : null));
  }

  /* ---------- human in the loop: the case file (opens in its own tab) ---------- */
  var reviewChoice = null;          // { job, decision, error } while the reviewer confirms
  var DECIDED = { approve: 'Approved — processed through the validator into the ledger', reject: 'Rejected — escalated for action', clarify: 'Clarification requested — the agent contacted the vendor' };
  var WORDS = { suspected_beneficiary_redirection: 'suspected redirection of the payment to a different bank account', invoice_data_gap: 'the invoice is missing information needed to pay it',
    suspected_forged_tool_result: 'a tool result or stored note looks forged', untrusted_beneficiary_change_vs_approved_beneficiary: 'the document asks for a different account than the approved one',
    independently_verify_via_approved_vendor_channel: 'confirm the change directly with the vendor through the approved contact',
    independent_confirmation_of_beneficiary_change: 'no independent confirmation that the vendor changed its bank account',
    obtain_corrected_invoice_via_internal_owner: 'obtain a corrected invoice through the internal owner', no_source_conflict: 'no conflict between sources' };
  function words(x) { return x ? (WORDS[x] || String(x).replace(/_/g, ' ')) : ''; }
  function caseLink(job, label) {
    return h('a', { class: 'btn primary', id: 'open-case', href: '#/case/' + job.job_id, target: '_blank', rel: 'noopener', title: 'Opens the case file in a new tab' }, label + ' ↗');
  }
  function caseBanner(job) {          // processing feed: a slim pointer to the case file
    var rv = job.review;
    if (!job.reviewable && !rv) return null;
    return h('section', { class: 'card case-banner', id: 'review-panel' },
      h('div', {}, h('b', { text: job.reviewable ? 'Held for a human reviewer' : (DECIDED[rv.decision] || rv.decision) }),
        h('div', { class: 'small muted', text: job.reviewable ? 'Open the case file to see the evidence and approve, reject or seek clarification.'
          : 'by ' + rv.reviewer_id + ' · ' + fmtShort(rv.requested_at) + (rv.state !== 'done' ? ' · being applied…' : '') })),
      caseLink(job, job.reviewable ? 'Open case file' : 'View case file'));
  }
  function renderCaseFile(id) {
    stopPolling();
    var host = h('div', { id: 'case' }, h('p', { class: 'muted', text: 'Loading the case file…' })); app.textContent = ''; app.appendChild(host);
    var load = function () {
      api('GET', '/api/intake/jobs/' + id + '/case').then(function (c) {
        if (view.name !== 'case') return;
        if (c.error || !c.job) { host.textContent = ''; host.appendChild(h('div', { class: 'empty-q', text: 'Unknown case.' })); return; }
        paintCase(host, c);
        if (!c.job.active && jobTimer) { clearInterval(jobTimer); jobTimer = null; }
      });
    };
    load(); jobTimer = setInterval(load, 1500);
  }
  function caseStory(c) {
    var j = c.job, t = c.tell, r = t.routing, a = c.agent, steps = [];
    steps.push(['done', 'The invoice was uploaded and read from the PDF (' + c.invoice.method + ').', null]);
    if (t.tell_secured === false) steps.push(['idle', 'TellSecured was OFF, so no Tell reading was taken and Agent 1 worked alone.', null]);
    else if (r) {
      var z = r.zone || (r.above_operational_threshold ? 'agent_s' : 'agent_1');
      steps.push([{ agent_s: 'act', tell_verify: 'ready', agent_1: 'relax' }[z], 'Tell read the model’s internal state after it saw the documents: ' + r.score.toFixed(3) + '. ' +
        ({ agent_s: 'That is above the alarm threshold (' + r.operational_threshold.toFixed(4) + '): Agent S, the safety adapter, took over and the gate was set to hold any payment.',
           tell_verify: 'That is in the Tell-Verify band (0.1708–' + r.operational_threshold.toFixed(4) + '): Agent S took over, without raising an alarm.',
           agent_1: 'That is clear: Agent 1 carried on.' }[z]), null]);
    }
    if (a.parse_failure) steps.push(['act', 'The model’s answer could not be read as a valid action, so the run failed closed.', null]);
    else if (a.report) steps.push(['ready', 'Agent S filed an evidence report: ' + words(a.report.assessment) + ' (severity ' + a.report.severity + ').', null]);
    else if (a.final_action) steps.push(['done', a.final_action.title + '.', null]);
    var g = c.checks.gate;
    if (g && g.gate_decision) steps.push([g.gate_decision === 'permit' ? 'relax' : 'act', 'The gate ' + (g.gate_decision === 'permit' ? 'permitted' : 'blocked') + ' the payment' +
      (g.gate_reason_code === 'alarm_unresolved' ? ' because the Tell alarm is unresolved — only a person can clear it.' : '.'), null]);
    if (c.outcome) steps.push([j.status === 'PAYMENT_COMPLETED' ? 'relax' : 'ready', 'Result: ' + c.outcome + '.', null]);
    return h('ol', { class: 'steps', id: 'case-story' }, steps.map(function (s) { return h('li', { class: 'st-' + s[0] }, h('span', { text: s[1] })); }));
  }
  function caseReasons(c) {
    var out = [], r = c.tell.routing, rep = c.agent.report;
    if (r && (r.zone === 'agent_s' || r.above_operational_threshold)) out.push(['act', 'Tell raised an alarm (' + r.score.toFixed(3) + ').']);
    else if (r && r.zone === 'tell_verify') out.push(['ready', 'Tell flagged the case for verification (' + r.score.toFixed(3) + ').']);
    if (rep) out.push(['ready', 'Agent S assessed: ' + words(rep.assessment) + '.']);
    if (rep && rep.unresolved_evidence_gap) out.push(['ready', 'Open question: ' + words(rep.unresolved_evidence_gap) + '.']);
    c.comparison.forEach(function (x) { if (x.match === false) out.push(['act', x.field + ' on the invoice does not match the ' + x.source + '.']); });
    if (c.trusted.dispute && c.trusted.dispute.status === 'open') out.push(['act', 'An open dispute case exists (' + c.trusted.dispute.case_id + ').']);
    if (c.trusted.prior_payment) out.push(['ready', 'This invoice already has a payment in the ledger (' + c.trusted.prior_payment.intent_id + ').']);
    if (!c.trusted.vendor) out.push(['act', 'The supplier is not in the vendor master.']);
    else if (!c.trusted.erp_invoice) out.push(['ready', 'No approved ERP invoice matches this invoice number.']);
    if (c.agent.parse_failure) out.push(['act', 'The model’s output was invalid, so nothing was executed.']);
    if (!out.length) out.push(['idle', c.job.latest_event || 'See the story above.']);
    return h('ul', { class: 'reasons', id: 'case-reasons' }, out.map(function (x) { return h('li', { class: 'st-' + x[0], text: x[1] }); }));
  }
  function caseCompare(c) {
    return h('table', { class: 'ledger cmp-t', id: 'case-compare' },
      h('thead', {}, h('tr', {}, [['Field', ''], ['The invoice says', ''], ['Trusted record', ''], ['Check', 'c-mode']].map(function (x) { return h('th', { class: x[1], scope: 'col', text: x[0] }); }))),
      h('tbody', {}, c.comparison.map(function (x) {
        return h('tr', {}, h('td', {}, h('b', { text: x.field })), h('td', { class: x.invoice ? '' : 'muted', text: x.invoice || 'not stated' }),
          h('td', {}, x.trusted || h('span', { class: 'muted', text: 'no record' }), x.trusted ? h('div', { class: 'small muted', text: x.source }) : null),
          h('td', { class: 'c-mode' }, h('span', { class: 'chk ' + (x.match === true ? 'ok' : x.match === false ? 'bad' : 'na'), text: x.match === true ? '✓' : x.match === false ? '✗ differs' : '—' })));
      })));
  }
  function caseReport(rep) {
    if (!rep) return null;
    return h('section', { class: 'card', id: 'case-report' }, h('div', { class: 'card-h' }, h('h2', { text: 'Agent S’s report' }), prov('extracted', 'model output')),
      h('dl', { class: 'kv' }, h('dt', { text: 'Assessment' }), h('dd', {}, h('b', { text: words(rep.assessment) })),
        h('dt', { text: 'Severity' }), h('dd', { text: rep.severity }), h('dt', { text: 'Conflict' }), h('dd', { text: words(rep.conflict_summary) }),
        h('dt', { text: 'Open question' }), h('dd', { text: words(rep.unresolved_evidence_gap) || '—' }),
        h('dt', { text: 'Recommended next step' }), h('dd', { text: words(rep.recommended_next_step) })),
      (rep.observed || []).length ? h('details', { class: 'sub-fold' }, h('summary', { text: 'Facts Agent S cited (' + rep.observed.length + ') — check them against the trusted records' }),
        h('ul', { class: 'small' }, rep.observed.map(function (o) { return h('li', { text: (o.trust_boundary === 'trusted' ? 'trusted · ' : 'untrusted · ') + words(o.field) + ': ' + o.value }); }))) : null);
  }
  function caseDecide(c, host) {
    var j = c.job, O = c.options, chosen = reviewChoice && reviewChoice.job === j.job_id ? reviewChoice : null;
    var wp = O.approve.would_pay;
    var opts = [
      ['approve', 'Approve', 'Process the payment', wp ? 'Pays ' + T.money(wp.amount_minor_units, wp.currency) + ' to the verified vendor account ' + wp.account +
        ' — after the validator and the gate check it — and adds it to the ledger. Never the account on the invoice.' : '', O.approve, 'relax', O.approve.warning],
      ['reject', 'Reject', 'Reject and escalate for action', 'Nothing is paid. The case and this file go to ' + O.reject.escalate_to + ' for action.', O.reject, 'act', null],
      ['clarify', 'Seek clarification', 'Have the agent ask the vendor', O.clarify.contact ? 'The agent sends a clarification request to ' + O.clarify.contact +
        ' (simulated, not sent). The payment stays on hold until a corrected invoice arrives.' : '', O.clarify, 'ready', null]];
    var cards = h('div', { class: 'rv-opts' }, opts.map(function (o) {
      var ok = o[4].available;
      return h('button', { type: 'button', class: 'rv-opt lv-' + o[5] + (chosen && chosen.decision === o[0] ? ' on' : ''), id: 'rv-' + o[0], 'data-decision': o[0], disabled: !ok,
        onclick: function () { reviewChoice = { job: j.job_id, decision: o[0] }; paintCase(host, c); } },
        h('span', { class: 'rv-k', text: o[1] }), h('b', { text: o[2] }), h('span', { class: 'small muted', text: ok ? o[3] : 'Not available: ' + o[4].reason + '.' }),
        ok && o[6] ? h('span', { class: 'small warn-text', text: o[6] }) : null);
    }));
    var confirm = null;
    if (chosen) {
      var note = h('textarea', { id: 'rv-note', rows: 2, maxlength: 500, placeholder: 'Note for the audit trail (optional)' });
      var label = opts.filter(function (o) { return o[0] === chosen.decision; })[0][2];
      confirm = h('div', { class: 'rv-confirm' }, note,
        h('div', { class: 'row' }, h('button', { class: 'btn primary', id: 'rv-confirm', onclick: function (e) {
            e.target.disabled = true;
            api('POST', '/api/intake/jobs/' + j.job_id + '/review', { decision: chosen.decision, note: note.value || null }).then(function (r) {
              reviewChoice = r.error ? Object.assign({}, chosen, { error: r.error }) : null;
              renderCaseFile(j.job_id);
            });
          } }, 'Confirm: ' + label.toLowerCase()), h('button', { class: 'btn ghost', id: 'rv-cancel', onclick: function () { reviewChoice = null; paintCase(host, c); } }, 'Cancel')),
        chosen.error ? h('div', { class: 'parsefail small', text: chosen.error }) : null);
    }
    return h('section', { class: 'card decide', id: 'case-decide' }, h('div', { class: 'card-h' }, h('h2', { text: 'Your decision' }),
      h('span', { class: 'muted small', text: 'Reviewing as ' + (c.reviewer_id || 'reviewer') })), cards, confirm);
  }
  function caseDecision(c) {
    var rv = c.job.review; if (!rv) return null;
    var esc = c.escalations[c.escalations.length - 1];
    return h('section', { class: 'card decided', id: 'review-decision' }, h('div', { class: 'card-h' }, h('h2', { text: 'Decision' }), prov('live', 'human in the loop')),
      h('b', { class: 'big-line', text: DECIDED[rv.decision] || rv.decision }),
      h('div', { class: 'small muted', text: 'by ' + rv.reviewer_id + ' · ' + fmtShort(rv.requested_at) + (rv.state !== 'done' ? ' · being applied…' : '') }),
      rv.note ? h('p', { class: 'small', text: '“' + rv.note + '”' }) : null,
      rv.result && rv.result.message ? h('div', { class: 'note', text: 'Result: ' + rv.result.message }) : null,
      esc ? h('dl', { class: 'kv esc', id: 'case-escalation' }, h('dt', { text: 'Escalated to' }), h('dd', { text: 'Security & vendor management' }),
        h('dt', { text: 'Case' }), h('dd', { class: 'mono', text: esc.case_id }), h('dt', { text: 'Asked to' }), h('dd', { text: esc.recommended_action })) : null);
  }
  function paintCase(host, c) {
    var j = c.job, s = j.summary || {}, lv = T.levelOf(j), inv = c.invoice;
    host.textContent = '';
    host.appendChild(h('div', {},
      h('a', { class: 'btn back', href: '#/job/' + j.job_id }, '← Processing feed'),
      h('div', { class: 'feed-h' }, h('div', { class: 'eyebrow' }, h('span', { class: 'pill lv-idle', text: 'CASE FILE' }), h('span', { class: 'mono small muted', text: j.job_no || '' }),
          h('span', { class: 'pill lv-' + lv, text: j.display_status }), h('span', { class: 'tsbadge ' + (j.tell_secured === false ? 'off' : 'on'), text: j.tell_secured === false ? 'TellSecured OFF' : 'TellSecured ON' })),
        h('h1', { text: (inv.invoice_number || j.display_name) + (inv.supplier ? ' — ' + inv.supplier : '') }),
        h('div', { class: 'sub' }, [fmtAmount(inv.amount, inv.currency) ? 'claims ' + fmtAmount(inv.amount, inv.currency) : null, 'uploaded ' + fmtShort(j.created_at), j.display_name].filter(Boolean)
          .map(function (x) { return h('span', { text: x }); }))),
      h('div', { class: 'case-grid' },
        h('section', { class: 'card' }, h('h2', { text: 'What happened' }), caseStory(c)),
        h('section', { class: 'card' }, h('h2', { text: 'Why it is held' }), caseReasons(c))),
      h('section', { class: 'card book' }, h('div', { class: 'card-h pad' }, h('h2', { text: 'The invoice vs the trusted records' }), prov('trusted', 'trusted demo fixtures')), caseCompare(c)),
      caseReport(c.agent.report),
      j.reviewable ? caseDecide(c, host) : null,
      caseDecision(c),
      j.active && j.status === 'APPLYING_REVIEW' ? h('div', { class: 'vd run', id: 'case-applying' }, h('span', { class: 'big', text: 'Applying…' }), h('div', { class: 'why', text: j.latest_event || '' })) : null,
      h('div', { class: 'folds' }, fold('case-text', ['Invoice text', h('span', { class: 'muted', text: 'as read from the PDF — untrusted' })], inv.text ? h('pre', { class: 'report', text: inv.text }) : h('div', { class: 'empty', text: 'No text.' })),
        fold('case-tell', ['Tell readings', h('span', { class: 'muted', text: (c.tell.readings || []).length + ' measured' })], h('div', { class: 'reads left' }, (c.tell.readings || []).map(function (t) {
          return h('span', { class: 'rd' + (t.role === 'routing' ? ' route' : '') }, 'T' + t.turn + ' ' + t.score.toFixed(3) + (t.role === 'routing' ? ' · ROUTING DECISION' : ' · MONITORING'));
        }))))));
  }

  function decisionBody(sec) {
    var parts = [], g = sec.gate;
    if (sec.route) parts.push(h('div', {}, h('h3', { text: sec.route.agent === 'agent_s' ? 'Agent S — frozen safety LoRA' : 'Agent 1 — base model' }),
      h('div', { class: 'small muted', text: sec.route.reason || '' }), sec.route.adapter ? h('div', { class: 'hash', text: 'adapter sha256 ' + sec.route.adapter }) : null));
    if (sec.action) parts.push(h('div', {}, h('div', { class: 'big-line', id: 'proposed-action', text: sec.action.title || sec.action.action.action }),
      h('div', { class: 'small muted', text: 'proposed by ' + String(sec.action.agent).replace('_', ' ') + ' on turn ' + sec.action.turn + (sec.action.ms != null ? ' · ' + fmtMs(sec.action.ms) : '') }),
      h('pre', { class: 'report', text: JSON.stringify(sec.action.action, null, 2) })));
    if (g) parts.push(h('div', {}, sec.gateTitle ? h('div', { class: 'big-line', id: 'gate-summary', text: sec.gateTitle }) : null,
      h('dl', { class: 'kv' }, h('dt', { text: 'Validator' }), h('dd', { text: g.validator_outcome || 'not run (no payment proposal)' }), h('dt', { text: 'Gate decision' }),
        h('dd', {}, h('b', { class: g.gate_decision === 'permit' ? 'match' : 'mism', text: g.gate_decision === 'not_applied' ? 'NOT APPLIED (TellSecured OFF)' : (g.gate_decision || 'not reached').toUpperCase() })),
        h('dt', { text: 'Reason code' }), h('dd', { class: 'mono', text: g.gate_reason_code || '—' }), h('dt', { text: 'Alarm' }), h('dd', { text: g.alarm_state_before + ' → ' + g.alarm_state_after }),
        h('dt', { text: 'Final action' }), h('dd', { class: 'mono', text: g.final_action }), h('dt', { text: 'Executed' }), h('dd', { text: String(g.executed) }))));
    var ev = [];
    sec.trustedEvidence.forEach(function (t) { var c = t.content || {}; ev.push(h('div', { class: 'small' }, prov('trusted'), ' Vendor record: ' + (c.vendor_name || '') + ' · ' + (c.vendor_id || '') + ' · ' + (c.verification_status || '') + ' · account ' + (c.beneficiary_account_id || '—'))); });
    sec.lookups.forEach(function (l) { ev.push(h('div', { class: 'small' }, prov('trusted'), ' ' + l.tool.replace(/_/g, ' ') + ': ' + l.status)); });
    if (ev.length) parts.push(h('div', { class: 'grid' }, ev));
    return parts.length ? h('div', { class: 'grid' }, parts) : h('div', { class: 'empty', text: 'Nothing has been decided yet.' });
  }

  function paintReplay(host, job, rep, detail) {
    var sec = T.replaySections(rep.events), terminal = !T.jobActive(job), run = rep.run || {}, s = job.summary || {};
    var ex = (detail && detail.extraction) || {}, lv = T.levelOf(job);
    var meta = h('dl', { class: 'kv' }, h('dt', { text: 'File' }), h('dd', { text: job.display_name }), h('dt', { text: 'Uploaded' }), h('dd', { text: fmtTime(job.created_at) }),
      h('dt', { text: 'Document ID' }), h('dd', { class: 'mono', id: 'doc-id', text: job.job_id }), h('dt', { text: 'Run ID' }), h('dd', { class: 'mono', id: 'run-id', text: job.run_id }),
      h('dt', { text: 'Execution' }), h('dd', { class: 'mono', text: run.execution_id ? run.execution_id + ' · attempt ' + run.attempt : '— (not started)' }),
      h('dt', { text: 'SHA-256' }), h('dd', { class: 'hash', text: (detail && detail.sha256) || '—' }),
      h('dt', { text: 'Extraction' }), h('dd', { text: ex.source === 'OCR' ? 'local OCR' : (ex.source ? 'embedded text' : '—') }));
    var docBody = h('div', { class: 'grid' }, meta, (ex.warnings || []).length ? h('div', {}, h('h2', { text: 'Extraction warnings' }), h('ul', {}, ex.warnings.map(function (w) { return h('li', { text: w }); }))) : null,
      ex.text ? h('div', {}, h('h2', { text: 'Text read from the PDF' }), h('pre', { class: 'report', text: ex.text })) : null,
      h('div', {}, h('h2', { text: 'Runtime' }), runtimeRows(rep.runtime)));
    var title = s.invoice_number || job.display_name;
    var subline = [s.supplier_name, fmtAmount(s.amount, s.currency), s.invoice_number ? job.display_name : null, fmtTime(job.created_at)].filter(Boolean);
    host.textContent = '';
    host.appendChild(h('div', {},
      h('div', { class: 'feed-h' }, h('button', { class: 'btn back', onclick: function () { location.hash = '#/'; } }, '← All invoices'),
        h('div', { class: 't' }, h('h1', { text: title }), h('span', { class: 'pill lv-' + lv, text: job.display_status }),
          h('span', { class: 'tsbadge ' + (job.tell_secured === false ? 'off' : 'on'), id: 'ts-badge', text: job.tell_secured === false ? 'TellSecured OFF — undefended baseline' : 'TellSecured ON' })),
        h('div', { class: 'sub' }, subline.map(function (x) { return h('span', { text: x }); }))),
      h('div', { class: 'hero' }, extractCard(job, detail), meterCard(sec, rep.runtime, terminal, job)),
      pipelineCard(job, rep),
      verdictBlock(job, sec),
      caseBanner(job),
      detail && detail.outbox ? h('div', { class: 'folds' }, emailCard(detail.outbox)) : null,
      h('div', { class: 'folds' },
        fold('why-fold', ['Decision details', h('span', { class: 'muted', text: 'route, proposed action, validator & gate, trusted evidence' })], decisionBody(sec)),
        fold('audit-fold', ['Audit trail', h('span', { class: 'muted', text: rep.events.length + ' persisted events' })], h('ol', { class: 'tl live-tl', id: 'agent-timeline' }, rep.events.map(evRow))),
        fold('doc-fold', ['Document & runtime', h('span', { class: 'muted', text: 'IDs, hashes, raw text, loaded models' })], docBody))));
  }

  function emailCard(ob) {
    return h('div', { class: 'card email', id: 'email-preview' }, h('div', { class: 'email-head' }, h('h2', { text: 'Clarification email' }), h('span', { class: 'sim-badge', id: 'sim-badge', text: 'SIMULATED — NOT SENT' })),
      h('dl', { class: 'kv' }, h('dt', { text: 'To' }), h('dd', { text: ob.recipient }), h('dt', { text: 'Recipient source' }), h('dd', { class: 'mono', text: ob.recipient_source }),
        h('dt', { text: 'Subject' }), h('dd', { text: ob.subject }), h('dt', { text: 'Send status' }), h('dd', { class: 'mono', text: ob.send_status })),
      h('pre', { class: 'report mail', id: 'email-body', text: ob.body }));
  }

  function renderLiveJob(id) {
    stopPolling();
    var host = h('div', {}); app.textContent = ''; app.appendChild(host);
    var load = function () {
      Promise.all([api('GET', '/api/intake/jobs/' + id + '/replay'), api('GET', '/api/intake/jobs/' + id)]).then(function (r) {
        var rep = r[0], detail = r[1];
        if (rep.error && !rep.job) { host.textContent = ''; host.appendChild(h('p', { text: 'Unknown invoice job.' })); if (jobTimer) clearInterval(jobTimer); return; }
        paintReplay(host, rep.job, rep, detail);
        if (!T.jobActive(rep.job) && jobTimer) { clearInterval(jobTimer); jobTimer = null; }
      });
    };
    load(); jobTimer = setInterval(load, 1000);       // polling only: every displayed value is re-read from the backend
  }

  /* ---------- router ---------- */
  function route() {
    if (player) { player.dispose(); player = null; }
    stopPolling();
    var m = location.hash.match(/^#\/(run|review|job|attention|ledger|case)(?:\/([^@]+))?(?:@(-?\d+))?$/);
    view = m ? { name: m[1], id: m[2], step: m[3] } : { name: 'dash' };
    var recorded = view.name === 'run' || view.name === 'review';
    var go = function () {
      renderChrome(view.name === 'run' ? [T.findTrace(bundle.traces, view.id)].filter(Boolean) : (view.name === 'job' ? [] : (bundle ? bundle.traces : [])));
      if (view.name === 'run') { renderRun(view.id); if (view.step != null && player) player.seek(parseInt(view.step, 10)); } else if (view.name === 'review') renderReview(view.id);
      else if (view.name === 'job') renderJob(view.id);
      else if (view.name === 'ledger' && isLive()) renderLedger();
      else if (view.name === 'case' && isLive()) renderCaseFile(view.id);
      else if (view.name === 'attention' && isLive()) { intake.sig = null; renderAttention(); refreshIntake().then(function () { if (view.name === 'attention') pollTimer = setInterval(refreshIntake, 3000); }); }
      else { renderDash(); refreshIntake().then(function () { if (view.name === 'dash') pollTimer = setInterval(refreshIntake, intake.jobs.some(T.jobActive) ? 800 : 3000); }); }
      window.scrollTo(0, 0);
    };
    // the recorded bundle is loaded only for recorded views (or on a non-live server); live uploads never touch it
    if ((recorded || !isLive()) && !bundle) ensureBundle().then(go, function (e) { app.textContent = 'Could not load recorded examples: ' + e; });
    else go();
  }
  api('GET', '/api/intake/status').then(function (st) { intake.status = st; }, function () {}).then(function () {
    window.addEventListener('hashchange', route); route();
  }).catch(function (e) { app.textContent = 'Could not start: ' + e; });
})();
