/* Login form. Contains no secret; posts the passcode over the same origin and lets the server set the session cookie. */
(function () {
  'use strict';
  var form = document.getElementById('login'), input = document.getElementById('passcode'),
    btn = document.getElementById('submit'), err = document.getElementById('error');
  function fail(msg) { err.textContent = msg; err.hidden = false; btn.disabled = false; input.select(); }
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!input.value) return fail('Enter the passcode.');
    btn.disabled = true; err.hidden = true;
    fetch('/auth/login', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-Tell-Intake': '1' },
      body: JSON.stringify({ passcode: input.value }) })
      .then(function (r) { return r.json().then(function (j) { return { status: r.status, body: j }; }); })
      .then(function (r) {
        input.value = '';
        if (r.status === 200) { location.replace('/'); return; }
        fail(r.status === 429 ? 'Too many attempts. Please wait a few minutes and try again.' : 'That passcode is not correct.');
      })
      .catch(function () { fail('Could not reach the server. Please try again.'); });
  });
})();
