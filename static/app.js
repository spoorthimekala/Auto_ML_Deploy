// Count-up numbers
document.querySelectorAll('[data-count]').forEach(el => {
  const end = parseFloat(el.dataset.count), dec = (el.dataset.count.split('.')[1] || '').length, t0 = performance.now();
  (function tick(t) {
    const p = Math.min((t - t0) / 900, 1);
    el.textContent = (end * (1 - Math.pow(1 - p, 3))).toFixed(dec);
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
});

// Upload page: drag & drop
const fi = document.getElementById('file'), dz = document.getElementById('dropzone');
if (fi && dz) {
  const show = () => {
    if (fi.files.length) {
      document.getElementById('fname').textContent = '' + fi.files[0].name;
      document.getElementById('upbtn').disabled = false;
    }
  };
  fi.addEventListener('change', show);
  ['dragenter', 'dragover'].forEach(e => dz.addEventListener(e, () => dz.classList.add('over')));
  ['dragleave', 'drop'].forEach(e => dz.addEventListener(e, () => dz.classList.remove('over')));
  dz.addEventListener('drop', ev => { fi.files = ev.dataTransfer.files; show(); });
}

// Configure page: live task badge + live training progress
const tf = document.getElementById('trainForm');
if (tf) {
  const sel = document.getElementById('target'), badge = document.getElementById('taskBadge');
  const tasks = JSON.parse(document.getElementById('taskData').textContent);
  const upd = () => { badge.textContent = 'Detected task: ' + tasks[sel.value]; badge.className = 'badge ' + tasks[sel.value]; };
  sel.addEventListener('change', upd); upd();

  const ov = document.getElementById('overlay'), bar = document.getElementById('bar'),
        pmsg = document.getElementById('pmsg'), log = document.getElementById('log'),
        err = document.getElementById('trainError');
  let last = '';
  async function poll() {
    const s = await (await fetch('/api/status')).json();
    bar.style.width = s.progress + '%';
    pmsg.textContent = s.message;
    if (s.message && s.message !== last) {
      last = s.message;
      const li = document.createElement('li'); li.textContent = s.message; log.appendChild(li);
    }
    if (s.state === 'done') { location.href = '/results'; return; }
    if (s.state === 'error') { ov.classList.remove('show'); err.textContent = s.error; return; }
    setTimeout(poll, 400);
  }
  tf.addEventListener('submit', async ev => {
    ev.preventDefault();
    err.textContent = ''; log.innerHTML = ''; last = ''; bar.style.width = '0%';
    ov.classList.add('show');
    const r = await fetch('/api/train', { method: 'POST', body: new FormData(tf) });
    if (!r.ok) { ov.classList.remove('show'); err.textContent = (await r.json()).error; return; }
    poll();
  });
}

// Predict page: live prediction as the user types
const pf = document.getElementById('predForm');
if (pf) {
  const out = document.getElementById('out'), perr = document.getElementById('predError');
  let timer;
  async function run() {
    const r = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.fromEntries(new FormData(pf)))
    });
    const j = await r.json();
    out.innerHTML = '';
    if (!r.ok) { perr.textContent = j.error; return; }
    perr.textContent = '';
    const big = document.createElement('div'); big.className = 'big pulse'; big.textContent = j.prediction;
    out.appendChild(big);
    if (j.probabilities) {
      Object.entries(j.probabilities).forEach(([cls, p]) => {
        const d = document.createElement('div'); d.className = 'bar';
        const row = document.createElement('div'); row.className = 'row';
        const a = document.createElement('span'); a.textContent = 'Class ' + cls;
        const b = document.createElement('span'); b.textContent = (p * 100).toFixed(1) + '%';
        row.append(a, b);
        const tr = document.createElement('div'); tr.className = 'track';
        const fl = document.createElement('div'); fl.className = 'fill'; fl.style.setProperty('--w', (p * 100) + '%');
        tr.appendChild(fl); d.append(row, tr); out.appendChild(d);
      });
    }
  }
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 350); };
  pf.addEventListener('input', schedule);
  pf.addEventListener('change', schedule);
  document.getElementById('resetBtn').addEventListener('click', () => { pf.reset(); schedule(); });
  run();
}