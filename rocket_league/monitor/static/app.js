// ---------------------------------------------------------------------------
// Chart factory — maintainAspectRatio:false lets the container div control
// height while Chart.js handles canvas DPI scaling (no pixelation).
// ---------------------------------------------------------------------------

const chartDefaults = {
  animation: false,
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { title: { display: true, text: 'Timesteps (M)', color: '#555' },
         ticks: { color: '#555', maxTicksLimit: 8 }, grid: { color: '#1e2028' } },
    y: { ticks: { color: '#555' }, grid: { color: '#1e2028' } }
  }
};

const mkChart = (id, color) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: color, borderWidth: 1.5,
           pointRadius: 0, fill: false, tension: 0.3 }] },
  options: { ...chartDefaults }
});

const mkMulti = (id, labels, colors) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: labels.map((l, i) => ({
    label: l, data: [], borderColor: colors[i], borderWidth: 1.5,
    pointRadius: 0, fill: false, tension: 0.3
  })) },
  options: {
    ...chartDefaults,
    plugins: { legend: { labels: { color: '#888', boxWidth: 10, font: { size: 10 } } } }
  }
});

// ---------------------------------------------------------------------------
// Chart instances
// ---------------------------------------------------------------------------

const charts = {
  // Training
  reward:  mkChart('c-reward', '#4fc3f7'),
  entropy: mkChart('c-entropy', '#ff8a65'),
  sps:     mkChart('c-sps', '#81c784'),
  updates: mkMulti('c-updates', ['Policy', 'Critic'], ['#ce93d8', '#fff176']),
  // Gameplay
  touch:   mkChart('c-touch', '#e6ee9c'),
  speed:   mkMulti('c-speed', ['Speed', 'Speed to Ball'], ['#4fc3f7', '#81c784']),
  boost:   mkChart('c-boost', '#ffcc80'),
  aerial:  mkMulti('c-aerial', ['In Air %', 'Touch Height'], ['#ce93d8', '#ef9a9a']),
};

let fetchedCount = 0;
let initialized = false;

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function fmt(n) {
  if (n == null) return '-';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return typeof n === 'number' ? n.toFixed(n < 10 ? 4 : 1) : n;
}

function fmtPct(n) {
  if (n == null) return '-';
  return (n * 100).toFixed(1) + '%';
}

function fmtTime(s) {
  if (!s) return '-';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + sec + 's';
}

// ---------------------------------------------------------------------------
// Data ingestion
// ---------------------------------------------------------------------------

function pushPt(chart, dsIdx, ts, val) {
  if (val == null) return;
  // Ensure labels array has this timestep (use dataset 0's label array)
  const ds = chart.data.datasets[dsIdx];
  if (dsIdx === 0) chart.data.labels.push(ts);
  ds.data.push(val);
}

function addDataPoint(d) {
  const ts = ((d.total_timesteps || 0) / 1e6).toFixed(2);

  // Training charts
  if (d.avg_reward != null)    pushPt(charts.reward,  0, ts, d.avg_reward);
  if (d.entropy != null)       pushPt(charts.entropy, 0, ts, d.entropy);
  if (d.sps != null)           pushPt(charts.sps,     0, ts, d.sps);
  if (d.policy_update != null || d.critic_update != null) {
    charts.updates.data.labels.push(ts);
    charts.updates.data.datasets[0].data.push(d.policy_update ?? null);
    charts.updates.data.datasets[1].data.push(d.critic_update ?? null);
  }

  // Gameplay charts
  if (d.player_ball_touch != null) pushPt(charts.touch, 0, ts, d.player_ball_touch);

  if (d.player_speed != null || d.player_speed_to_ball != null) {
    charts.speed.data.labels.push(ts);
    charts.speed.data.datasets[0].data.push(d.player_speed ?? null);
    charts.speed.data.datasets[1].data.push(d.player_speed_to_ball ?? null);
  }

  if (d.player_boost != null) pushPt(charts.boost, 0, ts, d.player_boost);

  if (d.player_in_air != null || d.touch_height != null) {
    charts.aerial.data.labels.push(ts);
    charts.aerial.data.datasets[0].data.push(d.player_in_air ?? null);
    charts.aerial.data.datasets[1].data.push(d.touch_height ?? null);
  }
}

// ---------------------------------------------------------------------------
// Stat cards
// ---------------------------------------------------------------------------

function updateStats(last) {
  document.getElementById('s-iter').textContent  = last.total_iterations || '-';
  document.getElementById('s-ts').textContent    = fmt(last.total_timesteps);
  document.getElementById('s-rew').textContent   = last.avg_reward != null ? last.avg_reward.toFixed(4) : '-';
  document.getElementById('s-sps').textContent   = fmt(last.sps);
  document.getElementById('s-ent').textContent   = last.entropy != null ? last.entropy.toFixed(4) : '-';
  document.getElementById('s-touch').textContent = fmtPct(last.player_ball_touch);
  document.getElementById('s-boost').textContent = last.player_boost != null ? last.player_boost.toFixed(1) : '-';
  document.getElementById('s-time').textContent  = fmtTime(last.wall_time);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

async function doStart() {
  const r = await fetch('/api/start', { method: 'POST' });
  const j = await r.json();
  if (!j.ok) alert(j.error);
}
async function doStop() {
  if (!confirm('Save checkpoint and stop training?')) return;
  const r = await fetch('/api/stop', { method: 'POST' });
  const j = await r.json();
  if (!j.ok) alert(j.error);
}
async function doKill() {
  if (!confirm('Kill training immediately WITHOUT saving?')) return;
  const r = await fetch('/api/kill', { method: 'POST' });
  const j = await r.json();
  if (!j.ok) alert(j.error);
}

function updateControls(status) {
  const badge = document.getElementById('status-badge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status.toUpperCase();
  document.getElementById('btn-start').disabled = (status !== 'idle');
  document.getElementById('btn-stop').disabled  = (status !== 'running');
  document.getElementById('btn-kill').disabled   = (status === 'idle');
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

function renderCheckpoints(ckpts) {
  const list = document.getElementById('ckpt-list');
  if (!ckpts || ckpts.length === 0) {
    list.innerHTML = '<li class="ckpt-empty">No checkpoints yet</li>';
    return;
  }
  list.innerHTML = ckpts.reverse().map(c => {
    const ts = c.timesteps >= 1e6
      ? (c.timesteps / 1e6).toFixed(1) + 'M'
      : (c.timesteps / 1e3).toFixed(0) + 'K';
    const detail = c.mean_return != null
      ? 'iter ' + c.iterations + ' &middot; mean return ' + c.mean_return.toFixed(1)
        + ' &middot; ' + (c.episodes || 0).toLocaleString() + ' eps'
      : 'iter ' + c.iterations;
    return '<li class="ckpt-item">'
      + '<span><span class="ckpt-ts">' + ts + ' steps</span>'
      + '  <span class="ckpt-detail">' + detail + '</span></span>'
      + (c.has_model ? '<span class="ckpt-badge">MODEL</span>' : '')
      + '</li>';
  }).join('');
}

// ---------------------------------------------------------------------------
// Polling loop
// ---------------------------------------------------------------------------

async function poll() {
  try {
    const sr = await fetch('/api/status');
    const sj = await sr.json();
    updateControls(sj.status);

    const mr = await fetch('/api/metrics?since=' + fetchedCount);
    const mj = await mr.json();

    if (!initialized && mj.total > 0) {
      const note = document.getElementById('history-note');
      note.style.display = 'flex';
      document.getElementById('history-text').textContent =
        mj.total + ' iterations loaded from previous sessions';
    }
    initialized = true;

    for (const d of mj.data) addDataPoint(d);

    if (mj.data.length > 0) {
      Object.values(charts).forEach(c => c.update());
      updateStats(mj.data[mj.data.length - 1]);
    }
    fetchedCount = mj.total;

    // Log
    const lr = await fetch('/api/log');
    const lj = await lr.json();
    const box = document.getElementById('log-box');
    box.textContent = lj.lines.join('\n');
    box.scrollTop = box.scrollHeight;

    // Checkpoints (less frequent)
    if (!poll._ckptCounter) poll._ckptCounter = 0;
    if (poll._ckptCounter++ % 5 === 0) {
      const cr = await fetch('/api/checkpoints');
      const cj = await cr.json();
      renderCheckpoints(cj);
    }
  } catch (e) { /* server not ready */ }
}

setInterval(poll, 2000);
poll();
