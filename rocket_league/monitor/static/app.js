// ---------------------------------------------------------------------------
// Chart factory
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
  // Performance
  timing:  mkMulti('c-timing',
    ['Collection', 'Env Step', 'Inference', 'PPO Learn'],
    ['#4fc3f7', '#81c784', '#ff8a65', '#ce93d8']),
  goals:   mkChart('c-goals', '#ef5350'),
};

let fetchedCount = 0;
let initialized = false;

// ---------------------------------------------------------------------------
// Formatting
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
// Toast notifications
// ---------------------------------------------------------------------------

function toast(msg, ok = true) {
  const el = document.createElement('div');
  el.className = 'toast ' + (ok ? 'toast-ok' : 'toast-err');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// ---------------------------------------------------------------------------
// Data ingestion
// ---------------------------------------------------------------------------

function pushPt(chart, dsIdx, ts, val) {
  if (val == null) return;
  if (dsIdx === 0) chart.data.labels.push(ts);
  chart.data.datasets[dsIdx].data.push(val);
}

function addDataPoint(d) {
  const ts = ((d.total_timesteps || 0) / 1e6).toFixed(2);

  // Training
  if (d.avg_reward != null)    pushPt(charts.reward,  0, ts, d.avg_reward);
  if (d.entropy != null)       pushPt(charts.entropy, 0, ts, d.entropy);
  if (d.sps != null)           pushPt(charts.sps,     0, ts, d.sps);
  if (d.policy_update != null || d.critic_update != null) {
    charts.updates.data.labels.push(ts);
    charts.updates.data.datasets[0].data.push(d.policy_update ?? null);
    charts.updates.data.datasets[1].data.push(d.critic_update ?? null);
  }

  // Gameplay
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

  // Performance timing
  if (d.collection_time != null || d.env_step_time != null ||
      d.inference_time != null || d.ppo_learn_time != null) {
    charts.timing.data.labels.push(ts);
    charts.timing.data.datasets[0].data.push(d.collection_time ?? null);
    charts.timing.data.datasets[1].data.push(d.env_step_time ?? null);
    charts.timing.data.datasets[2].data.push(d.inference_time ?? null);
    charts.timing.data.datasets[3].data.push(d.ppo_learn_time ?? null);
  }

  // Goals
  if (d.goal_speed != null) pushPt(charts.goals, 0, ts, d.goal_speed);
}

function clearCharts() {
  Object.values(charts).forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(ds => { ds.data = []; });
    c.update();
  });
  fetchedCount = 0;
  initialized = false;
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
// Bot management
// ---------------------------------------------------------------------------

async function loadBots() {
  try {
    const r = await fetch('/api/bots');
    const j = await r.json();
    const sel = document.getElementById('bot-select');
    sel.innerHTML = j.bots.map(b => {
      const ts = b.latest_timesteps >= 1e6
        ? (b.latest_timesteps / 1e6).toFixed(1) + 'M'
        : b.latest_timesteps > 0
        ? (b.latest_timesteps / 1e3).toFixed(0) + 'K'
        : 'new';
      return `<option value="${b.name}" ${b.name === j.current ? 'selected' : ''}>`
        + `${b.name} (${ts})</option>`;
    }).join('');
    if (j.bots.length === 0) {
      sel.innerHTML = '<option value="default">default (new)</option>';
    }
  } catch (e) { /* server not ready */ }
}

async function selectBot(name) {
  const r = await fetch('/api/bots/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const j = await r.json();
  if (j.ok) {
    clearCharts();
    toast('Switched to bot: ' + name);
    // Reload metrics for new bot
    poll();
  }
}

async function createBot() {
  const name = prompt('Enter bot name (letters, numbers, hyphens):');
  if (!name) return;
  const r = await fetch('/api/bots/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const j = await r.json();
  if (j.ok) {
    toast('Created bot: ' + j.name);
    await loadBots();
    await selectBot(j.name);
  } else {
    toast(j.error, false);
  }
}

async function deleteBot() {
  const name = document.getElementById('bot-select').value;
  if (!name) return;
  if (!confirm('Delete bot "' + name + '" and ALL its checkpoints/metrics? This cannot be undone.')) return;
  const r = await fetch('/api/bots/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const j = await r.json();
  if (j.ok) {
    toast('Deleted bot: ' + name);
    clearCharts();
    await loadBots();
    poll();
  } else {
    toast(j.error, false);
  }
}

// ---------------------------------------------------------------------------
// Training controls
// ---------------------------------------------------------------------------

async function doStart() {
  const bot = document.getElementById('bot-select').value;
  const r = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bot }),
  });
  const j = await r.json();
  if (j.ok) {
    toast('Training started: ' + (j.bot || bot));
  } else {
    toast(j.error, false);
  }
}

async function doStop() {
  if (!confirm('Save checkpoint and stop training?')) return;
  const r = await fetch('/api/stop', { method: 'POST' });
  const j = await r.json();
  if (!j.ok) toast(j.error, false);
  else toast('Saving and stopping...');
}

async function doKill() {
  if (!confirm('Kill training immediately WITHOUT saving?')) return;
  const r = await fetch('/api/kill', { method: 'POST' });
  const j = await r.json();
  if (!j.ok) toast(j.error, false);
  else toast('Process killed');
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
// Quick actions
// ---------------------------------------------------------------------------

async function openRewards() {
  const r = await fetch('/api/open-rewards', { method: 'POST' });
  const j = await r.json();
  if (j.ok) toast('Opened main.cpp in editor');
  else toast(j.error, false);
}

async function openViz() {
  const r = await fetch('/api/open-viz', { method: 'POST' });
  const j = await r.json();
  if (j.ok) toast('Launched RocketSimVis');
  else toast(j.error, false);
}

async function buildRLBot() {
  const bot = document.getElementById('bot-select').value;
  toast('Choose export folder...');
  const r = await fetch('/api/build-rlbot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bot }),
  });
  const j = await r.json();
  if (j.ok) {
    toast('Exported to: ' + j.path);
  } else if (j.error === 'Export cancelled') {
    toast('Export cancelled', false);
  } else {
    toast(j.error, false);
  }
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

    // Checkpoints (less frequent)
    if (!poll._ckptCounter) poll._ckptCounter = 0;
    if (poll._ckptCounter++ % 5 === 0) {
      const cr = await fetch('/api/checkpoints');
      const cj = await cr.json();
      renderCheckpoints(cj);
    }
  } catch (e) { /* server not ready */ }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadBots();
setInterval(poll, 2000);
poll();
