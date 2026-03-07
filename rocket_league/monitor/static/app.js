const mkChart = (id, color) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: color, borderWidth: 1.5,
           pointRadius: 0, fill: false, tension: 0.3 }] },
  options: {
    animation: false, responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: 'Timesteps (M)', color: '#555' },
           ticks: { color: '#555', maxTicksLimit: 8 }, grid: { color: '#1e2028' } },
      y: { ticks: { color: '#555' }, grid: { color: '#1e2028' } }
    }
  }
});
const mkMulti = (id, labels, colors) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: labels.map((l, i) => ({
    label: l, data: [], borderColor: colors[i], borderWidth: 1.5,
    pointRadius: 0, fill: false, tension: 0.3
  })) },
  options: {
    animation: false, responsive: true,
    plugins: { legend: { labels: { color: '#888', boxWidth: 10, font: { size: 10 } } } },
    scales: {
      x: { title: { display: true, text: 'Timesteps (M)', color: '#555' },
           ticks: { color: '#555', maxTicksLimit: 8 }, grid: { color: '#1e2028' } },
      y: { ticks: { color: '#555' }, grid: { color: '#1e2028' } }
    }
  }
});

const charts = {
  reward:  mkChart('c-reward', '#4fc3f7'),
  entropy: mkChart('c-entropy', '#ff8a65'),
  sps:     mkChart('c-sps', '#81c784'),
  updates: mkMulti('c-updates', ['Policy', 'Critic'], ['#ce93d8', '#fff176']),
};
let fetchedCount = 0;
let initialized = false;

function fmt(n) {
  if (n == null) return '-';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return typeof n === 'number' ? n.toFixed(n < 10 ? 4 : 1) : n;
}
function fmtTime(s) {
  if (!s) return '-';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + sec + 's';
}

function addDataPoint(d) {
  const ts = ((d.total_timesteps || 0) / 1e6).toFixed(2);
  if (d.avg_reward != null) {
    charts.reward.data.labels.push(ts);
    charts.reward.data.datasets[0].data.push(d.avg_reward);
  }
  if (d.entropy != null) {
    charts.entropy.data.labels.push(ts);
    charts.entropy.data.datasets[0].data.push(d.entropy);
  }
  if (d.sps != null) {
    charts.sps.data.labels.push(ts);
    charts.sps.data.datasets[0].data.push(d.sps);
  }
  if (d.policy_update != null || d.critic_update != null) {
    charts.updates.data.labels.push(ts);
    charts.updates.data.datasets[0].data.push(d.policy_update ?? null);
    charts.updates.data.datasets[1].data.push(d.critic_update ?? null);
  }
}

function updateStats(last) {
  document.getElementById('s-iter').textContent = last.total_iterations || '-';
  document.getElementById('s-ts').textContent = fmt(last.total_timesteps);
  document.getElementById('s-rew').textContent = last.avg_reward != null ? last.avg_reward.toFixed(4) : '-';
  document.getElementById('s-sps').textContent = fmt(last.sps);
  document.getElementById('s-ent').textContent = last.entropy != null ? last.entropy.toFixed(4) : '-';
  document.getElementById('s-time').textContent = fmtTime(last.wall_time);
}

// ---- Controls ----
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

// ---- Polling ----
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
