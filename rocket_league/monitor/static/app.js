// ---------------------------------------------------------------------------
// Notes — data + Chart.js plugin
// ---------------------------------------------------------------------------

let currentNotes = [];

const notePlugin = {
  id: 'noteAnnotations',
  afterDraw(chart) {
    if (!currentNotes.length) return;
    const { ctx, chartArea: { left, right, top, bottom }, scales: { x } } = chart;
    if (!x) return;
    const labels = chart.data.labels;
    if (!labels.length) return;

    for (const note of currentNotes) {
      // Find the x pixel for this note's timestep
      const tsMillion = (note.timestep / 1e6).toFixed(2);
      // Find nearest label index
      let bestIdx = -1, bestDist = Infinity;
      for (let i = 0; i < labels.length; i++) {
        const dist = Math.abs(parseFloat(labels[i]) - parseFloat(tsMillion));
        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
      }
      if (bestIdx < 0 || bestDist > 0.5) continue;

      const px = x.getPixelForValue(bestIdx);
      if (px < left || px > right) continue;

      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
      ctx.stroke();

      // Small diamond marker at top
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(79, 195, 247, 0.7)';
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px + 4, top + 5);
      ctx.lineTo(px, top + 10);
      ctx.lineTo(px - 4, top + 5);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }
};
Chart.register(notePlugin);

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
  ballSpeed:  mkChart('c-ballspeed', '#ff7043'),
  boostUsage: mkChart('c-boostusage', '#ab47bc'),
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
// Progress overlay (for long-running actions)
// ---------------------------------------------------------------------------

function showProgress(title, initialSteps) {
  document.getElementById('progress-title').textContent = title;
  document.getElementById('progress-spinner').className = 'progress-spinner';
  document.getElementById('progress-footer').style.display = 'none';
  document.getElementById('progress-output').className = 'progress-output';
  document.getElementById('progress-output').textContent = '';

  const list = document.getElementById('progress-steps');
  list.innerHTML = '';
  if (initialSteps) {
    initialSteps.forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      li.className = 'step-active';
      list.appendChild(li);
    });
  }

  document.getElementById('progress-overlay').classList.add('open');
}

function setProgressSteps(steps, allDone) {
  const list = document.getElementById('progress-steps');
  list.innerHTML = '';
  steps.forEach((s, i) => {
    const li = document.createElement('li');
    li.textContent = s;
    li.className = allDone ? 'step-done' : (i === steps.length - 1 ? 'step-active' : 'step-done');
    list.appendChild(li);
  });
}

function setProgressDone(title, ok, output) {
  document.getElementById('progress-title').textContent = title;
  const spinner = document.getElementById('progress-spinner');
  spinner.className = 'progress-spinner ' + (ok ? 'done' : 'fail');
  document.getElementById('progress-footer').style.display = '';

  // Mark all steps as done or fail
  const items = document.getElementById('progress-steps').querySelectorAll('li');
  items.forEach(li => {
    if (li.classList.contains('step-active')) {
      li.className = ok ? 'step-done' : 'step-fail';
    }
  });

  if (output) {
    const el = document.getElementById('progress-output');
    el.textContent = output;
    el.className = 'progress-output show';
  }
}

function closeProgress() {
  document.getElementById('progress-overlay').classList.remove('open');
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

  // Ball speed & boost usage
  if (d.ball_speed != null) pushPt(charts.ballSpeed, 0, ts, d.ball_speed);
  if (d.player_boost_usage != null) pushPt(charts.boostUsage, 0, ts, d.player_boost_usage);

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
  document.getElementById('s-ballspeed').textContent = last.ball_speed != null ? last.ball_speed.toFixed(0) : '-';
  document.getElementById('s-boostusage').textContent = fmtPct(last.player_boost_usage);
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
    loadNotes();
    poll();
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

  // Check if source files changed since last build
  try {
    const sr = await fetch('/api/source-status');
    const sj = await sr.json();
    if (sj.modified) {
      const files = sj.changed_files.join(', ');
      if (confirm(files + ' changed since last build.\n\nRebuild before training?')) {
        await doRebuild();
        // doRebuild shows progress overlay; user closes it when done.
        // We proceed to start training regardless (rebuild result shown in overlay).
      }
    }
  } catch (e) { /* source check failed, proceed anyway */ }

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

function updateControls(statusObj) {
  const status = statusObj.status || statusObj;
  const badge = document.getElementById('status-badge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status.toUpperCase();
  document.getElementById('btn-start').disabled = (status !== 'idle');
  document.getElementById('btn-stop').disabled  = (status !== 'running');
  document.getElementById('btn-kill').disabled   = (status === 'idle');

  // Test game button state
  const testBtn = document.querySelector('[onclick="doTestGame()"]');
  if (testBtn) {
    if (statusObj.test_game_running) {
      testBtn.textContent = 'Game Running';
      testBtn.disabled = true;
    } else {
      testBtn.textContent = 'Test Game';
      testBtn.disabled = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Bot Config Modal
// ---------------------------------------------------------------------------

let modalMode = 'create'; // 'create' or 'edit'

const DEFAULTS = {
  gamemode: '1v1',
  training: { numGames: 32, tickSkip: 8, randomSeed: 42, tsPerSave: 5000000 },
  ppo: {
    tsPerItr: 50000, batchSize: 50000, miniBatchSize: 50000,
    epochs: 2, entropyScale: 0.035, gaeGamma: 0.99, gaeLambda: 0.95,
    clipRange: 0.2, policyLR: 0.00015, criticLR: 0.00015,
  },
  network: { sharedHead: [256, 256], policy: [256, 256, 256], critic: [256, 256, 256] },
};

function fillModalFromConfig(cfg) {
  document.getElementById('cfg-gamemode').value = cfg.gamemode || '1v1';

  // Training
  const t = cfg.training || {};
  document.getElementById('cfg-numGames').value = t.numGames ?? 32;
  document.getElementById('cfg-tickSkip').value = t.tickSkip ?? 8;
  document.getElementById('cfg-randomSeed').value = t.randomSeed ?? 42;
  document.getElementById('cfg-tsPerSave').value = t.tsPerSave ?? 5000000;

  // PPO
  const p = cfg.ppo || {};
  document.getElementById('cfg-tsPerItr').value = p.tsPerItr ?? 50000;
  document.getElementById('cfg-batchSize').value = p.batchSize ?? 50000;
  document.getElementById('cfg-miniBatchSize').value = p.miniBatchSize ?? 50000;
  document.getElementById('cfg-epochs').value = p.epochs ?? 2;
  document.getElementById('cfg-entropyScale').value = p.entropyScale ?? 0.035;
  document.getElementById('cfg-gaeGamma').value = p.gaeGamma ?? 0.99;
  document.getElementById('cfg-gaeLambda').value = p.gaeLambda ?? 0.95;
  document.getElementById('cfg-clipRange').value = p.clipRange ?? 0.2;
  document.getElementById('cfg-policyLR').value = p.policyLR ?? 0.00015;
  document.getElementById('cfg-criticLR').value = p.criticLR ?? 0.00015;

  // Network
  const n = cfg.network || {};
  document.getElementById('cfg-sharedHead').value = (n.sharedHead || [256, 256]).join(', ');
  document.getElementById('cfg-policy').value = (n.policy || [256, 256, 256]).join(', ');
  document.getElementById('cfg-critic').value = (n.critic || [256, 256, 256]).join(', ');
}

function parseLayerSizes(str) {
  return str.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
}

function readModalConfig() {
  return {
    gamemode: document.getElementById('cfg-gamemode').value,
    training: {
      numGames: parseInt(document.getElementById('cfg-numGames').value) || 32,
      tickSkip: parseInt(document.getElementById('cfg-tickSkip').value) || 8,
      randomSeed: parseInt(document.getElementById('cfg-randomSeed').value) || 42,
      tsPerSave: parseInt(document.getElementById('cfg-tsPerSave').value) || 5000000,
    },
    ppo: {
      tsPerItr: parseInt(document.getElementById('cfg-tsPerItr').value) || 50000,
      batchSize: parseInt(document.getElementById('cfg-batchSize').value) || 50000,
      miniBatchSize: parseInt(document.getElementById('cfg-miniBatchSize').value) || 50000,
      epochs: parseInt(document.getElementById('cfg-epochs').value) || 2,
      entropyScale: parseFloat(document.getElementById('cfg-entropyScale').value) || 0.035,
      gaeGamma: parseFloat(document.getElementById('cfg-gaeGamma').value) || 0.99,
      gaeLambda: parseFloat(document.getElementById('cfg-gaeLambda').value) || 0.95,
      clipRange: parseFloat(document.getElementById('cfg-clipRange').value) || 0.2,
      policyLR: parseFloat(document.getElementById('cfg-policyLR').value) || 0.00015,
      criticLR: parseFloat(document.getElementById('cfg-criticLR').value) || 0.00015,
    },
    network: {
      sharedHead: parseLayerSizes(document.getElementById('cfg-sharedHead').value),
      policy: parseLayerSizes(document.getElementById('cfg-policy').value),
      critic: parseLayerSizes(document.getElementById('cfg-critic').value),
    },
  };
}

function openCreateModal() {
  modalMode = 'create';
  document.getElementById('modal-title').textContent = 'New Bot';
  document.getElementById('modal-submit').textContent = 'Create Bot';
  document.getElementById('modal-name-row').style.display = '';
  document.getElementById('cfg-name').value = '';
  fillModalFromConfig(DEFAULTS);
  document.getElementById('modal-overlay').classList.add('open');
}

async function openEditModal() {
  modalMode = 'edit';
  document.getElementById('modal-title').textContent = 'Edit Config';
  document.getElementById('modal-submit').textContent = 'Save Config';
  document.getElementById('modal-name-row').style.display = 'none';

  try {
    const bot = document.getElementById('bot-select').value;
    const r = await fetch('/api/bots/config?bot=' + encodeURIComponent(bot));
    const cfg = await r.json();
    fillModalFromConfig(cfg);
  } catch (e) {
    fillModalFromConfig(DEFAULTS);
  }
  document.getElementById('modal-overlay').classList.add('open');
}

function closeModal(e) {
  // If called from overlay click, only close if clicking the overlay itself
  if (e && e.target !== document.getElementById('modal-overlay')) return;
  document.getElementById('modal-overlay').classList.remove('open');
}

async function submitModal() {
  const config = readModalConfig();

  if (modalMode === 'create') {
    const name = document.getElementById('cfg-name').value.trim();
    if (!name) { toast('Enter a bot name', false); return; }
    const r = await fetch('/api/bots/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, config }),
    });
    const j = await r.json();
    if (j.ok) {
      toast('Created bot: ' + j.name);
      document.getElementById('modal-overlay').classList.remove('open');
      await loadBots();
      await selectBot(j.name);
    } else {
      toast(j.error, false);
    }
  } else {
    // Edit mode — save config for current bot
    const bot = document.getElementById('bot-select').value;
    const r = await fetch('/api/bots/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot, config }),
    });
    const j = await r.json();
    if (j.ok) {
      toast('Config saved for: ' + bot);
      document.getElementById('modal-overlay').classList.remove('open');
    } else {
      toast(j.error, false);
    }
  }
}

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeModal();
    closeNoteModal();
    closeProgress();
  }
});

// ---------------------------------------------------------------------------
// Quick actions (with streaming console log)
// ---------------------------------------------------------------------------

async function openRewards() {
  const r = await fetch('/api/open-rewards', { method: 'POST' });
  const j = await r.json();
  if (j.ok) toast('Opened reward files in editor');
  else toast(j.error, false);
}

/**
 * Stream a background task's console output into the progress modal.
 * Returns the task result when done, or null on error.
 */
async function streamTask(taskId, title) {
  showProgress(title, ['Starting...']);
  const outputEl = document.getElementById('progress-output');
  outputEl.className = 'progress-output show';
  outputEl.textContent = '';

  let sinceIdx = 0;
  let finished = false;
  let result = null;

  while (!finished) {
    await new Promise(r => setTimeout(r, 300));
    try {
      const r = await fetch('/api/task-status?id=' + taskId + '&since=' + sinceIdx);
      const j = await r.json();

      if (j.lines && j.lines.length > 0) {
        outputEl.textContent += j.lines.join('\n') + '\n';
        outputEl.scrollTop = outputEl.scrollHeight;
        sinceIdx = j.total_lines;
      }

      if (j.status !== 'running') {
        finished = true;
        result = j.result;
        const ok = result && result.ok;
        setProgressDone(ok ? title + ' — Succeeded' : title + ' — Failed', ok);
        fetchActivity();
      }
    } catch (e) { /* poll error, retry */ }
  }
  return result;
}

async function doRebuild() {
  try {
    const r = await fetch('/api/rebuild', { method: 'POST' });
    const j = await r.json();
    if (j.ok) {
      await streamTask('rebuild', 'Building...');
    } else {
      toast(j.error, false);
    }
  } catch (e) {
    toast('Network error: ' + e.message, false);
  }
}

async function doTestGame() {
  const bot = document.getElementById('bot-select').value;
  try {
    const r = await fetch('/api/test-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot }),
    });
    const j = await r.json();

    if (j.ok) {
      // Stream task output — test game keeps running after "done"
      streamTask('test-game', 'Test Game').then(() => {
        // Show Stop Game button once launch sequence finishes
        const footer = document.getElementById('progress-footer');
        footer.style.display = '';
        footer.innerHTML =
          '<button class="btn-sm btn-kill" onclick="stopTestGame()">Stop Game</button> ' +
          '<button class="btn-sm btn-action" onclick="closeProgress()">Close</button>';
      });
    } else {
      toast(j.error, false);
    }
  } catch (e) {
    toast('Network error: ' + e.message, false);
  }
}

async function stopTestGame() {
  try {
    const r = await fetch('/api/stop-test-game', { method: 'POST' });
    const j = await r.json();
    if (j.ok) {
      toast('Test game stopped');
      closeProgress();
      fetchActivity();
    } else {
      toast(j.error, false);
    }
  } catch (e) {
    toast('Error stopping game: ' + e.message, false);
  }
}

async function buildRLBot() {
  const bot = document.getElementById('bot-select').value;
  showProgress('Building for RLBot...', ['Waiting for folder selection...']);

  try {
    const r = await fetch('/api/build-rlbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot }),
    });
    const j = await r.json();

    if (j.ok) {
      await streamTask('build-rlbot', 'Building for RLBot...');
    } else if (j.error === 'Export cancelled') {
      closeProgress();
      toast('Export cancelled', false);
    } else {
      setProgressDone('RLBot Export Failed', false, j.error);
    }
  } catch (e) {
    setProgressDone('Export Error', false, 'Network error: ' + e.message);
  }
}

// ---------------------------------------------------------------------------
// Activity log
// ---------------------------------------------------------------------------

let activitySince = 0;

async function fetchActivity() {
  try {
    const r = await fetch('/api/activity?since=' + activitySince);
    const j = await r.json();
    if (j.entries && j.entries.length > 0) {
      const container = document.getElementById('activity-log');
      // Remove "no activity" placeholder
      const empty = container.querySelector('.activity-empty');
      if (empty) empty.remove();

      for (const e of j.entries) {
        const div = document.createElement('div');
        div.className = 'activity-entry';
        const tagClass = e.ok ? 'activity-tag-ok' : 'activity-tag-err';
        div.innerHTML =
          '<span class="activity-time">' + escHtml(e.time) + '</span>' +
          '<span class="activity-tag ' + tagClass + '">' + escHtml(e.action) + '</span>' +
          '<span class="activity-msg">' + escHtml(e.message) + '</span>';
        container.appendChild(div);
        // Keep scroll at bottom
        container.scrollTop = container.scrollHeight;
        // Track latest timestamp
        if (e.timestamp > activitySince) activitySince = e.timestamp;
      }

      // Trim old entries from DOM (keep last 50)
      while (container.children.length > 50) {
        container.removeChild(container.firstChild);
      }
    }
  } catch (e) { /* server not ready */ }
}

function escHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
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
    updateControls(sj);

    // On initial load, downsample to 500 points max to avoid lag.
    // Incremental polls (since > 0) get full resolution — just 1-2 new points.
    const maxPts = fetchedCount === 0 ? '&max_points=500' : '';
    const mr = await fetch('/api/metrics?since=' + fetchedCount + maxPts);
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

    // Activity log (every poll)
    fetchActivity();
  } catch (e) { /* server not ready */ }
}

// ---------------------------------------------------------------------------
// Notes — modal + tooltip
// ---------------------------------------------------------------------------

function openNoteModal() {
  const tsEl = document.getElementById('s-ts');
  const tsText = tsEl ? tsEl.textContent : '-';
  document.getElementById('note-timestep').value = tsText;
  document.getElementById('note-text').value = '';
  document.getElementById('note-overlay').classList.add('open');
  setTimeout(() => document.getElementById('note-text').focus(), 100);
}

function closeNoteModal(e) {
  if (e && e.target !== document.getElementById('note-overlay')) return;
  document.getElementById('note-overlay').classList.remove('open');
}

async function submitNote() {
  const tsRaw = document.getElementById('note-timestep').value;
  const text = document.getElementById('note-text').value.trim();
  if (!text) { toast('Enter a note', false); return; }

  // Parse timestep from display format (e.g. "5.00M" → 5000000, "150K" → 150000)
  let timestep = 0;
  if (tsRaw.endsWith('M')) timestep = Math.round(parseFloat(tsRaw) * 1e6);
  else if (tsRaw.endsWith('K')) timestep = Math.round(parseFloat(tsRaw) * 1e3);
  else timestep = parseInt(tsRaw) || 0;

  try {
    const r = await fetch('/api/notes/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ timestep, text }),
    });
    const j = await r.json();
    if (j.ok) {
      toast('Note added');
      closeNoteModal();
      await loadNotes();
    } else {
      toast(j.error || 'Failed to add note', false);
    }
  } catch (e) {
    toast('Error: ' + e.message, false);
  }
}

async function loadNotes() {
  try {
    const r = await fetch('/api/notes');
    const j = await r.json();
    currentNotes = j.notes || [];
    // Redraw all charts to show/hide note lines
    Object.values(charts).forEach(c => c.update());
  } catch (e) { /* server not ready */ }
}

// Tooltip on hover — find note lines near cursor on any chart canvas
function setupNoteTooltips() {
  const tooltip = document.getElementById('note-tooltip');
  document.querySelectorAll('.chart-canvas-wrap canvas').forEach(canvas => {
    canvas.addEventListener('mousemove', (e) => {
      if (!currentNotes.length) { tooltip.style.display = 'none'; return; }

      const chart = Chart.getChart(canvas);
      if (!chart || !chart.scales.x) { tooltip.style.display = 'none'; return; }

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const labels = chart.data.labels;
      let found = null;

      for (const note of currentNotes) {
        const tsMillion = (note.timestep / 1e6).toFixed(2);
        let bestIdx = -1, bestDist = Infinity;
        for (let i = 0; i < labels.length; i++) {
          const dist = Math.abs(parseFloat(labels[i]) - parseFloat(tsMillion));
          if (dist < bestDist) { bestDist = dist; bestIdx = i; }
        }
        if (bestIdx < 0 || bestDist > 0.5) continue;
        const px = chart.scales.x.getPixelForValue(bestIdx);
        if (Math.abs(mouseX - px) < 8) { found = note; break; }
      }

      if (found) {
        const ts = found.timestep >= 1e6
          ? (found.timestep / 1e6).toFixed(2) + 'M'
          : (found.timestep / 1e3).toFixed(0) + 'K';
        tooltip.innerHTML = '<strong>' + ts + ' steps</strong><br>' + escHtml(found.text);
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
      } else {
        tooltip.style.display = 'none';
      }
    });

    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadBots();
loadNotes();
setupNoteTooltips();
setInterval(poll, 2000);
poll();
