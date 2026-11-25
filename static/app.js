/* Dashboard prediksi reksa dana LSTM — frontend (vanilla JS + ECharts) */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const COLORS = { close: '#118dff', pred: '#e66c37', naive: '#a19f9d', fc: '#6b007b', band: 'rgba(107,0,123,0.12)', bar: '#118dff', bar2: '#e66c37', loss: '#118dff', val: '#e66c37', grid: '#edebe9', axis: '#605e5c' };
const QUOTE_EVERY_MS = 60_000;

const nf = (d) => new Intl.NumberFormat('id-ID', { minimumFractionDigits: d, maximumFractionDigits: d });
const f2 = (v) => (v == null || isNaN(v) ? '—' : nf(2).format(v));
const f1 = (v) => (v == null || isNaN(v) ? '—' : nf(1).format(v));
const pct = (v, d = 2) => (v == null || isNaN(v) ? '—' : (v > 0 ? '+' : v < 0 ? '−' : '') + nf(d).format(Math.abs(v)) + '%');
const dateFmt = new Intl.DateTimeFormat('id-ID', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
const fd = (s) => dateFmt.format(new Date(s + 'T00:00:00Z'));
const monthFmt = new Intl.DateTimeFormat('id-ID', { month: 'short', year: '2-digit', timeZone: 'UTC' });
const fm = (s) => monthFmt.format(new Date(s + '-01T00:00:00Z'));
const cls = (el, v) => { el.classList.toggle('up', v > 0); el.classList.toggle('down', v < 0); };

const state = { config: null, history: null, quote: null, result: null, runs: [], activeRun: null, polling: null };
const charts = {};

function chart(id) {
  if (!charts[id]) charts[id] = echarts.init(document.getElementById(id), null, { renderer: 'canvas' });
  return charts[id];
}
window.addEventListener('resize', () => Object.values(charts).forEach((c) => c.resize()));

const baseAxis = {
  axisLine: { lineStyle: { color: '#c8c6c4' } }, axisLabel: { color: COLORS.axis, fontSize: 11 },
  splitLine: { lineStyle: { color: COLORS.grid } }, axisTick: { show: false },
};
const baseOpt = {
  animationDuration: 400, textStyle: { fontFamily: 'Segoe UI, -apple-system, Roboto, sans-serif' },
  tooltip: { trigger: 'axis', backgroundColor: '#fff', borderColor: '#e1dfdd', textStyle: { color: '#252423', fontSize: 12 } },
  grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
};

/* ---------- Parameter panel ---------- */
function ticker() { return $('#ticker').value.trim().toUpperCase() || state.config.ticker; }
function start() { return $('#start').value || state.config.start; }

function setParams(p) {
  $('#features').replaceChildren(...state.config.features.map((f) => {
    const label = document.createElement('label');
    label.className = 'chip';
    const cb = Object.assign(document.createElement('input'), { type: 'checkbox', value: f, checked: p.features.includes(f), disabled: f === 'Close' });
    label.append(cb, f);
    return label;
  }));
  for (const [k, v] of Object.entries(p)) {
    const el = document.getElementById('p-' + k);
    if (!el) continue;
    if (el.tagName === 'SELECT' && ![...el.options].some((o) => o.value === String(v))) el.add(new Option(String(v)));
    el.value = String(v);
  }
  updateOutputs();
}

function readParams() {
  const p = { features: $$('#features input:checked').map((i) => i.value) };
  for (const el of $$('#params-form [name]')) p[el.name] = Number(el.value);
  return p;
}

function updateOutputs() {
  $$('#params-form output').forEach((o) => {
    const v = Number(document.getElementById(o.dataset.for).value);
    o.textContent = o.dataset.fmt === 'pct' ? Math.round(v * 100) + '%' : (o.dataset.for === 'p-dropout' ? v.toFixed(2) : v) + (o.dataset.suffix || '');
  });
}
$('#params-form').addEventListener('input', updateOutputs);

/* ---------- Data ---------- */
async function api(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(body.error || res.statusText), { status: res.status, body });
  return body;
}

async function loadHistory(refresh = false) {
  $('#subtitle').textContent = 'Memuat data ' + ticker() + '…';
  try {
    const h = await api(`/api/history?ticker=${encodeURIComponent(ticker())}&start=${start()}${refresh ? '&refresh=1' : ''}`);
    state.history = h;
    const last = h.rows[h.rows.length - 1];
    $('#subtitle').textContent = `${h.ticker} · ${h.rows.length} hari data · s/d ${fd(last.date)} · sumber ${h.source === 'yahoo' ? 'Yahoo Finance' : 'CSV cadangan'}`;
    const banner = $('#banner');
    banner.hidden = h.source === 'yahoo';
    banner.textContent = h.source === 'yahoo' ? '' : `Yahoo Finance tidak bisa diakses (${h.error}). Dashboard memakai data cadangan dari folder data/, yang berhenti di ${fd(last.date)}.`;
    renderHistoryKpis();
    renderPrice();
  } catch (e) {
    $('#subtitle').textContent = 'Gagal memuat data: ' + e.message;
  }
}

async function loadQuote() {
  try {
    const q = await api(`/api/quote?ticker=${encodeURIComponent(ticker())}`);
    state.quote = q;
    const live = q.source === 'yahoo';
    $('#live-dot').className = 'dot ' + (live ? 'on' : 'off');
    $('#live-price').textContent = f2(q.price);
    const chg = $('#live-chg');
    chg.textContent = pct(q.change_pct);
    cls(chg, q.change_pct);
    const t = new Date(q.at * 1000).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    $('#live-meta').textContent = live ? `Yahoo Finance · diperbarui ${t}` : `Offline · harga per ${fd(q.as_of)}`;
    renderHistoryKpis();
  } catch (e) {
    $('#live-dot').className = 'dot off';
    $('#live-meta').textContent = 'Harga tidak tersedia';
  }
}

/* ---------- Training ---------- */
async function train() {
  const btn = $('#train-btn');
  btn.disabled = true;
  btn.textContent = 'Melatih…';
  $('#progress').hidden = false;
  setProgress(0, 'Menyiapkan…');
  try {
    let jobId;
    try {
      ({ job_id: jobId } = await api('/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker(), start: start(), params: readParams() }),
      }));
    } catch (e) {
      if (e.status === 409 && e.body.job_id) jobId = e.body.job_id; else throw e;
    }
    const job = await pollJob(jobId);
    if (job.status === 'error') throw new Error(job.error);
    addRun(job.result);
    setProgress(100, `Selesai dalam ${f1(job.result.seconds)} detik · epoch terbaik ${job.result.best_epoch}`);
  } catch (e) {
    setProgress(0, 'Gagal: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Latih model';
  }
}

function pollJob(id) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const j = await api('/api/jobs/' + id);
        if (j.status === 'done' || j.status === 'error') return resolve(j);
        const label = j.status === 'training' ? `Epoch ${j.epoch} / ${j.epochs}` : 'Mengambil data…';
        const last = j.live_loss[j.live_loss.length - 1];
        setProgress(j.epochs ? (j.epoch / j.epochs) * 100 : 0, label + (last ? ` · loss ${last.loss.toExponential(2)}` : ''));
        if (j.live_loss.length) renderLoss(j.live_loss, null, true);
        setTimeout(tick, 800);
      } catch (e) { reject(e); }
    };
    tick();
  });
}

function setProgress(p, text) {
  $('#progress-bar').style.width = Math.min(100, p) + '%';
  $('#progress-text').textContent = text;
}

/* ---------- Runs (riwayat percobaan) ---------- */
const runsKey = () => 'lstm-runs:' + ticker();
function loadRuns() {
  try { state.runs = JSON.parse(localStorage.getItem(runsKey())) || []; } catch { state.runs = []; }
}
function saveRuns() {
  try { localStorage.setItem(runsKey(), JSON.stringify(state.runs.slice(-15))); } catch { /* penuh: abaikan */ }
}
function addRun(result) {
  result.id = (state.runs.at(-1)?.id || 0) + 1;
  state.runs.push(result);
  state.runs = state.runs.slice(-15);
  saveRuns();
  showRun(result);
}
function showRun(result) {
  state.result = result;
  state.activeRun = result.id;
  renderResult();
  renderRuns();
}

/* ---------- Rendering ---------- */
function renderHistoryKpis() {
  const h = state.history;
  if (!h) return;
  const rows = h.rows;
  const last = rows[rows.length - 1];
  const price = state.quote?.price ?? last.close;
  $('#k-price').textContent = f2(price);
  const sub = $('#k-price-sub');
  if (state.quote) { sub.textContent = pct(state.quote.change_pct) + ' hari ini'; cls(sub, state.quote.change_pct); }
  else sub.textContent = fd(last.date);
  const target = new Date(last.date + 'T00:00:00Z'); target.setUTCFullYear(target.getUTCFullYear() - 1);
  const base = rows.find((r) => new Date(r.date + 'T00:00:00Z') >= target);
  const ret = base ? (last.close / base.close - 1) * 100 : null;
  const k = $('#k-ret');
  k.textContent = pct(ret);
  cls(k, ret);
}

function renderResult() {
  const r = state.result;
  const m = r.metrics;
  $('#k-rmse').textContent = f2(m.rmse);
  $('#k-mae').textContent = 'MAE ' + f2(m.mae) + ' · R² ' + f2(m.r2);
  $('#k-mape').textContent = f2(m.mape) + '%';
  const kn = $('#k-naive');
  kn.textContent = pct(m.vs_naive_pct, 1);
  cls(kn, m.vs_naive_pct);
  $('#k-naive-sub').textContent = `RMSE naive ${f2(m.naive_rmse)} · ${m.vs_naive_pct > 0 ? 'model lebih baik' : 'model kalah'}`;
  $('#k-dir').textContent = f1(m.dir_acc) + '%';
  $('#k-dir-sub').textContent = `${m.dir_hit} benar · ${m.dir_miss} salah`;
  const fc = r.forecast[r.forecast.length - 1];
  const lastClose = r.test[r.test.length - 1].actual;
  $('#k-fc-label').textContent = `Forecast H+${r.forecast.length}`;
  const kf = $('#k-fc');
  kf.textContent = f2(fc.predicted);
  const chg = (fc.predicted / lastClose - 1) * 100;
  const sub = $('#k-fc-sub');
  sub.textContent = pct(chg) + ' · ' + fd(fc.date);
  cls(sub, chg);
  $('#loss-note').textContent = `epoch terbaik ${r.best_epoch} dari ${r.loss.length}`;
  renderPrice();
  renderLoss(r.loss, r.best_epoch);
  renderMape();
  renderHist();
  renderScatter();
  renderForecastTable();
}

function renderPrice() {
  const h = state.history;
  if (!h) return;
  const r = state.result && state.result.ticker === h.ticker ? state.result : null;
  const series = [{
    name: 'Close', type: 'line', showSymbol: false, data: h.rows.map((x) => [x.date, x.close]),
    lineStyle: { width: 2, color: COLORS.close }, itemStyle: { color: COLORS.close },
  }];
  if (r) {
    series.push(
      { name: 'Prediksi LSTM', type: 'line', showSymbol: false, data: r.test.map((x) => [x.date, x.predicted]), lineStyle: { width: 2, color: COLORS.pred }, itemStyle: { color: COLORS.pred } },
      { name: 'Naive (harga kemarin)', type: 'line', showSymbol: false, data: r.test.map((x) => [x.date, x.naive]), lineStyle: { width: 1, type: 'dotted', color: COLORS.naive }, itemStyle: { color: COLORS.naive } },
      { name: 'Rentang 95%', type: 'line', stack: 'band', showSymbol: false, data: r.forecast.map((x) => [x.date, x.lo]), lineStyle: { opacity: 0 }, itemStyle: { color: COLORS.band }, tooltip: { show: false } },
      { name: 'Rentang 95%', type: 'line', stack: 'band', showSymbol: false, data: r.forecast.map((x) => [x.date, x.hi - x.lo]), lineStyle: { opacity: 0 }, areaStyle: { color: COLORS.band }, itemStyle: { color: COLORS.band }, tooltip: { show: false } },
      { name: 'Forecast', type: 'line', showSymbol: false, data: r.forecast.map((x) => [x.date, x.predicted]), lineStyle: { width: 2, type: 'dashed', color: COLORS.fc }, itemStyle: { color: COLORS.fc } },
    );
  }
  const startZoom = Math.max(0, 100 - (400 / (h.rows.length + (r ? r.forecast.length : 0))) * 100);
  chart('c-price').setOption({
    ...baseOpt,
    legend: { type: 'scroll', top: 0, left: 0, right: 0, icon: 'roundRect', itemWidth: 14, itemHeight: 4, textStyle: { fontSize: 11, color: COLORS.axis }, selected: { 'Naive (harga kemarin)': false } },
    grid: { ...baseOpt.grid, top: 34, bottom: 46 },
    tooltip: { ...baseOpt.tooltip, valueFormatter: (v) => f2(v) },
    xAxis: { type: 'time', ...baseAxis, splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: (v) => nf(0).format(v) } },
    dataZoom: [{ type: 'inside', start: startZoom, end: 100 }, { type: 'slider', start: startZoom, end: 100, height: 22, bottom: 6, borderColor: '#e1dfdd', fillerColor: 'rgba(17,141,255,.12)', textStyle: { fontSize: 10 } }],
    series,
  }, { replaceMerge: ['series'] });
}

function renderLoss(loss, best, live = false) {
  chart('c-loss').setOption({
    ...baseOpt,
    animation: !live,
    legend: { top: 0, left: 0, textStyle: { fontSize: 11, color: COLORS.axis } },
    tooltip: { ...baseOpt.tooltip, valueFormatter: (v) => (v == null ? '—' : Number(v).toExponential(2)) },
    xAxis: { type: 'category', name: 'epoch', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: COLORS.axis, fontSize: 11 }, data: loss.map((x) => x.epoch), ...baseAxis, splitLine: { show: false } },
    yAxis: { type: 'log', ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: (v) => v.toExponential(0) } },
    grid: { ...baseOpt.grid, bottom: 24 },
    series: [
      { name: 'Loss latih', type: 'line', showSymbol: false, data: loss.map((x) => x.loss), lineStyle: { color: COLORS.loss, width: 2 }, itemStyle: { color: COLORS.loss },
        markLine: best ? { symbol: 'none', label: { formatter: 'terbaik', fontSize: 10, color: COLORS.axis }, lineStyle: { color: '#8a8886', type: 'dashed' }, data: [{ xAxis: String(best) }] } : undefined },
      { name: 'Loss validasi', type: 'line', showSymbol: false, data: loss.map((x) => x.val_loss), lineStyle: { color: COLORS.val, width: 2 }, itemStyle: { color: COLORS.val } },
    ],
  }, { replaceMerge: ['series'] });
}

function renderMape() {
  const groups = new Map();
  for (const t of state.result.test) {
    const k = t.date.slice(0, 7);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(Math.abs(t.actual - t.predicted) / t.actual * 100);
  }
  const rows = [...groups].map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length, v.length]);
  chart('c-mape').setOption({
    ...baseOpt,
    tooltip: { ...baseOpt.tooltip, formatter: (p) => `${fm(rows[p[0].dataIndex][0])}<br>MAPE <b>${f2(p[0].value)}%</b> · ${rows[p[0].dataIndex][2]} hari` },
    xAxis: { type: 'category', data: rows.map((r) => fm(r[0])), ...baseAxis, splitLine: { show: false } },
    yAxis: { type: 'value', ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: (v) => f1(v) + '%' } },
    grid: { ...baseOpt.grid, top: 20 },
    series: [{ type: 'bar', data: rows.map((r) => r[1]), barMaxWidth: 28, itemStyle: { color: COLORS.bar, borderRadius: [3, 3, 0, 0] },
      label: { show: rows.length <= 12, position: 'top', fontSize: 10, color: COLORS.axis, formatter: (p) => f2(p.value) } }],
  }, { replaceMerge: ['series'] });
}

function renderHist() {
  const e = state.result.test.map((t) => t.actual - t.predicted);
  const step = Math.max(0.5, Math.ceil((Math.max(...e) - Math.min(...e)) / 30 * 2) / 2);
  const lo = Math.floor(Math.min(...e) / step) * step, hi = Math.ceil(Math.max(...e) / step) * step || lo + step;
  const bins = [];
  for (let x = lo; x < hi - 1e-9; x += step) bins.push({ from: x, to: x + step, n: 0 });
  e.forEach((v) => { const b = bins[Math.min(bins.length - 1, Math.floor((v - lo) / step))]; if (b) b.n++; });
  chart('c-hist').setOption({
    ...baseOpt,
    tooltip: { ...baseOpt.tooltip, formatter: (p) => `Error ${f1(bins[p[0].dataIndex].from)} s/d ${f1(bins[p[0].dataIndex].to)}<br><b>${p[0].value}</b> hari` },
    xAxis: { type: 'category', data: bins.map((b) => f1(b.from)), ...baseAxis, splitLine: { show: false } },
    yAxis: { type: 'value', minInterval: 1, ...baseAxis },
    grid: { ...baseOpt.grid, top: 20 },
    series: [{ type: 'bar', data: bins.map((b) => ({ value: b.n, itemStyle: { color: b.to <= 0 ? COLORS.bar2 : COLORS.bar } })), barCategoryGap: '8%' }],
  }, { replaceMerge: ['series'] });
}

function renderScatter() {
  const pts = state.result.test.map((t) => [t.actual, t.predicted, t.date]);
  const all = pts.flatMap((p) => [p[0], p[1]]);
  const lo = Math.floor(Math.min(...all)) - 1, hi = Math.ceil(Math.max(...all)) + 1;
  chart('c-scatter').setOption({
    ...baseOpt,
    tooltip: { trigger: 'item', backgroundColor: '#fff', borderColor: '#e1dfdd', textStyle: { color: '#252423', fontSize: 12 }, formatter: (p) => p.seriesType === 'scatter' ? `${fd(p.value[2])}<br>Aktual <b>${f2(p.value[0])}</b><br>Prediksi <b>${f2(p.value[1])}</b>` : '' },
    xAxis: { type: 'value', min: lo, max: hi, name: 'aktual', nameLocation: 'middle', nameGap: 22, nameTextStyle: { color: COLORS.axis, fontSize: 11 }, ...baseAxis },
    yAxis: { type: 'value', min: lo, max: hi, name: 'prediksi', nameTextStyle: { color: COLORS.axis, fontSize: 11 }, ...baseAxis },
    grid: { ...baseOpt.grid, top: 28, bottom: 22 },
    series: [
      { type: 'line', data: [[lo, lo], [hi, hi]], showSymbol: false, silent: true, lineStyle: { color: '#8a8886', type: 'dashed', width: 1 } },
      { type: 'scatter', data: pts, symbolSize: 7, itemStyle: { color: COLORS.close, opacity: 0.6 } },
    ],
  }, { replaceMerge: ['series'] });
}

function renderForecastTable() {
  const r = state.result;
  const base = r.test[r.test.length - 1].actual;
  const pick = new Set([1, 2, 3, 4, 5, 10, 20, 30, 60, 90, 120]);
  const tb = $('#t-forecast tbody');
  tb.replaceChildren(...r.forecast.map((f, i) => [f, i + 1]).filter(([, h]) => pick.has(h) || h === r.forecast.length).map(([f, h]) => {
    const tr = document.createElement('tr');
    const chg = (f.predicted / base - 1) * 100;
    [fd(f.date), 'H+' + h, f2(f.predicted), `${f2(f.lo)} – ${f2(f.hi)}`, pct(chg)].forEach((t, i) => {
      const td = document.createElement('td');
      td.textContent = t;
      if (i === 4) cls(td, chg);
      tr.appendChild(td);
    });
    return tr;
  }));
}

function renderRuns() {
  const tb = $('#t-runs tbody');
  if (!state.runs.length) {
    const tr = document.createElement('tr');
    const td = Object.assign(document.createElement('td'), { colSpan: 9, className: 'empty', textContent: 'Belum ada percobaan. Atur variabel di kiri, lalu klik Latih model.' });
    tr.appendChild(td);
    tb.replaceChildren(tr);
    return;
  }
  const best = Math.min(...state.runs.map((r) => r.metrics.rmse));
  tb.replaceChildren(...[...state.runs].reverse().map((r) => {
    const p = r.params, m = r.metrics;
    const tr = document.createElement('tr');
    tr.classList.toggle('active', r.id === state.activeRun);
    const cells = [r.id, p.window, `${p.layers}×${p.units}, d${p.dropout}`, `${r.best_epoch}/${r.loss.length}`, p.features.length,
      f2(m.rmse), f2(m.mape) + '%', pct(m.vs_naive_pct, 1), f1(m.dir_acc) + '%'];
    cells.forEach((c, i) => {
      const td = document.createElement('td');
      td.textContent = c;
      if (i === 5 && m.rmse === best) td.className = 'best';
      if (i === 7) cls(td, m.vs_naive_pct);
      tr.appendChild(td);
    });
    tr.title = 'Fitur: ' + p.features.join(', ') + ` · batch ${p.batch_size} · lr ${p.learning_rate} · horizon ${p.horizon}`;
    tr.addEventListener('click', () => { setParams(p); showRun(r); });
    return tr;
  }));
}

/* ---------- Events & init ---------- */
$('#params-form').addEventListener('submit', (e) => { e.preventDefault(); train(); });
$('#reset-btn').addEventListener('click', () => setParams(state.config.defaults));
$('#refresh-data').addEventListener('click', () => { loadHistory(true); loadQuote(); });
$('#ticker-form').addEventListener('submit', (e) => {
  e.preventDefault();
  state.result = null;
  loadRuns();
  renderRuns();
  loadHistory().then(() => { if (state.runs.length) showRun(state.runs.at(-1)); });
  loadQuote();
});
$('#clear-runs').addEventListener('click', () => { state.runs = []; saveRuns(); renderRuns(); });

(async function init() {
  state.config = await api('/api/config');
  $('#ticker').value = state.config.ticker;
  $('#start').value = state.config.start;
  setParams(state.config.defaults);
  loadRuns();
  renderRuns();
  await loadHistory();
  loadQuote();
  setInterval(loadQuote, QUOTE_EVERY_MS);
  if (state.runs.length) showRun(state.runs.at(-1));
  else train(); // langsung latih sekali dengan parameter default supaya semua visual terisi
})();
