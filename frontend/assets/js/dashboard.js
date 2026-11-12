/**
 * Trader Dashboard Orchestrator
 * TradingView Dark Theme BTC/USDT 5m AI Indicator UI
 */

// 1. Central Configuration Object (Read dynamically from API, defaults as fallback)
const CONFIG = {
  symbol: "BTCUSDT",
  timeframe: "5m",
  intervalMinutes: 5,
  horizonCandles: 2,
  horizonMinutes: 10,
  bufferSize: 3000,
  costFloorBps: 23.0,
  apiBase: "http://localhost:8000",
  dashApiBase: "http://localhost:8001",
  wsUrl: "ws://localhost:8000/ws/live",
  binanceWsUrl: "wss://stream.binance.com:9443/ws/btcusdt@kline_5m",
  binanceRestUrl: "https://api.binance.com/api/v3/klines",
  bandConstants: {
    K_MOVE: 0.5,
    Z: 1.28,
    ATR_PERIOD: 14,
  },
  vocabulary: {
    actionMap: { LONG: "UP", SHORT: "DOWN", NEUTRAL: "NO CLEAR SIGNAL", NO_SIGNAL: "NO CLEAR SIGNAL" },
    trendMap: { BULLISH: "UPTREND", UPTREND: "UPTREND", BEARISH: "DOWNTREND", DOWNTREND: "DOWNTREND", SIDEWAYS: "SIDEWAYS" },
    phaseMap: { IMPULSE: "BREAKTHROUGH", BREAKTHROUGH: "BREAKTHROUGH", PULLBACK: "PULLBACK" },
  },
  benchmarks: {
    refTestAccuracy: 51.90,
    refTestCoverage: 52.50,
    refOofAccuracy: 52.16,
    refOofCoverage: 65.90,
  },
  colors: {
    up: "#26a69a",
    down: "#ef5350",
    neutral: "#787b86",
    accent: "#2962ff",
    bgApp: "#131722",
    bgPanel: "#1e222d",
    border: "#2a2e39",
    grid: "rgba(42, 46, 57, 0.5)",
    text: "#d1d4dc",
    textDim: "#787b86",
  },
};

// State Store
const STATE = {
  isMock: new URLSearchParams(window.location.search).get("mock") === "1",
  activeTab: "live", // "live" | "replay"
  logTab: "live",
  logFilter: "all", // "all" | "correct" | "wrong" | "pending"
  liveStatus: "OFFLINE", // "LIVE" | "RECONNECTING" | "STALE" | "MOCK" | "OFFLINE"
  lastTickTime: 0,
  lastPrediction: null,
  liveCandles: [],
  livePredictions: [],
  replayCandles: [],
  replayPredictions: [],
  replayTestRange: null,
  replayCurrentIndex: 0,
  replayPlaying: false,
  replaySpeed: 1, // 1x, 2x, 5x, 10x
  replayTimer: null,
  abortController: null,
  bandHits: { total: 0, inside: 0 },
  layerToggles: {
    markers: true,
    bands: true,
    smartPrice: true,
    classicBb: false,
    volume: true,
  },
};

// UI Elements Map
let DOM = {};

function initDOM() {
  DOM = {
    clockLocal: document.getElementById("clock-local"),
    clockServer: document.getElementById("clock-server"),
    statusDot: document.getElementById("status-dot"),
    statusText: document.getElementById("status-text"),
    staleBanner: document.getElementById("stale-banner"),
    btnLive: document.getElementById("btn-tab-live"),
    btnReplay: document.getElementById("btn-tab-replay"),
    liveView: document.getElementById("live-view"),
    replayView: document.getElementById("replay-view"),
    btnGuide: document.getElementById("btn-guide"),
    btnCloseGuide: document.getElementById("btn-close-guide"),
    guideDrawer: document.getElementById("guide-drawer"),
    // Prediction Panel
    predCandleTime: document.getElementById("pred-candle-time"),
    predCountdown: document.getElementById("pred-countdown"),
    predTrendBadge: document.getElementById("pred-trend-badge"),
    predPhaseBadge: document.getElementById("pred-phase-badge"),
    predContextText: document.getElementById("pred-context-text"),
    predModelResult: document.getElementById("pred-model-result"),
    predProbDown: document.getElementById("pred-prob-down"),
    predProbNeu: document.getElementById("pred-prob-neu"),
    predProbUp: document.getElementById("pred-prob-up"),
    probBarDown: document.getElementById("prob-bar-down"),
    probBarNeu: document.getElementById("prob-bar-neu"),
    probBarUp: document.getElementById("prob-bar-up"),
    predFinalBadge: document.getElementById("pred-final-badge"),
    predConfidence: document.getElementById("pred-confidence"),
    predSmartPrice: document.getElementById("pred-smart-price"),
    predUpperCap: document.getElementById("pred-upper-cap"),
    predLowerCap: document.getElementById("pred-lower-cap"),
    predBandContainment: document.getElementById("pred-band-containment"),
    // Log
    logTableBody: document.getElementById("log-table-body"),
    logTabLive: document.getElementById("log-tab-live"),
    logTabReplay: document.getElementById("log-tab-replay"),
    logStatCalls: document.getElementById("log-stat-calls"),
    logStatAccuracy: document.getElementById("log-stat-accuracy"),
    logStatCoverage: document.getElementById("log-stat-coverage"),
    logStatStreak: document.getElementById("log-stat-streak"),
    logStatBenchmark: document.getElementById("log-stat-benchmark"),
    btnClearReplay: document.getElementById("btn-clear-replay"),
    btnExportCsv: document.getElementById("btn-export-csv"),
    // Replay Controls
    btnTestNow: document.getElementById("btn-test-now"),
    btnPlayPause: document.getElementById("btn-play-pause"),
    btnStepNext: document.getElementById("btn-step-next"),
    btnStepPrev: document.getElementById("btn-step-prev"),
    btnSpeed: document.getElementById("btn-speed"),
    btnResetReplay: document.getElementById("btn-reset-replay"),
    replayDatePicker: document.getElementById("replay-date-picker"),
    replayDateBadge: document.getElementById("replay-date-badge"),
  };
}

// =====================================================================
// CHART MANAGEMENT (Lightweight Charts 4.2.x)
// =====================================================================
const CHARTS = {
  live: { chart: null, candles: null, volume: null, upper: null, lower: null, smart: null, bbUpper: null, bbLower: null, lastPriceLine: null },
  replay: { chart: null, candles: null, volume: null, upper: null, lower: null, smart: null, bbUpper: null, bbLower: null },
  overview: { chart: null, trainLine: null, valLine: null, testLine: null, markerLine: null },
};

function createChartInstance(container) {
  return LightweightCharts.createChart(container, {
    layout: {
      background: { type: "solid", color: CONFIG.colors.bgChart },
      textColor: CONFIG.colors.textDim,
      fontSize: 11,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, sans-serif',
    },
    grid: {
      vertLines: { color: CONFIG.colors.grid, style: 2 },
      horzLines: { color: CONFIG.colors.grid, style: 2 },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: CONFIG.colors.textDim, width: 1, style: 3, labelBackgroundColor: CONFIG.colors.bgPanel },
      horzLine: { color: CONFIG.colors.textDim, width: 1, style: 3, labelBackgroundColor: CONFIG.colors.bgPanel },
    },
    timeScale: {
      borderColor: CONFIG.colors.border,
      timeVisible: true,
      secondsVisible: false,
    },
    rightPriceScale: {
      borderColor: CONFIG.colors.border,
      scaleMargins: { top: 0.1, bottom: 0.2 },
    },
  });
}

function initCharts() {
  // 1. Live Chart
  const liveContainer = document.getElementById("live-chart-container");
  CHARTS.live.chart = createChartInstance(liveContainer);

  CHARTS.live.candles = CHARTS.live.chart.addCandlestickSeries({
    upColor: CONFIG.colors.up,
    downColor: CONFIG.colors.down,
    borderVisible: false,
    wickUpColor: CONFIG.colors.up,
    wickDownColor: CONFIG.colors.down,
  });

  CHARTS.live.volume = CHARTS.live.chart.addHistogramSeries({
    color: "#26a69a",
    priceFormat: { type: "volume" },
    priceScaleId: "",
    scaleMargins: { top: 0.8, bottom: 0 },
  });

  // Indicative Range Lines (Horizon t+2 forecast)
  CHARTS.live.upper = CHARTS.live.chart.addLineSeries({ color: "rgba(239, 83, 80, 0.7)", lineWidth: 1, lineStyle: 2 });
  CHARTS.live.lower = CHARTS.live.chart.addLineSeries({ color: "rgba(38, 166, 154, 0.7)", lineWidth: 1, lineStyle: 2 });
  CHARTS.live.smart = CHARTS.live.chart.addLineSeries({ color: CONFIG.colors.accent, lineWidth: 2, lineStyle: 0 });

  // Classic Bollinger Bands (20, 2)
  CHARTS.live.bbUpper = CHARTS.live.chart.addLineSeries({ color: "rgba(255,255,255,0.25)", lineWidth: 1, lineStyle: 1, visible: false });
  CHARTS.live.bbLower = CHARTS.live.chart.addLineSeries({ color: "rgba(255,255,255,0.25)", lineWidth: 1, lineStyle: 1, visible: false });

  // Crosshair Handler for Legend
  CHARTS.live.chart.subscribeCrosshairMove((param) => {
    updateLegend("live", param);
  });

  // 2. Replay Candlestick Chart
  const replayContainer = document.getElementById("replay-chart-container");
  CHARTS.replay.chart = createChartInstance(replayContainer);

  CHARTS.replay.candles = CHARTS.replay.chart.addCandlestickSeries({
    upColor: CONFIG.colors.up,
    downColor: CONFIG.colors.down,
    borderVisible: false,
    wickUpColor: CONFIG.colors.up,
    wickDownColor: CONFIG.colors.down,
  });

  CHARTS.replay.volume = CHARTS.replay.chart.addHistogramSeries({
    color: "#26a69a",
    priceFormat: { type: "volume" },
    priceScaleId: "",
    scaleMargins: { top: 0.8, bottom: 0 },
  });

  CHARTS.replay.upper = CHARTS.replay.chart.addLineSeries({ color: "rgba(239, 83, 80, 0.7)", lineWidth: 1, lineStyle: 2 });
  CHARTS.replay.lower = CHARTS.replay.chart.addLineSeries({ color: "rgba(38, 166, 154, 0.7)", lineWidth: 1, lineStyle: 2 });
  CHARTS.replay.smart = CHARTS.replay.chart.addLineSeries({ color: CONFIG.colors.accent, lineWidth: 2, lineStyle: 0 });

  CHARTS.replay.bbUpper = CHARTS.replay.chart.addLineSeries({ color: "rgba(255,255,255,0.25)", lineWidth: 1, lineStyle: 1, visible: false });
  CHARTS.replay.bbLower = CHARTS.replay.chart.addLineSeries({ color: "rgba(255,255,255,0.25)", lineWidth: 1, lineStyle: 1, visible: false });

  CHARTS.replay.chart.subscribeCrosshairMove((param) => {
    updateLegend("replay", param);
  });

  // 3. Overview Strip Chart
  const overviewContainer = document.getElementById("overview-chart-container");
  CHARTS.overview.chart = LightweightCharts.createChart(overviewContainer, {
    layout: {
      background: { type: "solid", color: CONFIG.colors.bgApp },
      textColor: CONFIG.colors.textDim,
      fontSize: 9,
    },
    grid: { vertLines: { visible: false }, horzLines: { visible: false } },
    timeScale: { visible: true, borderColor: CONFIG.colors.border },
    rightPriceScale: { visible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });

  CHARTS.overview.trainLine = CHARTS.overview.chart.addLineSeries({ color: "#42a5f5", lineWidth: 1 });
  CHARTS.overview.valLine = CHARTS.overview.chart.addLineSeries({ color: "#ab47bc", lineWidth: 1 });
  CHARTS.overview.testLine = CHARTS.overview.chart.addLineSeries({ color: CONFIG.colors.up, lineWidth: 2 });

  // Overview click handler to jump replay
  CHARTS.overview.chart.subscribeClick((param) => {
    if (!param.time) return;
    handleOverviewJump(param.time);
  });

  // Auto-resize observers with requestAnimationFrame throttling
  setupResizeObservers([
    { container: liveContainer, chart: CHARTS.live.chart },
    { container: replayContainer, chart: CHARTS.replay.chart },
    { container: overviewContainer, chart: CHARTS.overview.chart },
  ]);
}

function setupResizeObservers(chartList) {
  chartList.forEach(({ container, chart }) => {
    let rAF = null;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) return;
      const { width, height } = entries[0].contentRect;
      if (width === 0 || height === 0) return;
      if (rAF) cancelAnimationFrame(rAF);
      rAF = requestAnimationFrame(() => {
        chart.applyOptions({ width, height });
      });
    });
    observer.observe(container);
  });
}

function updateLegend(view, param) {
  const ohlcEl = document.getElementById(`${view}-legend-ohlc`);
  if (!param || !param.time || !param.seriesData) {
    return;
  }
  const series = CHARTS[view].candles;
  const candle = param.seriesData.get(series);
  if (!candle) return;

  const isUp = candle.close >= candle.open;
  const cls = isUp ? "up" : "down";
  ohlcEl.innerHTML = `
    <span>O: <b class="legend-val ${cls}">${candle.open.toFixed(2)}</b></span>
    <span>H: <b class="legend-val ${cls}">${candle.high.toFixed(2)}</b></span>
    <span>L: <b class="legend-val ${cls}">${candle.low.toFixed(2)}</b></span>
    <span>C: <b class="legend-val ${cls}">${candle.close.toFixed(2)}</b></span>
  `;
}

// =====================================================================
// TIME & CLOCKS (Browser Local Timezone vs UTC)
// =====================================================================
function initClocks() {
  function tick() {
    const now = new Date();
    // Local Time
    const localTimeStr = now.toLocaleTimeString();
    const tzName = Intl.DateTimeFormat().resolvedOptions().timeZone || "Local";
    DOM.clockLocal.textContent = `${localTimeStr} (${tzName})`;

    // Server Time (UTC)
    const utcHours = String(now.getUTCHours()).padStart(2, "0");
    const utcMins = String(now.getUTCMinutes()).padStart(2, "0");
    const utcSecs = String(now.getUTCSeconds()).padStart(2, "0");
    DOM.clockServer.textContent = `UTC ${utcHours}:${utcMins}:${utcSecs}`;

    // Candle-close countdown (5-minute boundaries)
    const secsInto5m = (now.getUTCMinutes() % 5) * 60 + now.getUTCSeconds();
    const secsRemaining = 300 - secsInto5m;
    const mRem = Math.floor(secsRemaining / 60);
    const sRem = String(secsRemaining % 60).padStart(2, "0");
    DOM.predCountdown.textContent = `${mRem}:${sRem}`;

    // Watchdog check: 15s stale check
    if (STATE.activeTab === "live" && !STATE.isMock) {
      if (STATE.lastTickTime > 0 && Date.now() - STATE.lastTickTime > 15000) {
        setLiveStatus("STALE", "Data feed stale (>15s)");
      }
    }
  }
  setInterval(tick, 1000);
  tick();
}

function setLiveStatus(status, text) {
  STATE.liveStatus = status;
  DOM.statusDot.className = `status-dot ${status.toLowerCase()}`;
  DOM.statusText.textContent = text || status;

  if (status === "STALE") {
    DOM.staleBanner.style.display = "block";
    DOM.staleBanner.textContent = "⚠️ Stale data: No new market ticks received for >15 seconds.";
  } else {
    DOM.staleBanner.style.display = "none";
  }
}

// =====================================================================
// PREDICTION PANEL RENDERING
// =====================================================================
function renderPredictionPanel(pred) {
  if (!pred) return;
  STATE.lastPrediction = pred;

  // Flash card border briefly
  const card = document.querySelector(".prediction-panel");
  card.style.backgroundColor = "rgba(41, 98, 255, 0.08)";
  setTimeout(() => (card.style.backgroundColor = "transparent"), 150);

  // Time & Trend
  const localDate = new Date(pred.open_time).toLocaleTimeString();
  DOM.predCandleTime.textContent = localDate;

  // Trend Badge
  const trend = (pred.trend || "SIDEWAYS").toUpperCase();
  DOM.predTrendBadge.textContent = trend;
  DOM.predTrendBadge.className = `badge ${trend.toLowerCase()}`;

  // Phase Badge
  const phase = (pred.phase || "PULLBACK").toUpperCase();
  DOM.predPhaseBadge.textContent = phase;
  DOM.predPhaseBadge.className = `badge ${phase.toLowerCase()}`;

  DOM.predContextText.textContent = pred.context_text || `${trend} | ${phase}`;

  // Model Result (with tooltip)
  DOM.predModelResult.textContent = pred.model_result || "DOWN";
  DOM.predModelResult.title = "Argmax of model probabilities before threshold override rule";

  // Probabilities
  const pFinal = pred.p_final || { down: 0.33, neutral: 0.34, up: 0.33 };
  const pDn = Math.round(pFinal.down * 100);
  const pNeu = Math.round(pFinal.neutral * 100);
  const pUp = Math.round(pFinal.up * 100);

  DOM.predProbDown.textContent = `${pDn}%`;
  DOM.predProbNeu.textContent = `${pNeu}%`;
  DOM.predProbUp.textContent = `${pUp}%`;

  DOM.probBarDown.style.width = `${pDn}%`;
  DOM.probBarNeu.style.width = `${pNeu}%`;
  DOM.probBarUp.style.width = `${pUp}%`;

  // Final Action Badge
  const label = (pred.label || "NO CLEAR SIGNAL").toUpperCase();
  DOM.predFinalBadge.textContent = label;
  if (label === "UP") {
    DOM.predFinalBadge.className = "decision-badge-large badge up";
  } else if (label === "DOWN") {
    DOM.predFinalBadge.className = "decision-badge-large badge down";
  } else {
    DOM.predFinalBadge.className = "decision-badge-large badge neutral";
    if (pred.reason) {
      DOM.predFinalBadge.textContent = `NO SIGNAL (${pred.reason})`;
    }
  }

  DOM.predConfidence.textContent = `${Math.round((pred.confidence || 0.4) * 100)}% Confidence`;

  // Smart Price & Caps
  const refClose = pred.close || 0;
  if (pred.smart_price && refClose > 0) {
    const spDiff = (((pred.smart_price - refClose) / refClose) * 100).toFixed(2);
    DOM.predSmartPrice.textContent = `${pred.smart_price.toFixed(2)} (${spDiff > 0 ? "+" : ""}${spDiff}%)`;
  } else {
    DOM.predSmartPrice.textContent = "n/a";
  }

  if (pred.upper && refClose > 0) {
    const upDiff = (((pred.upper - refClose) / refClose) * 100).toFixed(2);
    DOM.predUpperCap.textContent = `${pred.upper.toFixed(2)} (+${upDiff}%)`;
  } else {
    DOM.predUpperCap.textContent = "n/a";
  }

  if (pred.lower && refClose > 0) {
    const dnDiff = (((pred.lower - refClose) / refClose) * 100).toFixed(2);
    DOM.predLowerCap.textContent = `${pred.lower.toFixed(2)} (${dnDiff}%)`;
  } else {
    DOM.predLowerCap.textContent = "n/a";
  }

  // Band Containment Stat
  updateBandContainmentStat();
}

function updateBandContainmentStat() {
  const list = STATE.logTab === "live" ? STATE.livePredictions : STATE.replayPredictions;
  let totalResolved = 0;
  let insideCount = 0;

  list.forEach((p) => {
    if (p.status === "RESOLVED" && p.outcome_close && p.upper && p.lower) {
      totalResolved++;
      if (p.outcome_close >= p.lower && p.outcome_close <= p.upper) {
        insideCount++;
      }
    }
  });

  if (totalResolved > 0) {
    const pct = Math.round((insideCount / totalResolved) * 100);
    DOM.predBandContainment.textContent = `Contained in ${pct}% of ${totalResolved} cases`;
  } else {
    DOM.predBandContainment.textContent = "Awaiting resolutions...";
  }
}

// =====================================================================
// PREDICTION LOG TABLE & SUMMARY METRICS
// =====================================================================
function renderPredictionLog() {
  const isLive = STATE.logTab === "live";
  const list = isLive ? STATE.livePredictions : STATE.replayPredictions;

  // Filter
  const filtered = list.filter((item) => {
    if (STATE.logFilter === "all") return true;
    if (STATE.logFilter === "correct") return item.correct === 1;
    if (STATE.logFilter === "wrong") return item.correct === 0;
    if (STATE.logFilter === "pending") return item.status === "PENDING";
    return true;
  });

  // Table Rows
  DOM.logTableBody.innerHTML = "";
  filtered.forEach((row) => {
    const tr = document.createElement("tr");

    let rowClass = "";
    let verdictIcon = "⏳ PENDING (1/2)";
    if (row.status === "RESOLVED") {
      if (row.correct === 1) {
        rowClass = "correct";
        verdictIcon = "✓ CORRECT";
      } else if (row.correct === 0) {
        rowClass = "wrong";
        verdictIcon = "✗ WRONG";
      }
    } else if (row.status === "NO_CALL") {
      verdictIcon = "— NO CALL";
    } else if (row.status === "VOID") {
      verdictIcon = "⊘ VOID (0 move)";
    }

    tr.className = rowClass;

    const timeStr = new Date(row.open_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const confPct = Math.round((row.confidence || 0) * 100);
    const pSignal = row.close ? Number(row.close).toFixed(2) : "—";
    const pOutcome = row.outcome_close ? Number(row.outcome_close).toFixed(2) : "—";

    tr.innerHTML = `
      <td class="tabular-nums">${timeStr}</td>
      <td><span class="badge ${row.label ? row.label.toLowerCase() : 'neutral'}">${row.label || 'NO CALL'}</span></td>
      <td class="tabular-nums">${confPct}%</td>
      <td>${row.trend || '—'} / ${row.phase || '—'}</td>
      <td class="tabular-nums">${pSignal}</td>
      <td class="tabular-nums">${pOutcome}</td>
      <td><b>${verdictIcon}</b></td>
    `;

    // Click row -> scroll chart to that candle
    tr.addEventListener("click", () => {
      const chartObj = isLive ? CHARTS.live.chart : CHARTS.replay.chart;
      const ts = Math.floor(new Date(row.open_time).getTime() / 1000);
      chartObj.timeScale().scrollToPosition(0, true);
    });

    DOM.logTableBody.appendChild(tr);
  });

  // Summary Metrics Strip
  updateLogSummaryMetrics(list);
}

function updateLogSummaryMetrics(list) {
  let calls = 0;
  let correct = 0;
  let wrong = 0;
  let streak = 0;
  let streakType = null;

  list.forEach((p) => {
    if (p.label === "UP" || p.label === "DOWN") {
      calls++;
      if (p.correct === 1) {
        correct++;
        if (streakType === "win") streak++;
        else { streakType = "win"; streak = 1; }
      } else if (p.correct === 0) {
        wrong++;
        if (streakType === "loss") streak++;
        else { streakType = "loss"; streak = 1; }
      }
    }
  });

  const resolved = correct + wrong;
  const acc = resolved > 0 ? ((correct / resolved) * 100).toFixed(1) : "—";
  const cov = list.length > 0 ? Math.round((calls / list.length) * 100) : 0;

  DOM.logStatCalls.textContent = `${calls} (${resolved} resolved)`;
  DOM.logStatAccuracy.textContent = resolved > 0 ? `${acc}% ${resolved < 30 ? "(wide CI, n<30)" : ""}` : "—";
  DOM.logStatCoverage.textContent = `${cov}%`;
  DOM.logStatStreak.textContent = streak > 0 ? `${streak} ${streakType.toUpperCase()}` : "0";

  // Benchmarks comparison
  DOM.logStatBenchmark.textContent = `Ref: Held-out Test ${CONFIG.benchmarks.refTestAccuracy}% | OOF ${CONFIG.benchmarks.refOofAccuracy}%`;
}

// =====================================================================
// OVERLAYS & MARKERS
// =====================================================================
function attachPredictionToChart(view, pred) {
  const chartGroup = CHARTS[view];
  const candleTime = Math.floor(new Date(pred.open_time).getTime() / 1000);

  // 1. Forecast Lines at t + 2 candles (10 minutes forward)
  const targetTime = candleTime + CONFIG.horizonCandles * CONFIG.intervalMinutes * 60;

  if (STATE.layerToggles.bands && pred.upper && pred.lower) {
    chartGroup.upper.update({ time: targetTime, value: pred.upper });
    chartGroup.lower.update({ time: targetTime, value: pred.lower });
  }

  if (STATE.layerToggles.smartPrice && pred.smart_price) {
    chartGroup.smart.update({ time: targetTime, value: pred.smart_price });
  }

  // 2. Markers on closed candle
  updateAllMarkers(view);
}

function updateAllMarkers(view) {
  if (!STATE.layerToggles.markers) {
    CHARTS[view].candles.setMarkers([]);
    return;
  }

  const list = view === "live" ? STATE.livePredictions : STATE.replayPredictions;
  const markers = [];

  list.forEach((pred) => {
    const time = Math.floor(new Date(pred.open_time).getTime() / 1000);
    const label = pred.label || "NO CLEAR SIGNAL";
    const conf = Math.round((pred.confidence || 0) * 100);

    if (label === "UP") {
      markers.push({
        time: time,
        position: "belowBar",
        color: CONFIG.colors.up,
        shape: "arrowUp",
        text: `${conf}%`,
      });
    } else if (label === "DOWN") {
      markers.push({
        time: time,
        position: "aboveBar",
        color: CONFIG.colors.down,
        shape: "arrowDown",
        text: `${conf}%`,
      });
    } else {
      markers.push({
        time: time,
        position: "inBar",
        color: CONFIG.colors.neutral,
        shape: "circle",
        size: 0.5,
      });
    }
  });

  // Sort markers chronologically (Lightweight Charts requirement)
  markers.sort((a, b) => a.time - b.time);
  CHARTS[view].candles.setMarkers(markers);
}

// =====================================================================
// LIVE STREAMING & BACKEND SYNC
// =====================================================================
async function initLiveMode() {
  setLiveStatus("RECONNECTING", "Connecting to live feed...");

  if (STATE.isMock) {
    setupMockData();
    return;
  }

  try {
    // 1. Fetch system config from dashboard_api
    const cfgResp = await fetch(`${CONFIG.dashApiBase}/dash/config`);
    if (cfgResp.ok) {
      const cfg = await cfgResp.json();
      CONFIG.horizonCandles = cfg.horizon_candles;
      CONFIG.horizonMinutes = cfg.horizon_minutes;
      CONFIG.costFloorBps = cfg.cost_floor_bps;
      CONFIG.bufferSize = cfg.buffer_size;
      CONFIG.benchmarks = {
        refTestAccuracy: cfg.benchmarks.ref_test_accuracy_pct,
        refTestCoverage: cfg.benchmarks.ref_test_coverage_pct,
        refOofAccuracy: cfg.benchmarks.ref_oof_accuracy_pct,
        refOofCoverage: cfg.benchmarks.ref_oof_coverage_pct,
      };
    }

    // 2. Fetch live candles from /dash/live/candles
    const candlesResp = await fetch(`${CONFIG.dashApiBase}/dash/live/candles?limit=1000`);
    if (candlesResp.ok) {
      const rawCandles = await candlesResp.json();
      if (rawCandles.length > 0) {
        loadCandlesIntoChart("live", rawCandles);
      } else {
        // Fallback to Binance REST API
        await fetchBinanceRestKlines();
      }
    } else {
      await fetchBinanceRestKlines();
    }

    // 3. Fetch recent enriched predictions
    const predsResp = await fetch(`${CONFIG.dashApiBase}/dash/live/predictions?limit=200`);
    if (predsResp.ok) {
      const preds = await predsResp.json();
      STATE.livePredictions = preds;
      if (preds.length > 0) {
        renderPredictionPanel(preds[0]);
        preds.forEach((p) => attachPredictionToChart("live", p));
      }
      renderPredictionLog();
    }

    // 4. Connect to Backend WebSocket
    connectBackendWebSocket();

    // 5. Connect to Binance WebSocket for forming candle ticks
    connectBinanceWebSocket();
  } catch (err) {
    console.warn("[Live] Failed to connect to local backend, using Binance directly / Mock:", err);
    await fetchBinanceRestKlines();
    connectBinanceWebSocket();
    setLiveStatus("MOCK", "Binance Spot Live (No Backend)");
  }
}

async function fetchBinanceRestKlines() {
  try {
    const url = `${CONFIG.binanceRestUrl}?symbol=${CONFIG.symbol}&interval=${CONFIG.timeframe}&limit=500`;
    const resp = await fetch(url);
    const data = await resp.json();
    const candles = data.map((d) => ({
      open_time: new Date(d[0]).toISOString(),
      time: Math.floor(d[0] / 1000),
      open: parseFloat(d[1]),
      high: parseFloat(d[2]),
      low: parseFloat(d[3]),
      close: parseFloat(d[4]),
      volume: parseFloat(d[5]),
    }));
    loadCandlesIntoChart("live", candles);
  } catch (e) {
    console.error("Failed to load Binance REST klines:", e);
  }
}

function loadCandlesIntoChart(view, rawCandles) {
  const formattedCandles = [];
  const formattedVolumes = [];

  rawCandles.forEach((c) => {
    const time = c.time || Math.floor(new Date(c.open_time).getTime() / 1000);
    const o = Number(c.open);
    const h = Number(c.high);
    const l = Number(c.low);
    const cl = Number(c.close);
    const v = Number(c.volume);

    formattedCandles.push({ time, open: o, high: h, low: l, close: cl });
    formattedVolumes.push({
      time,
      value: v,
      color: cl >= o ? "rgba(38, 166, 154, 0.4)" : "rgba(239, 83, 80, 0.4)",
    });
  });

  // Sort and deduplicate by time
  formattedCandles.sort((a, b) => a.time - b.time);
  formattedVolumes.sort((a, b) => a.time - b.time);

  CHARTS[view].candles.setData(formattedCandles);
  CHARTS[view].volume.setData(formattedVolumes);
  STATE[`${view}Candles`] = formattedCandles;
}

function connectBackendWebSocket() {
  const ws = new WebSocket(CONFIG.wsUrl);

  ws.onopen = () => {
    console.log("[Backend WS] Connected to live prediction stream.");
    setLiveStatus("LIVE", "Live Forward Test");
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleBackendWsEvent(msg);
    } catch (e) {
      console.error("WS Parse error:", e);
    }
  };

  ws.onerror = () => {
    setLiveStatus("RECONNECTING", "Reconnecting to backend...");
  };

  ws.onclose = () => {
    setTimeout(connectBackendWebSocket, 5000);
  };
}

function handleBackendWsEvent(msg) {
  STATE.lastTickTime = Date.now();
  if (msg.event === "latest_prediction" || msg.event === "prediction_logged") {
    const pred = msg.data;
    // Prevent duplicate open_time
    const exists = STATE.livePredictions.some((p) => p.open_time === pred.candle_open_time);
    if (!exists) {
      const mapped = {
        open_time: pred.candle_open_time,
        close: pred.ref_close,
        trend: CONFIG.vocabulary.trendMap[pred.trend] || pred.trend,
        phase: CONFIG.vocabulary.phaseMap[pred.market_phase] || pred.market_phase,
        label: CONFIG.vocabulary.actionMap[pred.action_label] || pred.action_label,
        confidence: pred.confidence,
        p_final: { down: pred.p_down, neutral: pred.p_neutral, up: pred.p_up },
        status: pred.status,
        reason: pred.status_reason,
      };
      STATE.livePredictions.unshift(mapped);
      renderPredictionPanel(mapped);
      attachPredictionToChart("live", mapped);
      renderPredictionLog();
    }
  } else if (msg.event === "prediction_resolved") {
    const res = msg.data;
    const match = STATE.livePredictions.find((p) => p.open_time === res.candle_open_time);
    if (match) {
      match.status = "RESOLVED";
      match.correct = res.correct;
      match.outcome_close = res.outcome_close;
      match.return_bps_label = res.return_bps_label;
      renderPredictionLog();
      updateBandContainmentStat();
    }
  }
}

function connectBinanceWebSocket() {
  const ws = new WebSocket(CONFIG.binanceWsUrl);

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (!data.k) return;
      STATE.lastTickTime = Date.now();

      const k = data.k;
      const candle = {
        time: Math.floor(k.t / 1000),
        open: parseFloat(k.o),
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        close: parseFloat(k.c),
      };

      CHARTS.live.candles.update(candle);
      CHARTS.live.volume.update({
        time: candle.time,
        value: parseFloat(k.v),
        color: candle.close >= candle.open ? "rgba(38, 166, 154, 0.4)" : "rgba(239, 83, 80, 0.4)",
      });

      // Update Red dashed last-price line
      updateLastPriceLine("live", candle.close);
    } catch (err) {
      console.error("Binance WS error:", err);
    }
  };

  ws.onclose = () => {
    setTimeout(connectBinanceWebSocket, 3000);
  };
}

function updateLastPriceLine(view, price) {
  if (!CHARTS[view].lastPriceLine) {
    CHARTS[view].lastPriceLine = CHARTS[view].candles.createPriceLine({
      price: price,
      color: CONFIG.colors.down,
      lineWidth: 1,
      lineStyle: 2, // Dashed
      axisLabelVisible: true,
      title: "CLOSE",
    });
  } else {
    CHARTS[view].lastPriceLine.applyOptions({ price: price });
  }
}

// =====================================================================
// REPLAY ENGINE (Held-Out Test Split)
// =====================================================================
async function initReplayMode() {
  if (STATE.isMock) {
    const mockOverview = [];
    let p = 70000;
    const nowSec = Math.floor(Date.now() / 1000);
    for (let i = 0; i < 500; i++) {
      p += (Math.random() - 0.49) * 120;
      mockOverview.push({
        time: nowSec - (500 - i) * 3600,
        close: p,
        split: i < 250 ? "train" : (i < 380 ? "val" : "test"),
      });
    }
    renderOverviewChart(mockOverview);
    STATE.replayTestRange = {
      min_open_time: "2026-05-20T14:20:00.000Z",
      max_open_time: "2026-11-09T23:55:00.000Z",
    };
    setupReplayControls();

    const mockReplayCandles = [];
    let rcPrice = 85000;
    const startSec = Math.floor(new Date("2026-05-20T14:20:00.000Z").getTime() / 1000);
    for (let i = 0; i < 200; i++) {
      const t = startSec + i * 300;
      rcPrice += (Math.random() - 0.49) * 90;
      mockReplayCandles.push({
        time: t,
        open_time: new Date(t * 1000).toISOString(),
        open: rcPrice - 20,
        high: rcPrice + 40,
        low: rcPrice - 40,
        close: rcPrice,
        volume: 80 + Math.random() * 100,
      });
    }
    loadCandlesIntoChart("replay", mockReplayCandles);
    return;
  }

  try {
    // 1. Fetch overview downsampled history
    const ovResp = await fetch(`${CONFIG.dashApiBase}/dash/overview`);
    if (ovResp.ok) {
      const points = await ovResp.json();
      renderOverviewChart(points);
    }

    // 2. Fetch TEST range boundaries
    const rangeResp = await fetch(`${CONFIG.dashApiBase}/dash/replay/range`);
    if (rangeResp.ok) {
      STATE.replayTestRange = await rangeResp.json();
      setupReplayControls();
    }
  } catch (err) {
    console.warn("[Replay] Failed to load from backend:", err);
  }
}

function renderOverviewChart(points) {
  const trainPts = [];
  const valPts = [];
  const testPts = [];

  points.forEach((p) => {
    if (p.split === "train") trainPts.push({ time: p.time, value: p.close });
    else if (p.split === "val") valPts.push({ time: p.time, value: p.close });
    else testPts.push({ time: p.time, value: p.close });
  });

  CHARTS.overview.trainLine.setData(trainPts);
  CHARTS.overview.valLine.setData(valPts);
  CHARTS.overview.testLine.setData(testPts);
}

function setupReplayControls() {
  if (!STATE.replayTestRange) return;
  const minIso = STATE.replayTestRange.min_open_time;
  const maxIso = STATE.replayTestRange.max_open_time;

  DOM.replayDatePicker.min = minIso.substring(0, 16);
  DOM.replayDatePicker.max = maxIso.substring(0, 16);
  DOM.replayDatePicker.value = minIso.substring(0, 16);
  DOM.replayDateBadge.textContent = "TEST SPLIT";

  // Load initial slice around test start
  if (!STATE.isMock) {
    loadReplaySlice(minIso);
  }
}

async function loadReplaySlice(fromIso) {
  try {
    const fromDt = new Date(fromIso);
    const toDt = new Date(fromDt.getTime() + 300 * 5 * 60 * 1000);

    const resp = await fetch(
      `${CONFIG.dashApiBase}/dash/replay/candles?from=${encodeURIComponent(fromDt.toISOString())}&to=${encodeURIComponent(toDt.toISOString())}`
    );
    if (!resp.ok) return;

    const candles = await resp.json();
    loadCandlesIntoChart("replay", candles);
    STATE.replayCurrentIndex = 0;
  } catch (err) {
    console.error("Failed to load replay slice:", err);
  }
}

async function executeReplayStep() {
  const candles = STATE.replayCandles;
  if (!candles || candles.length === 0) return;
  if (STATE.replayCurrentIndex >= candles.length) {
    STATE.replayCurrentIndex = 0;
  }

  const currentCandle = candles[STATE.replayCurrentIndex];
  STATE.replayCurrentIndex++;

  const ts = currentCandle.open_time || new Date(currentCandle.time * 1000).toISOString();

  if (STATE.isMock) {
    const isUp = Math.random() > 0.48;
    const pUp = isUp ? 0.43 : 0.29;
    const pDn = isUp ? 0.27 : 0.41;
    const pNeu = 0.30;
    const pred = {
      open_time: ts,
      close: currentCandle.close,
      trend: isUp ? "UPTREND" : "DOWNTREND",
      phase: Math.random() > 0.5 ? "BREAKTHROUGH" : "PULLBACK",
      label: isUp ? "UP" : "DOWN",
      confidence: Math.max(pUp, pDn),
      p_final: { up: pUp, neutral: pNeu, down: pDn },
      smart_price: currentCandle.close + (isUp ? 75 : -75),
      upper: currentCandle.close + 160,
      lower: currentCandle.close - 160,
      status: "PENDING",
      correct: null,
      outcome_close: null,
    };
    STATE.replayPredictions.unshift(pred);
    if (STATE.replayPredictions.length > 2) {
      const older = STATE.replayPredictions[2];
      older.status = "RESOLVED";
      older.outcome_close = currentCandle.close;
      older.correct = older.label === "UP" ? (currentCandle.close > older.close ? 1 : 0) : (currentCandle.close < older.close ? 1 : 0);
    }
    renderPredictionPanel(pred);
    attachPredictionToChart("replay", pred);
    renderPredictionLog();
    return;
  }

  try {
    if (STATE.abortController) STATE.abortController.abort();
    STATE.abortController = new AbortController();

    const resp = await fetch(`${CONFIG.dashApiBase}/dash/replay/predict?ts=${encodeURIComponent(ts)}`, {
      signal: STATE.abortController.signal,
    });
    if (!resp.ok) return;

    const pred = await resp.json();
    STATE.replayPredictions.unshift(pred);
    renderPredictionPanel(pred);
    attachPredictionToChart("replay", pred);
    renderPredictionLog();
  } catch (e) {
    if (e.name !== "AbortError") console.error("Replay step error:", e);
  }
}

function togglePlayPause() {
  STATE.replayPlaying = !STATE.replayPlaying;
  DOM.btnPlayPause.textContent = STATE.replayPlaying ? "⏸ Pause" : "▶ Play";

  if (STATE.replayPlaying) {
    const delay = 1000 / STATE.replaySpeed;
    STATE.replayTimer = setInterval(executeReplayStep, delay);
  } else {
    clearInterval(STATE.replayTimer);
  }
}

// =====================================================================
// MOCK DEMO MODE (?mock=1)
// =====================================================================
function setupMockData() {
  setLiveStatus("MOCK", "Demo Mode (Mock)");
  const baseTime = Math.floor(Date.now() / 1000) - 300 * 300;
  let price = 91000.0;
  const mockCandles = [];

  for (let i = 0; i < 300; i++) {
    const t = baseTime + i * 300;
    const change = (Math.random() - 0.49) * 80;
    const o = price;
    const cl = price + change;
    const h = Math.max(o, cl) + Math.random() * 20;
    const l = Math.min(o, cl) - Math.random() * 20;
    const v = 50 + Math.random() * 150;
    price = cl;

    mockCandles.push({
      time: t,
      open_time: new Date(t * 1000).toISOString(),
      open: o,
      high: h,
      low: l,
      close: cl,
      volume: v,
    });
  }

  loadCandlesIntoChart("live", mockCandles);

  // Generate 20 mock predictions
  for (let i = 280; i < 300; i++) {
    const c = mockCandles[i];
    const isUp = Math.random() > 0.5;
    const pUp = isUp ? 0.44 : 0.28;
    const pDn = isUp ? 0.26 : 0.42;
    const pNeu = 0.3;

    const mockRow = {
      open_time: c.open_time,
      close: c.close,
      trend: isUp ? "UPTREND" : "DOWNTREND",
      phase: Math.random() > 0.5 ? "BREAKTHROUGH" : "PULLBACK",
      label: isUp ? "UP" : "DOWN",
      confidence: Math.max(pUp, pDn),
      p_final: { up: pUp, neutral: pNeu, down: pDn },
      smart_price: c.close + (isUp ? 80 : -80),
      upper: c.close + 180,
      lower: c.close - 180,
      status: i < 298 ? "RESOLVED" : "PENDING",
      correct: i < 298 ? (Math.random() > 0.45 ? 1 : 0) : null,
      outcome_close: i < 298 ? c.close + (isUp ? 40 : -40) : null,
    };
    STATE.livePredictions.unshift(mockRow);
    STATE.replayPredictions.unshift({ ...mockRow });
  }

  renderPredictionPanel(STATE.livePredictions[0]);
  STATE.livePredictions.forEach((p) => attachPredictionToChart("live", p));
  renderPredictionLog();
}

// =====================================================================
// EVENT LISTENERS & INITIALIZATION
// =====================================================================
function setupEventListeners() {
  // Tab Switch (Live vs Replay)
  DOM.btnLive.addEventListener("click", () => {
    STATE.activeTab = "live";
    DOM.btnLive.classList.add("active");
    DOM.btnReplay.classList.remove("active");
    DOM.liveView.style.display = "flex";
    DOM.replayView.style.display = "none";
  });

  DOM.btnReplay.addEventListener("click", () => {
    STATE.activeTab = "replay";
    DOM.btnReplay.classList.add("active");
    DOM.btnLive.classList.remove("active");
    DOM.replayView.style.display = "flex";
    DOM.liveView.style.display = "none";
    initReplayMode();
  });

  // Guide Drawer
  DOM.btnGuide.addEventListener("click", () => DOM.guideDrawer.classList.add("open"));
  DOM.btnCloseGuide.addEventListener("click", () => DOM.guideDrawer.classList.remove("open"));

  // Log Tab Switch
  DOM.logTabLive.addEventListener("click", () => {
    STATE.logTab = "live";
    DOM.logTabLive.classList.add("active");
    DOM.logTabReplay.classList.remove("active");
    renderPredictionLog();
  });

  DOM.logTabReplay.addEventListener("click", () => {
    STATE.logTab = "replay";
    DOM.logTabReplay.classList.add("active");
    DOM.logTabLive.classList.remove("active");
    renderPredictionLog();
  });

  // Log Filter Pills
  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.addEventListener("click", (e) => {
      document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
      e.target.classList.add("active");
      STATE.logFilter = e.target.getAttribute("data-filter");
      renderPredictionLog();
    });
  });

  // Replay Controls
  DOM.btnTestNow.addEventListener("click", executeReplayStep);
  DOM.btnPlayPause.addEventListener("click", togglePlayPause);
  DOM.btnStepNext.addEventListener("click", executeReplayStep);
  DOM.btnSpeed.addEventListener("click", () => {
    const speeds = [1, 2, 5, 10];
    const nextIdx = (speeds.indexOf(STATE.replaySpeed) + 1) % speeds.length;
    STATE.replaySpeed = speeds[nextIdx];
    DOM.btnSpeed.textContent = `${STATE.replaySpeed}x`;
    if (STATE.replayPlaying) {
      clearInterval(STATE.replayTimer);
      STATE.replayTimer = setInterval(executeReplayStep, 1000 / STATE.replaySpeed);
    }
  });

  // Layer Toggles
  document.querySelectorAll(".layer-toggle input").forEach((input) => {
    input.addEventListener("change", (e) => {
      const layer = e.target.getAttribute("data-layer");
      STATE.layerToggles[layer] = e.target.checked;
      if (layer === "markers") {
        updateAllMarkers(STATE.activeTab);
      } else if (layer === "classicBb") {
        CHARTS.live.bbUpper.applyOptions({ visible: e.target.checked });
        CHARTS.live.bbLower.applyOptions({ visible: e.target.checked });
      }
    });
  });

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === "ArrowRight") {
      executeReplayStep();
    }
  });
}

// Bootstrap
window.addEventListener("DOMContentLoaded", () => {
  initDOM();
  initCharts();
  initClocks();
  setupEventListeners();
  initLiveMode();
});

