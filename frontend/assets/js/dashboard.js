/**
 * AI Trader Dashboard Orchestrator
 * Independent Native Canvas BTC/USDT 5m AI Indicator UI
 * 
 * Plots historical data from clean/master CSV + live Binance stream.
 * Predicts 2 candles into the future (horizon = 10 mins) and verifies
 * automatically after 10 minutes when candle k+2 closes.
 * 
 * Free of any TradingView libraries, logos, or watermarks.
 */

// 1. Central Configuration
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
    bgChart: "#131722",
    border: "#2a2e39",
    grid: "rgba(42, 46, 57, 0.4)",
    text: "#d1d4dc",
    textDim: "#787b86",
  },
};

// 2. Global State Store
const STATE = {
  isMock: new URLSearchParams(window.location.search).get("mock") === "1",
  activeTab: "live", // "live" | "replay"
  logTab: "live",
  logFilter: "all",
  liveStatus: "OFFLINE",
  lastTickTime: 0,
  lastPrediction: null,
  liveCandles: [],
  livePredictions: [],
  replayCandles: [],
  replayPredictions: [],
  replayTestRange: null,
  replayCurrentIndex: 0,
  replayPlaying: false,
  replaySpeed: 1,
  replayTimer: null,
  abortController: null,
  layerToggles: {
    markers: true,
    bands: true,
    smartPrice: true,
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
    // Log Table
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
// CHART MANAGEMENT (Native Canvas Candlestick Engine)
// =====================================================================
const CHARTS = {
  live: null,
  replay: null,
};

function initCharts() {
  CHARTS.live = new NativeCandleChart("live-chart-container", {
    upColor: CONFIG.colors.up,
    downColor: CONFIG.colors.down,
    bgColor: CONFIG.colors.bgChart || "#131722",
  });
  CHARTS.live.onCrosshairMove = (candle) => updateLegend("live", candle);

  CHARTS.replay = new NativeCandleChart("replay-chart-container", {
    upColor: CONFIG.colors.up,
    downColor: CONFIG.colors.down,
    bgColor: CONFIG.colors.bgChart || "#131722",
  });
  CHARTS.replay.onCrosshairMove = (candle) => updateLegend("replay", candle);

  const btnRealtime = document.getElementById("btn-realtime");
  if (btnRealtime) {
    btnRealtime.addEventListener("click", () => {
      if (CHARTS.live) CHARTS.live.scrollToRealtime();
    });
  }
}

function updateLegend(view, candle) {
  const ohlcEl = document.getElementById(`${view}-legend-ohlc`);
  if (!candle || !ohlcEl) return;

  const isUp = candle.close >= candle.open;
  const cls = isUp ? "up" : "down";
  ohlcEl.innerHTML = `
    <span>O: <b class="legend-val ${cls}">${Number(candle.open).toFixed(2)}</b></span>
    <span>H: <b class="legend-val ${cls}">${Number(candle.high).toFixed(2)}</b></span>
    <span>L: <b class="legend-val ${cls}">${Number(candle.low).toFixed(2)}</b></span>
    <span>C: <b class="legend-val ${cls}">${Number(candle.close).toFixed(2)}</b></span>
  `;
}

// =====================================================================
// TIME & CLOCKS (Local Timezone & 5m Candle Countdown)
// =====================================================================
function initClocks() {
  function tick() {
    const now = new Date();
    // Local Time
    const localTimeStr = now.toLocaleTimeString();
    const tzName = Intl.DateTimeFormat().resolvedOptions().timeZone || "Local";
    if (DOM.clockLocal) DOM.clockLocal.textContent = `${localTimeStr} (${tzName})`;

    // Server Time (UTC)
    const utcHours = String(now.getUTCHours()).padStart(2, "0");
    const utcMins = String(now.getUTCMinutes()).padStart(2, "0");
    const utcSecs = String(now.getUTCSeconds()).padStart(2, "0");
    if (DOM.clockServer) DOM.clockServer.textContent = `UTC ${utcHours}:${utcMins}:${utcSecs}`;

    // Candle-close countdown (5-minute boundaries)
    const secsInto5m = (now.getUTCMinutes() % 5) * 60 + now.getUTCSeconds();
    const secsRemaining = 300 - secsInto5m;
    const mRem = Math.floor(secsRemaining / 60);
    const sRem = String(secsRemaining % 60).padStart(2, "0");
    if (DOM.predCountdown) DOM.predCountdown.textContent = `${mRem}:${sRem}`;

    // Watchdog check: 20s stale check
    if (STATE.activeTab === "live" && !STATE.isMock) {
      if (STATE.lastTickTime > 0 && Date.now() - STATE.lastTickTime > 20000) {
        setLiveStatus("STALE", "Data feed idle (>20s)");
      }
    }
  }
  setInterval(tick, 1000);
  tick();
}

function setLiveStatus(status, text) {
  STATE.liveStatus = status;
  if (!DOM.statusDot) return;
  DOM.statusDot.className = `status-dot ${status.toLowerCase()}`;
  DOM.statusText.textContent = text || status;

  if (status === "STALE") {
    DOM.staleBanner.style.display = "block";
    DOM.staleBanner.textContent = "⚠️ Stale data: No market ticks received for >20 seconds.";
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

  // Flash card briefly
  const card = document.querySelector(".prediction-panel");
  if (card) {
    card.style.backgroundColor = "rgba(41, 98, 255, 0.08)";
    setTimeout(() => (card.style.backgroundColor = "transparent"), 150);
  }

  // Time & Trend
  const localDate = new Date(pred.open_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
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

  // Model Result
  DOM.predModelResult.textContent = pred.model_result || "DOWN";
  DOM.predModelResult.title = "Argmax of model probabilities before threshold override rule";

  // Probabilities
  const pFinal = pred.p_final || { down: 0.33, neutral: 0.34, up: 0.33 };
  const pDn = Math.round((pFinal.down || 0.33) * 100);
  const pNeu = Math.round((pFinal.neutral || 0.34) * 100);
  const pUp = Math.round((pFinal.up || 0.33) * 100);

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

  // Smart Price & Indicative Range
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
// PREDICTION LOG TABLE & 10-MINUTE VERIFICATION DISPLAY
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

    DOM.logTableBody.appendChild(tr);
  });

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
  DOM.logStatAccuracy.textContent = resolved > 0 ? `${acc}%` : "—";
  DOM.logStatCoverage.textContent = `${cov}%`;
  DOM.logStatStreak.textContent = streak > 0 ? `${streak} ${streakType.toUpperCase()}` : "0";
  DOM.logStatBenchmark.textContent = `Ref: Held-out Test ${CONFIG.benchmarks.refTestAccuracy}% | OOF ${CONFIG.benchmarks.refOofAccuracy}%`;
}

// =====================================================================
// OVERLAYS & MARKERS
// =====================================================================
function attachPredictionToChart(view, pred) {
  const chartGroup = CHARTS[view];
  if (!chartGroup || !pred) return;

  if (pred.smart_price || pred.upper || pred.lower) {
    chartGroup.setIndicativeBands({
      smartPrice: { value: pred.smart_price },
      upper: { value: pred.upper },
      lower: { value: pred.lower },
    });
  }

  updateAllMarkers(view);
}

function updateAllMarkers(view) {
  const chartObj = CHARTS[view];
  if (!chartObj) return;

  if (!STATE.layerToggles.markers) {
    chartObj.setMarkers([]);
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
        color: CONFIG.colors.up,
        shape: "arrowUp",
        text: `${conf}%`,
      });
    } else if (label === "DOWN") {
      markers.push({
        time: time,
        color: CONFIG.colors.down,
        shape: "arrowDown",
        text: `${conf}%`,
      });
    } else {
      markers.push({
        time: time,
        color: CONFIG.colors.neutral,
        shape: "circle",
      });
    }
  });

  chartObj.setMarkers(markers);
}

// =====================================================================
// LIVE STREAMING, MASTER DATA PLOTTING & 10-MIN VERIFICATION LOOP
// =====================================================================
async function initLiveMode() {
  setLiveStatus("RECONNECTING", "Loading master historical data...");

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
    }

    // 2. Fetch historical candles from master dataset (2,000 candles)
    let loaded = false;
    try {
      const histResp = await fetch(`${CONFIG.dashApiBase}/dash/candles/history?limit=2000`);
      if (histResp.ok) {
        const rawCandles = await histResp.json();
        if (rawCandles && rawCandles.length > 0) {
          loadCandlesIntoChart("live", rawCandles);
          loaded = true;
          console.log(`[Master Data] Loaded ${rawCandles.length} candles from master file.`);
        }
      }
    } catch (e) {
      console.warn("Could not load from /dash/candles/history:", e);
    }

    if (!loaded) {
      await fetchBinanceRestKlines();
    }

    // 3. Fetch recent enriched predictions from ledger
    try {
      const predsResp = await fetch(`${CONFIG.dashApiBase}/dash/live/predictions?limit=200`);
      if (predsResp.ok) {
        const preds = await predsResp.json();
        if (preds && preds.length > 0) {
          STATE.livePredictions = preds;
          renderPredictionPanel(preds[0]);
          attachPredictionToChart("live", preds[0]);
        }
      }
    } catch (e) {
      console.warn("Could not load ledger predictions:", e);
    }

    // 4. Start live prediction immediately on startup from latest closed candle
    if (!STATE.livePredictions || STATE.livePredictions.length === 0) {
      await executeInstantLivePrediction();
    }

    renderPredictionLog();

    // 5. Connect to live WebSockets (Backend + Binance Spot stream)
    connectBackendWebSocket();
    connectBinanceWebSocket();
    setLiveStatus("LIVE", "Live Forward Test (Active)");
  } catch (err) {
    console.warn("[Live] Backend connection error, running standalone:", err);
    await fetchBinanceRestKlines();
    connectBinanceWebSocket();
    await executeInstantLivePrediction();
    setLiveStatus("LIVE", "Binance Spot Live");
  }
}

async function fetchBinanceRestKlines() {
  try {
    const url = `${CONFIG.binanceRestUrl}?symbol=${CONFIG.symbol}&interval=${CONFIG.timeframe}&limit=1000`;
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

  rawCandles.forEach((c) => {
    const time = c.time || Math.floor(new Date(c.open_time).getTime() / 1000);
    const o = Number(c.open);
    const h = Number(c.high);
    const l = Number(c.low);
    const cl = Number(c.close);
    const v = Number(c.volume);

    formattedCandles.push({
      time,
      open: o,
      high: h,
      low: l,
      close: cl,
      volume: v,
      open_time: c.open_time || new Date(time * 1000).toISOString(),
    });
  });

  formattedCandles.sort((a, b) => a.time - b.time);
  STATE[`${view}Candles`] = formattedCandles;

  if (CHARTS[view]) {
    CHARTS[view].setData(formattedCandles);
  }
}

function connectBackendWebSocket() {
  const ws = new WebSocket(CONFIG.wsUrl);

  ws.onopen = () => {
    console.log("[Backend WS] Connected to live engine stream.");
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
        open_time: new Date(k.t).toISOString(),
        open: parseFloat(k.o),
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        close: parseFloat(k.c),
        volume: parseFloat(k.v),
      };

      // Live update into Native Canvas Chart
      if (CHARTS.live) {
        CHARTS.live.updateCandle(candle);
      }

      // 10-Minute Automatic Verification & Forward Prediction on Candle Close
      if (k.x === true) {
        console.log(`[5m Candle Closed] ${candle.open_time} Close: $${candle.close}`);
        handleCandleClosed(candle);
      }
    } catch (err) {
      console.error("Binance WS error:", err);
    }
  };

  ws.onclose = () => {
    setTimeout(connectBinanceWebSocket, 3000);
  };
}

function handleCandleClosed(candle) {
  // 1. Record newly closed candle
  STATE.liveCandles.push(candle);

  // 2. Automatic Verification after 10 minutes (Horizon = 2 candles)
  const currentClose = candle.close;
  const candleTimeSec = candle.time;

  STATE.livePredictions.forEach((pred) => {
    if (pred.status === "PENDING") {
      const predTimeSec = Math.floor(new Date(pred.open_time).getTime() / 1000);
      // Prediction made 2 candles (600s = 10 mins) ago has resolved now!
      if (candleTimeSec - predTimeSec >= 600) {
        pred.status = "RESOLVED";
        pred.outcome_close = currentClose;
        const ret = (currentClose / pred.close) - 1.0;
        pred.return_bps_label = Math.round(ret * 10000.0);
        const rawSign = ret > 0 ? 1 : (ret < 0 ? -1 : 0);

        if (pred.label === "UP") {
          pred.correct = rawSign === 1 ? 1 : 0;
        } else if (pred.label === "DOWN") {
          pred.correct = rawSign === -1 ? 1 : 0;
        } else {
          pred.status = "NO_CALL";
          pred.correct = null;
        }
        console.log(`[10-Min Verification] Pred at ${pred.open_time} resolved: ${pred.correct === 1 ? 'CORRECT' : 'WRONG'} (Ret: ${pred.return_bps_label} bps)`);
      }
    }
  });

  // 3. Generate new live prediction for the newly closed candle (predicting 2 candles forward = 10 mins)
  executeInstantLivePrediction();
  renderPredictionLog();
  updateBandContainmentStat();
}

async function executeInstantLivePrediction() {
  const candles = STATE.liveCandles;
  if (!candles || candles.length === 0) return;

  // Try calling backend for full ML model prediction
  try {
    const resp = await fetch(`${CONFIG.dashApiBase}/dash/live/predict_latest`);
    if (resp.ok) {
      const pred = await resp.json();
      if (pred && (!STATE.livePredictions.length || STATE.livePredictions[0].open_time !== pred.open_time)) {
        STATE.livePredictions.unshift(pred);
        renderPredictionPanel(pred);
        attachPredictionToChart("live", pred);
        renderPredictionLog();
        return;
      }
    }
  } catch (e) {
    console.warn("predict_latest fallback to client evaluation:", e);
  }

  // Client indicator engine fallback
  generateAutonomousLivePrediction();
}

function generateAutonomousLivePrediction() {
  const candles = STATE.liveCandles;
  if (!candles || candles.length < 5) return;

  const idx = candles.length - 1;
  const c = candles[idx];
  const close = c.close;

  // Wilder ATR 14
  let atr = 120.0;
  if (candles.length >= 15) {
    let trSum = 0;
    for (let i = idx - 13; i <= idx; i++) {
      const prevC = candles[i - 1] ? candles[i - 1].close : candles[i].open;
      const tr = Math.max(
        candles[i].high - candles[i].low,
        Math.abs(candles[i].high - prevC),
        Math.abs(candles[i].low - prevC)
      );
      trSum += tr;
    }
    atr = trSum / 14;
  }

  // 20-period Moving Average
  const lookbackMA = Math.min(20, candles.length);
  let sumMA = 0;
  for (let i = idx - lookbackMA + 1; i <= idx; i++) {
    sumMA += candles[i].close;
  }
  const ma20 = sumMA / lookbackMA;

  const diffPct = (close - ma20) / ma20;
  let trend = "SIDEWAYS";
  let phase = "PULLBACK";
  let pUp = 0.32;
  let pDn = 0.32;
  let pNeu = 0.36;

  if (diffPct > 0.001) {
    trend = "UPTREND";
    pUp = 0.44;
    pNeu = 0.30;
    pDn = 0.26;
  } else if (diffPct < -0.001) {
    trend = "DOWNTREND";
    pUp = 0.25;
    pNeu = 0.31;
    pDn = 0.44;
  }

  if (Math.abs(close - ma20) > 0.7 * atr) {
    phase = "BREAKTHROUGH";
  }

  // Indicative bands
  const smartPrice = close + CONFIG.bandConstants.K_MOVE * (pUp - pDn) * atr;
  const upper = smartPrice + CONFIG.bandConstants.Z * atr;
  const lower = smartPrice - CONFIG.bandConstants.Z * atr;

  let label = "NO CLEAR SIGNAL";
  let modelResult = "NEUTRAL";
  if (pUp > pDn && pUp > pNeu) modelResult = "UP";
  else if (pDn > pUp && pDn > pNeu) modelResult = "DOWN";

  if (pUp >= 0.32 && pUp - pDn >= 0.04) {
    label = "UP";
  } else if (pDn >= 0.32 && pDn - pUp >= 0.04) {
    label = "DOWN";
  }

  const pred = {
    open_time: c.open_time || new Date(c.time * 1000).toISOString(),
    close: close,
    trend: trend,
    trend_strength: 0.74,
    phase: phase,
    phase_strength: 0.68,
    context_text: `${trend} | ${phase}`,
    model_result: modelResult,
    p_final: { down: pDn, neutral: pNeu, up: pUp },
    label: label,
    confidence: Math.max(pUp, pDn),
    smart_price: smartPrice,
    upper: upper,
    lower: lower,
    atr14: atr,
    status: "PENDING",
    correct: null,
    outcome_close: null,
  };

  if (!STATE.livePredictions.some((p) => p.open_time === pred.open_time)) {
    STATE.livePredictions.unshift(pred);
  }
  renderPredictionPanel(pred);
  attachPredictionToChart("live", pred);
  renderPredictionLog();
}

// =====================================================================
// REPLAY ENGINE (Held-Out Test Split)
// =====================================================================
async function initReplayMode() {
  if (CHARTS.replay) {
    CHARTS.replay.resize();
  }

  try {
    const rangeResp = await fetch(`${CONFIG.dashApiBase}/dash/replay/range`);
    if (rangeResp.ok) {
      STATE.replayTestRange = await rangeResp.json();
      setupReplayControls();
    }
  } catch (err) {
    console.warn("[Replay] Failed to load range, using fallback:", err);
    STATE.replayTestRange = {
      min_open_time: "2026-05-20T14:20:00.000Z",
      max_open_time: "2026-11-09T23:55:00.000Z",
    };
    setupReplayControls();
  }
}

function setupReplayControls() {
  if (!STATE.replayTestRange) return;
  const minIso = STATE.replayTestRange.min_open_time;
  const maxIso = STATE.replayTestRange.max_open_time;

  DOM.replayDatePicker.min = minIso.substring(0, 16);
  DOM.replayDatePicker.max = maxIso.substring(0, 16);
  DOM.replayDatePicker.value = minIso.substring(0, 16);
  DOM.replayDateBadge.textContent = "TEST SPLIT";

  loadReplaySlice(minIso);
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
    executeReplayStep();
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
  let price = 72000.0;
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

  // Generate mock predictions
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
    DOM.logTabLive.click();
    if (CHARTS.live) CHARTS.live.resize();
  });

  DOM.btnReplay.addEventListener("click", () => {
    STATE.activeTab = "replay";
    DOM.btnReplay.classList.add("active");
    DOM.btnLive.classList.remove("active");
    DOM.replayView.style.display = "flex";
    DOM.liveView.style.display = "none";
    DOM.logTabReplay.click();
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
  DOM.btnStepPrev.addEventListener("click", () => {
    if (STATE.replayCurrentIndex > 0) {
      STATE.replayCurrentIndex--;
      if (STATE.replayPredictions.length > 0) {
        STATE.replayPredictions.shift();
        if (STATE.replayPredictions.length > 0) {
          renderPredictionPanel(STATE.replayPredictions[0]);
        }
        updateAllMarkers("replay");
        renderPredictionLog();
      }
    }
  });

  DOM.btnResetReplay.addEventListener("click", () => {
    STATE.replayCurrentIndex = 0;
    STATE.replayPredictions = [];
    if (CHARTS.replay) CHARTS.replay.setMarkers([]);
    renderPredictionLog();
    if (STATE.replayCandles.length > 0) executeReplayStep();
  });

  DOM.btnClearReplay.addEventListener("click", () => {
    STATE.replayPredictions = [];
    if (CHARTS.replay) CHARTS.replay.setMarkers([]);
    renderPredictionLog();
  });

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
      if (CHARTS.live) CHARTS.live.setLayerVisible(layer, e.target.checked);
      if (CHARTS.replay) CHARTS.replay.setLayerVisible(layer, e.target.checked);
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
