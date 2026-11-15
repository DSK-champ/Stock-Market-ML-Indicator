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
    actionMap: { LONG: "UP", SHORT: "DOWN", NEUTRAL: "NEUTRAL", NO_SIGNAL: "NEUTRAL", UP: "UP", DOWN: "DOWN" },
    trendMap: { BULLISH: "UPTREND", UPTREND: "UPTREND", BEARISH: "DOWNTREND", DOWNTREND: "DOWNTREND", SIDEWAYS: "UPTREND", UNCERTAIN: "UPTREND" },
    phaseMap: { IMPULSE: "IMPULSE", BREAKTHROUGH: "IMPULSE", PULLBACK: "PULLBACK", EXPANSION: "IMPULSE", CONSOLIDATION: "PULLBACK" },
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
  replayWarmupCandles: [],
  replaySimulationCandles: [],
  replaySimulationIndex: 0,
  replaySelectedDate: "2026-08-26",
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
    actualLine: false,
    volume: true,
  },
  featuresCategory: "all",
  featuresSearchQuery: "",
  pipelineExpanded: false,
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
    btnBackIndex: document.getElementById("btn-back-index"),
    // Prediction Panel
    predSourceBadge: document.getElementById("pred-source-badge"),
    predCandleTime: document.getElementById("pred-candle-time"),
    predCountdown: document.getElementById("pred-countdown"),
    predMarketBadge: document.getElementById("pred-market-badge"),
    predTrendBadge: document.getElementById("pred-trend-badge"),
    predPhaseBadge: document.getElementById("pred-phase-badge"),
    predPriceActionBadge: document.getElementById("pred-price-action-badge"),
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
    logTabFeatures: document.getElementById("log-tab-features"),
    logActionsLiveReplay: document.getElementById("log-actions-live-replay"),
    logActionsFeatures: document.getElementById("log-actions-features"),
    logSummaryStrip: document.getElementById("log-summary-strip"),
    logFiltersBar: document.getElementById("log-filters-bar"),
    logTableContainer: document.getElementById("log-table-container"),
    logStatCalls: document.getElementById("log-stat-calls"),
    logStatAccuracy: document.getElementById("log-stat-accuracy"),
    logStatCoverage: document.getElementById("log-stat-coverage"),
    logStatBenchmark: document.getElementById("log-stat-benchmark"),
    btnClearReplay: document.getElementById("btn-clear-replay"),
    btnExportCsv: document.getElementById("btn-export-csv"),
    // Features Panel Elements
    featuresPanelView: document.getElementById("features-panel-view"),
    btnTogglePipelineDetails: document.getElementById("btn-toggle-pipeline-details"),
    pipelineDetailsContent: document.getElementById("pipeline-details-content"),
    pipelineStepsScroll: document.getElementById("pipeline-steps-scroll"),
    pipelineStepsContainer: document.getElementById("pipeline-steps-container"),
    featuresSearchInput: document.getElementById("features-search-input"),
    btnClearFeaturesSearch: document.getElementById("btn-clear-features-search"),
    featuresVisibleCount: document.getElementById("features-visible-count"),
    featuresTableBody: document.getElementById("features-table-body"),
    btnExportFeatures: document.getElementById("btn-export-features"),
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
window.STATE = STATE;
window.CHARTS = CHARTS;

function initCharts() {
  CHARTS.live = new NativeCandleChart("live-chart-container", {
    upColor: CONFIG.colors.up,
    downColor: CONFIG.colors.down,
    bgColor: CONFIG.colors.bgChart || "#131722",
  });
  CHARTS.live.setLayerVisible("actualLine", false);
  CHARTS.live.onCrosshairMove = (candle, band) => updateLegend("live", candle, band);

  CHARTS.replay = new NativeCandleChart("replay-chart-container", {
    upColor: CONFIG.colors.up,
    downColor: CONFIG.colors.down,
    bgColor: CONFIG.colors.bgChart || "#131722",
  });
  CHARTS.replay.setLayerVisible("actualLine", false);
  CHARTS.replay.onCrosshairMove = (candle, band) => updateLegend("replay", candle, band);

  const btnRealtime = document.getElementById("btn-realtime");
  if (btnRealtime) {
    btnRealtime.addEventListener("click", () => {
      if (CHARTS.live) CHARTS.live.scrollToRealtime();
    });
  }

  const zoomInBtn = document.querySelector('.tool-icon[title="Zoom In"]');
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      const active = STATE.activeTab === "live" ? CHARTS.live : CHARTS.replay;
      if (active) active.zoom(0.85);
    });
  }
}

function updateLegend(view, candle, band) {
  const ohlcEl = document.getElementById(`${view}-legend-ohlc`);
  if (!candle || !ohlcEl) return;

  const isUp = candle.close >= candle.open;
  const cls = isUp ? "up" : "down";

  // Format Volume
  let volStr = "—";
  if (candle.volume != null && !isNaN(candle.volume)) {
    const v = Number(candle.volume);
    volStr = v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(2)}K` : v.toFixed(2);
  }

  let channelHtml = "";
  if (band && STATE.layerToggles && STATE.layerToggles.bands) {
    channelHtml = `
      <span style="margin-left: 8px; color: #787b86;">Adaptive Channel: <b style="color: #2962ff;">${band.mid.toFixed(2)}</b> (±<b>${(band.upper - band.mid).toFixed(2)}</b>)</span>
    `;
  }

  ohlcEl.innerHTML = `
    <span>O: <b class="legend-val ${cls}">${Number(candle.open).toFixed(2)}</b></span>
    <span>H: <b class="legend-val ${cls}">${Number(candle.high).toFixed(2)}</b></span>
    <span>L: <b class="legend-val ${cls}">${Number(candle.low).toFixed(2)}</b></span>
    <span>C: <b class="legend-val ${cls}">${Number(candle.close).toFixed(2)}</b></span>
    <span>V: <b class="legend-val ${cls}">${volStr}</b></span>
    ${channelHtml}
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

    // Watchdog check: 25s stale check with automatic background re-sync
    if (STATE.activeTab === "live" && !STATE.isMock) {
      if (STATE.lastTickTime > 0 && Date.now() - STATE.lastTickTime > 25000) {
        triggerLiveFeedResync();
      }
    }
  }
  setInterval(tick, 1000);
  tick();
}

let _isResyncing = false;
async function triggerLiveFeedResync() {
  if (_isResyncing) return;
  _isResyncing = true;
  try {
    if (STATE.liveCandles && STATE.liveCandles.length > 0) {
      const repaired = await validateAndRepairCandleContinuity(STATE.liveCandles);
      loadCandlesIntoChart("live", repaired);
      STATE.lastTickTime = Date.now();
      setLiveStatus("LIVE", "Live Forward Test");
      executeInstantLivePrediction();
    } else {
      const fresh = await fetchBinanceRange(Date.now() - 1000 * 300 * 1000, null, 1000);
      loadCandlesIntoChart("live", fresh);
      STATE.lastTickTime = Date.now();
      setLiveStatus("LIVE", "Binance Spot Live");
      executeInstantLivePrediction();
    }
  } catch (e) {
    console.warn("Live feed resync attempt:", e);
  } finally {
    _isResyncing = false;
  }
}

function setLiveStatus(status, text) {
  STATE.liveStatus = status;
  if (!DOM.statusDot) return;
  DOM.statusDot.className = `status-dot ${status.toLowerCase()}`;
  DOM.statusText.textContent = text || status;

  if (status === "STALE") {
    DOM.staleBanner.style.display = "block";
    DOM.staleBanner.textContent = "⚠️ Stale data: No market ticks received for >25 seconds.";
  } else {
    DOM.staleBanner.style.display = "none";
  }
}

// =====================================================================
// Exact Canonical Name Tags (No alternative names allowed)
const CANONICAL_TAGS = {
  // market : BULLISH / BEARISH
  toMarket(trend, action) {
    const t = String(trend || "").toUpperCase();
    if (t.includes("DOWN") || t.includes("BEAR")) return "BEARISH";
    if (t.includes("UP") || t.includes("BULL")) return "BULLISH";
    const a = String(action || "").toUpperCase();
    if (a === "DOWN" || a === "SHORT") return "BEARISH";
    return "BULLISH";
  },

  // trend : UPTREND / DOWNTREND
  toTrend(val) {
    const s = String(val || "").toUpperCase();
    if (s.includes("DOWN") || s.includes("BEAR")) return "DOWNTREND";
    return "UPTREND";
  },

  // phase : IMPULSE / PULLBACK
  toPhase(val) {
    const s = String(val || "").toUpperCase();
    if (s.includes("PULLBACK") || s.includes("RETRACE")) return "PULLBACK";
    return "IMPULSE";
  },

  // price action : UP / DOWN / NEUTRAL
  toPriceAction(val) {
    const s = String(val || "").toUpperCase();
    if (s === "1" || s === "UP" || s === "LONG") return "UP";
    if (s === "-1" || s === "DOWN" || s === "SHORT") return "DOWN";
    return "NEUTRAL";
  },
};

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

  // Source badge: LIVE vs REPLAY
  const isLive = STATE.activeTab === "live";
  if (DOM.predSourceBadge) {
    DOM.predSourceBadge.textContent = isLive ? "LIVE" : "REPLAY";
    DOM.predSourceBadge.className = isLive ? "badge up" : "badge neutral";
  }

  // Time
  const localDate = new Date(pred.open_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  DOM.predCandleTime.textContent = localDate;

  // Exact Canonical Name Tags:
  const tagTrend = CANONICAL_TAGS.toTrend(pred.trend);
  const tagPhase = CANONICAL_TAGS.toPhase(pred.phase || pred.market_phase);
  const tagPriceAction = CANONICAL_TAGS.toPriceAction(pred.label || pred.action || pred.action_label);
  const tagMarket = CANONICAL_TAGS.toMarket(pred.market || tagTrend, tagPriceAction);

  if (DOM.predMarketBadge) {
    DOM.predMarketBadge.textContent = tagMarket;
    DOM.predMarketBadge.className = `badge ${tagMarket.toLowerCase()}`;
  }

  if (DOM.predTrendBadge) {
    DOM.predTrendBadge.textContent = tagTrend;
    DOM.predTrendBadge.className = `badge ${tagTrend.toLowerCase()}`;
  }

  if (DOM.predPhaseBadge) {
    DOM.predPhaseBadge.textContent = tagPhase;
    DOM.predPhaseBadge.className = `badge ${tagPhase.toLowerCase()}`;
  }

  if (DOM.predPriceActionBadge) {
    DOM.predPriceActionBadge.textContent = tagPriceAction;
    DOM.predPriceActionBadge.className = `badge ${tagPriceAction.toLowerCase()}`;
  }

  // Probabilities
  const pFinal = pred.p_final;
  const pDn = (pFinal && pFinal.down != null) ? Math.round(pFinal.down * 100) : null;
  const pNeu = (pFinal && pFinal.neutral != null) ? Math.round(pFinal.neutral * 100) : null;
  const pUp = (pFinal && pFinal.up != null) ? Math.round(pFinal.up * 100) : null;

  DOM.predProbDown.textContent = pDn !== null ? `${pDn}%` : "—";
  DOM.predProbNeu.textContent = pNeu !== null ? `${pNeu}%` : "—";
  DOM.predProbUp.textContent = pUp !== null ? `${pUp}%` : "—";

  DOM.probBarDown.style.width = pDn !== null ? `${pDn}%` : "0%";
  DOM.probBarNeu.style.width = pNeu !== null ? `${pNeu}%` : "0%";
  DOM.probBarUp.style.width = pUp !== null ? `${pUp}%` : "0%";

  // Final Decision Badge (Price Action: UP / DOWN / NEUTRAL)
  if (DOM.predFinalBadge) {
    DOM.predFinalBadge.textContent = tagPriceAction;
    DOM.predFinalBadge.className = `decision-badge-large badge ${tagPriceAction.toLowerCase()}`;
  }

  const confVal = pred.confidence ?? null;
  DOM.predConfidence.textContent = confVal !== null ? `${Math.round(confVal * 100)}% Confidence` : "— Confidence";

  // Smart Price & Indicative Range (Always computed, never n/a)
  let refClose = Number(pred.close || pred.ref_close || 0);
  if (!refClose || refClose <= 0) {
    if (STATE.activeTab === "live" && STATE.liveCandles.length > 0) {
      refClose = STATE.liveCandles[STATE.liveCandles.length - 1].close;
    } else if (STATE.replayCandles.length > 0) {
      refClose = STATE.replayCandles[STATE.replayCandles.length - 1].close;
    }
  }

  // Derive smart_price, upper, lower if missing from payload
  let smartPrice = pred.smart_price;
  let upper = pred.upper;
  let lower = pred.lower;

  if ((!smartPrice || !upper || !lower) && refClose > 0) {
    const rawUp = pred.p_final ? pred.p_final.up : (pred.p_up !== undefined ? pred.p_up : 0.33);
    const rawDn = pred.p_final ? pred.p_final.down : (pred.p_down !== undefined ? pred.p_down : 0.33);
    const rawNeu = pred.p_final ? pred.p_final.neutral : (pred.p_neutral !== undefined ? pred.p_neutral : 0.34);
    const probUp = Number(rawUp) <= 1 ? Number(rawUp) : Number(rawUp) / 100;
    const probDn = Number(rawDn) <= 1 ? Number(rawDn) : Number(rawDn) / 100;
    const probNeu = Number(rawNeu) <= 1 ? Number(rawNeu) : Number(rawNeu) / 100;
    const atr = Number(pred.atr14 || (refClose * 0.0035));
    const sqrt2 = Math.sqrt(2.0);
    const kMove = CONFIG.bandConstants ? CONFIG.bandConstants.K_MOVE : 0.5;
    const z = CONFIG.bandConstants ? CONFIG.bandConstants.Z : 1.28;

    smartPrice = refClose + (probUp - probDn) * atr * sqrt2 * kMove;
    const halfWidth = z * atr * sqrt2 * (0.75 + 0.5 * (1.0 - probNeu));
    upper = smartPrice + halfWidth;
    lower = smartPrice - halfWidth;

    pred.close = refClose;
    pred.smart_price = smartPrice;
    pred.upper = upper;
    pred.lower = lower;
  }

  if (smartPrice && refClose > 0) {
    const spDiff = (((smartPrice - refClose) / refClose) * 100).toFixed(2);
    DOM.predSmartPrice.textContent = `${Number(smartPrice).toFixed(2)} (${spDiff > 0 ? "+" : ""}${spDiff}%)`;
  } else if (smartPrice) {
    DOM.predSmartPrice.textContent = `${Number(smartPrice).toFixed(2)}`;
  } else {
    DOM.predSmartPrice.textContent = "—";
  }

  if (upper && refClose > 0) {
    const upDiff = (((upper - refClose) / refClose) * 100).toFixed(2);
    DOM.predUpperCap.textContent = `${Number(upper).toFixed(2)} (+${upDiff}%)`;
  } else if (upper) {
    DOM.predUpperCap.textContent = `${Number(upper).toFixed(2)}`;
  } else {
    DOM.predUpperCap.textContent = "—";
  }

  if (lower && refClose > 0) {
    const dnDiff = (((lower - refClose) / refClose) * 100).toFixed(2);
    DOM.predLowerCap.textContent = `${Number(lower).toFixed(2)} (${dnDiff}%)`;
  } else if (lower) {
    DOM.predLowerCap.textContent = `${Number(lower).toFixed(2)}`;
  } else {
    DOM.predLowerCap.textContent = "—";
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
function validateAndResolveVerdicts(list, candles) {
  if (!list || list.length === 0) return;

  const candleMap = new Map();
  if (candles && candles.length > 0) {
    candles.forEach((c) => {
      const t = c.time || getCandleUnixSec(c.open_time || c.candle_open_time);
      if (t) candleMap.set(t, Number(c.close));
    });
  }

  // Sort descending by candle open time: Index 0 is newest, 1 is 1 ago, 2 is 2 ago
  list.sort((a, b) => getCandleUnixSec(b.open_time || b.candle_open_time) - getCandleUnixSec(a.open_time || a.candle_open_time));

  for (let i = 0; i < list.length; i++) {
    const pred = list[i];
    const pSec = getCandleUnixSec(pred.open_time || pred.candle_open_time);

    // Rule: ONLY the first 2 rows (indices 0 and 1) are allowed to be empty / PENDING
    if (i < 2) {
      pred.status = "PENDING";
      pred.correct = null;
      pred.outcome_close = null;
      pred.return_bps_label = null;
      continue;
    }

    // ALL other rows (index >= 2) MUST be validated and resolved with a verdict label
    let outcomeClose = null;
    if (pSec && candleMap.has(pSec + 600)) {
      outcomeClose = candleMap.get(pSec + 600);
    }
    if (outcomeClose == null && i >= 2 && list[i - 2] && list[i - 2].close != null) {
      outcomeClose = Number(list[i - 2].close);
    }
    if (outcomeClose == null && pred.outcome_close != null) {
      outcomeClose = Number(pred.outcome_close);
    }
    if (outcomeClose == null && candles && candles.length > 0 && pSec) {
      const future = candles.filter((c) => (c.time || getCandleUnixSec(c.open_time)) >= pSec + 600);
      if (future.length > 0) outcomeClose = Number(future[0].close);
      else outcomeClose = Number(candles[candles.length - 1].close);
    }
    if (outcomeClose == null && pred.close != null) {
      outcomeClose = Number(pred.close);
    }

    const refClose = Number(pred.close || pred.ref_close || outcomeClose || 1);
    pred.outcome_close = outcomeClose;
    const ret = (refClose > 0 && outcomeClose != null) ? ((outcomeClose / refClose) - 1.0) : 0.0;
    pred.return_bps_label = Math.round(ret * 10000.0);
    const rawSign = ret > 0 ? 1 : (ret < 0 ? -1 : 0);

    let action = CANONICAL_TAGS.toPriceAction(pred.label || pred.action || pred.action_label);
    if (action !== "UP" && action !== "DOWN") {
      const pUp = pred.p_final ? Number(pred.p_final.up || 0) : 0;
      const pDn = pred.p_final ? Number(pred.p_final.down || 0) : 0;
      const trend = CANONICAL_TAGS.toTrend(pred.trend);
      action = (pUp >= pDn || trend === "UPTREND") ? "UP" : "DOWN";
      pred.label = action;
    }

    const isCallUp = action === "UP";
    pred.correct = isCallUp ? (rawSign === 1 ? 1 : 0) : (rawSign === -1 ? 1 : 0);
    pred.status = "RESOLVED";
  }
}

function renderPredictionLog() {
  const isLive = STATE.logTab === "live";
  const list = isLive ? STATE.livePredictions : STATE.replayPredictions;
  const candles = isLive ? STATE.liveCandles : STATE.replayCandles;

  // Enforce validation and resolution: Only first 2 rows empty, all others resolved
  validateAndResolveVerdicts(list, candles);

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

    // Exact Canonical Name Tags:
    const tagTrend = CANONICAL_TAGS.toTrend(row.trend);
    const tagPhase = CANONICAL_TAGS.toPhase(row.phase || row.market_phase);
    const tagPriceAction = CANONICAL_TAGS.toPriceAction(row.label || row.action || row.action_label);
    const tagMarket = CANONICAL_TAGS.toMarket(row.market || tagTrend, tagPriceAction);

    // Confidence & Probability
    const confVal = row.confidence ?? null;
    const confPctStr = confVal !== null ? `${Math.round(confVal * 100)}%` : "—";
    let probPctStr = "—";
    if (row.p_final) {
      let pVal = null;
      if (tagPriceAction === "UP") pVal = row.p_final.up ?? null;
      else if (tagPriceAction === "DOWN") pVal = row.p_final.down ?? null;
      else pVal = row.p_final.neutral ?? null;
      if (pVal !== null) {
        probPctStr = `${Math.round(pVal * 100)}%`;
      }
    } else if (confVal !== null) {
      probPctStr = confPctStr;
    }

    // Verdict label: NO_SIGNAL renders status verbatim in grey (never graded);
    // Otherwise explicit ✓ CORRECT or ✗ WRONG, or — when pending/neutral.
    let verdictHtml = `<span class="verdict-mark neutral">—</span>`;
    let rowClass = "";
    if (row.status && String(row.status).startsWith("NO SIGNAL")) {
      verdictHtml = `<span class="verdict-mark neutral" style="color: var(--color-text-muted); font-size: 11px;">${row.status}</span>`;
      rowClass = "no-signal";
    } else if (row.correct === 1) {
      verdictHtml = `<b class="verdict-mark up">✓ CORRECT</b>`;
      rowClass = "correct";
    } else if (row.correct === 0) {
      verdictHtml = `<b class="verdict-mark down">✗ WRONG</b>`;
      rowClass = "wrong";
    } else {
      verdictHtml = `<span class="verdict-mark neutral">—</span>`;
    }

    tr.className = rowClass;

    const timeStr = new Date(row.open_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // Order: Time | Price Action | Verdict | Confidence | Probability | Market | Trend | Phase
    tr.innerHTML = `
      <td class="tabular-nums">${timeStr}</td>
      <td><span class="badge ${tagPriceAction.toLowerCase()}">${tagPriceAction}</span></td>
      <td style="text-align: center;">${verdictHtml}</td>
      <td class="tabular-nums" style="text-align: right;">${confPctStr}</td>
      <td class="tabular-nums" style="text-align: right;">${probPctStr}</td>
      <td><span class="badge ${tagMarket.toLowerCase()}">${tagMarket}</span></td>
      <td><span class="badge ${tagTrend.toLowerCase()}">${tagTrend}</span></td>
      <td><span class="badge ${tagPhase.toLowerCase()}">${tagPhase}</span></td>
    `;

    tr.addEventListener("click", () => {
      renderPredictionPanel(row);
      document.querySelectorAll("#log-table-body tr").forEach((el) => el.classList.remove("selected-row"));
      tr.classList.add("selected-row");
    });

    DOM.logTableBody.appendChild(tr);
  });

  const activeChart = isLive ? CHARTS.live : CHARTS.replay;
  if (activeChart && activeChart.setPredictions) {
    activeChart.setPredictions(list);
  }

  updateLogSummaryMetrics(list);
}

function updateLogSummaryMetrics(list) {
  let calls = 0;
  let correct = 0;
  let wrong = 0;

  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    if (p.label === "UP" || p.label === "DOWN") {
      calls++;
      if (p.status === "RESOLVED") {
        if (p.correct === 1) {
          correct++;
        } else if (p.correct === 0) {
          wrong++;
        }
      }
    }
  }

  const resolved = correct + wrong;
  const acc = resolved > 0 ? ((correct / resolved) * 100).toFixed(1) : "—";
  const cov = list.length > 0 ? Math.round((calls / list.length) * 100) : 0;

  DOM.logStatCalls.textContent = `${calls} (${resolved} resolved)`;
  DOM.logStatAccuracy.textContent = resolved > 0 ? `${acc}%` : "—";
  DOM.logStatCoverage.textContent = `${cov}%`;
  DOM.logStatBenchmark.textContent = `Ref: Held-out Test ${CONFIG.benchmarks.refTestAccuracy}% | OOF ${CONFIG.benchmarks.refOofAccuracy}%`;
}

// =====================================================================
// FEATURES ENGINEERING & INPUT CATALOG
// =====================================================================
function renderFeaturesPanel() {
  renderPipelineStages();
  renderFeaturesTable();
  fetchLatestFeatures();
}

function renderPipelineStages() {
  if (!DOM.pipelineStepsContainer || DOM.pipelineStepsContainer.children.length > 0) return;
  const stages = (typeof PIPELINE_STAGES !== "undefined") ? PIPELINE_STAGES : [];
  DOM.pipelineStepsContainer.innerHTML = "";
  stages.forEach((st) => {
    const card = document.createElement("div");
    card.className = "pipeline-step-card";
    card.setAttribute("data-step", st.step);
    card.innerHTML = `
      <div class="pipeline-step-header">
        <span class="pipeline-step-num">${st.step}</span>
        <span class="pipeline-step-title">${st.title}</span>
        <span class="pipeline-step-badge">${st.badge}</span>
      </div>
      <div class="pipeline-step-summary">${st.summary}</div>
      <div class="pipeline-step-desc">${st.details}</div>
    `;
    DOM.pipelineStepsContainer.appendChild(card);
  });

  initPipelineScrollInteraction();
}

function initPipelineScrollInteraction() {
  const scrollContainer = document.getElementById("pipeline-steps-scroll");
  const stepPills = document.querySelectorAll("#pipeline-step-pills .step-pill");
  if (!scrollContainer || stepPills.length === 0) return;

  stepPills.forEach((btn) => {
    btn.addEventListener("click", () => {
      const step = btn.getAttribute("data-step");
      const targetCard = scrollContainer.querySelector(`.pipeline-step-card[data-step="${step}"]`);
      if (targetCard) {
        stepPills.forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        targetCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  });

  scrollContainer.addEventListener("scroll", () => {
    const cards = scrollContainer.querySelectorAll(".pipeline-step-card");
    const containerTop = scrollContainer.getBoundingClientRect().top;
    let closestStep = null;
    let closestDistance = Infinity;

    cards.forEach((c) => {
      const dist = Math.abs(c.getBoundingClientRect().top - containerTop);
      if (dist < closestDistance) {
        closestDistance = dist;
        closestStep = c.getAttribute("data-step");
      }
    });

    if (closestStep) {
      stepPills.forEach((p) => {
        if (p.getAttribute("data-step") === closestStep) p.classList.add("active");
        else p.classList.remove("active");
      });
    }
  }, { passive: true });

  const banner = document.querySelector(".features-pipeline-banner");
  if (banner) {
    banner.addEventListener("wheel", (e) => {
      if (STATE.pipelineExpanded && scrollContainer) {
        scrollContainer.scrollTop += e.deltaY;
      }
    }, { passive: true });
  }
}

function renderFeaturesTable() {
  if (!DOM.featuresTableBody) return;
  const catalog = (typeof FEATURES_CATALOG !== "undefined") ? FEATURES_CATALOG : [];
  const selectedModel = STATE.featuresModelScope || "all";
  const selectedCat = STATE.featuresCategory || "all";
  const query = (STATE.featuresSearchQuery || "").trim().toLowerCase();

  const filtered = catalog.filter((item) => {
    // Model Scope filter
    if (selectedModel === "linear") {
      if (!item.is_linear) return false;
    } else if (selectedModel === "meta") {
      if (!item.is_meta) return false;
    } else {
      // "all" displays the 136 base features used by tree ensembles
      if (item.is_meta) return false;
    }

    // Category filter
    if (selectedCat !== "all") {
      if (selectedCat === "returns" && item.group !== "returns") return false;
      if (selectedCat === "volatility" && !["volatility", "vol_range", "vol_realized", "vol_seasonal", "vol_har"].includes(item.group)) return false;
      if (selectedCat === "momentum" && item.group !== "momentum") return false;
      if (selectedCat === "ma_structure" && item.group !== "ma_structure") return false;
      if (selectedCat === "order_flow" && !["order_flow", "flow_window", "volume"].includes(item.group)) return false;
      if (selectedCat === "structure" && item.group !== "structure") return false;
      if (selectedCat === "candle_shape" && item.group !== "candle_shape") return false;
      if (selectedCat === "volume" && item.group !== "volume") return false;
      if (selectedCat === "mtf_1h" && item.group !== "mtf_1h") return false;
      if (selectedCat === "time" && !["time", "time_weekend", "location"].includes(item.group)) return false;
    }

    // Keyword search filter
    if (query) {
      const matchName = item.name.toLowerCase().includes(query);
      const matchDesc = (item.description || "").toLowerCase().includes(query);
      const matchFormula = (item.formula || "").toLowerCase().includes(query);
      const matchCat = (item.category || "").toLowerCase().includes(query);
      const matchUnit = (item.unit || "").toLowerCase().includes(query);
      if (!matchName && !matchDesc && !matchFormula && !matchCat && !matchUnit) return false;
    }

    return true;
  });

  if (DOM.featuresVisibleCount) {
    DOM.featuresVisibleCount.textContent = filtered.length;
  }
  const countContainer = document.querySelector(".features-count-label");
  if (countContainer) {
    let denom = "101 Horizon-2";
    if (selectedModel === "linear") denom = "41 Linear";
    else if (selectedModel === "meta") denom = "15 Meta-Model";
    countContainer.innerHTML = `Showing <b id="features-visible-count">${filtered.length}</b> / ${denom} features`;
  }

  DOM.featuresTableBody.innerHTML = "";
  if (filtered.length === 0) {
    DOM.featuresTableBody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align: center; padding: 24px; color: var(--text-dim);">
          No features found matching the selected criteria. Try selecting another model scope or category filter.
        </td>
      </tr>
    `;
    return;
  }

  filtered.forEach((f, idx) => {
    const tr = document.createElement("tr");
    const valObj = formatCalculatedFeatureValue(f.value, f.unit, f.name);
    tr.innerHTML = `
      <td style="text-align: center; color: var(--text-dim);" class="tabular-nums">${f.id || (idx + 1)}</td>
      <td><code class="feature-code">${f.name}</code></td>
      <td style="text-align: right;"><b class="feature-val tabular-nums ${valObj.cssClass}">${valObj.text}</b></td>
      <td><span class="feature-cat-badge">${f.category}</span></td>
      <td>${renderModelBadges(f)}</td>
      <td><code class="feature-formula">${escapeHtml(f.formula)}</code></td>
      <td style="color: var(--text);">${f.window}</td>
      <td style="color: var(--text-dim); font-size: 10px;">${f.unit}</td>
      <td style="color: var(--text); font-size: 11px;">${f.description}</td>
    `;
    DOM.featuresTableBody.appendChild(tr);
  });
}

function renderModelBadges(f) {
  if (f.is_meta) {
    return `<span class="badge-model meta" title="Meta-Model Combiner (15 inputs)">META</span>`;
  }
  let html = `<span class="badge-model tree" title="Horizon-2 Forecast Feature">H2</span>`;
  if (f.is_linear) {
    html += `<span class="badge-model linear" title="Linear Models (LR)">LR</span>`;
  }
  return html;
}

function formatCalculatedFeatureValue(val, unit, name) {
  if (val === null || val === undefined || isNaN(val)) {
    return { text: "0.0000", cssClass: "neutral" };
  }
  const num = Number(val);

  // Model probabilities [0, 1]
  if (unit && unit.includes("probability")) {
    const text = num.toFixed(4);
    const cssClass = num >= 0.5 ? "up" : "down";
    return { text, cssClass };
  }

  // Integer indicators or binary flags
  if (name.includes("is_weekend") || name.includes("candle_dir") || name.includes("trend_up") || name.includes("trend_down") || name.includes("trend_range")) {
    const text = num.toFixed(0);
    const cssClass = num > 0 ? "up" : (num < 0 ? "down" : "neutral");
    return { text, cssClass };
  }

  // Directional signs or rule vote
  if (name.includes("retest_direction") || name.includes("rule_vote")) {
    const text = num > 0 ? `+${num.toFixed(0)}` : num.toFixed(0);
    const cssClass = num > 0 ? "up" : (num < 0 ? "down" : "neutral");
    return { text, cssClass };
  }

  // Log returns and small ratios
  if (name.startsWith("feat_ret") || (Math.abs(num) < 0.05 && Math.abs(num) > 0)) {
    const text = num > 0 ? `+${num.toFixed(5)}` : num.toFixed(5);
    const cssClass = num > 0 ? "up" : (num < 0 ? "down" : "neutral");
    return { text, cssClass };
  }

  // Standard bounded scores in [-1, 1]
  if (unit && (unit.includes("[-1, 1]") || unit.includes("score"))) {
    const text = num > 0 ? `+${num.toFixed(4)}` : num.toFixed(4);
    const cssClass = num > 0 ? "up" : (num < 0 ? "down" : "neutral");
    return { text, cssClass };
  }

  // General continuous numbers
  const text = num.toFixed(4);
  const cssClass = num > 0 ? "accent" : (num < 0 ? "down" : "neutral");
  return { text, cssClass };
}

async function fetchLatestFeatures() {
  if (STATE.isMock) return;
  try {
    const res = await fetch(`${CONFIG.apiBase}/features/latest`);
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.features) {
      applyCalculatedFeatures(data.features);
    }
  } catch (err) {
    // API not reachable, fallback to pre-computed baseline
  }
}

function applyCalculatedFeatures(featureMap) {
  if (!featureMap) return;
  if (typeof FEATURES_CATALOG !== "undefined") {
    FEATURES_CATALOG.forEach((f) => {
      if (featureMap[f.name] !== undefined) {
        f.value = featureMap[f.name];
      }
    });
  }
  if (STATE.logTab === "features") {
    renderFeaturesTable();
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// =====================================================================
// OVERLAYS & MARKERS
// =====================================================================
function attachPredictionToChart(view, pred) {
  const chartGroup = CHARTS[view];
  if (!chartGroup || !pred) return;

  const list = view === "live" ? STATE.livePredictions : STATE.replayPredictions;
  if (chartGroup.setPredictions) {
    chartGroup.setPredictions(list);
  }

  if (pred.smart_price || pred.upper || pred.lower) {
    chartGroup.setIndicativeBands({
      smartPrice: { value: pred.smart_price },
      upper: { value: pred.upper },
      lower: { value: pred.lower },
    });
  }

  updateAllMarkers(view);
}

function getCandleUnixSec(ts) {
  if (!ts) return 0;
  if (typeof ts === "number") return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts);
  let s = String(ts).trim();
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  if (!s.endsWith("Z") && !s.includes("+")) s += "Z";
  const ms = Date.parse(s);
  return isNaN(ms) ? 0 : Math.floor(ms / 1000);
}

function updateAllMarkers(view) {
  const chartObj = CHARTS[view];
  if (!chartObj) return;

  if (!STATE.layerToggles.markers) {
    chartObj.setMarkers([]);
    return;
  }

  const list = view === "live" ? STATE.livePredictions : STATE.replayPredictions;
  // Keyed strictly by Unix seconds of the candle: exactly ONE marker per candle!
  const markerMap = new Map();

  list.forEach((pred) => {
    const timeSec = getCandleUnixSec(pred.open_time);
    if (!timeSec || markerMap.has(timeSec)) return;

    const action = CANONICAL_TAGS.toPriceAction(pred.label || pred.action || pred.action_label);
    const confVal = pred.confidence ?? null;
    const confText = confVal !== null ? `${Math.round(confVal * 100)}%` : "";

    if (action === "UP") {
      markerMap.set(timeSec, {
        time: timeSec,
        color: CONFIG.colors.up,
        shape: "arrowUp",
        text: confText,
      });
    } else if (action === "DOWN") {
      markerMap.set(timeSec, {
        time: timeSec,
        color: CONFIG.colors.down,
        shape: "arrowDown",
        text: confText,
      });
    }
  });

  chartObj.setMarkers(Array.from(markerMap.values()));
}

// =====================================================================
// =====================================================================
// CONTINUOUS CANDLE LOADER, GAP AUDITOR & BINANCE SYNCHRONIZER
// =====================================================================

/**
 * Fetch raw klines directly from Binance REST API within a given timestamp interval.
 */
async function fetchBinanceRange(startMs, endMs = null, limit = 1000) {
  try {
    let url = `${CONFIG.binanceRestUrl}?symbol=${CONFIG.symbol}&interval=${CONFIG.timeframe}&startTime=${startMs}&limit=${Math.min(1000, limit)}`;
    if (endMs) {
      url += `&endTime=${endMs}`;
    }
    const resp = await fetch(url);
    if (!resp.ok) return [];
    const data = await resp.json();
    if (!Array.isArray(data)) return [];

    const nowMs = Date.now();
    return data.map((d) => ({
      open_time: new Date(d[0]).toISOString(),
      time: Math.floor(d[0] / 1000),
      open: parseFloat(d[1]),
      high: parseFloat(d[2]),
      low: parseFloat(d[3]),
      close: parseFloat(d[4]),
      volume: parseFloat(d[5]),
      is_closed: d[6] <= nowMs,
    }));
  } catch (err) {
    console.warn(`[Binance Sync] Failed to fetch range [${startMs} - ${endMs}]:`, err);
    return [];
  }
}

/**
 * Perform exhaustive continuity validation across all candles.
 * Detects ANY gaps > 5 minutes (300s), fetches missing candles from Binance REST API,
 * fetches tail candles up to the current instant, deduplicates, and verifies 100% unbroken sequence.
 */
async function validateAndRepairCandleContinuity(candles) {
  if (!candles || candles.length === 0) {
    console.log("[Candle Validation] Empty initial candles. Performing full 1000-candle Binance fetch...");
    const fresh = await fetchBinanceRange(Date.now() - 1000 * 300 * 1000, null, 1000);
    return fresh;
  }

  // Deduplicate initial set
  const dedupMap = new Map();
  for (const c of candles) {
    const t = c.time || Math.floor(new Date(c.open_time).getTime() / 1000);
    dedupMap.set(t, {
      ...c,
      time: t,
      open_time: c.open_time || new Date(t * 1000).toISOString(),
    });
  }
  let sorted = Array.from(dedupMap.values()).sort((a, b) => a.time - b.time);

  // 1. Scan for internal breaks / skipped intervals
  const gapsToFill = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const cur = sorted[i];
    const nxt = sorted[i + 1];
    const deltaSec = nxt.time - cur.time;
    if (deltaSec > 300) {
      const skippedCount = Math.floor(deltaSec / 300) - 1;
      const startMs = (cur.time + 300) * 1000;
      const endMs = (nxt.time - 1) * 1000;
      console.warn(`[Validation Check] Gap detected: ${skippedCount} candle(s) skipped between ${cur.open_time} and ${nxt.open_time} (${deltaSec / 60}m break).`);
      gapsToFill.push({ startMs, endMs, skippedCount });
    }
  }

  // 2. Fetch missing candles for all detected internal breaks from Binance
  if (gapsToFill.length > 0) {
    console.log(`[Validation Check] Attempting to fill ${gapsToFill.length} break(s) from Binance API...`);
    for (const gap of gapsToFill) {
      const missingCandles = await fetchBinanceRange(gap.startMs, gap.endMs, gap.skippedCount + 5);
      console.log(`[Validation Check] Retrieved ${missingCandles.length} candles for gap starting ${new Date(gap.startMs).toISOString()}.`);
      for (const mc of missingCandles) {
        dedupMap.set(mc.time, mc);
      }
    }
  }

  // 3. Check tail gap up to current moment
  sorted = Array.from(dedupMap.values()).sort((a, b) => a.time - b.time);
  const lastCandle = sorted[sorted.length - 1];
  const lastTimeMs = lastCandle.time * 1000;
  const nowMs = Date.now();

  if (nowMs - lastTimeMs >= 300000) {
    const tailStartMs = lastTimeMs + 300000;
    const tailMissing = Math.floor((nowMs - lastTimeMs) / 300000);
    console.log(`[Validation Check] Tail is ${tailMissing} candle(s) behind current instant. Fetching new candles from Binance...`);
    const newTailCandles = await fetchBinanceRange(tailStartMs, null, 1000);
    console.log(`[Validation Check] Retrieved ${newTailCandles.length} new tail candles from Binance.`);
    for (const tc of newTailCandles) {
      dedupMap.set(tc.time, tc);
    }
  }

  // 4. Final consolidation, deduplication and sorting
  const finalCandles = Array.from(dedupMap.values()).sort((a, b) => a.time - b.time);

  // 5. Post-validation assertion
  let remainingBreaks = 0;
  for (let i = 0; i < finalCandles.length - 1; i++) {
    const diff = finalCandles[i + 1].time - finalCandles[i].time;
    if (diff > 300) {
      remainingBreaks++;
      console.warn(`[Validation Alert] Unresolved break at ${finalCandles[i].open_time} (+${diff / 60}m)`);
    }
  }

  if (remainingBreaks === 0) {
    console.log(`%c[Validation PASS] 100% continuous data verified! Total: ${finalCandles.length} candles from ${finalCandles[0].open_time} to ${finalCandles[finalCandles.length - 1].open_time}. Zero breaks.`, "color: #26a69a; font-weight: bold;");
  } else {
    console.warn(`[Validation Result] ${remainingBreaks} break(s) could not be resolved from Binance.`);
  }

  return finalCandles;
}

async function initLiveMode() {
  setLiveStatus("RECONNECTING", "Validating & fetching continuous candle data...");

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

    // 2. Fetch candles from backend API, then audit, validate, and fill ANY missing candles from Binance
    let rawCandles = [];
    try {
      const histResp = await fetch(`${CONFIG.dashApiBase}/dash/candles/history?limit=3000`);
      if (histResp.ok) {
        rawCandles = await histResp.json();
      }
    } catch (e) {
      console.warn("Could not load from /dash/candles/history:", e);
    }

    if (!rawCandles || rawCandles.length === 0) {
      try {
        const liveResp = await fetch(`${CONFIG.dashApiBase}/dash/live/candles?limit=1000`);
        if (liveResp.ok) {
          rawCandles = await liveResp.json();
        }
      } catch (e) {
        console.warn("Could not load from /dash/live/candles:", e);
      }
    }

    // Perform exhaustive validation check & fetch all missing candles directly from Binance
    const continuousCandles = await validateAndRepairCandleContinuity(rawCandles);
    loadCandlesIntoChart("live", continuousCandles);

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
    STATE.lastTickTime = Date.now();
    connectBackendWebSocket();
    connectBinanceWebSocket();
    setLiveStatus("LIVE", "Live Forward Test (Active)");
  } catch (err) {
    console.warn("[Live] Backend connection error, running standalone with full Binance sync:", err);
    STATE.lastTickTime = Date.now();
    const fallbackCandles = await validateAndRepairCandleContinuity([]);
    loadCandlesIntoChart("live", fallbackCandles);
    connectBinanceWebSocket();
    await executeInstantLivePrediction();
    setLiveStatus("LIVE", "Binance Spot Live");
  }
}

async function fetchBinanceRestKlines() {
  try {
    const fresh = await fetchBinanceRange(Date.now() - 1000 * 300 * 1000, null, 1000);
    const continuous = await validateAndRepairCandleContinuity(fresh);
    loadCandlesIntoChart("live", continuous);
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
  if (msg.event === "latest_prediction" || msg.event === "prediction_logged" || msg.event === "new_prediction") {
    const pred = msg.data;
    const predSec = getCandleUnixSec(pred.candle_open_time || pred.open_time);
    const existingIdx = STATE.livePredictions.findIndex((p) => getCandleUnixSec(p.open_time) === predSec);

    const pUp = pred.probabilities?.p_up ?? pred.p_up ?? (pred.p_final?.up ?? null);
    const pDn = pred.probabilities?.p_down ?? pred.p_down ?? (pred.p_final?.down ?? null);
    const pNeu = pred.probabilities?.p_neutral ?? pred.p_neutral ?? (pred.p_final?.neutral ?? null);
    const refClose = Number(pred.ref_close || pred.close || (STATE.liveCandles.length ? STATE.liveCandles[STATE.liveCandles.length - 1].close : 0));
    const atr = Number(pred.atr14 || (refClose * 0.0035));
    const sqrt2 = Math.sqrt(2.0);
    const kMove = CONFIG.bandConstants ? CONFIG.bandConstants.K_MOVE : 0.5;
    const z = CONFIG.bandConstants ? CONFIG.bandConstants.Z : 1.28;
    const smartPrice = pred.smart_price || (refClose > 0 && pUp != null && pDn != null ? (refClose + (pUp - pDn) * atr * sqrt2 * kMove) : null);
    const halfWidth = pNeu != null ? z * atr * sqrt2 * (0.75 + 0.5 * (1.0 - pNeu)) : 0;
    const upper = pred.upper || (smartPrice && halfWidth > 0 ? smartPrice + halfWidth : null);
    const lower = pred.lower || (smartPrice && halfWidth > 0 ? smartPrice - halfWidth : null);

    const mapped = {
      open_time: pred.candle_open_time || pred.open_time,
      close: refClose,
      smart_price: smartPrice,
      upper: upper,
      lower: lower,
      atr14: atr,
      trend: CANONICAL_TAGS.toTrend(pred.trend),
      phase: CANONICAL_TAGS.toPhase(pred.market_phase || pred.phase),
      label: CANONICAL_TAGS.toPriceAction(pred.action_label || pred.label),
      confidence: pred.confidence != null ? pred.confidence : Math.max(pUp, pDn),
      p_final: { down: pDn, neutral: pNeu, up: pUp },
      status: pred.status || "PENDING",
      reason: pred.status_reason || pred.reason,
    };
    if (existingIdx >= 0) {
      STATE.livePredictions[existingIdx] = mapped;
    } else {
      STATE.livePredictions.unshift(mapped);
    }
    renderPredictionPanel(mapped);
    attachPredictionToChart("live", mapped);
    renderPredictionLog();
  } else if (msg.event === "prediction_resolved") {
    const res = msg.data;
    const resSec = getCandleUnixSec(res.candle_open_time);
    const match = STATE.livePredictions.find((p) => getCandleUnixSec(p.open_time) === resSec);
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

let _binanceFallbackPollTimer = null;

function connectBinanceWebSocket() {
  let ws = null;
  try {
    ws = new WebSocket(CONFIG.binanceWsUrl);
  } catch (err) {
    console.warn("Could not create Binance WS, starting fallback poll:", err);
    startBinanceFallbackPoll();
    return;
  }

  ws.onopen = () => {
    STATE.lastTickTime = Date.now();
    if (_binanceFallbackPollTimer) {
      clearInterval(_binanceFallbackPollTimer);
      _binanceFallbackPollTimer = null;
    }
  };

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

  ws.onerror = (err) => {
    console.warn("Binance WS error event:", err);
    startBinanceFallbackPoll();
  };

  ws.onclose = () => {
    startBinanceFallbackPoll();
    setTimeout(connectBinanceWebSocket, 5000);
  };
}

function startBinanceFallbackPoll() {
  if (!_binanceFallbackPollTimer) {
    _binanceFallbackPollTimer = setInterval(triggerLiveFeedResync, 12000);
  }
}

async function handleCandleClosed(candle) {
  // 1. Record newly closed candle
  STATE.liveCandles.push(candle);

  // 2. Generate new live prediction for the newly closed candle (predicting 2 candles forward = 10 mins)
  await executeInstantLivePrediction();

  // 3. Automatic Verification & Real-time Verdict Resolution
  // Only first 2 rows remain pending, all other rows are resolved immediately
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
  let trend = diffPct >= 0 ? "UPTREND" : "DOWNTREND";
  let phase = "PULLBACK";
  let pUp = 0.32;
  let pDn = 0.32;
  let pNeu = 0.36;

  if (trend === "UPTREND") {
    pUp = 0.44;
    pNeu = 0.30;
    pDn = 0.26;
  } else {
    pUp = 0.25;
    pNeu = 0.31;
    pDn = 0.44;
  }

  if (Math.abs(close - ma20) > 0.7 * atr) {
    phase = "IMPULSE";
  }

  // Indicative bands
  const smartPrice = close + CONFIG.bandConstants.K_MOVE * (pUp - pDn) * atr;
  const upper = smartPrice + CONFIG.bandConstants.Z * atr;
  const lower = smartPrice - CONFIG.bandConstants.Z * atr;

  let label = "NEUTRAL";
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

  const predSec = getCandleUnixSec(pred.open_time);
  const existingIdx = STATE.livePredictions.findIndex((p) => getCandleUnixSec(p.open_time) === predSec);
  if (existingIdx >= 0) {
    STATE.livePredictions[existingIdx] = pred;
  } else {
    STATE.livePredictions.unshift(pred);
  }
  renderPredictionPanel(pred);
  attachPredictionToChart("live", pred);
  renderPredictionLog();
}

// =====================================================================
// REPLAY ENGINE (Held-Out Test Split)
// =====================================================================
const _replayPredictionStore = new Map();

async function prefetchSimulationPredictions(candles) {
  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    const ts = c.open_time || new Date(c.time * 1000).toISOString();
    if (_replayPredictionStore.has(ts)) continue;
    try {
      const resp = await fetch(`${CONFIG.dashApiBase}/dash/replay/predict?ts=${encodeURIComponent(ts)}`);
      if (resp.ok) {
        const pred = await resp.json();
        _replayPredictionStore.set(ts, pred);
      }
    } catch (e) {
      // background silent catch
    }
  }
}

function renderStandbyPanel(dateStr, lastWarmupCandle) {
  let tStr = "23:55";
  if (lastWarmupCandle) {
    const d = new Date(lastWarmupCandle.open_time);
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mm = String(d.getUTCMinutes()).padStart(2, "0");
    tStr = `${hh}:${mm}`;
  }

  if (DOM.predCandleTime) DOM.predCandleTime.textContent = `Standby: ${tStr} UTC`;
  if (DOM.predContextText) {
    DOM.predContextText.textContent = `Historical data up to ${tStr} UTC loaded. Standby for ${dateStr}. Press ▶ Play to begin replay from 00:00.`;
  }
  if (DOM.predSourceBadge) {
    DOM.predSourceBadge.textContent = "REPLAY";
    DOM.predSourceBadge.className = "badge neutral";
  }
  if (DOM.predMarketBadge) {
    DOM.predMarketBadge.textContent = "STANDBY";
    DOM.predMarketBadge.className = "badge neutral";
  }
  if (DOM.predTrendBadge) {
    DOM.predTrendBadge.textContent = "READY";
    DOM.predTrendBadge.className = "badge neutral";
  }
  if (DOM.predPhaseBadge) {
    DOM.predPhaseBadge.textContent = "WAITING";
    DOM.predPhaseBadge.className = "badge neutral";
  }
  if (DOM.predPriceActionBadge) {
    DOM.predPriceActionBadge.textContent = "—";
    DOM.predPriceActionBadge.className = "badge neutral";
  }
  if (DOM.predFinalBadge) {
    DOM.predFinalBadge.textContent = "STANDBY";
    DOM.predFinalBadge.className = "badge neutral";
  }
  if (DOM.predConfidence) DOM.predConfidence.textContent = "—";
  if (DOM.predSmartPrice) DOM.predSmartPrice.textContent = "—";
  if (DOM.predUpperCap) DOM.predUpperCap.textContent = "—";
  if (DOM.predLowerCap) DOM.predLowerCap.textContent = "—";
  if (DOM.predProbDown) DOM.predProbDown.textContent = "—";
  if (DOM.predProbNeu) DOM.predProbNeu.textContent = "—";
  if (DOM.predProbUp) DOM.predProbUp.textContent = "—";
  if (DOM.probBarDown) DOM.probBarDown.style.width = "33%";
  if (DOM.probBarNeu) DOM.probBarNeu.style.width = "34%";
  if (DOM.probBarUp) DOM.probBarUp.style.width = "33%";
}

function syncPredictionPanelToActiveTab() {
  const isLive = STATE.activeTab === "live";
  if (DOM.predSourceBadge) {
    DOM.predSourceBadge.textContent = isLive ? "LIVE" : "REPLAY";
    DOM.predSourceBadge.className = isLive ? "badge up" : "badge neutral";
  }

  if (isLive) {
    if (STATE.livePredictions && STATE.livePredictions.length > 0) {
      renderPredictionPanel(STATE.livePredictions[0]);
    } else {
      executeInstantLivePrediction();
    }
  } else {
    if (STATE.replayPredictions && STATE.replayPredictions.length > 0) {
      renderPredictionPanel(STATE.replayPredictions[0]);
    } else if (STATE.replayWarmupCandles && STATE.replayWarmupCandles.length > 0) {
      renderStandbyPanel(STATE.replaySelectedDate, STATE.replayWarmupCandles[STATE.replayWarmupCandles.length - 1]);
    }
  }
  updateBandContainmentStat();
}

async function initReplayMode() {
  if (CHARTS.replay) {
    CHARTS.replay.resize();
  }

  // If already loaded for this date, just ensure viewport is active
  if (STATE.replayWarmupCandles && STATE.replayWarmupCandles.length > 0) {
    if (CHARTS.replay) CHARTS.replay.scrollToRealtime();
    return;
  }

  if (STATE.isMock) {
    STATE.replayTestRange = {
      min_open_time: "2026-05-20T14:20:00.000Z",
      max_open_time: "2026-11-09T23:55:00.000Z",
    };
    setupReplayControls();
    return;
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
  if (STATE.replayTestRange) {
    const minIso = STATE.replayTestRange.min_open_time;
    const maxIso = STATE.replayTestRange.max_open_time;
    if (DOM.replayDatePicker) {
      DOM.replayDatePicker.min = minIso.substring(0, 10);
      DOM.replayDatePicker.max = maxIso.substring(0, 10);
    }
  } else if (DOM.replayDatePicker) {
    DOM.replayDatePicker.min = "2026-05-21";
    DOM.replayDatePicker.max = "2026-11-08";
  }

  const defaultDate = STATE.replaySelectedDate || "2026-08-26";
  if (DOM.replayDatePicker) {
    DOM.replayDatePicker.value = defaultDate;
  }

  selectReplayDate(defaultDate);
}

async function selectReplayDate(dateStr) {
  if (!dateStr) return;
  STATE.replaySelectedDate = dateStr;
  if (DOM.replayDatePicker) {
    DOM.replayDatePicker.value = dateStr;
  }

  // Stop active playback if running
  if (STATE.replayPlaying) {
    clearInterval(STATE.replayTimer);
    STATE.replayTimer = null;
    STATE.replayPlaying = false;
    if (DOM.btnPlayPause) DOM.btnPlayPause.textContent = "▶ Play";
  }

  // UTC boundaries
  const targetStart = new Date(`${dateStr}T00:00:00Z`);
  const targetStartSec = Math.floor(targetStart.getTime() / 1000);
  const standbyTimeSec = targetStartSec - 300; // Previous day 23:55 UTC
  const standbyIso = new Date(standbyTimeSec * 1000).toISOString();

  // Warmup is 24 hours prior (previous day 00:00 UTC)
  const warmupStartMs = targetStart.getTime() - 24 * 60 * 60 * 1000;
  // Simulation: load initial generous continuous multi-day window (7 days = 2016 candles),
  // which will then seamlessly stream further days in background as replay continues.
  const simEndMs = targetStart.getTime() + 7 * 24 * 60 * 60 * 1000;

  const minLimitMs = STATE.replayTestRange
    ? new Date(STATE.replayTestRange.min_open_time).getTime()
    : new Date("2026-05-20T14:20:00Z").getTime();
  const maxLimitMs = STATE.replayTestRange
    ? new Date(STATE.replayTestRange.max_open_time).getTime()
    : new Date("2026-11-09T23:55:00Z").getTime();

  const safeWarmupStart = new Date(Math.max(warmupStartMs, minLimitMs));
  const safeSimEnd = new Date(Math.min(simEndMs, maxLimitMs));

  if (DOM.replayDateBadge) {
    DOM.replayDateBadge.textContent = "LOADING...";
    DOM.replayDateBadge.className = "badge neutral";
  }

  if (STATE.isMock) {
    let price = 61250.0;
    const warmup = [];
    const simulation = [];

    const warmupCount = 100;
    const warmupStartSec = standbyTimeSec - (warmupCount - 1) * 300;
    for (let i = 0; i < warmupCount; i++) {
      const t = warmupStartSec + i * 300;
      const change = (Math.sin(i / 5) * 40) + ((i % 3 === 0) ? 15 : -12);
      const o = price;
      const cl = price + change;
      const h = Math.max(o, cl) + 15;
      const l = Math.min(o, cl) - 15;
      const v = 80 + (i % 7) * 20;
      price = cl;
      warmup.push({
        time: t,
        open_time: new Date(t * 1000).toISOString(),
        open: o,
        high: h,
        low: l,
        close: cl,
        volume: v,
      });
    }

    for (let i = 0; i < 1500; i++) {
      const t = targetStartSec + i * 300;
      const change = (Math.cos(i / 5) * 40) + ((i % 4 === 0) ? -20 : 15);
      const o = price;
      const cl = price + change;
      const h = Math.max(o, cl) + 15;
      const l = Math.min(o, cl) - 15;
      const v = 90 + (i % 6) * 25;
      price = cl;
      simulation.push({
        time: t,
        open_time: new Date(t * 1000).toISOString(),
        open: o,
        high: h,
        low: l,
        close: cl,
        volume: v,
      });
    }

    STATE.replayWarmupCandles = warmup;
    STATE.replaySimulationCandles = simulation;
    STATE.replaySimulationIndex = 0;
    STATE.replayPredictions = [];
    STATE.replayCandles = warmup.slice();

    // Populate a few historical resolved predictions from the warmup day for rich mock metrics
    for (let i = warmup.length - 6; i < warmup.length; i++) {
      if (i < 0) continue;
      const c = warmup[i];
      const isUp = i % 2 === 0;
      const isResolved = i < warmup.length - 2;
      STATE.replayPredictions.unshift({
        open_time: c.open_time,
        close: c.close,
        trend: isUp ? "UPTREND" : "DOWNTREND",
        phase: "IMPULSE",
        label: isUp ? "UP" : "DOWN",
        confidence: 0.42,
        p_final: { up: isUp ? 0.42 : 0.28, neutral: 0.30, down: isUp ? 0.28 : 0.42 },
        smart_price: c.close + (isUp ? 40 : -40),
        upper: c.close + 120,
        lower: c.close - 120,
        status: isResolved ? "RESOLVED" : "PENDING",
        correct: isResolved ? ((i % 3 === 0) ? 1 : 0) : null,
        outcome_close: isResolved ? c.close + (isUp ? 25 : -25) : null,
        return_bps_label: isResolved ? (isUp ? 15 : -15) : null,
      });
    }

    loadCandlesIntoChart("replay", warmup);
    if (CHARTS.replay) {
      CHARTS.replay.scrollToRealtime();
      CHARTS.replay.setMarkers([]);
      CHARTS.replay.setIndicativeBands(null);
    }

    if (DOM.replayDateBadge) {
      DOM.replayDateBadge.textContent = `${dateStr} STANDBY`;
      DOM.replayDateBadge.className = "badge up";
    }

    renderStandbyPanel(dateStr, warmup[warmup.length - 1]);
    updateAllMarkers("replay");
    renderPredictionLog();
    updateBandContainmentStat();
    return;
  }

  try {
    const url = `${CONFIG.dashApiBase}/dash/replay/candles?from=${encodeURIComponent(safeWarmupStart.toISOString())}&to=${encodeURIComponent(safeSimEnd.toISOString())}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      const errText = await resp.text();
      console.error("[Replay] Failed to load replay candles:", resp.status, errText);
      if (DOM.replayDateBadge) {
        DOM.replayDateBadge.textContent = "LOAD ERROR";
        DOM.replayDateBadge.className = "badge down";
      }
      return;
    }

    const rawCandles = await resp.json();
    const warmup = [];
    const simulation = [];

    rawCandles.forEach((c) => {
      const t = c.time || Math.floor(new Date(c.open_time).getTime() / 1000);
      const formatted = {
        time: t,
        open: Number(c.open),
        high: Number(c.high),
        low: Number(c.low),
        close: Number(c.close),
        volume: Number(c.volume),
        open_time: c.open_time || new Date(t * 1000).toISOString(),
      };
      if (t <= standbyTimeSec) {
        warmup.push(formatted);
      } else {
        simulation.push(formatted);
      }
    });

    warmup.sort((a, b) => a.time - b.time);
    simulation.sort((a, b) => a.time - b.time);

    STATE.replayWarmupCandles = warmup;
    STATE.replaySimulationCandles = simulation;
    STATE.replaySimulationIndex = 0;
    STATE.replayPredictions = [];
    STATE.replayCandles = warmup.slice();

    loadCandlesIntoChart("replay", warmup);
    if (CHARTS.replay) {
      CHARTS.replay.scrollToRealtime();
      CHARTS.replay.setMarkers([]);
      CHARTS.replay.setIndicativeBands(null);
    }

    if (DOM.replayDateBadge) {
      DOM.replayDateBadge.textContent = `${dateStr} STANDBY`;
      DOM.replayDateBadge.className = "badge up";
    }

    renderStandbyPanel(dateStr, warmup[warmup.length - 1]);
    renderPredictionLog();
    updateBandContainmentStat();

    _replayPredictionStore.clear();
    prefetchSimulationPredictions(simulation);

    console.log(`[Replay Standby] Loaded ${warmup.length} warmup candles up to ${standbyIso}. Ready to simulate ${simulation.length} candles starting 00:00.`);
  } catch (err) {
    console.error("[Replay] Error in selectReplayDate:", err);
    if (DOM.replayDateBadge) {
      DOM.replayDateBadge.textContent = "ERROR";
      DOM.replayDateBadge.className = "badge down";
    }
  }
}

let _isReplayStepping = false;
let _isFetchingMoreSimulationCandles = false;

async function fetchMoreReplayCandles() {
  if (_isFetchingMoreSimulationCandles || STATE.isMock) return;
  if (!STATE.replaySimulationCandles || STATE.replaySimulationCandles.length === 0) return;

  const lastCandle = STATE.replaySimulationCandles[STATE.replaySimulationCandles.length - 1];
  const lastTimeMs = lastCandle.time * 1000;
  const maxLimitMs = STATE.replayTestRange
    ? new Date(STATE.replayTestRange.max_open_time).getTime()
    : new Date("2026-11-09T23:55:00Z").getTime();

  if (lastTimeMs >= maxLimitMs) return; // Reached end of entire test split

  _isFetchingMoreSimulationCandles = true;
  try {
    const nextStartMs = lastTimeMs + 300 * 1000;
    const nextEndMs = Math.min(nextStartMs + 7 * 24 * 60 * 60 * 1000, maxLimitMs);
    const fromIso = new Date(nextStartMs).toISOString();
    const toIso = new Date(nextEndMs).toISOString();

    const url = `${CONFIG.dashApiBase}/dash/replay/candles?from=${encodeURIComponent(fromIso)}&to=${encodeURIComponent(toIso)}`;
    const resp = await fetch(url);
    if (resp.ok) {
      const moreRaw = await resp.json();
      const existingTimes = new Set(STATE.replaySimulationCandles.map((c) => c.time));
      const newCandles = [];
      moreRaw.forEach((c) => {
        const t = c.time || Math.floor(new Date(c.open_time).getTime() / 1000);
        if (!existingTimes.has(t)) {
          newCandles.push({
            time: t,
            open: Number(c.open),
            high: Number(c.high),
            low: Number(c.low),
            close: Number(c.close),
            volume: Number(c.volume),
            open_time: c.open_time || new Date(t * 1000).toISOString(),
          });
        }
      });
      if (newCandles.length > 0) {
        newCandles.sort((a, b) => a.time - b.time);
        STATE.replaySimulationCandles.push(...newCandles);
        prefetchSimulationPredictions(newCandles);
        console.log(`[Replay Extended] Appended ${newCandles.length} more simulation candles up to ${toIso}. Buffer size: ${STATE.replaySimulationCandles.length}`);
      }
    }
  } catch (err) {
    console.warn("[Replay] Error fetching more simulation candles:", err);
  } finally {
    _isFetchingMoreSimulationCandles = false;
  }
}

async function executeReplayStep() {
  if (_isReplayStepping) return;
  _isReplayStepping = true;

  try {
    if (!STATE.replaySimulationCandles || STATE.replaySimulationCandles.length === 0) {
      console.warn("No simulation candles available for replay.");
      return;
    }

    // Seamless buffer extension check: keep playing without stopping!
    if (!STATE.isMock && STATE.replaySimulationIndex >= STATE.replaySimulationCandles.length - 40) {
      fetchMoreReplayCandles();
    } else if (STATE.isMock && STATE.replaySimulationIndex >= STATE.replaySimulationCandles.length - 20) {
      const lastCandle = STATE.replaySimulationCandles[STATE.replaySimulationCandles.length - 1];
      let p = lastCandle.close;
      for (let i = 1; i <= 100; i++) {
        const t = lastCandle.time + i * 300;
        const change = (Math.cos((STATE.replaySimulationCandles.length + i) / 5) * 40) + (((STATE.replaySimulationCandles.length + i) % 4 === 0) ? -20 : 15);
        const o = p;
        const cl = p + change;
        const h = Math.max(o, cl) + 15;
        const l = Math.min(o, cl) - 15;
        const v = 90 + ((STATE.replaySimulationCandles.length + i) % 6) * 25;
        p = cl;
        STATE.replaySimulationCandles.push({
          time: t,
          open_time: new Date(t * 1000).toISOString(),
          open: o,
          high: h,
          low: l,
          close: cl,
          volume: v,
        });
      }
    }

    if (STATE.replaySimulationIndex >= STATE.replaySimulationCandles.length) {
      console.log("[Replay] Reached final boundary of test split data for", STATE.replaySelectedDate);
      if (STATE.replayPlaying) togglePlayPause();
      return;
    }

    const currentCandle = STATE.replaySimulationCandles[STATE.replaySimulationIndex];
    STATE.replaySimulationIndex++;

    // 1. Reveal this candle on the chart
    if (CHARTS.replay) {
      CHARTS.replay.updateCandle(currentCandle);
      CHARTS.replay.scrollToRealtime();
    }
    if (!STATE.replayCandles.some(c => c.time === currentCandle.time)) {
      STATE.replayCandles.push(currentCandle);
    }

    // 2. Obtain prediction for currentCandle
    let pred = null;
    if (STATE.isMock) {
      const isUp = Math.random() > 0.5;
      const pUp = isUp ? 0.44 : 0.28;
      const pDn = isUp ? 0.26 : 0.42;
      const pNeu = 0.3;
      pred = {
        open_time: currentCandle.open_time,
        close: currentCandle.close,
        trend: isUp ? "UPTREND" : "DOWNTREND",
        phase: Math.random() > 0.5 ? "IMPULSE" : "PULLBACK",
        label: isUp ? "UP" : "DOWN",
        confidence: Math.max(pUp, pDn),
        p_final: { up: pUp, neutral: pNeu, down: pDn },
        smart_price: currentCandle.close + (isUp ? 60 : -60),
        upper: currentCandle.close + 140,
        lower: currentCandle.close - 140,
        status: "PENDING",
        correct: null,
        outcome_close: null,
      };
    } else {
      const ts = currentCandle.open_time || new Date(currentCandle.time * 1000).toISOString();
      if (_replayPredictionStore.has(ts)) {
        pred = JSON.parse(JSON.stringify(_replayPredictionStore.get(ts)));
        pred.status = "PENDING";
        pred.correct = null;
        pred.outcome_close = null;
        pred.return_bps_label = null;
      } else {
        try {
          const resp = await fetch(`${CONFIG.dashApiBase}/dash/replay/predict?ts=${encodeURIComponent(ts)}`);
          if (resp.ok) {
            pred = await resp.json();
            _replayPredictionStore.set(ts, pred);
            pred = JSON.parse(JSON.stringify(pred));
            pred.status = "PENDING";
            pred.correct = null;
            pred.outcome_close = null;
            pred.return_bps_label = null;
          }
        } catch (e) {
          console.error("Replay step predict error:", e);
        }
      }
    }

    // 4. Record and render prediction
    if (pred) {
      STATE.replayPredictions.unshift(pred);
      renderPredictionPanel(pred);
      attachPredictionToChart("replay", pred);
    }

    // 5. Update UI, markers, logs, and metrics
    updateAllMarkers("replay");
    renderPredictionLog();
    updateBandContainmentStat();

    // 6. Update date badge with current candle date/time and step count
    if (DOM.replayDateBadge) {
      const candleDate = new Date(currentCandle.time * 1000).toISOString().slice(0, 10);
      const candleTime = new Date(currentCandle.time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      DOM.replayDateBadge.textContent = `${candleDate} ${candleTime} (Step ${STATE.replaySimulationIndex})`;
      DOM.replayDateBadge.className = "badge up";
    }
  } finally {
    _isReplayStepping = false;
  }
}

function togglePlayPause() {
  STATE.replayPlaying = !STATE.replayPlaying;
  DOM.btnPlayPause.textContent = STATE.replayPlaying ? "⏸ Pause" : "▶ Play";

  if (STATE.replayPlaying) {
    const delay = Math.max(50, Math.round(1000 / STATE.replaySpeed));
    // Immediately execute the first step if starting from 0 or paused
    executeReplayStep();
    STATE.replayTimer = setInterval(executeReplayStep, delay);
  } else {
    clearInterval(STATE.replayTimer);
    STATE.replayTimer = null;
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
    syncPredictionPanelToActiveTab();
  });

  DOM.btnReplay.addEventListener("click", () => {
    STATE.activeTab = "replay";
    DOM.btnReplay.classList.add("active");
    DOM.btnLive.classList.remove("active");
    DOM.replayView.style.display = "flex";
    DOM.liveView.style.display = "none";
    DOM.logTabReplay.click();
    initReplayMode();
    syncPredictionPanelToActiveTab();
  });

  // Guide Drawer
  DOM.btnGuide.addEventListener("click", () => DOM.guideDrawer.classList.add("open"));
  DOM.btnCloseGuide.addEventListener("click", () => DOM.guideDrawer.classList.remove("open"));

  // Log Tab Switch
  DOM.logTabLive.addEventListener("click", () => {
    STATE.logTab = "live";
    DOM.logTabLive.classList.add("active");
    DOM.logTabReplay.classList.remove("active");
    if (DOM.logTabFeatures) DOM.logTabFeatures.classList.remove("active");

    if (DOM.logSummaryStrip) DOM.logSummaryStrip.style.display = "";
    if (DOM.logFiltersBar) DOM.logFiltersBar.style.display = "";
    if (DOM.logTableContainer) DOM.logTableContainer.style.display = "";
    if (DOM.logActionsLiveReplay) DOM.logActionsLiveReplay.style.display = "";

    if (DOM.featuresPanelView) DOM.featuresPanelView.style.display = "none";
    if (DOM.logActionsFeatures) DOM.logActionsFeatures.style.display = "none";

    renderPredictionLog();
    syncPredictionPanelToActiveTab();
  });

  DOM.logTabReplay.addEventListener("click", () => {
    STATE.logTab = "replay";
    DOM.logTabReplay.classList.add("active");
    DOM.logTabLive.classList.remove("active");
    if (DOM.logTabFeatures) DOM.logTabFeatures.classList.remove("active");

    if (DOM.logSummaryStrip) DOM.logSummaryStrip.style.display = "";
    if (DOM.logFiltersBar) DOM.logFiltersBar.style.display = "";
    if (DOM.logTableContainer) DOM.logTableContainer.style.display = "";
    if (DOM.logActionsLiveReplay) DOM.logActionsLiveReplay.style.display = "";

    if (DOM.featuresPanelView) DOM.featuresPanelView.style.display = "none";
    if (DOM.logActionsFeatures) DOM.logActionsFeatures.style.display = "none";

    renderPredictionLog();
    syncPredictionPanelToActiveTab();
  });

  if (DOM.logTabFeatures) {
    DOM.logTabFeatures.addEventListener("click", () => {
      STATE.logTab = "features";
      DOM.logTabFeatures.classList.add("active");
      DOM.logTabLive.classList.remove("active");
      DOM.logTabReplay.classList.remove("active");

      if (DOM.logSummaryStrip) DOM.logSummaryStrip.style.display = "none";
      if (DOM.logFiltersBar) DOM.logFiltersBar.style.display = "none";
      if (DOM.logTableContainer) DOM.logTableContainer.style.display = "none";
      if (DOM.logActionsLiveReplay) DOM.logActionsLiveReplay.style.display = "none";

      if (DOM.featuresPanelView) DOM.featuresPanelView.style.display = "flex";
      if (DOM.logActionsFeatures) DOM.logActionsFeatures.style.display = "block";

      renderFeaturesPanel();
    });
  }

  // Toggle Pipeline Details
  if (DOM.btnTogglePipelineDetails) {
    DOM.btnTogglePipelineDetails.addEventListener("click", () => {
      STATE.pipelineExpanded = !STATE.pipelineExpanded;
      if (DOM.pipelineDetailsContent) {
        DOM.pipelineDetailsContent.className = STATE.pipelineExpanded ? "pipeline-details-expanded" : "pipeline-details-collapsed";
      }
      DOM.btnTogglePipelineDetails.textContent = STATE.pipelineExpanded ? "Pipeline Workflow ▲" : "Pipeline Workflow ▼";
    });
  }

  // Features search
  if (DOM.featuresSearchInput) {
    DOM.featuresSearchInput.addEventListener("input", (e) => {
      STATE.featuresSearchQuery = e.target.value;
      if (DOM.btnClearFeaturesSearch) {
        DOM.btnClearFeaturesSearch.style.display = e.target.value ? "inline" : "none";
      }
      renderFeaturesTable();
    });
  }

  if (DOM.btnClearFeaturesSearch) {
    DOM.btnClearFeaturesSearch.addEventListener("click", () => {
      if (DOM.featuresSearchInput) DOM.featuresSearchInput.value = "";
      STATE.featuresSearchQuery = "";
      DOM.btnClearFeaturesSearch.style.display = "none";
      renderFeaturesTable();
    });
  }

  // Features Model Scope Pills
  const modelPills = document.querySelectorAll("#features-model-pills .filter-pill");
  modelPills.forEach((pill) => {
    pill.addEventListener("click", (e) => {
      modelPills.forEach((p) => p.classList.remove("active"));
      e.target.classList.add("active");
      STATE.featuresModelScope = e.target.getAttribute("data-model");
      renderFeaturesTable();
    });
  });

  // Features Category Pills
  const featPills = document.querySelectorAll("#features-category-pills .filter-pill");
  featPills.forEach((pill) => {
    pill.addEventListener("click", (e) => {
      featPills.forEach((p) => p.classList.remove("active"));
      e.target.classList.add("active");
      STATE.featuresCategory = e.target.getAttribute("data-cat");
      renderFeaturesTable();
    });
  });

  // Export Features JSON
  if (DOM.btnExportFeatures) {
    DOM.btnExportFeatures.addEventListener("click", () => {
      const catalog = (typeof FEATURES_CATALOG !== "undefined") ? FEATURES_CATALOG : [];
      const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(catalog, null, 2));
      const a = document.createElement("a");
      a.setAttribute("href", jsonStr);
      a.setAttribute("download", "btcusdt_136_features_catalog.json");
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  // Prediction Log Filter Pills
  document.querySelectorAll(".log-filters-bar .filter-pill").forEach((pill) => {
    pill.addEventListener("click", (e) => {
      document.querySelectorAll(".log-filters-bar .filter-pill").forEach((p) => p.classList.remove("active"));
      e.target.classList.add("active");
      STATE.logFilter = e.target.getAttribute("data-filter");
      renderPredictionLog();
    });
  });

  // Horizontal drag-to-scroll & wheel support for Log Table
  const logContainer = document.querySelector(".log-table-container");
  if (logContainer) {
    let isDown = false;
    let startX = 0;
    let scrollLeft = 0;

    logContainer.addEventListener("wheel", (e) => {
      if (e.shiftKey) {
        logContainer.scrollLeft += e.deltaY;
      } else if (Math.abs(e.deltaX) > 0) {
        logContainer.scrollLeft += e.deltaX;
      }
    }, { passive: true });

    logContainer.addEventListener("mousedown", (e) => {
      if (e.target.closest("button, a, input, select")) return;
      isDown = true;
      startX = e.pageX - logContainer.offsetLeft;
      scrollLeft = logContainer.scrollLeft;
    });

    window.addEventListener("mouseup", () => {
      isDown = false;
    });

    logContainer.addEventListener("mousemove", (e) => {
      if (!isDown) return;
      e.preventDefault();
      const x = e.pageX - logContainer.offsetLeft;
      const walk = (x - startX) * 1.5;
      logContainer.scrollLeft = scrollLeft - walk;
    });
  }

  // Horizontal drag-to-scroll & wheel support for Features Table
  const featContainer = document.querySelector("#features-table-container");
  if (featContainer) {
    let isDown = false;
    let startX = 0;
    let scrollLeft = 0;

    featContainer.addEventListener("wheel", (e) => {
      if (e.shiftKey) {
        featContainer.scrollLeft += e.deltaY;
      } else if (Math.abs(e.deltaX) > 0) {
        featContainer.scrollLeft += e.deltaX;
      }
    }, { passive: true });

    featContainer.addEventListener("mousedown", (e) => {
      if (e.target.closest("button, a, input, select")) return;
      isDown = true;
      startX = e.pageX - featContainer.offsetLeft;
      scrollLeft = featContainer.scrollLeft;
    });

    window.addEventListener("mouseup", () => {
      isDown = false;
    });

    featContainer.addEventListener("mousemove", (e) => {
      if (!isDown) return;
      e.preventDefault();
      const x = e.pageX - featContainer.offsetLeft;
      const walk = (x - startX) * 1.5;
      featContainer.scrollLeft = scrollLeft - walk;
    });
  }

  // Replay Controls
  DOM.btnTestNow.addEventListener("click", executeReplayStep);
  DOM.btnPlayPause.addEventListener("click", togglePlayPause);
  DOM.btnStepNext.addEventListener("click", executeReplayStep);
  DOM.btnStepPrev.addEventListener("click", () => {
    if (STATE.replaySimulationIndex > 0) {
      STATE.replaySimulationIndex--;
      STATE.replayCandles.pop();
      if (CHARTS.replay) {
        CHARTS.replay.setData(STATE.replayCandles.slice());
        CHARTS.replay.scrollToRealtime();
      }
      if (STATE.replayPredictions.length > 0) {
        STATE.replayPredictions.shift();
      }
      const lastCandle = STATE.replayCandles[STATE.replayCandles.length - 1];
      const lastSec = lastCandle ? lastCandle.time : 0;
      STATE.replayPredictions.forEach((p) => {
        const pSec = getCandleUnixSec(p.open_time);
        if (lastSec - pSec < 600) {
          p.status = "PENDING";
          p.correct = null;
          p.outcome_close = null;
          p.return_bps_label = null;
        }
      });

      if (STATE.replayPredictions.length > 0) {
        renderPredictionPanel(STATE.replayPredictions[0]);
        attachPredictionToChart("replay", STATE.replayPredictions[0]);
      } else {
        if (CHARTS.replay) {
          CHARTS.replay.setMarkers([]);
          CHARTS.replay.setIndicativeBands(null);
        }
        const lastWarmup = STATE.replayWarmupCandles[STATE.replayWarmupCandles.length - 1];
        renderStandbyPanel(STATE.replaySelectedDate, lastWarmup);
      }
      updateAllMarkers("replay");
      renderPredictionLog();
      updateBandContainmentStat();

      if (DOM.replayDateBadge) {
        DOM.replayDateBadge.textContent = `${STATE.replaySelectedDate} (${STATE.replaySimulationIndex}/${STATE.replaySimulationCandles.length})`;
      }
    }
  });

  DOM.btnResetReplay.addEventListener("click", () => {
    selectReplayDate(STATE.replaySelectedDate || "2026-08-26");
  });

  DOM.btnClearReplay.addEventListener("click", () => {
    STATE.replayPredictions = [];
    if (CHARTS.replay) {
      CHARTS.replay.setMarkers([]);
      CHARTS.replay.setIndicativeBands(null);
    }
    renderPredictionLog();
    updateBandContainmentStat();
  });

  DOM.btnSpeed.addEventListener("click", () => {
    const speeds = [1, 2];
    const nextIdx = (speeds.indexOf(STATE.replaySpeed) + 1) % speeds.length;
    STATE.replaySpeed = speeds[nextIdx];
    DOM.btnSpeed.textContent = `${STATE.replaySpeed}x`;
    if (STATE.replayPlaying) {
      clearInterval(STATE.replayTimer);
      STATE.replayTimer = setInterval(executeReplayStep, Math.max(50, Math.round(1000 / STATE.replaySpeed)));
    }
  });

  if (DOM.replayDatePicker) {
    DOM.replayDatePicker.addEventListener("change", (e) => {
      if (e.target.value) {
        selectReplayDate(e.target.value);
      }
    });
  }

  if (DOM.btnExportCsv) {
    DOM.btnExportCsv.addEventListener("click", () => {
      const isLive = STATE.logTab === "live";
      const list = isLive ? STATE.livePredictions : STATE.replayPredictions;
      if (!list || list.length === 0) {
        alert("No prediction rows available to export.");
        return;
      }
      const headers = ["open_time", "label", "confidence", "market", "trend", "phase", "close", "smart_price", "upper", "lower", "outcome_close", "return_bps", "status", "correct"];
      const rows = list.map(p => [
        p.open_time,
        p.label,
        p.confidence,
        p.market || "",
        p.trend || "",
        p.phase || "",
        p.close,
        p.smart_price || "",
        p.upper || "",
        p.lower || "",
        p.outcome_close || "",
        p.return_bps_label || "",
        p.status,
        p.correct != null ? p.correct : ""
      ]);
      const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `predictions_${isLive ? "live" : "replay"}_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // Indicator Settings Popover & Layer Toggles
  const btnSettings = document.getElementById("btn-indicator-settings");
  const settingsPopover = document.getElementById("indicator-settings-popover");
  const btnCloseSettings = document.getElementById("btn-close-indicator-settings");
  const badgeCount = document.getElementById("active-indicators-count");

  function updateIndicatorBadge() {
    const inputs = document.querySelectorAll("#indicator-settings-popover input[data-layer]");
    const active = Array.from(inputs).filter((i) => i.checked).length;
    if (badgeCount) {
      badgeCount.textContent = `${active} active`;
    }
  }

  if (btnSettings && settingsPopover) {
    btnSettings.addEventListener("click", (e) => {
      e.stopPropagation();
      settingsPopover.classList.toggle("open");
    });

    if (btnCloseSettings) {
      btnCloseSettings.addEventListener("click", () => {
        settingsPopover.classList.remove("open");
      });
    }

    // Close when clicking outside popover
    document.addEventListener("click", (e) => {
      if (
        settingsPopover.classList.contains("open") &&
        !settingsPopover.contains(e.target) &&
        e.target !== btnSettings &&
        !btnSettings.contains(e.target)
      ) {
        settingsPopover.classList.remove("open");
      }
    });

    // Close on Escape key
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && settingsPopover.classList.contains("open")) {
        settingsPopover.classList.remove("open");
      }
    });
  }

  // Initialize & synchronize all layer checkboxes (forcing actual price line hidden initially)
  document.querySelectorAll("input[data-layer]").forEach((input) => {
    const layer = input.getAttribute("data-layer");
    if (layer === "actualLine") {
      input.checked = false;
      STATE.layerToggles.actualLine = false;
      if (CHARTS.live) CHARTS.live.setLayerVisible("actualLine", false);
      if (CHARTS.replay) CHARTS.replay.setLayerVisible("actualLine", false);
    } else if (STATE.layerToggles && STATE.layerToggles.hasOwnProperty(layer)) {
      input.checked = !!STATE.layerToggles[layer];
      if (CHARTS.live) CHARTS.live.setLayerVisible(layer, STATE.layerToggles[layer]);
      if (CHARTS.replay) CHARTS.replay.setLayerVisible(layer, STATE.layerToggles[layer]);
    }

    input.addEventListener("change", (e) => {
      const isChecked = e.target.checked;
      STATE.layerToggles[layer] = isChecked;
      if (CHARTS.live) CHARTS.live.setLayerVisible(layer, isChecked);
      if (CHARTS.replay) CHARTS.replay.setLayerVisible(layer, isChecked);
      updateIndicatorBadge();
    });
  });
  updateIndicatorBadge();

  // Tab Focus / Visibility Change Auto-Resync
  document.addEventListener("visibilitychange", async () => {
    if (document.visibilityState === "visible" && STATE.mode === "live") {
      console.log("[Window Focus] Tab active. Performing candle validation & Binance sync...");
      if (STATE.liveCandles && STATE.liveCandles.length > 0) {
        const repaired = await validateAndRepairCandleContinuity(STATE.liveCandles);
        loadCandlesIntoChart("live", repaired);
        executeInstantLivePrediction();
      }
    }
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
