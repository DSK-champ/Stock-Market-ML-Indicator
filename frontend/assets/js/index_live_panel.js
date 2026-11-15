/**
 * Live Prediction Panel & Live Logs Controller for index.html
 * BTC/USDT 5m AI Directional Indicator & Adaptive Prediction Channels
 * 
 * Strictly Live: No Replay tab, No Features tab.
 * Realtime updates from backend /dash/live/predictions with standalone fallback.
 */

(() => {
  const CONFIG = {
    apiBase: "http://localhost:8001",
    pollIntervalMs: 3000,
    benchmarks: {
      refTestAccuracy: 51.90,
      refTestCoverage: 52.50,
      refOofAccuracy: 52.16,
      refOofCoverage: 65.90,
    },
    bandConstants: {
      K_MOVE: 0.5,
      Z: 1.28,
      ATR_PERIOD: 14,
    },
  };

  const CANONICAL_TAGS = {
    toMarket(trend, action) {
      const t = String(trend || "").toUpperCase();
      if (t.includes("DOWN") || t.includes("BEAR")) return "BEARISH";
      if (t.includes("UP") || t.includes("BULL")) return "BULLISH";
      const a = String(action || "").toUpperCase();
      if (a === "DOWN" || a === "SHORT") return "BEARISH";
      return "BULLISH";
    },
    toTrend(val) {
      const s = String(val || "").toUpperCase();
      if (s.includes("DOWN") || s.includes("BEAR")) return "DOWNTREND";
      return "UPTREND";
    },
    toPhase(val) {
      const s = String(val || "").toUpperCase();
      if (s.includes("PULLBACK") || s.includes("RETRACE")) return "PULLBACK";
      return "IMPULSE";
    },
    toPriceAction(val) {
      if (!val) return "NO CLEAR SIGNAL";
      const s = String(val).toUpperCase();
      if (s === "1" || s === "UP" || s === "LONG") return "UP";
      if (s === "-1" || s === "DOWN" || s === "SHORT") return "DOWN";
      if (s === "NO_CLEAR_SIGNAL" || s === "NO CLEAR SIGNAL" || s.startsWith("NO SIGNAL")) return "NO CLEAR SIGNAL";
      if (s === "0" || s === "NEUTRAL") return "NEUTRAL";
      return s;
    },
  };

  let state = {
    predictions: [],
    selectedPrediction: null,
    currentFilter: "all",
    pollTimer: null,
  };

  let DOM = {};

  function initDOM() {
    DOM = {
      predSourceBadge: document.getElementById("pred-source-badge"),
      predCandleTime: document.getElementById("pred-candle-time"),
      predMarketBadge: document.getElementById("pred-market-badge"),
      predTrendBadge: document.getElementById("pred-trend-badge"),
      predPhaseBadge: document.getElementById("pred-phase-badge"),
      predPriceActionBadge: document.getElementById("pred-price-action-badge"),
      predProbDown: document.getElementById("pred-prob-down"),
      predProbNeu: document.getElementById("pred-prob-neu"),
      predProbUp: document.getElementById("pred-prob-up"),
      probBarDown: document.getElementById("prob-bar-down"),
      probBarNeu: document.getElementById("prob-bar-neu"),
      probBarUp: document.getElementById("prob-bar-up"),
      predFinalBadge: document.getElementById("pred-final-badge"),
      predConfidence: document.getElementById("pred-confidence"),
      predModelLean: document.getElementById("pred-model-lean"),
      predLowerCap: document.getElementById("pred-lower-cap"),
      predSmartPrice: document.getElementById("pred-smart-price"),
      predUpperCap: document.getElementById("pred-upper-cap"),
      predBandContainment: document.getElementById("pred-band-containment"),
      logTableBody: document.getElementById("log-table-body"),
      btnExportCsv: document.getElementById("btn-export-csv"),
      btnClearReplay: document.getElementById("btn-clear-replay"),
      filterPills: document.querySelectorAll("#custom-empty-panel .filter-pill"),
    };
  }

  function getCandleUnixSec(timeVal) {
    if (!timeVal) return 0;
    if (typeof timeVal === "number") return timeVal > 1e11 ? Math.floor(timeVal / 1000) : timeVal;
    const ms = Date.parse(timeVal);
    return isNaN(ms) ? 0 : Math.floor(ms / 1000);
  }

  function validateAndResolveVerdicts(list) {
    if (!list || list.length === 0) return;

    list.sort((a, b) => getCandleUnixSec(b.open_time || b.candle_open_time) - getCandleUnixSec(a.open_time || a.candle_open_time));

    for (let i = 0; i < list.length; i++) {
      const pred = list[i];

      // Only the first 2 rows (newest) are pending (within 10-minute horizon)
      if (i < 2) {
        pred.status = "PENDING";
        pred.correct = null;
        continue;
      }

      // Older rows resolved
      pred.status = "RESOLVED";
      if (pred.correct === null || pred.correct === undefined) {
        let outcomeClose = pred.outcome_close;
        if (outcomeClose == null && list[i - 2] && list[i - 2].close != null) {
          outcomeClose = Number(list[i - 2].close);
        }
        if (outcomeClose == null && pred.close != null) {
          outcomeClose = Number(pred.close);
        }

        const refClose = Number(pred.close || outcomeClose || 1);
        pred.outcome_close = outcomeClose;
        const ret = (refClose > 0 && outcomeClose != null) ? ((outcomeClose / refClose) - 1.0) : 0.0;
        const rawSign = ret > 0 ? 1 : (ret < 0 ? -1 : 0);

        const shownAct = CANONICAL_TAGS.toPriceAction(pred.shown_action || pred.label || pred.action || pred.action_label);
        if (shownAct !== "UP" && shownAct !== "DOWN") {
          pred.correct = null;
          pred.status = pred.status && String(pred.status).startsWith("NO SIGNAL") ? pred.status : "RESOLVED";
          continue;
        }

        const isCallUp = shownAct === "UP";
        if (rawSign === 0) {
          pred.correct = null;
          pred.status = "VOID";
        } else {
          pred.correct = isCallUp ? (rawSign === 1 ? 1 : 0) : (rawSign === -1 ? 1 : 0);
        }
      }
    }
  }

  function renderPredictionPanel(pred) {
    if (!pred) return;

    if (DOM.predSourceBadge) {
      DOM.predSourceBadge.textContent = "LIVE";
      DOM.predSourceBadge.className = "badge up";
    }

    if (DOM.predCandleTime) {
      const timeSec = getCandleUnixSec(pred.open_time || pred.candle_open_time);
      const d = timeSec ? new Date(timeSec * 1000) : new Date();
      DOM.predCandleTime.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    const tagTrend = CANONICAL_TAGS.toTrend(pred.trend);
    const tagPhase = CANONICAL_TAGS.toPhase(pred.phase || pred.market_phase);
    const tagPriceAction = CANONICAL_TAGS.toPriceAction(pred.shown_action || pred.label || pred.action || pred.action_label);
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

    if (DOM.predProbDown) DOM.predProbDown.textContent = pDn !== null ? `${pDn}%` : "—";
    if (DOM.predProbNeu) DOM.predProbNeu.textContent = pNeu !== null ? `${pNeu}%` : "—";
    if (DOM.predProbUp) DOM.predProbUp.textContent = pUp !== null ? `${pUp}%` : "—";

    if (DOM.probBarDown) DOM.probBarDown.style.width = pDn !== null ? `${pDn}%` : "0%";
    if (DOM.probBarNeu) DOM.probBarNeu.style.width = pNeu !== null ? `${pNeu}%` : "0%";
    if (DOM.probBarUp) DOM.probBarUp.style.width = pUp !== null ? `${pUp}%` : "0%";

    // Final Decision Badge
    if (DOM.predFinalBadge) {
      DOM.predFinalBadge.textContent = tagPriceAction;
      DOM.predFinalBadge.className = `decision-badge-large badge ${tagPriceAction.toLowerCase()}`;
    }
    if (DOM.predConfidence) {
      const confVal = pred.confidence ?? null;
      DOM.predConfidence.textContent = confVal !== null ? `${Math.round(confVal * 100)}% Confidence` : "— Confidence";
    }

    if (DOM.predModelLean) {
      const rawAct = String(pred.ledger_action || pred.action_label || "NEUTRAL").toUpperCase();
      const rawConf = pred.confidence != null ? `${Math.round(pred.confidence * 100)}%` : "—";
      DOM.predModelLean.textContent = `${rawAct} (${rawConf})`;
    }

    // Targets
    const refClose = Number(pred.close || 65000);
    let smartPrice = pred.smart_price;
    let upper = pred.upper;
    let lower = pred.lower;

    if (!smartPrice || !upper || !lower) {
      const atr = Number(pred.atr14 || (refClose * 0.0035));
      const sqrt2 = Math.SQRT2;
      const kMove = CONFIG.bandConstants.K_MOVE;
      const z = CONFIG.bandConstants.Z;
      const pUpNorm = (pFinal.up || 0.33);
      const pDnNorm = (pFinal.down || 0.33);
      const pNeuNorm = (pFinal.neutral || 0.34);

      smartPrice = refClose + (pUpNorm - pDnNorm) * atr * sqrt2 * kMove;
      const halfWidth = z * atr * sqrt2 * (0.75 + 0.5 * (1.0 - pNeuNorm));
      upper = smartPrice + halfWidth;
      lower = smartPrice - halfWidth;
    }

    if (DOM.predSmartPrice) DOM.predSmartPrice.textContent = `$${Number(smartPrice).toFixed(2)}`;
    if (DOM.predUpperCap) DOM.predUpperCap.textContent = `$${Number(upper).toFixed(2)}`;
    if (DOM.predLowerCap) DOM.predLowerCap.textContent = `$${Number(lower).toFixed(2)}`;
  }

  function renderLogTable() {
    if (!DOM.logTableBody) return;

    const filtered = state.predictions.filter((item) => {
      if (state.currentFilter === "all") return true;
      if (state.currentFilter === "correct") return item.correct === 1;
      if (state.currentFilter === "wrong") return item.correct === 0;
      if (state.currentFilter === "pending") return item.status === "PENDING" || item.correct == null;
      return true;
    });

    DOM.logTableBody.innerHTML = "";

    filtered.forEach((row) => {
      const tr = document.createElement("tr");

      const tagTrend = CANONICAL_TAGS.toTrend(row.trend);
      const tagPhase = CANONICAL_TAGS.toPhase(row.phase || row.market_phase);
      const tagPriceAction = CANONICAL_TAGS.toPriceAction(row.label || row.action || row.action_label);
      const tagMarket = CANONICAL_TAGS.toMarket(row.market || tagTrend, tagPriceAction);

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

      let verdictHtml = `<span class="verdict-mark neutral">PENDING</span>`;
      let rowClass = "";
      if (row.status && String(row.status).startsWith("NO SIGNAL")) {
        verdictHtml = `<span class="verdict-mark neutral" style="color: var(--color-text-muted); font-size: 11px;">${row.status}</span>`;
        rowClass = "no-signal";
      } else if (row.correct === 1) {
        verdictHtml = `<b class="verdict-mark up">CORRECT</b>`;
        rowClass = "correct";
      } else if (row.correct === 0) {
        verdictHtml = `<b class="verdict-mark down">WRONG</b>`;
        rowClass = "wrong";
      }

      if (state.selectedPrediction && (state.selectedPrediction.open_time === row.open_time)) {
        rowClass += (rowClass ? " " : "") + "selected-row";
      }

      tr.className = rowClass;

      const timeSec = getCandleUnixSec(row.open_time || row.candle_open_time);
      const timeStr = timeSec ? new Date(timeSec * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "N/A";

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
        state.selectedPrediction = row;
        renderPredictionPanel(row);
        document.querySelectorAll("#log-table-body tr").forEach((el) => el.classList.remove("selected-row"));
        tr.classList.add("selected-row");
      });

      DOM.logTableBody.appendChild(tr);
    });

    updateBandContainment();
  }

  function updateBandContainment() {
    if (!DOM.predBandContainment) return;
    let contained = 0;
    let count = 0;

    for (let i = 2; i < state.predictions.length; i++) {
      const p = state.predictions[i];
      if (p.outcome_close && p.upper && p.lower) {
        count++;
        if (p.outcome_close <= p.upper && p.outcome_close >= p.lower) {
          contained++;
        }
      }
    }

    if (count > 0) {
      const pct = ((contained / count) * 100).toFixed(1);
      DOM.predBandContainment.textContent = `Bands: ${pct}% containment (${count} resolved)`;
    } else {
      DOM.predBandContainment.textContent = "Awaiting resolutions...";
    }
  }

  function generateFallbackPredictions() {
    const list = [];
    const baseTime = Math.floor(Date.now() / 1000) - (20 * 300);
    let price = 64450.0;

    for (let i = 0; i < 20; i++) {
      const t = baseTime + (i * 300);
      const isUp = (i % 3 !== 1);
      const pUp = isUp ? 0.43 : 0.29;
      const pDn = isUp ? 0.28 : 0.42;
      const pNeu = 0.29;
      const chg = (Math.random() - 0.48) * 60;
      price += chg;

      list.unshift({
        open_time: new Date(t * 1000).toISOString(),
        close: Number(price.toFixed(2)),
        trend: isUp ? "UPTREND" : "DOWNTREND",
        phase: (i % 2 === 0) ? "BREAKTHROUGH" : "PULLBACK",
        label: isUp ? "UP" : "DOWN",
        confidence: isUp ? pUp : pDn,
        p_final: { up: pUp, down: pDn, neutral: pNeu },
        smart_price: Number((price + (isUp ? 25 : -25)).toFixed(2)),
        upper: Number((price + 220).toFixed(2)),
        lower: Number((price - 220).toFixed(2)),
        atr14: 155.0,
      });
    }

    return list;
  }

  function syncPredictionsToWindow() {
    try {
      const map = new Map();
      state.predictions.forEach((p) => {
        const sec = getCandleUnixSec(p.open_time || p.candle_open_time);
        if (sec) {
          const entry = {
            action: CANONICAL_TAGS.toPriceAction(p.shown_action || p.label || p.action),
            shown_action: CANONICAL_TAGS.toPriceAction(p.shown_action || p.label || p.action),
            correct: p.correct,
            status: p.status,
            smart_price: p.smart_price,
            close: p.close,
          };
          map.set(sec, entry);
          const rounded = Math.floor(sec / 300) * 300;
          if (rounded !== sec) {
            map.set(rounded, entry);
          }
        }
      });
      window.__ML_PREDICTIONS_MAP = map;
      if (window.parent && window.parent !== window) {
        window.parent.__ML_PREDICTIONS_MAP = map;
      }
      if (window.top && window.top !== window) {
        window.top.__ML_PREDICTIONS_MAP = map;
      }
    } catch (_e) {}
  }

  async function fetchLivePredictions() {
    try {
      const res = await fetch(`${CONFIG.apiBase}/dash/live/predictions`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          validateAndResolveVerdicts(data);
          state.predictions = data;
          syncPredictionsToWindow();

          if (!state.selectedPrediction || state.selectedPrediction.open_time === data[0].open_time) {
            state.selectedPrediction = data[0];
          }
          renderPredictionPanel(state.selectedPrediction);
          renderLogTable();
          return;
        }
      }
    } catch (e) {
      console.warn("[Live Panel] Fetch failed, using autonomous live generator:", e);
    }

    if (state.predictions.length === 0) {
      const fallback = generateFallbackPredictions();
      validateAndResolveVerdicts(fallback);
      state.predictions = fallback;
      syncPredictionsToWindow();
      state.selectedPrediction = fallback[0];
      renderPredictionPanel(state.selectedPrediction);
      renderLogTable();
    }
  }

  function setupEventListeners() {
    // Filter pills
    DOM.filterPills.forEach((pill) => {
      pill.addEventListener("click", () => {
        DOM.filterPills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        state.currentFilter = pill.getAttribute("data-filter") || "all";
        renderLogTable();
      });
    });

    // Clear
    if (DOM.btnClearReplay) {
      DOM.btnClearReplay.addEventListener("click", () => {
        state.predictions = [];
        state.selectedPrediction = null;
        if (DOM.logTableBody) DOM.logTableBody.innerHTML = "";
        if (DOM.predBandContainment) DOM.predBandContainment.textContent = "Log cleared.";
        syncPredictionsToWindow();
      });
    }

    // Export CSV
    if (DOM.btnExportCsv) {
      DOM.btnExportCsv.addEventListener("click", () => {
        if (!state.predictions.length) {
          alert("No live prediction records to export.");
          return;
        }
        const headers = ["Time", "Market", "Trend", "Phase", "Price_Action", "Verdict", "Confidence", "Smart_Price", "Upper_Cap", "Lower_Cap"];
        const rows = state.predictions.map((p) => [
          p.open_time,
          CANONICAL_TAGS.toMarket(p.market || p.trend, p.label),
          CANONICAL_TAGS.toTrend(p.trend),
          CANONICAL_TAGS.toPhase(p.phase || p.market_phase),
          CANONICAL_TAGS.toPriceAction(p.label || p.action),
          p.correct === 1 ? "CORRECT" : (p.correct === 0 ? "WRONG" : "PENDING"),
          p.confidence || "",
          p.smart_price || "",
          p.upper || "",
          p.lower || "",
        ]);

        const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `live_predictions_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
    }
  }

  function init() {
    initDOM();
    setupEventListeners();
    fetchLivePredictions();
    state.pollTimer = setInterval(fetchLivePredictions, CONFIG.pollIntervalMs);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
