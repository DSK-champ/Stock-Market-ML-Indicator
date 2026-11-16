/**
 * Trade Zones UI Panel Controller
 * Manages open/close state in memory only, binds WebSocket events,
 * and renders Active Signal Card, Levels, Model Evidence Chips,
 * Zone History Table with running hit rate, Switches, and Multipliers.
 */

(function (window) {
  "use strict";

  window.ZONE_SELECTED_TIMES = window.ZONE_SELECTED_TIMES || new Set();
  window.ZONE_USER_INTERACTED = false;

  class TradeZonesPanel {
    constructor() {
      this.isOpen = false;
      this.activeZone = null;
      this.historyZones = [];
      this.overlay = new window.ZoneChartOverlay();
      this.timerInterval = null;

      this.dom = {
        btnToggle: null,
        panel: null,
        btnClose: null,
        // Active Signal Card
        signalBadge: null,
        confidenceVal: null,
        agreementVal: null,
        agreementFill: null,
        regimePill: null,
        expiryCountdown: null,
        // Levels
        entryRange: null,
        targetPrice: null,
        stopPrice: null,
        riskReward: null,
        expectedNetBps: null,
        // Chips
        chipsContainer: null,
        // History
        historyBody: null,
        runningHitRate: null,
        runningNetBps: null,
        // Switches
        switchShowChart: null,
        switchActiveOnly: null,
        // Inputs
        inputKEntry: null,
        inputKStop: null,
        inputScale: null,
        inputBuffer: null,
      };
    }

    init() {
      this.cacheDom();
      this.bindEvents();
      this.startExpiryTimer();

      // Attach overlay & sync layers to window.CHARTS
      const syncCharts = () => {
        if (window.CHARTS) {
          if (window.CHARTS.live) {
            this.overlay.attachToChart(window.CHARTS.live);
            window.CHARTS.live.layers.tradeZones = (window.ZONE_CONFIG.show_on_chart !== false);
          }
          if (window.CHARTS.replay) {
            this.overlay.attachToChart(window.CHARTS.replay);
            window.CHARTS.replay.layers.tradeZones = (window.ZONE_CONFIG.show_on_chart !== false);
          }
        }
      };
      syncCharts();
      setInterval(syncCharts, 1000);

      // Populate and continuously synchronize zones from window.STATE
      const syncStateZones = () => {
        if (window.STATE && window.ZoneCalculator) {
          const isLive = (window.STATE.activeTab || "live") === "live";
          const preds = isLive ? window.STATE.livePredictions : window.STATE.replayPredictions;
          const candles = isLive ? window.STATE.liveCandles : window.STATE.replayCandles;

          if (preds && preds.length > 0) {
            const latestPred = preds[0];
            const z = window.ZoneCalculator.calculateTradeZone(latestPred, candles);
            this.activeZone = z;
            this.renderAll();
          }
        }
      };
      syncStateZones();
      setInterval(syncStateZones, 1000);

      // Initial fetch of active and historical zones via API
      this.fetchInitialZones();
    }

    cacheDom() {
      this.dom.btnToggle = document.getElementById("btn-trade-zones");
      this.dom.panel = document.getElementById("trade-zones-panel");
      this.dom.btnClose = document.getElementById("btn-close-trade-zones");

      this.dom.signalBadge = document.getElementById("tz-signal-badge");
      this.dom.confidenceVal = document.getElementById("tz-confidence-val");
      this.dom.agreementVal = document.getElementById("tz-agreement-val");
      this.dom.agreementFill = document.getElementById("tz-agreement-fill");
      this.dom.regimePill = document.getElementById("tz-regime-pill");
      this.dom.expiryCountdown = document.getElementById("tz-expiry-countdown");

      this.dom.entryRange = document.getElementById("tz-level-entry");
      this.dom.targetPrice = document.getElementById("tz-level-target");
      this.dom.stopPrice = document.getElementById("tz-level-stop");
      this.dom.riskReward = document.getElementById("tz-level-rr");
      this.dom.expectedNetBps = document.getElementById("tz-level-net-bps");

      this.dom.chipsContainer = document.getElementById("tz-chips-container");

      this.dom.historyBody = document.getElementById("tz-history-body");
      this.dom.runningHitRate = document.getElementById("tz-running-hit-rate");
      this.dom.runningNetBps = document.getElementById("tz-running-net-bps");

      this.dom.switchShowChart = document.getElementById("tz-switch-show-chart");
      this.dom.switchActiveOnly = document.getElementById("tz-switch-active-only");

      this.dom.inputKEntry = document.getElementById("tz-input-k-entry");
      this.dom.inputKStop = document.getElementById("tz-input-k-stop");
      this.dom.inputScale = document.getElementById("tz-input-scale");
      this.dom.inputBuffer = document.getElementById("tz-input-buffer");
    }

    bindEvents() {
      if (this.dom.btnToggle) {
        this.dom.btnToggle.addEventListener("click", () => this.togglePanel());
      }
      if (this.dom.btnClose) {
        this.dom.btnClose.addEventListener("click", () => this.setPanelOpen(false));
      }

      if (this.dom.switchShowChart) {
        this.dom.switchShowChart.addEventListener("change", (e) => {
          window.ZONE_CONFIG.show_on_chart = e.target.checked;
          if (window.CHARTS) {
            if (window.CHARTS.live) {
              window.CHARTS.live.layers.tradeZones = e.target.checked;
              window.CHARTS.live.render();
            }
            if (window.CHARTS.replay) {
              window.CHARTS.replay.layers.tradeZones = e.target.checked;
              window.CHARTS.replay.render();
            }
          }
        });
      }

      if (this.dom.switchActiveOnly) {
        this.dom.switchActiveOnly.addEventListener("change", (e) => {
          window.ZONE_CONFIG.active_only = e.target.checked;
          if (window.CHARTS) {
            if (window.CHARTS.live) window.CHARTS.live.render();
            if (window.CHARTS.replay) window.CHARTS.replay.render();
          }
        });
      }

      const updateMultipliers = () => {
        if (this.dom.inputKEntry) window.ZONE_CONFIG.k_entry = parseFloat(this.dom.inputKEntry.value) || 0.75;
        if (this.dom.inputKStop) window.ZONE_CONFIG.k_stop = parseFloat(this.dom.inputKStop.value) || 1.25;
        if (this.dom.inputScale) window.ZONE_CONFIG.scale = parseFloat(this.dom.inputScale.value) || 1.50;
        if (this.dom.inputBuffer) window.ZONE_CONFIG.buffer_bps = parseFloat(this.dom.inputBuffer.value) || 5.0;

        // Recompute active zone if available
        if (this.activeZone && window.ZoneCalculator) {
          const isLive = (window.STATE && window.STATE.activeTab === "live");
          const candles = (window.CHARTS && window.CHARTS[isLive ? "live" : "replay"]) ? window.CHARTS[isLive ? "live" : "replay"].candles : [];
          this.activeZone = window.ZoneCalculator.calculateTradeZone(this.activeZone, candles);
          this.renderActiveCard(this.activeZone);
          this.overlay.setZones(this.historyZones, this.activeZone);
        }
        if (window.CHARTS) {
          if (window.CHARTS.live) window.CHARTS.live.render();
          if (window.CHARTS.replay) window.CHARTS.replay.render();
        }
      };

      [this.dom.inputKEntry, this.dom.inputKStop, this.dom.inputScale, this.dom.inputBuffer].forEach((inp) => {
        if (inp) inp.addEventListener("input", updateMultipliers);
      });
    }

    togglePanel() {
      this.setPanelOpen(!this.isOpen);
    }

    setPanelOpen(open) {
      this.isOpen = !!open;

      const mainRow = document.querySelector(".main-row") || document.querySelector("main");
      if (mainRow) {
        if (this.isOpen) {
          mainRow.classList.add("trade-zones-open");
        } else {
          mainRow.classList.remove("trade-zones-open");
        }
      }

      if (this.dom.panel) {
        this.dom.panel.hidden = !this.isOpen;
      }
      if (this.dom.btnToggle) {
        if (this.isOpen) {
          this.dom.btnToggle.classList.add("active");
          this.dom.btnToggle.setAttribute("aria-expanded", "true");
        } else {
          this.dom.btnToggle.classList.remove("active");
          this.dom.btnToggle.setAttribute("aria-expanded", "false");
        }
      }

      // Notify chart overlay of open state
      this.overlay.setPanelOpen(this.isOpen);

      if (window.CHARTS) {
        if (window.CHARTS.live) {
          window.CHARTS.live.layers.tradeZones = (window.ZONE_CONFIG.show_on_chart !== false);
          window.CHARTS.live.resize();
        }
        if (window.CHARTS.replay) {
          window.CHARTS.replay.layers.tradeZones = (window.ZONE_CONFIG.show_on_chart !== false);
          window.CHARTS.replay.resize();
        }
      }

      // Trigger resize event so charts adapt smoothly
      window.dispatchEvent(new Event("resize"));
    }

    async fetchInitialZones() {
      try {
        const res = await fetch("http://localhost:8001/dash/trade_zones/history?limit=30");
        if (res.ok) {
          const data = await res.json();
          if (data && data.zones) {
            this.historyZones = data.zones;
            const active = data.zones.find((z) => z.status === "ACTIVE") || data.zones[0];
            if (active) this.activeZone = active;
            this.renderAll();
          }
        }
      } catch (err) {
        // Fallback: try direct backend API port 8000
        try {
          const res2 = await fetch("http://localhost:8000/api/live/trade_zones/history?limit=30");
          if (res2.ok) {
            const data2 = await res2.json();
            if (data2 && data2.zones) {
              this.historyZones = data2.zones;
              const active = data2.zones.find((z) => z.status === "ACTIVE") || data2.zones[0];
              if (active) this.activeZone = active;
              this.renderAll();
            }
          }
        } catch (e2) {
          // In offline/replay, zones are calculated directly on candles
        }
      }
    }

    onNewZone(zone) {
      if (!zone) return;
      this.activeZone = zone;

      // Update history list
      const existingIdx = this.historyZones.findIndex((z) => z.zone_id === zone.zone_id);
      if (existingIdx !== -1) {
        this.historyZones[existingIdx] = zone;
      } else {
        this.historyZones.unshift(zone);
        if (this.historyZones.length > 50) this.historyZones.pop();
      }

      this.renderAll();
    }

    onZoneResolved(resolvedZone) {
      if (!resolvedZone) return;
      const idx = this.historyZones.findIndex((z) => z.zone_id === resolvedZone.zone_id);
      if (idx !== -1) {
        this.historyZones[idx] = resolvedZone;
      } else {
        this.historyZones.unshift(resolvedZone);
      }
      if (this.activeZone && this.activeZone.zone_id === resolvedZone.zone_id) {
        this.activeZone = resolvedZone;
      }
      this.renderAll();
    }

    renderAll() {
      if (this.activeZone) {
        this.renderActiveCard(this.activeZone);
      }
      this.renderHistoryTable();
      this.overlay.setZones(this.historyZones, this.activeZone);
    }

    renderActiveCard(zone) {
      const isLong = zone.direction === "LONG";
      const isShort = zone.direction === "SHORT";
      const isNeu = zone.direction === "NEUTRAL";

      // 1. Signal Badge
      if (this.dom.signalBadge) {
        this.dom.signalBadge.textContent = zone.direction;
        this.dom.signalBadge.className = `tz-badge-large ${zone.direction.toLowerCase()}`;
      }

      // 2. Confidence
      if (this.dom.confidenceVal) {
        this.dom.confidenceVal.textContent = `${Math.round(zone.confidence * 100)}% Confidence`;
      }

      // 3. Agreement Score
      if (this.dom.agreementVal) {
        this.dom.agreementVal.textContent = `${zone.agreement_score}%`;
      }
      if (this.dom.agreementFill) {
        this.dom.agreementFill.style.width = `${zone.agreement_score}%`;
        if (zone.has_conflicts) {
          this.dom.agreementFill.classList.add("conflicts");
        } else {
          this.dom.agreementFill.classList.remove("conflicts");
        }
      }

      // 4. Regime Pill
      if (this.dom.regimePill) {
        this.dom.regimePill.textContent = zone.regime;
        this.dom.regimePill.className = `tz-regime-pill ${zone.regime}`;
      }

      // 5. Levels
      if (this.dom.entryRange) {
        this.dom.entryRange.textContent = isNeu ? "No Entry" : `$${zone.entry_min.toLocaleString()} – $${zone.entry_max.toLocaleString()}`;
      }
      if (this.dom.targetPrice) {
        this.dom.targetPrice.textContent = isNeu ? "—" : `$${zone.target_price.toLocaleString()}`;
      }
      if (this.dom.stopPrice) {
        this.dom.stopPrice.textContent = isNeu ? "—" : `$${zone.stop_price.toLocaleString()}`;
      }
      if (this.dom.riskReward) {
        this.dom.riskReward.textContent = isNeu ? "—" : `1 : ${zone.risk_reward}`;
      }
      if (this.dom.expectedNetBps) {
        this.dom.expectedNetBps.textContent = "2 Candles (10m)";
        this.dom.expectedNetBps.style.color = "#2962ff";
      }

      // 6. Evidence Chips (12 chips)
      if (this.dom.chipsContainer && zone.evidence_chips) {
        this.dom.chipsContainer.innerHTML = "";
        zone.evidence_chips.forEach((chip) => {
          const div = document.createElement("div");
          const stateClass = chip.bullish ? "bullish" : (chip.bearish ? "bearish" : "neutral");
          div.className = `tz-chip ${stateClass}`;
          div.innerHTML = `
            <span class="tz-chip-name">${chip.name}</span>
            <span class="tz-chip-val">${chip.value}</span>
          `;
          this.dom.chipsContainer.appendChild(div);
        });
      }
    }

    renderHistoryTable() {
      if (!this.dom.historyBody) return;

      const isLive = (!window.STATE || (window.STATE.activeTab || "live") === "live");
      const rawPreds = isLive ? (window.STATE?.livePredictions || []) : (window.STATE?.replayPredictions || []);
      const candles = isLive ? (window.STATE?.liveCandles || []) : (window.STATE?.replayCandles || []);

      if (typeof window.validateAndResolveVerdicts === "function") {
        try {
          window.validateAndResolveVerdicts(rawPreds, candles);
        } catch (e) {
          // ignore
        }
      }

      const toSec = (ts) => {
        if (!ts) return 0;
        if (typeof ts === "number") return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts);
        let s = String(ts).trim().replace(" ", "T");
        if (!s.endsWith("Z") && !s.includes("+")) s += "Z";
        const ms = Date.parse(s);
        return isNaN(ms) ? 0 : Math.floor(ms / 1000);
      };

      // STRICT DEDUPLICATION: Ensure no predictions with duplicate candle times exist
      const seenTimes = new Set();
      const preds = [];
      for (let i = 0; i < rawPreds.length; i++) {
        const p = rawPreds[i];
        if (!p) continue;
        const sec = toSec(p.open_time || p.candle_open_time);
        if (sec && !seenTimes.has(sec)) {
          seenTimes.add(sec);
          preds.push(p);
        }
      }

      // Running Stats: Strictly Wins, Losses, and Win Rate (NO net profit/loss, NO bps!)
      let wins = 0;
      let losses = 0;
      preds.forEach((p) => {
        if (p.correct === 1) wins++;
        else if (p.correct === 0) losses++;
      });
      const totalResolved = wins + losses;
      const winRate = totalResolved > 0 ? Math.round((wins / totalResolved) * 100) : 0;

      if (this.dom.runningHitRate) {
        this.dom.runningHitRate.textContent = totalResolved > 0 ? `${winRate}% Win Rate` : "—%";
      }
      if (this.dom.runningNetBps) {
        this.dom.runningNetBps.textContent = `Wins: ${wins} | Losses: ${losses}`;
      }

      this.dom.historyBody.innerHTML = "";

      const activePred = preds[0] || null;
      const activeSec = activePred ? toSec(activePred.open_time || activePred.candle_open_time) : 0;

      const rowsToShow = preds.slice(0, 30);
      rowsToShow.forEach((row) => {
        const tr = document.createElement("tr");

        const rowSec = toSec(row.open_time || row.candle_open_time);
        const isLatestCandle = (rowSec === activeSec);

        // Dynamically show the latest candle's zone by default!
        // Historical candles show ONLY if specifically checked in window.ZONE_SELECTED_TIMES
        let isChecked = false;
        if (isLatestCandle) {
          isChecked = !window.ZONE_ACTIVE_DISABLED;
        } else {
          isChecked = !!(window.ZONE_SELECTED_TIMES && window.ZONE_SELECTED_TIMES.has(rowSec));
        }

        // Exact match with main prediction log timestamp (e.g. "05:15 PM")
        let timeStr = "--:--";
        if (row.open_time) {
          try {
            timeStr = new Date(row.open_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
          } catch (e) {
            timeStr = String(row.open_time);
          }
        }

        // Canonical direction mapping
        let rawAction = "";
        if (window.CANONICAL_TAGS && window.CANONICAL_TAGS.toPriceAction) {
          rawAction = window.CANONICAL_TAGS.toPriceAction(row.shown_action || row.label || row.action || row.action_label);
        } else {
          rawAction = String(row.shown_action || row.label || row.action || "NEUTRAL").toUpperCase();
        }

        let dir = "NEUTRAL";
        let dirColor = "#787b86";
        if (rawAction === "UP" || rawAction === "LONG" || rawAction === "1") {
          dir = "LONG";
          dirColor = "#00e676";
        } else if (rawAction === "DOWN" || rawAction === "SHORT" || rawAction === "-1") {
          dir = "SHORT";
          dirColor = "#ff1744";
        }

        // Entry Price
        const entryPrice = Math.round(row.close || row.ref_close || 0);
        const entryStr = entryPrice > 0 ? `$${entryPrice.toLocaleString()}` : "—";

        // Verdict Badge: Strictly PROFIT, LOSS, ACTIVE, NO SIGNAL, or — (No bps!)
        let verdictBadge = "";
        if (row.status && String(row.status).startsWith("NO SIGNAL")) {
          verdictBadge = `<span class="tz-outcome-badge neutral">NO SIGNAL</span>`;
        } else if (row.correct === 1) {
          verdictBadge = `<span class="tz-outcome-badge target-hit">PROFIT</span>`;
        } else if (row.correct === 0) {
          verdictBadge = `<span class="tz-outcome-badge stopped-out">LOSS</span>`;
        } else if (dir === "NEUTRAL") {
          verdictBadge = `<span class="tz-outcome-badge neutral">—</span>`;
        } else {
          verdictBadge = `<span class="tz-outcome-badge active">ACTIVE</span>`;
        }

        tr.innerHTML = `
          <td style="text-align: center;">
            <input type="checkbox" class="tz-zone-toggle" data-sec="${rowSec}" ${isLatestCandle ? 'data-latest="true"' : ""} ${isChecked ? "checked" : ""}>
          </td>
          <td class="tabular-nums">${timeStr}</td>
          <td style="color: ${dirColor}; font-weight: 700;">${dir}</td>
          <td class="tabular-nums">${entryStr}</td>
          <td style="text-align: right;">${verdictBadge}</td>
        `;

        // Row click to toggle zone
        tr.addEventListener("click", (e) => {
          if (e.target.tagName && e.target.tagName.toLowerCase() === "input") return;
          const chk = tr.querySelector(".tz-zone-toggle");
          if (chk) {
            chk.checked = !chk.checked;
            chk.dispatchEvent(new Event("change"));
          }
        });

        this.dom.historyBody.appendChild(tr);
      });

      // Bind checkbox change events to toggle zones on chart
      this.dom.historyBody.querySelectorAll(".tz-zone-toggle").forEach((chk) => {
        chk.addEventListener("change", (e) => {
          const sec = Number(e.target.getAttribute("data-sec"));
          const isLatest = e.target.getAttribute("data-latest") === "true";
          if (!window.ZONE_SELECTED_TIMES) window.ZONE_SELECTED_TIMES = new Set();

          if (isLatest) {
            window.ZONE_ACTIVE_DISABLED = !e.target.checked;
          } else {
            if (e.target.checked) {
              window.ZONE_SELECTED_TIMES.add(sec);
            } else {
              window.ZONE_SELECTED_TIMES.delete(sec);
            }
          }
          if (window.CHARTS) {
            if (window.CHARTS.live) window.CHARTS.live.render();
            if (window.CHARTS.replay) window.CHARTS.replay.render();
          }
        });
      });
    }

    startExpiryTimer() {
      if (this.timerInterval) clearInterval(this.timerInterval);
      this.timerInterval = setInterval(() => {
        if (!this.activeZone || !this.dom.expiryCountdown) return;

        const targetTimeStr = this.activeZone.target_candle_open_time;
        if (!targetTimeStr) {
          this.dom.expiryCountdown.textContent = "2 candles (10m)";
          return;
        }

        const targetMs = Date.parse(targetTimeStr.replace(" ", "T"));
        const nowMs = Date.now();
        const diffSec = Math.max(0, Math.floor((targetMs - nowMs) / 1000));

        if (diffSec <= 0) {
          this.dom.expiryCountdown.textContent = "Resolving at close...";
        } else {
          const m = Math.floor(diffSec / 60);
          const s = diffSec % 60;
          this.dom.expiryCountdown.textContent = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")} to expiry`;
        }
      }, 1000);
    }
  }

  window.TradeZonesPanel = new TradeZonesPanel();

  // Auto-init on DOMContentLoaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => window.TradeZonesPanel.init());
  } else {
    window.TradeZonesPanel.init();
  }
})(window);
