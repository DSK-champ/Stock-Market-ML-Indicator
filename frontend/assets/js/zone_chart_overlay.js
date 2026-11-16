/**
 * Model-Informed Trade Zones Chart Overlay Primitive
 * Projects horizontal semi-transparent entry and stop zones into future bars (t0 -> t+2)
 * matching TradingView-style support/resistance projections.
 * Strictly active ONLY when the Trade Zones panel is open and "Show zones on chart" is ON.
 */

(function (window) {
  "use strict";

  class ZoneChartOverlay {
    constructor() {
      this.zones = [];
      this.activeZone = null;
      this.panelOpen = false;
      this.charts = new Set();
      this._boundRender = this.render.bind(this);
    }

    setPanelOpen(isOpen) {
      this.panelOpen = !!isOpen;
      this.charts.forEach((c) => { if (c) c.render(); });
    }

    setZones(zones, activeZone) {
      this.zones = zones || [];
      if (activeZone) this.activeZone = activeZone;
      this.charts.forEach((c) => { if (c) c.render(); });
    }

    attachToChart(chart) {
      if (!chart) return;
      this.charts.add(chart);
      if (!chart.overlayPlugins) chart.overlayPlugins = [];
      if (!chart.overlayPlugins.includes(this._boundRender)) {
        chart.overlayPlugins.push(this._boundRender);
      }
    }

    detachFromChart(chart) {
      if (!chart || !chart.overlayPlugins) return;
      this.charts.delete(chart);
      const idx = chart.overlayPlugins.indexOf(this._boundRender);
      if (idx !== -1) chart.overlayPlugins.splice(idx, 1);
      chart.render();
    }

    render(ctx, renderState) {
      const cfg = window.ZONE_CONFIG || {};
      if (cfg.show_on_chart === false) {
        return; // Switched off -> draw nothing, 0 overhead
      }
      // NativeChart natively renders trade zones in drawTradeZones; return early to guarantee zero duplicate zones!
      if (renderState && (renderState.nativeRenderedTradeZones || renderState.candles)) {
        return;
      }

      const {
        intStart,
        endIndex,
        getX,
        getY,
        candleSlotWidth,
        candleWidth,
        chartWidth,
        chartHeight,
        candles,
      } = renderState;

      if (!candles || candles.length === 0) return;

      const nCandles = candles.length;
      const timeToIdx = new Map();
      candles.forEach((c, idx) => {
        if (c && c.time != null) {
          const tSec = typeof c.time === "number" ? (c.time > 1e11 ? Math.floor(c.time / 1000) : c.time) : 0;
          if (tSec) timeToIdx.set(tSec, idx);
        }
      });

      const toSec = (ts) => {
        if (!ts) return 0;
        if (typeof ts === "number") return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts);
        let s = String(ts).trim().replace(" ", "T");
        if (!s.endsWith("Z") && !s.includes("+")) s += "Z";
        const ms = Date.parse(s);
        return isNaN(ms) ? 0 : Math.floor(ms / 1000);
      };

      // Determine which zones to draw
      const selectedTimes = window.ZONE_SELECTED_TIMES;
      let zonesToDraw = [];
      if (selectedTimes && selectedTimes.size > 0) {
        this.zones.forEach((z) => {
          const zSec = toSec(z.candle_open_time);
          if (selectedTimes.has(zSec)) zonesToDraw.push(z);
        });
        if (this.activeZone) {
          const aSec = toSec(this.activeZone.candle_open_time);
          if (selectedTimes.has(aSec) && !zonesToDraw.some((zd) => zd.zone_id === this.activeZone.zone_id)) {
            zonesToDraw.push(this.activeZone);
          }
        }
      } else if (!window.ZONE_USER_INTERACTED && this.activeZone) {
        zonesToDraw = [this.activeZone];
      } else if (cfg.active_only) {
        if (this.activeZone) zonesToDraw = [this.activeZone];
      } else {
        zonesToDraw = this.zones.slice(-15);
        if (this.activeZone && !zonesToDraw.some((z) => z.zone_id === this.activeZone.zone_id)) {
          zonesToDraw.push(this.activeZone);
        }
      }

      ctx.save();

      zonesToDraw.forEach((zone) => {
        if (!zone) return;

        const openSec = toSec(zone.candle_open_time);
        let candIdx = timeToIdx.get(openSec);

        // Fallback matching if exact second not present
        if (candIdx == null) {
          for (let i = nCandles - 1; i >= 0; i--) {
            const cSec = toSec(candles[i].open_time || candles[i].time);
            if (cSec && Math.abs(cSec - openSec) < 300) {
              candIdx = i;
              break;
            }
          }
        }

        if (candIdx == null) return;

        const isActive = zone.status === "ACTIVE";
        const isLong = zone.direction === "LONG";
        const isShort = zone.direction === "SHORT";
        const isNeutral = zone.direction === "NEUTRAL";

        // Geometry: start from signal candle, extend across to expiry (t+2) and into future bars!
        const startX = getX(candIdx) - candleWidth * 0.5;
        const targetIdx = candIdx + 2;

        let endX = 0;
        if (targetIdx < nCandles) {
          endX = getX(targetIdx) + candleWidth * 0.5;
        } else {
          endX = getX(candIdx) + (targetIdx - candIdx) * candleSlotWidth + candleWidth * 0.5;
        }

        // If zone is active/live, project further into future space so trader sees it ahead of time!
        if (isActive) {
          const futureX = getX(nCandles - 1) + 2.5 * candleSlotWidth;
          endX = Math.max(endX, futureX);
        }

        const width = Math.max(20, endX - startX);

        // Opacity adjustments: dim zone if low edge or conflicting signals
        let baseOpacity = 0.20;
        if (zone.is_low_edge) baseOpacity = 0.12;
        if (zone.has_conflicts) baseOpacity *= 0.65;

        // Recoloring finished zones by outcome
        let statusTint = null;
        if (zone.status === "TARGET_HIT") statusTint = "rgba(0, 230, 118, ";
        else if (zone.status === "STOPPED_OUT") statusTint = "rgba(255, 23, 68, ";
        else if (zone.status === "FLIPPED_EXIT") statusTint = "rgba(255, 152, 0, ";
        else if (zone.status === "EXPIRED") statusTint = "rgba(120, 123, 134, ";

        // 1. NEUTRAL "No Trade" Band
        if (isNeutral) {
          const yMid = getY(zone.ref_close);
          const bandHeight = Math.max(6, (zone.ref_close * 0.0023 / (renderState.maxPrice - renderState.minPrice)) * chartHeight);
          ctx.fillStyle = "rgba(120, 123, 134, 0.10)";
          ctx.fillRect(startX, yMid - bandHeight * 0.5, width, bandHeight);

          ctx.strokeStyle = "rgba(120, 123, 134, 0.4)";
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(startX, yMid);
          ctx.lineTo(startX + width, yMid);
          ctx.stroke();
          ctx.setLineDash([]);

          if (isActive) {
            ctx.fillStyle = "#787b86";
            ctx.font = "bold 9px sans-serif";
            ctx.textAlign = "left";
            ctx.fillText("⚪ NO TRADE (23 bps hurdle)", startX + 4, yMid - 4);
          }
          return;
        }

        // 2. ENTRY ZONE (Green for Long, Red for Short)
        const yEntryMin = getY(zone.entry_min);
        const yEntryMax = getY(zone.entry_max);
        const entryTopY = Math.min(yEntryMin, yEntryMax);
        const entryHeight = Math.max(3, Math.abs(yEntryMin - yEntryMax));

        const entryColor = statusTint ? `${statusTint}${baseOpacity})` : (isLong ? `rgba(0, 230, 118, ${baseOpacity})` : `rgba(255, 23, 68, ${baseOpacity})`);
        ctx.fillStyle = entryColor;
        ctx.fillRect(startX, entryTopY, width, entryHeight);

        // Entry Border
        ctx.strokeStyle = isLong ? "rgba(0, 230, 118, 0.5)" : "rgba(255, 23, 68, 0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(startX, entryTopY, width, entryHeight);

        // 3. STOP LOSS / BACK-OUT ZONE (Red)
        const yStop = getY(zone.stop_price);
        let stopTopY = 0;
        let stopHeight = 0;

        if (isLong) {
          // Stop is below entry_min
          stopTopY = Math.min(yEntryMin, yStop);
          stopHeight = Math.max(3, Math.abs(yEntryMin - yStop));
        } else {
          // Stop is above entry_max
          stopTopY = Math.min(yEntryMax, yStop);
          stopHeight = Math.max(3, Math.abs(yEntryMax - yStop));
        }

        const stopColor = statusTint ? `${statusTint}${baseOpacity * 0.75})` : `rgba(255, 23, 68, ${baseOpacity * 0.75})`;
        ctx.fillStyle = stopColor;
        ctx.fillRect(startX, stopTopY, width, stopHeight);

        ctx.strokeStyle = "rgba(255, 23, 68, 0.4)";
        ctx.setLineDash([2, 2]);
        ctx.strokeRect(startX, stopTopY, width, stopHeight);
        ctx.setLineDash([]);

        // 4. TARGET LINE (Dashed Green)
        const yTarget = getY(zone.target_price);
        ctx.strokeStyle = isLong ? "#00e676" : "#ff1744";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(startX, yTarget);
        ctx.lineTo(startX + width, yTarget);
        ctx.stroke();
        ctx.setLineDash([]);

        // 5. STOP LINE (Dashed Red)
        ctx.strokeStyle = "#ff1744";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(startX, yStop);
        ctx.lineTo(startX + width, yStop);
        ctx.stroke();
        ctx.setLineDash([]);

        // 6. Labels & Outcome Tags on the right side of the zone
        ctx.font = "bold 9px monospace, sans-serif";
        ctx.textAlign = "left";

        const tagX = startX + width - 95;
        if (tagX > startX) {
          // Target Badge
          ctx.fillStyle = "rgba(19, 23, 34, 0.85)";
          ctx.fillRect(tagX - 2, yTarget - 12, 95, 12);
          ctx.fillStyle = isLong ? "#00e676" : "#ff1744";
          ctx.fillText(`🎯 TARGET $${zone.target_price}`, tagX, yTarget - 3);

          // Stop Badge
          ctx.fillStyle = "rgba(19, 23, 34, 0.85)";
          ctx.fillRect(tagX - 2, yStop - 12, 95, 12);
          ctx.fillStyle = "#ff1744";
          ctx.fillText(`🛑 STOP $${zone.stop_price}`, tagX, yStop - 3);

          // Entry Mid Label
          const yMid = (entryTopY + entryTopY + entryHeight) / 2;
          ctx.fillStyle = "rgba(19, 23, 34, 0.85)";
          ctx.fillRect(tagX - 2, yMid - 6, 95, 12);
          ctx.fillStyle = "#d1d4dc";
          ctx.fillText(`🟢 ENTRY $${zone.entry_mid}`, tagX, yMid + 3);
        }

        // 7. Outcome Tag for resolved zones: strictly PROFIT, LOSS, FLIPPED, or EXPIRED
        if (!isActive && statusTint) {
          const outcomeText = zone.status === "TARGET_HIT" ? "PROFIT" :
            zone.status === "STOPPED_OUT" ? "LOSS" :
            zone.status === "FLIPPED_EXIT" ? "FLIPPED EXIT" : "EXPIRED";

          ctx.fillStyle = "rgba(19, 23, 34, 0.9)";
          ctx.fillRect(startX + 4, entryTopY - 14, 75, 13);
          ctx.fillStyle = zone.status === "TARGET_HIT" ? "#00e676" : (zone.status === "STOPPED_OUT" ? "#ff1744" : "#ff9800");
          ctx.font = "bold 9px sans-serif";
          ctx.fillText(outcomeText, startX + 7, entryTopY - 4);
        }

        // 8. Direction Marker on Signal Candle
        const c = candles[candIdx];
        if (c) {
          ctx.font = "bold 9px sans-serif";
          ctx.textAlign = "center";
          const markerX = getX(candIdx);

          if (isLong) {
            const markerY = getY(c.low) + 14;
            ctx.fillStyle = "#00e676";
            ctx.beginPath();
            ctx.moveTo(markerX, markerY - 5);
            ctx.lineTo(markerX - 4, markerY + 3);
            ctx.lineTo(markerX + 4, markerY + 3);
            ctx.closePath();
            ctx.fill();
            ctx.fillText("ENTRY", markerX, markerY + 12);
          } else if (isShort) {
            const markerY = getY(c.high) - 14;
            ctx.fillStyle = "#ff1744";
            ctx.beginPath();
            ctx.moveTo(markerX, markerY + 5);
            ctx.lineTo(markerX - 4, markerY - 3);
            ctx.lineTo(markerX + 4, markerY - 3);
            ctx.closePath();
            ctx.fill();
            ctx.fillText("EXIT", markerX, markerY - 7);
          }
        }
      });

      ctx.restore();
    }
  }

  window.ZoneChartOverlay = ZoneChartOverlay;
})(window);
