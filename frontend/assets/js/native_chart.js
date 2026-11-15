/**
 * Native Canvas Candlestick Chart Engine
 * Pure HTML5 Canvas2D implementation for BTC/USDT 5m AI Indicator Dashboard.
 * 
 * Free of any TradingView libraries, logos, or watermarks.
 * 
 * Features:
 * - TradingView-identical cursor-anchored zoom (keeps cursor candle stationary)
 * - Time-axis horizontal drag to zoom in/out
 * - Price-axis vertical drag to scale/compress price range (double-click to auto-fit)
 * - Full-chart continuous Adaptive Indicator Channel across ALL historical candles
 * - Translucent channel fill, upper/lower bounds, and baseline
 * - Forward-projected 10-minute (2-candle) predictive rays and target levels
 * - High-DPI crisp rendering, crosshair with price/time badges, volume histogram
 */

class NativeCandleChart {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container #${containerId} not found`);
    }

    this.options = {
      upColor: options.upColor || "#26a69a",
      downColor: options.downColor || "#ef5350",
      bgColor: options.bgColor || "#131722",
      gridColor: options.gridColor || "rgba(42, 46, 57, 0.4)",
      textColor: options.textColor || "#787b86",
      accentColor: options.accentColor || "#2962ff",
      priceAxisWidth: 70,
      timeAxisHeight: 26,
      volumeHeightRatio: 0.18,
      minCandlesVisible: 15,
      maxCandlesVisible: 1500,
      ...options,
    };

    // Canvas setup
    this.canvas = document.createElement("canvas");
    this.canvas.className = "native-chart-canvas";
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.canvas.style.display = "block";
    this.canvas.style.backgroundColor = this.options.bgColor;
    this.container.innerHTML = "";
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");

    // Data stores
    this.candles = []; // [{ time, open_time, open, high, low, close, volume }]
    this.markers = []; // [{ time, shape: 'arrowUp'|'arrowDown'|'circle', color, text }]
    this.predictions = []; // All historical and active predictions
    this.indicatorBands = []; // Full-chart continuous indicator: [{ upper, lower, mid }]
    this.overlays = {
      smartPrice: null, // { time, value }
      upperBand: null,  // { time, value }
      lowerBand: null,  // { time, value }
    };
    this.lastPrice = null;

    // Layer visibility
    this.layers = {
      markers: true,
      bands: true,
      smartPrice: true,
      actualLine: false,
      volume: true,
    };

    // Viewport state (TradingView smooth continuous coordinates)
    this.startIndex = 0.0;
    this.visibleCount = 80.0;
    this.dragMode = null; // null | 'pan' | 'timeZoom' | 'priceScale'
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.dragStartStartIndex = 0;
    this.dragStartVisibleCount = 80;
    this.customPriceRange = null; // { min, max } when price axis is dragged
    this.dragStartPriceRange = null;
    this.mousePos = null;

    // Callbacks
    this.onCrosshairMove = null;

    // Resize handling
    this.dpr = window.devicePixelRatio || 1;
    this.resize();
    this.initEventListeners();
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    if (this.width === 0 || this.height === 0) return;

    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    this.chartWidth = this.width - this.options.priceAxisWidth;
    this.chartHeight = this.height - this.options.timeAxisHeight;
    this.requestRender();
  }

  setData(candles) {
    this.candles = candles || [];
    this.computeIndicatorBands();

    // Anchor viewport to the right (latest candles)
    if (this.candles.length > 0) {
      this.startIndex = Math.max(0, this.candles.length - this.visibleCount);
      this.lastPrice = this.candles[this.candles.length - 1].close;
    }
    this.requestRender();
  }

  computeIndicatorBands() {
    // Compute 20-period Adaptive Volatility Channel across ALL historical candles
    const n = this.candles.length;
    this.indicatorBands = new Array(n);
    const period = 20;
    const mult = 2.0;

    for (let i = 0; i < n; i++) {
      if (i < 2) {
        this.indicatorBands[i] = null;
        continue;
      }
      const start = Math.max(0, i - period + 1);
      const count = i - start + 1;
      let sum = 0;
      for (let j = start; j <= i; j++) {
        sum += this.candles[j].close;
      }
      const mean = sum / count;

      let varianceSum = 0;
      for (let j = start; j <= i; j++) {
        varianceSum += Math.pow(this.candles[j].close - mean, 2);
      }
      const std = Math.sqrt(varianceSum / count);

      this.indicatorBands[i] = {
        mid: mean,
        upper: mean + mult * std,
        lower: mean - mult * std,
      };
    }
  }

  updateCandle(candle) {
    if (!candle) return;
    this.lastPrice = candle.close;
    const n = this.candles.length;

    if (n === 0) {
      this.candles.push(candle);
      this.computeIndicatorBands();
    } else {
      const last = this.candles[n - 1];
      if (last.time === candle.time) {
        // Update forming candle
        this.candles[n - 1] = candle;
        // Recompute latest band point
        const start = Math.max(0, n - 20);
        let sum = 0;
        for (let j = start; j < n; j++) sum += this.candles[j].close;
        const mean = sum / (n - start);
        let varSum = 0;
        for (let j = start; j < n; j++) varSum += Math.pow(this.candles[j].close - mean, 2);
        const std = Math.sqrt(varSum / (n - start));
        this.indicatorBands[n - 1] = {
          mid: mean,
          upper: mean + 2.0 * std,
          lower: mean - 2.0 * std,
        };
      } else if (candle.time > last.time) {
        // New candle opened
        this.candles.push(candle);
        this.computeIndicatorBands();
        // Stick to realtime edge if already viewing latest
        if (this.startIndex >= n - this.visibleCount - 3) {
          this.startIndex = Math.max(0, this.candles.length - this.visibleCount);
        }
      }
    }
    this.requestRender();
  }

  setMarkers(markers) {
    if (!markers || markers.length === 0) {
      this.markers = [];
      this.requestRender();
      return;
    }
    // Strictly deduplicate by candle timestamp: exactly 1 marker per candle
    const uniqueMap = new Map();
    markers.forEach((m) => {
      if (m && m.time != null) {
        uniqueMap.set(m.time, m);
      }
    });
    this.markers = Array.from(uniqueMap.values());
    this.requestRender();
  }

  setPredictions(predictions) {
    this.predictions = Array.isArray(predictions) ? predictions.slice() : [];
    this.requestRender();
  }

  setIndicativeBands(bands) {
    if (!bands) {
      this.overlays.smartPrice = null;
      this.overlays.upperBand = null;
      this.overlays.lowerBand = null;
    } else {
      this.overlays.smartPrice = bands.smartPrice || null;
      this.overlays.upperBand = bands.upper || null;
      this.overlays.lowerBand = bands.lower || null;
    }
    this.requestRender();
  }

  setLayerVisible(layer, visible) {
    if (this.layers.hasOwnProperty(layer)) {
      this.layers[layer] = visible;
      this.requestRender();
    }
  }

  scrollToRealtime() {
    this.customPriceRange = null;
    this.startIndex = Math.max(0, this.candles.length - this.visibleCount);
    this.requestRender();
  }

  // ===================================================================
  // TRADINGVIEW INTERACTION MECHANISMS (Cursor-Anchored Zoom & Axis Drag)
  // ===================================================================
  initEventListeners() {
    window.addEventListener("resize", () => this.resize());

    // Double click anywhere to reset vertical scale to auto-fit
    this.canvas.addEventListener("dblclick", () => {
      this.customPriceRange = null;
      this.requestRender();
    });

    // Mouse Down
    this.canvas.addEventListener("mousedown", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;
      this.dragStartStartIndex = this.startIndex;
      this.dragStartVisibleCount = this.visibleCount;
      this.dragStartPriceRange = this.currentPriceRange ? { ...this.currentPriceRange } : null;

      if (x >= this.chartWidth) {
        // Dragging on Right Price Axis -> Vertical stretch/compress
        this.dragMode = "priceScale";
        this.canvas.style.cursor = "ns-resize";
      } else if (y >= this.chartHeight) {
        // Dragging on Bottom Time Axis -> Horizontal zoom
        this.dragMode = "timeZoom";
        this.canvas.style.cursor = "ew-resize";
      } else {
        // Dragging inside Chart Area -> Free 2D Pan (horizontally and vertically)
        this.dragMode = "pan";
        this.canvas.style.cursor = "grabbing";
      }
    });

    // Mouse Move
    window.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (x >= 0 && x <= this.width && y >= 0 && y <= this.height) {
        this.mousePos = { x, y };
        if (!this.dragMode) {
          if (x >= this.chartWidth) {
            this.canvas.style.cursor = "ns-resize";
          } else if (y >= this.chartHeight) {
            this.canvas.style.cursor = "ew-resize";
          } else {
            this.canvas.style.cursor = "crosshair";
          }
        }
      } else {
        this.mousePos = null;
      }

      if (this.dragMode === "pan") {
        const deltaX = e.clientX - this.dragStartX;
        const candleStep = this.chartWidth / this.visibleCount;
        const shift = deltaX / candleStep;
        // Allow moving past the most recent candle so user can position it freely
        const maxStart = Math.max(1, this.candles.length - 2);
        const minStart = -this.visibleCount + 3;
        this.startIndex = Math.max(minStart, Math.min(maxStart, this.dragStartStartIndex - shift));

        // Allow dragging up and down freely ONLY when vertical drag is intentional (> 8px)
        const deltaY = e.clientY - this.dragStartY;
        if (this.dragStartPriceRange && Math.abs(deltaY) > 8 && Math.abs(deltaY) > Math.abs(deltaX) * 0.8) {
          const priceSpan = this.dragStartPriceRange.max - this.dragStartPriceRange.min;
          const priceShift = (deltaY / this.chartHeight) * priceSpan;
          this.customPriceRange = {
            min: this.dragStartPriceRange.min + priceShift,
            max: this.dragStartPriceRange.max + priceShift,
          };
        }
        this.requestRender();
      } else if (this.dragMode === "timeZoom") {
        // Horizontal time-axis drag zoom anchored towards most recent candle
        const deltaX = e.clientX - this.dragStartX;
        const zoomFactor = Math.pow(1.004, -deltaX);
        const newVisible = Math.max(
          this.options.minCandlesVisible,
          Math.min(this.options.maxCandlesVisible, this.dragStartVisibleCount * zoomFactor)
        );
        const rightEdge = this.dragStartStartIndex + this.dragStartVisibleCount;
        this.visibleCount = newVisible;
        const maxStart = Math.max(1, this.candles.length - 2);
        const minStart = -newVisible + 3;
        this.startIndex = Math.max(minStart, Math.min(maxStart, rightEdge - newVisible));
        this.requestRender();
      } else if (this.dragMode === "priceScale" && this.dragStartPriceRange) {
        // Vertical price-axis drag scale
        const deltaY = e.clientY - this.dragStartY;
        const scaleFactor = Math.pow(1.005, deltaY);
        const midPrice = (this.dragStartPriceRange.max + this.dragStartPriceRange.min) / 2;
        const halfSpan = ((this.dragStartPriceRange.max - this.dragStartPriceRange.min) / 2) * scaleFactor;
        this.customPriceRange = {
          min: midPrice - halfSpan,
          max: midPrice + halfSpan,
        };
        this.requestRender();
      } else if (this.mousePos) {
        this.requestRender();
      }
    });

    // Mouse Up
    window.addEventListener("mouseup", () => {
      if (this.dragMode) {
        this.dragMode = null;
        this.canvas.style.cursor = "crosshair";
      }
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.mousePos = null;
      this.requestRender();
    });

    // ===================================================================
    // Two-finger Touchpad, Pinch-to-Zoom & Wheel Gestures:
    // - Pinch to Zoom (e.ctrlKey or touch pinch):
    //   Expand fingers = zoom IN
    //   Pinch fingers = zoom OUT
    // - 2 finger scroll UP (deltaY < 0) = zoom out
    // - 2 finger scroll DOWN (deltaY > 0) = zoom in towards most recent candle
    // - 2 finger scroll left to right (deltaX < 0) = move chart left to right
    // - 2 finger scroll right to left (deltaX > 0) = move chart right to left
    // ===================================================================
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();

      // 1. Pinch-to-Zoom gesture (Trackpad pinch sets e.ctrlKey === true)
      if (e.ctrlKey) {
        this.customPriceRange = null; // Auto-scale price dynamically on zoom
        // Expand fingers (deltaY < 0): ZOOM IN
        // Pinch fingers (deltaY > 0): ZOOM OUT
        const zoomDelta = e.deltaY;
        const zoomFactor = Math.pow(1.008, zoomDelta);
        const newVisible = Math.max(
          this.options.minCandlesVisible,
          Math.min(this.options.maxCandlesVisible, this.visibleCount * zoomFactor)
        );

        if (newVisible < this.visibleCount) {
          // Zoom IN: expand candles, keep recent candle in frame
          const recentIndex = this.candles.length - 1;
          this.visibleCount = newVisible;
          if (this.startIndex + newVisible < recentIndex) {
            this.startIndex = recentIndex - newVisible + 1;
          } else if (this.startIndex > recentIndex) {
            this.startIndex = recentIndex - 2;
          }
        } else {
          // Zoom OUT: pinch in, reveal more candles
          const rightEdge = this.startIndex + this.visibleCount;
          this.visibleCount = newVisible;
          this.startIndex = rightEdge - newVisible;
        }

        const maxStart = Math.max(1, this.candles.length - 2);
        const minStart = -newVisible + 3;
        this.startIndex = Math.max(minStart, Math.min(maxStart, this.startIndex));
        this.requestRender();
        return;
      }

      const absX = Math.abs(e.deltaX);
      const absY = Math.abs(e.deltaY);

      // 2. Horizontal gesture: 2 finger scroll left to right / right to left
      if (absX > absY && absX > 1.0) {
        const candleStep = this.chartWidth / this.visibleCount;
        const shiftCandles = (e.deltaX / candleStep) * 0.45;
        const maxStart = Math.max(1, this.candles.length - 2);
        const minStart = -this.visibleCount + 3;
        this.startIndex = Math.max(minStart, Math.min(maxStart, this.startIndex + shiftCandles));
        this.requestRender();
        return;
      }

      // 3. Vertical gesture:
      // 2 finger scroll UP (deltaY < 0) = zoom out
      // 2 finger scroll DOWN (deltaY > 0) = zoom in towards most recent candle so it stays in frame
      if (absY > 0.5) {
        this.customPriceRange = null; // Auto-scale price dynamically on zoom
        if (e.deltaY < 0) {
          // Zoom OUT: expand visible candles
          const newVisible = Math.min(
            this.options.maxCandlesVisible,
            this.visibleCount * 1.085
          );
          const rightEdge = this.startIndex + this.visibleCount;
          this.visibleCount = newVisible;
          const maxStart = Math.max(1, this.candles.length - 2);
          const minStart = -newVisible + 3;
          this.startIndex = Math.max(minStart, Math.min(maxStart, rightEdge - newVisible));
        } else {
          // Zoom IN: zoom towards most recent candle so it ALWAYS stays in frame
          const newVisible = Math.max(
            this.options.minCandlesVisible,
            this.visibleCount * 0.92
          );
          const recentIndex = this.candles.length - 1;
          this.visibleCount = newVisible;
          if (this.startIndex + newVisible < recentIndex) {
            this.startIndex = recentIndex - newVisible + 1;
          } else if (this.startIndex > recentIndex) {
            this.startIndex = recentIndex - 2;
          }
          const maxStart = Math.max(1, this.candles.length - 2);
          const minStart = -newVisible + 3;
          this.startIndex = Math.max(minStart, Math.min(maxStart, this.startIndex));
        }
        this.requestRender();
      }
    }, { passive: false });

    // Touchscreen 2-finger pinch gesture handler
    let touchStartDist = 0;
    let touchStartVisibleCount = 80;

    this.canvas.addEventListener("touchstart", (e) => {
      if (e.touches && e.touches.length === 2) {
        touchStartDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        touchStartVisibleCount = this.visibleCount;
      }
    }, { passive: true });

    this.canvas.addEventListener("touchmove", (e) => {
      if (e.touches && e.touches.length === 2 && touchStartDist > 0) {
        e.preventDefault();
        this.customPriceRange = null; // Auto-scale price dynamically on zoom
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        // Expand fingers (dist > touchStartDist) -> scale < 1 -> ZOOM IN
        // Pinch fingers (dist < touchStartDist) -> scale > 1 -> ZOOM OUT
        const scale = touchStartDist / dist;
        const newVisible = Math.max(
          this.options.minCandlesVisible,
          Math.min(this.options.maxCandlesVisible, touchStartVisibleCount * scale)
        );
        this.visibleCount = newVisible;
        const maxStart = Math.max(1, this.candles.length - 2);
        const minStart = -newVisible + 3;
        this.startIndex = Math.max(minStart, Math.min(maxStart, this.startIndex));
        this.requestRender();
      }
    }, { passive: false });

    this.canvas.addEventListener("touchend", () => {
      touchStartDist = 0;
    });
  }

  zoom(factor) {
    if (!this.candles || this.candles.length === 0) return;
    this.customPriceRange = null; // Auto-scale price dynamically on zoom

    const newVisible = Math.max(
      this.options.minCandlesVisible,
      Math.min(this.options.maxCandlesVisible, this.visibleCount * factor)
    );

    // Check if user is at or near the most recent candle (real-time edge)
    const isAtRealtime = (this.startIndex + this.visibleCount) >= (this.candles.length - 3);

    if (isAtRealtime) {
      // Pin strictly to the most recent candle at the right edge
      this.visibleCount = newVisible;
      this.startIndex = Math.max(0, this.candles.length - newVisible);
    } else {
      // If panned into history, zoom towards the rightmost visible candle (towards recent)
      const rightEdge = this.startIndex + this.visibleCount;
      this.visibleCount = newVisible;
      const maxStart = Math.max(0, this.candles.length - newVisible);
      this.startIndex = Math.max(0, Math.min(maxStart, rightEdge - newVisible));
    }

    this.requestRender();
  }

  requestRender() {
    if (this._renderPending) return;
    this._renderPending = true;
    requestAnimationFrame(() => {
      this._renderPending = false;
      this.render();
    });
  }

  render() {
    if (!this.ctx || this.width === 0 || this.height === 0) return;
    const ctx = this.ctx;

    ctx.fillStyle = this.options.bgColor;
    ctx.fillRect(0, 0, this.width, this.height);

    if (this.candles.length === 0) {
      ctx.fillStyle = this.options.textColor;
      ctx.font = "13px -apple-system, BlinkMacSystemFont, Roboto, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Loading master market data...", this.chartWidth / 2, this.chartHeight / 2);
      return;
    }

    const intStart = Math.floor(this.startIndex);
    const endIndex = Math.ceil(this.startIndex + this.visibleCount) + 1;
    const candleStart = Math.max(0, intStart);
    const candleEnd = Math.min(this.candles.length, endIndex);

    // Dynamic candle-centric price auto-range calculation
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    let maxVolume = 0;

    for (let i = candleStart; i < candleEnd; i++) {
      const c = this.candles[i];
      if (!c) continue;
      if (c.low < minPrice) minPrice = c.low;
      if (c.high > maxPrice) maxPrice = c.high;
      if (c.volume > maxVolume) maxVolume = c.volume;
    }

    if (!isFinite(minPrice) || !isFinite(maxPrice)) {
      if (this.candles.length > 0) {
        const lastC = this.candles[this.candles.length - 1];
        minPrice = lastC.low * 0.99;
        maxPrice = lastC.high * 1.01;
      } else {
        minPrice = 0;
        maxPrice = 100;
      }
    }

    const candleSpan = maxPrice - minPrice || 1.0;

    // Accommodate forecast target lines smoothly without squishing the candles
    if (this.layers.bands && this.overlays.upperBand) {
      const targetUp = this.overlays.upperBand.value;
      maxPrice = Math.max(maxPrice, Math.min(targetUp, maxPrice + candleSpan * 0.35));
    }
    if (this.layers.bands && this.overlays.lowerBand) {
      const targetDn = this.overlays.lowerBand.value;
      minPrice = Math.min(minPrice, Math.max(targetDn, minPrice - candleSpan * 0.35));
    }

    // Apply custom price scale if explicitly dragged by user, else use dynamic auto-scale
    if (this.customPriceRange) {
      minPrice = this.customPriceRange.min;
      maxPrice = this.customPriceRange.max;
    } else {
      const priceRange = maxPrice - minPrice || 1.0;
      // Headroom: 10% on top, 16% on bottom for volume bars & indicators
      minPrice -= priceRange * 0.16;
      maxPrice += priceRange * 0.10;
      this.currentPriceRange = { min: minPrice, max: maxPrice };
    }

    // Geometry conversions
    const candleSlotWidth = this.chartWidth / this.visibleCount;
    const candleWidth = Math.max(1.5, candleSlotWidth * 0.72);

    const getY = (price) => {
      return this.chartHeight - ((price - minPrice) / (maxPrice - minPrice)) * this.chartHeight;
    };
    const getPrice = (y) => {
      return maxPrice - (y / this.chartHeight) * (maxPrice - minPrice);
    };
    const getX = (globalIndex) => {
      return (globalIndex - this.startIndex) * candleSlotWidth + candleSlotWidth / 2;
    };

    // 1. Grid
    this.drawGrid(ctx, minPrice, maxPrice);

    // 2. Volume Bars
    if (this.layers.volume && maxVolume > 0) {
      this.drawVolume(ctx, intStart, endIndex, getX, candleWidth, maxVolume);
    }

    // 3. FULL-CHART CONTINUOUS ADAPTIVE INDICATOR CHANNEL
    if (this.layers.bands && this.indicatorBands.length > 0) {
      this.drawFullChartIndicatorChannel(ctx, intStart, endIndex, getX, getY);
    }

    // 4. Candlesticks
    for (let i = intStart; i < endIndex; i++) {
      const c = this.candles[i];
      if (!c) continue;
      const x = getX(i);
      const isUp = c.close >= c.open;
      const color = isUp ? this.options.upColor : this.options.downColor;

      const yHigh = getY(c.high);
      const yLow = getY(c.low);
      const yOpen = getY(c.open);
      const yClose = getY(c.close);

      // Wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(Math.round(x), Math.round(yHigh));
      ctx.lineTo(Math.round(x), Math.round(yLow));
      ctx.stroke();

      // Real Body
      const bodyTop = Math.min(yOpen, yClose);
      const bodyBottom = Math.max(yOpen, yClose);
      const bodyHeight = Math.max(1.5, bodyBottom - bodyTop);

      ctx.fillStyle = color;
      ctx.fillRect(
        Math.round(x - candleWidth / 2),
        Math.round(bodyTop),
        Math.round(candleWidth),
        Math.round(bodyHeight)
      );
    }

    // 5. AI Prediction Markers (Directional Up / Down)
    if (this.layers.markers && this.markers.length > 0) {
      this.drawMarkers(ctx, intStart, endIndex, getX, getY);
    }

    // 6. Actual Price Line (Candle Body Midpoints)
    if (this.layers.actualLine) {
      this.drawActualBodyMidLine(ctx, intStart, endIndex, getX, getY);
    }

    // 7. Persistent Predicted Directions Line & Smart Forecast Rays
    if (this.layers.smartPrice) {
      this.drawSmartPredictedLineAndForecastRays(ctx, intStart, endIndex, getX, getY, candleSlotWidth);
    }

    // 7. Last Price Line with price badge
    if (this.lastPrice) {
      this.drawLastPriceLine(ctx, getY(this.lastPrice), this.lastPrice);
    }

    // 8. Time Axis at Bottom
    this.drawTimeAxis(ctx, intStart, endIndex, getX);

    // 9. Interactive Crosshair & Tooltip
    if (this.mousePos && this.mousePos.x <= this.chartWidth && this.mousePos.y <= this.chartHeight) {
      this.drawCrosshair(ctx, intStart, endIndex, candleSlotWidth, getY, getPrice);
    }
  }

  drawGrid(ctx, minPrice, maxPrice) {
    const numHorizontal = 6;
    ctx.strokeStyle = this.options.gridColor;
    ctx.lineWidth = 1;
    ctx.fillStyle = this.options.textColor;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, Roboto, sans-serif";
    ctx.textAlign = "left";

    for (let i = 0; i <= numHorizontal; i++) {
      const y = Math.round((i / numHorizontal) * this.chartHeight);
      const price = maxPrice - (i / numHorizontal) * (maxPrice - minPrice);

      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(0, y);
      ctx.lineTo(this.chartWidth, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillText(price.toFixed(2), this.chartWidth + 6, y + 4);
    }

    ctx.strokeStyle = "rgba(42, 46, 57, 0.8)";
    ctx.beginPath();
    ctx.moveTo(this.chartWidth, 0);
    ctx.lineTo(this.chartWidth, this.height);
    ctx.stroke();
  }

  drawVolume(ctx, intStart, endIndex, getX, candleWidth, maxVolume) {
    const volHeight = this.chartHeight * this.options.volumeHeightRatio;
    const volBaseY = this.chartHeight;

    for (let i = intStart; i < endIndex; i++) {
      const c = this.candles[i];
      if (!c) continue;
      const x = getX(i);
      const barH = (c.volume / maxVolume) * volHeight;
      const isUp = c.close >= c.open;
      ctx.fillStyle = isUp ? "rgba(38, 166, 154, 0.35)" : "rgba(239, 83, 80, 0.35)";
      ctx.fillRect(
        Math.round(x - candleWidth / 2),
        Math.round(volBaseY - barH),
        Math.round(candleWidth),
        Math.round(barH)
      );
    }
  }

  // ===================================================================
  // FULL-CHART CONTINUOUS ADAPTIVE INDICATOR CHANNEL
  // ===================================================================
  drawFullChartIndicatorChannel(ctx, intStart, endIndex, getX, getY) {
    if (!this.indicatorBands || this.indicatorBands.length === 0) return;

    // 1. Draw Translucent Shaded Channel Polygon
    ctx.beginPath();
    let hasStarted = false;

    // Trace Upper Line forward
    for (let i = intStart; i < endIndex; i++) {
      const b = this.indicatorBands[i];
      if (!b) continue;
      const x = getX(i);
      const yUp = getY(b.upper);
      if (!hasStarted) {
        ctx.moveTo(x, yUp);
        hasStarted = true;
      } else {
        ctx.lineTo(x, yUp);
      }
    }

    // Trace Lower Line backward
    for (let i = endIndex - 1; i >= intStart; i--) {
      const b = this.indicatorBands[i];
      if (!b) continue;
      const x = getX(i);
      const yDn = getY(b.lower);
      ctx.lineTo(x, yDn);
    }

    if (hasStarted) {
      ctx.closePath();
      ctx.fillStyle = "rgba(41, 98, 255, 0.08)";
      ctx.fill();
    }

    // 2. Draw Very Thin Upper Outline
    ctx.beginPath();
    let upperStarted = false;
    for (let i = intStart; i < endIndex; i++) {
      const b = this.indicatorBands[i];
      if (!b) continue;
      const x = getX(i);
      const yUp = getY(b.upper);
      if (!upperStarted) {
        ctx.moveTo(x, yUp);
        upperStarted = true;
      } else {
        ctx.lineTo(x, yUp);
      }
    }
    if (upperStarted) {
      ctx.strokeStyle = "rgba(41, 98, 255, 0.45)"; // Very thin outline
      ctx.lineWidth = 1.0;
      ctx.stroke();
    }

    // 3. Draw Very Thin Lower Outline
    ctx.beginPath();
    let lowerStarted = false;
    for (let i = intStart; i < endIndex; i++) {
      const b = this.indicatorBands[i];
      if (!b) continue;
      const x = getX(i);
      const yDn = getY(b.lower);
      if (!lowerStarted) {
        ctx.moveTo(x, yDn);
        lowerStarted = true;
      } else {
        ctx.lineTo(x, yDn);
      }
    }
    if (lowerStarted) {
      ctx.strokeStyle = "rgba(41, 98, 255, 0.45)"; // Very thin outline
      ctx.lineWidth = 1.0;
      ctx.stroke();
    }

    // 4. Draw Very Thin Moving Average Midline (dashed)
    ctx.beginPath();
    let midStarted = false;
    for (let i = intStart; i < endIndex; i++) {
      const b = this.indicatorBands[i];
      if (!b) continue;
      const x = getX(i);
      const yMid = getY(b.mid);
      if (!midStarted) {
        ctx.moveTo(x, yMid);
        midStarted = true;
      } else {
        ctx.lineTo(x, yMid);
      }
    }
    if (midStarted) {
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(41, 98, 255, 0.28)";
      ctx.lineWidth = 1.0;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  drawMarkers(ctx, intStart, endIndex, getX, getY) {
    const timeMap = new Map();
    for (let i = intStart; i < endIndex; i++) {
      const c = this.candles[i];
      if (c) timeMap.set(c.time, { idx: i, candle: c });
    }

    // Guard: each candle timestamp can ONLY receive at most one marker!
    const drawnCandleTimes = new Set();

    this.markers.forEach((m) => {
      if (!m || m.time == null || drawnCandleTimes.has(m.time)) return;
      const entry = timeMap.get(m.time);
      if (!entry) return;

      drawnCandleTimes.add(m.time);

      const x = getX(entry.idx);
      const c = entry.candle;

      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";

      if (m.shape === "arrowUp") {
        const y = getY(c.low) + 16;
        ctx.fillStyle = this.options.upColor;
        ctx.beginPath();
        ctx.moveTo(x, y - 6);
        ctx.lineTo(x - 5, y + 4);
        ctx.lineTo(x + 5, y + 4);
        ctx.closePath();
        ctx.fill();
        if (m.text) {
          ctx.fillText(m.text, x, y + 15);
        }
      } else if (m.shape === "arrowDown") {
        const y = getY(c.high) - 16;
        ctx.fillStyle = this.options.downColor;
        ctx.beginPath();
        ctx.moveTo(x, y + 6);
        ctx.lineTo(x - 5, y - 4);
        ctx.lineTo(x + 5, y - 4);
        ctx.closePath();
        ctx.fill();
        if (m.text) {
          ctx.fillText(m.text, x, y - 10);
        }
      } else {
        const y = getY((c.open + c.close) / 2);
        ctx.fillStyle = this.options.textColor;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // ===================================================================
  // ACTUAL PRICE LINE (CANDLE BODY MIDPOINTS)
  // ===================================================================
  drawActualBodyMidLine(ctx, intStart, endIndex, getX, getY) {
    if (!this.layers.actualLine || this.candles.length === 0) return;

    const candleStart = Math.max(0, intStart);
    const candleEnd = Math.min(this.candles.length, endIndex);
    if (candleEnd <= candleStart) return;

    ctx.save();
    // 1. Line connecting body midpoints (split into groups if gap > 3 candles)
    ctx.beginPath();
    let started = false;
    let prevTime = null;
    for (let i = candleStart; i < candleEnd; i++) {
      const c = this.candles[i];
      if (!c) continue;
      const midPrice = (Number(c.open) + Number(c.close)) / 2.0;
      const x = getX(i);
      const y = getY(midPrice);

      const isGap = prevTime != null && (c.time - prevTime > 3 * 300);
      if (!started || isGap) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
      prevTime = c.time;
    }

    if (started) {
      ctx.strokeStyle = "rgba(255, 179, 0, 0.85)"; // Rich Golden Amber
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    ctx.restore();
  }

  // ===================================================================
  // PERSISTENT PREDICTED DIRECTIONS LINE & SMART FORECAST RAYS
  // ===================================================================
  drawSmartPredictedLineAndForecastRays(ctx, intStart, endIndex, getX, getY, candleSlotWidth) {
    if (this.candles.length === 0) return;

    const timeToCandleIdx = new Map();
    for (let i = 0; i < this.candles.length; i++) {
      timeToCandleIdx.set(this.candles[i].time, i);
    }

    const nCandles = this.candles.length;
    const lastCandleIdx = nCandles - 1;

    // Helper to extract unix seconds
    const toSec = (ts) => {
      if (!ts) return 0;
      if (typeof ts === "number") return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts);
      let s = String(ts).trim().replace(" ", "T");
      if (!s.endsWith("Z") && !s.includes("+")) s += "Z";
      const ms = Date.parse(s);
      return isNaN(ms) ? 0 : Math.floor(ms / 1000);
    };

    ctx.save();

    // 1. Collect and draw persistent anchored rays for each prediction
    // "the indictor thats plotting where the price might go should keep the point of prediction where it is
    // it shouldnt chnage it and the line should continue to exist even if the cnadle has become old and outdated"
    const targetPoints = [];
    const preds = this.predictions || [];

    // Identify the latest candle prediction index so we ONLY write the label/price on the current candle
    let latestCandlePredIdx = -1;
    for (let i = preds.length - 1; i >= 0; i--) {
      const pSec = toSec(preds[i].open_time || preds[i].candle_open_time);
      if (pSec && timeToCandleIdx.has(pSec)) {
        latestCandlePredIdx = i;
        break;
      }
    }

    preds.forEach((p, pIdx) => {
      const pSec = toSec(p.open_time || p.candle_open_time);
      if (!pSec || !timeToCandleIdx.has(pSec)) return;

      const candIdx = timeToCandleIdx.get(pSec);
      const isLatestCandle = (pIdx === latestCandlePredIdx) || (candIdx >= lastCandleIdx);
      const c = this.candles[candIdx];
      const startX = getX(candIdx);
      const basePrice = Number(c.close);
      const startY = getY(basePrice);

      const targetIdx = candIdx + 2;
      let targetX = 0;
      if (targetIdx < nCandles) {
        targetX = getX(targetIdx);
      } else {
        targetX = getX(candIdx) + (targetIdx - candIdx) * candleSlotWidth;
      }

      // Compute raw smart price delta
      let smartPrice = p.smart_price != null ? Number(p.smart_price) : null;
      const isCallUp = (p.label === "UP" || (p.action && String(p.action).toUpperCase() === "UP"));
      if (smartPrice == null) {
        smartPrice = basePrice + (isCallUp ? 1 : -1) * (basePrice * 0.0035);
      }

      const rawDelta = smartPrice - basePrice;
      const delta = Math.abs(rawDelta) > 1e-6 ? rawDelta : (isCallUp ? basePrice * 0.003 : -basePrice * 0.003);

      const isPending = (p.status === "PENDING" || p.correct == null);
      const isCorrect = (p.correct === 1);

      // Smart direction plotting:
      // - If right: plot that point (full smartPrice)
      // - If wrong: plot half-way in predicted direction
      // - If pending: plot that point (full smartPrice)
      let plottedPrice = smartPrice;
      if (!isPending) {
        if (isCorrect) {
          plottedPrice = smartPrice;
        } else {
          plottedPrice = basePrice + 0.5 * delta;
        }
      }

      const targetY = getY(plottedPrice);

      // Draw persistent prediction ray from (startX, startY) to (targetX, targetY)
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(targetX, targetY);

      if (isPending) {
        ctx.strokeStyle = "#2962ff";
        ctx.lineWidth = isLatestCandle ? 2.0 : 1.2;
        ctx.stroke();

        // Terminal glowing node
        ctx.fillStyle = "#2962ff";
        ctx.beginPath();
        ctx.arc(targetX, targetY, isLatestCandle ? 4.0 : 3.0, 0, Math.PI * 2);
        ctx.fill();

        // Only write price and label on the indicator for the current/latest candle (never past candles)
        if (isLatestCandle) {
          ctx.font = "bold 10px sans-serif";
          ctx.fillStyle = "#2962ff";
          ctx.textAlign = "left";
          ctx.fillText(`Smart: ${plottedPrice.toFixed(1)}`, targetX + 6, targetY + 3);
        }
      } else {
          if (isCorrect) {
            ctx.strokeStyle = "rgba(38, 166, 154, 0.45)";
            ctx.lineWidth = 1.2;
          } else {
            ctx.strokeStyle = "rgba(239, 83, 80, 0.40)";
            ctx.lineWidth = 1.2;
          }
          ctx.stroke();

          // Target dot
          ctx.fillStyle = isCorrect ? "#26a69a" : "#ef5350";
          ctx.beginPath();
          ctx.arc(targetX, targetY, 3.0, 0, Math.PI * 2);
          ctx.fill();
        }

        targetPoints.push({
          x: targetX,
          y: targetY,
          candIdx: candIdx,
          targetIdx: targetIdx,
          time: pSec,
          price: plottedPrice,
          isCorrect: isCorrect,
          isPending: isPending
        });
      });

      // 2. CONNECT TARGET POINTS INTO THE "PREDICTED DIRECTIONS LINE"
      // If two predictions have more than 3 candles gap between them, do not draw between them (split into separate groups)
      if (targetPoints.length > 1) {
        targetPoints.sort((a, b) => (a.candIdx != null && b.candIdx != null) ? a.candIdx - b.candIdx : a.x - b.x);

        ctx.beginPath();
        let started = false;
        for (let i = 0; i < targetPoints.length; i++) {
          const pt = targetPoints[i];
          if (!started) {
            ctx.moveTo(pt.x, pt.y);
            started = true;
            continue;
          }

          const prevPt = targetPoints[i - 1];
          const candleGap = (pt.candIdx != null && prevPt.candIdx != null)
            ? (pt.candIdx - prevPt.candIdx)
            : Math.round(Math.abs(pt.x - prevPt.x) / candleSlotWidth);

          // If more than 3 candles gap, do not draw between them; start a new distinct prediction group
          if (candleGap > 3) {
            ctx.moveTo(pt.x, pt.y);
          } else {
            ctx.lineTo(pt.x, pt.y);
          }
        }

        if (started) {
          ctx.strokeStyle = "rgba(41, 150, 255, 0.75)"; // Sleek cyan-blue predicted trajectory
          ctx.lineWidth = 1.6;
          ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // 3. Draw active upper/lower caps for latest pending prediction
      if (this.layers.bands && this.overlays.upperBand) {
        const yUp = getY(this.overlays.upperBand.value);
        const startX = getX(lastCandleIdx);
        const targetX = startX + 2 * candleSlotWidth;

        ctx.strokeStyle = "rgba(239, 83, 80, 0.85)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(startX, yUp);
        ctx.lineTo(targetX, yUp);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = this.options.downColor;
        ctx.font = "9px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`Upper: ${this.overlays.upperBand.value.toFixed(1)}`, targetX + 4, yUp + 3);
      }

      if (this.layers.bands && this.overlays.lowerBand) {
        const yDn = getY(this.overlays.lowerBand.value);
        const startX = getX(lastCandleIdx);
        const targetX = startX + 2 * candleSlotWidth;

        ctx.strokeStyle = "rgba(38, 166, 154, 0.85)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(startX, yDn);
        ctx.lineTo(targetX, yDn);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = this.options.upColor;
        ctx.font = "9px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`Lower: ${this.overlays.lowerBand.value.toFixed(1)}`, targetX + 4, yDn + 3);
      }

      ctx.restore();
    }

  drawLastPriceLine(ctx, y, price) {
    let isUp = true;
    const n = this.candles.length;
    if (n > 0) {
      const lastCandle = this.candles[n - 1];
      if (lastCandle) {
        const curPrice = (price !== undefined && price !== null) ? Number(price) : Number(lastCandle.close);
        const openPrice = Number(lastCandle.open);
        if (curPrice !== openPrice) {
          isUp = curPrice > openPrice;
        } else if (n > 1 && this.candles[n - 2]) {
          isUp = curPrice >= Number(this.candles[n - 2].close);
        } else {
          isUp = true;
        }
      }
    }

    const color = isUp ? this.options.upColor : this.options.downColor;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(this.chartWidth, y);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = color;
    ctx.fillRect(this.chartWidth, y - 9, this.options.priceAxisWidth, 18);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px tabular-nums sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(Number(price).toFixed(2), this.chartWidth + 6, y + 4);
  }

  drawTimeAxis(ctx, intStart, endIndex, getX) {
    ctx.strokeStyle = "rgba(42, 46, 57, 0.8)";
    ctx.beginPath();
    ctx.moveTo(0, this.chartHeight);
    ctx.lineTo(this.width, this.chartHeight);
    ctx.stroke();

    ctx.fillStyle = this.options.textColor;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";

    const step = Math.max(1, Math.round(this.visibleCount / 7));
    const lastC = this.candles[this.candles.length - 1];
    const firstC = this.candles[0];

    for (let i = intStart; i < endIndex; i += step) {
      const x = getX(i);
      if (x < -40 || x > this.chartWidth + 40) continue;

      let unixSec = null;
      if (i >= 0 && i < this.candles.length) {
        unixSec = this.candles[i].time;
      } else if (lastC && i >= this.candles.length) {
        unixSec = lastC.time + (i - (this.candles.length - 1)) * 300;
      } else if (firstC && i < 0) {
        unixSec = firstC.time + i * 300;
      }

      if (!unixSec) continue;
      const d = new Date(unixSec * 1000);
      const timeStr = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const dateStr = `${d.getMonth() + 1}/${d.getDate()}`;
      ctx.fillText(`${dateStr} ${timeStr}`, x, this.chartHeight + 16);
    }
  }

  drawCrosshair(ctx, intStart, endIndex, candleSlotWidth, getY, getPrice) {
    const { x, y } = this.mousePos;

    // Fixed to 5-minute intervals: find exact candle or projected slot under cursor
    const candleIdx = Math.floor(this.startIndex + x / candleSlotWidth);
    // Snapped X: always passes through the middle of the candle stick only
    const candleCenterX = (candleIdx - this.startIndex) * candleSlotWidth + candleSlotWidth / 2;

    ctx.strokeStyle = "rgba(209, 212, 220, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    ctx.beginPath();
    // Horizontal crosshair follows cursor price
    ctx.moveTo(0, y);
    ctx.lineTo(this.chartWidth, y);
    // Vertical guideline is fixed to 5m interval and passes through middle of candle
    ctx.moveTo(Math.round(candleCenterX), 0);
    ctx.lineTo(Math.round(candleCenterX), this.chartHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    // Hovered Price Badge on right axis
    const hoveredPrice = getPrice(y);
    ctx.fillStyle = "#1e222d";
    ctx.fillRect(this.chartWidth, y - 9, this.options.priceAxisWidth, 18);
    ctx.strokeStyle = this.options.accentColor;
    ctx.strokeRect(this.chartWidth, y - 9, this.options.priceAxisWidth, 18);
    ctx.fillStyle = "#ffffff";
    ctx.font = "10px tabular-nums sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(hoveredPrice.toFixed(2), this.chartWidth + 6, y + 4);

    // Hovered Candle Data Pill on bottom time axis (snapped to candle center)
    let candleTime = null;
    let hoveredCandle = null;
    let hoveredBand = null;

    if (candleIdx >= 0 && candleIdx < this.candles.length) {
      hoveredCandle = this.candles[candleIdx];
      hoveredBand = this.indicatorBands[candleIdx];
      candleTime = hoveredCandle.time;
    } else if (this.candles.length > 0) {
      const lastC = this.candles[this.candles.length - 1];
      const firstC = this.candles[0];
      if (candleIdx >= this.candles.length && lastC) {
        candleTime = lastC.time + (candleIdx - (this.candles.length - 1)) * 300;
      } else if (firstC) {
        candleTime = firstC.time + candleIdx * 300;
      }
    }

    if (candleTime) {
      const d = new Date(candleTime * 1000);
      const timeStr = `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

      ctx.fillStyle = "#1e222d";
      ctx.fillRect(Math.round(candleCenterX - 55), this.chartHeight, 110, 20);
      ctx.strokeStyle = this.options.accentColor;
      ctx.strokeRect(Math.round(candleCenterX - 55), this.chartHeight, 110, 20);
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(timeStr, Math.round(candleCenterX), this.chartHeight + 14);
    }

    if (this.onCrosshairMove) {
      this.onCrosshairMove(hoveredCandle, hoveredBand);
    }
  }
}

window.NativeCandleChart = NativeCandleChart;
