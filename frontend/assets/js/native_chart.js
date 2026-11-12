/**
 * Native Canvas Candlestick Chart Engine
 * Pure HTML5 Canvas2D implementation for BTC/USDT 5m AI Indicator Dashboard.
 * 
 * Free of any TradingView libraries, logos, or watermarks.
 * Features: High-DPI rendering, pan/zoom, crosshair, volume histogram,
 * dynamic grid, AI prediction markers, and indicative forecast bands.
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
      candleWidth: 7,
      candleSpacing: 3,
      minCandlesVisible: 20,
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
      volume: true,
    };

    // Viewport state
    this.startIndex = 0;
    this.visibleCount = 80;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartIndex = 0;
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
    // Anchor viewport to the right (most recent) by default
    if (this.candles.length > 0) {
      this.startIndex = Math.max(0, this.candles.length - this.visibleCount);
      this.lastPrice = this.candles[this.candles.length - 1].close;
    }
    this.requestRender();
  }

  updateCandle(candle) {
    if (!candle) return;
    this.lastPrice = candle.close;
    if (this.candles.length === 0) {
      this.candles.push(candle);
    } else {
      const last = this.candles[this.candles.length - 1];
      if (last.time === candle.time) {
        // Update current forming candle
        this.candles[this.candles.length - 1] = candle;
      } else if (candle.time > last.time) {
        // New candle opened
        this.candles.push(candle);
        // Follow right edge if already at edge
        if (this.startIndex >= this.candles.length - this.visibleCount - 2) {
          this.startIndex = Math.max(0, this.candles.length - this.visibleCount);
        }
      }
    }
    this.requestRender();
  }

  setMarkers(markers) {
    this.markers = markers || [];
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
    this.startIndex = Math.max(0, this.candles.length - this.visibleCount);
    this.requestRender();
  }

  initEventListeners() {
    window.addEventListener("resize", () => this.resize());

    // Mouse drag pan
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX;
      this.dragStartIndex = this.startIndex;
      this.canvas.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (x >= 0 && x <= this.width && y >= 0 && y <= this.height) {
        this.mousePos = { x, y };
      } else {
        this.mousePos = null;
      }

      if (this.isDragging) {
        const deltaX = e.clientX - this.dragStartX;
        const candleStep = this.chartWidth / this.visibleCount;
        const indexShift = Math.round(deltaX / candleStep);
        const maxStart = Math.max(0, this.candles.length - this.visibleCount);
        this.startIndex = Math.max(0, Math.min(maxStart, this.dragStartIndex - indexShift));
      }

      this.requestRender();
    });

    window.addEventListener("mouseup", () => {
      if (this.isDragging) {
        this.isDragging = false;
        this.canvas.style.cursor = "crosshair";
      }
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.mousePos = null;
      this.requestRender();
    });

    // Zoom on wheel
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.15 : 0.85;
      const newVisible = Math.max(
        this.options.minCandlesVisible,
        Math.min(this.options.maxCandlesVisible, Math.round(this.visibleCount * zoomFactor))
      );

      const diff = newVisible - this.visibleCount;
      this.visibleCount = newVisible;
      const maxStart = Math.max(0, this.candles.length - this.visibleCount);
      this.startIndex = Math.max(0, Math.min(maxStart, Math.round(this.startIndex - diff / 2)));
      this.requestRender();
    }, { passive: false });
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

    // Clear background
    ctx.fillStyle = this.options.bgColor;
    ctx.fillRect(0, 0, this.width, this.height);

    if (this.candles.length === 0) {
      ctx.fillStyle = this.options.textColor;
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Loading master market data...", this.chartWidth / 2, this.chartHeight / 2);
      return;
    }

    // Slice visible candles (+2 lookahead for forward bands)
    const endIndex = Math.min(this.candles.length, this.startIndex + this.visibleCount);
    const visibleCandles = this.candles.slice(this.startIndex, endIndex);
    if (visibleCandles.length === 0) return;

    // Calculate Price Min / Max in visible window
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    let maxVolume = 0;

    visibleCandles.forEach((c) => {
      if (c.low < minPrice) minPrice = c.low;
      if (c.high > maxPrice) maxPrice = c.high;
      if (c.volume > maxVolume) maxVolume = c.volume;
    });

    // Include indicative band extremes if visible
    if (this.layers.bands && this.overlays.upperBand) {
      maxPrice = Math.max(maxPrice, this.overlays.upperBand.value);
    }
    if (this.layers.bands && this.overlays.lowerBand) {
      minPrice = Math.min(minPrice, this.overlays.lowerBand.value);
    }

    // Add 8% vertical padding
    const priceRange = maxPrice - minPrice || 1.0;
    minPrice -= priceRange * 0.06;
    maxPrice += priceRange * 0.06;

    // Coordinate converters
    const candleSlotWidth = this.chartWidth / this.visibleCount;
    const candleWidth = Math.max(1.5, candleSlotWidth * 0.72);

    const getY = (price) => {
      return this.chartHeight - ((price - minPrice) / (maxPrice - minPrice)) * this.chartHeight;
    };
    const getPrice = (y) => {
      return maxPrice - (y / this.chartHeight) * (maxPrice - minPrice);
    };
    const getX = (index) => {
      return index * candleSlotWidth + candleSlotWidth / 2;
    };

    // 1. Draw Grid Lines & Price Labels
    this.drawGrid(ctx, minPrice, maxPrice);

    // 2. Draw Volume Bars
    if (this.layers.volume && maxVolume > 0) {
      this.drawVolume(ctx, visibleCandles, candleSlotWidth, candleWidth, maxVolume);
    }

    // 3. Draw Candlesticks
    visibleCandles.forEach((c, i) => {
      const x = getX(i);
      const isUp = c.close >= c.open;
      const color = isUp ? this.options.upColor : this.options.downColor;

      const yHigh = getY(c.high);
      const yLow = getY(c.low);
      const yOpen = getY(c.open);
      const yClose = getY(c.close);

      // High / Low Wick
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
    });

    // 4. Draw AI Prediction Markers
    if (this.layers.markers && this.markers.length > 0) {
      this.drawMarkers(ctx, visibleCandles, getX, getY);
    }

    // 5. Draw Indicative Bands & Smart Price Horizon Projections
    if (this.overlays.smartPrice || this.overlays.upperBand || this.overlays.lowerBand) {
      this.drawForecastBands(ctx, visibleCandles, getX, getY);
    }

    // 6. Draw Current Price Line
    if (this.lastPrice) {
      this.drawLastPriceLine(ctx, getY(this.lastPrice), this.lastPrice);
    }

    // 7. Draw Time Axis at Bottom
    this.drawTimeAxis(ctx, visibleCandles, candleSlotWidth);

    // 8. Draw Crosshair & Hover Tooltip
    if (this.mousePos && this.mousePos.x <= this.chartWidth && this.mousePos.y <= this.chartHeight) {
      this.drawCrosshair(ctx, visibleCandles, candleSlotWidth, getY, getPrice);
    }
  }

  drawGrid(ctx, minPrice, maxPrice) {
    const numHorizontal = 6;
    ctx.strokeStyle = this.options.gridColor;
    ctx.lineWidth = 1;
    ctx.fillStyle = this.options.textColor;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, Roboto, sans-serif";
    ctx.textAlign = "left";

    // Horizontal Price Lines
    for (let i = 0; i <= numHorizontal; i++) {
      const y = Math.round((i / numHorizontal) * this.chartHeight);
      const price = maxPrice - (i / numHorizontal) * (maxPrice - minPrice);

      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(0, y);
      ctx.lineTo(this.chartWidth, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label on Price Axis
      ctx.fillText(price.toFixed(2), this.chartWidth + 6, y + 4);
    }

    // Price Axis Border
    ctx.strokeStyle = "rgba(42, 46, 57, 0.8)";
    ctx.beginPath();
    ctx.moveTo(this.chartWidth, 0);
    ctx.lineTo(this.chartWidth, this.height);
    ctx.stroke();
  }

  drawVolume(ctx, visibleCandles, candleSlotWidth, candleWidth, maxVolume) {
    const volHeight = this.chartHeight * this.options.volumeHeightRatio;
    const volBaseY = this.chartHeight;

    visibleCandles.forEach((c, i) => {
      const x = i * candleSlotWidth + candleSlotWidth / 2;
      const barH = (c.volume / maxVolume) * volHeight;
      const isUp = c.close >= c.open;
      ctx.fillStyle = isUp ? "rgba(38, 166, 154, 0.35)" : "rgba(239, 83, 80, 0.35)";
      ctx.fillRect(
        Math.round(x - candleWidth / 2),
        Math.round(volBaseY - barH),
        Math.round(candleWidth),
        Math.round(barH)
      );
    });
  }

  drawMarkers(ctx, visibleCandles, getX, getY) {
    const candleTimeMap = new Map();
    visibleCandles.forEach((c, i) => candleTimeMap.set(c.time, { idx: i, candle: c }));

    this.markers.forEach((m) => {
      const entry = candleTimeMap.get(m.time);
      if (!entry) return;

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

  drawForecastBands(ctx, visibleCandles, getX, getY) {
    if (visibleCandles.length === 0) return;
    const lastVisibleIdx = visibleCandles.length - 1;
    const startX = getX(lastVisibleIdx);
    // Horizon is 2 candles forward (10 min)
    const slotW = this.chartWidth / this.visibleCount;
    const targetX = startX + 2 * slotW;

    // Upper Cap
    if (this.layers.bands && this.overlays.upperBand) {
      const yUp = getY(this.overlays.upperBand.value);
      ctx.strokeStyle = "rgba(239, 83, 80, 0.85)";
      ctx.lineWidth = 1.5;
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

    // Lower Cap
    if (this.layers.bands && this.overlays.lowerBand) {
      const yDn = getY(this.overlays.lowerBand.value);
      ctx.strokeStyle = "rgba(38, 166, 154, 0.85)";
      ctx.lineWidth = 1.5;
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

    // Smart Price Ray
    if (this.layers.smartPrice && this.overlays.smartPrice) {
      const ySmart = getY(this.overlays.smartPrice.value);
      ctx.strokeStyle = this.options.accentColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startX, getY(visibleCandles[lastVisibleIdx].close));
      ctx.lineTo(targetX, ySmart);
      ctx.stroke();

      ctx.fillStyle = this.options.accentColor;
      ctx.beginPath();
      ctx.arc(targetX, ySmart, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = "bold 10px sans-serif";
      ctx.fillText(`Smart: ${this.overlays.smartPrice.value.toFixed(1)}`, targetX + 6, ySmart + 3);
    }
  }

  drawLastPriceLine(ctx, y, price) {
    ctx.strokeStyle = this.options.downColor;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(this.chartWidth, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Price badge on axis
    ctx.fillStyle = this.options.downColor;
    ctx.fillRect(this.chartWidth, y - 9, this.options.priceAxisWidth, 18);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px tabular-nums sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(price.toFixed(2), this.chartWidth + 6, y + 4);
  }

  drawTimeAxis(ctx, visibleCandles, candleSlotWidth) {
    ctx.strokeStyle = "rgba(42, 46, 57, 0.8)";
    ctx.beginPath();
    ctx.moveTo(0, this.chartHeight);
    ctx.lineTo(this.width, this.chartHeight);
    ctx.stroke();

    ctx.fillStyle = this.options.textColor;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";

    const step = Math.max(1, Math.round(this.visibleCount / 7));
    for (let i = 0; i < visibleCandles.length; i += step) {
      const c = visibleCandles[i];
      const x = i * candleSlotWidth + candleSlotWidth / 2;
      const d = new Date(c.time * 1000);
      const timeStr = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const dateStr = `${d.getMonth() + 1}/${d.getDate()}`;
      ctx.fillText(`${dateStr} ${timeStr}`, x, this.chartHeight + 16);
    }
  }

  drawCrosshair(ctx, visibleCandles, candleSlotWidth, getY, getPrice) {
    const { x, y } = this.mousePos;

    // Crosshair Lines
    ctx.strokeStyle = "rgba(209, 212, 220, 0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(this.chartWidth, y);
    ctx.moveTo(x, 0);
    ctx.lineTo(x, this.chartHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    // Hovered Price Badge on Y axis
    const hoveredPrice = getPrice(y);
    ctx.fillStyle = "#1e222d";
    ctx.fillRect(this.chartWidth, y - 9, this.options.priceAxisWidth, 18);
    ctx.strokeStyle = this.options.accentColor;
    ctx.strokeRect(this.chartWidth, y - 9, this.options.priceAxisWidth, 18);
    ctx.fillStyle = "#ffffff";
    ctx.font = "10px tabular-nums sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(hoveredPrice.toFixed(2), this.chartWidth + 6, y + 4);

    // Hovered Candle Data
    const candleIdx = Math.floor(x / candleSlotWidth);
    if (candleIdx >= 0 && candleIdx < visibleCandles.length) {
      const c = visibleCandles[candleIdx];
      const d = new Date(c.time * 1000);
      const timeStr = `${d.toLocaleDateString()} ${d.toLocaleTimeString()}`;

      // Time Badge on X axis
      ctx.fillStyle = "#1e222d";
      ctx.fillRect(x - 55, this.chartHeight, 110, 20);
      ctx.strokeStyle = this.options.accentColor;
      ctx.strokeRect(x - 55, this.chartHeight, 110, 20);
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(timeStr, x, this.chartHeight + 14);

      // Trigger crosshair move callback for legend
      if (this.onCrosshairMove) {
        this.onCrosshairMove(c);
      }
    }
  }
}

window.NativeCandleChart = NativeCandleChart;
