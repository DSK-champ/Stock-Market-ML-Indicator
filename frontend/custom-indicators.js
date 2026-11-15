function getCustomIndicators(PineJS) {
	return Promise.resolve([{
		name: "2401CS45",
		metainfo: {
			_metainfoVersion: 56,
			id: "2401CS45@tv-basicstudies-1",
			name: "2401CS45",
			description: "2401CS45",
			shortDescription: "2401CS45 (Price Action & Predictions)",
			is_price_study: true,
			isCustomIndicator: true,
			plots: [
				{ id: "plot_0", type: "line" },
				{ id: "plot_1", type: "line" },
				{ id: "plot_2", type: "line" },
				{ id: "plot_3", type: "line" },
				{ id: "plot_4", type: "shapes" },
				{ id: "plot_5", type: "shapes" },
				{ id: "plot_6", type: "shapes" }
			],
			defaults: {
				styles: {
					plot_0: { linestyle: 0, linewidth: 2, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#2962ff" },
					plot_1: { linestyle: 0, linewidth: 1, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#ef5350" },
					plot_2: { linestyle: 0, linewidth: 1, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#26a69a" },
					plot_3: { linestyle: 0, linewidth: 2, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#ff9800" },
					plot_4: { linestyle: 0, linewidth: 3, plottype: "shape_arrow_up", trackPrice: false, location: "BelowBar", transparency: 0, visible: true, color: "#26a69a" },
					plot_5: { linestyle: 0, linewidth: 3, plottype: "shape_arrow_down", trackPrice: false, location: "AboveBar", transparency: 0, visible: true, color: "#ef5350" },
					plot_6: { linestyle: 0, linewidth: 2, plottype: "shape_circle", trackPrice: false, location: "BelowBar", transparency: 0, visible: true, color: "#787b86" }
				},
				filledAreasStyle: {
					fill_0: { color: "#2962ff", transparency: 90, visible: true }
				},
				inputs: { length: 20, mult: 2.0 }
			},
			styles: {
				plot_0: { title: "Smart Prediction Baseline", histogramBase: 0, joinPoints: false, isHidden: false },
				plot_1: { title: "Upper Cap (+2.0σ Channel)", histogramBase: 0, joinPoints: false, isHidden: false },
				plot_2: { title: "Lower Cap (-2.0σ Channel)", histogramBase: 0, joinPoints: false, isHidden: false },
				plot_3: { title: "Predicted Direction Path (Accuracy Scaled)", histogramBase: 0, joinPoints: false, isHidden: false },
				plot_4: { title: "Marker: UP (Buy)", histogramBase: 0, joinPoints: false, isHidden: false },
				plot_5: { title: "Marker: DOWN (Sell)", histogramBase: 0, joinPoints: false, isHidden: false },
				plot_6: { title: "Marker: NEUTRAL", histogramBase: 0, joinPoints: false, isHidden: false }
			},
			filledAreas: [{ id: "fill_0", objAId: "plot_1", objBId: "plot_2", type: "plot_plot", title: "Adaptive Prediction Channel" }],
			inputs: [
				{ id: "length", name: "Length", defval: 20, type: "integer", min: 1, max: 10000 },
				{ id: "mult", name: "StdDev Multiplier", defval: 2.0, type: "float", min: 0.001, max: 50 }
			],
			format: { type: "inherit" }
		},
		constructor: function() {
			this.main = function(context, inputs) {
				this._context = context;
				this._input = inputs;
				var length = this._input(0);
				var multiplier = this._input(1);

				var close = PineJS.Std.close(this._context);
				var sourceClose = this._context.new_var(close);
				var baseline = PineJS.Std.ema(sourceClose, length, this._context);
				var atrVal = PineJS.Std.atr(length, this._context);
				if (isNaN(atrVal) || atrVal === 0) {
					atrVal = close * 0.0035;
				}

				// 1. Adaptive Prediction Channels (unchanged baseline & caps)
				var bandWidth = multiplier * atrVal * 1.414;
				var upper = baseline + bandWidth;
				var lower = baseline - bandWidth;

				// 2. Identify Candle Prediction from Actual ML Ledger
				var predMap = null;
				if (typeof window !== "undefined") {
					predMap = window.__ML_PREDICTIONS_MAP ||
					          (window.parent && window.parent.__ML_PREDICTIONS_MAP) ||
					          (window.top && window.top.__ML_PREDICTIONS_MAP);
				}

				var rawTime = PineJS.Std.time(this._context);
				var timeSec = !isNaN(rawTime) ? Math.floor(rawTime / 1000) : 0;
				var roundedSec = Math.floor(timeSec / 300) * 300;

				var mlPred = null;
				if (predMap && timeSec > 0) {
					if (predMap.has(timeSec)) {
						mlPred = predMap.get(timeSec);
					} else if (predMap.has(roundedSec)) {
						mlPred = predMap.get(roundedSec);
					}
				}

				// 3. Markers and Prediction Path: STRICTLY for candles actually predicted
				var upMarker = NaN;
				var downMarker = NaN;
				var neuMarker = NaN;
				var plottedPath = NaN;

				if (mlPred) {
					var action = String(mlPred.shown_action || mlPred.action || mlPred.label || "").toUpperCase();
					var isUp = (action === "UP" || action === "LONG" || action === "1");
					var isDown = (action === "DOWN" || action === "SHORT" || action === "-1");
					var isNeu = (!isUp && !isDown);

					// Candle Markers
					if (isUp) upMarker = 1;
					else if (isDown) downMarker = 1;
					else neuMarker = 1;

					// Accuracy-Adjusted Predicted Direction Path (matches blank.html / native_chart.js)
					var basePrice = Number(mlPred.close || close);
					var smartPrice = mlPred.smart_price != null ? Number(mlPred.smart_price) : null;
					if (smartPrice == null || isNaN(smartPrice)) {
						smartPrice = basePrice + (isUp ? 1 : (isDown ? -1 : 0)) * (atrVal * 1.25);
					}

					var rawDelta = smartPrice - basePrice;
					var delta = Math.abs(rawDelta) > 1e-6 ? rawDelta : ((isUp ? 1 : (isDown ? -1 : 0)) * atrVal * 1.25);

					var isPending = (mlPred.status === "PENDING" || mlPred.correct == null);
					var isCorrect = (mlPred.correct === 1);

					if (isPending) {
						plottedPath = smartPrice;
					} else if (isCorrect) {
						plottedPath = smartPrice; // full direction if right
					} else if (isNeu) {
						plottedPath = basePrice;
					} else {
						plottedPath = basePrice + 0.38 * delta; // moves in that direction but slightly if wrong
					}
				}

				return [baseline, upper, lower, plottedPath, upMarker, downMarker, neuMarker];
			};
		}
	}, {
		name: "2401CS45 - BB",
		metainfo: {
			_metainfoVersion: 53,
			id: "2401CS45 - BB@tv-basicstudies-1",
			name: "2401CS45 - BB",
			description: "2401CS45 - BB",
			shortDescription: "BB",
			is_price_study: true,
			isCustomIndicator: true,
			plots: [
				{ id: "plot_0", type: "line" },
				{ id: "plot_1", type: "line" },
				{ id: "plot_2", type: "line" }
			],
			defaults: {
				styles: {
					plot_0: { linestyle: 0, linewidth: 1, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#FF6D00" },
					plot_1: { linestyle: 0, linewidth: 1, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#2196F3" },
					plot_2: { linestyle: 0, linewidth: 1, plottype: 0, trackPrice: false, transparency: 0, visible: true, color: "#2196F3" }
				},
				filledAreasStyle: {
					fill_0: { color: "#2196F3", transparency: 95, visible: true }
				},
				inputs: { length: 20, mult: 2 }
			},
			styles: {
				plot_0: { title: "Median", histogramBase: 0, joinPoints: false },
				plot_1: { title: "Upper", histogramBase: 0, joinPoints: false },
				plot_2: { title: "Lower", histogramBase: 0, joinPoints: false }
			},
			filledAreas: [{ id: "fill_0", objAId: "plot_1", objBId: "plot_2", type: "plot_plot", title: "Plots Background" }],
			inputs: [
				{ id: "length", name: "Length", defval: 20, type: "integer", min: 1, max: 10000 },
				{ id: "mult", name: "StdDev", defval: 2, type: "float", min: 0.001, max: 50 }
			],
			format: { type: "inherit" }
		},
		constructor: function() {
			this.main = function(context, inputs) {
				if (!this._loggedOnce) {
					console.log("INSIDE MAIN typeof window:", typeof window);
					this._loggedOnce = true;
				}
				this._context = context;
				this._input = inputs;
				var length = this._input(0);
				var multiplier = this._input(1);
				var source = context.new_var(PineJS.Std.close(context));
				var median = PineJS.Std.sma(source, length, context);
				var deviation = PineJS.Std.stdev(context.new_var(PineJS.Std.close(context)), length, context);
				var bandWidth = multiplier * deviation;
				return [median, median + bandWidth, median - bandWidth];
			};
		}
	}]);
}
