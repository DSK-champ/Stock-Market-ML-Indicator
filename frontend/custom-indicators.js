function getCustomIndicators(PineJS) {
	return Promise.resolve([{
		name: "2401CS45",
		metainfo: {
			_metainfoVersion: 53,
			id: "2401CS45@tv-basicstudies-1",
			name: "2401CS45",
			description: "2401CS45",
			shortDescription: "2401CS45",
			is_price_study: true,
			isCustomIndicator: true,
			plots: [{ id: "plot_0", type: "line" }],
			defaults: {
				styles: {
					plot_0: {
						linestyle: 0,
						linewidth: 1,
						plottype: 0,
						trackPrice: false,
						transparency: 0,
						visible: true,
						color: "#2196F3"
					}
				},
				inputs: { length: 9 }
			},
			styles: {
				plot_0: { title: "MA", histogramBase: 0, joinPoints: false }
			},
			inputs: [{ id: "length", name: "Length", defval: 9, type: "integer", min: 1, max: 10000 }],
			format: { type: "inherit" }
		},
		constructor: function() {
			this.main = function(context, inputs) {
				this._context = context;
				this._input = inputs;
				var source = context.new_var(PineJS.Std.close(context));
				return [PineJS.Std.sma(source, this._input(0), context)];
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
