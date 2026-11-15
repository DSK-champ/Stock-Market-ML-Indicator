// Horizon-2 Feature Engineering Catalog (Zero Different Horizon Features)
// Dedicated Features for BTCUSDT 5m Horizon-2 Directional Predictor (10-minute forecast)

const PIPELINE_STAGES = [
  {
    "step": 1,
    "title": "Raw Candle Ingestion & Gap Auditing",
    "badge": "Data Foundation",
    "summary": "5m continuous OHLCV klines + Taker Volume + Trade Count are streamed from Binance Spot (BTCUSDT).",
    "details": "Candle timestamps are tracked to audit sequence continuity. Any missing intervals increment the gap segment counter to prevent rolling indicators from crossing data boundaries."
  },
  {
    "step": 2,
    "title": "Strict Anti-Leakage & Causality Protocol",
    "badge": "Zero Future Leakage",
    "summary": "All features at candle t are computed exclusively from closed candles <= t with zero lookahead bias.",
    "details": "Modifying any future candle (t+1 onwards) produces 0.000000000000 delta on historical features. A rolling lookback buffer of N >= 2,018 bars (~7.0 days) ensures zero NaNs across all seasonal and HAR components."
  },
  {
    "step": 3,
    "title": "Stationarity & Scale Normalization",
    "badge": "Covariate Invariance",
    "summary": "Raw USD prices are strictly excluded; all price series are converted to stationary ratios and normalized metrics.",
    "details": "Employs log-returns ln(C_t / C_{t-k}), ATR-14 normalized distances (close - S) / ATR, bounded rolling Z-scores, and harmonic trigonometric coordinates (sin/cos) for 24h diurnal and 168h weekly cycles."
  },
  {
    "step": 4,
    "title": "Multi-Domain Feature Extraction",
    "badge": "Horizon-2 Specialized",
    "summary": "Captures momentum, extreme volatility, microstructure order flow, multi-timeframe trends, and market structure.",
    "details": "All features of different horizons (horizon 3, 6, 12 lookbacks) have been removed, leaving only the Horizon-2 active features and general category indicators."
  },
  {
    "step": 5,
    "title": "Horizon-2 Model Input Vector",
    "badge": "ML Model Ingestion",
    "summary": "The dedicated Horizon-2 feature set (101 features) feeds the 10-minute forecast models.",
    "details": "Supplied to Horizon-2 directional ensembles, standardized via StandardScaler for linear models, and combined with 15 Meta-Model inputs."
  }
];

const FEATURES_CATALOG = [
  {
    "id": 1,
    "name": "feat_ret_1",
    "value": -0.001659,
    "group": "returns",
    "category": "Returns",
    "formula": "ln(close[t] / close[t-1])",
    "unit": "fraction",
    "window": "1 bar (5 mins)",
    "description": "Log return over 1 candle (5m)",
    "role": "Horizon 2 Base Input",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 2,
    "name": "feat_ret_2",
    "value": 0.000634,
    "group": "returns",
    "category": "Returns",
    "formula": "ln(close[t] / close[t-2])",
    "unit": "log return",
    "window": "2 bars (10 mins - Horizon 2)",
    "description": "10-minute return spanning exactly 2 bars, matching the Horizon-2 forecast window",
    "role": "Horizon 2 Core Return",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 3,
    "name": "feat_vol_sigma_288",
    "value": 0.001309,
    "group": "volatility",
    "category": "Rolling Volatility",
    "formula": "rolling_sigma(ret_1, 288)",
    "unit": "fraction",
    "window": "288 bars (24 hours)",
    "description": "Causal rolling return std over 288 candles (24h)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 4,
    "name": "feat_vol_natr_14",
    "value": 0.002563,
    "group": "volatility",
    "category": "Rolling Volatility",
    "formula": "ATR(14) / close",
    "unit": "ratio",
    "window": "14 bars (70 mins)",
    "description": "Normalized Average True Range over 14 candles",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 5,
    "name": "feat_vol_bb_pct_20",
    "value": 0.681048,
    "group": "volatility",
    "category": "Rolling Volatility",
    "formula": "(close - lower) / (upper - lower)",
    "unit": "ratio in [0, 1]",
    "window": "20 bars (100 mins)",
    "description": "Bollinger Bands %B (20-period, 2 sigma)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 6,
    "name": "feat_vol_bb_width_20",
    "value": 0.007619,
    "group": "volatility",
    "category": "Rolling Volatility",
    "formula": "(upper - lower) / SMA(20)",
    "unit": "ratio",
    "window": "20 bars (100 mins)",
    "description": "Bollinger Bands bandwidth normalized by SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 7,
    "name": "feat_shape_body_pct",
    "value": 0.00166,
    "group": "candle_shape",
    "category": "Candle Shape",
    "formula": "|close - open| / close",
    "unit": "fraction",
    "window": "1 bar (5 mins)",
    "description": "Candle real body size relative to close price",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 8,
    "name": "feat_shape_body_to_range",
    "value": 0.830067,
    "group": "candle_shape",
    "category": "Candle Shape",
    "formula": "|close - open| / (high - low)",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "Candle body size relative to full range [0, 1]",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 9,
    "name": "feat_shape_upper_wick_pct",
    "value": 0.000106,
    "group": "candle_shape",
    "category": "Candle Shape",
    "formula": "(high - max(open, close)) / close",
    "unit": "fraction",
    "window": "1 bar (5 mins)",
    "description": "Upper wick length relative to close price",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 10,
    "name": "feat_shape_lower_wick_pct",
    "value": 0.000234,
    "group": "candle_shape",
    "category": "Candle Shape",
    "formula": "(min(open, close) - low) / close",
    "unit": "fraction",
    "window": "1 bar (5 mins)",
    "description": "Lower wick length relative to close price",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 11,
    "name": "feat_shape_close_pos",
    "value": 0.117111,
    "group": "candle_shape",
    "category": "Candle Shape",
    "formula": "(close - low) / (high - low)",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "Close position within candle range [0, 1]",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 12,
    "name": "feat_shape_candle_dir",
    "value": -1.0,
    "group": "candle_shape",
    "category": "Candle Shape",
    "formula": "sign(close - open)",
    "unit": "integer {-1,0,1}",
    "window": "1 bar (5 mins)",
    "description": "Directional sign of candle body (+1 bullish, -1 bearish, 0 doji)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 13,
    "name": "feat_mom_sma_ratio_10",
    "value": 0.001578,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "close / SMA(10) - 1.0",
    "unit": "ratio",
    "window": "10 bars (50 mins)",
    "description": "Distance of close from 10-candle SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 14,
    "name": "feat_mom_sma_ratio_20",
    "value": 0.001379,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "close / SMA(20) - 1.0",
    "unit": "ratio",
    "window": "20 bars (100 mins)",
    "description": "Distance of close from 20-candle SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 15,
    "name": "feat_mom_sma_ratio_50",
    "value": 0.000597,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "close / SMA(50) - 1.0",
    "unit": "ratio",
    "window": "60 bars (5 hours)",
    "description": "Distance of close from 50-candle SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 16,
    "name": "feat_mom_ema_ratio_10",
    "value": 0.001148,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "close / EMA(10) - 1.0",
    "unit": "ratio",
    "window": "10 bars (50 mins)",
    "description": "Distance of close from 10-candle EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 17,
    "name": "feat_mom_ema_ratio_20",
    "value": 0.001634,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "close / EMA(20) - 1.0",
    "unit": "ratio",
    "window": "20 bars (100 mins)",
    "description": "Distance of close from 20-candle EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 18,
    "name": "feat_mom_ema_ratio_50",
    "value": -0.000786,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "close / EMA(50) - 1.0",
    "unit": "ratio",
    "window": "60 bars (5 hours)",
    "description": "Distance of close from 50-candle EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 19,
    "name": "feat_mom_sma_ratio_10_50",
    "value": -0.00098,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "SMA(10) / SMA(50) - 1.0",
    "unit": "ratio",
    "window": "60 bars (5 hours)",
    "description": "Spread between short and medium SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 20,
    "name": "feat_mom_rsi_7",
    "value": 0.586245,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "RSI(7) / 100.0",
    "unit": "oscillator",
    "window": "7 bars (35 mins)",
    "description": "Relative Strength Index over 7 candles [0, 1]",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 21,
    "name": "feat_mom_rsi_14",
    "value": 0.537961,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "RSI(14) / 100.0",
    "unit": "oscillator",
    "window": "14 bars (70 mins)",
    "description": "Relative Strength Index over 14 candles [0, 1]",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 22,
    "name": "feat_mom_macd_line",
    "value": 0.000166,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "(EMA(12) - EMA(26)) / close",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "MACD line normalized by close price",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 23,
    "name": "feat_mom_macd_signal",
    "value": -0.000369,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "EMA(macd_line, 9) / close",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "MACD 9-period signal line normalized by close",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 24,
    "name": "feat_mom_macd_hist",
    "value": 0.000534,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "(macd_line - macd_signal) / close",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "MACD histogram normalized by close",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 25,
    "name": "feat_mom_adx_14",
    "value": 0.184342,
    "group": "momentum",
    "category": "Momentum Oscillators",
    "formula": "ADX(14) / 100.0",
    "unit": "oscillator",
    "window": "14 bars (70 mins)",
    "description": "Average Directional Index over 14 candles [0, 1]",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 26,
    "name": "feat_volume_log",
    "value": 4.179426,
    "group": "volume",
    "category": "Volume & Activity",
    "formula": "ln(1 + volume)",
    "unit": "log volume",
    "window": "1 bar (5 mins)",
    "description": "Log volume",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 27,
    "name": "feat_volume_change_1",
    "value": -0.257578,
    "group": "volume",
    "category": "Volume & Activity",
    "formula": "volume[t] / volume[t-1] - 1.0",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "1-candle fractional volume change",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 28,
    "name": "feat_of_taker_buy_ratio",
    "value": 0.505491,
    "group": "order_flow",
    "category": "Order Flow Imbalance",
    "formula": "taker_buy_base_volume / volume",
    "unit": "ratio",
    "window": "1 bar (5 mins)",
    "description": "Taker buy volume fraction in [0, 1]",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 29,
    "name": "feat_of_avg_trade_size",
    "value": 0.005112,
    "group": "order_flow",
    "category": "Order Flow Imbalance",
    "formula": "volume / number_of_trades",
    "unit": "BTC / trade",
    "window": "1 bar (5 mins)",
    "description": "Average trade size in BTC",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 30,
    "name": "feat_time_sin_hour",
    "value": -0.965926,
    "group": "time",
    "category": "Cyclical Time",
    "formula": "sin(2 * pi * hour / 24)",
    "unit": "coordinate in [-1, 1]",
    "window": "1 bar (5 mins)",
    "description": "Sine transform of UTC hour of day",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 31,
    "name": "feat_time_cos_hour",
    "value": -0.258819,
    "group": "time",
    "category": "Cyclical Time",
    "formula": "cos(2 * pi * hour / 24)",
    "unit": "coordinate in [-1, 1]",
    "window": "1 bar (5 mins)",
    "description": "Cosine transform of UTC hour of day",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 32,
    "name": "feat_time_sin_weekday",
    "value": 0.433884,
    "group": "time",
    "category": "Cyclical Time",
    "formula": "sin(2 * pi * weekday / 7)",
    "unit": "coordinate in [-1, 1]",
    "window": "288 bars (24 hours)",
    "description": "Sine transform of day of week (Monday=0)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 33,
    "name": "feat_time_cos_weekday",
    "value": -0.900969,
    "group": "time",
    "category": "Cyclical Time",
    "formula": "cos(2 * pi * weekday / 7)",
    "unit": "coordinate in [-1, 1]",
    "window": "288 bars (24 hours)",
    "description": "Cosine transform of day of week (Monday=0)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 34,
    "name": "feat_mtf_1h_ret",
    "value": 0.000626,
    "group": "mtf_1h",
    "category": "MTF 1-Hour Aggregated",
    "formula": "ln(close_1h / open_1h) shifted 1h",
    "unit": "fraction",
    "window": "1H completed (12 bars)",
    "description": "Log return of last COMPLETED 1h candle",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 35,
    "name": "feat_mtf_1h_range",
    "value": 0.01007,
    "group": "mtf_1h",
    "category": "MTF 1-Hour Aggregated",
    "formula": "(high_1h - low_1h) / close_1h shifted 1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "High-low range of last COMPLETED 1h candle normalized by close",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 36,
    "name": "feat_mtf_1h_vol_ratio",
    "value": 1.585974,
    "group": "mtf_1h",
    "category": "MTF 1-Hour Aggregated",
    "formula": "vol_1h / SMA(vol_1h, 24) shifted 1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Volume of last COMPLETED 1h candle vs 24h rolling mean",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 37,
    "name": "feat_vol_parkinson_288",
    "value": 0.001228,
    "group": "vol_range",
    "category": "Range Volatility",
    "formula": "sqrt(1/(4 ln 2 * 288) * sum(ln(H/L)^2))",
    "unit": "fraction",
    "window": "288 bars (24 hours)",
    "description": "Parkinson volatility over 288 candles (24h)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 38,
    "name": "feat_vol_gk_288",
    "value": 0.001194,
    "group": "vol_range",
    "category": "Range Volatility",
    "formula": "sqrt(1/288 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))",
    "unit": "fraction",
    "window": "288 bars (24 hours)",
    "description": "Garman-Klass volatility over 288 candles (24h)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 39,
    "name": "feat_vol_seasonal_slot_abs_ret_7d",
    "value": 0.000766,
    "group": "vol_seasonal",
    "category": "Seasonal Volatility",
    "formula": "mean(|r_5m|[t - 288*k]) for k=1..7",
    "unit": "fraction",
    "window": "2016 bars (~7 days)",
    "description": "Same 5m time slot mean absolute return over prior 7 days",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 40,
    "name": "feat_vol_seasonal_volume_ratio_7d",
    "value": 1.766167,
    "group": "vol_seasonal",
    "category": "Seasonal Volatility",
    "formula": "vol[t] / mean(vol[t - 288*k]) for k=1..7",
    "unit": "ratio",
    "window": "2016 bars (~7 days)",
    "description": "Ratio of current volume to 7-day same-slot mean volume",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 41,
    "name": "feat_vol_seasonal_trades_ratio_7d",
    "value": 1.040562,
    "group": "vol_seasonal",
    "category": "Seasonal Volatility",
    "formula": "trades[t] / mean(trades[t - 288*k]) for k=1..7",
    "unit": "ratio",
    "window": "2016 bars (~7 days)",
    "description": "Ratio of current trades to 7-day same-slot mean trades",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 42,
    "name": "feat_vol_har_rv_72",
    "value": 0.002093,
    "group": "vol_har",
    "category": "HAR Volatility",
    "formula": "rolling_sigma(r_5m, 72)",
    "unit": "fraction",
    "window": "72 bars (6 hours)",
    "description": "Realized volatility over 6h (72 candles)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 43,
    "name": "feat_vol_har_rv_2016",
    "value": 0.001239,
    "group": "vol_har",
    "category": "HAR Volatility",
    "formula": "rolling_sigma(r_5m, 2016)",
    "unit": "fraction",
    "window": "2016 bars (~7 days)",
    "description": "Realized volatility over 7 days (2016 candles)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 44,
    "name": "feat_vol_har_ratio_288_2016",
    "value": 1.056686,
    "group": "vol_har",
    "category": "HAR Volatility",
    "formula": "rv_288 / rv_2016",
    "unit": "ratio",
    "window": "2016 bars (~7 days)",
    "description": "HAR volatility ratio: 24h RV / 7d RV",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 45,
    "name": "feat_loc_position_288",
    "value": 0.188546,
    "group": "location",
    "category": "Range Location",
    "formula": "(close - min(L_288)) / (max(H_288) - min(L_288))",
    "unit": "fraction [0, 1]",
    "window": "288 bars (24 hours)",
    "description": "Position of close within 24h high-low channel",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 46,
    "name": "feat_loc_dist_high_288",
    "value": 0.040123,
    "group": "location",
    "category": "Range Location",
    "formula": "(max(H_288) - close) / close",
    "unit": "fraction",
    "window": "288 bars (24 hours)",
    "description": "Fractional distance of close below 24h high",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 47,
    "name": "feat_loc_dist_low_288",
    "value": 0.009323,
    "group": "location",
    "category": "Range Location",
    "formula": "(close - min(L_288)) / close",
    "unit": "fraction",
    "window": "288 bars (24 hours)",
    "description": "Fractional distance of close above 24h low",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 48,
    "name": "feat_time_is_weekend",
    "value": 0.0,
    "group": "time_weekend",
    "category": "Weekend & Weekly Time",
    "formula": "1.0 if weekday >= 5 else 0.0",
    "unit": "binary {0, 1}",
    "window": "288 bars (24 hours)",
    "description": "Weekend indicator (Saturday or Sunday UTC)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 49,
    "name": "feat_time_sin_week",
    "value": -0.204252,
    "group": "time_weekend",
    "category": "Weekend & Weekly Time",
    "formula": "sin(2 * pi * (weekday*24 + hour + min/60) / 168)",
    "unit": "coordinate in [-1, 1]",
    "window": "288 bars (24 hours)",
    "description": "Sine transform of 168-hour weekly cycle",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 50,
    "name": "feat_time_cos_week",
    "value": -0.978918,
    "group": "time_weekend",
    "category": "Weekend & Weekly Time",
    "formula": "cos(2 * pi * (weekday*24 + hour + min/60) / 168)",
    "unit": "coordinate in [-1, 1]",
    "window": "288 bars (24 hours)",
    "description": "Cosine transform of 168-hour weekly cycle",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": true,
    "is_meta": false,
    "models": [
      "RF/GB",
      "LR"
    ]
  },
  {
    "id": 51,
    "name": "feat_ma_sma_ratio_5",
    "value": 0.000718,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma5) / sma5",
    "unit": "ratio",
    "window": "5 bars (25 mins)",
    "description": "Ratio of 5m close to 5-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 52,
    "name": "feat_ma_sma_ratio_25",
    "value": 0.00251,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma25) / sma25",
    "unit": "ratio",
    "window": "25 bars (125 mins)",
    "description": "Ratio of 5m close to 25-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 53,
    "name": "feat_ma_sma_ratio_100",
    "value": -0.008398,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma100) / sma100",
    "unit": "ratio",
    "window": "100 bars (~8.3 hours)",
    "description": "Ratio of 5m close to 100-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 54,
    "name": "feat_ma_ema_ratio_5",
    "value": 0.000426,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema5) / ema5",
    "unit": "ratio",
    "window": "5 bars (25 mins)",
    "description": "Ratio of 5m close to 5-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 55,
    "name": "feat_ma_ema_ratio_25",
    "value": 0.001533,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema25) / ema25",
    "unit": "ratio",
    "window": "25 bars (125 mins)",
    "description": "Ratio of 5m close to 25-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 56,
    "name": "feat_ma_ema_ratio_100",
    "value": -0.006332,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema100) / ema100",
    "unit": "ratio",
    "window": "100 bars (~8.3 hours)",
    "description": "Ratio of 5m close to 100-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 57,
    "name": "feat_ma_sma_slope_5",
    "value": 0.001721,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma5_t / sma5_{t-5} - 1",
    "unit": "fraction",
    "window": "5 bars (25 mins)",
    "description": "5-candle slope of 5-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 58,
    "name": "feat_ma_sma_slope_10",
    "value": 0.000615,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma10_t / sma10_{t-5} - 1",
    "unit": "fraction",
    "window": "10 bars (50 mins)",
    "description": "5-candle slope of 10-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 59,
    "name": "feat_ma_sma_slope_25",
    "value": 0.001188,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma25_t / sma25_{t-5} - 1",
    "unit": "fraction",
    "window": "25 bars (125 mins)",
    "description": "5-candle slope of 25-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 60,
    "name": "feat_ma_sma_slope_50",
    "value": -0.001012,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma50_t / sma50_{t-5} - 1",
    "unit": "fraction",
    "window": "60 bars (5 hours)",
    "description": "5-candle slope of 50-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 61,
    "name": "feat_ma_sma_slope_100",
    "value": -0.001113,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma100_t / sma100_{t-5} - 1",
    "unit": "fraction",
    "window": "100 bars (~8.3 hours)",
    "description": "5-candle slope of 100-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 62,
    "name": "feat_ma_ema_slope_5",
    "value": 0.002187,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "ema5_t / ema5_{t-5} - 1",
    "unit": "fraction",
    "window": "5 bars (25 mins)",
    "description": "5-candle slope of 5-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 63,
    "name": "feat_ma_ema_slope_10",
    "value": 0.001342,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "ema10_t / ema10_{t-5} - 1",
    "unit": "fraction",
    "window": "10 bars (50 mins)",
    "description": "5-candle slope of 10-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 64,
    "name": "feat_ma_ema_slope_25",
    "value": 0.000472,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "ema25_t / ema25_{t-5} - 1",
    "unit": "fraction",
    "window": "25 bars (125 mins)",
    "description": "5-candle slope of 25-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 65,
    "name": "feat_ma_ema_slope_50",
    "value": -0.000316,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "ema50_t / ema50_{t-5} - 1",
    "unit": "fraction",
    "window": "60 bars (5 hours)",
    "description": "5-candle slope of 50-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 66,
    "name": "feat_ma_ema_slope_100",
    "value": -0.000737,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "ema100_t / ema100_{t-5} - 1",
    "unit": "fraction",
    "window": "100 bars (~8.3 hours)",
    "description": "5-candle slope of 100-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 67,
    "name": "feat_ma_alignment_sma_5m",
    "value": -0.4,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(2*sum(short>long) - 10) / 10",
    "unit": "score in [-1, 1]",
    "window": "5 bars (25 mins)",
    "description": "Pairwise alignment score of 5m SMAs (5,10,25,50,100)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 68,
    "name": "feat_ma_alignment_ema_5m",
    "value": -0.4,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(2*sum(short>long) - 10) / 10",
    "unit": "score in [-1, 1]",
    "window": "5 bars (25 mins)",
    "description": "Pairwise alignment score of 5m EMAs (5,10,25,50,100)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 69,
    "name": "feat_ma_1h_sma_ratio_5",
    "value": -0.002485,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma5_1h) / sma5_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 5-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 70,
    "name": "feat_ma_1h_sma_ratio_10",
    "value": -0.011601,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma10_1h) / sma10_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 10-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 71,
    "name": "feat_ma_1h_sma_ratio_25",
    "value": -0.022037,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma25_1h) / sma25_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 25-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 72,
    "name": "feat_ma_1h_sma_ratio_50",
    "value": -0.022979,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma50_1h) / sma50_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 50-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 73,
    "name": "feat_ma_1h_sma_ratio_100",
    "value": -0.031449,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - sma100_1h) / sma100_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 100-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 74,
    "name": "feat_ma_1h_ema_ratio_5",
    "value": -0.002272,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema5_1h) / ema5_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 5-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 75,
    "name": "feat_ma_1h_ema_ratio_10",
    "value": -0.008687,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema10_1h) / ema10_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 10-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 76,
    "name": "feat_ma_1h_ema_ratio_25",
    "value": -0.017411,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema25_1h) / ema25_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 25-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 77,
    "name": "feat_ma_1h_ema_ratio_50",
    "value": -0.0232,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema50_1h) / ema50_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 50-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 78,
    "name": "feat_ma_1h_ema_ratio_100",
    "value": -0.031194,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(close - ema100_1h) / ema100_1h",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Ratio of 5m close to completed 1h 100-period EMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 79,
    "name": "feat_ma_1h_sma_slope_5",
    "value": -0.018112,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma5_{1h,t} / sma5_{1h,t-5} - 1",
    "unit": "fraction",
    "window": "1H completed (12 bars)",
    "description": "5-hour slope of completed 1h 5-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 80,
    "name": "feat_ma_1h_sma_slope_10",
    "value": -0.010789,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma10_{1h,t} / sma10_{1h,t-5} - 1",
    "unit": "fraction",
    "window": "1H completed (12 bars)",
    "description": "5-hour slope of completed 1h 10-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 81,
    "name": "feat_ma_1h_sma_slope_25",
    "value": -0.004721,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma25_{1h,t} / sma25_{1h,t-5} - 1",
    "unit": "fraction",
    "window": "1H completed (12 bars)",
    "description": "5-hour slope of completed 1h 25-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 82,
    "name": "feat_ma_1h_sma_slope_50",
    "value": -0.002891,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma50_{1h,t} / sma50_{1h,t-5} - 1",
    "unit": "fraction",
    "window": "1H completed (12 bars)",
    "description": "5-hour slope of completed 1h 50-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 83,
    "name": "feat_ma_1h_sma_slope_100",
    "value": -0.00258,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "sma100_{1h,t} / sma100_{1h,t-5} - 1",
    "unit": "fraction",
    "window": "1H completed (12 bars)",
    "description": "5-hour slope of completed 1h 100-period SMA",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 84,
    "name": "feat_ma_alignment_sma_1h",
    "value": -1.0,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(2*sum(short>long) - 10) / 10",
    "unit": "score in [-1, 1]",
    "window": "1H completed (12 bars)",
    "description": "Pairwise alignment score of completed 1h SMAs",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 85,
    "name": "feat_ma_alignment_ema_1h",
    "value": -1.0,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "(2*sum(short>long) - 10) / 10",
    "unit": "score in [-1, 1]",
    "window": "1H completed (12 bars)",
    "description": "Pairwise alignment score of completed 1h EMAs",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 86,
    "name": "feat_ma_1h_natr_14",
    "value": 0.007,
    "group": "ma_structure",
    "category": "Moving Average Structure",
    "formula": "atr14_1h / close",
    "unit": "ratio",
    "window": "1H completed (12 bars)",
    "description": "Completed 1h 14-period Wilder ATR normalized by 5m close",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 87,
    "name": "feat_struct_dist_support_atr",
    "value": 2.301799,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "(close - S) / ATR",
    "unit": "ATR multiple",
    "window": "1 bar (5 mins)",
    "description": "Distance from close to nearest support in 5m ATR units",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 88,
    "name": "feat_struct_dist_support_pct",
    "value": 0.0059,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "(close - S) / close",
    "unit": "fraction",
    "window": "1 bar (5 mins)",
    "description": "Fractional distance from close to nearest support",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 89,
    "name": "feat_struct_dist_resistance_atr",
    "value": 1.673093,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "(R - close) / ATR",
    "unit": "ATR multiple",
    "window": "1 bar (5 mins)",
    "description": "Distance from close to nearest resistance in 5m ATR units",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 90,
    "name": "feat_struct_dist_resistance_pct",
    "value": 0.004288,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "(R - close) / close",
    "unit": "fraction",
    "window": "1 bar (5 mins)",
    "description": "Fractional distance from close to nearest resistance",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 91,
    "name": "feat_struct_support_strength",
    "value": 1.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "level.touches",
    "unit": "count",
    "window": "1 bar (5 mins)",
    "description": "Touch count of nearest active support level",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 92,
    "name": "feat_struct_resistance_strength",
    "value": 2.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "level.touches",
    "unit": "count",
    "window": "1 bar (5 mins)",
    "description": "Touch count of nearest active resistance level",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 93,
    "name": "feat_struct_trend_up",
    "value": 0.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "1.0 if trend == Up else 0.0",
    "unit": "binary {0, 1}",
    "window": "1 bar (5 mins)",
    "description": "Dow Theory uptrend indicator (HH and HL persist)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 94,
    "name": "feat_struct_trend_down",
    "value": 1.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "1.0 if trend == Down else 0.0",
    "unit": "binary {0, 1}",
    "window": "1 bar (5 mins)",
    "description": "Dow Theory downtrend indicator (LH and LL persist)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 95,
    "name": "feat_struct_trend_range",
    "value": 0.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "1.0 if trend == Range else 0.0",
    "unit": "binary {0, 1}",
    "window": "1 bar (5 mins)",
    "description": "Dow Theory sideways/range market indicator",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 96,
    "name": "feat_struct_trend_age",
    "value": 59.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "t - trend_start",
    "unit": "count",
    "window": "1 bar (5 mins)",
    "description": "Candles elapsed since current Dow trend state began",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 97,
    "name": "feat_struct_pullback_depth_atr",
    "value": 3.637431,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "(peak - close) / ATR or (close - trough) / ATR",
    "unit": "ATR multiple",
    "window": "1 bar (5 mins)",
    "description": "Pullback depth from trend extreme in ATR units",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 98,
    "name": "feat_struct_bars_since_break",
    "value": 0.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "bars since break event",
    "unit": "count",
    "window": "1 bar (5 mins)",
    "description": "Candles elapsed since last support/resistance level break",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 99,
    "name": "feat_struct_bars_since_retest",
    "value": 39.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "bars since retest event",
    "unit": "count",
    "window": "1 bar (5 mins)",
    "description": "Candles elapsed since last flipped level retest",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 100,
    "name": "feat_struct_retest_direction",
    "value": 0.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "retest direction",
    "unit": "trinary {-1, 0, 1}",
    "window": "1 bar (5 mins)",
    "description": "Direction of last flipped level retest (+1 bullish, -1 bearish, 0 neutral)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 101,
    "name": "feat_struct_rule_vote",
    "value": 0.0,
    "group": "structure",
    "category": "Causal Market Structure",
    "formula": "Section 9.2 rule vote",
    "unit": "trinary {-1, 0, 1}",
    "window": "1 bar (5 mins)",
    "description": "Rule-based 4th voter signal per decisions table (+1, -1, 0)",
    "role": "Horizon 2 Feature",
    "is_tree": true,
    "is_linear": false,
    "is_meta": false,
    "models": [
      "RF/GB"
    ]
  },
  {
    "id": 102,
    "name": "bin_lr_p_up",
    "value": 0.4987,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(y = +1 | X_linear, Logistic Regression)",
    "unit": "probability [0, 1]",
    "window": "Trained Logistic Regression on Horizon-2 decorrelated features",
    "description": "Binary Logistic Regression directional probability of UP forecast",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 103,
    "name": "bin_rf_p_up",
    "value": 0.5076,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(y = +1 | X, Random Forest)",
    "unit": "probability [0, 1]",
    "window": "Trained 100-tree RF ensemble for Horizon 2",
    "description": "Binary Random Forest directional probability of UP forecast",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 104,
    "name": "bin_gb_p_up",
    "value": 0.5072,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(y = +1 | X, Gradient Boosting)",
    "unit": "probability [0, 1]",
    "window": "Trained Gradient Boosting ensemble for Horizon 2",
    "description": "Binary Gradient Boosting directional probability of UP forecast",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 105,
    "name": "tri_lr_p_up",
    "value": 0.3855,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(class = LONG | X_linear, Multiclass LR)",
    "unit": "probability [0, 1]",
    "window": "Trained 3-class Logistic Regression for Horizon 2",
    "description": "Three-class Logistic Regression probability of LONG outcome",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 106,
    "name": "tri_lr_p_down",
    "value": 0.3859,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(class = SHORT | X_linear, Multiclass LR)",
    "unit": "probability [0, 1]",
    "window": "Trained 3-class Logistic Regression for Horizon 2",
    "description": "Three-class Logistic Regression probability of SHORT outcome",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 107,
    "name": "tri_rf_p_up",
    "value": 0.3831,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(class = LONG | X, Multiclass RF)",
    "unit": "probability [0, 1]",
    "window": "Trained 3-class Random Forest for Horizon 2",
    "description": "Three-class Random Forest probability of LONG outcome",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 108,
    "name": "tri_rf_p_down",
    "value": 0.3817,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(class = SHORT | X, Multiclass RF)",
    "unit": "probability [0, 1]",
    "window": "Trained 3-class Random Forest for Horizon 2",
    "description": "Three-class Random Forest probability of SHORT outcome",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 109,
    "name": "tri_gb_p_up",
    "value": 0.3981,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(class = LONG | X, Multiclass GB)",
    "unit": "probability [0, 1]",
    "window": "Trained 3-class Gradient Boosting for Horizon 2",
    "description": "Three-class Gradient Boosting probability of LONG outcome",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 110,
    "name": "tri_gb_p_down",
    "value": 0.3583,
    "group": "meta_probs",
    "category": "Model Probabilities",
    "formula": "P(class = SHORT | X, Multiclass GB)",
    "unit": "probability [0, 1]",
    "window": "Trained 3-class Gradient Boosting for Horizon 2",
    "description": "Three-class Gradient Boosting probability of SHORT outcome",
    "role": "Meta Stacking Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 111,
    "name": "ta_rule_vote",
    "value": 0.0,
    "group": "meta_ta",
    "category": "Technical Analysis Engine",
    "formula": "Composite rule voter signal in {-1, 0, 1}",
    "unit": "trinary {-1, 0, 1}",
    "window": "Current candle",
    "description": "Direct rule-based trend vote feeding Meta-Model Combiner",
    "role": "Meta TA Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 112,
    "name": "ta_loc",
    "value": -0.6229,
    "group": "meta_ta",
    "category": "Technical Analysis Engine",
    "formula": "(close - min(L_288)) / (max(H_288) - min(L_288)) * 2 - 1",
    "unit": "score in [-1, 1]",
    "window": "288 bars (24 hours)",
    "description": "Position within 24-hour channel normalized to [-1, 1] feeding Meta-Model Combiner",
    "role": "Meta TA Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 113,
    "name": "ta_trend",
    "value": -1.0,
    "group": "meta_ta",
    "category": "Technical Analysis Engine",
    "formula": "Dow Theory direction: +1 Up, -1 Down, 0 Range",
    "unit": "trinary {-1, 0, 1}",
    "window": "Current Dow swing structure",
    "description": "Dow Theory directional trend state feeding Meta-Model Combiner",
    "role": "Meta TA Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 114,
    "name": "ta_ema50",
    "value": -0.04178,
    "group": "meta_ta",
    "category": "Technical Analysis Engine",
    "formula": "(close - EMA50) / EMA50",
    "unit": "ratio",
    "window": "50 bars (~4.2 hours)",
    "description": "Distance to 50-period EMA feeding Meta-Model Combiner",
    "role": "Meta TA Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 115,
    "name": "ta_rsi",
    "value": 0.0759,
    "group": "meta_ta",
    "category": "Technical Analysis Engine",
    "formula": "RSI(14) / 50.0 - 1.0 in [-1, 1]",
    "unit": "score in [-1, 1]",
    "window": "14 bars (70 mins)",
    "description": "RSI-14 scaled to [-1, 1] feeding Meta-Model Combiner",
    "role": "Meta TA Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  },
  {
    "id": 116,
    "name": "ta_align",
    "value": -0.4,
    "group": "meta_ta",
    "category": "Technical Analysis Engine",
    "formula": "Pairwise alignment of 5m EMAs in [-1, 1]",
    "unit": "score in [-1, 1]",
    "window": "5m multi-span EMAs",
    "description": "Pairwise EMA alignment score feeding Meta-Model Combiner",
    "role": "Meta TA Input",
    "is_tree": false,
    "is_linear": false,
    "is_meta": true,
    "models": [
      "META"
    ]
  }
];

const LATEST_CALCULATED_FEATURES = {
  "feat_ret_1": -0.001659,
  "feat_ret_3": 0.001636,
  "feat_ret_6": 0.002695,
  "feat_ret_12": 0.001191,
  "feat_ret_24": 0.007724,
  "feat_vol_sigma_288": 0.001309,
  "feat_vol_sigma_60": 0.002178,
  "feat_vol_sigma_12": 0.001699,
  "feat_vol_ratio_60_288": 1.664387,
  "feat_vol_ratio_12_60": 0.779909,
  "feat_vol_natr_14": 0.002563,
  "feat_vol_bb_pct_20": 0.681048,
  "feat_vol_bb_width_20": 0.007619,
  "feat_vol_range_12": 0.007666,
  "feat_shape_body_pct": 0.00166,
  "feat_shape_body_to_range": 0.830067,
  "feat_shape_upper_wick_pct": 0.000106,
  "feat_shape_lower_wick_pct": 0.000234,
  "feat_shape_close_pos": 0.117111,
  "feat_shape_candle_dir": -1.0,
  "feat_mom_sma_ratio_10": 0.001578,
  "feat_mom_sma_ratio_20": 0.001379,
  "feat_mom_sma_ratio_50": 0.000597,
  "feat_mom_ema_ratio_10": 0.001148,
  "feat_mom_ema_ratio_20": 0.001634,
  "feat_mom_ema_ratio_50": -0.000786,
  "feat_mom_sma_ratio_10_50": -0.00098,
  "feat_mom_rsi_7": 0.586245,
  "feat_mom_rsi_14": 0.537961,
  "feat_mom_macd_line": 0.000166,
  "feat_mom_macd_signal": -0.000369,
  "feat_mom_macd_hist": 0.000534,
  "feat_mom_adx_14": 0.184342,
  "feat_volume_log": 4.179426,
  "feat_volume_zscore_60": -0.610256,
  "feat_volume_zscore_12": -0.37954,
  "feat_volume_change_1": -0.257578,
  "feat_volume_ratio_12_60": 0.622771,
  "feat_of_taker_buy_ratio": 0.505491,
  "feat_of_taker_buy_zscore_60": 0.07556,
  "feat_of_trades_zscore_60": -1.035405,
  "feat_of_trades_zscore_12": -1.500048,
  "feat_of_avg_trade_size": 0.005112,
  "feat_of_avg_trade_size_zscore_60": 0.472657,
  "feat_time_sin_hour": -0.965926,
  "feat_time_cos_hour": -0.258819,
  "feat_time_sin_weekday": 0.433884,
  "feat_time_cos_weekday": -0.900969,
  "feat_mtf_1h_ret": 0.000626,
  "feat_mtf_1h_range": 0.01007,
  "feat_mtf_1h_vol_ratio": 1.585974,
  "feat_mtf_1h_close_vs_sma_6": -0.006809,
  "feat_mtf_1h_rsi_6": 0.117759,
  "feat_vol_parkinson_12": 0.001388,
  "feat_vol_parkinson_60": 0.001993,
  "feat_vol_parkinson_288": 0.001228,
  "feat_vol_gk_12": 0.001282,
  "feat_vol_gk_60": 0.001919,
  "feat_vol_gk_288": 0.001194,
  "feat_vol_realized_abs_sum_6": 0.00659,
  "feat_vol_realized_abs_sum_12": 0.016648,
  "feat_vol_realized_abs_sum_36": 0.063896,
  "feat_vol_realized_max_abs_12": 0.002723,
  "feat_vol_realized_max_abs_60": 0.008139,
  "feat_vol_realized_semi_var_up_ratio_60": 0.324211,
  "feat_vol_seasonal_slot_abs_ret_7d": 0.000766,
  "feat_vol_seasonal_vol_ratio_12": 1.810307,
  "feat_vol_seasonal_volume_ratio_7d": 1.766167,
  "feat_vol_seasonal_trades_ratio_7d": 1.040562,
  "feat_vol_har_rv_72": 0.002093,
  "feat_vol_har_rv_2016": 0.001239,
  "feat_vol_har_ratio_12_288": 1.298071,
  "feat_vol_har_ratio_288_2016": 1.056686,
  "feat_flow_window_net_6": 0.082921,
  "feat_flow_window_net_12": 0.014901,
  "feat_flow_window_net_36": 0.018962,
  "feat_flow_window_signed_flow_6": -0.082921,
  "feat_flow_window_signed_flow_12": -0.014901,
  "feat_flow_window_signed_flow_36": -0.018962,
  "feat_loc_position_288": 0.188546,
  "feat_loc_dist_high_288": 0.040123,
  "feat_loc_dist_low_288": 0.009323,
  "feat_time_is_weekend": 0.0,
  "feat_time_sin_week": -0.204252,
  "feat_time_cos_week": -0.978918,
  "feat_ma_sma_ratio_5": 0.000718,
  "feat_ma_sma_ratio_25": 0.00251,
  "feat_ma_sma_ratio_100": -0.008398,
  "feat_ma_ema_ratio_5": 0.000426,
  "feat_ma_ema_ratio_25": 0.001533,
  "feat_ma_ema_ratio_100": -0.006332,
  "feat_ma_sma_slope_5": 0.001721,
  "feat_ma_sma_slope_10": 0.000615,
  "feat_ma_sma_slope_25": 0.001188,
  "feat_ma_sma_slope_50": -0.001012,
  "feat_ma_sma_slope_100": -0.001113,
  "feat_ma_ema_slope_5": 0.002187,
  "feat_ma_ema_slope_10": 0.001342,
  "feat_ma_ema_slope_25": 0.000472,
  "feat_ma_ema_slope_50": -0.000316,
  "feat_ma_ema_slope_100": -0.000737,
  "feat_ma_alignment_sma_5m": -0.4,
  "feat_ma_alignment_ema_5m": -0.4,
  "feat_ma_1h_sma_ratio_5": -0.002485,
  "feat_ma_1h_sma_ratio_10": -0.011601,
  "feat_ma_1h_sma_ratio_25": -0.022037,
  "feat_ma_1h_sma_ratio_50": -0.022979,
  "feat_ma_1h_sma_ratio_100": -0.031449,
  "feat_ma_1h_ema_ratio_5": -0.002272,
  "feat_ma_1h_ema_ratio_10": -0.008687,
  "feat_ma_1h_ema_ratio_25": -0.017411,
  "feat_ma_1h_ema_ratio_50": -0.0232,
  "feat_ma_1h_ema_ratio_100": -0.031194,
  "feat_ma_1h_sma_slope_5": -0.018112,
  "feat_ma_1h_sma_slope_10": -0.010789,
  "feat_ma_1h_sma_slope_25": -0.004721,
  "feat_ma_1h_sma_slope_50": -0.002891,
  "feat_ma_1h_sma_slope_100": -0.00258,
  "feat_ma_alignment_sma_1h": -1.0,
  "feat_ma_alignment_ema_1h": -1.0,
  "feat_ma_1h_natr_14": 0.007,
  "feat_struct_dist_support_atr": 2.301799,
  "feat_struct_dist_support_pct": 0.0059,
  "feat_struct_dist_resistance_atr": 1.673093,
  "feat_struct_dist_resistance_pct": 0.004288,
  "feat_struct_support_strength": 1.0,
  "feat_struct_resistance_strength": 2.0,
  "feat_struct_trend_up": 0.0,
  "feat_struct_trend_down": 1.0,
  "feat_struct_trend_range": 0.0,
  "feat_struct_trend_age": 59.0,
  "feat_struct_pullback_depth_atr": 3.637431,
  "feat_struct_bars_since_break": 0.0,
  "feat_struct_bars_since_retest": 39.0,
  "feat_struct_retest_direction": 0.0,
  "feat_struct_rule_vote": 0.0,
  "feat_ret_2": 0.000634,
  "bin_lr_p_up": 0.4987,
  "bin_rf_p_up": 0.5076,
  "bin_gb_p_up": 0.5072,
  "tri_lr_p_up": 0.3855,
  "tri_lr_p_down": 0.3859,
  "tri_rf_p_up": 0.3831,
  "tri_rf_p_down": 0.3817,
  "tri_gb_p_up": 0.3981,
  "tri_gb_p_down": 0.3583,
  "ta_rule_vote": 0.0,
  "ta_loc": -0.6229,
  "ta_trend": -1.0,
  "ta_ema50": -0.04178,
  "ta_rsi": 0.0759,
  "ta_align": -0.4
};

if (typeof window !== "undefined") {
  window.PIPELINE_STAGES = PIPELINE_STAGES;
  window.FEATURES_CATALOG = FEATURES_CATALOG;
  window.LATEST_CALCULATED_FEATURES = LATEST_CALCULATED_FEATURES;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { PIPELINE_STAGES, FEATURES_CATALOG, LATEST_CALCULATED_FEATURES };
}
