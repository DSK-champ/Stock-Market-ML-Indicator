# BTC/USDT 5-Minute Causal Feature Engineering & Calculation Manual

> **Document Scope:** Exhaustive calculation reference manual for all 136 causal machine learning features engineered for 5-minute Bitcoin (`BTCUSDT`) spot trading.
> **Anti-Leakage Guarantee:** Every feature at timestamp $t$ is strictly causal, computed exclusively from closed candles $\le t$. Zero future leakage ($t+1$ or later) across all 136 features.
> **Scale Invariance:** Zero raw USD price levels are exposed to models; all inputs are transformed into unitless stationary log-returns, moving average ratios, Z-scores, ATR-normalized distances, or bounded cyclical coordinates.

---

## Table of Contents

1. [Returns & Price Differentials (Group: `returns`, 5 Features)](#returns)
2. [Volatility, Rolling Sigmas & Envelopes (Group: `volatility`, 9 Features)](#volatility)
3. [Candlestick Anatomy & Price Distribution (Group: `candle_shape`, 6 Features)](#candle_shape)
4. [Momentum & Trend Oscillators (Group: `momentum`, 13 Features)](#momentum)
5. [Trading Volume Dynamics (Group: `volume`, 5 Features)](#volume)
6. [Microstructure & Order Flow (Group: `order_flow`, 6 Features)](#order_flow)
7. [Intraday & Weekly Cyclical Time (Group: `time`, 4 Features)](#time)
8. [Multi-Timeframe 1-Hour Aggregations (Group: `mtf_1h`, 5 Features)](#mtf_1h)
9. [Extreme Value Range Volatility (Parkinson & Garman-Klass) (Group: `vol_range`, 6 Features)](#vol_range)
10. [Realized Volatility & Jump Dynamics (Group: `vol_realized`, 6 Features)](#vol_realized)
11. [Diurnal Seasonal Volatility & Baseline Ratios (Group: `vol_seasonal`, 4 Features)](#vol_seasonal)
12. [Heterogeneous Autoregressive Volatility (HAR) (Group: `vol_har`, 4 Features)](#vol_har)
13. [Windowed Cumulative Order Flow & Signed Flow (Group: `flow_window`, 6 Features)](#flow_window)
14. [Price Location Relative to 24-Hour Range (Group: `location`, 3 Features)](#location)
15. [Weekend Regimes & Weekly Cyclical Coordinates (Group: `time_weekend`, 3 Features)](#time_weekend)
16. [Moving Average Term Structure & Trend Alignment (Group: `ma_structure`, 36 Features)](#ma_structure)
17. [Market Structure, Support/Resistance & Dow Theory (Group: `structure`, 15 Features)](#structure)
18. [Summary Taxonomy & Model Deployment Scope](#summary-taxonomy)

---

<a id="returns"></a>
## Returns & Price Differentials

- **Group Key:** `returns`
- **Feature Count:** 5
- **Raw Inputs Required:** `close`
- **Overview:** Logarithmic price returns across multi-scale lookback horizons.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **`feat_ret_1`** | `ln(close[t] / close[t-1])` | fraction | Log return over 1 candle (5m) |
| 2 | **`feat_ret_3`** | `ln(close[t] / close[t-3])` | fraction | Log return over 3 candles (15m) |
| 3 | **`feat_ret_6`** | `ln(close[t] / close[t-6])` | fraction | Log return over 6 candles (30m) |
| 4 | **`feat_ret_12`** | `ln(close[t] / close[t-12])` | fraction | Log return over 12 candles (60m / 1h) |
| 5 | **`feat_ret_24`** | `ln(close[t] / close[t-24])` | fraction | Log return over 24 candles (120m / 2h) |

### Detailed Calculation Breakdowns

#### `feat_ret_1`
- **Description:** Log return over 1 candle (5m)
- **Mathematical Formula:** `ln(close[t] / close[t-1])`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes candle close at current bar $t$.
  2. Computes the natural logarithm of current close divided by the closed price 1 bars ago ($t - 1$, representing 5 minutes prior).
  3. Formula: `ln(close[t] / close[t - 1])`. Guarantees exact stationarity with zero USD level dependency.

#### `feat_ret_3`
- **Description:** Log return over 3 candles (15m)
- **Mathematical Formula:** `ln(close[t] / close[t-3])`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes candle close at current bar $t$.
  2. Computes the natural logarithm of current close divided by the closed price 3 bars ago ($t - 3$, representing 15 minutes prior).
  3. Formula: `ln(close[t] / close[t - 3])`. Guarantees exact stationarity with zero USD level dependency.

#### `feat_ret_6`
- **Description:** Log return over 6 candles (30m)
- **Mathematical Formula:** `ln(close[t] / close[t-6])`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes candle close at current bar $t$.
  2. Computes the natural logarithm of current close divided by the closed price 6 bars ago ($t - 6$, representing 30 minutes prior).
  3. Formula: `ln(close[t] / close[t - 6])`. Guarantees exact stationarity with zero USD level dependency.

#### `feat_ret_12`
- **Description:** Log return over 12 candles (60m / 1h)
- **Mathematical Formula:** `ln(close[t] / close[t-12])`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes candle close at current bar $t$.
  2. Computes the natural logarithm of current close divided by the closed price 12 bars ago ($t - 12$, representing 60 minutes prior).
  3. Formula: `ln(close[t] / close[t - 12])`. Guarantees exact stationarity with zero USD level dependency.

#### `feat_ret_24`
- **Description:** Log return over 24 candles (120m / 2h)
- **Mathematical Formula:** `ln(close[t] / close[t-24])`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes candle close at current bar $t$.
  2. Computes the natural logarithm of current close divided by the closed price 24 bars ago ($t - 24$, representing 120 minutes prior).
  3. Formula: `ln(close[t] / close[t - 24])`. Guarantees exact stationarity with zero USD level dependency.

---

<a id="volatility"></a>
## Volatility, Rolling Sigmas & Envelopes

- **Group Key:** `volatility`
- **Feature Count:** 9
- **Raw Inputs Required:** `high, low, close`
- **Overview:** Causal rolling standard deviations, normalized ATR, Bollinger Bands, and high-low range measures.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 6 | **`feat_vol_sigma_288`** | `rolling_sigma(ret_1, 288)` | fraction | Causal rolling return std over 288 candles (24h) |
| 7 | **`feat_vol_sigma_60`** | `rolling_sigma(ret_1, 60)` | fraction | Causal rolling return std over 60 candles (5h) |
| 8 | **`feat_vol_sigma_12`** | `rolling_sigma(ret_1, 12)` | fraction | Causal rolling return std over 12 candles (1h) |
| 9 | **`feat_vol_ratio_60_288`** | `sigma_60 / sigma_288` | ratio | Volatility ratio: 5h sigma / 24h sigma |
| 10 | **`feat_vol_ratio_12_60`** | `sigma_12 / sigma_60` | ratio | Volatility ratio: 1h sigma / 5h sigma |
| 11 | **`feat_vol_natr_14`** | `ATR(14) / close` | ratio | Normalized Average True Range over 14 candles |
| 12 | **`feat_vol_bb_pct_20`** | `(close - lower) / (upper - lower)` | ratio in [0, 1] | Bollinger Bands %B (20-period, 2 sigma) |
| 13 | **`feat_vol_bb_width_20`** | `(upper - lower) / SMA(20)` | ratio | Bollinger Bands bandwidth normalized by SMA |
| 14 | **`feat_vol_range_12`** | `(max(H_12) - min(L_12)) / close` | ratio | Normalized 12-candle high-low range |

### Detailed Calculation Breakdowns

#### `feat_vol_sigma_288`
- **Description:** Causal rolling return std over 288 candles (24h)
- **Mathematical Formula:** `rolling_sigma(ret_1, 288)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates 1-candle 5m log returns $r_t = \ln(C_t / C_{t-1})$.
  2. Computes causal sample standard deviation over a rolling window of 288 candles (24.0 hours).
  3. Uses `min_periods=2` and strictly closed bars to prevent any future boundary leakage.

#### `feat_vol_sigma_60`
- **Description:** Causal rolling return std over 60 candles (5h)
- **Mathematical Formula:** `rolling_sigma(ret_1, 60)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates 1-candle 5m log returns $r_t = \ln(C_t / C_{t-1})$.
  2. Computes causal sample standard deviation over a rolling window of 60 candles (5.0 hours).
  3. Uses `min_periods=2` and strictly closed bars to prevent any future boundary leakage.

#### `feat_vol_sigma_12`
- **Description:** Causal rolling return std over 12 candles (1h)
- **Mathematical Formula:** `rolling_sigma(ret_1, 12)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates 1-candle 5m log returns $r_t = \ln(C_t / C_{t-1})$.
  2. Computes causal sample standard deviation over a rolling window of 12 candles (1.0 hours).
  3. Uses `min_periods=2` and strictly closed bars to prevent any future boundary leakage.

#### `feat_vol_ratio_60_288`
- **Description:** Volatility ratio: 5h sigma / 24h sigma
- **Mathematical Formula:** `sigma_60 / sigma_288`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates rolling volatility $\sigma_{60}$ and $\sigma_{288}$.
  2. Divides shorter-window volatility by longer-window volatility: `sigma_60 / sigma_288`.
  3. Ratios > 1.0 indicate expanding volatility compression; ratios < 1.0 indicate contracting volatility regimes.

#### `feat_vol_ratio_12_60`
- **Description:** Volatility ratio: 1h sigma / 5h sigma
- **Mathematical Formula:** `sigma_12 / sigma_60`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates rolling volatility $\sigma_{12}$ and $\sigma_{60}$.
  2. Divides shorter-window volatility by longer-window volatility: `sigma_12 / sigma_60`.
  3. Ratios > 1.0 indicate expanding volatility compression; ratios < 1.0 indicate contracting volatility regimes.

#### `feat_vol_natr_14`
- **Description:** Normalized Average True Range over 14 candles
- **Mathematical Formula:** `ATR(14) / close`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes True Range: $TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$.
  2. Applies Wilder's 14-period recursive RMA (alpha = 1/14).
  3. Normalizes by dividing ATR by candle close $C_t$: `ATR_14 / close`. Scale invariant.

#### `feat_vol_bb_pct_20`
- **Description:** Bollinger Bands %B (20-period, 2 sigma)
- **Mathematical Formula:** `(close - lower) / (upper - lower)`
- **Unit / Range:** `ratio in [0, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 20-period rolling close SMA $\mu$ and standard deviation $\sigma$.
  2. Forms Upper Band $(\mu + 2\sigma)$ and Lower Band $(\mu - 2\sigma)$.
  3. Computes %B oscillator: `(close - lower_band) / (upper_band - lower_band)`. Values near 1.0 indicate upper envelope contact; values near 0.0 indicate lower envelope contact.

#### `feat_vol_bb_width_20`
- **Description:** Bollinger Bands bandwidth normalized by SMA
- **Mathematical Formula:** `(upper - lower) / SMA(20)`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 20-period Bollinger upper and lower envelope difference: $Band = Upper - Lower = 4\sigma$.
  2. Normalizes bandwidth by 20-period moving average: `Band / SMA_20`. Measures relative volatility expansion/squeeze.

#### `feat_vol_range_12`
- **Description:** Normalized 12-candle high-low range
- **Mathematical Formula:** `(max(H_12) - min(L_12)) / close`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes highest high over last 12 candles (1 hour) and lowest low over last 12 candles.
  2. Computes total span: $High_{12} - Low_{12}$.
  3. Normalizes by dividing by current close: `(max_high_12 - min_low_12) / close`.

---

<a id="candle_shape"></a>
## Candlestick Anatomy & Price Distribution

- **Group Key:** `candle_shape`
- **Feature Count:** 6
- **Raw Inputs Required:** `open, high, low, close`
- **Overview:** Intra-candle structural geometries including body percentages, wick lengths, close position, and polarity.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 15 | **`feat_shape_body_pct`** | `\|close - open\| / close` | fraction | Candle real body size relative to close price |
| 16 | **`feat_shape_body_to_range`** | `\|close - open\| / (high - low)` | ratio | Candle body size relative to full range [0, 1] |
| 17 | **`feat_shape_upper_wick_pct`** | `(high - max(open, close)) / close` | fraction | Upper wick length relative to close price |
| 18 | **`feat_shape_lower_wick_pct`** | `(min(open, close) - low) / close` | fraction | Lower wick length relative to close price |
| 19 | **`feat_shape_close_pos`** | `(close - low) / (high - low)` | ratio | Close position within candle range [0, 1] |
| 20 | **`feat_shape_candle_dir`** | `sign(close - open)` | integer {-1,0,1} | Directional sign of candle body (+1 bullish, -1 bearish, 0 doji) |

### Detailed Calculation Breakdowns

#### `feat_shape_body_pct`
- **Description:** Candle real body size relative to close price
- **Mathematical Formula:** `|close - open| / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates absolute candle body height: $|C_t - O_t|$.
  2. Divides by candle close: `abs(close - open) / close`. Measures pure body displacement percentage.

#### `feat_shape_body_to_range`
- **Description:** Candle body size relative to full range [0, 1]
- **Mathematical Formula:** `|close - open| / (high - low)`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates candle body $|C_t - O_t|$ and full high-low span $(H_t - L_t)$.
  2. Computes ratio: `abs(close - open) / (high - low)`. 1.0 represents full marubozu; 0.0 represents a doji.

#### `feat_shape_upper_wick_pct`
- **Description:** Upper wick length relative to close price
- **Mathematical Formula:** `(high - max(open, close)) / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Identifies top of the candle body: $\max(C_t, O_t)$.
  2. Calculates upper shadow: $H_t - \max(C_t, O_t)$.
  3. Normalizes by dividing by close: `(high - max(close, open)) / close`.

#### `feat_shape_lower_wick_pct`
- **Description:** Lower wick length relative to close price
- **Mathematical Formula:** `(min(open, close) - low) / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Identifies bottom of the candle body: $\min(C_t, O_t)$.
  2. Calculates lower shadow: $\min(C_t, O_t) - L_t$.
  3. Normalizes by dividing by close: `(min(close, open) - low) / close`.

#### `feat_shape_close_pos`
- **Description:** Close position within candle range [0, 1]
- **Mathematical Formula:** `(close - low) / (high - low)`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates where close sits within the candle's high-low span: `(close - low) / (high - low)`.
  2. Bounded in [0.0, 1.0]. 1.0 = close at candle high; 0.0 = close at candle low; 0.5 = midpoint.

#### `feat_shape_candle_dir`
- **Description:** Directional sign of candle body (+1 bullish, -1 bearish, 0 doji)
- **Mathematical Formula:** `sign(close - open)`
- **Unit / Range:** `integer {-1,0,1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates direction of candle: `sign(close - open)`.
  2. Returns +1.0 for bullish green candle, -1.0 for bearish red candle, 0.0 for exact flat doji.

---

<a id="momentum"></a>
## Momentum & Trend Oscillators

- **Group Key:** `momentum`
- **Feature Count:** 13
- **Raw Inputs Required:** `close, high, low`
- **Overview:** Moving average ratios, dual-period RSI, MACD components, and Average Directional Index (ADX).

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 21 | **`feat_mom_sma_ratio_10`** | `close / SMA(10) - 1.0` | ratio | Distance of close from 10-candle SMA |
| 22 | **`feat_mom_sma_ratio_20`** | `close / SMA(20) - 1.0` | ratio | Distance of close from 20-candle SMA |
| 23 | **`feat_mom_sma_ratio_50`** | `close / SMA(50) - 1.0` | ratio | Distance of close from 50-candle SMA |
| 24 | **`feat_mom_ema_ratio_10`** | `close / EMA(10) - 1.0` | ratio | Distance of close from 10-candle EMA |
| 25 | **`feat_mom_ema_ratio_20`** | `close / EMA(20) - 1.0` | ratio | Distance of close from 20-candle EMA |
| 26 | **`feat_mom_ema_ratio_50`** | `close / EMA(50) - 1.0` | ratio | Distance of close from 50-candle EMA |
| 27 | **`feat_mom_sma_ratio_10_50`** | `SMA(10) / SMA(50) - 1.0` | ratio | Spread between short and medium SMA |
| 28 | **`feat_mom_rsi_7`** | `RSI(7) / 100.0` | oscillator | Relative Strength Index over 7 candles [0, 1] |
| 29 | **`feat_mom_rsi_14`** | `RSI(14) / 100.0` | oscillator | Relative Strength Index over 14 candles [0, 1] |
| 30 | **`feat_mom_macd_line`** | `(EMA(12) - EMA(26)) / close` | ratio | MACD line normalized by close price |
| 31 | **`feat_mom_macd_signal`** | `EMA(macd_line, 9) / close` | ratio | MACD 9-period signal line normalized by close |
| 32 | **`feat_mom_macd_hist`** | `(macd_line - macd_signal) / close` | ratio | MACD histogram normalized by close |
| 33 | **`feat_mom_adx_14`** | `ADX(14) / 100.0` | oscillator | Average Directional Index over 14 candles [0, 1] |

### Detailed Calculation Breakdowns

#### `feat_mom_sma_ratio_10`
- **Description:** Distance of close from 10-candle SMA
- **Mathematical Formula:** `close / SMA(10) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 10-period causal SMA of close price.
  2. Calculates normalized distance ratio: `(close - SMA_10) / SMA_10`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_sma_ratio_20`
- **Description:** Distance of close from 20-candle SMA
- **Mathematical Formula:** `close / SMA(20) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 20-period causal SMA of close price.
  2. Calculates normalized distance ratio: `(close - SMA_20) / SMA_20`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_sma_ratio_50`
- **Description:** Distance of close from 50-candle SMA
- **Mathematical Formula:** `close / SMA(50) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 50-period causal SMA of close price.
  2. Calculates normalized distance ratio: `(close - SMA_50) / SMA_50`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_ema_ratio_10`
- **Description:** Distance of close from 10-candle EMA
- **Mathematical Formula:** `close / EMA(10) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 10-period causal EMA of close price.
  2. Calculates normalized distance ratio: `(close - EMA_10) / EMA_10`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_ema_ratio_20`
- **Description:** Distance of close from 20-candle EMA
- **Mathematical Formula:** `close / EMA(20) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 20-period causal EMA of close price.
  2. Calculates normalized distance ratio: `(close - EMA_20) / EMA_20`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_ema_ratio_50`
- **Description:** Distance of close from 50-candle EMA
- **Mathematical Formula:** `close / EMA(50) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 50-period causal EMA of close price.
  2. Calculates normalized distance ratio: `(close - EMA_50) / EMA_50`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_sma_ratio_10_50`
- **Description:** Spread between short and medium SMA
- **Mathematical Formula:** `SMA(10) / SMA(50) - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 50-period causal SMA of close price.
  2. Calculates normalized distance ratio: `(close - SMA_50) / SMA_50`.
  3. Expresses how far price has deviated above (+) or below (-) the moving average baseline.

#### `feat_mom_rsi_7`
- **Description:** Relative Strength Index over 7 candles [0, 1]
- **Mathematical Formula:** `RSI(7) / 100.0`
- **Unit / Range:** `oscillator`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-bar price changes $\Delta C = C_t - C_{t-1}$.
  2. Separates into positive gains $\max(\Delta C, 0)$ and losses $\max(-\Delta C, 0)$.
  3. Smooths gains and losses using Wilder's RMA over 7 periods.
  4. Computes Relative Strength $RS = AvgGain / AvgLoss$ and $RSI = 100 - (100 / (1 + RS))$.
  5. Scaled into [0.0, 1.0] by dividing by 100: `RSI / 100`.

#### `feat_mom_rsi_14`
- **Description:** Relative Strength Index over 14 candles [0, 1]
- **Mathematical Formula:** `RSI(14) / 100.0`
- **Unit / Range:** `oscillator`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-bar price changes $\Delta C = C_t - C_{t-1}$.
  2. Separates into positive gains $\max(\Delta C, 0)$ and losses $\max(-\Delta C, 0)$.
  3. Smooths gains and losses using Wilder's RMA over 14 periods.
  4. Computes Relative Strength $RS = AvgGain / AvgLoss$ and $RSI = 100 - (100 / (1 + RS))$.
  5. Scaled into [0.0, 1.0] by dividing by 100: `RSI / 100`.

#### `feat_mom_macd_line`
- **Description:** MACD line normalized by close price
- **Mathematical Formula:** `(EMA(12) - EMA(26)) / close`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes Fast EMA (12-period) and Slow EMA (26-period) on close.
  2. Calculates raw MACD difference: $EMA_{12} - EMA_{26}$.
  3. Normalizes by dividing by close: `(EMA_12 - EMA_26) / close`.

#### `feat_mom_macd_signal`
- **Description:** MACD 9-period signal line normalized by close
- **Mathematical Formula:** `EMA(macd_line, 9) / close`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes MACD line $(EMA_{12} - EMA_{26})$.
  2. Calculates 9-period EMA of the MACD line.
  3. Normalizes by dividing by close: `EMA_9(MACD_raw) / close`.

#### `feat_mom_macd_hist`
- **Description:** MACD histogram normalized by close
- **Mathematical Formula:** `(macd_line - macd_signal) / close`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes MACD Line and MACD Signal Line.
  2. Calculates MACD Histogram: $MACD_{line} - MACD_{signal}$.
  3. Normalizes by dividing by close: `(MACD_raw - MACD_signal_raw) / close`.

#### `feat_mom_adx_14`
- **Description:** Average Directional Index over 14 candles [0, 1]
- **Mathematical Formula:** `ADX(14) / 100.0`
- **Unit / Range:** `oscillator`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates directional movement: $+DM = \max(H_t - H_{t-1}, 0)$ and $-DM = \max(L_{t-1} - L_t, 0)$.
  2. Normalizes by 14-period Wilder ATR to produce $+DI_{14}$ and $-DI_{14}$.
  3. Calculates Directional Index $DX = 100 \times |+DI - -DI| / (+DI + -DI)$.
  4. Smooths DX via 14-period Wilder RMA to yield ADX.
  5. Scaled to [0.0, 1.0] by dividing by 100: `ADX / 100`.

---

<a id="volume"></a>
## Trading Volume Dynamics

- **Group Key:** `volume`
- **Feature Count:** 5
- **Raw Inputs Required:** `volume`
- **Overview:** Log-transformed volume, rolling volume Z-scores, rate-of-change, and short-to-medium moving average volume ratios.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 34 | **`feat_volume_log`** | `ln(1 + volume)` | log volume | Log volume |
| 35 | **`feat_volume_zscore_60`** | `zscore(volume, 60)` | z-score | Rolling volume Z-score over 60 candles |
| 36 | **`feat_volume_zscore_12`** | `zscore(volume, 12)` | z-score | Rolling volume Z-score over 12 candles |
| 37 | **`feat_volume_change_1`** | `volume[t] / volume[t-1] - 1.0` | ratio | 1-candle fractional volume change |
| 38 | **`feat_volume_ratio_12_60`** | `SMA(volume, 12) / SMA(volume, 60)` | ratio | Volume trend ratio: short / medium volume SMA |

### Detailed Calculation Breakdowns

#### `feat_volume_log`
- **Description:** Log volume
- **Mathematical Formula:** `ln(1 + volume)`
- **Unit / Range:** `log volume`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes raw 5-minute trading volume $V_t$.
  2. Computes natural log of 1 + volume: `ln(1 + volume)`. Compresses volume spikes into a stationary bell-like distribution.

#### `feat_volume_zscore_60`
- **Description:** Rolling volume Z-score over 60 candles
- **Mathematical Formula:** `zscore(volume, 60)`
- **Unit / Range:** `z-score`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates rolling mean $\mu_V$ and standard deviation $\sigma_V$ of trading volume over 60 candles.
  2. Computes rolling Z-score: `(volume - mean_60) / std_60`. Identifies statistical volume surges relative to the 60-candle baseline.

#### `feat_volume_zscore_12`
- **Description:** Rolling volume Z-score over 12 candles
- **Mathematical Formula:** `zscore(volume, 12)`
- **Unit / Range:** `z-score`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates rolling mean $\mu_V$ and standard deviation $\sigma_V$ of trading volume over 12 candles.
  2. Computes rolling Z-score: `(volume - mean_12) / std_12`. Identifies statistical volume surges relative to the 12-candle baseline.

#### `feat_volume_change_1`
- **Description:** 1-candle fractional volume change
- **Mathematical Formula:** `volume[t] / volume[t-1] - 1.0`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates percentage volume delta from previous 5m candle: `(volume[t] / volume[t-1]) - 1.0`. Measures sudden volume acceleration.

#### `feat_volume_ratio_12_60`
- **Description:** Volume trend ratio: short / medium volume SMA
- **Mathematical Formula:** `SMA(volume, 12) / SMA(volume, 60)`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Calculates 12-candle (1-hour) volume SMA and 60-candle (5-hour) volume SMA.
  2. Computes ratio: `SMA_vol_12 / SMA_vol_60`. Values > 1.0 indicate elevated short-term volume intensity.

---

<a id="order_flow"></a>
## Microstructure & Order Flow

- **Group Key:** `order_flow`
- **Feature Count:** 6
- **Raw Inputs Required:** `volume, taker_buy_base_volume, number_of_trades`
- **Overview:** Aggressive taker buy volume ratios, order imbalance Z-scores, and trade size dynamics.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 39 | **`feat_of_taker_buy_ratio`** | `taker_buy_base_volume / volume` | ratio | Taker buy volume fraction in [0, 1] |
| 40 | **`feat_of_taker_buy_zscore_60`** | `zscore(taker_buy_ratio, 60)` | z-score | Rolling Z-score of taker buy ratio over 60 candles |
| 41 | **`feat_of_trades_zscore_60`** | `zscore(number_of_trades, 60)` | z-score | Rolling Z-score of number of trades over 60 candles |
| 42 | **`feat_of_trades_zscore_12`** | `zscore(number_of_trades, 12)` | z-score | Rolling Z-score of number of trades over 12 candles |
| 43 | **`feat_of_avg_trade_size`** | `volume / number_of_trades` | BTC / trade | Average trade size in BTC |
| 44 | **`feat_of_avg_trade_size_zscore_60`** | `zscore(avg_trade_size, 60)` | z-score | Rolling Z-score of average trade size over 60 candles |

### Detailed Calculation Breakdowns

#### `feat_of_taker_buy_ratio`
- **Description:** Taker buy volume fraction in [0, 1]
- **Mathematical Formula:** `taker_buy_base_volume / volume`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes taker buy base volume (market buy order fills) and total candle volume.
  2. Computes ratio: `taker_buy_volume / total_volume`. Clipped to [0.0, 1.0]. Values > 0.5 indicate net aggressive buyer dominance.

#### `feat_of_taker_buy_zscore_60`
- **Description:** Rolling Z-score of taker buy ratio over 60 candles
- **Mathematical Formula:** `zscore(taker_buy_ratio, 60)`
- **Unit / Range:** `z-score`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes taker buy ratio over each 5m bar.
  2. Calculates 60-period rolling mean and standard deviation of taker buy ratio.
  3. Computes Z-score: `(taker_ratio - mean_60) / std_60`.

#### `feat_of_trades_zscore_60`
- **Description:** Rolling Z-score of number of trades over 60 candles
- **Mathematical Formula:** `zscore(number_of_trades, 60)`
- **Unit / Range:** `z-score`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes Binance trade count ($N_{trades}$) per 5m bar.
  2. Computes rolling 60-candle mean and standard deviation.
  3. Calculates rolling Z-score: `(trades - mean_60) / std_60`. Detects microstructural trade frequency bursts.

#### `feat_of_trades_zscore_12`
- **Description:** Rolling Z-score of number of trades over 12 candles
- **Mathematical Formula:** `zscore(number_of_trades, 12)`
- **Unit / Range:** `z-score`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes Binance trade count ($N_{trades}$) per 5m bar.
  2. Computes rolling 12-candle mean and standard deviation.
  3. Calculates rolling Z-score: `(trades - mean_12) / std_12`. Detects microstructural trade frequency bursts.

#### `feat_of_avg_trade_size`
- **Description:** Average trade size in BTC
- **Mathematical Formula:** `volume / number_of_trades`
- **Unit / Range:** `BTC / trade`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Divides total candle volume by number of trades: `volume / number_of_trades`.
  2. Measures average trade execution size. Larger values reflect institutional participation; smaller values reflect retail activity.

#### `feat_of_avg_trade_size_zscore_60`
- **Description:** Rolling Z-score of average trade size over 60 candles
- **Mathematical Formula:** `zscore(avg_trade_size, 60)`
- **Unit / Range:** `z-score`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates average trade size `volume / trades` per bar.
  2. Computes rolling 60-period mean and standard deviation.
  3. Calculates Z-score: `(avg_trade_size - mean_60) / std_60`.

---

<a id="time"></a>
## Intraday & Weekly Cyclical Time

- **Group Key:** `time`
- **Feature Count:** 4
- **Raw Inputs Required:** `timestamp`
- **Overview:** Harmonic sine/cosine encodings of UTC hour (24-hour diurnal cycle) and day of week (7-day cycle).

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 45 | **`feat_time_sin_hour`** | `sin(2 * pi * hour / 24)` | coordinate in [-1, 1] | Sine transform of UTC hour of day |
| 46 | **`feat_time_cos_hour`** | `cos(2 * pi * hour / 24)` | coordinate in [-1, 1] | Cosine transform of UTC hour of day |
| 47 | **`feat_time_sin_weekday`** | `sin(2 * pi * weekday / 7)` | coordinate in [-1, 1] | Sine transform of day of week (Monday=0) |
| 48 | **`feat_time_cos_weekday`** | `cos(2 * pi * weekday / 7)` | coordinate in [-1, 1] | Cosine transform of day of week (Monday=0) |

### Detailed Calculation Breakdowns

#### `feat_time_sin_hour`
- **Description:** Sine transform of UTC hour of day
- **Mathematical Formula:** `sin(2 * pi * hour / 24)`
- **Unit / Range:** `coordinate in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Extracts UTC hour $h \in [0, 23]$.
  2. Computes sine harmonic: `sin(2 * pi * hour / 24.0)`. Cyclical 24-hour diurnal encoding.

#### `feat_time_cos_hour`
- **Description:** Cosine transform of UTC hour of day
- **Mathematical Formula:** `cos(2 * pi * hour / 24)`
- **Unit / Range:** `coordinate in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Extracts UTC hour $h \in [0, 23]$.
  2. Computes cosine harmonic: `cos(2 * pi * hour / 24.0)`. Orthogonal diurnal coordinate.

#### `feat_time_sin_weekday`
- **Description:** Sine transform of day of week (Monday=0)
- **Mathematical Formula:** `sin(2 * pi * weekday / 7)`
- **Unit / Range:** `coordinate in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Extracts weekday index $w \in [0, 6]$ (Monday = 0).
  2. Computes sine harmonic: `sin(2 * pi * weekday / 7.0)`. Captures weekly periodicity.

#### `feat_time_cos_weekday`
- **Description:** Cosine transform of day of week (Monday=0)
- **Mathematical Formula:** `cos(2 * pi * weekday / 7)`
- **Unit / Range:** `coordinate in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Extracts weekday index $w \in [0, 6]$ (Monday = 0).
  2. Computes cosine harmonic: `cos(2 * pi * weekday / 7.0)`. Orthogonal weekly coordinate.

---

<a id="mtf_1h"></a>
## Multi-Timeframe 1-Hour Aggregations

- **Group Key:** `mtf_1h`
- **Feature Count:** 5
- **Raw Inputs Required:** `open, high, low, close, volume (resampled to 1h)`
- **Overview:** Features computed strictly from fully COMPLETED 1-hour candles, forward-filled with 1-hour shift for zero leakage.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 49 | **`feat_mtf_1h_ret`** | `ln(close_1h / open_1h) shifted 1h` | fraction | Log return of last COMPLETED 1h candle |
| 50 | **`feat_mtf_1h_range`** | `(high_1h - low_1h) / close_1h shifted 1h` | ratio | High-low range of last COMPLETED 1h candle normalized by close |
| 51 | **`feat_mtf_1h_vol_ratio`** | `vol_1h / SMA(vol_1h, 24) shifted 1h` | ratio | Volume of last COMPLETED 1h candle vs 24h rolling mean |
| 52 | **`feat_mtf_1h_close_vs_sma_6`** | `(close_1h - SMA(close_1h, 6)) / SMA shifted 1h` | ratio | Distance of last completed 1h close from 6h SMA |
| 53 | **`feat_mtf_1h_rsi_6`** | `RSI(close_1h, 6) / 100.0 shifted 1h` | oscillator | RSI(6) of completed 1h candles [0, 1] |

### Detailed Calculation Breakdowns

#### `feat_mtf_1h_ret`
- **Description:** Log return of last COMPLETED 1h candle
- **Mathematical Formula:** `ln(close_1h / open_1h) shifted 1h`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples 5m bars to 1h candles (`closed='left', label='left'`).
  2. Computes 1h metric: ret.
  3. **Causal Shift:** Shifts the 1h metric by 1 full hour (`shift(1)`) so hour H is only known at H+1h onwards.
  4. Forward-fills onto the 5m time index (`reindex(method='ffill')`). Zero lookahead guaranteed.

#### `feat_mtf_1h_range`
- **Description:** High-low range of last COMPLETED 1h candle normalized by close
- **Mathematical Formula:** `(high_1h - low_1h) / close_1h shifted 1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples 5m bars to 1h candles (`closed='left', label='left'`).
  2. Computes 1h metric: range.
  3. **Causal Shift:** Shifts the 1h metric by 1 full hour (`shift(1)`) so hour H is only known at H+1h onwards.
  4. Forward-fills onto the 5m time index (`reindex(method='ffill')`). Zero lookahead guaranteed.

#### `feat_mtf_1h_vol_ratio`
- **Description:** Volume of last COMPLETED 1h candle vs 24h rolling mean
- **Mathematical Formula:** `vol_1h / SMA(vol_1h, 24) shifted 1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples 5m bars to 1h candles (`closed='left', label='left'`).
  2. Computes 1h metric: vol_ratio.
  3. **Causal Shift:** Shifts the 1h metric by 1 full hour (`shift(1)`) so hour H is only known at H+1h onwards.
  4. Forward-fills onto the 5m time index (`reindex(method='ffill')`). Zero lookahead guaranteed.

#### `feat_mtf_1h_close_vs_sma_6`
- **Description:** Distance of last completed 1h close from 6h SMA
- **Mathematical Formula:** `(close_1h - SMA(close_1h, 6)) / SMA shifted 1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples 5m bars to 1h candles (`closed='left', label='left'`).
  2. Computes 1h metric: close_vs_sma_6.
  3. **Causal Shift:** Shifts the 1h metric by 1 full hour (`shift(1)`) so hour H is only known at H+1h onwards.
  4. Forward-fills onto the 5m time index (`reindex(method='ffill')`). Zero lookahead guaranteed.

#### `feat_mtf_1h_rsi_6`
- **Description:** RSI(6) of completed 1h candles [0, 1]
- **Mathematical Formula:** `RSI(close_1h, 6) / 100.0 shifted 1h`
- **Unit / Range:** `oscillator`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples 5m bars to 1h candles (`closed='left', label='left'`).
  2. Computes 1h metric: rsi_6.
  3. **Causal Shift:** Shifts the 1h metric by 1 full hour (`shift(1)`) so hour H is only known at H+1h onwards.
  4. Forward-fills onto the 5m time index (`reindex(method='ffill')`). Zero lookahead guaranteed.

---

<a id="vol_range"></a>
## Extreme Value Range Volatility (Parkinson & Garman-Klass)

- **Group Key:** `vol_range`
- **Feature Count:** 6
- **Raw Inputs Required:** `open, high, low, close`
- **Overview:** Continuous high-low and opening-closing range volatility estimators across 1h, 5h, and 24h rolling windows.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 54 | **`feat_vol_parkinson_12`** | `sqrt(1/(4 ln 2 * 12) * sum(ln(H/L)^2))` | fraction | Parkinson volatility over 12 candles (1h) |
| 55 | **`feat_vol_parkinson_60`** | `sqrt(1/(4 ln 2 * 60) * sum(ln(H/L)^2))` | fraction | Parkinson volatility over 60 candles (5h) |
| 56 | **`feat_vol_parkinson_288`** | `sqrt(1/(4 ln 2 * 288) * sum(ln(H/L)^2))` | fraction | Parkinson volatility over 288 candles (24h) |
| 57 | **`feat_vol_gk_12`** | `sqrt(1/12 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))` | fraction | Garman-Klass volatility over 12 candles (1h) |
| 58 | **`feat_vol_gk_60`** | `sqrt(1/60 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))` | fraction | Garman-Klass volatility over 60 candles (5h) |
| 59 | **`feat_vol_gk_288`** | `sqrt(1/288 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))` | fraction | Garman-Klass volatility over 288 candles (24h) |

### Detailed Calculation Breakdowns

#### `feat_vol_parkinson_12`
- **Description:** Parkinson volatility over 12 candles (1h)
- **Mathematical Formula:** `sqrt(1/(4 ln 2 * 12) * sum(ln(H/L)^2))`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes high-to-low log ratio: $\ln(High / Low)$.
  2. Applies Parkinson variance estimator: $\sigma^2 = \frac{1}{4 \ln(2)} \times \text{mean}(\ln(H/L)^2, \text{window}=12)$.
  3. Takes square root: `sqrt(parkinson_var_12)`. Captures continuous intra-bar volatility without opening gap noise.

#### `feat_vol_parkinson_60`
- **Description:** Parkinson volatility over 60 candles (5h)
- **Mathematical Formula:** `sqrt(1/(4 ln 2 * 60) * sum(ln(H/L)^2))`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes high-to-low log ratio: $\ln(High / Low)$.
  2. Applies Parkinson variance estimator: $\sigma^2 = \frac{1}{4 \ln(2)} \times \text{mean}(\ln(H/L)^2, \text{window}=60)$.
  3. Takes square root: `sqrt(parkinson_var_60)`. Captures continuous intra-bar volatility without opening gap noise.

#### `feat_vol_parkinson_288`
- **Description:** Parkinson volatility over 288 candles (24h)
- **Mathematical Formula:** `sqrt(1/(4 ln 2 * 288) * sum(ln(H/L)^2))`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes high-to-low log ratio: $\ln(High / Low)$.
  2. Applies Parkinson variance estimator: $\sigma^2 = \frac{1}{4 \ln(2)} \times \text{mean}(\ln(H/L)^2, \text{window}=288)$.
  3. Takes square root: `sqrt(parkinson_var_288)`. Captures continuous intra-bar volatility without opening gap noise.

#### `feat_vol_gk_12`
- **Description:** Garman-Klass volatility over 12 candles (1h)
- **Mathematical Formula:** `sqrt(1/12 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates Garman-Klass log ratios: $u = \ln(H/L)$ and $c = \ln(C/O)$.
  2. Computes unbiased point variance: $\sigma_{GK}^2 = 0.5 u^2 - (2\ln(2) - 1) c^2$.
  3. Averages over rolling 12 candles and takes square root: `sqrt(rolling_mean(gk_sq, 12))`.

#### `feat_vol_gk_60`
- **Description:** Garman-Klass volatility over 60 candles (5h)
- **Mathematical Formula:** `sqrt(1/60 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates Garman-Klass log ratios: $u = \ln(H/L)$ and $c = \ln(C/O)$.
  2. Computes unbiased point variance: $\sigma_{GK}^2 = 0.5 u^2 - (2\ln(2) - 1) c^2$.
  3. Averages over rolling 60 candles and takes square root: `sqrt(rolling_mean(gk_sq, 60))`.

#### `feat_vol_gk_288`
- **Description:** Garman-Klass volatility over 288 candles (24h)
- **Mathematical Formula:** `sqrt(1/288 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates Garman-Klass log ratios: $u = \ln(H/L)$ and $c = \ln(C/O)$.
  2. Computes unbiased point variance: $\sigma_{GK}^2 = 0.5 u^2 - (2\ln(2) - 1) c^2$.
  3. Averages over rolling 288 candles and takes square root: `sqrt(rolling_mean(gk_sq, 288))`.

---

<a id="vol_realized"></a>
## Realized Volatility & Jump Dynamics

- **Group Key:** `vol_realized`
- **Feature Count:** 6
- **Raw Inputs Required:** `close (5m log returns)`
- **Overview:** Sum of absolute returns, maximum absolute returns, and directional upside/downside semi-variance shares.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 60 | **`feat_vol_realized_abs_sum_6`** | `sum(\|r_5m\|, W=6)` | fraction | Realized absolute variation over 6 candles (30m) |
| 61 | **`feat_vol_realized_abs_sum_12`** | `sum(\|r_5m\|, W=12)` | fraction | Realized absolute variation over 12 candles (1h) |
| 62 | **`feat_vol_realized_abs_sum_36`** | `sum(\|r_5m\|, W=36)` | fraction | Realized absolute variation over 36 candles (3h) |
| 63 | **`feat_vol_realized_max_abs_12`** | `max(\|r_5m\|, W=12)` | fraction | Maximum single-candle absolute return over 12 candles (1h) |
| 64 | **`feat_vol_realized_max_abs_60`** | `max(\|r_5m\|, W=60)` | fraction | Maximum single-candle absolute return over 60 candles (5h) |
| 65 | **`feat_vol_realized_semi_var_up_ratio_60`** | `SV^+ / (SV^+ + SV^-)` | fraction [0, 1] | Upside semivariance share over 60 candles |

### Detailed Calculation Breakdowns

#### `feat_vol_realized_abs_sum_6`
- **Description:** Realized absolute variation over 6 candles (30m)
- **Mathematical Formula:** `sum(|r_5m|, W=6)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-candle absolute log returns: $|\ln(C_t / C_{t-1})|$.
  2. Sums absolute returns over the last 6 candles: `rolling_sum(abs_ret, 6)`. High-frequency integrated volatility proxy.

#### `feat_vol_realized_abs_sum_12`
- **Description:** Realized absolute variation over 12 candles (1h)
- **Mathematical Formula:** `sum(|r_5m|, W=12)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-candle absolute log returns: $|\ln(C_t / C_{t-1})|$.
  2. Sums absolute returns over the last 12 candles: `rolling_sum(abs_ret, 12)`. High-frequency integrated volatility proxy.

#### `feat_vol_realized_abs_sum_36`
- **Description:** Realized absolute variation over 36 candles (3h)
- **Mathematical Formula:** `sum(|r_5m|, W=36)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-candle absolute log returns: $|\ln(C_t / C_{t-1})|$.
  2. Sums absolute returns over the last 36 candles: `rolling_sum(abs_ret, 36)`. High-frequency integrated volatility proxy.

#### `feat_vol_realized_max_abs_12`
- **Description:** Maximum single-candle absolute return over 12 candles (1h)
- **Mathematical Formula:** `max(|r_5m|, W=12)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-candle absolute log returns: $|\ln(C_t / C_{t-1})|$.
  2. Takes rolling maximum over last 12 candles: `rolling_max(abs_ret, 12)`. Detects jump discontinuities and flash spikes.

#### `feat_vol_realized_max_abs_60`
- **Description:** Maximum single-candle absolute return over 60 candles (5h)
- **Mathematical Formula:** `max(|r_5m|, W=60)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 1-candle absolute log returns: $|\ln(C_t / C_{t-1})|$.
  2. Takes rolling maximum over last 60 candles: `rolling_max(abs_ret, 60)`. Detects jump discontinuities and flash spikes.

#### `feat_vol_realized_semi_var_up_ratio_60`
- **Description:** Upside semivariance share over 60 candles
- **Mathematical Formula:** `SV^+ / (SV^+ + SV^-)`
- **Unit / Range:** `fraction [0, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Separates squared returns into upside $(r_t^+)^2$ and downside $(r_t^-)^2$.
  2. Computes 60-candle rolling sums: $SemiVar_{up}$ and $SemiVar_{down}$.
  3. Computes upside variance fraction: `semi_up_60 / (semi_up_60 + semi_down_60)`. Values > 0.5 reflect upside volatility dominance; < 0.5 reflect downside panic volatility.

---

<a id="vol_seasonal"></a>
## Diurnal Seasonal Volatility & Baseline Ratios

- **Group Key:** `vol_seasonal`
- **Feature Count:** 4
- **Raw Inputs Required:** `close, volume, number_of_trades`
- **Overview:** 7-day seasonal slot baselines capturing recurring time-of-day volatility, volume, and trade count spikes.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 66 | **`feat_vol_seasonal_slot_abs_ret_7d`** | `mean(\|r_5m\|[t - 288*k]) for k=1..7` | fraction | Same 5m time slot mean absolute return over prior 7 days |
| 67 | **`feat_vol_seasonal_vol_ratio_12`** | `mean(\|r_5m\|, 12) / slot_abs_ret_7d` | ratio | Ratio of 1h mean absolute return to 7-day seasonal slot baseline |
| 68 | **`feat_vol_seasonal_volume_ratio_7d`** | `vol[t] / mean(vol[t - 288*k]) for k=1..7` | ratio | Ratio of current volume to 7-day same-slot mean volume |
| 69 | **`feat_vol_seasonal_trades_ratio_7d`** | `trades[t] / mean(trades[t - 288*k]) for k=1..7` | ratio | Ratio of current trades to 7-day same-slot mean trades |

### Detailed Calculation Breakdowns

#### `feat_vol_seasonal_slot_abs_ret_7d`
- **Description:** Same 5m time slot mean absolute return over prior 7 days
- **Mathematical Formula:** `mean(|r_5m|[t - 288*k]) for k=1..7`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Looks up the exact same 5m time-of-day slot across previous 7 calendar days ($k \times 288$ bars ago, for $k=1..7$).
  2. Averages absolute returns across those 7 historical days.
  3. Provides a baseline expected volatility for this specific 5m time of day.

#### `feat_vol_seasonal_vol_ratio_12`
- **Description:** Ratio of 1h mean absolute return to 7-day seasonal slot baseline
- **Mathematical Formula:** `mean(|r_5m|, 12) / slot_abs_ret_7d`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates rolling volatility $\sigma_{ratio}$ and $\sigma_{12}$.
  2. Divides shorter-window volatility by longer-window volatility: `sigma_ratio / sigma_12`.
  3. Ratios > 1.0 indicate expanding volatility compression; ratios < 1.0 indicate contracting volatility regimes.

#### `feat_vol_seasonal_volume_ratio_7d`
- **Description:** Ratio of current volume to 7-day same-slot mean volume
- **Mathematical Formula:** `vol[t] / mean(vol[t - 288*k]) for k=1..7`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Averages total volume for this same 5m time-of-day slot over previous 7 days.
  2. Computes ratio of current volume to seasonal slot baseline: `volume / seasonal_volume_7d`.

#### `feat_vol_seasonal_trades_ratio_7d`
- **Description:** Ratio of current trades to 7-day same-slot mean trades
- **Mathematical Formula:** `trades[t] / mean(trades[t - 288*k]) for k=1..7`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Averages trade count for this same 5m time-of-day slot over previous 7 days.
  2. Computes ratio of current trade count to seasonal slot baseline: `trades / seasonal_trades_7d`.

---

<a id="vol_har"></a>
## Heterogeneous Autoregressive Volatility (HAR)

- **Group Key:** `vol_har`
- **Feature Count:** 4
- **Raw Inputs Required:** `close (5m log returns)`
- **Overview:** Realized volatility across intraday, 6-hour, daily, and weekly horizons, with term-structure ratios.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 70 | **`feat_vol_har_rv_72`** | `rolling_sigma(r_5m, 72)` | fraction | Realized volatility over 6h (72 candles) |
| 71 | **`feat_vol_har_rv_2016`** | `rolling_sigma(r_5m, 2016)` | fraction | Realized volatility over 7 days (2016 candles) |
| 72 | **`feat_vol_har_ratio_12_288`** | `rv_12 / rv_288` | ratio | HAR volatility ratio: 1h RV / 24h RV |
| 73 | **`feat_vol_har_ratio_288_2016`** | `rv_288 / rv_2016` | ratio | HAR volatility ratio: 24h RV / 7d RV |

### Detailed Calculation Breakdowns

#### `feat_vol_har_rv_72`
- **Description:** Realized volatility over 6h (72 candles)
- **Mathematical Formula:** `rolling_sigma(r_5m, 72)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 5m log returns standard deviation over 72 candles (6 hours).
  2. Formula: `rolling_std(ret_1, 72)`. Captures medium-intraday market maker volatility horizon.

#### `feat_vol_har_rv_2016`
- **Description:** Realized volatility over 7 days (2016 candles)
- **Mathematical Formula:** `rolling_sigma(r_5m, 2016)`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 5m log returns standard deviation over 2,016 candles (7 full days).
  2. Formula: `rolling_std(ret_1, 2016)`. Long-term weekly volatility baseline for HAR models.

#### `feat_vol_har_ratio_12_288`
- **Description:** HAR volatility ratio: 1h RV / 24h RV
- **Mathematical Formula:** `rv_12 / rv_288`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Divides 1-hour realized volatility (12 bars) by 24-hour realized volatility (288 bars).
  2. Formula: `rv_12 / rv_288`. High values denote acute short-term volatility shocks.

#### `feat_vol_har_ratio_288_2016`
- **Description:** HAR volatility ratio: 24h RV / 7d RV
- **Mathematical Formula:** `rv_288 / rv_2016`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Divides 24-hour realized volatility (288 bars) by 7-day realized volatility (2016 bars).
  2. Formula: `rv_288 / rv_2016`. Captures daily vs weekly volatility regime shifts.

---

<a id="flow_window"></a>
## Windowed Cumulative Order Flow & Signed Flow

- **Group Key:** `flow_window`
- **Feature Count:** 6
- **Raw Inputs Required:** `volume, taker_buy_base_volume, close`
- **Overview:** Multi-horizon cumulative net taker buy pressure and directionally signed volume flows (30m, 1h, 3h).

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 74 | **`feat_flow_window_net_6`** | `sum(2*taker_buy - vol) / sum(vol)` | fraction [-1, 1] | Normalized taker net flow over 6 candles (30m) |
| 75 | **`feat_flow_window_net_12`** | `sum(2*taker_buy - vol) / sum(vol)` | fraction [-1, 1] | Normalized taker net flow over 12 candles (1h) |
| 76 | **`feat_flow_window_net_36`** | `sum(2*taker_buy - vol) / sum(vol)` | fraction [-1, 1] | Normalized taker net flow over 36 candles (3h) |
| 77 | **`feat_flow_window_signed_flow_6`** | `net_flow_6 * sign(ret_1)` | fraction [-1, 1] | Price-signed net flow over 6 candles |
| 78 | **`feat_flow_window_signed_flow_12`** | `net_flow_12 * sign(ret_1)` | fraction [-1, 1] | Price-signed net flow over 12 candles |
| 79 | **`feat_flow_window_signed_flow_36`** | `net_flow_36 * sign(ret_1)` | fraction [-1, 1] | Price-signed net flow over 36 candles |

### Detailed Calculation Breakdowns

#### `feat_flow_window_net_6`
- **Description:** Normalized taker net flow over 6 candles (30m)
- **Mathematical Formula:** `sum(2*taker_buy - vol) / sum(vol)`
- **Unit / Range:** `fraction [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates raw net taker flow per candle: $NetFlow = 2 \times TakerBuyVol - TotalVol$.
  2. Sums net flow over 6 candles and sums total volume over 6 candles.
  3. Computes normalized net flow ratio: `sum(net_flow, 6) / sum(volume, 6)`. Bounded in [-1.0, 1.0].

#### `feat_flow_window_net_12`
- **Description:** Normalized taker net flow over 12 candles (1h)
- **Mathematical Formula:** `sum(2*taker_buy - vol) / sum(vol)`
- **Unit / Range:** `fraction [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates raw net taker flow per candle: $NetFlow = 2 \times TakerBuyVol - TotalVol$.
  2. Sums net flow over 12 candles and sums total volume over 12 candles.
  3. Computes normalized net flow ratio: `sum(net_flow, 12) / sum(volume, 12)`. Bounded in [-1.0, 1.0].

#### `feat_flow_window_net_36`
- **Description:** Normalized taker net flow over 36 candles (3h)
- **Mathematical Formula:** `sum(2*taker_buy - vol) / sum(vol)`
- **Unit / Range:** `fraction [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates raw net taker flow per candle: $NetFlow = 2 \times TakerBuyVol - TotalVol$.
  2. Sums net flow over 36 candles and sums total volume over 36 candles.
  3. Computes normalized net flow ratio: `sum(net_flow, 36) / sum(volume, 36)`. Bounded in [-1.0, 1.0].

#### `feat_flow_window_signed_flow_6`
- **Description:** Price-signed net flow over 6 candles
- **Mathematical Formula:** `net_flow_6 * sign(ret_1)`
- **Unit / Range:** `fraction [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes normalized windowed net flow over 6 candles.
  2. Multiplies by the directional sign of 1-candle return: `norm_flow_6 * sign(ret_1)`. Positive when buying pressure confirms price increase; negative on price absorption.

#### `feat_flow_window_signed_flow_12`
- **Description:** Price-signed net flow over 12 candles
- **Mathematical Formula:** `net_flow_12 * sign(ret_1)`
- **Unit / Range:** `fraction [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes normalized windowed net flow over 12 candles.
  2. Multiplies by the directional sign of 1-candle return: `norm_flow_12 * sign(ret_1)`. Positive when buying pressure confirms price increase; negative on price absorption.

#### `feat_flow_window_signed_flow_36`
- **Description:** Price-signed net flow over 36 candles
- **Mathematical Formula:** `net_flow_36 * sign(ret_1)`
- **Unit / Range:** `fraction [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes normalized windowed net flow over 36 candles.
  2. Multiplies by the directional sign of 1-candle return: `norm_flow_36 * sign(ret_1)`. Positive when buying pressure confirms price increase; negative on price absorption.

---

<a id="location"></a>
## Price Location Relative to 24-Hour Range

- **Group Key:** `location`
- **Feature Count:** 3
- **Raw Inputs Required:** `high, low, close`
- **Overview:** Current close position within the rolling 24-hour high/low extreme channel, and normalized distance to bounds.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 80 | **`feat_loc_position_288`** | `(close - min(L_288)) / (max(H_288) - min(L_288))` | fraction [0, 1] | Position of close within 24h high-low channel |
| 81 | **`feat_loc_dist_high_288`** | `(max(H_288) - close) / close` | fraction | Fractional distance of close below 24h high |
| 82 | **`feat_loc_dist_low_288`** | `(close - min(L_288)) / close` | fraction | Fractional distance of close above 24h low |

### Detailed Calculation Breakdowns

#### `feat_loc_position_288`
- **Description:** Position of close within 24h high-low channel
- **Mathematical Formula:** `(close - min(L_288)) / (max(H_288) - min(L_288))`
- **Unit / Range:** `fraction [0, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Finds highest high and lowest low over rolling 288 candles (24 hours).
  2. Calculates relative position: `(close - low_288) / (high_288 - low_288)`.
  3. Clipped to [0.0, 1.0]. 1.0 = at 24h high; 0.0 = at 24h low; 0.5 = at 24h midpoint.

#### `feat_loc_dist_high_288`
- **Description:** Fractional distance of close below 24h high
- **Mathematical Formula:** `(max(H_288) - close) / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Finds highest high over rolling 288 candles (24 hours).
  2. Computes normalized distance: `(high_288 - close) / close`. Measures percentage distance to 24-hour resistance.

#### `feat_loc_dist_low_288`
- **Description:** Fractional distance of close above 24h low
- **Mathematical Formula:** `(close - min(L_288)) / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Finds lowest low over rolling 288 candles (24 hours).
  2. Computes normalized distance: `(close - low_288) / close`. Measures percentage distance to 24-hour support.

---

<a id="time_weekend"></a>
## Weekend Regimes & Weekly Cyclical Coordinates

- **Group Key:** `time_weekend`
- **Feature Count:** 3
- **Raw Inputs Required:** `timestamp`
- **Overview:** Binary weekend indicator and continuous 168-hour weekly cyclical phase coordinates.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 83 | **`feat_time_is_weekend`** | `1.0 if weekday >= 5 else 0.0` | binary {0, 1} | Weekend indicator (Saturday or Sunday UTC) |
| 84 | **`feat_time_sin_week`** | `sin(2 * pi * (weekday*24 + hour + min/60) / 168)` | coordinate in [-1, 1] | Sine transform of 168-hour weekly cycle |
| 85 | **`feat_time_cos_week`** | `cos(2 * pi * (weekday*24 + hour + min/60) / 168)` | coordinate in [-1, 1] | Cosine transform of 168-hour weekly cycle |

### Detailed Calculation Breakdowns

#### `feat_time_is_weekend`
- **Description:** Weekend indicator (Saturday or Sunday UTC)
- **Mathematical Formula:** `1.0 if weekday >= 5 else 0.0`
- **Unit / Range:** `binary {0, 1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Checks if current UTC timestamp is Saturday (5) or Sunday (6).
  2. Returns binary 1.0 on weekends, 0.0 on weekdays. Isolates low-liquidity weekend regimes.

#### `feat_time_sin_week`
- **Description:** Sine transform of 168-hour weekly cycle
- **Mathematical Formula:** `sin(2 * pi * (weekday*24 + hour + min/60) / 168)`
- **Unit / Range:** `coordinate in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes continuous exact minute in the 168-hour week: $t_{week} = weekday \times 24 + hour + minute/60$.
  2. Computes sine harmonic: `sin(2 * pi * hour_of_week / 168.0)`.

#### `feat_time_cos_week`
- **Description:** Cosine transform of 168-hour weekly cycle
- **Mathematical Formula:** `cos(2 * pi * (weekday*24 + hour + min/60) / 168)`
- **Unit / Range:** `coordinate in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes continuous exact minute in the 168-hour week: $t_{week} = weekday \times 24 + hour + minute/60$.
  2. Computes cosine harmonic: `cos(2 * pi * hour_of_week / 168.0)`.

---

<a id="ma_structure"></a>
## Moving Average Term Structure & Trend Alignment

- **Group Key:** `ma_structure`
- **Feature Count:** 36
- **Raw Inputs Required:** `close (5m & 1h completed)`
- **Overview:** 5m and 1h moving average distance ratios, 5-bar momentum slopes, and pairwise multi-MA alignment scores.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 86 | **`feat_ma_sma_ratio_5`** | `(close - sma5) / sma5` | ratio | Ratio of 5m close to 5-period SMA |
| 87 | **`feat_ma_sma_ratio_25`** | `(close - sma25) / sma25` | ratio | Ratio of 5m close to 25-period SMA |
| 88 | **`feat_ma_sma_ratio_100`** | `(close - sma100) / sma100` | ratio | Ratio of 5m close to 100-period SMA |
| 89 | **`feat_ma_ema_ratio_5`** | `(close - ema5) / ema5` | ratio | Ratio of 5m close to 5-period EMA |
| 90 | **`feat_ma_ema_ratio_25`** | `(close - ema25) / ema25` | ratio | Ratio of 5m close to 25-period EMA |
| 91 | **`feat_ma_ema_ratio_100`** | `(close - ema100) / ema100` | ratio | Ratio of 5m close to 100-period EMA |
| 92 | **`feat_ma_sma_slope_5`** | `sma5_t / sma5_{t-5} - 1` | fraction | 5-candle slope of 5-period SMA |
| 93 | **`feat_ma_sma_slope_10`** | `sma10_t / sma10_{t-5} - 1` | fraction | 5-candle slope of 10-period SMA |
| 94 | **`feat_ma_sma_slope_25`** | `sma25_t / sma25_{t-5} - 1` | fraction | 5-candle slope of 25-period SMA |
| 95 | **`feat_ma_sma_slope_50`** | `sma50_t / sma50_{t-5} - 1` | fraction | 5-candle slope of 50-period SMA |
| 96 | **`feat_ma_sma_slope_100`** | `sma100_t / sma100_{t-5} - 1` | fraction | 5-candle slope of 100-period SMA |
| 97 | **`feat_ma_ema_slope_5`** | `ema5_t / ema5_{t-5} - 1` | fraction | 5-candle slope of 5-period EMA |
| 98 | **`feat_ma_ema_slope_10`** | `ema10_t / ema10_{t-5} - 1` | fraction | 5-candle slope of 10-period EMA |
| 99 | **`feat_ma_ema_slope_25`** | `ema25_t / ema25_{t-5} - 1` | fraction | 5-candle slope of 25-period EMA |
| 100 | **`feat_ma_ema_slope_50`** | `ema50_t / ema50_{t-5} - 1` | fraction | 5-candle slope of 50-period EMA |
| 101 | **`feat_ma_ema_slope_100`** | `ema100_t / ema100_{t-5} - 1` | fraction | 5-candle slope of 100-period EMA |
| 102 | **`feat_ma_alignment_sma_5m`** | `(2*sum(short>long) - 10) / 10` | score in [-1, 1] | Pairwise alignment score of 5m SMAs (5,10,25,50,100) |
| 103 | **`feat_ma_alignment_ema_5m`** | `(2*sum(short>long) - 10) / 10` | score in [-1, 1] | Pairwise alignment score of 5m EMAs (5,10,25,50,100) |
| 104 | **`feat_ma_1h_sma_ratio_5`** | `(close - sma5_1h) / sma5_1h` | ratio | Ratio of 5m close to completed 1h 5-period SMA |
| 105 | **`feat_ma_1h_sma_ratio_10`** | `(close - sma10_1h) / sma10_1h` | ratio | Ratio of 5m close to completed 1h 10-period SMA |
| 106 | **`feat_ma_1h_sma_ratio_25`** | `(close - sma25_1h) / sma25_1h` | ratio | Ratio of 5m close to completed 1h 25-period SMA |
| 107 | **`feat_ma_1h_sma_ratio_50`** | `(close - sma50_1h) / sma50_1h` | ratio | Ratio of 5m close to completed 1h 50-period SMA |
| 108 | **`feat_ma_1h_sma_ratio_100`** | `(close - sma100_1h) / sma100_1h` | ratio | Ratio of 5m close to completed 1h 100-period SMA |
| 109 | **`feat_ma_1h_ema_ratio_5`** | `(close - ema5_1h) / ema5_1h` | ratio | Ratio of 5m close to completed 1h 5-period EMA |
| 110 | **`feat_ma_1h_ema_ratio_10`** | `(close - ema10_1h) / ema10_1h` | ratio | Ratio of 5m close to completed 1h 10-period EMA |
| 111 | **`feat_ma_1h_ema_ratio_25`** | `(close - ema25_1h) / ema25_1h` | ratio | Ratio of 5m close to completed 1h 25-period EMA |
| 112 | **`feat_ma_1h_ema_ratio_50`** | `(close - ema50_1h) / ema50_1h` | ratio | Ratio of 5m close to completed 1h 50-period EMA |
| 113 | **`feat_ma_1h_ema_ratio_100`** | `(close - ema100_1h) / ema100_1h` | ratio | Ratio of 5m close to completed 1h 100-period EMA |
| 114 | **`feat_ma_1h_sma_slope_5`** | `sma5_{1h,t} / sma5_{1h,t-5} - 1` | fraction | 5-hour slope of completed 1h 5-period SMA |
| 115 | **`feat_ma_1h_sma_slope_10`** | `sma10_{1h,t} / sma10_{1h,t-5} - 1` | fraction | 5-hour slope of completed 1h 10-period SMA |
| 116 | **`feat_ma_1h_sma_slope_25`** | `sma25_{1h,t} / sma25_{1h,t-5} - 1` | fraction | 5-hour slope of completed 1h 25-period SMA |
| 117 | **`feat_ma_1h_sma_slope_50`** | `sma50_{1h,t} / sma50_{1h,t-5} - 1` | fraction | 5-hour slope of completed 1h 50-period SMA |
| 118 | **`feat_ma_1h_sma_slope_100`** | `sma100_{1h,t} / sma100_{1h,t-5} - 1` | fraction | 5-hour slope of completed 1h 100-period SMA |
| 119 | **`feat_ma_alignment_sma_1h`** | `(2*sum(short>long) - 10) / 10` | score in [-1, 1] | Pairwise alignment score of completed 1h SMAs |
| 120 | **`feat_ma_alignment_ema_1h`** | `(2*sum(short>long) - 10) / 10` | score in [-1, 1] | Pairwise alignment score of completed 1h EMAs |
| 121 | **`feat_ma_1h_natr_14`** | `atr14_1h / close` | ratio | Completed 1h 14-period Wilder ATR normalized by 5m close |

### Detailed Calculation Breakdowns

#### `feat_ma_sma_ratio_5`
- **Description:** Ratio of 5m close to 5-period SMA
- **Mathematical Formula:** `(close - sma5) / sma5`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates inputs: [].
  2. Mathematical Formula: `(close - sma5) / sma5`.
  3. Operational Definition: Ratio of 5m close to 5-period SMA.

#### `feat_ma_sma_ratio_25`
- **Description:** Ratio of 5m close to 25-period SMA
- **Mathematical Formula:** `(close - sma25) / sma25`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates inputs: [].
  2. Mathematical Formula: `(close - sma25) / sma25`.
  3. Operational Definition: Ratio of 5m close to 25-period SMA.

#### `feat_ma_sma_ratio_100`
- **Description:** Ratio of 5m close to 100-period SMA
- **Mathematical Formula:** `(close - sma100) / sma100`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates inputs: [].
  2. Mathematical Formula: `(close - sma100) / sma100`.
  3. Operational Definition: Ratio of 5m close to 100-period SMA.

#### `feat_ma_ema_ratio_5`
- **Description:** Ratio of 5m close to 5-period EMA
- **Mathematical Formula:** `(close - ema5) / ema5`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates inputs: [].
  2. Mathematical Formula: `(close - ema5) / ema5`.
  3. Operational Definition: Ratio of 5m close to 5-period EMA.

#### `feat_ma_ema_ratio_25`
- **Description:** Ratio of 5m close to 25-period EMA
- **Mathematical Formula:** `(close - ema25) / ema25`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates inputs: [].
  2. Mathematical Formula: `(close - ema25) / ema25`.
  3. Operational Definition: Ratio of 5m close to 25-period EMA.

#### `feat_ma_ema_ratio_100`
- **Description:** Ratio of 5m close to 100-period EMA
- **Mathematical Formula:** `(close - ema100) / ema100`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates inputs: [].
  2. Mathematical Formula: `(close - ema100) / ema100`.
  3. Operational Definition: Ratio of 5m close to 100-period EMA.

#### `feat_ma_sma_slope_5`
- **Description:** 5-candle slope of 5-period SMA
- **Mathematical Formula:** `sma5_t / sma5_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 5-period SMA of close prices.
  2. Evaluates 5-candle slope percentage: `(SMA_5[t] / SMA_5[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_sma_slope_10`
- **Description:** 5-candle slope of 10-period SMA
- **Mathematical Formula:** `sma10_t / sma10_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 10-period SMA of close prices.
  2. Evaluates 5-candle slope percentage: `(SMA_10[t] / SMA_10[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_sma_slope_25`
- **Description:** 5-candle slope of 25-period SMA
- **Mathematical Formula:** `sma25_t / sma25_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 25-period SMA of close prices.
  2. Evaluates 5-candle slope percentage: `(SMA_25[t] / SMA_25[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_sma_slope_50`
- **Description:** 5-candle slope of 50-period SMA
- **Mathematical Formula:** `sma50_t / sma50_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 50-period SMA of close prices.
  2. Evaluates 5-candle slope percentage: `(SMA_50[t] / SMA_50[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_sma_slope_100`
- **Description:** 5-candle slope of 100-period SMA
- **Mathematical Formula:** `sma100_t / sma100_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 100-period SMA of close prices.
  2. Evaluates 5-candle slope percentage: `(SMA_100[t] / SMA_100[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_ema_slope_5`
- **Description:** 5-candle slope of 5-period EMA
- **Mathematical Formula:** `ema5_t / ema5_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 5-period EMA of close prices.
  2. Evaluates 5-candle slope percentage: `(EMA_5[t] / EMA_5[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_ema_slope_10`
- **Description:** 5-candle slope of 10-period EMA
- **Mathematical Formula:** `ema10_t / ema10_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 10-period EMA of close prices.
  2. Evaluates 5-candle slope percentage: `(EMA_10[t] / EMA_10[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_ema_slope_25`
- **Description:** 5-candle slope of 25-period EMA
- **Mathematical Formula:** `ema25_t / ema25_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 25-period EMA of close prices.
  2. Evaluates 5-candle slope percentage: `(EMA_25[t] / EMA_25[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_ema_slope_50`
- **Description:** 5-candle slope of 50-period EMA
- **Mathematical Formula:** `ema50_t / ema50_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 50-period EMA of close prices.
  2. Evaluates 5-candle slope percentage: `(EMA_50[t] / EMA_50[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_ema_slope_100`
- **Description:** 5-candle slope of 100-period EMA
- **Mathematical Formula:** `ema100_t / ema100_{t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes 100-period EMA of close prices.
  2. Evaluates 5-candle slope percentage: `(EMA_100[t] / EMA_100[t - 5]) - 1.0`.
  3. Positive values indicate upward trending moving average; negative values indicate downward slope.

#### `feat_ma_alignment_sma_5m`
- **Description:** Pairwise alignment score of 5m SMAs (5,10,25,50,100)
- **Mathematical Formula:** `(2*sum(short>long) - 10) / 10`
- **Unit / Range:** `score in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes 5 moving averages (SMA 5, 10, 25, 50, 100) on 5m candles.
  2. Evaluates all 10 pairwise comparisons $(MA_i > MA_j)$ for $i < j$.
  3. Computes alignment score: `(2 * count_agree - 10) / 10`. Yields +1.0 for perfect bullish stacking (5 > 10 > 25 > 50 > 100) and -1.0 for perfect bearish stacking.

#### `feat_ma_alignment_ema_5m`
- **Description:** Pairwise alignment score of 5m EMAs (5,10,25,50,100)
- **Mathematical Formula:** `(2*sum(short>long) - 10) / 10`
- **Unit / Range:** `score in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes 5 moving averages (EMA 5, 10, 25, 50, 100) on 5m candles.
  2. Evaluates all 10 pairwise comparisons $(MA_i > MA_j)$ for $i < j$.
  3. Computes alignment score: `(2 * count_agree - 10) / 10`. Yields +1.0 for perfect bullish stacking (5 > 10 > 25 > 50 > 100) and -1.0 for perfect bearish stacking.

#### `feat_ma_1h_sma_ratio_5`
- **Description:** Ratio of 5m close to completed 1h 5-period SMA
- **Mathematical Formula:** `(close - sma5_1h) / sma5_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 5-hour SMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - SMA_1h) / SMA_1h`.

#### `feat_ma_1h_sma_ratio_10`
- **Description:** Ratio of 5m close to completed 1h 10-period SMA
- **Mathematical Formula:** `(close - sma10_1h) / sma10_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 10-hour SMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - SMA_1h) / SMA_1h`.

#### `feat_ma_1h_sma_ratio_25`
- **Description:** Ratio of 5m close to completed 1h 25-period SMA
- **Mathematical Formula:** `(close - sma25_1h) / sma25_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 25-hour SMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - SMA_1h) / SMA_1h`.

#### `feat_ma_1h_sma_ratio_50`
- **Description:** Ratio of 5m close to completed 1h 50-period SMA
- **Mathematical Formula:** `(close - sma50_1h) / sma50_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 50-hour SMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - SMA_1h) / SMA_1h`.

#### `feat_ma_1h_sma_ratio_100`
- **Description:** Ratio of 5m close to completed 1h 100-period SMA
- **Mathematical Formula:** `(close - sma100_1h) / sma100_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 100-hour SMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - SMA_1h) / SMA_1h`.

#### `feat_ma_1h_ema_ratio_5`
- **Description:** Ratio of 5m close to completed 1h 5-period EMA
- **Mathematical Formula:** `(close - ema5_1h) / ema5_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 5-hour EMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - EMA_1h) / EMA_1h`.

#### `feat_ma_1h_ema_ratio_10`
- **Description:** Ratio of 5m close to completed 1h 10-period EMA
- **Mathematical Formula:** `(close - ema10_1h) / ema10_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 10-hour EMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - EMA_1h) / EMA_1h`.

#### `feat_ma_1h_ema_ratio_25`
- **Description:** Ratio of 5m close to completed 1h 25-period EMA
- **Mathematical Formula:** `(close - ema25_1h) / ema25_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 25-hour EMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - EMA_1h) / EMA_1h`.

#### `feat_ma_1h_ema_ratio_50`
- **Description:** Ratio of 5m close to completed 1h 50-period EMA
- **Mathematical Formula:** `(close - ema50_1h) / ema50_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 50-hour EMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - EMA_1h) / EMA_1h`.

#### `feat_ma_1h_ema_ratio_100`
- **Description:** Ratio of 5m close to completed 1h 100-period EMA
- **Mathematical Formula:** `(close - ema100_1h) / ema100_1h`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Resamples close to 1-hour completed bars and computes 100-hour EMA.
  2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.
  3. Computes percentage distance: `(close_5m - EMA_1h) / EMA_1h`.

#### `feat_ma_1h_sma_slope_5`
- **Description:** 5-hour slope of completed 1h 5-period SMA
- **Mathematical Formula:** `sma5_{1h,t} / sma5_{1h,t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates 5-hour SMA on completed 1h candles.
  2. Calculates 5-hour slope: `(SMA_1h[T] / SMA_1h[T - 5]) - 1.0`.
  3. Shifts by 1 full hour and aligns to 5m time index.

#### `feat_ma_1h_sma_slope_10`
- **Description:** 5-hour slope of completed 1h 10-period SMA
- **Mathematical Formula:** `sma10_{1h,t} / sma10_{1h,t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates 10-hour SMA on completed 1h candles.
  2. Calculates 5-hour slope: `(SMA_1h[T] / SMA_1h[T - 5]) - 1.0`.
  3. Shifts by 1 full hour and aligns to 5m time index.

#### `feat_ma_1h_sma_slope_25`
- **Description:** 5-hour slope of completed 1h 25-period SMA
- **Mathematical Formula:** `sma25_{1h,t} / sma25_{1h,t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates 25-hour SMA on completed 1h candles.
  2. Calculates 5-hour slope: `(SMA_1h[T] / SMA_1h[T - 5]) - 1.0`.
  3. Shifts by 1 full hour and aligns to 5m time index.

#### `feat_ma_1h_sma_slope_50`
- **Description:** 5-hour slope of completed 1h 50-period SMA
- **Mathematical Formula:** `sma50_{1h,t} / sma50_{1h,t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates 50-hour SMA on completed 1h candles.
  2. Calculates 5-hour slope: `(SMA_1h[T] / SMA_1h[T - 5]) - 1.0`.
  3. Shifts by 1 full hour and aligns to 5m time index.

#### `feat_ma_1h_sma_slope_100`
- **Description:** 5-hour slope of completed 1h 100-period SMA
- **Mathematical Formula:** `sma100_{1h,t} / sma100_{1h,t-5} - 1`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates 100-hour SMA on completed 1h candles.
  2. Calculates 5-hour slope: `(SMA_1h[T] / SMA_1h[T - 5]) - 1.0`.
  3. Shifts by 1 full hour and aligns to 5m time index.

#### `feat_ma_alignment_sma_1h`
- **Description:** Pairwise alignment score of completed 1h SMAs
- **Mathematical Formula:** `(2*sum(short>long) - 10) / 10`
- **Unit / Range:** `score in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes 5 moving averages (SMA 5, 10, 25, 50, 100) on 1h candles.
  2. Evaluates all 10 pairwise comparisons $(MA_i > MA_j)$ for $i < j$.
  3. Computes alignment score: `(2 * count_agree - 10) / 10`. Yields +1.0 for perfect bullish stacking (5 > 10 > 25 > 50 > 100) and -1.0 for perfect bearish stacking.

#### `feat_ma_alignment_ema_1h`
- **Description:** Pairwise alignment score of completed 1h EMAs
- **Mathematical Formula:** `(2*sum(short>long) - 10) / 10`
- **Unit / Range:** `score in [-1, 1]`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Takes 5 moving averages (EMA 5, 10, 25, 50, 100) on 1h candles.
  2. Evaluates all 10 pairwise comparisons $(MA_i > MA_j)$ for $i < j$.
  3. Computes alignment score: `(2 * count_agree - 10) / 10`. Yields +1.0 for perfect bullish stacking (5 > 10 > 25 > 50 > 100) and -1.0 for perfect bearish stacking.

#### `feat_ma_1h_natr_14`
- **Description:** Completed 1h 14-period Wilder ATR normalized by 5m close
- **Mathematical Formula:** `atr14_1h / close`
- **Unit / Range:** `ratio`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Computes Wilder 14-period ATR on completed 1-hour candles.
  2. Shifts by 1 full hour and forward-fills to 5m index.
  3. Normalizes by dividing by current 5m close: `ATR_14_1h / close_5m`.

---

<a id="structure"></a>
## Market Structure, Support/Resistance & Dow Theory

- **Group Key:** `structure`
- **Feature Count:** 15
- **Raw Inputs Required:** `open, high, low, close, ATR-14`
- **Overview:** Non-repainting causal swing pivots, horizontal support/resistance levels, breakout/retest ages, and Dow rule state machine.

| # | Feature Column Name | Formula | Unit / Scale | Description |
| :-: | :--- | :--- | :--- | :--- |
| 122 | **`feat_struct_dist_support_atr`** | `(close - S) / ATR` | ATR multiple | Distance from close to nearest support in 5m ATR units |
| 123 | **`feat_struct_dist_support_pct`** | `(close - S) / close` | fraction | Fractional distance from close to nearest support |
| 124 | **`feat_struct_dist_resistance_atr`** | `(R - close) / ATR` | ATR multiple | Distance from close to nearest resistance in 5m ATR units |
| 125 | **`feat_struct_dist_resistance_pct`** | `(R - close) / close` | fraction | Fractional distance from close to nearest resistance |
| 126 | **`feat_struct_support_strength`** | `level.touches` | count | Touch count of nearest active support level |
| 127 | **`feat_struct_resistance_strength`** | `level.touches` | count | Touch count of nearest active resistance level |
| 128 | **`feat_struct_trend_up`** | `1.0 if trend == Up else 0.0` | binary {0, 1} | Dow Theory uptrend indicator (HH and HL persist) |
| 129 | **`feat_struct_trend_down`** | `1.0 if trend == Down else 0.0` | binary {0, 1} | Dow Theory downtrend indicator (LH and LL persist) |
| 130 | **`feat_struct_trend_range`** | `1.0 if trend == Range else 0.0` | binary {0, 1} | Dow Theory sideways/range market indicator |
| 131 | **`feat_struct_trend_age`** | `t - trend_start` | count | Candles elapsed since current Dow trend state began |
| 132 | **`feat_struct_pullback_depth_atr`** | `(peak - close) / ATR or (close - trough) / ATR` | ATR multiple | Pullback depth from trend extreme in ATR units |
| 133 | **`feat_struct_bars_since_break`** | `bars since break event` | count | Candles elapsed since last support/resistance level break |
| 134 | **`feat_struct_bars_since_retest`** | `bars since retest event` | count | Candles elapsed since last flipped level retest |
| 135 | **`feat_struct_retest_direction`** | `retest direction` | trinary {-1, 0, 1} | Direction of last flipped level retest (+1 bullish, -1 bearish, 0 neutral) |
| 136 | **`feat_struct_rule_vote`** | `Section 9.2 rule vote` | trinary {-1, 0, 1} | Rule-based 4th voter signal per decisions table (+1, -1, 0) |

### Detailed Calculation Breakdowns

#### `feat_struct_dist_support_atr`
- **Description:** Distance from close to nearest support in 5m ATR units
- **Mathematical Formula:** `(close - S) / ATR`
- **Unit / Range:** `ATR multiple`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Identifies most recent confirmed append-only causal support level $S$.
  2. Computes distance in ATR units: `(close - S) / ATR_14`. Positive when price is above support.

#### `feat_struct_dist_support_pct`
- **Description:** Fractional distance from close to nearest support
- **Mathematical Formula:** `(close - S) / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Identifies most recent confirmed causal support level $S$.
  2. Computes percentage distance: `(close - S) / close`.

#### `feat_struct_dist_resistance_atr`
- **Description:** Distance from close to nearest resistance in 5m ATR units
- **Mathematical Formula:** `(R - close) / ATR`
- **Unit / Range:** `ATR multiple`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Identifies most recent confirmed append-only causal resistance level $R$.
  2. Computes distance in ATR units: `(R - close) / ATR_14`. Positive when price is below resistance.

#### `feat_struct_dist_resistance_pct`
- **Description:** Fractional distance from close to nearest resistance
- **Mathematical Formula:** `(R - close) / close`
- **Unit / Range:** `fraction`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Identifies most recent confirmed causal resistance level $R$.
  2. Computes percentage distance: `(R - close) / close`.

#### `feat_struct_support_strength`
- **Description:** Touch count of nearest active support level
- **Mathematical Formula:** `level.touches`
- **Unit / Range:** `count`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Counts number of confirmed swing lows that have merged into this active support level.
  2. Normalizes by dividing by maximum buffer count (capped at 5.0). Higher values signify major structural support.

#### `feat_struct_resistance_strength`
- **Description:** Touch count of nearest active resistance level
- **Mathematical Formula:** `level.touches`
- **Unit / Range:** `count`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Counts number of confirmed swing highs that have merged into this active resistance level.
  2. Normalizes by dividing by maximum buffer count (capped at 5.0). Higher values signify major structural resistance.

#### `feat_struct_trend_up`
- **Description:** Dow Theory uptrend indicator (HH and HL persist)
- **Mathematical Formula:** `1.0 if trend == Up else 0.0`
- **Unit / Range:** `binary {0, 1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates causal Dow Theory trend state machine.
  2. Returns binary 1.0 if market is in confirmed Higher-High + Higher-Low uptrend; otherwise 0.0.

#### `feat_struct_trend_down`
- **Description:** Dow Theory downtrend indicator (LH and LL persist)
- **Mathematical Formula:** `1.0 if trend == Down else 0.0`
- **Unit / Range:** `binary {0, 1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates causal Dow Theory trend state machine.
  2. Returns binary 1.0 if market is in confirmed Lower-High + Lower-Low downtrend; otherwise 0.0.

#### `feat_struct_trend_range`
- **Description:** Dow Theory sideways/range market indicator
- **Mathematical Formula:** `1.0 if trend == Range else 0.0`
- **Unit / Range:** `binary {0, 1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Evaluates causal Dow Theory trend state machine.
  2. Returns binary 1.0 if market is in consolidating/ranging consolidation; otherwise 0.0.

#### `feat_struct_trend_age`
- **Description:** Candles elapsed since current Dow trend state began
- **Mathematical Formula:** `t - trend_start`
- **Unit / Range:** `count`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Counts the number of continuous 5m candles since the current Dow trend regime began.
  2. Normalizes by dividing by 288 (24 hours). Quantifies trend maturity.

#### `feat_struct_pullback_depth_atr`
- **Description:** Pullback depth from trend extreme in ATR units
- **Mathematical Formula:** `(peak - close) / ATR or (close - trough) / ATR`
- **Unit / Range:** `ATR multiple`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Measures current price retracement from the most recent swing extreme in the trend direction.
  2. Normalizes depth by ATR-14: `abs(extreme_price - close) / ATR_14`.

#### `feat_struct_bars_since_break`
- **Description:** Candles elapsed since last support/resistance level break
- **Mathematical Formula:** `bars since break event`
- **Unit / Range:** `count`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Counts elapsed candles since the most recent confirmed breakout beyond a key support or resistance level.
  2. Bounded by 288.

#### `feat_struct_bars_since_retest`
- **Description:** Candles elapsed since last flipped level retest
- **Mathematical Formula:** `bars since retest event`
- **Unit / Range:** `count`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Counts elapsed candles since price returned to retest a broken level within 0.5 * ATR-14.
  2. Bounded by 288.

#### `feat_struct_retest_direction`
- **Description:** Direction of last flipped level retest (+1 bullish, -1 bearish, 0 neutral)
- **Mathematical Formula:** `retest direction`
- **Unit / Range:** `trinary {-1, 0, 1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Direction of recent level retest: +1.0 for bullish retest (testing old resistance as new support), -1.0 for bearish retest (testing old support as new resistance), 0.0 for none.

#### `feat_struct_rule_vote`
- **Description:** Rule-based 4th voter signal per decisions table (+1, -1, 0)
- **Mathematical Formula:** `Section 9.2 rule vote`
- **Unit / Range:** `trinary {-1, 0, 1}`
- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).
- **Calculation Steps:**
  1. Deterministic 4th Voter combining Dow trend, S/R proximity, and retest confirmation.
  2. Returns categorical voting score: +1.0 (Buy/Long), -1.0 (Sell/Short), 0.0 (Neutral/No Action).

---

<a id="summary-taxonomy"></a>
## Summary Taxonomy & Model Deployment Scope

### 1. Total Feature Count by Group

| Feature Group | Number of Features | Primary Domain |
| :--- | :-: | :--- |
| **`returns`** | 5 | Returns & Price Differentials |
| **`volatility`** | 9 | Volatility, Rolling Sigmas & Envelopes |
| **`candle_shape`** | 6 | Candlestick Anatomy & Price Distribution |
| **`momentum`** | 13 | Momentum & Trend Oscillators |
| **`volume`** | 5 | Trading Volume Dynamics |
| **`order_flow`** | 6 | Microstructure & Order Flow |
| **`time`** | 4 | Intraday & Weekly Cyclical Time |
| **`mtf_1h`** | 5 | Multi-Timeframe 1-Hour Aggregations |
| **`vol_range`** | 6 | Extreme Value Range Volatility (Parkinson & Garman-Klass) |
| **`vol_realized`** | 6 | Realized Volatility & Jump Dynamics |
| **`vol_seasonal`** | 4 | Diurnal Seasonal Volatility & Baseline Ratios |
| **`vol_har`** | 4 | Heterogeneous Autoregressive Volatility (HAR) |
| **`flow_window`** | 6 | Windowed Cumulative Order Flow & Signed Flow |
| **`location`** | 3 | Price Location Relative to 24-Hour Range |
| **`time_weekend`** | 3 | Weekend Regimes & Weekly Cyclical Coordinates |
| **`ma_structure`** | 36 | Moving Average Term Structure & Trend Alignment |
| **`structure`** | 15 | Market Structure, Support/Resistance & Dow Theory |
| **TOTAL** | **136** | **Full 17-Domain Causal Space** |

### 2. Horizon-2 Deployment (10-Minute Forecast)
- **Active Features (101 Features):** Dedicated to Horizon 2 (2 candles ahead = 10m). Excludes redundant copies belonging to alternative forecasting horizons (3, 6, 12).
- **Tree Ensembles (LightGBM, XGBoost, CatBoost, Random Forest):** Ingest full feature vector without requiring scale transformation.
- **Linear Models (Ridge, Logistic Regression):** Standardized using `StandardScaler` fitted exclusively on training splits (Folds 1..5) with frozen parameters.
- **Meta-Model Combiner (15 Inputs):** Stacks 9 sub-model probability outputs + 6 continuous market condition features (`feat_vol_natr_14`, `feat_mom_rsi_14`, `feat_mom_adx_14`, `feat_of_taker_buy_ratio`, `feat_ma_alignment_sma_5m`, `feat_struct_rule_vote`).
