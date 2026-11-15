"""Script to generate a comprehensive, clean, and direct Feature Calculation Reference Document.
Output paths:
- ML/reports/features_calculation_reference.md
- reports/features_calculation_reference.md
"""

import sys
from pathlib import Path
import json
import re

# Add ML directory to path
ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))

import src.column_dictionary as cd
from src.features import ALL_FEATURE_COLUMNS, FEATURE_GROUPS

# Group display metadata
GROUP_INFO = {
    "returns": {
        "title": "Returns & Price Differentials",
        "description": "Logarithmic price returns across multi-scale lookback horizons.",
        "input_series": "close"
    },
    "volatility": {
        "title": "Volatility, Rolling Sigmas & Envelopes",
        "description": "Causal rolling standard deviations, normalized ATR, Bollinger Bands, and high-low range measures.",
        "input_series": "high, low, close"
    },
    "candle_shape": {
        "title": "Candlestick Anatomy & Price Distribution",
        "description": "Intra-candle structural geometries including body percentages, wick lengths, close position, and polarity.",
        "input_series": "open, high, low, close"
    },
    "momentum": {
        "title": "Momentum & Trend Oscillators",
        "description": "Moving average ratios, dual-period RSI, MACD components, and Average Directional Index (ADX).",
        "input_series": "close, high, low"
    },
    "volume": {
        "title": "Trading Volume Dynamics",
        "description": "Log-transformed volume, rolling volume Z-scores, rate-of-change, and short-to-medium moving average volume ratios.",
        "input_series": "volume"
    },
    "order_flow": {
        "title": "Microstructure & Order Flow",
        "description": "Aggressive taker buy volume ratios, order imbalance Z-scores, and trade size dynamics.",
        "input_series": "volume, taker_buy_base_volume, number_of_trades"
    },
    "time": {
        "title": "Intraday & Weekly Cyclical Time",
        "description": "Harmonic sine/cosine encodings of UTC hour (24-hour diurnal cycle) and day of week (7-day cycle).",
        "input_series": "timestamp"
    },
    "mtf_1h": {
        "title": "Multi-Timeframe 1-Hour Aggregations",
        "description": "Features computed strictly from fully COMPLETED 1-hour candles, forward-filled with 1-hour shift for zero leakage.",
        "input_series": "open, high, low, close, volume (resampled to 1h)"
    },
    "vol_range": {
        "title": "Extreme Value Range Volatility (Parkinson & Garman-Klass)",
        "description": "Continuous high-low and opening-closing range volatility estimators across 1h, 5h, and 24h rolling windows.",
        "input_series": "open, high, low, close"
    },
    "vol_realized": {
        "title": "Realized Volatility & Jump Dynamics",
        "description": "Sum of absolute returns, maximum absolute returns, and directional upside/downside semi-variance shares.",
        "input_series": "close (5m log returns)"
    },
    "vol_seasonal": {
        "title": "Diurnal Seasonal Volatility & Baseline Ratios",
        "description": "7-day seasonal slot baselines capturing recurring time-of-day volatility, volume, and trade count spikes.",
        "input_series": "close, volume, number_of_trades"
    },
    "vol_har": {
        "title": "Heterogeneous Autoregressive Volatility (HAR)",
        "description": "Realized volatility across intraday, 6-hour, daily, and weekly horizons, with term-structure ratios.",
        "input_series": "close (5m log returns)"
    },
    "flow_window": {
        "title": "Windowed Cumulative Order Flow & Signed Flow",
        "description": "Multi-horizon cumulative net taker buy pressure and directionally signed volume flows (30m, 1h, 3h).",
        "input_series": "volume, taker_buy_base_volume, close"
    },
    "location": {
        "title": "Price Location Relative to 24-Hour Range",
        "description": "Current close position within the rolling 24-hour high/low extreme channel, and normalized distance to bounds.",
        "input_series": "high, low, close"
    },
    "time_weekend": {
        "title": "Weekend Regimes & Weekly Cyclical Coordinates",
        "description": "Binary weekend indicator and continuous 168-hour weekly cyclical phase coordinates.",
        "input_series": "timestamp"
    },
    "ma_structure": {
        "title": "Moving Average Term Structure & Trend Alignment",
        "description": "5m and 1h moving average distance ratios, 5-bar momentum slopes, and pairwise multi-MA alignment scores.",
        "input_series": "close (5m & 1h completed)"
    },
    "structure": {
        "title": "Market Structure, Support/Resistance & Dow Theory",
        "description": "Non-repainting causal swing pivots, horizontal support/resistance levels, breakout/retest ages, and Dow rule state machine.",
        "input_series": "open, high, low, close, ATR-14"
    }
}

# Detailed step-by-step calculation descriptions for features
def get_detailed_calculation(col_name: str, meta: dict) -> str:
    formula = meta.get("formula", "")
    desc = meta.get("description", "")
    
    # Custom tailored explanations based on feature category
    if col_name.startswith("feat_ret_"):
        lag = col_name.split("_")[-1]
        minutes = int(lag) * 5
        return f"1. Takes candle close at current bar $t$.\n2. Computes the natural logarithm of current close divided by the closed price {lag} bars ago ($t - {lag}$, representing {minutes} minutes prior).\n3. Formula: `ln(close[t] / close[t - {lag}])`. Guarantees exact stationarity with zero USD level dependency."
    
    if "sigma" in col_name and not "har" in col_name:
        w = col_name.split("_")[-1]
        hours = round(int(w) * 5 / 60, 1)
        return f"1. Calculates 1-candle 5m log returns $r_t = \\ln(C_t / C_{{t-1}})$.\n2. Computes causal sample standard deviation over a rolling window of {w} candles ({hours} hours).\n3. Uses `min_periods=2` and strictly closed bars to prevent any future boundary leakage."
        
    if "vol_ratio_" in col_name:
        parts = col_name.split("_")
        w1, w2 = parts[-2], parts[-1]
        return f"1. Evaluates rolling volatility $\\sigma_{{{w1}}}$ and $\\sigma_{{{w2}}}$.\n2. Divides shorter-window volatility by longer-window volatility: `sigma_{w1} / sigma_{w2}`.\n3. Ratios > 1.0 indicate expanding volatility compression; ratios < 1.0 indicate contracting volatility regimes."

    if col_name == "feat_vol_natr_14":
        return f"1. Computes True Range: $TR_t = \\max(H_t - L_t, |H_t - C_{{t-1}}|, |L_t - C_{{t-1}}|)$.\n2. Applies Wilder's 14-period recursive RMA (alpha = 1/14).\n3. Normalizes by dividing ATR by candle close $C_t$: `ATR_14 / close`. Scale invariant."

    if "bb_" in col_name:
        if "pct" in col_name:
            return f"1. Computes 20-period rolling close SMA $\\mu$ and standard deviation $\\sigma$.\n2. Forms Upper Band $(\\mu + 2\\sigma)$ and Lower Band $(\\mu - 2\\sigma)$.\n3. Computes %B oscillator: `(close - lower_band) / (upper_band - lower_band)`. Values near 1.0 indicate upper envelope contact; values near 0.0 indicate lower envelope contact."
        else:
            return f"1. Computes 20-period Bollinger upper and lower envelope difference: $Band = Upper - Lower = 4\\sigma$.\n2. Normalizes bandwidth by 20-period moving average: `Band / SMA_20`. Measures relative volatility expansion/squeeze."

    if col_name == "feat_vol_range_12":
        return f"1. Computes highest high over last 12 candles (1 hour) and lowest low over last 12 candles.\n2. Computes total span: $High_{{12}} - Low_{{12}}$.\n3. Normalizes by dividing by current close: `(max_high_12 - min_low_12) / close`."

    if col_name.startswith("feat_shape_"):
        if "body_pct" in col_name:
            return "1. Calculates absolute candle body height: $|C_t - O_t|$.\n2. Divides by candle close: `abs(close - open) / close`. Measures pure body displacement percentage."
        if "body_to_range" in col_name:
            return "1. Calculates candle body $|C_t - O_t|$ and full high-low span $(H_t - L_t)$.\n2. Computes ratio: `abs(close - open) / (high - low)`. 1.0 represents full marubozu; 0.0 represents a doji."
        if "upper_wick_pct" in col_name:
            return "1. Identifies top of the candle body: $\\max(C_t, O_t)$.\n2. Calculates upper shadow: $H_t - \\max(C_t, O_t)$.\n3. Normalizes by dividing by close: `(high - max(close, open)) / close`."
        if "lower_wick_pct" in col_name:
            return "1. Identifies bottom of the candle body: $\\min(C_t, O_t)$.\n2. Calculates lower shadow: $\\min(C_t, O_t) - L_t$.\n3. Normalizes by dividing by close: `(min(close, open) - low) / close`."
        if "close_pos" in col_name:
            return "1. Calculates where close sits within the candle's high-low span: `(close - low) / (high - low)`.\n2. Bounded in [0.0, 1.0]. 1.0 = close at candle high; 0.0 = close at candle low; 0.5 = midpoint."
        if "candle_dir" in col_name:
            return "1. Evaluates direction of candle: `sign(close - open)`.\n2. Returns +1.0 for bullish green candle, -1.0 for bearish red candle, 0.0 for exact flat doji."

    if col_name.startswith("feat_mom_sma_ratio_") or col_name.startswith("feat_mom_ema_ratio_"):
        parts = col_name.split("_")
        kind = parts[2].upper()
        w = parts[-1]
        return f"1. Computes {w}-period causal {kind} of close price.\n2. Calculates normalized distance ratio: `(close - {kind}_{w}) / {kind}_{w}`.\n3. Expresses how far price has deviated above (+) or below (-) the moving average baseline."

    if col_name == "feat_mom_sma_ratio_10_50":
        return "1. Computes 10-period and 50-period simple moving averages.\n2. Calculates percentage spread: `(SMA_10 - SMA_50) / SMA_50`. Identifies short vs intermediate trend divergence and MA crossovers."

    if col_name.startswith("feat_mom_rsi_"):
        period = col_name.split("_")[-1]
        return f"1. Computes 1-bar price changes $\\Delta C = C_t - C_{{t-1}}$.\n2. Separates into positive gains $\\max(\\Delta C, 0)$ and losses $\\max(-\\Delta C, 0)$.\n3. Smooths gains and losses using Wilder's RMA over {period} periods.\n4. Computes Relative Strength $RS = AvgGain / AvgLoss$ and $RSI = 100 - (100 / (1 + RS))$.\n5. Scaled into [0.0, 1.0] by dividing by 100: `RSI / 100`."

    if col_name.startswith("feat_mom_macd_"):
        sub = col_name.split("_")[-1]
        if sub == "line":
            return "1. Computes Fast EMA (12-period) and Slow EMA (26-period) on close.\n2. Calculates raw MACD difference: $EMA_{12} - EMA_{26}$.\n3. Normalizes by dividing by close: `(EMA_12 - EMA_26) / close`."
        elif sub == "signal":
            return "1. Computes MACD line $(EMA_{12} - EMA_{26})$.\n2. Calculates 9-period EMA of the MACD line.\n3. Normalizes by dividing by close: `EMA_9(MACD_raw) / close`."
        elif sub == "hist":
            return "1. Computes MACD Line and MACD Signal Line.\n2. Calculates MACD Histogram: $MACD_{line} - MACD_{signal}$.\n3. Normalizes by dividing by close: `(MACD_raw - MACD_signal_raw) / close`."

    if col_name == "feat_mom_adx_14":
        return "1. Evaluates directional movement: $+DM = \\max(H_t - H_{t-1}, 0)$ and $-DM = \\max(L_{t-1} - L_t, 0)$.\n2. Normalizes by 14-period Wilder ATR to produce $+DI_{14}$ and $-DI_{14}$.\n3. Calculates Directional Index $DX = 100 \\times |+DI - -DI| / (+DI + -DI)$.\n4. Smooths DX via 14-period Wilder RMA to yield ADX.\n5. Scaled to [0.0, 1.0] by dividing by 100: `ADX / 100`."

    if col_name == "feat_volume_log":
        return "1. Takes raw 5-minute trading volume $V_t$.\n2. Computes natural log of 1 + volume: `ln(1 + volume)`. Compresses volume spikes into a stationary bell-like distribution."

    if "volume_zscore_" in col_name:
        w = col_name.split("_")[-1]
        return f"1. Evaluates rolling mean $\\mu_V$ and standard deviation $\\sigma_V$ of trading volume over {w} candles.\n2. Computes rolling Z-score: `(volume - mean_{w}) / std_{w}`. Identifies statistical volume surges relative to the {w}-candle baseline."

    if col_name == "feat_volume_change_1":
        return "1. Calculates percentage volume delta from previous 5m candle: `(volume[t] / volume[t-1]) - 1.0`. Measures sudden volume acceleration."

    if col_name == "feat_volume_ratio_12_60":
        return "1. Calculates 12-candle (1-hour) volume SMA and 60-candle (5-hour) volume SMA.\n2. Computes ratio: `SMA_vol_12 / SMA_vol_60`. Values > 1.0 indicate elevated short-term volume intensity."

    if col_name.startswith("feat_of_"):
        if "taker_buy_ratio" in col_name and not "zscore" in col_name:
            return "1. Takes taker buy base volume (market buy order fills) and total candle volume.\n2. Computes ratio: `taker_buy_volume / total_volume`. Clipped to [0.0, 1.0]. Values > 0.5 indicate net aggressive buyer dominance."
        if "taker_buy_zscore_60" in col_name:
            return "1. Computes taker buy ratio over each 5m bar.\n2. Calculates 60-period rolling mean and standard deviation of taker buy ratio.\n3. Computes Z-score: `(taker_ratio - mean_60) / std_60`."
        if "trades_zscore_" in col_name:
            w = col_name.split("_")[-1]
            return f"1. Takes Binance trade count ($N_{{trades}}$) per 5m bar.\n2. Computes rolling {w}-candle mean and standard deviation.\n3. Calculates rolling Z-score: `(trades - mean_{w}) / std_{w}`. Detects microstructural trade frequency bursts."
        if col_name == "feat_of_avg_trade_size":
            return "1. Divides total candle volume by number of trades: `volume / number_of_trades`.\n2. Measures average trade execution size. Larger values reflect institutional participation; smaller values reflect retail activity."
        if col_name == "feat_of_avg_trade_size_zscore_60":
            return "1. Evaluates average trade size `volume / trades` per bar.\n2. Computes rolling 60-period mean and standard deviation.\n3. Calculates Z-score: `(avg_trade_size - mean_60) / std_60`."

    if col_name.startswith("feat_time_"):
        if "sin_hour" in col_name:
            return "1. Extracts UTC hour $h \\in [0, 23]$.\n2. Computes sine harmonic: `sin(2 * pi * hour / 24.0)`. Cyclical 24-hour diurnal encoding."
        if "cos_hour" in col_name:
            return "1. Extracts UTC hour $h \\in [0, 23]$.\n2. Computes cosine harmonic: `cos(2 * pi * hour / 24.0)`. Orthogonal diurnal coordinate."
        if "sin_weekday" in col_name:
            return "1. Extracts weekday index $w \\in [0, 6]$ (Monday = 0).\n2. Computes sine harmonic: `sin(2 * pi * weekday / 7.0)`. Captures weekly periodicity."
        if "cos_weekday" in col_name:
            return "1. Extracts weekday index $w \\in [0, 6]$ (Monday = 0).\n2. Computes cosine harmonic: `cos(2 * pi * weekday / 7.0)`. Orthogonal weekly coordinate."
        if "is_weekend" in col_name:
            return "1. Checks if current UTC timestamp is Saturday (5) or Sunday (6).\n2. Returns binary 1.0 on weekends, 0.0 on weekdays. Isolates low-liquidity weekend regimes."
        if "sin_week" in col_name:
            return "1. Computes continuous exact minute in the 168-hour week: $t_{week} = weekday \\times 24 + hour + minute/60$.\n2. Computes sine harmonic: `sin(2 * pi * hour_of_week / 168.0)`."
        if "cos_week" in col_name:
            return "1. Computes continuous exact minute in the 168-hour week: $t_{week} = weekday \\times 24 + hour + minute/60$.\n2. Computes cosine harmonic: `cos(2 * pi * hour_of_week / 168.0)`."

    if col_name.startswith("feat_mtf_1h_"):
        sub = col_name.replace("feat_mtf_1h_", "")
        return f"1. Resamples 5m bars to 1h candles (`closed='left', label='left'`).\n2. Computes 1h metric: {sub}.\n3. **Causal Shift:** Shifts the 1h metric by 1 full hour (`shift(1)`) so hour H is only known at H+1h onwards.\n4. Forward-fills onto the 5m time index (`reindex(method='ffill')`). Zero lookahead guaranteed."

    if col_name.startswith("feat_vol_parkinson_"):
        w = col_name.split("_")[-1]
        return f"1. Computes high-to-low log ratio: $\\ln(High / Low)$.\n2. Applies Parkinson variance estimator: $\\sigma^2 = \\frac{{1}}{{4 \\ln(2)}} \\times \\text{{mean}}(\\ln(H/L)^2, \\text{{window}}={w})$.\n3. Takes square root: `sqrt(parkinson_var_{w})`. Captures continuous intra-bar volatility without opening gap noise."

    if col_name.startswith("feat_vol_gk_"):
        w = col_name.split("_")[-1]
        return f"1. Evaluates Garman-Klass log ratios: $u = \\ln(H/L)$ and $c = \\ln(C/O)$.\n2. Computes unbiased point variance: $\\sigma_{{GK}}^2 = 0.5 u^2 - (2\\ln(2) - 1) c^2$.\n3. Averages over rolling {w} candles and takes square root: `sqrt(rolling_mean(gk_sq, {w}))`."

    if col_name.startswith("feat_vol_realized_abs_sum_"):
        w = col_name.split("_")[-1]
        return f"1. Computes 1-candle absolute log returns: $|\\ln(C_t / C_{{t-1}})|$.\n2. Sums absolute returns over the last {w} candles: `rolling_sum(abs_ret, {w})`. High-frequency integrated volatility proxy."

    if col_name.startswith("feat_vol_realized_max_abs_"):
        w = col_name.split("_")[-1]
        return f"1. Computes 1-candle absolute log returns: $|\\ln(C_t / C_{{t-1}})|$.\n2. Takes rolling maximum over last {w} candles: `rolling_max(abs_ret, {w})`. Detects jump discontinuities and flash spikes."

    if col_name == "feat_vol_realized_semi_var_up_ratio_60":
        return "1. Separates squared returns into upside $(r_t^+)^2$ and downside $(r_t^-)^2$.\n2. Computes 60-candle rolling sums: $SemiVar_{up}$ and $SemiVar_{down}$.\n3. Computes upside variance fraction: `semi_up_60 / (semi_up_60 + semi_down_60)`. Values > 0.5 reflect upside volatility dominance; < 0.5 reflect downside panic volatility."

    if col_name.startswith("feat_vol_seasonal_"):
        if "slot_abs_ret_7d" in col_name:
            return "1. Looks up the exact same 5m time-of-day slot across previous 7 calendar days ($k \\times 288$ bars ago, for $k=1..7$).\n2. Averages absolute returns across those 7 historical days.\n3. Provides a baseline expected volatility for this specific 5m time of day."
        if "vol_ratio_12" in col_name:
            return "1. Computes current 1-hour realized volatility (`abs_sum_12 / 12`).\n2. Divides by the 7-day same-slot expected volatility: `curr_vol_12 / slot_abs_ret_7d`. Ratios > 1.0 indicate abnormal volatility relative to historical time-of-day expectations."
        if "volume_ratio_7d" in col_name:
            return "1. Averages total volume for this same 5m time-of-day slot over previous 7 days.\n2. Computes ratio of current volume to seasonal slot baseline: `volume / seasonal_volume_7d`."
        if "trades_ratio_7d" in col_name:
            return "1. Averages trade count for this same 5m time-of-day slot over previous 7 days.\n2. Computes ratio of current trade count to seasonal slot baseline: `trades / seasonal_trades_7d`."

    if col_name.startswith("feat_vol_har_"):
        if "rv_72" in col_name:
            return "1. Computes 5m log returns standard deviation over 72 candles (6 hours).\n2. Formula: `rolling_std(ret_1, 72)`. Captures medium-intraday market maker volatility horizon."
        if "rv_2016" in col_name:
            return "1. Computes 5m log returns standard deviation over 2,016 candles (7 full days).\n2. Formula: `rolling_std(ret_1, 2016)`. Long-term weekly volatility baseline for HAR models."
        if "ratio_12_288" in col_name:
            return "1. Divides 1-hour realized volatility (12 bars) by 24-hour realized volatility (288 bars).\n2. Formula: `rv_12 / rv_288`. High values denote acute short-term volatility shocks."
        if "ratio_288_2016" in col_name:
            return "1. Divides 24-hour realized volatility (288 bars) by 7-day realized volatility (2016 bars).\n2. Formula: `rv_288 / rv_2016`. Captures daily vs weekly volatility regime shifts."

    if col_name.startswith("feat_flow_window_"):
        w = col_name.split("_")[-1]
        if "net_" in col_name:
            return f"1. Evaluates raw net taker flow per candle: $NetFlow = 2 \\times TakerBuyVol - TotalVol$.\n2. Sums net flow over {w} candles and sums total volume over {w} candles.\n3. Computes normalized net flow ratio: `sum(net_flow, {w}) / sum(volume, {w})`. Bounded in [-1.0, 1.0]."
        if "signed_flow_" in col_name:
            return f"1. Takes normalized windowed net flow over {w} candles.\n2. Multiplies by the directional sign of 1-candle return: `norm_flow_{w} * sign(ret_1)`. Positive when buying pressure confirms price increase; negative on price absorption."

    if col_name.startswith("feat_loc_"):
        if "position_288" in col_name:
            return "1. Finds highest high and lowest low over rolling 288 candles (24 hours).\n2. Calculates relative position: `(close - low_288) / (high_288 - low_288)`.\n3. Clipped to [0.0, 1.0]. 1.0 = at 24h high; 0.0 = at 24h low; 0.5 = at 24h midpoint."
        if "dist_high_288" in col_name:
            return "1. Finds highest high over rolling 288 candles (24 hours).\n2. Computes normalized distance: `(high_288 - close) / close`. Measures percentage distance to 24-hour resistance."
        if "dist_low_288" in col_name:
            return "1. Finds lowest low over rolling 288 candles (24 hours).\n2. Computes normalized distance: `(close - low_288) / close`. Measures percentage distance to 24-hour support."

    if col_name.startswith("feat_ma_sma_slope_") or col_name.startswith("feat_ma_ema_slope_"):
        parts = col_name.split("_")
        kind = parts[2].upper()
        w = parts[-1]
        return f"1. Computes {w}-period {kind} of close prices.\n2. Evaluates 5-candle slope percentage: `({kind}_{w}[t] / {kind}_{w}[t - 5]) - 1.0`.\n3. Positive values indicate upward trending moving average; negative values indicate downward slope."

    if col_name.startswith("feat_ma_1h_"):
        parts = col_name.split("_")
        if "ratio" in col_name:
            kind = parts[3].upper()
            w = parts[-1]
            return f"1. Resamples close to 1-hour completed bars and computes {w}-hour {kind}.\n2. Shifts by 1 full hour (`shift(1)`) and forward-fills to 5m index.\n3. Computes percentage distance: `(close_5m - {kind}_1h) / {kind}_1h`."
        if "slope" in col_name:
            kind = parts[3].upper()
            w = parts[-1]
            return f"1. Evaluates {w}-hour {kind} on completed 1h candles.\n2. Calculates 5-hour slope: `({kind}_1h[T] / {kind}_1h[T - 5]) - 1.0`.\n3. Shifts by 1 full hour and aligns to 5m time index."
        if "natr_14" in col_name:
            return "1. Computes Wilder 14-period ATR on completed 1-hour candles.\n2. Shifts by 1 full hour and forward-fills to 5m index.\n3. Normalizes by dividing by current 5m close: `ATR_14_1h / close_5m`."

    if "alignment" in col_name:
        parts = col_name.split("_")
        kind = parts[3].upper()
        tf = parts[4]
        return f"1. Takes 5 moving averages ({kind} 5, 10, 25, 50, 100) on {tf} candles.\n2. Evaluates all 10 pairwise comparisons $(MA_i > MA_j)$ for $i < j$.\n3. Computes alignment score: `(2 * count_agree - 10) / 10`. Yields +1.0 for perfect bullish stacking (5 > 10 > 25 > 50 > 100) and -1.0 for perfect bearish stacking."

    if col_name.startswith("feat_struct_"):
        if "dist_support_atr" in col_name:
            return "1. Identifies most recent confirmed append-only causal support level $S$.\n2. Computes distance in ATR units: `(close - S) / ATR_14`. Positive when price is above support."
        if "dist_support_pct" in col_name:
            return "1. Identifies most recent confirmed causal support level $S$.\n2. Computes percentage distance: `(close - S) / close`."
        if "dist_resistance_atr" in col_name:
            return "1. Identifies most recent confirmed append-only causal resistance level $R$.\n2. Computes distance in ATR units: `(R - close) / ATR_14`. Positive when price is below resistance."
        if "dist_resistance_pct" in col_name:
            return "1. Identifies most recent confirmed causal resistance level $R$.\n2. Computes percentage distance: `(R - close) / close`."
        if "support_strength" in col_name:
            return "1. Counts number of confirmed swing lows that have merged into this active support level.\n2. Normalizes by dividing by maximum buffer count (capped at 5.0). Higher values signify major structural support."
        if "resistance_strength" in col_name:
            return "1. Counts number of confirmed swing highs that have merged into this active resistance level.\n2. Normalizes by dividing by maximum buffer count (capped at 5.0). Higher values signify major structural resistance."
        if "trend_up" in col_name:
            return "1. Evaluates causal Dow Theory trend state machine.\n2. Returns binary 1.0 if market is in confirmed Higher-High + Higher-Low uptrend; otherwise 0.0."
        if "trend_down" in col_name:
            return "1. Evaluates causal Dow Theory trend state machine.\n2. Returns binary 1.0 if market is in confirmed Lower-High + Lower-Low downtrend; otherwise 0.0."
        if "trend_range" in col_name:
            return "1. Evaluates causal Dow Theory trend state machine.\n2. Returns binary 1.0 if market is in consolidating/ranging consolidation; otherwise 0.0."
        if "trend_age" in col_name:
            return "1. Counts the number of continuous 5m candles since the current Dow trend regime began.\n2. Normalizes by dividing by 288 (24 hours). Quantifies trend maturity."
        if "pullback_depth_atr" in col_name:
            return "1. Measures current price retracement from the most recent swing extreme in the trend direction.\n2. Normalizes depth by ATR-14: `abs(extreme_price - close) / ATR_14`."
        if "bars_since_break" in col_name:
            return "1. Counts elapsed candles since the most recent confirmed breakout beyond a key support or resistance level.\n2. Bounded by 288."
        if "bars_since_retest" in col_name:
            return "1. Counts elapsed candles since price returned to retest a broken level within 0.5 * ATR-14.\n2. Bounded by 288."
        if "retest_direction" in col_name:
            return "1. Direction of recent level retest: +1.0 for bullish retest (testing old resistance as new support), -1.0 for bearish retest (testing old support as new resistance), 0.0 for none."
        if "rule_vote" in col_name:
            return "1. Deterministic 4th Voter combining Dow trend, S/R proximity, and retest confirmation.\n2. Returns categorical voting score: +1.0 (Buy/Long), -1.0 (Sell/Short), 0.0 (Neutral/No Action)."

    # Fallback to formula and description
    return f"1. Evaluates inputs: {meta.get('source_columns', 'OHLCV')}.\n2. Mathematical Formula: `{formula}`.\n3. Operational Definition: {desc}."


def generate_markdown() -> str:
    md = []
    md.append("# BTC/USDT 5-Minute Causal Feature Engineering & Calculation Manual")
    md.append("")
    md.append("> **Document Scope:** Exhaustive calculation reference manual for all 136 causal machine learning features engineered for 5-minute Bitcoin (`BTCUSDT`) spot trading.")
    md.append("> **Anti-Leakage Guarantee:** Every feature at timestamp $t$ is strictly causal, computed exclusively from closed candles $\\le t$. Zero future leakage ($t+1$ or later) across all 136 features.")
    md.append("> **Scale Invariance:** Zero raw USD price levels are exposed to models; all inputs are transformed into unitless stationary log-returns, moving average ratios, Z-scores, ATR-normalized distances, or bounded cyclical coordinates.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Table of Contents")
    md.append("")
    
    idx = 1
    for grp_key in FEATURE_GROUPS.keys():
        info = GROUP_INFO.get(grp_key, {})
        title = info.get("title", grp_key)
        count = len(FEATURE_GROUPS[grp_key])
        md.append(f"{idx}. [{title} (Group: `{grp_key}`, {count} Features)](#{grp_key})")
        idx += 1
    md.append(f"{idx}. [Summary Taxonomy & Model Deployment Scope](#summary-taxonomy)")
    md.append("")
    md.append("---")
    md.append("")

    total_counter = 0
    for grp_key, cols in FEATURE_GROUPS.items():
        info = GROUP_INFO.get(grp_key, {})
        title = info.get("title", grp_key)
        grp_desc = info.get("description", "")
        inputs = info.get("input_series", "OHLCV")
        
        md.append(f"<a id=\"{grp_key}\"></a>")
        md.append(f"## {title}")
        md.append("")
        md.append(f"- **Group Key:** `{grp_key}`")
        md.append(f"- **Feature Count:** {len(cols)}")
        md.append(f"- **Raw Inputs Required:** `{inputs}`")
        md.append(f"- **Overview:** {grp_desc}")
        md.append("")
        
        # Summary table for the group
        md.append("| # | Feature Column Name | Formula | Unit / Scale | Description |")
        md.append("| :-: | :--- | :--- | :--- | :--- |")
        for c in cols:
            total_counter += 1
            meta = cd.get_column_meta(c)
            formula_esc = meta.get("formula", "").replace("|", "\\|")
            desc_short = meta.get("description", "").replace("|", "\\|")
            unit_val = meta.get("unit", "").replace("|", "\\|")
            md.append(f"| {total_counter} | **`{c}`** | `{formula_esc}` | {unit_val} | {desc_short} |")
        md.append("")

        # Detailed cards for each feature
        md.append("### Detailed Calculation Breakdowns")
        md.append("")
        for c in cols:
            meta = cd.get_column_meta(c)
            calc_details = get_detailed_calculation(c, meta)
            
            md.append(f"#### `{c}`")
            md.append(f"- **Description:** {meta.get('description', '')}")
            md.append(f"- **Mathematical Formula:** `{meta.get('formula', '')}`")
            md.append(f"- **Unit / Range:** `{meta.get('unit', '')}`")
            md.append(f"- **Causality / Lookback:** Lookback is strictly backward-looking (closed candles only, zero lookahead).")
            md.append(f"- **Calculation Steps:**")
            for step_line in calc_details.split("\n"):
                md.append(f"  {step_line}")
            md.append("")
        
        md.append("---")
        md.append("")

    # Summary section
    md.append("<a id=\"summary-taxonomy\"></a>")
    md.append("## Summary Taxonomy & Model Deployment Scope")
    md.append("")
    md.append("### 1. Total Feature Count by Group")
    md.append("")
    md.append("| Feature Group | Number of Features | Primary Domain |")
    md.append("| :--- | :-: | :--- |")
    for grp_key, cols in FEATURE_GROUPS.items():
        info = GROUP_INFO.get(grp_key, {})
        md.append(f"| **`{grp_key}`** | {len(cols)} | {info.get('title', grp_key)} |")
    md.append(f"| **TOTAL** | **{len(ALL_FEATURE_COLUMNS)}** | **Full 17-Domain Causal Space** |")
    md.append("")
    md.append("### 2. Horizon-2 Deployment (10-Minute Forecast)")
    md.append("- **Active Features (101 Features):** Dedicated to Horizon 2 (2 candles ahead = 10m). Excludes redundant copies belonging to alternative forecasting horizons (3, 6, 12).")
    md.append("- **Tree Ensembles (LightGBM, XGBoost, CatBoost, Random Forest):** Ingest full feature vector without requiring scale transformation.")
    md.append("- **Linear Models (Ridge, Logistic Regression):** Standardized using `StandardScaler` fitted exclusively on training splits (Folds 1..5) with frozen parameters.")
    md.append("- **Meta-Model Combiner (15 Inputs):** Stacks 9 sub-model probability outputs + 6 continuous market condition features (`feat_vol_natr_14`, `feat_mom_rsi_14`, `feat_mom_adx_14`, `feat_of_taker_buy_ratio`, `feat_ma_alignment_sma_5m`, `feat_struct_rule_vote`).")
    md.append("")

    return "\n".join(md)


def main():
    content = generate_markdown()
    
    # Save to ML/reports/features_calculation_reference.md
    out_path_1 = ML_DIR / "reports" / "features_calculation_reference.md"
    out_path_1.parent.mkdir(parents=True, exist_ok=True)
    out_path_1.write_text(content, encoding="utf-8")
    print(f"✓ Generated {out_path_1} ({len(content)} bytes)")

    # Save to root reports/features_calculation_reference.md
    root_reports = ML_DIR.parent / "reports"
    root_reports.mkdir(parents=True, exist_ok=True)
    out_path_2 = root_reports / "features_calculation_reference.md"
    out_path_2.write_text(content, encoding="utf-8")
    print(f"✓ Generated {out_path_2} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
