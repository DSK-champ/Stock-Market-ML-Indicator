"""Causal feature engineering — Stock-Market-ML-Predictor v3.

All features are computed from CLOSED candle history only. No feature at time t
uses price or volume information from t+1 or later.

Feature groups:
---------------
1. returns       : log-return lags (1, 3, 6, 12, 24) and open gap
2. volatility    : causal rolling std (12, 60, 288), vol ratios, natr_14, bollinger %B and bandwidth, range_12
3. candle_shape  : body pct, body-to-range, upper/lower wicks, close position in range, candle dir
                   (Note: hl_spread is excluded to avoid collinearity: hl_spread = body + upper_wick + lower_wick)
4. momentum      : SMA ratios (10, 20, 50), EMA ratios (10, 20, 50), SMA cross, RSI (7, 14), MACD, ADX (14)
5. volume        : log volume, rolling z-scores (12, 60), 1-candle change, volume ratio 12/60
6. order_flow    : taker-buy ratio, taker-buy z-score (60), trades z-scores (12, 60), average trade size, avg trade size z-score
                   (Note: taker_imbalance is dropped as it is a pure affine copy of taker_buy_ratio)
7. time          : cyclical sin/cos hour of day and day of week
8. mtf_1h        : 1h multi-timeframe features from COMPLETED 1h candles only (shifted by 1h, ffilled)
9. vol_range     : Parkinson & Garman-Klass volatility over 12, 60, 288 candles
10. vol_realized : realized absolute variation sums, max abs returns, upside/downside semivariance shares
11. vol_seasonal : 7-day same-slot mean absolute return, vol ratio 12, same-slot volume and trade ratios
12. vol_har      : HAR realized volatility over 12, 72, 288, 2016 candles and ratios
13. flow_window  : normalized cumulative taker net flow and return-signed flow over 6, 12, 36 candles
14. location     : close position in 288-candle range, fractional distances to high/low
15. time_weekend : weekend indicator and 168-hour continuous cyclical encoding

Warm-up:
--------
LOOKBACK_CANDLES = 2018 (minimum rows needed before all features are strictly populated; 7d buffer).
No NaN or inf after LOOKBACK_CANDLES.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src import config
from src.utils import rolling_sigma, setup_logger

logger = setup_logger("features")

LOOKBACK_CANDLES: int = config.LOOKBACK_CANDLES  # 288 candles (24 hours of 5m bars)

# ---------------------------------------------------------------------------
# Feature Group Definitions
# ---------------------------------------------------------------------------
FEATURE_GROUPS: Dict[str, List[str]] = {
    "returns": [
        "feat_ret_1",
        "feat_ret_3",
        "feat_ret_6",
        "feat_ret_12",
        "feat_ret_24",
    ],
    "volatility": [
        "feat_vol_sigma_288",
        "feat_vol_sigma_60",
        "feat_vol_sigma_12",
        "feat_vol_ratio_60_288",
        "feat_vol_ratio_12_60",
        "feat_vol_natr_14",
        "feat_vol_bb_pct_20",
        "feat_vol_bb_width_20",
        "feat_vol_range_12",
    ],
    "candle_shape": [
        "feat_shape_body_pct",
        "feat_shape_body_to_range",
        "feat_shape_upper_wick_pct",
        "feat_shape_lower_wick_pct",
        "feat_shape_close_pos",
        "feat_shape_candle_dir",
    ],
    "momentum": [
        "feat_mom_sma_ratio_10",
        "feat_mom_sma_ratio_20",
        "feat_mom_sma_ratio_50",
        "feat_mom_ema_ratio_10",
        "feat_mom_ema_ratio_20",
        "feat_mom_ema_ratio_50",
        "feat_mom_sma_ratio_10_50",
        "feat_mom_rsi_7",
        "feat_mom_rsi_14",
        "feat_mom_macd_line",
        "feat_mom_macd_signal",
        "feat_mom_macd_hist",
        "feat_mom_adx_14",
    ],
    "volume": [
        "feat_volume_log",
        "feat_volume_zscore_60",
        "feat_volume_zscore_12",
        "feat_volume_change_1",
        "feat_volume_ratio_12_60",
    ],
    "order_flow": [
        "feat_of_taker_buy_ratio",
        "feat_of_taker_buy_zscore_60",
        "feat_of_trades_zscore_60",
        "feat_of_trades_zscore_12",
        "feat_of_avg_trade_size",
        "feat_of_avg_trade_size_zscore_60",
    ],
    "time": [
        "feat_time_sin_hour",
        "feat_time_cos_hour",
        "feat_time_sin_weekday",
        "feat_time_cos_weekday",
    ],
    "mtf_1h": [
        "feat_mtf_1h_ret",
        "feat_mtf_1h_range",
        "feat_mtf_1h_vol_ratio",
        "feat_mtf_1h_close_vs_sma_6",
        "feat_mtf_1h_rsi_6",
    ],
    "vol_range": [
        "feat_vol_parkinson_12",
        "feat_vol_parkinson_60",
        "feat_vol_parkinson_288",
        "feat_vol_gk_12",
        "feat_vol_gk_60",
        "feat_vol_gk_288",
    ],
    "vol_realized": [
        "feat_vol_realized_abs_sum_6",
        "feat_vol_realized_abs_sum_12",
        "feat_vol_realized_abs_sum_36",
        "feat_vol_realized_max_abs_12",
        "feat_vol_realized_max_abs_60",
        "feat_vol_realized_semi_var_up_ratio_60",
    ],
    "vol_seasonal": [
        "feat_vol_seasonal_slot_abs_ret_7d",
        "feat_vol_seasonal_vol_ratio_12",
        "feat_vol_seasonal_volume_ratio_7d",
        "feat_vol_seasonal_trades_ratio_7d",
    ],
    "vol_har": [
        "feat_vol_har_rv_72",
        "feat_vol_har_rv_2016",
        "feat_vol_har_ratio_12_288",
        "feat_vol_har_ratio_288_2016",
    ],
    "flow_window": [
        "feat_flow_window_net_6",
        "feat_flow_window_net_12",
        "feat_flow_window_net_36",
        "feat_flow_window_signed_flow_6",
        "feat_flow_window_signed_flow_12",
        "feat_flow_window_signed_flow_36",
    ],
    "location": [
        "feat_loc_position_288",
        "feat_loc_dist_high_288",
        "feat_loc_dist_low_288",
    ],
    "time_weekend": [
        "feat_time_is_weekend",
        "feat_time_sin_week",
        "feat_time_cos_week",
    ],
    "ma_structure": [
        # 5m MA ratios (windows 5, 25, 100) — 10 and 50 are reused from momentum group
        "feat_ma_sma_ratio_5",
        "feat_ma_sma_ratio_25",
        "feat_ma_sma_ratio_100",
        "feat_ma_ema_ratio_5",
        "feat_ma_ema_ratio_25",
        "feat_ma_ema_ratio_100",
        # 5m MA slopes (windows 5, 10, 25, 50, 100) — MA_t / MA_{t-5} - 1
        "feat_ma_sma_slope_5",
        "feat_ma_sma_slope_10",
        "feat_ma_sma_slope_25",
        "feat_ma_sma_slope_50",
        "feat_ma_sma_slope_100",
        "feat_ma_ema_slope_5",
        "feat_ma_ema_slope_10",
        "feat_ma_ema_slope_25",
        "feat_ma_ema_slope_50",
        "feat_ma_ema_slope_100",
        # 5m MA alignment scores in [-1, 1]
        "feat_ma_alignment_sma_5m",
        "feat_ma_alignment_ema_5m",
        # 1h completed MA ratios (windows 5, 10, 25, 50, 100) — (close_5m - MA_1h) / MA_1h
        "feat_ma_1h_sma_ratio_5",
        "feat_ma_1h_sma_ratio_10",
        "feat_ma_1h_sma_ratio_25",
        "feat_ma_1h_sma_ratio_50",
        "feat_ma_1h_sma_ratio_100",
        "feat_ma_1h_ema_ratio_5",
        "feat_ma_1h_ema_ratio_10",
        "feat_ma_1h_ema_ratio_25",
        "feat_ma_1h_ema_ratio_50",
        "feat_ma_1h_ema_ratio_100",
        # 1h completed MA slopes (windows 5, 10, 25, 50, 100) — MA_{t,1h} / MA_{t-5,1h} - 1
        "feat_ma_1h_sma_slope_5",
        "feat_ma_1h_sma_slope_10",
        "feat_ma_1h_sma_slope_25",
        "feat_ma_1h_sma_slope_50",
        "feat_ma_1h_sma_slope_100",
        # 1h MA alignment scores in [-1, 1]
        "feat_ma_alignment_sma_1h",
        "feat_ma_alignment_ema_1h",
        # 1h Wilder ATR-14 normalized by 5m close
        "feat_ma_1h_natr_14",
    ],
    "structure": [
        "feat_struct_dist_support_atr",
        "feat_struct_dist_support_pct",
        "feat_struct_dist_resistance_atr",
        "feat_struct_dist_resistance_pct",
        "feat_struct_support_strength",
        "feat_struct_resistance_strength",
        "feat_struct_trend_up",
        "feat_struct_trend_down",
        "feat_struct_trend_range",
        "feat_struct_trend_age",
        "feat_struct_pullback_depth_atr",
        "feat_struct_bars_since_break",
        "feat_struct_bars_since_retest",
        "feat_struct_retest_direction",
        "feat_struct_rule_vote",
    ],
}

ALL_FEATURE_COLUMNS: List[str] = [col for cols in FEATURE_GROUPS.values() for col in cols]


# ===========================================================================
# Internal Causal Math Helpers
# ===========================================================================

def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Smoothed Moving Average (RMA) — strictly causal recursive EMA."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range: max(H - L, |H - prev_C|, |L - prev_C|) — causal."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range via Wilder's RMA — causal."""
    return _wilder_rma(_true_range(df), period)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100] — causal."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Where loss was 0 but gain > 0 -> 100; where both 0 -> 50
    rsi = rsi.where(avg_loss != 0.0, np.where(avg_gain != 0.0, 100.0, 50.0))
    return rsi


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Standard causal EMA (adjust=False)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) in [0, 100] — causal."""
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    dm_plus = (high - prev_high).clip(lower=0.0)
    dm_minus = (prev_low - low).clip(lower=0.0)
    both_pos = (dm_plus > 0) & (dm_minus > 0)
    larger_plus = dm_plus >= dm_minus
    dm_plus = dm_plus.where(~both_pos | larger_plus, 0.0)
    dm_minus = dm_minus.where(~both_pos | ~larger_plus, 0.0)

    tr = _true_range(df)
    atr_smooth = _wilder_rma(tr, period)
    di_plus_smooth = _wilder_rma(dm_plus, period)
    di_minus_smooth = _wilder_rma(dm_minus, period)

    di_plus = 100.0 * di_plus_smooth / atr_smooth.replace(0.0, np.nan)
    di_minus = 100.0 * di_minus_smooth / atr_smooth.replace(0.0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    adx = _wilder_rma(dx, period)
    return adx


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling Z-score: (x - mean) / std with min_periods=2."""
    mu = series.rolling(window, min_periods=2).mean()
    sd = series.rolling(window, min_periods=2).std()
    sd_clean = sd.replace(0.0, np.nan)
    z = (series - mu) / sd_clean
    return z.fillna(0.0)


# ===========================================================================
# Main Feature Builder
# ===========================================================================

def build_features(df: pd.DataFrame, sigma_window: int = 288) -> pd.DataFrame:
    """Build all 53 causal features from cleaned OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: open, high, low, close, volume,
        taker_buy_base_volume, number_of_trades.
        UTC DatetimeIndex.
    sigma_window : int
        Window for long-run rolling sigma (default 288 candles = 24h).

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly ALL_FEATURE_COLUMNS, aligned to df.index.
        First LOOKBACK_CANDLES rows contain warm-up NaNs; zero NaNs after LOOKBACK_CANDLES.
    """
    required = {
        "open", "high", "low", "close", "volume",
        "taker_buy_base_volume", "number_of_trades"
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df is missing required columns: {missing}")

    out = {}
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    eps = 1e-12

    # -----------------------------------------------------------
    # G1: Returns (open_gap removed per Section 8 decision)
    # -----------------------------------------------------------
    out["feat_ret_1"] = np.log(close / close.shift(1).replace(0.0, np.nan))
    out["feat_ret_3"] = np.log(close / close.shift(3).replace(0.0, np.nan))
    out["feat_ret_6"] = np.log(close / close.shift(6).replace(0.0, np.nan))
    out["feat_ret_12"] = np.log(close / close.shift(12).replace(0.0, np.nan))
    out["feat_ret_24"] = np.log(close / close.shift(24).replace(0.0, np.nan))

    # -----------------------------------------------------------
    # G2: Volatility
    # -----------------------------------------------------------
    ret_1 = out["feat_ret_1"]
    out["feat_vol_sigma_288"] = rolling_sigma(ret_1, window=sigma_window)
    out["feat_vol_sigma_60"] = rolling_sigma(ret_1, window=60)
    out["feat_vol_sigma_12"] = rolling_sigma(ret_1, window=12)
    out["feat_vol_ratio_60_288"] = out["feat_vol_sigma_60"] / out["feat_vol_sigma_288"].replace(0.0, np.nan)
    out["feat_vol_ratio_12_60"] = out["feat_vol_sigma_12"] / out["feat_vol_sigma_60"].replace(0.0, np.nan)

    atr14 = _atr(df, period=14)
    out["feat_vol_natr_14"] = atr14 / close.replace(0.0, np.nan)

    bb_mean = close.rolling(20, min_periods=2).mean()
    bb_std = close.rolling(20, min_periods=2).std()
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std
    bb_band = (bb_upper - bb_lower).replace(0.0, np.nan)
    out["feat_vol_bb_pct_20"] = (close - bb_lower) / bb_band
    out["feat_vol_bb_width_20"] = bb_band / bb_mean.replace(0.0, np.nan)

    range_12_raw = high.rolling(12, min_periods=2).max() - low.rolling(12, min_periods=2).min()
    out["feat_vol_range_12"] = range_12_raw / close.replace(0.0, np.nan)

    # -----------------------------------------------------------
    # G3: Candle shape
    # -----------------------------------------------------------
    candle_span = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    out["feat_shape_body_pct"] = body / close.replace(0.0, np.nan)
    out["feat_shape_body_to_range"] = (body / candle_span).fillna(0.0)
    out["feat_shape_upper_wick_pct"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / close.replace(0.0, np.nan)
    out["feat_shape_lower_wick_pct"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / close.replace(0.0, np.nan)
    out["feat_shape_close_pos"] = ((close - low) / candle_span).fillna(0.5)
    out["feat_shape_candle_dir"] = np.sign(close - open_)

    # -----------------------------------------------------------
    # G4: Momentum
    # -----------------------------------------------------------
    sma10 = close.rolling(10, min_periods=2).mean()
    sma20 = close.rolling(20, min_periods=2).mean()
    sma50 = close.rolling(50, min_periods=2).mean()
    out["feat_mom_sma_ratio_10"] = (close - sma10) / sma10.replace(0.0, np.nan)
    out["feat_mom_sma_ratio_20"] = (close - sma20) / sma20.replace(0.0, np.nan)
    out["feat_mom_sma_ratio_50"] = (close - sma50) / sma50.replace(0.0, np.nan)

    ema10 = _ema(close, 10)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    out["feat_mom_ema_ratio_10"] = (close - ema10) / ema10.replace(0.0, np.nan)
    out["feat_mom_ema_ratio_20"] = (close - ema20) / ema20.replace(0.0, np.nan)
    out["feat_mom_ema_ratio_50"] = (close - ema50) / ema50.replace(0.0, np.nan)
    out["feat_mom_sma_ratio_10_50"] = (sma10 - sma50) / sma50.replace(0.0, np.nan)

    out["feat_mom_rsi_7"] = _rsi(close, 7) / 100.0
    out["feat_mom_rsi_14"] = _rsi(close, 14) / 100.0

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_raw = ema12 - ema26
    macd_sig_raw = _ema(macd_raw, 9)
    out["feat_mom_macd_line"] = macd_raw / close.replace(0.0, np.nan)
    out["feat_mom_macd_signal"] = macd_sig_raw / close.replace(0.0, np.nan)
    out["feat_mom_macd_hist"] = (macd_raw - macd_sig_raw) / close.replace(0.0, np.nan)

    out["feat_mom_adx_14"] = _adx(df, 14) / 100.0

    # -----------------------------------------------------------
    # G5: Volume
    # -----------------------------------------------------------
    out["feat_volume_log"] = np.log1p(vol.clip(lower=0.0))
    out["feat_volume_zscore_60"] = _zscore(vol, 60)
    out["feat_volume_zscore_12"] = _zscore(vol, 12)
    out["feat_volume_change_1"] = (vol / vol.shift(1).replace(0.0, np.nan) - 1.0).fillna(0.0)

    vol_sma12 = vol.rolling(12, min_periods=2).mean()
    vol_sma60 = vol.rolling(60, min_periods=2).mean()
    out["feat_volume_ratio_12_60"] = (vol_sma12 / vol_sma60.replace(0.0, np.nan)).fillna(1.0)

    # -----------------------------------------------------------
    # G6: Order flow
    # -----------------------------------------------------------
    tbv = df["taker_buy_base_volume"].clip(lower=0.0)
    taker_ratio = (tbv / vol.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    out["feat_of_taker_buy_ratio"] = taker_ratio
    out["feat_of_taker_buy_zscore_60"] = _zscore(taker_ratio, 60)

    trades = df["number_of_trades"].clip(lower=0.0)
    out["feat_of_trades_zscore_60"] = _zscore(trades, 60)
    out["feat_of_trades_zscore_12"] = _zscore(trades, 12)

    avg_size = vol / trades.replace(0.0, np.nan)
    out["feat_of_avg_trade_size"] = avg_size.fillna(0.0)
    out["feat_of_avg_trade_size_zscore_60"] = _zscore(avg_size, 60)

    # -----------------------------------------------------------
    # G7: Time
    # -----------------------------------------------------------
    hours = df.index.hour.to_series(index=df.index).astype(float)
    weekdays = pd.Series(df.index.day_of_week if hasattr(df.index, "day_of_week") else df.index.dayofweek, index=df.index).astype(float)
    out["feat_time_sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0)
    out["feat_time_cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0)
    out["feat_time_sin_weekday"] = np.sin(2.0 * np.pi * weekdays / 7.0)
    out["feat_time_cos_weekday"] = np.cos(2.0 * np.pi * weekdays / 7.0)

    # -----------------------------------------------------------
    # G8: Multi-timeframe 1h features from COMPLETED 1h candles only
    # -----------------------------------------------------------
    # Resample 5m bars to 1h bars (closed='left', label='left')
    df_1h = df[["open", "high", "low", "close", "volume"]].resample(
        "1h", closed="left", label="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Features computed on 1h bars
    ret_1h_raw = np.log(df_1h["close"] / df_1h["open"].replace(0.0, np.nan))
    range_1h_raw = (df_1h["high"] - df_1h["low"]) / df_1h["close"].replace(0.0, np.nan)
    vol_mean_24_1h = df_1h["volume"].rolling(24, min_periods=2).mean()
    vol_ratio_1h_raw = (df_1h["volume"] / vol_mean_24_1h.replace(0.0, np.nan)).fillna(1.0)
    sma6_1h = df_1h["close"].rolling(6, min_periods=2).mean()
    close_vs_sma6_1h_raw = (df_1h["close"] - sma6_1h) / sma6_1h.replace(0.0, np.nan)
    rsi6_1h_raw = _rsi(df_1h["close"], 6) / 100.0

    # Strict causality: shift 1h metrics by 1 full hour so hour H is only known at H+1h onwards
    ret_1h_shifted = ret_1h_raw.shift(1)
    range_1h_shifted = range_1h_raw.shift(1)
    vol_ratio_1h_shifted = vol_ratio_1h_raw.shift(1)
    close_vs_sma6_1h_shifted = close_vs_sma6_1h_raw.shift(1)
    rsi6_1h_shifted = rsi6_1h_raw.shift(1)

    # Align / forward fill onto 5m index
    out["feat_mtf_1h_ret"] = ret_1h_shifted.reindex(df.index, method="ffill")
    out["feat_mtf_1h_range"] = range_1h_shifted.reindex(df.index, method="ffill")
    out["feat_mtf_1h_vol_ratio"] = vol_ratio_1h_shifted.reindex(df.index, method="ffill")
    out["feat_mtf_1h_close_vs_sma_6"] = close_vs_sma6_1h_shifted.reindex(df.index, method="ffill")
    out["feat_mtf_1h_rsi_6"] = rsi6_1h_shifted.reindex(df.index, method="ffill")

    # -----------------------------------------------------------
    # G9: vol_range (Parkinson and Garman-Klass over 12, 60, 288)
    # -----------------------------------------------------------
    log_hl = np.log(high / low.replace(0.0, np.nan))
    log_hl_sq = log_hl ** 2
    parkinson_const = 1.0 / (4.0 * np.log(2.0))
    for w in [12, 60, 288]:
        out[f"feat_vol_parkinson_{w}"] = np.sqrt(
            parkinson_const * log_hl_sq.rolling(w, min_periods=2).mean()
        )

    log_co = np.log(close / open_.replace(0.0, np.nan))
    log_co_sq = log_co ** 2
    gk_const = 2.0 * np.log(2.0) - 1.0
    gk_sq = 0.5 * log_hl_sq - gk_const * log_co_sq
    for w in [12, 60, 288]:
        out[f"feat_vol_gk_{w}"] = np.sqrt(
            gk_sq.rolling(w, min_periods=2).mean().clip(lower=0.0)
        )

    # -----------------------------------------------------------
    # G10: vol_realized (abs return sums, max abs returns, semivariance shares)
    # -----------------------------------------------------------
    abs_ret_1 = ret_1.abs()
    out["feat_vol_realized_abs_sum_6"] = abs_ret_1.rolling(6, min_periods=2).sum()
    out["feat_vol_realized_abs_sum_12"] = abs_ret_1.rolling(12, min_periods=2).sum()
    out["feat_vol_realized_abs_sum_36"] = abs_ret_1.rolling(36, min_periods=2).sum()
    out["feat_vol_realized_max_abs_12"] = abs_ret_1.rolling(12, min_periods=2).max()
    out["feat_vol_realized_max_abs_60"] = abs_ret_1.rolling(60, min_periods=2).max()

    ret_pos_sq = ret_1.clip(lower=0.0) ** 2
    ret_neg_sq = (-ret_1).clip(lower=0.0) ** 2
    semi_up_60 = ret_pos_sq.rolling(60, min_periods=2).sum()
    semi_down_60 = ret_neg_sq.rolling(60, min_periods=2).sum()
    semi_tot_60 = semi_up_60 + semi_down_60
    out["feat_vol_realized_semi_var_up_ratio_60"] = (semi_up_60 / semi_tot_60.replace(0.0, np.nan)).fillna(0.5)

    # -----------------------------------------------------------
    # G11: vol_seasonal (7-day same-slot mean absolute return and ratios)
    # -----------------------------------------------------------
    seasonal_offsets = [288 * k for k in range(1, 8)]  # [288, 576, 864, 1152, 1440, 1728, 2016]
    slot_abs_ret_7d = sum(abs_ret_1.shift(k) for k in seasonal_offsets) / 7.0
    out["feat_vol_seasonal_slot_abs_ret_7d"] = slot_abs_ret_7d

    curr_vol_12 = out["feat_vol_realized_abs_sum_12"] / 12.0
    out["feat_vol_seasonal_vol_ratio_12"] = (curr_vol_12 / slot_abs_ret_7d.replace(0.0, np.nan)).fillna(1.0)

    slot_vol_7d = sum(vol.shift(k) for k in seasonal_offsets) / 7.0
    out["feat_vol_seasonal_volume_ratio_7d"] = (vol / slot_vol_7d.replace(0.0, np.nan)).fillna(1.0)

    slot_tr_7d = sum(trades.shift(k) for k in seasonal_offsets) / 7.0
    out["feat_vol_seasonal_trades_ratio_7d"] = (trades / slot_tr_7d.replace(0.0, np.nan)).fillna(1.0)

    # -----------------------------------------------------------
    # G12: vol_har (realized volatility over 1h, 6h, 24h, 7d and ratios)
    # -----------------------------------------------------------
    # Note: har_rv_12 and har_rv_288 are exact mathematical duplicates of feat_vol_sigma_12
    # and feat_vol_sigma_288; they are used locally to compute HAR ratios but excluded from feature columns.
    rv_12 = ret_1.rolling(12, min_periods=2).std()
    rv_288 = ret_1.rolling(288, min_periods=2).std()
    out["feat_vol_har_rv_72"] = ret_1.rolling(72, min_periods=2).std()
    out["feat_vol_har_rv_2016"] = ret_1.rolling(2016, min_periods=2).std()
    out["feat_vol_har_ratio_12_288"] = (rv_12 / rv_288.replace(0.0, np.nan)).fillna(1.0)
    out["feat_vol_har_ratio_288_2016"] = (rv_288 / out["feat_vol_har_rv_2016"].replace(0.0, np.nan)).fillna(1.0)

    # -----------------------------------------------------------
    # G13: flow_window (cumulative taker net flow and signed flow over 6, 12, 36)
    # -----------------------------------------------------------
    flow_raw = 2.0 * tbv - vol
    sign_ret_1 = np.sign(ret_1).fillna(0.0)
    for w in [6, 12, 36]:
        flow_sum_w = flow_raw.rolling(w, min_periods=1).sum()
        vol_sum_w = vol.rolling(w, min_periods=1).sum()
        norm_flow_w = (flow_sum_w / vol_sum_w.replace(0.0, np.nan)).fillna(0.0)
        out[f"feat_flow_window_net_{w}"] = norm_flow_w
        out[f"feat_flow_window_signed_flow_{w}"] = norm_flow_w * sign_ret_1

    # -----------------------------------------------------------
    # G14: location (288-candle high-low range position and distances)
    # -----------------------------------------------------------
    high_288 = high.rolling(288, min_periods=2).max()
    low_288 = low.rolling(288, min_periods=2).min()
    span_288 = (high_288 - low_288).replace(0.0, np.nan)
    out["feat_loc_position_288"] = ((close - low_288) / span_288).clip(0.0, 1.0).fillna(0.5)
    out["feat_loc_dist_high_288"] = ((high_288 - close) / close.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    out["feat_loc_dist_low_288"] = ((close - low_288) / close.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)

    # -----------------------------------------------------------
    # G15: time_weekend (weekend indicator and 168-hour cyclical encoding)
    # -----------------------------------------------------------
    out["feat_time_is_weekend"] = np.where(weekdays >= 5.0, 1.0, 0.0)
    minutes = df.index.minute.to_series(index=df.index).astype(float)
    hour_of_week = weekdays * 24.0 + hours + minutes / 60.0
    out["feat_time_sin_week"] = np.sin(2.0 * np.pi * hour_of_week / 168.0)
    out["feat_time_cos_week"] = np.cos(2.0 * np.pi * hour_of_week / 168.0)

    # -----------------------------------------------------------
    # G16: ma_structure (MAs on 5m and completed 1h, slopes, alignment, 1h ATR)
    # -----------------------------------------------------------
    # 1. 5m MAs: windows 5, 10, 25, 50, 100
    sma5 = close.rolling(5, min_periods=2).mean()
    sma25 = close.rolling(25, min_periods=2).mean()
    sma100 = close.rolling(100, min_periods=2).mean()

    ema5 = _ema(close, 5)
    ema25 = _ema(close, 25)
    ema100 = _ema(close, 100)

    # 5m MA ratios (5, 25, 100) — 10 and 50 reused from momentum group
    out["feat_ma_sma_ratio_5"] = (close - sma5) / sma5.replace(0.0, np.nan)
    out["feat_ma_sma_ratio_25"] = (close - sma25) / sma25.replace(0.0, np.nan)
    out["feat_ma_sma_ratio_100"] = (close - sma100) / sma100.replace(0.0, np.nan)
    out["feat_ma_ema_ratio_5"] = (close - ema5) / ema5.replace(0.0, np.nan)
    out["feat_ma_ema_ratio_25"] = (close - ema25) / ema25.replace(0.0, np.nan)
    out["feat_ma_ema_ratio_100"] = (close - ema100) / ema100.replace(0.0, np.nan)

    # 5m MA slopes: MA_t / MA_{t-5} - 1 (5-candle lag)
    out["feat_ma_sma_slope_5"] = (sma5 / sma5.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_sma_slope_10"] = (sma10 / sma10.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_sma_slope_25"] = (sma25 / sma25.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_sma_slope_50"] = (sma50 / sma50.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_sma_slope_100"] = (sma100 / sma100.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)

    out["feat_ma_ema_slope_5"] = (ema5 / ema5.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_ema_slope_10"] = (ema10 / ema10.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_ema_slope_25"] = (ema25 / ema25.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_ema_slope_50"] = (ema50 / ema50.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
    out["feat_ma_ema_slope_100"] = (ema100 / ema100.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)

    # 5m Alignment scores: 10 pairs from {5, 10, 25, 50, 100} in [-1, 1]
    sma_5m_list = [sma5, sma10, sma25, sma50, sma100]
    ema_5m_list = [ema5, ema10, ema25, ema50, ema100]
    sma_agree_5m = sum(np.where(sma_5m_list[i] > sma_5m_list[j], 1.0, 0.0) for i in range(5) for j in range(i + 1, 5))
    ema_agree_5m = sum(np.where(ema_5m_list[i] > ema_5m_list[j], 1.0, 0.0) for i in range(5) for j in range(i + 1, 5))
    out["feat_ma_alignment_sma_5m"] = (2.0 * sma_agree_5m - 10.0) / 10.0
    out["feat_ma_alignment_ema_5m"] = (2.0 * ema_agree_5m - 10.0) / 10.0

    # 2. Completed 1h MAs: windows 5, 10, 25, 50, 100 on df_1h
    sma_1h_dict = {}
    ema_1h_dict = {}
    for w in [5, 10, 25, 50, 100]:
        s1 = df_1h["close"].rolling(w, min_periods=2).mean()
        e1 = _ema(df_1h["close"], w)
        sma_1h_dict[w] = s1
        ema_1h_dict[w] = e1
        s1_ffill = s1.shift(1).reindex(df.index, method="ffill")
        e1_ffill = e1.shift(1).reindex(df.index, method="ffill")
        out[f"feat_ma_1h_sma_ratio_{w}"] = (close - s1_ffill) / s1_ffill.replace(0.0, np.nan)
        out[f"feat_ma_1h_ema_ratio_{w}"] = (close - e1_ffill) / e1_ffill.replace(0.0, np.nan)
        slope_1h = (s1 / s1.shift(5).replace(0.0, np.nan) - 1.0).fillna(0.0)
        out[f"feat_ma_1h_sma_slope_{w}"] = slope_1h.shift(1).reindex(df.index, method="ffill").fillna(0.0)

    # 1h Alignment scores in [-1, 1]
    sma_1h_list = [sma_1h_dict[w] for w in [5, 10, 25, 50, 100]]
    ema_1h_list = [ema_1h_dict[w] for w in [5, 10, 25, 50, 100]]
    sma_agree_1h = sum(np.where(sma_1h_list[i] > sma_1h_list[j], 1.0, 0.0) for i in range(5) for j in range(i + 1, 5))
    ema_agree_1h = sum(np.where(ema_1h_list[i] > ema_1h_list[j], 1.0, 0.0) for i in range(5) for j in range(i + 1, 5))
    align_sma_1h = pd.Series((2.0 * sma_agree_1h - 10.0) / 10.0, index=df_1h.index)
    align_ema_1h = pd.Series((2.0 * ema_agree_1h - 10.0) / 10.0, index=df_1h.index)
    out["feat_ma_alignment_sma_1h"] = align_sma_1h.shift(1).reindex(df.index, method="ffill").fillna(0.0)
    out["feat_ma_alignment_ema_1h"] = align_ema_1h.shift(1).reindex(df.index, method="ffill").fillna(0.0)

    # 1h Wilder ATR-14 normalized by 5m close
    atr_1h = _atr(df_1h, 14)
    out["feat_ma_1h_natr_14"] = (atr_1h.shift(1).reindex(df.index, method="ffill")) / close.replace(0.0, np.nan)

    # -----------------------------------------------------------
    # G17: structure (causal pivots, support/resistance, Dow trend, 4th voter)
    # -----------------------------------------------------------
    from src.structure import build_structure, STRUCTURE_FEATURE_COLUMNS
    df_struct, _ = build_structure(df)
    for col in STRUCTURE_FEATURE_COLUMNS:
        out[col] = df_struct[col]

    out_df = pd.DataFrame(out, index=df.index)

    # -----------------------------------------------------------
    # Gap boundary mask protection
    # -----------------------------------------------------------
    if "gap_id" in df.columns:
        gap = df["gap_id"]
        for col in ALL_FEATURE_COLUMNS:
            if "feat_struct_" in col:
                # Structure engine already handles gap resets internally
                continue
            elif "2016" in col or "seasonal" in col:
                w_gap = 2016
            elif "feat_ma_1h_" in col or "alignment_sma_1h" in col or "alignment_ema_1h" in col:
                w_gap = 1260  # 105 hours * 12
            elif "288" in col:
                w_gap = 288
            elif "100" in col:
                w_gap = 100
            elif "72" in col or "60" in col or "50" in col:
                w_gap = 72
            elif "36" in col:
                w_gap = 36
            elif "25" in col or "24" in col or "20" in col:
                w_gap = 25
            elif "14" in col or "12" in col or "10" in col:
                w_gap = 14
            elif "6" in col or "5" in col:
                w_gap = 6
            elif "3" in col:
                w_gap = 3
            else:
                w_gap = 2
            same_gap = (gap == gap.shift(w_gap - 1))
            out_df[col] = out_df[col].where(same_gap, np.nan)

    # Final check of columns
    missing_cols = set(ALL_FEATURE_COLUMNS) - set(out_df.columns)
    if missing_cols:
        raise RuntimeError(f"Engineered features missing expected columns: {missing_cols}")

    return out_df[ALL_FEATURE_COLUMNS]


def drop_warmup_rows(features_df: pd.DataFrame, n: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    """Drop the first n rows (warm-up period) to guarantee zero NaNs."""
    return features_df.iloc[n:].copy()


def get_feature_columns() -> List[str]:
    """Return all 88 engineered feature column names (role 'feature' only)."""
    return list(ALL_FEATURE_COLUMNS)
