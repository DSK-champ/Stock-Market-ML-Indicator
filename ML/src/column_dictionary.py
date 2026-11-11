"""Column dictionary for all data files in the Stock-Market-ML-Predictor project.

Every column in every data file (raw, cleaned, targets, features, master) must have an entry here.
Roles strictly follow the project taxonomy:
  market | feature | target | threshold | flag

Functions provided:
  get_column_meta(col_name) -> Dict
  get_feature_columns() -> List[str] (guarantees role == 'feature' only, no target/threshold leakage)
  load_master(split="train") -> pd.DataFrame (safe loader with test partition locking)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

from src import config
from src.features import ALL_FEATURE_COLUMNS, FEATURE_GROUPS


# ---------------------------------------------------------------------------
# Raw market data (Binance kline columns)
# ---------------------------------------------------------------------------
_RAW: Dict[str, Any] = {
    "open_time": {
        "description": "Candle open timestamp in Unix milliseconds (Binance kline field 1).",
        "unit": "ms (Unix epoch)",
        "formula": "",
        "group": "raw",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "close_time": {
        "description": "Candle close timestamp in Unix milliseconds (Binance kline field 7). "
                       "Equal to open_time + 5 minutes - 1 ms for 5m candles.",
        "unit": "ms (Unix epoch)",
        "formula": "open_time + 300_000 - 1",
        "group": "raw",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["open_time"],
    },
    "ignore": {
        "description": "Unused Binance kline field 12. Always '0'; dropped during preprocessing.",
        "unit": "none",
        "formula": "",
        "group": "raw",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
}

# ---------------------------------------------------------------------------
# Cleaned market data (clean.csv)
# ---------------------------------------------------------------------------
_CLEAN: Dict[str, Any] = {
    "open": {
        "description": "Candle open price in USDT (Binance kline field 2). "
                       "First traded price in the 5-minute interval.",
        "unit": "USDT",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,  # Model uses returns/ratios, not raw price levels
        "source_columns": [],
    },
    "high": {
        "description": "Highest traded price within the 5-minute candle (Binance kline field 3).",
        "unit": "USDT",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "low": {
        "description": "Lowest traded price within the 5-minute candle (Binance kline field 4).",
        "unit": "USDT",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "close": {
        "description": "Candle close price in USDT (Binance kline field 5). "
                       "Last traded price in the 5-minute interval. Used as reference for all labels.",
        "unit": "USDT",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "volume": {
        "description": "Total traded volume in BTC (base asset) during the candle (Binance kline field 6).",
        "unit": "BTC",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "quote_volume": {
        "description": "Total traded volume in USDT (quote asset) during the candle (Binance kline field 8).",
        "unit": "USDT",
        "formula": "sum(price_i × qty_i) for all trades in candle",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "number_of_trades": {
        "description": "Count of individual trades executed during the 5-minute candle (Binance kline field 9).",
        "unit": "count",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "taker_buy_base_volume": {
        "description": "Volume in BTC bought by taker orders during candle (Binance kline field 10).",
        "unit": "BTC",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "taker_buy_quote_volume": {
        "description": "Volume in USDT bought by taker orders during candle (Binance kline field 11).",
        "unit": "USDT",
        "formula": "",
        "group": "market",
        "role": "market",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "gap_id": {
        "description": "Contiguous segment identifier. Increments at every non-consecutive 5m gap. "
                       "Rows with same gap_id form a continuous OHLCV segment.",
        "unit": "integer index",
        "formula": "cumsum(is_gap_boundary)",
        "group": "metadata",
        "role": "flag",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["open_time"],
    },
    "is_gap_boundary": {
        "description": "True if row is the first candle of a new contiguous segment.",
        "unit": "boolean",
        "formula": "time_diff(open_time) > 5min OR row == 0",
        "group": "metadata",
        "role": "flag",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["open_time"],
    },
}

# ---------------------------------------------------------------------------
# Target columns
# ---------------------------------------------------------------------------
def _target_base_entries() -> Dict[str, Any]:
    """Generate column_dict entries for all targets_base.csv columns."""
    entries: Dict[str, Any] = {}
    for N in [2, 3, 6, 12]:
        minutes = N * 5
        entries[f"fwd_ret_{N}"] = {
            "description": f"Forward close-to-close return over {N} candles ({minutes} min).",
            "unit": "fraction",
            "formula": f"close[t+{N}] / close[t] - 1",
            "group": "targets_base",
            "role": "target",
            "is_future_looking": True,
            "allowed_as_model_input": False,
            "source_columns": ["close"],
        }
        entries[f"fwd_ret_open_{N}"] = {
            "description": f"Conservative forward return for backtesting over {N} candles ({minutes} min). Entry at open[t+1].",
            "unit": "fraction",
            "formula": f"close[t+{N}] / open[t+1] - 1",
            "group": "targets_base",
            "role": "target",
            "is_future_looking": True,
            "allowed_as_model_input": False,
            "source_columns": ["close", "open"],
        }
        entries[f"raw_sign_{N}"] = {
            "description": f"Raw directional sign of fwd_ret_{N} (+1, -1, 0).",
            "unit": "integer {-1, 0, +1}",
            "formula": f"sign(fwd_ret_{N})",
            "group": "targets_base",
            "role": "target",
            "is_future_looking": True,
            "allowed_as_model_input": False,
            "source_columns": [f"fwd_ret_{N}"],
        }
        entries[f"bin_dir_{N}"] = {
            "description": f"Binary directional target at horizon N={N} ({minutes} min): 1 if fwd_ret_{N} > 0, 0 if < 0, NaN if 0 or invalid.",
            "unit": "binary {0, 1}",
            "formula": f"1.0 if fwd_ret_{N} > 0 else (0.0 if fwd_ret_{N} < 0 else NaN)",
            "group": "targets_base",
            "role": "target",
            "is_future_looking": True,
            "allowed_as_model_input": False,
            "source_columns": [f"fwd_ret_{N}", f"raw_sign_{N}"],
        }
        entries[f"has_valid_target_{N}"] = {
            "description": f"True if row t has a valid forward label for horizon N={N} (no gap crossing, not in boundary).",
            "unit": "boolean",
            "formula": f"gap_id[t] == gap_id[t+{N}] AND t not in last {N} rows",
            "group": "targets_base",
            "role": "flag",
            "is_future_looking": True,
            "allowed_as_model_input": False,
            "source_columns": ["gap_id"],
        }
    return entries


_RULE_DESCRIPTIONS = {
    "S1": "Symmetric cost-floor threshold: theta_up = theta_down = c (0.23%).",
    "S2": "Symmetric fixed threshold: theta_up = theta_down = 0.30%.",
    "Q1": "Symmetric quantile threshold: theta = max(c, median |ret|). Floor binds (Q1 == S1).",
    "V1": "Volatility-adjusted threshold (half-sigma): theta = max(c, 0.5 * sigma * sqrt(N)).",
    "V2": "Volatility-adjusted threshold (one-sigma): theta = max(c, 1.0 * sigma * sqrt(N)).",
    "A1": "Asymmetric threshold: theta_up = 0.23%, theta_down = 0.30%.",
    "A2": "Asymmetric threshold: theta_up = 0.30%, theta_down = 0.23%.",
    "F05": "Research sub-cost threshold: theta_up = theta_down = 0.05% (move below cost c=0.23%; label is NOT tradable after costs; a signal on a move below c loses money).",
    "F10": "Research sub-cost threshold: theta_up = theta_down = 0.10% (move below cost c=0.23%; label is NOT tradable after costs; a signal on a move below c loses money).",
    "F15": "Research sub-cost threshold: theta_up = theta_down = 0.15% (move below cost c=0.23%; label is NOT tradable after costs; a signal on a move below c loses money).",
}


def _target_label_entries() -> Dict[str, Any]:
    """Generate column_dict entries for all targets_labels.csv columns."""
    entries: Dict[str, Any] = {}
    for N in [2, 3, 6, 12]:
        minutes = N * 5
        # big_move_N binary target for Setup D
        entries[f"big_move_{N}"] = {
            "description": f"1 if |forward return over {N} candles| exceeds the round-trip cost c (0.23%), else 0; basis for the worth-it model",
            "unit": "binary {0, 1} or NaN",
            "formula": f"1.0 if target_{N}_S1 != 0 else 0.0 (NaN if invalid)",
            "group": "targets_labels",
            "role": "target",
            "is_future_looking": True,
            "allowed_as_model_input": False,
            "source_columns": [f"target_{N}_S1"],
            "horizon": N,
        }
        for rule in ["S1", "S2", "Q1", "V1", "V2", "A1", "A2", "F05", "F10", "F15"]:
            rule_desc = _RULE_DESCRIPTIONS[rule]
            is_sub_cost = rule in ("F05", "F10", "F15")
            entries[f"target_{N}_{rule}"] = {
                "description": f"3-class label for horizon N={N} ({minutes} min), rule {rule} (+1 Up, -1 Down, 0 Neutral). {rule_desc}",
                "unit": "integer {-1, 0, +1} or NaN",
                "formula": f"Label rule {rule} evaluated against forward return fwd_ret_{N}",
                "group": "targets_labels",
                "role": "target",
                "is_future_looking": True,
                "allowed_as_model_input": False,
                "sub_cost": is_sub_cost,
                "source_columns": [f"fwd_ret_{N}"],
                "horizon": N,
                "theta_rule": rule,
            }
            entries[f"theta_up_{N}_{rule}"] = {
                "description": f"Up threshold for horizon N={N}, rule {rule}. {rule_desc}",
                "unit": "fraction",
                "formula": f"Rule {rule} threshold formula",
                "group": "targets_labels",
                "role": "threshold",
                "is_future_looking": False,
                "allowed_as_model_input": False,
                "source_columns": (["close"] if rule in ("V1", "V2") else []),
                "horizon": N,
                "theta_rule": rule,
            }
            entries[f"theta_down_{N}_{rule}"] = {
                "description": f"Down threshold for horizon N={N}, rule {rule}. {rule_desc}",
                "unit": "fraction",
                "formula": f"Rule {rule} threshold formula",
                "group": "targets_labels",
                "role": "threshold",
                "is_future_looking": False,
                "allowed_as_model_input": False,
                "source_columns": (["close"] if rule in ("V1", "V2") else []),
                "horizon": N,
                "theta_rule": rule,
            }
    return entries


# ---------------------------------------------------------------------------
# Engineered Feature Columns (88 features across 15 groups)
# ---------------------------------------------------------------------------
_FEATURE_DESCRIPTIONS: Dict[str, Tuple[str, str, str]] = {
    # Returns (G1)
    "feat_ret_1": ("Log return over 1 candle (5m)", "fraction", "ln(close[t] / close[t-1])"),
    "feat_ret_3": ("Log return over 3 candles (15m)", "fraction", "ln(close[t] / close[t-3])"),
    "feat_ret_6": ("Log return over 6 candles (30m)", "fraction", "ln(close[t] / close[t-6])"),
    "feat_ret_12": ("Log return over 12 candles (60m / 1h)", "fraction", "ln(close[t] / close[t-12])"),
    "feat_ret_24": ("Log return over 24 candles (120m / 2h)", "fraction", "ln(close[t] / close[t-24])"),

    # Volatility (G2)
    "feat_vol_sigma_288": ("Causal rolling return std over 288 candles (24h)", "fraction", "rolling_sigma(ret_1, 288)"),
    "feat_vol_sigma_60": ("Causal rolling return std over 60 candles (5h)", "fraction", "rolling_sigma(ret_1, 60)"),
    "feat_vol_sigma_12": ("Causal rolling return std over 12 candles (1h)", "fraction", "rolling_sigma(ret_1, 12)"),
    "feat_vol_ratio_60_288": ("Volatility ratio: 5h sigma / 24h sigma", "ratio", "sigma_60 / sigma_288"),
    "feat_vol_ratio_12_60": ("Volatility ratio: 1h sigma / 5h sigma", "ratio", "sigma_12 / sigma_60"),
    "feat_vol_natr_14": ("Normalized Average True Range over 14 candles", "ratio", "ATR(14) / close"),
    "feat_vol_bb_pct_20": ("Bollinger Bands %B (20-period, 2 sigma)", "ratio in [0, 1]", "(close - lower) / (upper - lower)"),
    "feat_vol_bb_width_20": ("Bollinger Bands bandwidth normalized by SMA", "ratio", "(upper - lower) / SMA(20)"),
    "feat_vol_range_12": ("Normalized 12-candle high-low range", "ratio", "(max(H_12) - min(L_12)) / close"),

    # Candle Shape (G3)
    "feat_shape_body_pct": ("Candle real body size relative to close price", "fraction", "|close - open| / close"),
    "feat_shape_body_to_range": ("Candle body size relative to full range [0, 1]", "ratio", "|close - open| / (high - low)"),
    "feat_shape_upper_wick_pct": ("Upper wick length relative to close price", "fraction", "(high - max(open, close)) / close"),
    "feat_shape_lower_wick_pct": ("Lower wick length relative to close price", "fraction", "(min(open, close) - low) / close"),
    "feat_shape_close_pos": ("Close position within candle range [0, 1]", "ratio", "(close - low) / (high - low)"),
    "feat_shape_candle_dir": ("Directional sign of candle body (+1 bullish, -1 bearish, 0 doji)", "integer {-1,0,1}", "sign(close - open)"),

    # Momentum (G4)
    "feat_mom_sma_ratio_10": ("Distance of close from 10-candle SMA", "ratio", "close / SMA(10) - 1.0"),
    "feat_mom_sma_ratio_20": ("Distance of close from 20-candle SMA", "ratio", "close / SMA(20) - 1.0"),
    "feat_mom_sma_ratio_50": ("Distance of close from 50-candle SMA", "ratio", "close / SMA(50) - 1.0"),
    "feat_mom_ema_ratio_10": ("Distance of close from 10-candle EMA", "ratio", "close / EMA(10) - 1.0"),
    "feat_mom_ema_ratio_20": ("Distance of close from 20-candle EMA", "ratio", "close / EMA(20) - 1.0"),
    "feat_mom_ema_ratio_50": ("Distance of close from 50-candle EMA", "ratio", "close / EMA(50) - 1.0"),
    "feat_mom_sma_ratio_10_50": ("Spread between short and medium SMA", "ratio", "SMA(10) / SMA(50) - 1.0"),
    "feat_mom_rsi_7": ("Relative Strength Index over 7 candles [0, 1]", "oscillator", "RSI(7) / 100.0"),
    "feat_mom_rsi_14": ("Relative Strength Index over 14 candles [0, 1]", "oscillator", "RSI(14) / 100.0"),
    "feat_mom_macd_line": ("MACD line normalized by close price", "ratio", "(EMA(12) - EMA(26)) / close"),
    "feat_mom_macd_signal": ("MACD 9-period signal line normalized by close", "ratio", "EMA(macd_line, 9) / close"),
    "feat_mom_macd_hist": ("MACD histogram normalized by close", "ratio", "(macd_line - macd_signal) / close"),
    "feat_mom_adx_14": ("Average Directional Index over 14 candles [0, 1]", "oscillator", "ADX(14) / 100.0"),

    # Volume (G5)
    "feat_volume_log": ("Log volume", "log volume", "ln(1 + volume)"),
    "feat_volume_zscore_60": ("Rolling volume Z-score over 60 candles", "z-score", "zscore(volume, 60)"),
    "feat_volume_zscore_12": ("Rolling volume Z-score over 12 candles", "z-score", "zscore(volume, 12)"),
    "feat_volume_change_1": ("1-candle fractional volume change", "ratio", "volume[t] / volume[t-1] - 1.0"),
    "feat_volume_ratio_12_60": ("Volume trend ratio: short / medium volume SMA", "ratio", "SMA(volume, 12) / SMA(volume, 60)"),

    # Order flow (G6)
    "feat_of_taker_buy_ratio": ("Taker buy volume fraction in [0, 1]", "ratio", "taker_buy_base_volume / volume"),
    "feat_of_taker_buy_zscore_60": ("Rolling Z-score of taker buy ratio over 60 candles", "z-score", "zscore(taker_buy_ratio, 60)"),
    "feat_of_trades_zscore_60": ("Rolling Z-score of number of trades over 60 candles", "z-score", "zscore(number_of_trades, 60)"),
    "feat_of_trades_zscore_12": ("Rolling Z-score of number of trades over 12 candles", "z-score", "zscore(number_of_trades, 12)"),
    "feat_of_avg_trade_size": ("Average trade size in BTC", "BTC / trade", "volume / number_of_trades"),
    "feat_of_avg_trade_size_zscore_60": ("Rolling Z-score of average trade size over 60 candles", "z-score", "zscore(avg_trade_size, 60)"),

    # Time (G7)
    "feat_time_sin_hour": ("Sine transform of UTC hour of day", "coordinate in [-1, 1]", "sin(2 * pi * hour / 24)"),
    "feat_time_cos_hour": ("Cosine transform of UTC hour of day", "coordinate in [-1, 1]", "cos(2 * pi * hour / 24)"),
    "feat_time_sin_weekday": ("Sine transform of day of week (Monday=0)", "coordinate in [-1, 1]", "sin(2 * pi * weekday / 7)"),
    "feat_time_cos_weekday": ("Cosine transform of day of week (Monday=0)", "coordinate in [-1, 1]", "cos(2 * pi * weekday / 7)"),

    # MTF 1h (G8)
    "feat_mtf_1h_ret": ("Log return of last COMPLETED 1h candle", "fraction", "ln(close_1h / open_1h) shifted 1h"),
    "feat_mtf_1h_range": ("High-low range of last COMPLETED 1h candle normalized by close", "ratio", "(high_1h - low_1h) / close_1h shifted 1h"),
    "feat_mtf_1h_vol_ratio": ("Volume of last COMPLETED 1h candle vs 24h rolling mean", "ratio", "vol_1h / SMA(vol_1h, 24) shifted 1h"),
    "feat_mtf_1h_close_vs_sma_6": ("Distance of last completed 1h close from 6h SMA", "ratio", "(close_1h - SMA(close_1h, 6)) / SMA shifted 1h"),
    "feat_mtf_1h_rsi_6": ("RSI(6) of completed 1h candles [0, 1]", "oscillator", "RSI(close_1h, 6) / 100.0 shifted 1h"),

    # Range-based Volatility (G9)
    "feat_vol_parkinson_12": ("Parkinson volatility over 12 candles (1h)", "fraction", "sqrt(1/(4 ln 2 * 12) * sum(ln(H/L)^2))"),
    "feat_vol_parkinson_60": ("Parkinson volatility over 60 candles (5h)", "fraction", "sqrt(1/(4 ln 2 * 60) * sum(ln(H/L)^2))"),
    "feat_vol_parkinson_288": ("Parkinson volatility over 288 candles (24h)", "fraction", "sqrt(1/(4 ln 2 * 288) * sum(ln(H/L)^2))"),
    "feat_vol_gk_12": ("Garman-Klass volatility over 12 candles (1h)", "fraction", "sqrt(1/12 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))"),
    "feat_vol_gk_60": ("Garman-Klass volatility over 60 candles (5h)", "fraction", "sqrt(1/60 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))"),
    "feat_vol_gk_288": ("Garman-Klass volatility over 288 candles (24h)", "fraction", "sqrt(1/288 * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))"),

    # Realized Volatility & Semivariance (G10)
    "feat_vol_realized_abs_sum_6": ("Realized absolute variation over 6 candles (30m)", "fraction", "sum(|r_5m|, W=6)"),
    "feat_vol_realized_abs_sum_12": ("Realized absolute variation over 12 candles (1h)", "fraction", "sum(|r_5m|, W=12)"),
    "feat_vol_realized_abs_sum_36": ("Realized absolute variation over 36 candles (3h)", "fraction", "sum(|r_5m|, W=36)"),
    "feat_vol_realized_max_abs_12": ("Maximum single-candle absolute return over 12 candles (1h)", "fraction", "max(|r_5m|, W=12)"),
    "feat_vol_realized_max_abs_60": ("Maximum single-candle absolute return over 60 candles (5h)", "fraction", "max(|r_5m|, W=60)"),
    "feat_vol_realized_semi_var_up_ratio_60": ("Upside semivariance share over 60 candles", "fraction [0, 1]", "SV^+ / (SV^+ + SV^-)"),

    # Seasonal Volatility & Volume (G11)
    "feat_vol_seasonal_slot_abs_ret_7d": ("Same 5m time slot mean absolute return over prior 7 days", "fraction", "mean(|r_5m|[t - 288*k]) for k=1..7"),
    "feat_vol_seasonal_vol_ratio_12": ("Ratio of 1h mean absolute return to 7-day seasonal slot baseline", "ratio", "mean(|r_5m|, 12) / slot_abs_ret_7d"),
    "feat_vol_seasonal_volume_ratio_7d": ("Ratio of current volume to 7-day same-slot mean volume", "ratio", "vol[t] / mean(vol[t - 288*k]) for k=1..7"),
    "feat_vol_seasonal_trades_ratio_7d": ("Ratio of current trades to 7-day same-slot mean trades", "ratio", "trades[t] / mean(trades[t - 288*k]) for k=1..7"),

    # HAR Multi-Horizon Volatility (G12)
    "feat_vol_har_rv_72": ("Realized volatility over 6h (72 candles)", "fraction", "rolling_sigma(r_5m, 72)"),
    "feat_vol_har_rv_2016": ("Realized volatility over 7 days (2016 candles)", "fraction", "rolling_sigma(r_5m, 2016)"),
    "feat_vol_har_ratio_12_288": ("HAR volatility ratio: 1h RV / 24h RV", "ratio", "rv_12 / rv_288"),
    "feat_vol_har_ratio_288_2016": ("HAR volatility ratio: 24h RV / 7d RV", "ratio", "rv_288 / rv_2016"),

    # Windowed Order Flow (G13)
    "feat_flow_window_net_6": ("Normalized taker net flow over 6 candles (30m)", "fraction [-1, 1]", "sum(2*taker_buy - vol) / sum(vol)"),
    "feat_flow_window_net_12": ("Normalized taker net flow over 12 candles (1h)", "fraction [-1, 1]", "sum(2*taker_buy - vol) / sum(vol)"),
    "feat_flow_window_net_36": ("Normalized taker net flow over 36 candles (3h)", "fraction [-1, 1]", "sum(2*taker_buy - vol) / sum(vol)"),
    "feat_flow_window_signed_flow_6": ("Price-signed net flow over 6 candles", "fraction [-1, 1]", "net_flow_6 * sign(ret_1)"),
    "feat_flow_window_signed_flow_12": ("Price-signed net flow over 12 candles", "fraction [-1, 1]", "net_flow_12 * sign(ret_1)"),
    "feat_flow_window_signed_flow_36": ("Price-signed net flow over 36 candles", "fraction [-1, 1]", "net_flow_36 * sign(ret_1)"),

    # Channel Location & Support/Resistance (G14)
    "feat_loc_position_288": ("Position of close within 24h high-low channel", "fraction [0, 1]", "(close - min(L_288)) / (max(H_288) - min(L_288))"),
    "feat_loc_dist_high_288": ("Fractional distance of close below 24h high", "fraction", "(max(H_288) - close) / close"),
    "feat_loc_dist_low_288": ("Fractional distance of close above 24h low", "fraction", "(close - min(L_288)) / close"),

    # Weekend & Continuous Week Time (G15)
    "feat_time_is_weekend": ("Weekend indicator (Saturday or Sunday UTC)", "binary {0, 1}", "1.0 if weekday >= 5 else 0.0"),
    "feat_time_sin_week": ("Sine transform of 168-hour weekly cycle", "coordinate in [-1, 1]", "sin(2 * pi * (weekday*24 + hour + min/60) / 168)"),
    "feat_time_cos_week": ("Cosine transform of 168-hour weekly cycle", "coordinate in [-1, 1]", "cos(2 * pi * (weekday*24 + hour + min/60) / 168)"),

    # Moving Average Structure & Multi-Timeframe MAs (G16)
    "feat_ma_sma_ratio_5": ("Ratio of 5m close to 5-period SMA", "ratio", "(close - sma5) / sma5"),
    "feat_ma_sma_ratio_25": ("Ratio of 5m close to 25-period SMA", "ratio", "(close - sma25) / sma25"),
    "feat_ma_sma_ratio_100": ("Ratio of 5m close to 100-period SMA", "ratio", "(close - sma100) / sma100"),
    "feat_ma_ema_ratio_5": ("Ratio of 5m close to 5-period EMA", "ratio", "(close - ema5) / ema5"),
    "feat_ma_ema_ratio_25": ("Ratio of 5m close to 25-period EMA", "ratio", "(close - ema25) / ema25"),
    "feat_ma_ema_ratio_100": ("Ratio of 5m close to 100-period EMA", "ratio", "(close - ema100) / ema100"),

    "feat_ma_sma_slope_5": ("5-candle slope of 5-period SMA", "fraction", "sma5_t / sma5_{t-5} - 1"),
    "feat_ma_sma_slope_10": ("5-candle slope of 10-period SMA", "fraction", "sma10_t / sma10_{t-5} - 1"),
    "feat_ma_sma_slope_25": ("5-candle slope of 25-period SMA", "fraction", "sma25_t / sma25_{t-5} - 1"),
    "feat_ma_sma_slope_50": ("5-candle slope of 50-period SMA", "fraction", "sma50_t / sma50_{t-5} - 1"),
    "feat_ma_sma_slope_100": ("5-candle slope of 100-period SMA", "fraction", "sma100_t / sma100_{t-5} - 1"),

    "feat_ma_ema_slope_5": ("5-candle slope of 5-period EMA", "fraction", "ema5_t / ema5_{t-5} - 1"),
    "feat_ma_ema_slope_10": ("5-candle slope of 10-period EMA", "fraction", "ema10_t / ema10_{t-5} - 1"),
    "feat_ma_ema_slope_25": ("5-candle slope of 25-period EMA", "fraction", "ema25_t / ema25_{t-5} - 1"),
    "feat_ma_ema_slope_50": ("5-candle slope of 50-period EMA", "fraction", "ema50_t / ema50_{t-5} - 1"),
    "feat_ma_ema_slope_100": ("5-candle slope of 100-period EMA", "fraction", "ema100_t / ema100_{t-5} - 1"),

    "feat_ma_alignment_sma_5m": ("Pairwise alignment score of 5m SMAs (5,10,25,50,100)", "score in [-1, 1]", "(2*sum(short>long) - 10) / 10"),
    "feat_ma_alignment_ema_5m": ("Pairwise alignment score of 5m EMAs (5,10,25,50,100)", "score in [-1, 1]", "(2*sum(short>long) - 10) / 10"),

    "feat_ma_1h_sma_ratio_5": ("Ratio of 5m close to completed 1h 5-period SMA", "ratio", "(close - sma5_1h) / sma5_1h"),
    "feat_ma_1h_sma_ratio_10": ("Ratio of 5m close to completed 1h 10-period SMA", "ratio", "(close - sma10_1h) / sma10_1h"),
    "feat_ma_1h_sma_ratio_25": ("Ratio of 5m close to completed 1h 25-period SMA", "ratio", "(close - sma25_1h) / sma25_1h"),
    "feat_ma_1h_sma_ratio_50": ("Ratio of 5m close to completed 1h 50-period SMA", "ratio", "(close - sma50_1h) / sma50_1h"),
    "feat_ma_1h_sma_ratio_100": ("Ratio of 5m close to completed 1h 100-period SMA", "ratio", "(close - sma100_1h) / sma100_1h"),

    "feat_ma_1h_ema_ratio_5": ("Ratio of 5m close to completed 1h 5-period EMA", "ratio", "(close - ema5_1h) / ema5_1h"),
    "feat_ma_1h_ema_ratio_10": ("Ratio of 5m close to completed 1h 10-period EMA", "ratio", "(close - ema10_1h) / ema10_1h"),
    "feat_ma_1h_ema_ratio_25": ("Ratio of 5m close to completed 1h 25-period EMA", "ratio", "(close - ema25_1h) / ema25_1h"),
    "feat_ma_1h_ema_ratio_50": ("Ratio of 5m close to completed 1h 50-period EMA", "ratio", "(close - ema50_1h) / ema50_1h"),
    "feat_ma_1h_ema_ratio_100": ("Ratio of 5m close to completed 1h 100-period EMA", "ratio", "(close - ema100_1h) / ema100_1h"),

    "feat_ma_1h_sma_slope_5": ("5-hour slope of completed 1h 5-period SMA", "fraction", "sma5_{1h,t} / sma5_{1h,t-5} - 1"),
    "feat_ma_1h_sma_slope_10": ("5-hour slope of completed 1h 10-period SMA", "fraction", "sma10_{1h,t} / sma10_{1h,t-5} - 1"),
    "feat_ma_1h_sma_slope_25": ("5-hour slope of completed 1h 25-period SMA", "fraction", "sma25_{1h,t} / sma25_{1h,t-5} - 1"),
    "feat_ma_1h_sma_slope_50": ("5-hour slope of completed 1h 50-period SMA", "fraction", "sma50_{1h,t} / sma50_{1h,t-5} - 1"),
    "feat_ma_1h_sma_slope_100": ("5-hour slope of completed 1h 100-period SMA", "fraction", "sma100_{1h,t} / sma100_{1h,t-5} - 1"),

    "feat_ma_alignment_sma_1h": ("Pairwise alignment score of completed 1h SMAs", "score in [-1, 1]", "(2*sum(short>long) - 10) / 10"),
    "feat_ma_alignment_ema_1h": ("Pairwise alignment score of completed 1h EMAs", "score in [-1, 1]", "(2*sum(short>long) - 10) / 10"),
    "feat_ma_1h_natr_14": ("Completed 1h 14-period Wilder ATR normalized by 5m close", "ratio", "atr14_1h / close"),

    # Causal Market Structure & Dow Trend (G17)
    "feat_struct_dist_support_atr": ("Distance from close to nearest support in 5m ATR units", "ATR multiple", "(close - S) / ATR"),
    "feat_struct_dist_support_pct": ("Fractional distance from close to nearest support", "fraction", "(close - S) / close"),
    "feat_struct_dist_resistance_atr": ("Distance from close to nearest resistance in 5m ATR units", "ATR multiple", "(R - close) / ATR"),
    "feat_struct_dist_resistance_pct": ("Fractional distance from close to nearest resistance", "fraction", "(R - close) / close"),
    "feat_struct_support_strength": ("Touch count of nearest active support level", "count", "level.touches"),
    "feat_struct_resistance_strength": ("Touch count of nearest active resistance level", "count", "level.touches"),
    "feat_struct_trend_up": ("Dow Theory uptrend indicator (HH and HL persist)", "binary {0, 1}", "1.0 if trend == Up else 0.0"),
    "feat_struct_trend_down": ("Dow Theory downtrend indicator (LH and LL persist)", "binary {0, 1}", "1.0 if trend == Down else 0.0"),
    "feat_struct_trend_range": ("Dow Theory sideways/range market indicator", "binary {0, 1}", "1.0 if trend == Range else 0.0"),
    "feat_struct_trend_age": ("Candles elapsed since current Dow trend state began", "count", "t - trend_start"),
    "feat_struct_pullback_depth_atr": ("Pullback depth from trend extreme in ATR units", "ATR multiple", "(peak - close) / ATR or (close - trough) / ATR"),
    "feat_struct_bars_since_break": ("Candles elapsed since last support/resistance level break", "count", "bars since break event"),
    "feat_struct_bars_since_retest": ("Candles elapsed since last flipped level retest", "count", "bars since retest event"),
    "feat_struct_retest_direction": ("Direction of last flipped level retest (+1 bullish, -1 bearish, 0 neutral)", "trinary {-1, 0, 1}", "retest direction"),
    "feat_struct_rule_vote": ("Rule-based 4th voter signal per decisions table (+1, -1, 0)", "trinary {-1, 0, 1}", "Section 9.2 rule vote"),
}


def _feature_entries() -> Dict[str, Any]:
    """Generate column_dict entries for all engineered features."""
    entries: Dict[str, Any] = {}
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            desc, unit, formula = _FEATURE_DESCRIPTIONS.get(col, (f"Engineered feature {col}", "unitless", ""))
            entry = {
                "description": desc,
                "unit": unit,
                "formula": formula,
                "group": group_name,
                "role": "feature",
                "is_future_looking": False,
                "allowed_as_model_input": True,
                "source_columns": [],
            }
            if col == "feat_struct_rule_vote":
                entry["tag"] = "rule_vote"
            entries[col] = entry
    return entries


# ---------------------------------------------------------------------------
# Drawings: raw price levels and candidate targets/stops (drawings.csv)
# ---------------------------------------------------------------------------
_DRAWINGS: Dict[str, Any] = {
    "level_support_price": {
        "description": "Raw USDT price of the nearest active/flipped support level below close.",
        "unit": "USDT",
        "formula": "max(level.price for level in active_support if level.price < close)",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close", "high", "low"],
    },
    "level_resistance_price": {
        "description": "Raw USDT price of the nearest active/flipped resistance level above close.",
        "unit": "USDT",
        "formula": "min(level.price for level in active_resistance if level.price > close)",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close", "high", "low"],
    },
    "candidate_target_price": {
        "description": "Target price in USDT for active structural rule signals; NaN when vote == 0.",
        "unit": "USDT",
        "formula": "Next resistance above (vote=+1) or support below (vote=-1), min 1.5 ATR",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close", "high", "low"],
    },
    "candidate_stop_price": {
        "description": "Stop-loss price in USDT for active structural rule signals; NaN when vote == 0.",
        "unit": "USDT",
        "formula": "Beyond opposite level by buffer (0.2 ATR), min 1.5 ATR distance",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close", "high", "low"],
    },
    "target_dist_pct": {
        "description": "Absolute fractional distance from close to candidate target price.",
        "unit": "fraction",
        "formula": "|target - close| / close",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close"],
    },
    "stop_dist_pct": {
        "description": "Absolute fractional distance from close to candidate stop price.",
        "unit": "fraction",
        "formula": "|stop - close| / close",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close"],
    },
    "target_dist_c_mult": {
        "description": "Candidate target distance expressed as a multiple of round-trip cost c (0.23%).",
        "unit": "multiple of c",
        "formula": "target_dist_pct / 0.0023",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close"],
    },
    "stop_dist_c_mult": {
        "description": "Candidate stop distance expressed as a multiple of round-trip cost c (0.23%).",
        "unit": "multiple of c",
        "formula": "stop_dist_pct / 0.0023",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": ["close"],
    },
    "reward_to_risk": {
        "description": "Ratio of target distance to stop distance for active structural rule signals.",
        "unit": "ratio",
        "formula": "target_dist_pct / stop_dist_pct",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
    "rule_reason_code": {
        "description": "Causal text trigger code for the 4th rule voter (BULL_RETEST, BEAR_RETEST, BULL_PULLBACK_MA, BEAR_PULLBACK_MA, NEUTRAL).",
        "unit": "category string",
        "formula": "Decision table in docs/decisions.md",
        "group": "drawing",
        "role": "drawing",
        "is_future_looking": False,
        "allowed_as_model_input": False,
        "source_columns": [],
    },
}


# ---------------------------------------------------------------------------
# Master dictionary: merge all groups
# ---------------------------------------------------------------------------
COLUMN_DICT: Dict[str, Any] = {}
COLUMN_DICT.update(_RAW)
COLUMN_DICT.update(_CLEAN)
COLUMN_DICT.update(_target_base_entries())
COLUMN_DICT.update(_target_label_entries())
COLUMN_DICT.update(_feature_entries())
COLUMN_DICT.update(_DRAWINGS)


def get_column_meta(col_name: str) -> Dict[str, Any]:
    """Return metadata for a column."""
    if col_name not in COLUMN_DICT:
        raise KeyError(
            f"Column '{col_name}' has no entry in src/column_dictionary.py. "
            f"Add a description before generating detail JSON files."
        )
    return COLUMN_DICT[col_name]


COLUMN_ROLES: List[str] = ["market", "feature", "target", "threshold", "flag", "drawing"]


def get_column_role(col_name: str) -> str:
    """Return canonical role for a column: market | feature | target | threshold | flag."""
    return get_column_meta(col_name)["role"]


def get_columns_by_role(role: str) -> List[str]:
    """Return all registered column names with the given role."""
    return [col for col, meta in COLUMN_DICT.items() if meta["role"] == role]


def get_feature_columns(
    df: Optional[pd.DataFrame] = None,
    feature_set: str = "all",
) -> List[str]:
    """Return ONLY columns with role 'feature'.

    Guarantees:
      - All returned columns have role == 'feature'
      - No column with role 'target' or 'threshold' is ever returned
      - No returned column name contains 'fwd_', 'target_', 'raw_sign', 'bin_dir_', or 'big_move_'
    All model training and feature evaluation code MUST use this function.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Optional DataFrame to filter against existing columns.
    feature_set : str
        'all' | 'FS_ALL': all 136 features.
        'FS_CORE' | 'core': 85 baseline features (no ma_structure, no structure).
        'FS_TA' | 'ta': 54 features (ma_structure, structure, location).
        'LINEAR_FEATURES' | 'linear': 70 features for linear models (uncorrelated, non-redundant).
    """
    feature_cols = [col for col, meta in COLUMN_DICT.items() if meta["role"] == "feature"]

    # Filter by feature_set
    fs_upper = feature_set.upper()
    if fs_upper in ("FS_CORE", "CORE"):
        core_groups = {
            "returns", "volatility", "candle_shape", "momentum", "volume",
            "order_flow", "time", "mtf_1h", "vol_range", "vol_realized",
            "vol_seasonal", "vol_har", "flow_window", "location", "time_weekend"
        }
        feature_cols = [c for c in feature_cols if COLUMN_DICT[c]["group"] in core_groups]
    elif fs_upper in ("FS_TA", "TA"):
        ta_groups = {"ma_structure", "structure", "location"}
        feature_cols = [c for c in feature_cols if COLUMN_DICT[c]["group"] in ta_groups]
    elif fs_upper in ("LINEAR_FEATURES", "LINEAR"):
        from src.config import LINEAR_FEATURES
        feature_cols = [c for c in LINEAR_FEATURES]
    elif fs_upper in ("FS_ALL", "ALL"):
        pass
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be 'all', 'FS_CORE', 'FS_TA', or 'LINEAR_FEATURES'.")

    # Assert safety contracts
    for col in feature_cols:
        meta = COLUMN_DICT[col]
        assert meta["role"] == "feature", f"Invalid role for {col}: {meta['role']}"
        assert not meta["is_future_looking"], f"Feature {col} is marked future-looking"
        assert not col.startswith("fwd_"), f"Feature {col} contains fwd_ prefix"
        assert not col.startswith("target_"), f"Feature {col} contains target_ prefix"
        assert not col.startswith("raw_sign_"), f"Feature {col} contains raw_sign_ prefix"
        assert not col.startswith("bin_dir_"), f"Feature {col} contains bin_dir_ prefix"
        assert not col.startswith("big_move_"), f"Feature {col} contains big_move_ prefix"

    if df is not None:
        return [c for c in feature_cols if c in df.columns]
    return feature_cols


def load_master(
    split: str = "train",
    filepath: Optional[Path] = None,
    unlock_test: bool = False,
) -> pd.DataFrame:
    """Safe loader for master.csv with strict test partition locking.

    Parameters
    ----------
    split : str
        'train' (default), 'val', 'train_val', 'test', or 'all'.
    filepath : Optional[Path]
        Custom path; defaults to config.DATA_PROCESSED_DIR / 'master.csv'.
    unlock_test : bool
        Safety lock. Must be True to load 'test' or 'all'.

    Returns
    -------
    pd.DataFrame
        Sliced partition with UTC DatetimeIndex.

    Raises
    ------
    PermissionError
        If split in ('test', 'all') and unlock_test is False.
    """
    from src.io_utils import load_master_csv, MASTER_CSV_PATH
    target_path = filepath or MASTER_CSV_PATH
    return load_master_csv(filepath=target_path, split=split, unlock_test=unlock_test)


def all_column_names() -> List[str]:
    """Return sorted list of all registered column names."""
    return sorted(COLUMN_DICT.keys())
