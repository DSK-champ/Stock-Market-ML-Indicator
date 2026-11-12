"""TA State Branch: Deterministic Trend, Phase and Context Identification.

Strictly causal, non-repainting technical analysis state per candle:
- Trend: UP vs DOWN, strength in [0, 1], p_up in [0, 1].
- Phase: PULLBACK vs BREAKTHROUGH, strength in [0, 1], p_pullback in [0, 1].
- Context: 4 discrete regimes ("UPTREND | PULLBACK", "UPTREND | BREAKTHROUGH",
                              "DOWNTREND | PULLBACK", "DOWNTREND | BREAKTHROUGH").
Per-candle output table: data/processed/ta_state.csv.gz.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src import config
from src.features import _atr
from src.io_utils import save_horizon_oof_csv_gz, load_horizon_oof_csv_gz
from src.column_dictionary import load_master
from src.utils import setup_logger

logger = setup_logger("ta_state")

TA_STATE_CSV_GZ = config.DATA_PROCESSED_DIR / "ta_state.csv.gz"


def derive_break_direction(
    df: pd.DataFrame,
    k_5m: int = 5,
    tau_merge_atr: float = 0.5,
    beta_break_atr: float = 0.2,
    max_levels: int = 50,
) -> pd.Series:
    """Derive the direction (+1 bullish, -1 bearish) of the last confirmed level break.

    Causal, read-only function mirroring the level-tracking loop of build_structure.
    Does NOT modify master.csv or any existing pipeline output.
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=int, index=df.index, name="last_break_direction")

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    gap_id = (
        df["gap_id"].to_numpy(dtype=np.int64)
        if "gap_id" in df.columns
        else np.zeros(n, dtype=np.int64)
    )

    if "feat_vol_natr_14" in df.columns:
        atr = (df["feat_vol_natr_14"] * df["close"]).to_numpy(dtype=np.float64)
    else:
        atr_series = _atr(df, 14).bfill().ffill()
        atr = atr_series.to_numpy(dtype=np.float64)

    # 1. Detect 5m Swing Pivots (confirmed at i + k_5m)
    sh_confirmed: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
    sl_confirmed: List[List[Tuple[float, int]]] = [[] for _ in range(n)]

    for i in range(k_5m, n - k_5m):
        if gap_id[i - k_5m] != gap_id[i + k_5m]:
            continue
        h_i = high[i]
        l_i = low[i]
        if (
            h_i == np.max(high[i - k_5m : i + k_5m + 1])
            and h_i > high[i - 1]
            and h_i > high[i + 1]
        ):
            sh_confirmed[i + k_5m].append((h_i, i))
        if (
            l_i == np.min(low[i - k_5m : i + k_5m + 1])
            and l_i < low[i - 1]
            and l_i < low[i + 1]
        ):
            sl_confirmed[i + k_5m].append((l_i, i))

    active_pivots: deque[Tuple[float, str, int, int]] = deque(maxlen=max_levels)
    swing_highs: deque[Tuple[float, int]] = deque(maxlen=25)
    swing_lows: deque[Tuple[float, int]] = deque(maxlen=25)

    curr_trend = 0
    last_hl = np.nan
    last_lh = np.nan
    last_break_dir = 0
    break_dirs = np.zeros(n, dtype=np.int32)

    for t in range(n):
        c = close[t]
        h = high[t]
        l = low[t]
        a = max(atr[t], 1e-12)

        if t > 0 and gap_id[t] != gap_id[t - 1]:
            active_pivots.clear()
            swing_highs.clear()
            swing_lows.clear()
            curr_trend = 0
            last_hl = np.nan
            last_lh = np.nan
            last_break_dir = 0

        for sh_p, p_idx in sh_confirmed[t]:
            active_pivots.append((sh_p, "resistance", t, p_idx))
            swing_highs.append((sh_p, p_idx))
        for sl_p, p_idx in sl_confirmed[t]:
            active_pivots.append((sl_p, "support", t, p_idx))
            swing_lows.append((sl_p, p_idx))

        levels: List[Dict] = []
        for p_price, p_type, p_conf_t, p_idx in active_pivots:
            merged = False
            for lev in levels:
                if abs(p_price - lev["price"]) <= tau_merge_atr * a:
                    lev["touches"] += 1
                    merged = True
                    break
            if not merged:
                levels.append({
                    "price": p_price,
                    "type": p_type,
                    "touches": 1,
                    "effective_role": p_type,
                    "status": "active",
                })

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            h_curr, h_prev = swing_highs[-1][0], swing_highs[-2][0]
            l_curr, l_prev = swing_lows[-1][0], swing_lows[-2][0]
            if h_curr > h_prev and l_curr > l_prev:
                if curr_trend != 1:
                    curr_trend = 1
                last_hl = l_curr
            elif h_curr < h_prev and l_curr < l_prev:
                if curr_trend != -1:
                    curr_trend = -1
                last_lh = h_curr

        # Check trend break
        if curr_trend == 1:
            if not np.isnan(last_hl) and c < (last_hl - beta_break_atr * a):
                curr_trend = 0
                last_break_dir = -1
        elif curr_trend == -1:
            if not np.isnan(last_lh) and c > (last_lh + beta_break_atr * a):
                curr_trend = 0
                last_break_dir = 1

        # Check level break
        for lev in levels:
            if lev["effective_role"] == "support":
                if c < (lev["price"] - beta_break_atr * a):
                    lev["effective_role"] = "resistance"
                    lev["status"] = "flipped"
                    last_break_dir = -1
            elif lev["effective_role"] == "resistance":
                if c > (lev["price"] + beta_break_atr * a):
                    lev["effective_role"] = "support"
                    lev["status"] = "flipped"
                    last_break_dir = 1

        break_dirs[t] = last_break_dir

    return pd.Series(break_dirs, index=df.index, name="last_break_direction")


def compute_ta_state(
    df: pd.DataFrame,
    break_direction: Optional[Union[pd.Series, np.ndarray]] = None,
) -> pd.DataFrame:
    """Compute the deterministic, causal TA trend and phase state per candle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing required structure and MA features.
    break_direction : Optional[Union[pd.Series, np.ndarray]]
        Optional precomputed break direction series. If None, derived read-only.

    Returns
    -------
    pd.DataFrame
        DataFrame with open_time, ta_trend_label, ta_trend_strength, ta_trend_p_up,
        ta_phase_label, ta_phase_strength, ta_phase_p_pullback, ta_context.
    """
    req_cols = [
        "feat_struct_trend_up",
        "feat_struct_trend_down",
        "feat_ma_alignment_ema_5m",
        "feat_ma_alignment_ema_1h",
        "feat_struct_pullback_depth_atr",
        "feat_struct_bars_since_break",
    ]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features for TA state computation: {missing}")

    # Trend (UP / DOWN)
    trend_up = df["feat_struct_trend_up"].to_numpy(dtype=np.float64)
    trend_down = df["feat_struct_trend_down"].to_numpy(dtype=np.float64)
    ema_5m = df["feat_ma_alignment_ema_5m"].to_numpy(dtype=np.float64)
    ema_1h = df["feat_ma_alignment_ema_1h"].to_numpy(dtype=np.float64)

    trend_score = 0.5 * (trend_up - trend_down) + 0.3 * ema_5m + 0.2 * ema_1h
    trend_score = np.clip(trend_score, -1.0, 1.0)

    ta_trend_label = np.where(trend_score >= 0.0, "UP", "DOWN")
    ta_trend_strength = np.abs(trend_score)
    ta_trend_p_up = 0.5 + 0.5 * trend_score

    # Phase (PULLBACK / BREAKTHROUGH)
    pb_depth = df["feat_struct_pullback_depth_atr"].to_numpy(dtype=np.float64)
    pullback_score = np.clip(pb_depth / 2.0, 0.0, 1.0)

    if break_direction is None:
        break_dir_arr = derive_break_direction(df).to_numpy(dtype=np.int32)
    elif isinstance(break_direction, pd.Series):
        break_dir_arr = break_direction.to_numpy(dtype=np.int32)
    else:
        break_dir_arr = np.asarray(break_direction, dtype=np.int32)

    bars_break = df["feat_struct_bars_since_break"].to_numpy(dtype=np.float64)
    is_trend_dir = np.where(
        ta_trend_label == "UP",
        break_dir_arr == 1,
        break_dir_arr == -1,
    )
    breakthrough_score = np.where(
        is_trend_dir,
        np.exp(-bars_break / 6.0),
        0.0,
    )

    sum_scores = pullback_score + breakthrough_score
    ta_phase_p_pullback = np.where(
        sum_scores > 0.05,
        pullback_score / np.maximum(sum_scores, 1e-12),
        0.5,
    )
    ta_phase_p_pullback = np.clip(ta_phase_p_pullback, 0.0, 1.0)

    ta_phase_label = np.where(ta_phase_p_pullback >= 0.5, "PULLBACK", "BREAKTHROUGH")
    ta_phase_strength = np.abs(2.0 * ta_phase_p_pullback - 1.0)

    # 4 Canonical contexts
    contexts = [
        f"{t}TREND | {p}"
        for t, p in zip(ta_trend_label, ta_phase_label)
    ]

    out_df = pd.DataFrame(
        {
            "open_time": df.index if isinstance(df.index, pd.DatetimeIndex) else df["open_time"],
            "ta_trend_label": ta_trend_label,
            "ta_trend_strength": np.round(ta_trend_strength, 4),
            "ta_trend_p_up": np.round(ta_trend_p_up, 4),
            "ta_phase_label": ta_phase_label,
            "ta_phase_strength": np.round(ta_phase_strength, 4),
            "ta_phase_p_pullback": np.round(ta_phase_p_pullback, 4),
            "ta_context": contexts,
        },
        index=df.index,
    )
    return out_df


def build_and_save_ta_state(
    filepath: Optional[Path] = None,
    unlock_all: bool = True,
) -> Path:
    """Build and save ta_state.csv.gz across train (and unanalyzed test) rows."""
    out_path = filepath or TA_STATE_CSV_GZ
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading master data to compute TA state...")
    df_all = load_master(split="all", unlock_test=unlock_all)
    logger.info("Computing TA state for %d rows...", len(df_all))

    ta_df = compute_ta_state(df_all)
    ta_df.to_csv(out_path, compression="gzip", index=False)
    logger.info("Saved TA state table to %s (%d rows)", out_path, len(ta_df))
    return out_path

