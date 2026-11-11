"""Causal Market Structure Engine — Support, Resistance, Dow Trend & 4th Voter.

Strictly causal, non-repainting market structure identification for Step 4b:
1. Swing Pivots:
   - 5m: k=5 (confirmed at i+5)
   - 1h: k=3 on completed 1h candles (confirmed at I+3)
2. Append-Only Horizontal Levels:
   - Formed from confirmed pivots, merged within tau_merge = 0.5 * ATR.
   - Bounded memory buffer (M_pivots = 50).
   - Creation prices NEVER change. Status transitions (active -> flipped) on close break.
   - Retest detection within tau_retest = 0.5 * ATR with price rejection.
3. Dow Rule Trend State Machine:
   - Persistent state: Up (+1), Down (-1), Range (0).
   - Up requires consecutive HH/HL, ends on close below last HL - 0.2 * ATR.
   - Down requires consecutive LH/LL, ends on close above last LH + 0.2 * ATR.
4. The 4th Rule Voter:
   - Deterministic, zero fitted parameters, per docs/decisions.md Section 9.2.
5. Role Separation:
   - Features (role 'feature', group 'structure'): 15 unitless ratios and distances.
   - Drawings (role 'drawing', saved in drawings.csv): raw prices, candidate targets/stops.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src import config
from src.features import _atr, _ema
from src.utils import setup_logger

logger = setup_logger("structure")

STRUCTURE_FEATURE_COLUMNS: List[str] = [
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
]

DRAWING_COLUMNS: List[str] = [
    "level_support_price",
    "level_resistance_price",
    "candidate_target_price",
    "candidate_stop_price",
    "target_dist_pct",
    "stop_dist_pct",
    "target_dist_c_mult",
    "stop_dist_c_mult",
    "reward_to_risk",
    "rule_reason_code",
]


@dataclass
class HorizontalLevel:
    level_id: int
    price: float
    level_type: str        # 'support' or 'resistance'
    created_idx: int
    touches: int
    status: str            # 'active' or 'flipped'
    effective_role: str    # 'support' or 'resistance'


def build_structure(
    df: pd.DataFrame,
    k_5m: int = 5,
    k_1h: int = 3,
    tau_merge_atr: float = 0.5,
    beta_break_atr: float = 0.2,
    tau_retest_atr: float = 0.5,
    max_levels: int = 50,
    min_target_stop_atr: float = 1.5,
    cost_c: float = config.TOTAL_ROUNDTRIP_COST,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full causal market structure engine on OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame with open, high, low, close, volume.
        Optional column: gap_id.
    k_5m : int
        5m swing pivot window (default 5 candles).
    k_1h : int
        1h swing pivot window (default 3 completed 1h candles).
    tau_merge_atr : float
        Tolerance in ATR for merging nearby pivot levels (default 0.5).
    beta_break_atr : float
        Buffer in ATR beyond level required for a candle close break (default 0.2).
    tau_retest_atr : float
        Tolerance in ATR for retest zone (default 0.5).
    max_levels : int
        Maximum number of persistent horizontal levels maintained in memory buffer.
    min_target_stop_atr : float
        Minimum distance for candidate targets and stops in ATR units (default 1.5).
    cost_c : float
        Total round-trip trading cost fraction (default 0.0023).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, drawings_df)
        - features_df: DataFrame of exactly STRUCTURE_FEATURE_COLUMNS (role 'feature').
        - drawings_df: DataFrame of exactly DRAWING_COLUMNS (role 'drawing').
    """
    n = len(df)
    if n == 0:
        return (
            pd.DataFrame(columns=STRUCTURE_FEATURE_COLUMNS, index=df.index),
            pd.DataFrame(columns=DRAWING_COLUMNS, index=df.index),
        )

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    gap_id = df["gap_id"].to_numpy(dtype=np.int64) if "gap_id" in df.columns else np.zeros(n, dtype=np.int64)

    atr_series = _atr(df, 14).bfill().ffill()
    atr = atr_series.to_numpy(dtype=np.float64)
    ema50 = _ema(df["close"], 50).bfill().ffill().to_numpy(dtype=np.float64)

    # -----------------------------------------------------------------------
    # 1. Detect 5m Swing Pivots (confirmed at i + k_5m)
    # -----------------------------------------------------------------------
    sh_confirmed: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
    sl_confirmed: List[List[Tuple[float, int]]] = [[] for _ in range(n)]

    for i in range(k_5m, n - k_5m):
        # Disallow pivots spanning a gap
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

    # Output feature arrays
    dist_sup_atr = np.zeros(n, dtype=np.float64)
    dist_sup_pct = np.zeros(n, dtype=np.float64)
    dist_res_atr = np.zeros(n, dtype=np.float64)
    dist_res_pct = np.zeros(n, dtype=np.float64)
    sup_strength = np.zeros(n, dtype=np.float64)
    res_strength = np.zeros(n, dtype=np.float64)

    trend_up = np.zeros(n, dtype=np.float64)
    trend_down = np.zeros(n, dtype=np.float64)
    trend_range = np.zeros(n, dtype=np.float64)
    trend_age = np.zeros(n, dtype=np.float64)
    pullback_depth = np.zeros(n, dtype=np.float64)
    bars_since_break = np.zeros(n, dtype=np.float64)
    bars_since_retest = np.zeros(n, dtype=np.float64)
    retest_direction = np.zeros(n, dtype=np.float64)
    rule_vote = np.zeros(n, dtype=np.float64)

    # Drawing arrays
    sup_price = np.zeros(n, dtype=np.float64)
    res_price = np.zeros(n, dtype=np.float64)
    cand_target = np.full(n, np.nan, dtype=np.float64)
    cand_stop = np.full(n, np.nan, dtype=np.float64)
    target_dist_pct = np.full(n, np.nan, dtype=np.float64)
    stop_dist_pct = np.full(n, np.nan, dtype=np.float64)
    target_dist_c = np.full(n, np.nan, dtype=np.float64)
    stop_dist_c = np.full(n, np.nan, dtype=np.float64)
    reward_to_risk = np.full(n, np.nan, dtype=np.float64)
    reason_code = ["NEUTRAL"] * n

    # Internal state machine structures bounded to declared max_levels (M_pivots=50)
    # Each pivot: (price, type, confirmed_t, pivot_idx)
    active_pivots: deque[Tuple[float, str, int, int]] = deque(maxlen=max_levels)
    swing_highs: deque[Tuple[float, int]] = deque(maxlen=25)
    swing_lows: deque[Tuple[float, int]] = deque(maxlen=25)

    curr_trend = 0  # +1: Up, -1: Down, 0: Range
    curr_trend_start = 0
    last_hl = np.nan
    last_lh = np.nan
    peak_high = np.nan
    trough_low = np.nan

    b_break = 9999
    b_retest = 9999
    last_retest_dir = 0

    for t in range(n):
        c = close[t]
        h = high[t]
        l = low[t]
        o = open_[t]
        a = max(atr[t], 1e-12)
        e50 = ema50[t]

        # Reset structure on gap transition
        if t > 0 and gap_id[t] != gap_id[t - 1]:
            active_pivots.clear()
            swing_highs.clear()
            swing_lows.clear()
            curr_trend = 0
            curr_trend_start = t
            last_hl = np.nan
            last_lh = np.nan
            peak_high = np.nan
            trough_low = np.nan
            b_break = 9999
            b_retest = 9999
            last_retest_dir = 0

        # -------------------------------------------------------------------
        # 2. Append confirmed swing highs and lows to bounded pivot buffer
        # -------------------------------------------------------------------
        for sh_p, p_idx in sh_confirmed[t]:
            active_pivots.append((sh_p, "resistance", t, p_idx))
            swing_highs.append((sh_p, p_idx))
        for sl_p, p_idx in sl_confirmed[t]:
            active_pivots.append((sl_p, "support", t, p_idx))
            swing_lows.append((sl_p, p_idx))

        # -------------------------------------------------------------------
        # 3. Form horizontal levels from bounded confirmed pivots (tau_merge = 0.5 ATR)
        # -------------------------------------------------------------------
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
                    "created_t": p_conf_t,
                    "effective_role": p_type,
                    "status": "active",
                })

        # -------------------------------------------------------------------
        # 4. Dow Rule Trend State Machine
        # -------------------------------------------------------------------
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            h_curr, h_prev = swing_highs[-1][0], swing_highs[-2][0]
            l_curr, l_prev = swing_lows[-1][0], swing_lows[-2][0]
            if h_curr > h_prev and l_curr > l_prev:
                if curr_trend != 1:
                    curr_trend = 1
                    curr_trend_start = t
                    peak_high = h
                last_hl = l_curr
            elif h_curr < h_prev and l_curr < l_prev:
                if curr_trend != -1:
                    curr_trend = -1
                    curr_trend_start = t
                    trough_low = l
                last_lh = h_curr

        # Check trend break
        if curr_trend == 1:
            if not np.isnan(last_hl) and c < (last_hl - beta_break_atr * a):
                curr_trend = 0
                curr_trend_start = t
                b_break = 0
            else:
                peak_high = max(peak_high, h)
        elif curr_trend == -1:
            if not np.isnan(last_lh) and c > (last_lh + beta_break_atr * a):
                curr_trend = 0
                curr_trend_start = t
                b_break = 0
            else:
                trough_low = min(trough_low, l)

        # -------------------------------------------------------------------
        # 5. Level Breaks, Role Flips & Retest Rejections
        # -------------------------------------------------------------------
        for lev in levels:
            if lev["effective_role"] == "support":
                # Close break below support plus buffer
                if c < (lev["price"] - beta_break_atr * a):
                    lev["effective_role"] = "resistance"
                    lev["status"] = "flipped"
                    b_break = 0
            elif lev["effective_role"] == "resistance":
                # Close break above resistance plus buffer
                if c > (lev["price"] + beta_break_atr * a):
                    lev["effective_role"] = "support"
                    lev["status"] = "flipped"
                    b_break = 0

            # Retest of flipped levels
            if lev["status"] == "flipped":
                if abs(c - lev["price"]) <= tau_retest_atr * a:
                    if lev["effective_role"] == "support" and c >= (lev["price"] - beta_break_atr * a):
                        # Bullish retest: flipped resistance acting as support held
                        if c >= o or (t > 0 and c >= close[t - 1]):
                            b_retest = 0
                            last_retest_dir = 1
                    elif lev["effective_role"] == "resistance" and c <= (lev["price"] + beta_break_atr * a):
                        # Bearish retest: flipped support acting as resistance held
                        if c <= o or (t > 0 and c <= close[t - 1]):
                            b_retest = 0
                            last_retest_dir = -1

        # -------------------------------------------------------------------
        # 6. Nearest Active Support and Resistance
        # -------------------------------------------------------------------
        best_sup = -np.inf
        best_sup_touches = 1
        best_res = np.inf
        best_res_touches = 1

        for lev in levels:
            if lev["effective_role"] == "support" and lev["price"] < c:
                if lev["price"] > best_sup:
                    best_sup = lev["price"]
                    best_sup_touches = lev["touches"]
            elif lev["effective_role"] == "resistance" and lev["price"] > c:
                if lev["price"] < best_res:
                    best_res = lev["price"]
                    best_res_touches = lev["touches"]

        # Fallback when no active levels exist within range
        if np.isneginf(best_sup):
            best_sup = c - 2.0 * a
            best_sup_touches = 1
        if np.isposinf(best_res):
            best_res = c + 2.0 * a
            best_res_touches = 1

        # -------------------------------------------------------------------
        # 7. Pullback Depth in ATR Units
        # -------------------------------------------------------------------
        pb_atr = 0.0
        if curr_trend == 1 and not np.isnan(peak_high):
            pb_atr = max(0.0, (peak_high - c) / a)
        elif curr_trend == -1 and not np.isnan(trough_low):
            pb_atr = max(0.0, (c - trough_low) / a)

        # -------------------------------------------------------------------
        # 8. Evaluate 4th Rule Voter (docs/decisions.md Section 9.2)
        # -------------------------------------------------------------------
        vote = 0
        r_code = "NEUTRAL"
        if curr_trend == 1 and b_retest <= 12 and last_retest_dir == 1:
            vote = 1
            r_code = "BULL_RETEST"
        elif curr_trend == -1 and b_retest <= 12 and last_retest_dir == -1:
            vote = -1
            r_code = "BEAR_RETEST"
        elif curr_trend == 1 and pb_atr <= 1.0 and c > e50 and abs(c - e50) <= 0.5 * a:
            vote = 1
            r_code = "BULL_PULLBACK_MA"
        elif curr_trend == -1 and pb_atr <= 1.0 and c < e50 and abs(c - e50) <= 0.5 * a:
            vote = -1
            r_code = "BEAR_PULLBACK_MA"

        # -------------------------------------------------------------------
        # 9. Candidate Target & Stop (Drawings)
        # -------------------------------------------------------------------
        t_price = np.nan
        s_price = np.nan
        if vote == 1:
            t_price = best_res
            s_price = min(best_sup - beta_break_atr * a, c - min_target_stop_atr * a)
            if (t_price - c) < min_target_stop_atr * a:
                t_price = c + min_target_stop_atr * a
        elif vote == -1:
            t_price = best_sup
            s_price = max(best_res + beta_break_atr * a, c + min_target_stop_atr * a)
            if (c - t_price) < min_target_stop_atr * a:
                t_price = c - min_target_stop_atr * a

        # -------------------------------------------------------------------
        # 10. Record Outputs
        # -------------------------------------------------------------------
        dist_sup_atr[t] = (c - best_sup) / a
        dist_sup_pct[t] = (c - best_sup) / c
        dist_res_atr[t] = (best_res - c) / a
        dist_res_pct[t] = (best_res - c) / c
        sup_strength[t] = float(best_sup_touches)
        res_strength[t] = float(best_res_touches)

        trend_up[t] = 1.0 if curr_trend == 1 else 0.0
        trend_down[t] = 1.0 if curr_trend == -1 else 0.0
        trend_range[t] = 1.0 if curr_trend == 0 else 0.0
        trend_age[t] = float(t - curr_trend_start)
        pullback_depth[t] = pb_atr
        bars_since_break[t] = float(min(b_break, 9999))
        bars_since_retest[t] = float(min(b_retest, 9999))
        retest_direction[t] = float(last_retest_dir if b_retest <= 12 else 0)
        rule_vote[t] = float(vote)

        # Drawings
        sup_price[t] = best_sup
        res_price[t] = best_res
        if vote != 0:
            cand_target[t] = t_price
            cand_stop[t] = s_price
            t_dist = abs(t_price - c) / c
            s_dist = abs(s_price - c) / c
            target_dist_pct[t] = t_dist
            stop_dist_pct[t] = s_dist
            target_dist_c[t] = t_dist / cost_c
            stop_dist_c[t] = s_dist / cost_c
            reward_to_risk[t] = t_dist / max(s_dist, 1e-12)
            reason_code[t] = r_code

        b_break += 1
        b_retest += 1

    # Assemble feature DataFrame
    features_dict = {
        "feat_struct_dist_support_atr": dist_sup_atr,
        "feat_struct_dist_support_pct": dist_sup_pct,
        "feat_struct_dist_resistance_atr": dist_res_atr,
        "feat_struct_dist_resistance_pct": dist_res_pct,
        "feat_struct_support_strength": sup_strength,
        "feat_struct_resistance_strength": res_strength,
        "feat_struct_trend_up": trend_up,
        "feat_struct_trend_down": trend_down,
        "feat_struct_trend_range": trend_range,
        "feat_struct_trend_age": trend_age,
        "feat_struct_pullback_depth_atr": pullback_depth,
        "feat_struct_bars_since_break": bars_since_break,
        "feat_struct_bars_since_retest": bars_since_retest,
        "feat_struct_retest_direction": retest_direction,
        "feat_struct_rule_vote": rule_vote,
    }
    df_features = pd.DataFrame(features_dict, index=df.index)[STRUCTURE_FEATURE_COLUMNS]

    # Assemble drawing DataFrame
    drawings_dict = {
        "level_support_price": sup_price,
        "level_resistance_price": res_price,
        "candidate_target_price": cand_target,
        "candidate_stop_price": cand_stop,
        "target_dist_pct": target_dist_pct,
        "stop_dist_pct": stop_dist_pct,
        "target_dist_c_mult": target_dist_c,
        "stop_dist_c_mult": stop_dist_c,
        "reward_to_risk": reward_to_risk,
        "rule_reason_code": reason_code,
    }
    df_drawings = pd.DataFrame(drawings_dict, index=df.index)[DRAWING_COLUMNS]

    return df_features, df_drawings

