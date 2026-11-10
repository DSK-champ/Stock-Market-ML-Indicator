"""Target label generation — Stock-Market-ML-Predictor v3 specification.

Computes multi-horizon, multi-rule directional targets for BTCUSDT 5-minute data.

Base quantities per horizon N ∈ {2, 3, 6, 12} candles
-------------------------------------------------------
  fwd_ret_N        = Close_{t+N} / Close_t - 1
  fwd_ret_open_N   = Close_{t+N} / Open_{t+1} - 1   (conservative backtesting entry)
  raw_sign_N       = sign(fwd_ret_N)                 (+1 / -1 / 0 / NaN)
  has_valid_target_N = False when the horizon crosses a gap OR row is in the last N rows

Threshold rules (all stored with per-row θ_up and θ_down)
----------------------------------------------------------
  S1  : θ_up = θ_down = c                           (symmetric cost floor)
  S2  : θ_up = θ_down = 0.003                       (symmetric 0.30%)
  Q1  : θ = max(c, 50th-pct of |fwd_ret| from TRAIN positives/negatives)
           computed ONCE on the train split, frozen, applied unchanged to val/test
  V1  : θ_t = max(c, 0.5 · σ_t · √N)               (causal rolling vol, ½-σ scale)
  V2  : θ_t = max(c, 1.0 · σ_t · √N)               (causal rolling vol, 1-σ scale)
  A1  : θ_up = c,     θ_down = 0.003                (asymmetric: easier UP bar)
  A2  : θ_up = 0.003, θ_down = c                    (asymmetric: easier DOWN bar)

Label assignment (applies to all rules)
----------------------------------------
  +1 (Up)      : fwd_ret_N > +θ_up
  -1 (Down)    : fwd_ret_N < -θ_down
   0 (Neutral) : otherwise (including dead zone)
  NaN          : has_valid_target_N = False

Invalidation rules
------------------
  has_valid_target_N = False if:
    1. gap_id at row t differs from gap_id at row t+N   (horizon crosses a data gap)
    2. row is among the last N rows of the input DataFrame (insufficient future data)

Economic cost constant
----------------------
  c = TOTAL_ROUNDTRIP_COST = 0.0023 (fees 0.20% + slippage 0.02% + half-spread 0.01%)
  Source: Binance VIP 0 taker fee [VERIFIED]; slippage/spread [ASSUMPTION].
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.utils import rolling_sigma, setup_logger

logger = setup_logger("labels")

# ---------------------------------------------------------------------------
# Public constants re-exported for convenience
# ---------------------------------------------------------------------------
COST = config.TOTAL_ROUNDTRIP_COST          # c = 0.0023
CANDIDATE_HORIZONS: List[int] = config.CANDIDATE_HORIZONS  # [2, 3, 6, 12]
RULES: List[str] = ["S1", "S2", "Q1", "V1", "V2", "A1", "A2"]

# Fixed absolute thresholds
_S2_THETA = 0.003    # 0.30%
_V1_K = 0.5          # half-σ scale factor
_V2_K = 1.0          # 1-σ scale factor
_SIGMA_WINDOW = 288  # 24 hours of 5-minute bars [ASSUMPTION — see utils.rolling_sigma]


# ===========================================================================
# Internal helpers
# ===========================================================================

def _build_fwd_ret(
    close: np.ndarray,
    open_next: np.ndarray,
    gap_id: np.ndarray,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute forward return arrays for a single horizon N.

    Returns
    -------
    fwd_ret, fwd_ret_open, raw_sign, valid_mask  — all float64, length = len(close)
    """
    n = len(close)
    fwd_ret = np.full(n, np.nan, dtype="float64")
    fwd_ret_open = np.full(n, np.nan, dtype="float64")
    raw_sign = np.full(n, np.nan, dtype="float64")
    valid_mask = np.zeros(n, dtype=bool)

    if n <= N:
        return fwd_ret, fwd_ret_open, raw_sign, valid_mask

    # Rows that can have a label: indices 0 .. n-N-1
    idx = n - N
    same_gap = gap_id[:idx] == gap_id[N:n]   # True where horizon stays in same segment

    # Forward return using close-to-close
    fr = (close[N:n] / close[:idx]) - 1.0
    # Conservative entry: open of t+1 to close of t+N
    fr_open = (close[N:n] / open_next[:idx]) - 1.0

    fwd_ret[:idx] = np.where(same_gap, fr, np.nan)
    fwd_ret_open[:idx] = np.where(same_gap, fr_open, np.nan)
    valid_mask[:idx] = same_gap

    # raw_sign: +1 / -1 / 0
    with np.errstate(invalid="ignore"):
        rs = np.where(fr > 0.0, 1.0, np.where(fr < 0.0, -1.0, 0.0))
    raw_sign[:idx] = np.where(same_gap, rs, np.nan)

    return fwd_ret, fwd_ret_open, raw_sign, valid_mask


def _apply_threshold(
    fwd_ret: np.ndarray,
    theta_up: np.ndarray,
    theta_down: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Apply symmetric or asymmetric threshold to produce 3-class label.

    Returns float64 array: 1.0 (Up), -1.0 (Down), 0.0 (Neutral), NaN (invalid).
    """
    label = np.where(
        ~valid_mask,
        np.nan,
        np.where(
            fwd_ret > theta_up,
            1.0,
            np.where(
                fwd_ret < -theta_down,
                -1.0,
                0.0,
            ),
        ),
    )
    return label.astype("float64")


# ===========================================================================
# Q1 threshold computation (TRAIN only)
# ===========================================================================

def compute_q1_thresholds(
    df_train: pd.DataFrame,
    horizons: List[int] = CANDIDATE_HORIZONS,
    cost: float = COST,
) -> Dict[str, float]:
    """Compute Q1 (50th-percentile of absolute fwd_ret) thresholds from TRAIN rows only.

    The result is a frozen dict that must be applied unchanged to val/test.
    Floor at c is enforced; floor_binding is logged.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split ONLY (no val or test rows).
    horizons : list of int
    cost : float
        Round-trip economic cost c (floor).

    Returns
    -------
    dict mapping  'q1_up_N' and 'q1_down_N'  →  float threshold (≥ cost)
    """
    if "close" not in df_train.columns:
        raise KeyError("df_train must have a 'close' column.")

    close = df_train["close"].to_numpy(dtype="float64")
    open_ = df_train["open"].to_numpy(dtype="float64") if "open" in df_train.columns else close
    gap_id = df_train["gap_id"].to_numpy() if "gap_id" in df_train.columns else np.ones(len(close), dtype=int)

    # open_next[t] = open of candle t+1 (used for fwd_ret_open)
    open_next = np.roll(open_, -1)  # last element wraps but it's in last-N NaN zone

    thresholds: Dict[str, float] = {}

    for N in horizons:
        fwd_ret, _, _, valid_mask = _build_fwd_ret(close, open_next, gap_id, N)

        valid_ret = fwd_ret[valid_mask & ~np.isnan(fwd_ret)]

        pos_ret = valid_ret[valid_ret > 0.0]
        neg_ret = valid_ret[valid_ret < 0.0]

        # 50th percentile of POSITIVE returns for Up threshold
        if len(pos_ret) > 0:
            raw_up = float(np.percentile(pos_ret, 50))
        else:
            raw_up = cost
            logger.warning("N=%d: no positive returns in train; Q1 up = c", N)

        # 50th percentile of ABS of NEGATIVE returns for Down threshold
        if len(neg_ret) > 0:
            raw_down = float(np.percentile(np.abs(neg_ret), 50))
        else:
            raw_down = cost
            logger.warning("N=%d: no negative returns in train; Q1 down = c", N)

        theta_up = max(cost, raw_up)
        theta_down = max(cost, raw_down)

        bound_up = theta_up > raw_up + 1e-12
        bound_down = theta_down > raw_down + 1e-12

        logger.info(
            "Q1 N=%2d: raw_up=%.4f%% raw_down=%.4f%% → θ_up=%.4f%% θ_down=%.4f%% "
            "(floor binds: up=%s, down=%s)",
            N, raw_up * 100, raw_down * 100,
            theta_up * 100, theta_down * 100,
            bound_up, bound_down,
        )

        thresholds[f"q1_up_{N}"] = theta_up
        thresholds[f"q1_down_{N}"] = theta_down

    return thresholds


# ===========================================================================
# Main label computation
# ===========================================================================

def compute_all_targets(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    horizons: List[int] = CANDIDATE_HORIZONS,
    cost: float = COST,
    sigma_window: int = _SIGMA_WINDOW,
    q1_thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute all base quantities and rule-specific labels for every horizon N.

    Parameters
    ----------
    df : pd.DataFrame
        The split to label (train or val; NOT test).  Must include columns:
        ``open``, ``close``, ``gap_id``.  DatetimeIndex in UTC.
    train_df : pd.DataFrame
        Training split (used for Q1 threshold computation if q1_thresholds is None).
    horizons : list of int
        Candidate horizons to process.
    cost : float
        Round-trip economic cost c (floor for all θ).
    sigma_window : int
        Rolling-sigma look-back in candles (default 288 = 24 h).
    q1_thresholds : dict, optional
        Pre-computed Q1 thresholds from ``compute_q1_thresholds``.
        Pass explicitly to avoid recomputing each call.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame aligned to df.index with columns:
          fwd_ret_N, fwd_ret_open_N, raw_sign_N, has_valid_target_N,
          target_N_{rule}, theta_up_N_{rule}, theta_down_N_{rule}
        for every N ∈ horizons and rule ∈ RULES.
    """
    required = {"close", "open", "gap_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df is missing required columns: {missing}")

    n = len(df)
    close = df["close"].to_numpy(dtype="float64")
    open_ = df["open"].to_numpy(dtype="float64")
    gap_id = df["gap_id"].to_numpy()

    # open_next[t] = open of candle t+1 (conservative entry for fwd_ret_open)
    open_next = np.empty(n, dtype="float64")
    open_next[:-1] = open_[1:]
    open_next[-1] = np.nan   # last row: no next candle

    # Causal rolling sigma on this split's log returns
    # Uses only past data — verified causal (see test_v_rule_sigma_causal)
    log_ret = pd.Series(
        np.concatenate([[np.nan], np.log(close[1:] / close[:-1])]),
        index=df.index,
    )
    sigma_series = rolling_sigma(log_ret, window=sigma_window)
    sigma = sigma_series.to_numpy(dtype="float64")

    # Compute Q1 thresholds from train (if not supplied)
    if q1_thresholds is None:
        q1_thresholds = compute_q1_thresholds(train_df, horizons=horizons, cost=cost)

    out = pd.DataFrame(index=df.index)

    for N in horizons:
        fwd_ret, fwd_ret_open, raw_sign, valid_mask = _build_fwd_ret(
            close, open_next, gap_id, N
        )

        # Store base quantities
        out[f"fwd_ret_{N}"] = fwd_ret
        out[f"fwd_ret_open_{N}"] = fwd_ret_open
        out[f"raw_sign_{N}"] = raw_sign
        out[f"has_valid_target_{N}"] = valid_mask

        # -----------------------------------------------------------------
        # S1: θ_up = θ_down = c
        # -----------------------------------------------------------------
        tu_S1 = np.full(n, cost, dtype="float64")
        td_S1 = np.full(n, cost, dtype="float64")
        out[f"theta_up_{N}_S1"] = tu_S1
        out[f"theta_down_{N}_S1"] = td_S1
        out[f"target_{N}_S1"] = _apply_threshold(fwd_ret, tu_S1, td_S1, valid_mask)

        # -----------------------------------------------------------------
        # S2: θ_up = θ_down = 0.003
        # -----------------------------------------------------------------
        tu_S2 = np.full(n, _S2_THETA, dtype="float64")
        td_S2 = np.full(n, _S2_THETA, dtype="float64")
        out[f"theta_up_{N}_S2"] = tu_S2
        out[f"theta_down_{N}_S2"] = td_S2
        out[f"target_{N}_S2"] = _apply_threshold(fwd_ret, tu_S2, td_S2, valid_mask)

        # -----------------------------------------------------------------
        # Q1: 50th pct of pos/neg fwd_ret from TRAIN, floored at c
        # -----------------------------------------------------------------
        tu_Q1 = np.full(n, q1_thresholds[f"q1_up_{N}"], dtype="float64")
        td_Q1 = np.full(n, q1_thresholds[f"q1_down_{N}"], dtype="float64")
        out[f"theta_up_{N}_Q1"] = tu_Q1
        out[f"theta_down_{N}_Q1"] = td_Q1
        out[f"target_{N}_Q1"] = _apply_threshold(fwd_ret, tu_Q1, td_Q1, valid_mask)

        # -----------------------------------------------------------------
        # V1: θ_t = max(c, 0.5 · σ_t · √N)   — causal rolling vol
        # V2: θ_t = max(c, 1.0 · σ_t · √N)
        # -----------------------------------------------------------------
        sqrt_N = np.sqrt(N)
        for rule, k in [("V1", _V1_K), ("V2", _V2_K)]:
            raw_theta = k * sigma * sqrt_N
            # Where sigma is NaN (e.g. very first row), fall back to c (floor)
            raw_theta_safe = np.where(np.isnan(raw_theta), 0.0, raw_theta)
            theta = np.maximum(cost, raw_theta_safe)
            # Count how many rows floor binds (raw < c) for valid rows
            floor_binds = int(np.sum(
                (raw_theta < cost) & valid_mask & ~np.isnan(raw_theta)
            ))
            if floor_binds > 0:
                logger.debug(
                    "%s N=%d: floor binds on %d valid rows (raw_theta < c)",
                    rule, N, floor_binds,
                )
            out[f"theta_up_{N}_{rule}"] = theta
            out[f"theta_down_{N}_{rule}"] = theta
            out[f"target_{N}_{rule}"] = _apply_threshold(fwd_ret, theta, theta, valid_mask)

        # -----------------------------------------------------------------
        # A1: θ_up = c,     θ_down = 0.003   (easier bar for Up signals)
        # A2: θ_up = 0.003, θ_down = c       (easier bar for Down signals)
        # -----------------------------------------------------------------
        tu_A1 = np.full(n, cost, dtype="float64")
        td_A1 = np.full(n, _S2_THETA, dtype="float64")
        out[f"theta_up_{N}_A1"] = tu_A1
        out[f"theta_down_{N}_A1"] = td_A1
        out[f"target_{N}_A1"] = _apply_threshold(fwd_ret, tu_A1, td_A1, valid_mask)

        tu_A2 = np.full(n, _S2_THETA, dtype="float64")
        td_A2 = np.full(n, cost, dtype="float64")
        out[f"theta_up_{N}_A2"] = tu_A2
        out[f"theta_down_{N}_A2"] = td_A2
        out[f"target_{N}_A2"] = _apply_threshold(fwd_ret, tu_A2, td_A2, valid_mask)

    return out


# ===========================================================================
# Summary helpers
# ===========================================================================

def create_label_summary(
    targets_df: pd.DataFrame,
    split_name: str,
    horizons: List[int] = CANDIDATE_HORIZONS,
    rules: List[str] = RULES,
) -> pd.DataFrame:
    """Count and percentage breakdown of Up / Down / Neutral / Invalid per horizon × rule.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Output of ``compute_all_targets``.
    split_name : str
        Label for the split (e.g. 'train', 'val').
    horizons, rules : list
        Subset to summarize.

    Returns
    -------
    pd.DataFrame with columns:
        split, horizon, rule, n_total, n_valid, n_up, n_down, n_neutral, n_invalid,
        pct_up, pct_down, pct_neutral, pct_invalid
    """
    records = []
    for N in horizons:
        valid_col = f"has_valid_target_{N}"
        for rule in rules:
            target_col = f"target_{N}_{rule}"
            if target_col not in targets_df.columns:
                continue

            tgt = targets_df[target_col]
            valid_flag = targets_df[valid_col] if valid_col in targets_df.columns else tgt.notna()

            n_total = len(tgt)
            n_valid = int(valid_flag.sum())
            n_invalid = n_total - n_valid
            n_up = int((tgt == 1.0).sum())
            n_down = int((tgt == -1.0).sum())
            n_neutral = int((tgt == 0.0).sum())

            def pct(x: int) -> float:
                return round(x / n_valid * 100, 2) if n_valid > 0 else 0.0

            records.append({
                "split": split_name,
                "horizon": N,
                "rule": rule,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_up": n_up,
                "n_down": n_down,
                "n_neutral": n_neutral,
                "n_invalid": n_invalid,
                "pct_up": pct(n_up),
                "pct_down": pct(n_down),
                "pct_neutral": pct(n_neutral),
                "pct_invalid": round(n_invalid / n_total * 100, 2) if n_total > 0 else 0.0,
            })

    return pd.DataFrame(records)


def floor_binding_report(
    targets_df: pd.DataFrame,
    train_q1: Dict[str, float],
    horizons: List[int] = CANDIDATE_HORIZONS,
    cost: float = COST,
    sigma_window: int = _SIGMA_WINDOW,
) -> pd.DataFrame:
    """Report how many valid rows the floor at c binds for Q1 and V rules.

    Returns a DataFrame with columns: horizon, rule, n_valid, n_floor_bound, pct_floor_bound.
    """
    records = []
    for N in horizons:
        valid_col = f"has_valid_target_{N}"
        if valid_col not in targets_df.columns:
            continue
        valid_mask = targets_df[valid_col].to_numpy()
        n_valid = int(valid_mask.sum())

        # Q1
        for side in ("up", "down"):
            q1_val = train_q1.get(f"q1_{side}_{N}", np.nan)
            if np.isnan(q1_val):
                continue
            bound = int(np.sum(valid_mask))  # all rows use the same frozen Q1 value
            # floor binds if q1_val == cost (i.e. the raw percentile was ≤ cost)
            binds = q1_val <= cost + 1e-10
            records.append({
                "horizon": N,
                "rule": f"Q1_{side}",
                "n_valid": n_valid,
                "floor_binds_all": binds,
                "theta": round(q1_val * 100, 4),
                "note": f"θ=c={cost*100:.2f}% (floor always active)" if binds else f"θ={q1_val*100:.4f}%",
            })

        # V1 / V2
        for rule, k in [("V1", _V1_K), ("V2", _V2_K)]:
            theta_col = f"theta_up_{N}_{rule}"
            if theta_col not in targets_df.columns:
                continue
            theta_arr = targets_df[theta_col].to_numpy()
            bound_rows = int(np.sum(
                (theta_arr <= cost + 1e-10) & valid_mask
            ))
            records.append({
                "horizon": N,
                "rule": rule,
                "n_valid": n_valid,
                "n_floor_bound": bound_rows,
                "pct_floor_bound": round(bound_rows / n_valid * 100, 2) if n_valid > 0 else 0.0,
                "note": "",
            })

    return pd.DataFrame(records)


# ===========================================================================
# Legacy compatibility shim
# (keeps existing test_labels.py imports working during migration)
# ===========================================================================

def create_forward_labels(
    df: pd.DataFrame,
    horizon: int = config.HORIZON_CANDLES,
    dead_zone: float = config.LABEL_DEAD_ZONE,
) -> pd.DataFrame:
    """Backward-compatible single-horizon, single-rule label function.

    Uses S1 logic (θ = dead_zone, symmetric).  Retained to avoid breaking
    existing unit tests that import this name.  New code should call
    ``compute_all_targets`` instead.

    Returns DataFrame with columns: forward_return, target, raw_sign.
    """
    close = df["close"].to_numpy(dtype="float64")
    open_ = df["open"].to_numpy(dtype="float64") if "open" in df.columns else close
    gap_id = df["gap_id"].to_numpy() if "gap_id" in df.columns else np.ones(len(close), dtype=int)

    n = len(close)
    open_next = np.empty(n, dtype="float64")
    open_next[:-1] = open_[1:]
    open_next[-1] = np.nan

    fwd_ret, _, raw_sign, valid_mask = _build_fwd_ret(close, open_next, gap_id, horizon)

    theta = np.full(n, dead_zone, dtype="float64")
    target = _apply_threshold(fwd_ret, theta, theta, valid_mask)

    return pd.DataFrame(
        {"forward_return": fwd_ret, "target": target, "raw_sign": raw_sign},
        index=df.index,
    )


def compute_label_balance_report(labels_series: pd.Series) -> dict:
    """Compute frequency breakdown of a single labels series.  Legacy helper."""
    total_rows = len(labels_series)
    valid_mask = labels_series.notna()
    valid_count = int(valid_mask.sum())
    nan_count = total_rows - valid_count

    up_count = int((labels_series == config.LABEL_UP).sum())
    down_count = int((labels_series == config.LABEL_DOWN).sum())
    neutral_count = int((labels_series == config.LABEL_NEUTRAL).sum())

    def pct_of_valid(c: int) -> float:
        return round((c / valid_count) * 100.0, 2) if valid_count > 0 else 0.0

    return {
        "total_candles": total_rows,
        "valid_labeled_candles": valid_count,
        "unlabeled_nan_candles": nan_count,
        "up_count": up_count,
        "up_pct": pct_of_valid(up_count),
        "down_count": down_count,
        "down_pct": pct_of_valid(down_count),
        "neutral_count": neutral_count,
        "neutral_pct": pct_of_valid(neutral_count),
    }


def compute_dead_zone_sensitivity(
    df: pd.DataFrame,
    thresholds: List[float] = [0.0010, 0.0020, 0.0023, 0.0030],
    horizon: int = config.HORIZON_CANDLES,
) -> pd.DataFrame:
    """Dead-zone sensitivity table (legacy helper, used by notebook 01)."""
    labels_df = create_forward_labels(df, horizon=horizon, dead_zone=0.0)
    valid_ret = labels_df["forward_return"].dropna()

    records = []
    n_valid = len(valid_ret)
    for thresh in thresholds:
        inside_count = int((valid_ret.abs() <= thresh).sum())
        up_count = int((valid_ret > thresh).sum())
        down_count = int((valid_ret < -thresh).sum())
        inside_pct = (inside_count / n_valid) * 100.0 if n_valid > 0 else 0.0
        up_pct = (up_count / n_valid) * 100.0 if n_valid > 0 else 0.0
        down_pct = (down_count / n_valid) * 100.0 if n_valid > 0 else 0.0

        records.append({
            "dead_zone_pct": round(thresh * 100.0, 3),
            "inside_dead_zone_count": inside_count,
            "inside_dead_zone_share_pct": round(inside_pct, 2),
            "up_count": up_count,
            "up_share_pct": round(up_pct, 2),
            "down_count": down_count,
            "down_share_pct": round(down_pct, 2),
        })

    return pd.DataFrame(records)
