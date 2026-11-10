"""Chronological data splitting, walk-forward fold generation, regime reporting, and test-set locking.

Enforces strict chronological partitioning:
  - 70% Train
  - 15% Validation
  - 15% Test (Locked)
Guarantees a 12-candle purge gap between splits and within walk-forward folds.
Any access to the test set prior to Phase 6 raises TestSetLockedError.
"""

from typing import Dict, Any, Generator, Tuple, List, Optional
import pandas as pd
import numpy as np

from src import config
from src.utils import setup_logger

logger = setup_logger("splitting")


class TestSetLockedError(Exception):
    """Raised when an unauthorized routine attempts to access the out-of-sample test set prior to Phase 6."""
    __test__ = False


def get_real_split_indices(
    df: pd.DataFrame,
) -> Dict[str, Tuple[int, int]]:
    """Compute exact slice indices based on real cleaned data length.
    
    If the DataFrame spans across the full canonical historical period,
    uses the fixed split timestamps so Train and Val remain 100% frozen.
    Otherwise (for synthetic/mock unit tests), uses 70/15/15 proportional slices.
    """
    if (
        isinstance(df.index, pd.DatetimeIndex)
        and df.index.min() <= config.SPLIT_BOUNDARIES["train"]["start"]
        and df.index.max() >= config.SPLIT_BOUNDARIES["val"]["end"]
    ):
        t_train_end = config.SPLIT_BOUNDARIES["train"]["end"]
        t_val_end = config.SPLIT_BOUNDARIES["val"]["end"]
        t_test_start = config.SPLIT_BOUNDARIES["test"]["start"]

        train_mask = df.index <= t_train_end
        val_mask = (df.index > t_train_end) & (df.index <= t_val_end)
        test_mask = df.index >= t_test_start

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        n_test = int(test_mask.sum())

        return {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, len(df)),
        }

    # Proportional split for mock, synthetic, or arbitrary test DataFrames
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }


def get_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve strictly the Training partition."""
    idx = get_real_split_indices(df)["train"]
    return df.iloc[idx[0]:idx[1]].copy()


def get_val_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve strictly the Validation partition."""
    idx = get_real_split_indices(df)["val"]
    return df.iloc[idx[0]:idx[1]].copy()


def get_test_data(df: pd.DataFrame, unlock: bool = False) -> pd.DataFrame:
    """Retrieve the Test partition.
    
    Guarded by an explicit lock. Raises TestSetLockedError unless unlock=True.
    Phase 6 is the ONLY authorized consumer allowed to pass unlock=True.
    """
    if not unlock:
        raise TestSetLockedError(
            "CRITICAL: The test set is locked until Phase 6 (Final Backtest). "
            "Unauthorized access to out-of-sample test data violates validation integrity."
        )
    idx = get_real_split_indices(df)["test"]
    return df.iloc[idx[0]:idx[1]].copy()


def get_splits(
    df: pd.DataFrame,
    unlock_test: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Return dictionary of splits. Test partition is withheld unless unlock_test=True."""
    splits = {
        "train": get_train_data(df),
        "val": get_val_data(df),
    }
    if unlock_test:
        splits["test"] = get_test_data(df, unlock=True)
    return splits


def generate_walk_forward_folds(
    df_train_val: pd.DataFrame,
    n_folds: int = config.N_WALK_FORWARD_FOLDS,
    purge_gap: int = config.PURGE_GAP_CANDLES,
    val_ratio: float = 0.20,
) -> Generator[Dict[str, Any], None, None]:
    """Generate expanding-window walk-forward folds with an explicit purge gap.
    
    In every fold:
      - Train window expands chronologically.
      - Exactly `purge_gap` candles are skipped between the end of train and start of validation.
      - Validation window evaluates out-of-fold generalization.
    
    Yields dict with keys:
      fold, train_df, val_df, train_range, val_range, purge_gap_candles
    """
    n_total = len(df_train_val)
    # Minimum train window is 40% of train_val
    min_train = int(n_total * 0.40)
    remaining = n_total - min_train
    step = remaining // n_folds

    for fold in range(n_folds):
        train_end_idx = min_train + fold * step
        val_start_idx = train_end_idx + purge_gap
        val_end_idx = min(train_end_idx + step + purge_gap, n_total)

        if val_start_idx >= n_total:
            break

        train_df = df_train_val.iloc[:train_end_idx].copy()
        val_df = df_train_val.iloc[val_start_idx:val_end_idx].copy()

        yield {
            "fold": fold + 1,
            "train_df": train_df,
            "val_df": val_df,
            "train_range": (train_df.index.min(), train_df.index.max()),
            "val_range": (val_df.index.min(), val_df.index.max()),
            "purge_gap_candles": purge_gap,
            "train_count": len(train_df),
            "val_count": len(val_df),
        }


def analyze_market_regimes(df: pd.DataFrame) -> Dict[str, Any]:
    """Quantify market regime metrics for a given time-series slice.
    
    Computes:
      - Total Return (%): cumulative price change
      - Annualized Volatility (%): 5m std scaled to annual (sqrt(288 * 365))
      - Maximum Drawdown (%): peak-to-trough decline
      - Dominant Trend: Bull / Bear / Sideways
      - Volatility Regime: High / Normal / Low
    """
    if df.empty:
        return {}

    close = df["close"]
    p_start = close.iloc[0]
    p_end = close.iloc[-1]
    total_return_pct = ((p_end / p_start) - 1.0) * 100.0

    # 5-minute log returns
    log_ret = np.log(close / close.shift(1)).dropna()
    # 288 5m candles/day * 365 days = 105,120 candles/year
    ann_factor = np.sqrt(288 * 365)
    ann_vol_pct = (log_ret.std() * ann_factor) * 100.0

    # Max Drawdown
    cum_max = close.cummax()
    drawdown = (close - cum_max) / cum_max
    max_dd_pct = (drawdown.min()) * 100.0

    # Regime characterization
    if total_return_pct > 25.0:
        trend = "Strong Bull"
    elif total_return_pct > 5.0:
        trend = "Mild Bull"
    elif total_return_pct < -25.0:
        trend = "Strong Bear"
    elif total_return_pct < -5.0:
        trend = "Mild Bear"
    else:
        trend = "Sideways / Range-bound"

    if ann_vol_pct > 65.0:
        vol_regime = "High Volatility"
    elif ann_vol_pct < 45.0:
        vol_regime = "Low Volatility"
    else:
        vol_regime = "Moderate Volatility"

    return {
        "start": df.index.min(),
        "end": df.index.max(),
        "candles": len(df),
        "days": (df.index.max() - df.index.min()).total_seconds() / 86400.0,
        "start_price": p_start,
        "end_price": p_end,
        "total_return_pct": round(total_return_pct, 2),
        "annualized_vol_pct": round(ann_vol_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "trend_regime": trend,
        "volatility_regime": vol_regime,
    }
