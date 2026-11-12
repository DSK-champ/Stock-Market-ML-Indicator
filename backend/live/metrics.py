"""Statistical metrics, confidence intervals, and baseline evaluations for live predictions."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

from backend.live import config


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf((1.0 + confidence) / 2.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def compute_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 12,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute moving-block bootstrap confidence interval for autocorrelated returns/verdicts."""
    n = len(y_true)
    if n < block_size * 2:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    acc_samples = []

    for _ in range(n_bootstrap):
        # Sample starting indices
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample_indices = np.concatenate([np.arange(s, s + block_size) for s in start_indices])[:n]
        boot_true = y_true[sample_indices]
        boot_pred = y_pred[sample_indices]
        acc_samples.append((boot_true == boot_pred).mean())

    low = float(np.percentile(acc_samples, 2.5))
    high = float(np.percentile(acc_samples, 97.5))
    return low, high


class LiveMetricsCalculator:
    """Calculates live scoring metrics from SQLite prediction ledger."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or config.LIVE_LEDGER_DB_PATH

    def _get_resolved_dataframe(self) -> pd.DataFrame:
        """Load resolved ledger rows into pandas DataFrame."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM predictions
                WHERE status IN ('RESOLVED', 'NO_CALL', 'VOID')
                ORDER BY candle_open_time ASC
            """
            df = pd.read_sql_query(query, conn)
        return df

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute full statistical metrics package on resolved predictions."""
        df = self._get_resolved_dataframe()

        total_candles = len(df)
        if total_candles == 0:
            return {
                "total_candles_evaluated": 0,
                "status": "No resolved predictions available yet.",
                "too_few_samples": True,
                "calls_needed_for_95pct_vs_coinflip": 2000,
            }

        # Filter directional calls (LONG / SHORT) with non-null verdicts
        dir_mask = df["action"].isin([1, -1]) & df["correct"].notna()
        df_dir = df[dir_mask].copy()
        n_dir = len(df_dir)

        # 1. Coverage & Signals per Day
        coverage_pct = (n_dir / total_candles) * 100.0 if total_candles > 0 else 0.0
        signals_per_day = coverage_pct * 288.0 / 100.0

        # 2. Directional Accuracy
        if n_dir > 0:
            k_correct = int(df_dir["correct"].sum())
            dir_acc = float(k_correct / n_dir) * 100.0
            wilson_low, wilson_high = compute_wilson_ci(k_correct, n_dir, confidence=0.95)

            # Block bootstrap
            y_true_dir = df_dir["actual_raw_sign_2"].values
            y_pred_dir = df_dir["action"].values
            boot_low, boot_high = compute_block_bootstrap_ci(y_true_dir, y_pred_dir, block_size=12)

            # Binomial tests
            p_val_coinflip = float(stats.binomtest(k_correct, n_dir, p=0.50, alternative="two-sided").pvalue)
            p_val_claimed = float(stats.binomtest(k_correct, n_dir, p=0.5216, alternative="two-sided").pvalue)

            # Accuracy by action
            longs = df_dir[df_dir["action"] == 1]
            shorts = df_dir[df_dir["action"] == -1]
            long_acc = float(longs["correct"].mean() * 100.0) if len(longs) > 0 else 0.0
            short_acc = float(shorts["correct"].mean() * 100.0) if len(shorts) > 0 else 0.0

            # Rolling windows
            acc_last_50 = float(df_dir["correct"].tail(50).mean() * 100.0) if n_dir >= 50 else dir_acc
            acc_last_200 = float(df_dir["correct"].tail(200).mean() * 100.0) if n_dir >= 200 else dir_acc

            # Last 24h
            if "candle_open_time" in df_dir.columns:
                last_time = pd.to_datetime(df_dir["candle_open_time"].max(), utc=True)
                cutoff_24h = last_time - pd.Timedelta(hours=24)
                df_24h = df_dir[pd.to_datetime(df_dir["candle_open_time"], utc=True) >= cutoff_24h]
                acc_last_24h = float(df_24h["correct"].mean() * 100.0) if len(df_24h) > 0 else dir_acc
            else:
                acc_last_24h = dir_acc

            # Baselines on same rows
            always_up_rate = float((y_true_dir == 1.0).mean() * 100.0)
            always_down_rate = float((y_true_dir == -1.0).mean() * 100.0)

            # Returns
            label_rets = df_dir["return_bps_label"].values
            gross_bps = float(np.mean(df_dir["action"].values * label_rets)) if len(label_rets) > 0 else 0.0
            net_bps = gross_bps - config.COST_FLOOR_BPS
        else:
            k_correct = 0
            dir_acc = 0.0
            wilson_low, wilson_high = 0.0, 0.0
            boot_low, boot_high = 0.0, 0.0
            p_val_coinflip = 1.0
            p_val_claimed = 1.0
            long_acc = 0.0
            short_acc = 0.0
            acc_last_50 = 0.0
            acc_last_200 = 0.0
            acc_last_24h = 0.0
            always_up_rate = 0.0
            always_down_rate = 0.0
            gross_bps = 0.0
            net_bps = 0.0

        # 3. 3-Class Confusion Matrix (against actual_target_2_S1)
        valid_tri = df[df["actual_target_2_S1"].notna()]
        if len(valid_tri) > 0:
            y_tri_true = valid_tri["actual_target_2_S1"].values.astype(int)
            y_tri_pred = valid_tri["action"].values.astype(int)

            c_matrix = []
            for act_true in [-1, 0, 1]:
                row = []
                for act_pred in [-1, 0, 1]:
                    cnt = int(np.sum((y_tri_true == act_true) & (y_tri_pred == act_pred)))
                    row.append(cnt)
                c_matrix.append(row)

            always_neu_rate = float((y_tri_true == 0).mean() * 100.0)
            three_class_acc = float((y_tri_true == y_tri_pred).mean() * 100.0)
        else:
            c_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            always_neu_rate = 0.0
            three_class_acc = 0.0

        return {
            "total_resolved_candles": total_candles,
            "directional_calls_count": n_dir,
            "too_few_samples": n_dir < 2000,
            "calls_needed_for_95pct_vs_coinflip": 2000,
            "directional_accuracy": {
                "overall_pct": round(dir_acc, 2),
                "last_50_calls_pct": round(acc_last_50, 2),
                "last_200_calls_pct": round(acc_last_200, 2),
                "last_24h_pct": round(acc_last_24h, 2),
                "long_accuracy_pct": round(long_acc, 2),
                "short_accuracy_pct": round(short_acc, 2),
                "wilson_95_ci": [round(wilson_low * 100.0, 2), round(wilson_high * 100.0, 2)],
                "block_bootstrap_95_ci": [round(boot_low * 100.0, 2), round(boot_high * 100.0, 2)],
                "p_value_vs_50pct_coinflip": round(p_val_coinflip, 4),
                "p_value_vs_52_16pct_claimed": round(p_val_claimed, 4),
            },
            "coverage": {
                "active_coverage_pct": round(coverage_pct, 2),
                "signals_per_day": round(signals_per_day, 1),
                "claimed_coverage_pct": 65.90,
                "claimed_signals_per_day": 189.79,
            },
            "returns": {
                "gross_profit_per_trade_bps": round(gross_bps, 2),
                "fee_deducted_per_trade_bps": round(config.COST_FLOOR_BPS, 2),
                "net_profit_per_trade_bps": round(net_bps, 2),
            },
            "benchmarks_on_same_rows": {
                "always_up_rate_pct": round(always_up_rate, 2),
                "always_down_rate_pct": round(always_down_rate, 2),
                "always_neutral_rate_pct": round(always_neu_rate, 2),
                "three_class_accuracy_pct": round(three_class_acc, 2),
            },
            "confusion_matrix_3class": {
                "labels": ["DOWN (-1)", "NEUTRAL (0)", "UP (+1)"],
                "matrix": c_matrix,
            },
            "notes": (
                "The models were trained on historical Bitcoin Spot data (2023-2025). "
                "Live market performance may fluctuate across differing volatility or trend regimes."
            ),
        }

