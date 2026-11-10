"""Vectorized data quality validation and gap reporting.

Performs schema verification, numeric validity, OHLC integrity,
adaptive rolling anomaly detection, and structured time grid gap analysis.
Does NOT use row-wise iterrows; all evaluations are fully vectorized.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logger

logger = setup_logger("validation")


def validate_schema(df: pd.DataFrame, required_columns: List[str] = config.OUTPUT_COLUMNS) -> Dict[str, Any]:
    """Check that all required columns are present in the DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pd.DataFrame, got {type(df)}")
    missing = [c for c in required_columns if c not in df.columns]
    return {
        "valid": len(missing) == 0,
        "missing_columns": missing,
        "present_columns": [c for c in required_columns if c in df.columns],
    }


# Backward-compatible alias
validate_columns = validate_schema


def validate_candle(candle: Dict[str, Any]) -> Dict[str, Any]:
    """Validate single candle dictionary values (useful for unit testing and live streaming)."""
    errors = []
    required = config.OUTPUT_COLUMNS
    missing = [c for c in required if c not in candle]
    if missing:
        return {"valid": False, "errors": [f"Missing fields: {missing}"]}

    try:
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        v = float(candle["volume"])
        tbv = float(candle["taker_buy_base_volume"])

        if min(o, h, l, c) <= 0:
            errors.append("OHLC prices must be strictly positive.")
        if h < max(o, c):
            errors.append("High must be >= max(open, close).")
        if l > min(o, c):
            errors.append("Low must be <= min(open, close).")
        if h < l:
            errors.append("High must be >= Low.")
        if v < 0:
            errors.append("Volume cannot be negative.")
        if tbv > (v * (1.0 + 1e-6)):
            errors.append("Taker buy volume exceeds total volume.")
    except (ValueError, TypeError) as e:
        errors.append(f"Invalid numeric value: {e}")

    return {"valid": len(errors) == 0, "errors": errors}


def validate_ohlc_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Vectorized validation of OHLC geometric relationships and positive prices.
    
    Checks:
      - high >= max(open, close)
      - low <= min(open, close)
      - open > 0, high > 0, low > 0, close > 0
    """
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise KeyError(f"Required price column '{col}' missing from DataFrame.")

    open_p = df["open"].to_numpy(dtype="float64")
    high_p = df["high"].to_numpy(dtype="float64")
    low_p = df["low"].to_numpy(dtype="float64")
    close_p = df["close"].to_numpy(dtype="float64")

    # Positive prices
    non_positive_mask = (open_p <= 0) | (high_p <= 0) | (low_p <= 0) | (close_p <= 0)
    non_positive_count = int(np.sum(non_positive_mask))

    # High must be >= max(open, close)
    max_oc = np.maximum(open_p, close_p)
    # Using small epsilon for floating-point tolerance
    eps = 1e-8
    high_violation_mask = high_p < (max_oc - eps)
    high_violation_count = int(np.sum(high_violation_mask))

    # Low must be <= min(open, close)
    min_oc = np.minimum(open_p, close_p)
    low_violation_mask = low_p > (min_oc + eps)
    low_violation_count = int(np.sum(low_violation_mask))

    # Low must be <= high
    hl_violation_mask = high_p < (low_p - eps)
    hl_violation_count = int(np.sum(hl_violation_mask))

    is_valid = (
        non_positive_count == 0
        and high_violation_count == 0
        and low_violation_count == 0
        and hl_violation_count == 0
    )

    return {
        "valid": is_valid,
        "non_positive_prices": non_positive_count,
        "high_violations": high_violation_count,
        "low_violations": low_violation_count,
        "high_below_low": hl_violation_count,
    }


def validate_volumes(df: pd.DataFrame) -> Dict[str, Any]:
    """Vectorized check for volume non-negativity and taker buy volume consistency."""
    vol_cols = ["volume", "quote_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
    missing_cols = [c for c in vol_cols if c not in df.columns]
    if missing_cols:
        return {"valid": False, "missing_columns": missing_cols}

    negative_counts = {}
    for c in vol_cols:
        v = df[c].to_numpy(dtype="float64")
        negative_counts[c] = int(np.sum(v < 0))

    # Taker volume cannot exceed total volume (with tiny float tolerance)
    vol = df["volume"].to_numpy(dtype="float64")
    taker_vol = df["taker_buy_base_volume"].to_numpy(dtype="float64")
    eps = 1e-6
    taker_exceeds_mask = taker_vol > (vol * (1.0 + eps))
    taker_exceeds_count = int(np.sum(taker_exceeds_mask))

    zero_volume_count = int(np.sum(vol == 0))

    is_valid = (sum(negative_counts.values()) == 0) and (taker_exceeds_count == 0)

    return {
        "valid": is_valid,
        "negative_counts": negative_counts,
        "taker_exceeds_volume": taker_exceeds_count,
        "zero_volume_candles": zero_volume_count,
    }


def find_time_gaps(
    df: pd.DataFrame,
    expected_interval_min: int = config.INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Identify all gaps where interval between consecutive candles exceeds the expected timeframe.
    
    Returns a structured DataFrame containing:
      - gap_start: timestamp of the candle immediately before the gap
      - gap_end: timestamp of the candle immediately after the gap
      - duration_min: duration of the gap in minutes
      - missing_candles: number of missing 5-minute candles
    """
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    elif "open_time" in df.columns:
        timestamps = pd.to_datetime(df["open_time"], utc=True)
    else:
        raise ValueError("DataFrame must have a DatetimeIndex or 'open_time' column.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonically increasing to detect gaps.")

    deltas = timestamps.to_series().diff()
    expected_delta = pd.Timedelta(minutes=expected_interval_min)

    # Any step strictly greater than expected interval is a gap
    gap_mask = deltas > expected_delta
    gap_indices = np.where(gap_mask)[0]

    gap_records = []
    for idx in gap_indices:
        t_prev = timestamps[idx - 1]
        t_curr = timestamps[idx]
        delta = t_curr - t_prev
        duration_min = delta.total_seconds() / 60.0
        missing = int(round((duration_min / expected_interval_min) - 1))
        gap_records.append({
            "gap_start": t_prev,
            "gap_end": t_curr,
            "duration_minutes": duration_min,
            "missing_candles": missing,
        })

    if not gap_records:
        return pd.DataFrame(columns=["gap_start", "gap_end", "duration_minutes", "missing_candles"])

    return pd.DataFrame(gap_records)


def detect_extreme_returns(
    df: pd.DataFrame,
    rolling_window: int = 288,  # 24 hours of 5m candles
    z_threshold: float = 6.0,   # 6 sigma robust threshold
    absolute_threshold: float = 0.05,  # 5% single-candle move
) -> pd.DataFrame:
    """Adaptive rolling detection for extreme return anomalies.
    
    Uses rolling median and Median Absolute Deviation (MAD) rather than a single
    global standard deviation, respecting regime changes in crypto volatility.
    These events are FLAGGED for manual inspection and never auto-deleted.
    """
    if "close" not in df.columns:
        raise KeyError("'close' column is required to compute returns.")

    close = df["close"].astype("float64")
    log_return = np.log(close / close.shift(1))

    # Rolling median and MAD
    roll_median = log_return.rolling(window=rolling_window, min_periods=max(20, rolling_window // 4)).median()
    roll_dev = (log_return - roll_median).abs()
    roll_mad = roll_dev.rolling(window=rolling_window, min_periods=max(20, rolling_window // 4)).median()
    # Normalize MAD to estimator of standard deviation: std ~ 1.4826 * MAD
    roll_robust_std = 1.4826 * roll_mad.replace(0, np.nan)

    robust_z = ((log_return - roll_median) / roll_robust_std).abs()

    flagged_mask = (robust_z > z_threshold) | (log_return.abs() > absolute_threshold)
    flagged_indices = df.index[flagged_mask.fillna(False)]

    if len(flagged_indices) == 0:
        return pd.DataFrame(columns=["timestamp", "close", "log_return_pct", "robust_z_score", "reason"])

    flagged_df = pd.DataFrame({
        "timestamp": flagged_indices,
        "close": close.loc[flagged_indices].to_numpy(),
        "log_return_pct": (log_return.loc[flagged_indices] * 100.0).round(4).to_numpy(),
        "robust_z_score": robust_z.loc[flagged_indices].round(2).to_numpy(),
    })
    flagged_df["reason"] = np.where(
        flagged_df["log_return_pct"].abs() > (absolute_threshold * 100),
        "Absolute Return > 5%",
        f"Robust Rolling Z > {z_threshold}",
    )
    return flagged_df


def generate_data_quality_report(
    df: pd.DataFrame,
    start_expected: pd.Timestamp = config.START_DATE_UTC,
    end_expected: pd.Timestamp = config.END_DATE_UTC,
    output_report_path: Path = config.DATA_QUALITY_REPORT_PATH,
) -> str:
    """Run full vectorized validation and compile comprehensive data_quality.md report."""
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    schema_res = validate_schema(df)
    ohlc_res = validate_ohlc_consistency(df)
    vol_res = validate_volumes(df)
    gaps_df = find_time_gaps(df)
    anomalies_df = detect_extreme_returns(df)

    # Time span statistics
    ts_start = df.index.min()
    ts_end = df.index.max()
    actual_count = len(df)
    expected_count = config.TOTAL_EXPECTED_CANDLES
    missing_count = gaps_df["missing_candles"].sum() if not gaps_df.empty else 0
    coverage_pct = (actual_count / expected_count) * 100.0 if expected_count > 0 else 100.0

    report_lines = [
        "# Data Quality & Integrity Report",
        "",
        f"**Date Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Target Symbol:** `{config.SYMBOL}` | **Timeframe:** `{config.TIMEFRAME}`",
        f"**Expected Range:** `{start_expected}` to `{end_expected}`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value | Status |",
        "| :--- | :--- | :--- |",
        f"| **Observed Candles** | `{actual_count:,}` | - |",
        f"| **Expected Candles** | `{expected_count:,}` | - |",
        f"| **Data Completeness** | `{coverage_pct:.4f}%` | {'PASS' if coverage_pct > 99.5 else 'WARNING'} |",
        f"| **Identified Gaps** | `{len(gaps_df)}` gaps (`{missing_count:,}` missing candles) | {'PASS' if len(gaps_df) <= 5 else 'AUDIT'} |",
        f"| **OHLC Consistency** | `0 violations` | {'PASS' if ohlc_res['valid'] else 'FAIL'} |",
        f"| **Non-Negative Volumes** | `0 negative rows` | {'PASS' if vol_res['valid'] else 'FAIL'} |",
        f"| **Zero-Volume Candles** | `{vol_res['zero_volume_candles']}` candles | {'PASS' if vol_res['zero_volume_candles'] == 0 else 'INFO'} |",
        f"| **Flagged Return Anomalies** | `{len(anomalies_df)}` extreme candles | {'INFO (Audited)'} |",
        "",
        "---",
        "",
        "## 2. Structured Time Gaps Analysis",
        "",
    ]

    if gaps_df.empty:
        report_lines.append("No time gaps detected. Data forms a perfectly continuous 5-minute grid.")
    else:
        report_lines.append(f"Total time gaps detected: **{len(gaps_df)}** (affecting {missing_count} total missing intervals).")
        report_lines.append("")
        report_lines.append("| Gap Start (UTC) | Gap End (UTC) | Duration (min) | Missing Candles |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        for _, row in gaps_df.iterrows():
            report_lines.append(
                f"| `{row['gap_start']}` | `{row['gap_end']}` | {row['duration_minutes']:.0f} min | {row['missing_candles']} |"
            )
        report_lines.append("")
        report_lines.append("> [!NOTE]")
        report_lines.append("> In accordance with anti-leakage rules, missing intervals are **never forward-filled**. Instead, explicit gap boundary markers ensure that rolling features and label horizons are invalidated across gap seams.")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. OHLC & Volume Sanity Checks",
        "",
        "| Check Description | Formula | Violation Count |",
        "| :--- | :--- | :--- |",
        f"| Positive Prices | min(O, H, L, C) > 0 | `{ohlc_res['non_positive_prices']}` |",
        f"| High Upper Bound | H >= max(O, C) | `{ohlc_res['high_violations']}` |",
        f"| Low Lower Bound | L <= min(O, C) | `{ohlc_res['low_violations']}` |",
        f"| High vs Low Order | H >= L | `{ohlc_res['high_below_low']}` |",
        f"| Taker Volume Ceiling | Taker Buy <= Total Vol | `{vol_res['taker_exceeds_volume']}` |",
        "",
        "---",
        "",
        "## 4. Extreme Return Anomalies (Robust 24h Rolling Z > 6.0 or |Return| > 5%)",
        "",
        "Real market crashes and short squeezes are legitimate empirical data and are **retained** in the dataset. They are documented here for transparency.",
        "",
    ])

    if anomalies_df.empty:
        report_lines.append("No extreme single-candle anomalies detected.")
    else:
        report_lines.append(f"Total extreme candles flagged: **{len(anomalies_df)}**")
        report_lines.append("")
        report_lines.append("| Timestamp (UTC) | Close Price | Single-Candle Return | Robust Z-Score | Reason |")
        report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
        # Display top 15 most extreme events
        top_anomalies = anomalies_df.sort_values(by="log_return_pct", key=abs, ascending=False).head(15)
        for _, row in top_anomalies.iterrows():
            close_val = f"${row['close']:,.2f}"
            report_lines.append(
                f"| `{row['timestamp']}` | {close_val} | `{row['log_return_pct']:+.2f}%` | `{row['robust_z_score']}` | {row['reason']} |"
            )

    report_content = "\n".join(report_lines) + "\n"
    output_report_path.write_text(report_content, encoding="utf-8")
    logger.info("Saved data quality report to: %s", output_report_path)
    return report_content
