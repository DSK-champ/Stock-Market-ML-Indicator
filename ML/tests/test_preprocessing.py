"""Unit tests for data preprocessing and validation.

Verifies:
  - Synthetic gap detection and structured reporting
  - Duplicate row deduplication and index integrity
  - Vectorized detection of invalid OHLC geometric violations
  - Rejection of unfinished / future-dated candles
  - Correct dtypes, column retention, and Parquet round-trip fidelity
All tests run locally on synthetic fixtures with zero network dependency.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import process_raw_dataframe
from src.validation import (
    validate_ohlc_consistency,
    validate_volumes,
    find_time_gaps,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Create a minimal synthetic raw DataFrame conforming to Binance API schema."""
    base_ts = 1700000000000  # ms epoch
    rows = []
    price = 35000.0
    for i in range(10):
        open_time = base_ts + i * 300_000
        close_time = open_time + 299_999
        rows.append([
            open_time,
            str(price),
            str(price + 50.0),
            str(price - 30.0),
            str(price + 10.0),
            "10.5",
            close_time,
            "367500.0",
            120,
            "6.2",
            "217000.0",
            "0",
        ])
        price += 10.0

    return pd.DataFrame(rows, columns=config.BINANCE_COLUMNS)


def test_synthetic_duplicate_caught(sample_raw_data: pd.DataFrame):
    """Verify that duplicate timestamps are caught and deduplicated."""
    # Duplicate the 3rd row
    dup_row = sample_raw_data.iloc[[2]].copy()
    data_with_dup = pd.concat([sample_raw_data, dup_row], ignore_index=True)

    assert len(data_with_dup) == 11
    # Process
    t_start = pd.to_datetime(sample_raw_data["open_time"].iloc[0], unit="ms", utc=True)
    t_end = pd.to_datetime(sample_raw_data["open_time"].iloc[-1] + 300_000, unit="ms", utc=True)
    clean_df = process_raw_dataframe(data_with_dup, start_utc=t_start, end_utc=t_end)

    assert len(clean_df) == 10
    assert not clean_df.index.duplicated().any()
    assert clean_df.index.is_monotonic_increasing


def test_synthetic_gap_detection(sample_raw_data: pd.DataFrame):
    """Verify that a time gap is detected, reported with exact missing candles, and increments gap_id."""
    # Introduce a 20-minute gap by dropping rows 4, 5, 6
    gapped_raw = sample_raw_data.drop(index=[4, 5, 6]).reset_index(drop=True)

    t_start = pd.to_datetime(sample_raw_data["open_time"].iloc[0], unit="ms", utc=True)
    t_end = pd.to_datetime(sample_raw_data["open_time"].iloc[-1] + 300_000, unit="ms", utc=True)
    clean_df = process_raw_dataframe(gapped_raw, start_utc=t_start, end_utc=t_end)

    # 1. Validation gap detection
    gaps_table = find_time_gaps(clean_df)
    assert len(gaps_table) == 1
    gap_row = gaps_table.iloc[0]
    assert gap_row["missing_candles"] == 3
    assert gap_row["duration_minutes"] == 20.0

    # 2. Gap segment marker check
    assert clean_df["gap_id"].nunique() == 2  # Two distinct continuous chunks
    assert clean_df["is_gap_boundary"].sum() == 2  # Start of series + after the gap


def test_bad_ohlc_detected():
    """Verify that invalid OHLC geometry and negative prices are caught by vectorized validation."""
    timestamps = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")

    # Case 1: High lower than Close
    df_bad_high = pd.DataFrame({
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [105.0, 95.0, 105.0, 105.0],  # Row 1 high < open/close
        "low": [95.0, 90.0, 95.0, 95.0],
        "close": [102.0, 98.0, 102.0, 102.0],
    }, index=timestamps)
    res_high = validate_ohlc_consistency(df_bad_high)
    assert res_high["valid"] is False
    assert res_high["high_violations"] == 1

    # Case 2: Low higher than Open
    df_bad_low = pd.DataFrame({
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [110.0, 110.0, 110.0, 110.0],
        "low": [95.0, 102.0, 95.0, 95.0],  # Row 1 low > open/close
        "close": [105.0, 105.0, 105.0, 105.0],
    }, index=timestamps)
    res_low = validate_ohlc_consistency(df_bad_low)
    assert res_low["valid"] is False
    assert res_low["low_violations"] == 1

    # Case 3: Negative price
    df_neg = pd.DataFrame({
        "open": [100.0, -10.0, 100.0, 100.0],
        "high": [105.0, 105.0, 105.0, 105.0],
        "low": [95.0, 95.0, 95.0, 95.0],
        "close": [102.0, 102.0, 102.0, 102.0],
    }, index=timestamps)
    res_neg = validate_ohlc_consistency(df_neg)
    assert res_neg["valid"] is False
    assert res_neg["non_positive_prices"] == 1


def test_future_dated_candle_rejected(sample_raw_data: pd.DataFrame):
    """Verify that candles with open_time >= end_utc or close_time in the future are rejected."""
    # Add an unfinished candle dated in year 2099
    future_open = int(pd.Timestamp("2099-01-01 00:00:00+00:00").timestamp() * 1000)
    future_row = sample_raw_data.iloc[[0]].copy()
    future_row["open_time"] = future_open
    future_row["close_time"] = future_open + 299_999

    data_with_future = pd.concat([sample_raw_data, future_row], ignore_index=True)

    t_start = pd.to_datetime(sample_raw_data["open_time"].iloc[0], unit="ms", utc=True)
    t_end = pd.to_datetime(sample_raw_data["open_time"].iloc[-1] + 300_000, unit="ms", utc=True)
    clean_df = process_raw_dataframe(data_with_future, start_utc=t_start, end_utc=t_end)

    # Future candle must be filtered out
    assert clean_df.index.max() < pd.Timestamp("2099-01-01 00:00:00+00:00")
    assert len(clean_df) == 10


def test_clean_parquet_roundtrip_fidelity(sample_raw_data: pd.DataFrame, tmp_path):
    """Verify clean DataFrame preserves column names, float64 types, and UTC DatetimeIndex across Parquet save/load."""
    t_start = pd.to_datetime(sample_raw_data["open_time"].iloc[0], unit="ms", utc=True)
    t_end = pd.to_datetime(sample_raw_data["open_time"].iloc[-1] + 300_000, unit="ms", utc=True)
    clean_df = process_raw_dataframe(sample_raw_data, start_utc=t_start, end_utc=t_end)

    test_file = tmp_path / "test_clean.parquet"
    clean_df.to_parquet(test_file)

    loaded_df = pd.read_parquet(test_file)
    assert isinstance(loaded_df.index, pd.DatetimeIndex)
    assert loaded_df.index.tz is not None
    assert str(loaded_df.index.tz) == "UTC"
    assert loaded_df["close"].dtype == np.float64
    assert "ignore" not in loaded_df.columns
    assert "close_time" not in loaded_df.columns
    assert "quote_volume" in loaded_df.columns


def test_clean_csv_roundtrip_fidelity(sample_raw_data: pd.DataFrame, tmp_path):
    """Verify clean DataFrame preserves column schema, UTC DatetimeIndex, and exact values via io_utils."""
    from src import io_utils
    t_start = pd.to_datetime(sample_raw_data["open_time"].iloc[0], unit="ms", utc=True)
    t_end = pd.to_datetime(sample_raw_data["open_time"].iloc[-1] + 300_000, unit="ms", utc=True)
    clean_df = process_raw_dataframe(sample_raw_data, start_utc=t_start, end_utc=t_end)

    test_csv_path = tmp_path / "test_clean.csv"
    io_utils.save_clean_market_data_csv(clean_df, test_csv_path)

    loaded_df = io_utils.load_clean_market_data_csv(test_csv_path)
    assert isinstance(loaded_df.index, pd.DatetimeIndex)
    assert str(loaded_df.index.tz) == "UTC"
    assert loaded_df.shape == clean_df.shape
    for col in config.NUMERIC_COLUMNS:
        np.testing.assert_allclose(loaded_df[col].values, clean_df[col].values, rtol=1e-8, atol=1e-8)
    assert "taker_buy_base_volume" in loaded_df.columns
    pd.testing.assert_frame_equal(clean_df, loaded_df)
