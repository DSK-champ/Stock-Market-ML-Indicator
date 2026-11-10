"""Data preprocessing pipeline: type casting, gap indexing, and clean Parquet export.

Transforms raw Binance data into a clean, strictly indexed time-series dataset.
Enforces UTC DatetimeIndex, standard column dtypes, deduplication, and
establishes contiguous group markers (`gap_id`, `is_gap_boundary`) so rolling features
and forward labels can never span across exchange downtime or data gaps.
Does NOT forward-fill missing prices.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src import config, io_utils
from src.utils import setup_logger
from src.validation import validate_ohlc_consistency, validate_volumes, find_time_gaps

logger = setup_logger("preprocessing")


def process_raw_dataframe(
    df_raw: pd.DataFrame,
    start_utc: pd.Timestamp = config.START_DATE_UTC,
    end_utc: pd.Timestamp = config.END_DATE_UTC,
) -> pd.DataFrame:
    """Clean and standardize raw Binance klines DataFrame into a production-grade time-indexed table.
    
    Steps:
      1. Parse open_time as UTC datetime.
      2. Set UTC DatetimeIndex and sort chronologically.
      3. Drop duplicate timestamps (keeping last).
      4. Drop 'ignore' and 'close_time' columns.
      5. Cast all prices and volume measurements to float64.
      6. Filter to exact [start_utc, end_utc) window.
      7. Detect gaps and assign `gap_id` and `is_gap_boundary` columns.
    """
    if "open_time" not in df_raw.columns:
        raise KeyError("Raw DataFrame must contain 'open_time' column.")

    df = df_raw.copy()

    # 1. Convert open_time to UTC datetime
    # Binance timestamps are milliseconds
    if pd.api.types.is_numeric_dtype(df["open_time"]):
        dt_index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    else:
        dt_index = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)

    df.index = dt_index
    df.index.name = "open_time"

    # 2. Sort chronologically
    df = df.sort_index()

    # 3. Deduplicate timestamps
    before_dedup = len(df)
    df = df[~df.index.duplicated(keep="last")]
    dedup_dropped = before_dedup - len(df)
    if dedup_dropped > 0:
        logger.info("Dropped %d duplicate timestamp rows.", dedup_dropped)

    # 4. Filter to exact date window [start_utc, end_utc)
    df = df[(df.index >= start_utc) & (df.index < end_utc)]

    # 5. Drop unused columns
    cols_to_drop = [c for c in ["ignore", "close_time", "open_time"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 6. Cast price and volume columns to float64
    for col in config.NUMERIC_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Expected numeric column '{col}' missing from raw data.")
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")

    # Retain strictly the standardized columns
    retained_cols = [c for c in config.OUTPUT_COLUMNS if c != "open_time"]
    df = df[retained_cols]

    # 7. Add gap identifiers without forward-filling prices
    # diff() gives interval from previous row to current row
    time_diff = df.index.to_series().diff()
    expected_delta = pd.Timedelta(minutes=config.INTERVAL_MINUTES)

    # Row 0 or any row where delta > 5 minutes is a gap boundary
    is_gap_boundary = (time_diff > expected_delta)
    is_gap_boundary.iloc[0] = True  # Start of dataset is a boundary

    # Contiguous segment ID increments at each gap boundary
    gap_id = is_gap_boundary.cumsum().astype("int32")

    df["gap_id"] = gap_id
    df["is_gap_boundary"] = is_gap_boundary.astype("bool")

    return df


def run_preprocessing_pipeline(
    raw_path: Path = config.RAW_DATA_PATH,
    clean_path: Path = config.CLEAN_DATA_PATH,
) -> pd.DataFrame:
    """Execute end-to-end preprocessing from raw data to clean dataset."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run extraction first.")

    logger.info("Loading raw data from: %s", raw_path)
    if str(raw_path).endswith(".csv"):
        df_raw = io_utils.load_raw_untouched_csv(raw_path)
    else:
        df_raw = pd.read_parquet(raw_path)
    logger.info("Raw data loaded: %d rows, %d columns", len(df_raw), len(df_raw.columns))

    df_clean = process_raw_dataframe(df_raw)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    if str(clean_path).endswith(".csv"):
        io_utils.save_clean_market_data_csv(df_clean, clean_path)
    else:
        df_clean.to_parquet(clean_path)
    logger.info("Clean dataset saved to %s (%d rows, %d columns)", clean_path, len(df_clean), len(df_clean.columns))

    return df_clean
