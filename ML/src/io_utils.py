"""Centralized I/O utilities for CSV market data storage and retrieval.

Enforces strict single-entry reading and writing:
  - Timezone-aware ISO 8601 UTC open_time
  - Full floating-point precision (repr-level)
  - Explicit dtype casting (float64 for market measurements)
  - Zero ad-hoc pd.read_csv calls across the project
"""

from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logger

logger = setup_logger("io_utils")


def save_clean_market_data_csv(df: pd.DataFrame, filepath: Union[str, Path] = config.DATA_PROCESSED_DIR / "clean.csv") -> Path:
    """Save cleaned time-indexed market data to CSV with ISO UTC timestamps and full precision."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()

    # Ensure index is open_time in UTC
    if isinstance(df_to_save.index, pd.DatetimeIndex):
        if df_to_save.index.tz is None:
            df_to_save.index = df_to_save.index.tz_localize("UTC")
        else:
            df_to_save.index = df_to_save.index.tz_convert("UTC")
        df_to_save.index.name = "open_time"
        # Export with ISO 8601 formatting and full IEEE 754 repr-level float precision
        df_to_save.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        if "open_time" not in df_to_save.columns:
            raise KeyError("DataFrame must possess an 'open_time' index or column.")
        df_to_save["open_time"] = pd.to_datetime(df_to_save["open_time"], utc=True)
        df_to_save.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S%z")

    logger.info("Saved clean market data CSV to %s (%d rows)", path, len(df_to_save))
    return path


def load_clean_market_data_csv(filepath: Union[str, Path] = config.DATA_PROCESSED_DIR / "clean.csv") -> pd.DataFrame:
    """Load clean market data CSV, establishing UTC DatetimeIndex and strict float64 dtypes."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clean CSV file not found: {path}")

    # Explicit dtype specifications
    dtype_dict = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_volume": "float64",
        "number_of_trades": "float64",
        "taker_buy_base_volume": "float64",
        "taker_buy_quote_volume": "float64",
    }
    if "gap_id" in pd.read_csv(path, nrows=1).columns:
        dtype_dict["gap_id"] = "int32"
    if "is_gap_boundary" in pd.read_csv(path, nrows=1).columns:
        dtype_dict["is_gap_boundary"] = "bool"

    df = pd.read_csv(
        path,
        parse_dates=["open_time"],
        index_col="open_time",
        dtype=dtype_dict,
    )

    # Ensure DatetimeIndex is explicitly UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


def save_raw_untouched_csv(df_raw: pd.DataFrame, filepath: Union[str, Path] = config.DATA_RAW_DIR / "BTCUSDT_5m_raw.csv") -> Path:
    """Save untouched raw Binance klines to CSV preserving all 12 raw columns exactly."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Raw data exported as-is without index modification at full repr-level precision
    df_raw.to_csv(path, index=False)
    logger.info("Saved untouched raw CSV to %s (%d rows)", path, len(df_raw))
    return path


def load_raw_untouched_csv(filepath: Union[str, Path] = config.DATA_RAW_DIR / "BTCUSDT_5m_raw.csv") -> pd.DataFrame:
    """Load untouched raw Binance CSV maintaining original column schemas."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV file not found: {path}")

    df_raw = pd.read_csv(path, dtype={"open_time": "int64", "close_time": "int64"})
    return df_raw
