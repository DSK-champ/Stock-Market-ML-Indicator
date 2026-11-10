"""Resumable Binance Kline Data Extraction.

Downloads historical klines from Binance API with chunked checkpointing,
rate-limit backoff, dynamic timestamp unit verification, and future-candle rejection.
Saves raw API data completely untouched to data/raw/.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timezone

import pandas as pd
import requests

from src import config
from src.utils import setup_logger

logger = setup_logger("extraction")

CHUNKS_DIR = config.DATA_RAW_DIR / "chunks"


def detect_timestamp_unit(sample_ts: int) -> str:
    """Verify timestamp unit (ms vs us vs s) from raw integer value.
    
    Binance documentation specifies milliseconds (13 digits for 2020-2030).
    This function dynamically verifies the unit from data rather than assuming.
    """
    ts_val = int(sample_ts)
    if 1e11 <= ts_val < 1e14:
        return "ms"
    elif 1e14 <= ts_val < 1e17:
        return "us"
    elif 1e8 <= ts_val < 1e11:
        return "s"
    raise ValueError(f"Unrecognized timestamp magnitude: {ts_val}")


def interval_to_ms(interval: str) -> int:
    """Convert standard interval string (e.g. '5m', '1h', '1d') to milliseconds."""
    units = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    if not isinstance(interval, str) or len(interval) < 2:
        raise ValueError(f"Invalid interval: {interval}")
    unit = interval[-1]
    amount = interval[:-1]
    if unit not in units or not amount.isdigit():
        raise ValueError(f"Unsupported interval format: {interval}")
    return int(amount) * units[unit]


def fetch_klines_page(
    symbol: str = config.SYMBOL,
    interval: str = config.TIMEFRAME,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = config.DEFAULT_PAGE_LIMIT,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
) -> List[List[Any]]:
    """Fetch a single page of klines directly from Binance API.
    
    Returns the untouched raw list of lists as returned by the API.
    Respects rate limits and handles HTTP 429/418 with exponential backoff.
    """
    client = session or requests.Session()
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    for attempt in range(max_retries):
        try:
            resp = client.get(config.BINANCE_BASE_URL, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code == 418:
                retry_after = int(resp.headers.get("Retry-After", 10 * (2 ** attempt)))
                logger.warning("Rate limit hit (%d). Backing off for %ds...", resp.status_code, retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Binance API response payload: {data}")
            return data
        except (requests.RequestException, RuntimeError) as e:
            if attempt == max_retries - 1:
                logger.error("Failed fetching klines after %d attempts: %s", max_retries, e)
                raise
            sleep_sec = 2 ** attempt
            logger.warning("Request failed (%s). Retrying in %ds...", e, sleep_sec)
            time.sleep(sleep_sec)
    return []


def get_latest_checkpoint_time(chunks_dir: Path = CHUNKS_DIR) -> Optional[int]:
    """Inspect existing raw chunk files on disk to find the highest fetched open_time."""
    if not chunks_dir.exists():
        return None
    chunk_files = sorted(chunks_dir.glob("raw_chunk_*.parquet"))
    if not chunk_files:
        return None
    
    latest_ts = 0
    for cf in chunk_files:
        try:
            df_chunk = pd.read_parquet(cf, columns=["open_time"])
            if not df_chunk.empty:
                max_ts = int(df_chunk["open_time"].max())
                if max_ts > latest_ts:
                    latest_ts = max_ts
        except Exception as e:
            logger.warning("Could not read chunk %s: %s", cf, e)
    
    return latest_ts if latest_ts > 0 else None


def download_historical_raw_data(
    symbol: str = config.SYMBOL,
    interval: str = config.TIMEFRAME,
    start_time_str: str = config.START_DATE_STR,
    end_time_str: str = config.END_DATE_STR,
    output_parquet_path: Path = config.RAW_PARQUET_PATH,
    chunks_dir: Path = CHUNKS_DIR,
    pause_sec: float = 0.05,
    chunk_size_candles: int = 20_000,
) -> pd.DataFrame:
    """Download historical Binance klines with chunked checkpoints.
    
    Raw data is preserved completely UNTOUCHED with all 12 Binance columns.
    Resumes automatically from the last checkpoint if interrupted.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    start_ms = int(pd.Timestamp(start_time_str).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_time_str).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval_ms = interval_to_ms(interval)

    # Resume from checkpoint if available
    latest_checkpoint = get_latest_checkpoint_time(chunks_dir)
    if latest_checkpoint is not None and latest_checkpoint >= start_ms:
        cursor_ms = latest_checkpoint + interval_ms
        logger.info("Resuming extraction from checkpoint: %s (ms: %d)", 
                    pd.to_datetime(cursor_ms, unit="ms", utc=True), cursor_ms)
    else:
        cursor_ms = start_ms
        logger.info("Starting fresh extraction from: %s (ms: %d)", start_time_str, start_ms)

    session = requests.Session()
    current_chunk_rows: List[List[Any]] = []
    chunk_index = len(list(chunks_dir.glob("raw_chunk_*.parquet")))

    total_expected = config.TOTAL_EXPECTED_CANDLES
    logger.info("Target date range: %s to %s (Expected ~%d candles)", start_time_str, end_time_str, total_expected)

    while cursor_ms < end_ms:
        # Request batch
        page = fetch_klines_page(
            symbol=symbol,
            interval=interval,
            start_ms=cursor_ms,
            end_ms=end_ms - 1,  # Open time must be strictly before end_ms
            limit=config.DEFAULT_PAGE_LIMIT,
            session=session,
        )

        if not page:
            logger.info("No more candles returned by Binance at cursor %d. Pagination complete.", cursor_ms)
            break

        # Check timestamp unit on first page received
        ts_unit = detect_timestamp_unit(page[0][0])
        if ts_unit != "ms":
            raise ValueError(f"Expected millisecond timestamps from Binance, detected '{ts_unit}'.")

        for candle in page:
            candle_open_ms = int(candle[0])
            candle_close_ms = int(candle[6])

            # Rule 3: Reject in-progress or future candle
            if candle_close_ms > now_ms:
                logger.info("Skipping unfinished candle with close_time in the future: %d", candle_close_ms)
                continue
            if candle_open_ms >= end_ms:
                continue

            current_chunk_rows.append(candle)

        last_candle_open_ms = int(page[-1][0])
        next_cursor_ms = last_candle_open_ms + interval_ms
        if next_cursor_ms <= cursor_ms:
            raise RuntimeError(f"Pagination made no forward progress: cursor={cursor_ms}, next={next_cursor_ms}")
        cursor_ms = next_cursor_ms

        # Checkpoint if chunk reaches threshold
        if len(current_chunk_rows) >= chunk_size_candles:
            chunk_file = chunks_dir / f"raw_chunk_{chunk_index:04d}.parquet"
            df_chunk = pd.DataFrame(current_chunk_rows, columns=config.BINANCE_COLUMNS)
            df_chunk.to_parquet(chunk_file, index=False)
            logger.info("Saved raw checkpoint chunk %s (%d rows)", chunk_file.name, len(df_chunk))
            chunk_index += 1
            current_chunk_rows = []

        if pause_sec > 0:
            time.sleep(pause_sec)

    # Save remaining rows in final chunk
    if current_chunk_rows:
        chunk_file = chunks_dir / f"raw_chunk_{chunk_index:04d}.parquet"
        df_chunk = pd.DataFrame(current_chunk_rows, columns=config.BINANCE_COLUMNS)
        df_chunk.to_parquet(chunk_file, index=False)
        logger.info("Saved final raw checkpoint chunk %s (%d rows)", chunk_file.name, len(df_chunk))

    # Assemble complete raw dataset from all chunks
    all_chunk_files = sorted(chunks_dir.glob("raw_chunk_*.parquet"))
    if not all_chunk_files:
        raise RuntimeError("No raw chunks downloaded or found on disk.")

    logger.info("Combining %d raw chunks into complete raw dataset...", len(all_chunk_files))
    chunk_dfs = [pd.read_parquet(cf) for cf in all_chunk_files]
    df_raw = pd.concat(chunk_dfs, ignore_index=True)

    # Ensure strictly untouched values, but drop any duplicate rows that may span chunk seams
    # Note: raw strings / integers are preserved exactly as Binance returned them
    df_raw = df_raw.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Filter to exact requested range
    df_raw["_open_int"] = df_raw["open_time"].astype("int64")
    df_raw = df_raw[(df_raw["_open_int"] >= start_ms) & (df_raw["_open_int"] < end_ms)].drop(columns=["_open_int"]).reset_index(drop=True)

    # Save complete raw untouched dataset
    df_raw.to_parquet(output_parquet_path, index=False)
    logger.info("Untouched raw dataset saved successfully to: %s (%d rows, %d columns)",
                output_parquet_path, len(df_raw), len(df_raw.columns))

    return df_raw


def fetch_historical_data(
    symbol: str = config.SYMBOL,
    interval: str = config.TIMEFRAME,
    days: float = 0.1,
    limit: int = 10,
) -> pd.DataFrame:
    """Convenience helper to fetch recent closed candles for testing and live inference."""
    raw_data = fetch_klines_page(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw_data, columns=config.BINANCE_COLUMNS)
    return df


if __name__ == "__main__":
    download_historical_raw_data()
