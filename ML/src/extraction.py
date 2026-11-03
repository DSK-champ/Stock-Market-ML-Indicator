from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests


BASE_URL = "https://data-api.binance.vision/api/v3/klines"
DEFAULT_LIMIT = 1000

OUTPUT_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "quote_volume",
    "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
]

BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
]


# ============================================================
# INPUT VALIDATION AND CONVERSIONS
# ============================================================

def datetime_to_ms(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):
        if not pd.notna(value):
            raise ValueError("Timestamp must be finite.")
        return int(value)

    timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError("Invalid timestamp.")

    # Treat naive datetimes as UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return int(timestamp.timestamp() * 1000)

# =============================================================

def interval_to_ms(interval):
    """Convert supported Binance intervals to milliseconds."""

    units = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }

    if not isinstance(interval, str) or len(interval) < 2:
        raise ValueError("Invalid interval. Examples: 5m, 1h, 1d.")

    unit = interval[-1]
    amount_text = interval[:-1]

    if unit not in units or not amount_text.isdigit():
        raise ValueError(
            "Unsupported interval. Use minute, hour, day, or week "
            "intervals, such as 5m, 1h, 1d, or 1w."
        )

    amount = int(amount_text)

    if amount <= 0:
        raise ValueError("Interval amount must be positive.")

    return amount * units[unit]

#==================================================================

def validate_extraction_inputs(
    symbol, interval, start_time, end_time, limit
):
    """Validate extraction parameters before making API requests."""

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string.")

    if symbol != symbol.strip() or not symbol.isalnum():
        raise ValueError("Symbol must contain only letters and digits.")

    interval_to_ms(interval)

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("Limit must be an integer.")

    if not 1 <= limit <= DEFAULT_LIMIT:
        raise ValueError(f"Limit must be between 1 and {DEFAULT_LIMIT}.")

    start_ms = datetime_to_ms(start_time)
    end_ms = datetime_to_ms(end_time)

    if start_ms is not None and end_ms is not None:
        if start_ms >= end_ms:
            raise ValueError("start_time must be earlier than end_time.")

    return start_ms, end_ms


# ============================================================
#             FETCH A SINGLE PAGE FROM BINANCE
# ============================================================

def fetch_klines(symbol="BTCUSDT",interval="5m",start_time=None,end_time=None,limit=DEFAULT_LIMIT,session=None,):

    start_ms, end_ms = validate_extraction_inputs(symbol, interval, start_time, end_time, limit)

    params = {
        "symbol"  : symbol.upper(),
        "interval": interval,
        "limit"   : limit,
    }

    if start_ms is not None: params["startTime"] = start_ms
    if end_ms   is not None: params["endTime"]   = end_ms

    client = session if session is not None else requests

    response = client.get(BASE_URL, params=params, timeout=30,)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list): raise RuntimeError(f"Unexpected Binance response: {data}")
    if not data:                   return pd.DataFrame(columns=BINANCE_COLUMNS)

    if any(not isinstance(row, list) or len(row) != 12 for row in data):
        raise RuntimeError("Binance returned an unexpected candle format.")

    df = pd.DataFrame(data, columns=BINANCE_COLUMNS)

    # Convert timestamps to timezone-aware UTC datetimes
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="raise"),unit="ms",utc=True,)

    # Convert market measurements to numeric values
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


# ============================================================
#                    FETCH HISTORICAL DATA
# ============================================================

def fetch_historical_data(symbol="BTCUSDT",interval="5m",days=180,start_time=None,end_time=None,limit=DEFAULT_LIMIT,pause_seconds=0.1,):
    """Fetch historical candles over a requested time range.

    Provide either:
      - days: how many days to retrieve, or
      - start_time and optionally end_time: explicit date range.

    If end_time is omitted, the current UTC time is used.
    """

    if not isinstance(days, (int, float)) or isinstance(days, bool): raise ValueError("days must be a positive number.")
    if days <= 0:                                                    raise ValueError("days must be positive.")
    if pause_seconds < 0:                                            raise ValueError("pause_seconds cannot be negative.")

    interval_ms = interval_to_ms(interval)
    end_ms      = datetime_to_ms(end_time)

    if end_ms   is None: end_ms = datetime_to_ms(datetime.now(timezone.utc))
    
    start_ms = datetime_to_ms(start_time)
    if start_ms is None:   start_ms = end_ms - int(days * 86_400_000)
    if start_ms >= end_ms: raise ValueError("start_time must be earlier than end_time.")

    # Validate the remaining inputs before starting pagination
    validate_extraction_inputs(symbol, interval, start_ms, end_ms, limit)

    pages = []
    cursor_ms = start_ms

    # Paginate until the requested time range is exhausted
    while cursor_ms < end_ms:
        page = fetch_klines(symbol=symbol,interval=interval,start_time=cursor_ms,end_time=end_ms,limit=limit,)

        if page.empty: break

        pages.append(page)

        last_open_ms = datetime_to_ms(page["open_time"].iloc[-1])

        # if the API returned no forward progress we stop
        next_cursor_ms = last_open_ms + interval_ms

        if next_cursor_ms <= cursor_ms: raise RuntimeError("Pagination made no forward progress.")

        cursor_ms = next_cursor_ms

        if pause_seconds: time.sleep(pause_seconds)

    if not pages: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.concat(pages, ignore_index=True)

    # Remove any overlap between pages and keep chronological order
    df = df.drop_duplicates(subset="open_time", keep="last")
    df = df.sort_values("open_time").reset_index(drop=True)

    # Standardize output columns
    df = df[OUTPUT_COLUMNS]

    # Exclude candles whose opening time is outside the requested range
    df = df[
        (df["open_time"] >= pd.to_datetime(start_ms, unit="ms", utc=True))
        & (df["open_time"] < pd.to_datetime( end_ms, unit="ms", utc=True))
    ].reset_index(drop=True)

    return df


# ============================================================
#                STANDARDIZE AND FINALIZE DATA
# ============================================================

def standardize_data(df):
    """Return data with the standard schema and chronological order."""

    if not isinstance(df, pd.DataFrame): raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing_columns: raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = df[OUTPUT_COLUMNS].copy()
    result["open_time"] = pd.to_datetime(result["open_time"],utc=True,errors="raise",)

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column],errors="raise",)

    result = (result.drop_duplicates(subset="open_time", keep="last").sort_values("open_time").reset_index(drop=True))
    return result


# ============================================================
#                      SAVE RAW DATA
# ============================================================

def save_raw_data(df, output_path=None, overwrite=False):


    if output_path is None:
        output_path = (Path(__file__).resolve().parents[1]/"data"/"raw"/"BTCUSDT_5m.csv")
    else:
        output_path = Path(output_path)

    output_path = output_path.resolve()

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Choose another path or pass overwrite=True.")

    clean_df = standardize_data(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"Saved {len(clean_df):,} candles to: {output_path}")

    return output_path
