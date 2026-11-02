
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# Configuration
BASE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
DAYS_BACK = 180
LIMIT = 1000
CANDLE_DURATION = pd.Timedelta(minutes=5)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "BTCUSDT_5m.csv"
)

# Binance's exact kline response order.
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Columns retained in the raw dataset.
KEEP_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def download_candles() -> pd.DataFrame:
    """Download and validate completed Binance candles."""

    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - DAYS_BACK * 24 * 60 * 60 * 1000
    next_start = start_time
    all_rows = []

    with requests.Session() as session:
        while next_start < end_time:
            params = {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": next_start,
                "endTime": end_time,
                "limit": LIMIT,
            }

            for attempt in range(5):
                try:
                    response = session.get(
                        BASE_URL,
                        params=params,
                        timeout=30,
                    )
                    response.raise_for_status()
                    batch = response.json()

                    if not isinstance(batch, list):
                        raise ValueError(
                            f"Unexpected Binance response: {batch}"
                        )
                    break

                except (requests.RequestException, ValueError):
                    if attempt == 4:
                        raise

                    wait_seconds = 2 ** attempt
                    print(
                        f"Request failed. Retrying in "
                        f"{wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)

            if not batch:
                break

            all_rows.extend(batch)

            last_open_time = batch[-1][0]
            next_start = last_open_time + 1

            print(f"Downloaded {len(all_rows):,} candles...")
            time.sleep(0.1)

    if not all_rows:
        raise RuntimeError("Binance returned no candle data.")

    df = pd.DataFrame(all_rows, columns=COLUMNS)

    # Remove duplicates and sort chronologically.
    df = (
        df.drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # Exclude candles that have not yet closed.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = df[df["close_time"] < now_ms].copy()

    if df.empty:
        raise RuntimeError("No completed candles were returned.")

    # Convert timestamps to UTC datetimes.
    df["open_time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    # Convert market measurements to numeric types.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype("int64")

    # Validate candle price ranges and nonnegative activity values.
    highest_expected = df[["open", "close"]].max(axis=1)
    lowest_expected = df[["open", "close"]].min(axis=1)

    invalid = (
        (df["high"] < highest_expected)
        | (df["low"] > lowest_expected)
        | (df["high"] < df["low"])
        | (df[numeric_columns] < 0).any(axis=1)
        | (df["number_of_trades"] < 0)
        | (df["taker_buy_base_volume"] > df["volume"])
        | (df["taker_buy_quote_volume"] > df["quote_volume"])
    )

    if invalid.any():
        raise ValueError(
            f"Found {invalid.sum()} invalid candles. "
            "Inspect the downloaded data before proceeding."
        )

    # Keep useful raw market fields; discard close_time and ignore.
    return df[KEEP_COLUMNS].reset_index(drop=True)


def main() -> None:
    """Download, inspect, and save the dataset."""

    print(f"Downloading {SYMBOL} {INTERVAL} candles...")
    df = download_candles()

    # Detect missing five-minute intervals.
    time_differences = df["open_time"].diff()
    gaps = int((time_differences > CANDLE_DURATION).sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDownload complete.")
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"First candle: {df['open_time'].iloc[0]}")
    print(f"Last candle:  {df['open_time'].iloc[-1]}")
    print(f"Missing five-minute intervals detected: {gaps}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
