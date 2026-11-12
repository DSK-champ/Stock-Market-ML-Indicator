"""Sync missing market candles from Binance API into clean.csv and master.csv.

Extracts missing 5m klines from the end of clean.csv / master.csv up to current
completed candle, calculates engineered features, and safely appends them.
Also synchronizes candles into backend/data/candles.db.
"""

import sys
import time
import urllib.request
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = BASE_DIR / "ML"
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(BASE_DIR))

from src import config as ml_config
from src.features import build_features
from backend.live import config as live_config

CLEAN_CSV_PATH = ML_DIR / "data" / "processed" / "clean.csv"
MASTER_CSV_PATH = ML_DIR / "data" / "processed" / "master.csv"
CANDLES_DB_PATH = BASE_DIR / "backend" / "data" / "candles.db"


def fetch_binance_klines_gap(start_time_ms: int, end_time_ms: int) -> list:
    """Fetch 5m klines from Binance Spot API paginating by 1000 candles."""
    all_klines = []
    curr_start = start_time_ms

    while curr_start <= end_time_ms:
        url = (
            f"https://api.binance.com/api/v3/klines?"
            f"symbol={live_config.SYMBOL}&interval={live_config.INTERVAL}&"
            f"startTime={curr_start}&endTime={end_time_ms}&limit=1000"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if not data:
            break

        all_klines.extend(data)
        last_open_time = data[-1][0]
        curr_start = last_open_time + 5 * 60 * 1000

        if len(data) < 1000:
            break

        time.sleep(0.2)

    return all_klines


def sync_missing_candles():
    print("=" * 80)
    print("SYNCING HISTORICAL MARKET DATA FROM BINANCE API")
    print("=" * 80)

    # 1. Load clean.csv and master.csv to find status
    print("1. Checking clean.csv and master.csv state...")
    df_clean = pd.read_csv(CLEAN_CSV_PATH)
    clean_last_dt = pd.to_datetime(df_clean.iloc[-1]["open_time"], utc=True)
    clean_last_gap_id = int(df_clean.iloc[-1].get("gap_id", 0))

    master_tail = pd.read_csv(MASTER_CSV_PATH, usecols=["open_time"]).tail(1)
    master_last_dt = pd.to_datetime(master_tail.iloc[-1]["open_time"], utc=True)

    print(f"   clean.csv rows:  {len(df_clean):,} | Last: {clean_last_dt}")
    print(f"   master.csv last: {master_last_dt}")

    now_utc = datetime.now(timezone.utc)
    # The latest fully closed 5m candle
    latest_closed_sec = (int(now_utc.timestamp()) // 300) * 300
    latest_closed_ms = (latest_closed_sec - 300) * 1000

    start_fetch_ms = int(clean_last_dt.timestamp() * 1000) + 5 * 60 * 1000

    # 2. Check if clean.csv needs newer Binance candles
    if start_fetch_ms <= latest_closed_ms:
        print(f"\n2. Fetching newer Binance klines from {pd.to_datetime(start_fetch_ms, unit='ms', utc=True)} to {pd.to_datetime(latest_closed_ms, unit='ms', utc=True)}...")
        raw_klines = fetch_binance_klines_gap(start_fetch_ms, latest_closed_ms)
        print(f"   Retrieved {len(raw_klines)} newer candles from Binance API.")

        if raw_klines:
            new_clean_rows = []
            curr_gap_id = clean_last_gap_id
            prev_ts_ms = int(clean_last_dt.timestamp() * 1000)

            for k in raw_klines:
                open_time_ms = int(k[0])
                is_gap = (open_time_ms - prev_ts_ms) > (5 * 60 * 1000 + 1000)
                if is_gap:
                    curr_gap_id += 1

                open_time_dt = pd.to_datetime(open_time_ms, unit="ms", utc=True)
                open_time_fmt = open_time_dt.strftime("%Y-%m-%d %H:%M:%S+0000")

                row = {
                    "open_time": open_time_fmt,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "number_of_trades": float(k[8]),
                    "taker_buy_base_volume": float(k[9]),
                    "taker_buy_quote_volume": float(k[10]),
                    "gap_id": curr_gap_id,
                    "is_gap_boundary": is_gap,
                }
                new_clean_rows.append(row)
                prev_ts_ms = open_time_ms

            df_new_clean = pd.DataFrame(new_clean_rows)
            df_new_clean.to_csv(CLEAN_CSV_PATH, mode="a", header=False, index=False)
            print(f"   [DONE] Appended {len(df_new_clean)} rows to clean.csv.")
            df_clean = pd.read_csv(CLEAN_CSV_PATH)
    else:
        print("\n2. clean.csv is already up to date with Binance API.")

    # 3. Identify rows in clean.csv that need to be appended to master.csv
    df_clean["dt"] = pd.to_datetime(df_clean["open_time"], utc=True)
    missing_for_master = df_clean[df_clean["dt"] > master_last_dt].copy()
    print(f"\n3. Found {len(missing_for_master)} rows in clean.csv to append to master.csv.")

    if len(missing_for_master) == 0:
        print("[OK] master.csv is already fully synced with clean.csv.")
    else:
        # 4. Compute 136 features on clean with rolling warm-up
        print("4. Computing 136 engineered causal features for missing rows...")
        warmup_count = 1000
        full_slice = df_clean.tail(warmup_count + len(missing_for_master)).copy()
        full_slice = full_slice.drop(columns=["dt"])
        full_slice["open_time"] = pd.to_datetime(full_slice["open_time"], utc=True)
        full_slice = full_slice.set_index("open_time").sort_index()

        features_full = build_features(full_slice)
        new_features = features_full.tail(len(missing_for_master)).copy()

        # 5. Build master rows
        print("5. Aligning columns with master.csv schema...")
        master_header = pd.read_csv(MASTER_CSV_PATH, nrows=1)
        master_cols = list(master_header.columns)

        missing_indexed = missing_for_master.drop(columns=["dt"]).copy()
        missing_indexed["open_time_dt"] = pd.to_datetime(missing_indexed["open_time"], utc=True)
        missing_indexed = missing_indexed.set_index("open_time_dt")

        combined = pd.concat([missing_indexed, new_features], axis=1)

        # Reset index and maintain open_time column format
        combined = combined.reset_index(drop=True)

        # Populate forward returns if outcome candles exist
        horizons = [2, 3, 6, 12]
        for h in horizons:
            combined[f"fwd_ret_{h}"] = (combined["close"].shift(-h) / combined["close"]) - 1.0
            combined[f"raw_sign_{h}"] = np.sign(combined[f"fwd_ret_{h}"])

        # Populate any missing master columns with NaN
        for col in master_cols:
            if col not in combined.columns:
                combined[col] = np.nan

        # Reorder columns
        master_append_df = combined[master_cols]
        master_append_df.to_csv(MASTER_CSV_PATH, mode="a", header=False, index=False)
        print(f"   [DONE] master.csv appended {len(master_append_df)} rows successfully.")

    # 6. Store latest candles into backend/data/candles.db
    print("\n6. Syncing into backend/data/candles.db...")
    if CANDLES_DB_PATH.exists():
        conn = sqlite3.connect(CANDLES_DB_PATH)
        recent_sync = df_clean.tail(1000)
        now_iso = datetime.now(timezone.utc).isoformat()
        for _, row in recent_sync.iterrows():
            ts_str = str(row["open_time"]).replace("+0000", "+00:00")
            close_time_dt = pd.to_datetime(ts_str) + pd.Timedelta(minutes=5)
            close_time_str = close_time_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
            conn.execute(
                """
                INSERT OR REPLACE INTO candles (
                    open_time, open, high, low, close, volume,
                    quote_volume, number_of_trades, taker_buy_base_volume,
                    taker_buy_quote_volume, close_time, received_at, source, is_reconciled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_str,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    float(row["quote_volume"]),
                    float(row["number_of_trades"]),
                    float(row["taker_buy_base_volume"]),
                    float(row["taker_buy_quote_volume"]),
                    close_time_str,
                    now_iso,
                    "BINANCE_REST_GAP_SYNC",
                    1,
                ),
            )
        conn.commit()
        conn.close()
        print(f"   [DONE] Synchronized recent candles into candles.db.")

    final_clean = pd.read_csv(CLEAN_CSV_PATH).tail(1)
    last_time = final_clean.iloc[-1]["open_time"]
    last_price = float(final_clean.iloc[-1]["close"])

    print("\n" + "=" * 80)
    print(f"SYNC COMPLETE: Total candles in clean & master: {len(df_clean):,}")
    print(f"Latest Recorded Candle Time:  {last_time}")
    print(f"Latest BTC/USDT Close Price:   ${last_price:,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    sync_missing_candles()
