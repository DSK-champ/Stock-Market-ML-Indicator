"""Candle storage, validation, REST backfill, and reconciliation engine."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from backend.live import config


class CandleStore:
    """Manages SQLite candle persistence, historical backfilling, and REST reconciliation."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or config.CANDLES_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    open_time TEXT PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_volume REAL NOT NULL,
                    number_of_trades REAL NOT NULL,
                    taker_buy_base_volume REAL NOT NULL,
                    taker_buy_quote_volume REAL NOT NULL,
                    close_time TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    is_reconciled INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_close_time ON candles(close_time)")
            conn.commit()

    def validate_candle(self, c: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate candle numerical soundness and temporal integrity."""
        try:
            o, h, l, cl, v = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]), float(c["volume"])
            for val in [o, h, l, cl, v]:
                if np.isnan(val) or np.isinf(val):
                    return False, f"Non-finite value detected: {val}"

            if h < max(o, cl):
                return False, f"Invalid high {h} < max(open={o}, close={cl})"
            if l > min(o, cl):
                return False, f"Invalid low {l} > min(open={o}, close={cl})"
            if v < 0:
                return False, f"Negative volume: {v}"

            # Validate 5-minute alignment if epoch timestamp provided
            if "open_time_ms" in c:
                if c["open_time_ms"] % config.INTERVAL_MS != 0:
                    return False, f"open_time {c['open_time_ms']} not aligned to 5-minute boundary"

            return True, None
        except Exception as e:
            return False, f"Validation exception: {str(e)}"

    def store_candle(self, candle: Dict[str, Any], source: str = "WS") -> bool:
        """Store or update a validated closed candle in SQLite.

        Returns True if inserted/updated successfully, False if rejected.
        """
        is_valid, reason = self.validate_candle(candle)
        if not is_valid:
            print(f"[ERROR] Candle rejected: {reason} | Payload: {candle}")
            return False

        now_str = datetime.now(timezone.utc).isoformat()
        open_time = candle["open_time"]

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO candles (
                    open_time, open, high, low, close, volume,
                    quote_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                    close_time, received_at, source, is_reconciled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                open_time,
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
                float(candle["volume"]),
                float(candle.get("quote_volume", candle.get("quote_asset_volume", 0.0))),
                float(candle.get("number_of_trades", 0.0)),
                float(candle.get("taker_buy_base_volume", 0.0)),
                float(candle.get("taker_buy_quote_volume", candle.get("taker_buy_quote_asset_volume", 0.0))),
                str(candle["close_time"]),
                now_str,
                source,
                1 if source == "REST" else 0,
            ))
            conn.commit()
        return True

    def get_candle(self, open_time: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single candle by open_time."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM candles WHERE open_time = ?", (open_time,)).fetchone()
            return dict(row) if row else None

    def get_latest_open_time(self) -> Optional[datetime]:
        """Get the latest recorded open_time."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT MAX(open_time) as max_time FROM candles").fetchone()
            if row and row["max_time"]:
                return pd.to_datetime(row["max_time"], utc=True)
            return None

    def get_recent_candles(self, limit: int = config.BUFFER_SIZE) -> pd.DataFrame:
        """Retrieve recent candles formatted as a model-ready pandas DataFrame."""
        with self._get_connection() as conn:
            query = f"SELECT * FROM candles ORDER BY open_time DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return pd.DataFrame()

        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.sort_values("open_time").set_index("open_time")
        return df

    def backfill_from_rest(self, limit: int = config.BUFFER_SIZE) -> int:
        """Fetch historical closed candles from Binance Public REST API and store."""
        print(f"[INFO] Backfilling last {limit} candles from Binance REST API...")
        url = f"{config.REST_BASE_URL}/klines"
        params = {
            "symbol": config.SYMBOL,
            "interval": config.INTERVAL,
            "limit": min(limit + 1, 1000),  # Fetch extra to discard current unclosed candle
        }
        res = requests.get(url, params=params, timeout=15).json()

        if not isinstance(res, list):
            raise RuntimeError(f"Unexpected Binance REST response: {res}")

        now_ms = int(time.time() * 1000)
        stored_count = 0

        for k in res:
            open_ms = int(k[0])
            close_ms = int(k[6])

            # Drop candle if still forming (close time is in the future)
            if close_ms > now_ms:
                continue

            open_time_dt = pd.to_datetime(open_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00")
            close_time_dt = pd.to_datetime(close_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00")

            candle_dict = {
                "open_time": open_time_dt,
                "open_time_ms": open_ms,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
                "number_of_trades": float(k[8]),
                "taker_buy_base_volume": float(k[9]),
                "taker_buy_quote_volume": float(k[10]),
                "close_time": close_time_dt,
            }

            if self.store_candle(candle_dict, source="REST"):
                stored_count += 1

        print(f"[OK] Backfilled {stored_count} closed candles successfully.")
        return stored_count

    def reconcile_candle(self, open_time_iso: str) -> bool:
        """Reconcile WebSocket candle against Binance REST truth 3 seconds after close."""
        try:
            open_dt = pd.to_datetime(open_time_iso, utc=True)
            open_ms = int(open_dt.timestamp() * 1000)

            url = f"{config.REST_BASE_URL}/klines"
            params = {
                "symbol": config.SYMBOL,
                "interval": config.INTERVAL,
                "startTime": open_ms,
                "limit": 1,
            }
            res = requests.get(url, params=params, timeout=5).json()
            if not isinstance(res, list) or len(res) == 0:
                return False

            k = res[0]
            if int(k[0]) != open_ms:
                return False

            existing = self.get_candle(open_time_iso)
            if not existing:
                return False

            rest_close = float(k[4])
            ws_close = float(existing["close"])

            if abs(rest_close - ws_close) > 1e-4:
                print(f"[RECONCILED] Mismatch at {open_time_iso}: WS close={ws_close} vs REST close={rest_close}. Overwriting.")
                with self._get_connection() as conn:
                    conn.execute("""
                        UPDATE candles SET
                            open = ?, high = ?, low = ?, close = ?, volume = ?,
                            quote_volume = ?, number_of_trades = ?, taker_buy_base_volume = ?,
                            taker_buy_quote_volume = ?, source = 'REST_RECONCILED', is_reconciled = 1
                        WHERE open_time = ?
                    """, (
                        float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
                        float(k[7]), float(k[8]), float(k[9]), float(k[10]), open_time_iso
                    ))
                    conn.commit()
            else:
                with self._get_connection() as conn:
                    conn.execute("UPDATE candles SET is_reconciled = 1 WHERE open_time = ?", (open_time_iso,))
                    conn.commit()
            return True
        except Exception as e:
            print(f"[WARN] Reconciliation failed for {open_time_iso}: {e}")
            return False

