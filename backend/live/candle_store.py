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
        """Fetch historical closed candles from Binance Public REST API with pagination."""
        print(f"[INFO] Backfilling last {limit} candles from Binance REST API (paginated)...")
        url = f"{config.REST_BASE_URL}/klines"
        all_candles: List[Dict[str, Any]] = []
        now_ms = int(time.time() * 1000)
        end_time_ms: Optional[int] = None

        while len(all_candles) < limit:
            chunk_size = min(1000, limit - len(all_candles) + 5)
            params: Dict[str, Any] = {
                "symbol": config.SYMBOL,
                "interval": config.INTERVAL,
                "limit": chunk_size,
            }
            if end_time_ms is not None:
                params["endTime"] = end_time_ms

            try:
                res = requests.get(url, params=params, timeout=15).json()
            except Exception as e:
                print(f"[WARN] REST backfill request failed: {e}")
                break

            if not isinstance(res, list) or len(res) == 0:
                break

            chunk_candles = []
            for k in res:
                open_ms = int(k[0])
                close_ms = int(k[6])
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
                chunk_candles.append(candle_dict)

            if not chunk_candles:
                break

            oldest_open_ms = min(c["open_time_ms"] for c in chunk_candles)
            end_time_ms = oldest_open_ms - 1
            all_candles = chunk_candles + all_candles
            if len(chunk_candles) < 2:
                break

        # Dedup and take tail limit
        seen = set()
        unique_candles = []
        for c in sorted(all_candles, key=lambda x: x["open_time_ms"]):
            if c["open_time"] not in seen:
                seen.add(c["open_time"])
                unique_candles.append(c)

        if len(unique_candles) > limit:
            unique_candles = unique_candles[-limit:]

        # Fast bulk insert inside a single SQLite transaction
        stored_count = 0
        with self._get_connection() as conn:
            for c in unique_candles:
                try:
                    conn.execute("""
                        INSERT INTO candles (
                            open_time, open, high, low, close, volume,
                            quote_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                            close_time, received_at, source, is_reconciled
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(open_time) DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close,
                            volume = excluded.volume,
                            quote_volume = excluded.quote_volume,
                            number_of_trades = excluded.number_of_trades,
                            taker_buy_base_volume = excluded.taker_buy_base_volume,
                            taker_buy_quote_volume = excluded.taker_buy_quote_volume,
                            close_time = excluded.close_time,
                            is_reconciled = 1
                    """, (
                        c["open_time"], c["open"], c["high"], c["low"], c["close"], c["volume"],
                        c["quote_volume"], c["number_of_trades"], c["taker_buy_base_volume"], c["taker_buy_quote_volume"],
                        c["close_time"], datetime.now(timezone.utc).isoformat(), "REST"
                    ))
                    stored_count += 1
                except Exception as e:
                    print(f"[WARN] Error inserting candle {c['open_time']}: {e}")
            conn.commit()

        print(f"[OK] Backfilled {stored_count} closed candles successfully with pagination.")
        return stored_count

    def fetch_binance_range(self, start_ms: int, end_ms: Optional[int] = None, max_candles: int = 1000) -> List[Dict[str, Any]]:
        """Fetch closed klines from Binance Public REST API within a specific timestamp window."""
        url = f"{config.REST_BASE_URL}/klines"
        all_candles: List[Dict[str, Any]] = []
        curr_start = start_ms
        now_ms = int(time.time() * 1000)

        while True:
            params: Dict[str, Any] = {
                "symbol": config.SYMBOL,
                "interval": config.INTERVAL,
                "startTime": curr_start,
                "limit": min(1000, max_candles - len(all_candles)),
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            try:
                res = requests.get(url, params=params, timeout=10).json()
            except Exception as e:
                print(f"[WARN] Binance REST range fetch error: {e}")
                break

            if not isinstance(res, list) or len(res) == 0:
                break

            page_added = 0
            for k in res:
                open_ms = int(k[0])
                close_ms = int(k[6])
                if close_ms > now_ms:
                    continue
                o_dt = pd.to_datetime(open_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00")
                c_dt = pd.to_datetime(close_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00")
                all_candles.append({
                    "open_time": o_dt,
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
                    "close_time": c_dt,
                })
                page_added += 1

            if page_added == 0 or len(all_candles) >= max_candles or len(res) < 1000:
                break

            latest_open_ms = max(int(k[0]) for k in res)
            curr_start = latest_open_ms + config.INTERVAL_MS
            if end_ms is not None and curr_start > end_ms:
                break

        return all_candles

    def validate_and_repair_continuity(self, limit: int = config.BUFFER_SIZE) -> Dict[str, Any]:
        """Audit SQLite candles for skipped intervals and repair every gap from Binance."""
        with self._get_connection() as conn:
            query = "SELECT open_time, open, high, low, close FROM candles ORDER BY open_time ASC"
            df = pd.read_sql_query(query, conn)

        if df.empty:
            count = self.backfill_from_rest(limit=limit)
            return {"gaps_found": 0, "candles_repaired": count, "is_continuous": True}

        df["dt"] = pd.to_datetime(df["open_time"], utc=True)
        df["diff"] = df["dt"].diff()
        # Any step greater than 5 minutes (300 seconds) indicates skipped candles!
        gaps = df[df["diff"] > pd.Timedelta(minutes=5)]
        
        repaired_count = 0
        now_ms = int(time.time() * 1000)

        # 1. Repair internal gaps
        if not gaps.empty:
            print(f"[CANDLE AUDIT] Detected {len(gaps)} continuity break(s) in candles.db. Repairing from Binance...")
            for idx, row in gaps.iterrows():
                prev_dt = df.loc[idx - 1, "dt"]
                curr_dt = row["dt"]
                start_ms = int((prev_dt + pd.Timedelta(minutes=5)).timestamp() * 1000)
                end_ms = int((curr_dt - pd.Timedelta(seconds=1)).timestamp() * 1000)
                missing_est = int((curr_dt - prev_dt).total_seconds() / 300) - 1

                print(f"[CANDLE AUDIT] Filling break: {missing_est} skipped candles between {prev_dt} and {curr_dt}...")
                gap_candles = self.fetch_binance_range(start_ms=start_ms, end_ms=end_ms, max_candles=missing_est + 10)
                
                with self._get_connection() as conn:
                    for c in gap_candles:
                        conn.execute("""
                            INSERT OR REPLACE INTO candles (
                                open_time, open, high, low, close, volume,
                                quote_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                                close_time, received_at, source, is_reconciled
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """, (
                            c["open_time"], c["open"], c["high"], c["low"], c["close"], c["volume"],
                            c["quote_volume"], c["number_of_trades"], c["taker_buy_base_volume"], c["taker_buy_quote_volume"],
                            c["close_time"], datetime.now(timezone.utc).isoformat(), "REST_GAPFILL"
                        ))
                        repaired_count += 1
                    conn.commit()

        # 2. Repair tail gap up to current moment
        max_dt = df["dt"].max()
        tail_start_ms = int((max_dt + pd.Timedelta(minutes=5)).timestamp() * 1000)
        if now_ms - tail_start_ms >= 300000:
            tail_missing = int((now_ms - tail_start_ms) / 300000)
            print(f"[CANDLE AUDIT] Tail data is {tail_missing} candles behind current time. Fetching latest from Binance...")
            tail_candles = self.fetch_binance_range(start_ms=tail_start_ms, max_candles=tail_missing + 10)
            with self._get_connection() as conn:
                for c in tail_candles:
                    conn.execute("""
                        INSERT OR REPLACE INTO candles (
                            open_time, open, high, low, close, volume,
                            quote_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                            close_time, received_at, source, is_reconciled
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (
                        c["open_time"], c["open"], c["high"], c["low"], c["close"], c["volume"],
                        c["quote_volume"], c["number_of_trades"], c["taker_buy_base_volume"], c["taker_buy_quote_volume"],
                        c["close_time"], datetime.now(timezone.utc).isoformat(), "REST_TAIL"
                    ))
                    repaired_count += 1
                conn.commit()

        # 3. Post-repair validation assertion
        with self._get_connection() as conn:
            df_check = pd.read_sql_query("SELECT open_time FROM candles ORDER BY open_time ASC", conn)
        df_check["dt"] = pd.to_datetime(df_check["open_time"], utc=True)
        remaining_gaps = len(df_check[df_check["dt"].diff() > pd.Timedelta(minutes=5)])
        is_continuous = (remaining_gaps == 0)

        print(f"[CANDLE AUDIT COMPLETE] Repaired {repaired_count} candles across {len(gaps)} break(s). Total candles: {len(df_check)}. Unbroken continuous: {is_continuous}.")
        return {
            "gaps_found": len(gaps),
            "candles_repaired": repaired_count,
            "total_candles": len(df_check),
            "remaining_gaps": remaining_gaps,
            "is_continuous": is_continuous,
            "latest_open_time": df_check["open_time"].iloc[-1] if not df_check.empty else None,
        }

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

