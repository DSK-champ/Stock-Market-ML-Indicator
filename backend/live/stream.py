"""Binance WebSocket client for real-time 5m kline streaming."""

from typing import Dict, Any, Optional, List
import asyncio
import json
import time
import random
import pandas as pd
import websockets
from datetime import datetime, timezone

from backend.live import config
from backend.live.candle_store import CandleStore
from backend.live.predictor import PredictorWorker


class BinanceKlineStream:
    """Robust, production-grade Binance 5m kline WebSocket consumer."""

    def __init__(
        self,
        candle_store: CandleStore,
        predictor: PredictorWorker,
        endpoints: Optional[List[str]] = None,
    ) -> None:
        self.candle_store = candle_store
        self.predictor = predictor
        self.endpoints = endpoints or config.WS_ENDPOINTS
        self.current_endpoint_idx = 0

        self.state = "DOWN"  # CONNECTING / LIVE / RECONNECTING / DOWN
        self.last_message_time: Optional[float] = None
        self.last_closed_candle_time: Optional[str] = None
        self.reconnect_count = 0
        self.measured_latency_ms = 0.0
        self.forming_candle: Optional[Dict[str, Any]] = None

        self._running = False
        self._last_processed_open_ms: Optional[int] = None
        self._main_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start streaming loop."""
        if self._running:
            return
        self._running = True
        self._main_task = asyncio.create_task(self._run_loop())
        print(f"[OK] BinanceKlineStream started. Primary endpoint: {self.endpoints[0]}")

    async def stop(self) -> None:
        """Gracefully stop streaming."""
        self._running = False
        self.state = "DOWN"
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        if self._watchdog_task:
            self._watchdog_task.cancel()
        print("[OK] BinanceKlineStream stopped.")

    async def _run_loop(self) -> None:
        """Main connection and auto-recovery loop with backoff and failover."""
        backoff = config.INITIAL_RECONNECT_BACKOFF

        while self._running:
            endpoint = self.endpoints[self.current_endpoint_idx]
            self.state = "CONNECTING"
            conn_start_time = time.time()
            print(f"[WS] Connecting to {endpoint} (Attempt #{self.reconnect_count + 1})...")

            try:
                async with websockets.connect(
                    endpoint,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self.state = "LIVE"
                    self.last_message_time = time.time()
                    backoff = config.INITIAL_RECONNECT_BACKOFF
                    print(f"[WS] Connected successfully to {endpoint} (LIVE).")

                    # Start watchdog
                    self._watchdog_task = asyncio.create_task(self._watchdog_loop())

                    while self._running:
                        # Check proactive 23-hour rotation limit
                        if time.time() - conn_start_time >= config.PROACTIVE_RECONNECT_SECONDS:
                            print("[WS] 23-hour connection threshold reached. Proactively reconnecting at candle boundary...")
                            break

                        try:
                            msg_str = await asyncio.wait_for(ws.recv(), timeout=config.WATCHDOG_TIMEOUT_SECONDS)
                        except asyncio.TimeoutError:
                            print("[WARN] Watchdog timeout: no frame received in 15 seconds. Reconnecting...")
                            break

                        self.last_message_time = time.time()
                        msg = json.loads(msg_str)

                        # Check serverShutdown
                        if msg.get("e") == "serverShutdown":
                            print("[WARN] Server sent serverShutdown event. Reconnecting immediately...")
                            break

                        if msg.get("e") == "kline":
                            await self._handle_kline_event(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WARN] Connection dropped: {e}")
            finally:
                if self._watchdog_task:
                    self._watchdog_task.cancel()

            if not self._running:
                break

            self.state = "RECONNECTING"
            self.reconnect_count += 1

            # Failover endpoint
            self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.endpoints)

            # Jittered exponential backoff
            jitter = random.uniform(0.1, 0.5)
            sleep_time = min(backoff + jitter, config.MAX_RECONNECT_BACKOFF)
            print(f"[WS] Reconnecting in {sleep_time:.2f}s (backoff={backoff:.1f}s)...")
            await asyncio.sleep(sleep_time)
            backoff = min(backoff * 2.0, config.MAX_RECONNECT_BACKOFF)

            # Backfill any missed candles after reconnection
            try:
                self.candle_store.backfill_from_rest(limit=10)
            except Exception as e:
                print(f"[WARN] Post-reconnect backfill failed: {e}")

    async def _handle_kline_event(self, msg: Dict[str, Any]) -> None:
        """Parse, validate, and process incoming kline event."""
        recv_time = time.time()
        event_time_ms = int(msg.get("E", 0))

        # Clock drift detection
        drift_sec = abs(recv_time - (event_time_ms / 1000.0))
        if drift_sec > config.CLOCK_DRIFT_WARN_SECONDS:
            print(f"[WARN] Clock drift detected: {drift_sec:.2f}s difference between local clock and Binance event time.")

        k = msg.get("k", {})
        open_ms = int(k.get("t", 0))
        close_ms = int(k.get("T", 0))
        is_closed = bool(k.get("x", False))

        # Always update current forming candle view
        self.forming_candle = {
            "open_time_ms": open_ms,
            "close": float(k.get("c", 0.0)),
            "high": float(k.get("h", 0.0)),
            "low": float(k.get("l", 0.0)),
            "volume": float(k.get("v", 0.0)),
            "is_closed": is_closed,
        }

        # Act ONLY when k.x is True (closed candle)
        if not is_closed:
            return

        # De-duplicate by candle open time
        if self._last_processed_open_ms == open_ms:
            return
        self._last_processed_open_ms = open_ms

        open_time_dt = pd.to_datetime(open_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00")
        close_time_dt = pd.to_datetime(close_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00")

        # Measured latency: local receive time minus candle close timestamp
        self.measured_latency_ms = max(0.0, (recv_time * 1000.0) - close_ms)
        self.last_closed_candle_time = open_time_dt

        candle_payload = {
            "open_time": open_time_dt,
            "open_time_ms": open_ms,
            "open": float(k.get("o", 0.0)),
            "high": float(k.get("h", 0.0)),
            "low": float(k.get("l", 0.0)),
            "close": float(k.get("c", 0.0)),
            "volume": float(k.get("v", 0.0)),
            "quote_volume": float(k.get("q", 0.0)),
            "number_of_trades": float(k.get("n", 0.0)),
            "taker_buy_base_volume": float(k.get("V", 0.0)),
            "taker_buy_quote_volume": float(k.get("Q", 0.0)),
            "close_time": close_time_dt,
            "exchange_event_time_ms": event_time_ms,
            "exchange_close_time": close_time_dt,
        }

        # Store in SQLite
        if not self.candle_store.store_candle(candle_payload, source="WS"):
            print(f"[WARN] Candle {open_time_dt} failed validation.")
            return

        print(f"[CLOSED] Candle {open_time_dt} | Close: {candle_payload['close']:.2f} | Latency: {self.measured_latency_ms:.1f}ms")

        # Schedule REST reconciliation after 3 seconds
        asyncio.create_task(self._schedule_reconciliation(open_time_dt))

        # Enqueue for model inference
        await self.predictor.enqueue_candle(candle_payload)

    async def _schedule_reconciliation(self, open_time_dt: str) -> None:
        """Wait 3 seconds then reconcile WebSocket candle against REST truth."""
        await asyncio.sleep(config.RECONCILIATION_DELAY_SECONDS)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.candle_store.reconcile_candle, open_time_dt)

    async def _watchdog_loop(self) -> None:
        """Watchdog checking for quiet connection."""
        while self._running:
            await asyncio.sleep(5)
            if self.last_message_time and (time.time() - self.last_message_time > config.WATCHDOG_TIMEOUT_SECONDS):
                print("[WARN] Watchdog alert: no frame received in >15s. Forcing reconnect...")
                break

