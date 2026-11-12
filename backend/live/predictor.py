"""Prediction worker executing FIFO inference, thread isolation, and automated resolution."""

from typing import Dict, Any, Optional, Callable, List
import asyncio
import time
import subprocess
import hashlib
import pandas as pd
from datetime import datetime, timezone

from backend.live import config
from backend.live.candle_store import CandleStore
from backend.live.ledger import PredictionLedger
from backend.engine import TradingEngine


def get_git_commit() -> str:
    """Retrieve git HEAD commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()[:10]
    except Exception:
        return "unknown"


def compute_config_hash() -> str:
    """Compute deterministic hash of runtime decision config."""
    cfg_str = f"{config.CONTEXT_WEIGHT}_{config.OVERRIDE_THRESHOLD}_{config.OVERRIDE_MARGIN}_{config.COST_FLOOR_BPS}"
    return hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:8]


class PredictorWorker:
    """Asynchronous prediction worker with FIFO serialization and thread safety."""

    def __init__(
        self,
        candle_store: CandleStore,
        ledger: PredictionLedger,
        buffer_size: int = config.BUFFER_SIZE,
        event_callbacks: Optional[List[Callable[[str, Dict[str, Any]], None]]] = None,
    ) -> None:
        self.candle_store = candle_store
        self.ledger = ledger
        self.buffer_size = buffer_size
        self.engine = TradingEngine.get_instance(
            context_weight=config.CONTEXT_WEIGHT,
            override_threshold=config.OVERRIDE_THRESHOLD,
            override_margin=config.OVERRIDE_MARGIN,
            cost_floor_bps=config.COST_FLOOR_BPS,
        )
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.event_callbacks = event_callbacks or []
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self.git_commit = get_git_commit()
        self.config_hash = compute_config_hash()

    def add_callback(self, cb: Callable[[str, Dict[str, Any]], None]) -> None:
        self.event_callbacks.append(cb)

    def _broadcast(self, event_type: str, payload: Dict[str, Any]) -> None:
        for cb in self.event_callbacks:
            try:
                cb(event_type, payload)
            except Exception as e:
                print(f"[WARN] Error in event callback: {e}")

    async def start(self) -> None:
        """Start the async queue consumer task."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._consume_loop())
        print("[OK] PredictorWorker started.")

    async def stop(self) -> None:
        """Stop worker cleanly."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        print("[OK] PredictorWorker stopped.")

    async def enqueue_candle(self, candle: Dict[str, Any]) -> None:
        """Enqueue closed candle for processing."""
        await self.queue.put(candle)

    async def _consume_loop(self) -> None:
        """FIFO worker loop."""
        while self._running:
            try:
                candle = await self.queue.get()
                await self._process_candle(candle)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] PredictorWorker error: {e}")
                await asyncio.sleep(0.5)

    async def _process_candle(self, candle: Dict[str, Any]) -> None:
        """Process incoming closed candle."""
        open_time = candle["open_time"]
        ref_close = float(candle["close"])

        # Calculate target candle k+2 open_time (+10 minutes)
        candle_dt = pd.to_datetime(open_time, utc=True)
        target_dt = candle_dt + pd.Timedelta(minutes=config.INTERVAL_MINUTES * config.HORIZON)
        target_open_time = target_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

        # 1. Fetch rolling buffer from SQLite
        recent_df = self.candle_store.get_recent_candles(limit=self.buffer_size)
        buffer_len = len(recent_df)

        if buffer_len < config.MIN_BUFFER_SIZE:
            # Short buffer -> NO_SIGNAL
            pred_record = {
                "candle_open_time": open_time,
                "ref_close": ref_close,
                "target_candle_open_time": target_open_time,
                "action": 0,
                "action_label": "NEUTRAL",
                "confidence": 0.0,
                "probabilities": {"p_up": 0.3333, "p_neutral": 0.3334, "p_down": 0.3333},
                "trend": "uncertain",
                "market_phase": "uncertain",
                "model_version": f"TradingEngine-v2.0-N{config.HORIZON}",
                "git_commit": self.git_commit,
                "config_hash": self.config_hash,
                "buffer_length": buffer_len,
                "compute_latency_ms": 0.0,
                "status": "NO_SIGNAL",
                "status_reason": f"Buffer short ({buffer_len} < {config.MIN_BUFFER_SIZE} required)",
            }
            self.ledger.record_prediction(pred_record)
            self._broadcast("new_prediction", pred_record)
            return

        # 2. Check for buffer gap
        diffs = recent_df.index.to_series().diff().dropna()
        has_gap = (diffs > pd.Timedelta(minutes=5)).any() if len(diffs) > 0 else False
        if has_gap:
            pred_record = {
                "candle_open_time": open_time,
                "ref_close": ref_close,
                "target_candle_open_time": target_open_time,
                "action": 0,
                "action_label": "NEUTRAL",
                "confidence": 0.0,
                "probabilities": {"p_up": 0.3333, "p_neutral": 0.3334, "p_down": 0.3333},
                "trend": "uncertain",
                "market_phase": "uncertain",
                "model_version": f"TradingEngine-v2.0-N{config.HORIZON}",
                "git_commit": self.git_commit,
                "config_hash": self.config_hash,
                "buffer_length": buffer_len,
                "compute_latency_ms": 0.0,
                "status": "NO_SIGNAL",
                "status_reason": "Buffer contains unfillable temporal gap",
            }
            self.ledger.record_prediction(pred_record)
            self._broadcast("new_prediction", pred_record)
            return

        # 3. Execute Model in Thread Executor under Lock
        t0 = time.time()
        async with self.lock:
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, self.engine.predict_candle, recent_df)
            except Exception as e:
                print(f"[ERROR] Inference failed on {open_time}: {e}")
                pred_record = {
                    "candle_open_time": open_time,
                    "ref_close": ref_close,
                    "target_candle_open_time": target_open_time,
                    "action": 0,
                    "action_label": "NEUTRAL",
                    "confidence": 0.0,
                    "probabilities": {"p_up": 0.3333, "p_neutral": 0.3334, "p_down": 0.3333},
                    "trend": "uncertain",
                    "market_phase": "uncertain",
                    "model_version": f"TradingEngine-v2.0-N{config.HORIZON}",
                    "git_commit": self.git_commit,
                    "config_hash": self.config_hash,
                    "buffer_length": buffer_len,
                    "compute_latency_ms": (time.time() - t0) * 1000.0,
                    "status": "NO_SIGNAL",
                    "status_reason": f"Model inference exception: {str(e)}",
                }
                self.ledger.record_prediction(pred_record)
                self._broadcast("new_prediction", pred_record)
                return

        latency_ms = (time.time() - t0) * 1000.0

        # 4. Form Record & Persist
        pred_record = {
            "candle_open_time": open_time,
            "ref_close": ref_close,
            "target_candle_open_time": target_open_time,
            "action": result["action"],
            "action_label": result["action_label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "trend": result["technical_analysis"]["trend"],
            "market_phase": result["technical_analysis"]["market_phase"],
            "model_version": f"TradingEngine-v2.0-N{config.HORIZON}",
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "buffer_length": buffer_len,
            "compute_latency_ms": latency_ms,
            "status": "PENDING",
            "status_reason": None,
        }
        self.ledger.record_prediction(pred_record)
        self._broadcast("new_prediction", pred_record)

        # 5. Resolve Pending Predictions (if candle k+2 has arrived)
        resolved_list = self.ledger.resolve_pending(self.candle_store)
        for r in resolved_list:
            self._broadcast("prediction_resolved", r)

