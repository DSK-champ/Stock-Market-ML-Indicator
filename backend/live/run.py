"""Live & Replay CLI Runner for the BTCUSDT AI Trading Indicator."""

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import pandas as pd

# Add ML and backend to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent / "ML"))

from backend.live import config
from backend.live.candle_store import CandleStore
from backend.live.ledger import PredictionLedger
from backend.live.predictor import PredictorWorker
from backend.live.stream import BinanceKlineStream
from backend.live.api import LiveApiServer
from backend.live.metrics import LiveMetricsCalculator
from src.column_dictionary import load_master


class ReplayServer:
    """Mock streaming engine replaying Train-period candles at N-speed."""

    def __init__(
        self,
        candle_store: CandleStore,
        predictor: PredictorWorker,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        speed: float = 60.0,
    ) -> None:
        self.candle_store = candle_store
        self.predictor = predictor
        self.from_date = from_date
        self.to_date = to_date
        self.speed = speed
        self.state = "REPLAY"
        self.measured_latency_ms = 0.5
        self.reconnect_count = 0
        self.last_closed_candle_time = None
        self.forming_candle = None

    async def run(self) -> None:
        print(f"[REPLAY] Loading candles for replay ({self.from_date} to {self.to_date})...")
        clean_csv = BASE_DIR.parent / "ML" / "data" / "processed" / "clean.csv"
        if clean_csv.exists():
            df_full = pd.read_csv(clean_csv)
            df_full["open_time"] = pd.to_datetime(df_full["open_time"], utc=True)
            df_full = df_full.set_index("open_time").sort_index()
        else:
            df_full = load_master(split="train")

        from_dt = pd.to_datetime(self.from_date, utc=True) if self.from_date else df_full.index[1000]
        to_dt = pd.to_datetime(self.to_date, utc=True) if self.to_date else from_dt + pd.Timedelta(hours=2)

        # 1. Warm-up prefill: load up to buffer_size candles strictly before from_dt
        warmup_df = df_full[df_full.index < from_dt].tail(self.predictor.buffer_size)
        print(f"[REPLAY] Pre-filling CandleStore with {len(warmup_df)} warm-up candles prior to {from_dt}...")
        for i in range(len(warmup_df)):
            row = warmup_df.iloc[i]
            ts_str = warmup_df.index[i].strftime("%Y-%m-%d %H:%M:%S+00:00")
            c_dict = {
                "open_time": ts_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "quote_volume": float(row.get("quote_volume", row["volume"] * row["close"])),
                "number_of_trades": float(row.get("number_of_trades", 1000.0)),
                "taker_buy_base_volume": float(row.get("taker_buy_base_volume", row["volume"] * 0.5)),
                "taker_buy_quote_volume": float(row.get("taker_buy_quote_volume", row["volume"] * row["close"] * 0.5)),
                "close_time": (warmup_df.index[i] + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
            }
            self.candle_store.store_candle(c_dict, source="WARMUP")

        # 2. Slice replay candles (and include 2 outcome candles after to_dt for full resolution)
        replay_df = df_full[(df_full.index >= from_dt) & (df_full.index <= to_dt)]
        outcome_df = df_full[df_full.index > to_dt].head(config.HORIZON)

        total_candles = len(replay_df) + len(outcome_df)
        print(f"[REPLAY] Replaying {len(replay_df)} test candles (+{len(outcome_df)} outcome resolution candles) at {self.speed}x speed...")
        sleep_sec = (config.INTERVAL_MINUTES * 60) / max(1.0, self.speed)

        all_stream_df = pd.concat([replay_df, outcome_df])
        for i in range(len(all_stream_df)):
            row = all_stream_df.iloc[i]
            ts = all_stream_df.index[i]
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S+00:00")
            self.last_closed_candle_time = ts_str

            candle_dict = {
                "open_time": ts_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "quote_volume": float(row.get("quote_volume", row["volume"] * row["close"])),
                "number_of_trades": float(row.get("number_of_trades", 1000.0)),
                "taker_buy_base_volume": float(row.get("taker_buy_base_volume", row["volume"] * 0.5)),
                "taker_buy_quote_volume": float(row.get("taker_buy_quote_volume", row["volume"] * row["close"] * 0.5)),
                "close_time": (ts + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
            }

            self.candle_store.store_candle(candle_dict, source="REPLAY")
            await self.predictor.enqueue_candle(candle_dict)

            if sleep_sec > 0.001:
                await asyncio.sleep(sleep_sec)

        # Allow final queue flush
        await self.predictor.queue.join()
        print("\n" + "=" * 80)
        print("[REPLAY] Replay complete. All candles processed.")
        print("=" * 80)

        calc = LiveMetricsCalculator(db_path=self.predictor.ledger.db_path)
        metrics = calc.compute_metrics()
        print(f"Total Resolved Candles: {metrics.get('total_resolved_candles')}")
        print(f"Directional Calls:      {metrics.get('directional_calls_count')}")
        print(f"Directional Acc:        {metrics.get('directional_accuracy', {}).get('overall_pct')}%")
        print(f"Wilson 95% CI:          {metrics.get('directional_accuracy', {}).get('wilson_95_ci')}")
        print(f"Too Few Samples (<2000):{metrics.get('too_few_samples')}")
        print("=" * 80 + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Live Forward Testing & Replay Engine")
    parser.add_argument("--mode", choices=["live", "replay"], default="live", help="Execution mode")
    parser.add_argument("--from", dest="from_date", type=str, default=None, help="Replay start date (ISO)")
    parser.add_argument("--to", dest="to_date", type=str, default=None, help="Replay end date (ISO)")
    parser.add_argument("--speed", type=float, default=60.0, help="Replay speed multiplier")
    parser.add_argument("--dry-run", action="store_true", help="Run without persisting to DB")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")

    args = parser.parse_args()

    print("=" * 80)
    print(f"STARTING LIVE FORWARD TESTING LAYER (MODE = {args.mode.upper()})")
    print("=" * 80)

    # Database selection
    if args.dry_run:
        candles_db = Path(":memory:")
        ledger_db = Path(":memory:")
    elif args.mode == "replay":
        candles_db = config.DATA_DIR / "replay_candles.db"
        ledger_db = config.REPLAY_LEDGER_DB_PATH
    else:
        candles_db = config.CANDLES_DB_PATH
        ledger_db = config.LIVE_LEDGER_DB_PATH

    candle_store = CandleStore(db_path=candles_db)
    ledger = PredictionLedger(db_path=ledger_db)
    predictor = PredictorWorker(candle_store=candle_store, ledger=ledger, buffer_size=config.BUFFER_SIZE)

    stream_engine = None
    if args.mode == "live":
        # 1. Startup REST backfill
        candle_store.backfill_from_rest(limit=config.BUFFER_SIZE)
        stream_engine = BinanceKlineStream(candle_store=candle_store, predictor=predictor)
    else:
        stream_engine = ReplayServer(
            candle_store=candle_store,
            predictor=predictor,
            from_date=args.from_date,
            to_date=args.to_date,
            speed=args.speed,
        )

    api_server = LiveApiServer(
        candle_store=candle_store,
        ledger=ledger,
        predictor=predictor,
        stream_state_provider=stream_engine,
        host=args.host,
        port=args.port,
    )

    # Start services
    await predictor.start()
    await api_server.start()

    # Graceful shutdown handler
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        print("\n[SHUTDOWN] Intercepted stop signal. Shutting down gracefully...")
        stop_event.set()

    for s in [signal.SIGINT, signal.SIGTERM]:
        try:
            loop.add_signal_handler(s, _signal_handler)
        except NotImplementedError:
            pass

    try:
        if args.mode == "live":
            await stream_engine.start()
            await stop_event.wait()
        else:
            await stream_engine.run()
    finally:
        print("[SHUTDOWN] Stopping all live services and flushing ledger...")
        if args.mode == "live":
            await stream_engine.stop()
        await api_server.stop()
        await predictor.stop()
        print("[SHUTDOWN] All services stopped cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
