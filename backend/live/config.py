"""Configuration settings for the Live Forward Testing Backend."""

from pathlib import Path
from typing import List, Dict, Any

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CANDLES_DB_PATH = DATA_DIR / "candles.db"
LIVE_LEDGER_DB_PATH = DATA_DIR / "live_ledger.db"
REPLAY_LEDGER_DB_PATH = DATA_DIR / "replay_ledger.db"

# Market & Instrument
SYMBOL = "BTCUSDT"
SYMBOL_STREAM = SYMBOL.lower()
INTERVAL = "5m"
INTERVAL_MINUTES = 5
INTERVAL_MS = INTERVAL_MINUTES * 60 * 1000

# Binance Endpoints
WS_ENDPOINTS: List[str] = [
    f"wss://stream.binance.com:9443/ws/{SYMBOL_STREAM}@kline_{INTERVAL}",
    f"wss://stream.binance.com:443/ws/{SYMBOL_STREAM}@kline_{INTERVAL}",
    f"wss://data-stream.binance.vision/ws/{SYMBOL_STREAM}@kline_{INTERVAL}",
]
REST_BASE_URL = "https://api.binance.com/api/v3"

# Model & Economic Parameters (Frozen)
HORIZON = 2                  # N = 2 candles (10 minutes)
COST_FLOOR_BPS = 23.0        # Round-trip exchange friction in basis points (0.0023)
CONTEXT_WEIGHT = 0.05
OVERRIDE_THRESHOLD = 0.32
OVERRIDE_MARGIN = 0.04

# Buffer & Timing Rules
BUFFER_SIZE = 1000            # Rolling candles lookback window
MIN_BUFFER_SIZE = 100         # Absolute minimum required candles for reliable indicators
WATCHDOG_TIMEOUT_SECONDS = 15.0
RECONCILIATION_DELAY_SECONDS = 3.0
PROACTIVE_RECONNECT_SECONDS = 23 * 3600  # 23 hours
CLOCK_DRIFT_WARN_SECONDS = 2.0

# WebSocket Limits
MAX_RECONNECT_BACKOFF = 60.0
INITIAL_RECONNECT_BACKOFF = 1.0

