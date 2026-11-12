"""Configuration settings for the Live Forward Testing Backend."""

from pathlib import Path
from typing import List, Dict, Any

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

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

# Reference Benchmarks (Frozen Project Scorecard)
REF_TEST_ACCURACY = 51.90    # Held-out Test directional accuracy (primary benchmark)
REF_TEST_COVERAGE = 52.50    # Held-out Test coverage
REF_OOF_ACCURACY = 52.16     # Out-of-fold CV (thresholds tuned on it)
REF_OOF_COVERAGE = 65.90     # Out-of-fold CV coverage
REF_EDGE_BPS = 190.0         # 1.9% theoretical directional edge (51.90% - 50.0%)

# Buffer, Parity & Timing Rules
BUFFER_SIZE = 3000           # Rolling candles lookback window (>= 2016 for 7-day seasonal features)
MIN_BUFFER_SIZE = 2016        # Absolute minimum required candles for 0 NaN features
REQUIRE_CLEAN_FEATURES = True # If True, any imputed feature on the last row emits NO_SIGNAL
WATCHDOG_TIMEOUT_SECONDS = 15.0
RECONCILIATION_DELAY_SECONDS = 3.0
PROACTIVE_RECONNECT_SECONDS = 23 * 3600  # 23 hours
CLOCK_DRIFT_WARN_SECONDS = 2.0

# WebSocket Limits
MAX_RECONNECT_BACKOFF = 60.0
INITIAL_RECONNECT_BACKOFF = 1.0

