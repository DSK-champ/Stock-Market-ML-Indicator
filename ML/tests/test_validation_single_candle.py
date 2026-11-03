import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.validation import validate_candle

candle = {
    "open_time": "2026-10-31 10:20:00+00:00",
    "open": 109500.0,
    "high": 109600.0,
    "low": 109400.0,
    "close": 109550.0,
    "volume": 12.5,
    "quote_volume": 1368750.0,
    "number_of_trades": 1500,
    "taker_buy_base_volume": 7.0,
    "taker_buy_quote_volume": 766000.0,
}

result = validate_candle(candle)

print(result)