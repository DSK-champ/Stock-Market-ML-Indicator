"""Backend Model Interface.

Exposes the unified TradingEngine to the backend API, websocket server,
and frontend dashboard.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add ML directory to Python path if not already present
ML_DIR = Path(__file__).resolve().parent.parent / "ML"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.trading_engine import TradingEngine, get_engine, predict_one, get_model_metrics

__all__ = ["TradingEngine", "get_engine", "predict_one", "get_model_metrics"]

