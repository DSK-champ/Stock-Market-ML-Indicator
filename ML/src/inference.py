"""Real-Time Inference Interface for Live Backend Integration.

Routes requests directly to the unified TradingEngine.
Guarantees strict causal, zero-repainting predictions on closed candles.
"""

from typing import Dict, Any, Optional
import pandas as pd
from src.trading_engine import TradingEngine, get_engine, predict_one, get_model_metrics

__all__ = ["TradingEngine", "get_engine", "predict_one", "get_model_metrics"]
