"""Real-time inference interface for live backend integration.

Phase 7 will implement predict_one(candles_df) with strict closed-candle causal checks.
"""

from typing import Dict, Any
import pandas as pd


def predict_one(candles_df: pd.DataFrame) -> Dict[str, Any]:
    """Given lookback closed candles, generate calibrated prediction.
    
    Stub for Phase 7.
    """
    return {
        "direction": "NO_SIGNAL",
        "confidence": 0.0,
        "probabilities": {"UP": 0.0, "DOWN": 0.0, "NEUTRAL": 1.0},
        "timestamp": None,
        "reason": "Stub implementation",
    }

