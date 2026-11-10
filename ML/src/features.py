"""Causal feature engineering.

Phase 3 will implement BlackRock-aligned factor categories:
returns, volatility, candle shape, momentum, volume, order flow, time, and multi-timeframe.
"""

import pandas as pd

LOOKBACK_CANDLES: int = 120


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build strictly causal features from closed candle history.
    
    Stub for Phase 3.
    """
    return df.copy()

