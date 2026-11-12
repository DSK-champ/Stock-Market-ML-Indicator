"""Indicative Price Range and ATR Band Calculation Module.

Formulas:
- ATR14: Wilder's Smoothed Average True Range over period 14 on closed candles.
- smart_price = close_t + (p_up - p_down) * ATR14 * sqrt(2) * K_MOVE
- half_width = Z * ATR14 * sqrt(2) * (0.75 + 0.5 * (1 - p_neutral))
- upper = smart_price + half_width
- lower = smart_price - half_width

Defaults: K_MOVE = 0.5, Z = 1.28.
These values represent an indicative visual range based on ATR and model probabilities,
not a formal statistical prediction interval.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_wilder_atr14(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Wilder's ATR over closed candles in df.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have 'high', 'low', 'close' columns.
    period : int
        Smoothing period (default 14).
    """
    if len(df) < period + 1:
        # Fallback to simple high - low if insufficient history
        high = df["high"].values
        low = df["low"].values
        return float(np.mean(high - low)) if len(df) > 0 else 100.0

    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    n = len(df)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.zeros(n, dtype=np.float64)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / float(period)

    val = float(atr[-1])
    return val if np.isfinite(val) and val > 0 else float(np.mean(high[-period:] - low[-period:]))


def compute_indicative_bands(
    close_price: float,
    atr14: float,
    p_up: float,
    p_down: float,
    p_neutral: float,
    k_move: float = 0.5,
    z: float = 1.28,
) -> Dict[str, float]:
    """Calculate smart price and upper/lower indicative bands.
    
    Parameters
    ----------
    close_price : float
        Reference closing price of candle t.
    atr14 : float
        Wilder's ATR 14.
    p_up, p_down, p_neutral : float
        Model probabilities on the simplex.
    k_move : float
        Directional multiplier (default 0.5).
    z : float
        Half-width multiplier (default 1.28).
    
    Returns
    -------
    dict
        smart_price, upper, lower, half_width, atr14.
    """
    sqrt2 = np.sqrt(2.0)
    # smart_price = close_t + (p_up - p_down) * ATR14 * sqrt(2) * K_MOVE
    smart_price = float(close_price + (p_up - p_down) * atr14 * sqrt2 * k_move)
    
    # half_width = Z * ATR14 * sqrt(2) * (0.75 + 0.5 * (1 - p_neutral))
    half_width = float(z * atr14 * sqrt2 * (0.75 + 0.5 * (1.0 - p_neutral)))
    
    upper = float(smart_price + half_width)
    lower = float(smart_price - half_width)

    return {
        "atr14": round(float(atr14), 4),
        "smart_price": round(float(smart_price), 2),
        "upper": round(float(upper), 2),
        "lower": round(float(lower), 2),
        "half_width": round(float(half_width), 2),
    }

