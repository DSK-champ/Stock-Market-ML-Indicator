"""Configuration constants and multipliers for Model-Informed Trade Zones.

All tunable multipliers for zone depth, stop placement, expected-move scaling,
and friction buffers are centralized here.
"""

from typing import Dict, Any

DEFAULT_ZONE_CONFIG: Dict[str, Any] = {
    # Entry zone depth multiplier against ATR(14)
    "k_entry": 0.75,
    # Stop distance multiplier against volatility
    "k_stop": 1.25,
    # Scaling factor for Garman-Klass expected price move
    "scale": 1.50,
    # Extra buffer in basis points added to 23 bps friction hurdle
    "buffer_bps": 5.0,
    # Exchange fee + round-trip taker friction floor
    "cost_floor_bps": 23.0,
    # Minimum directional confidence threshold for active holding
    "theta_dir": 0.32,
    # Stop multiplier expansion during high-volatility regimes
    "high_vol_multiplier": 1.35,
    # Prediction horizon in 5-minute candles (2 candles = 10 minutes)
    "horizon_candles": 2,
    # High volatility ATR / price threshold (70 bps)
    "high_vol_atr_threshold": 0.0070,
    # Swing lookback period for Dow swing high/low detection
    "swing_lookback": 12,
}
