"""Empirical quantile band and calibrated smart price calculator (F4).

Derived from TRAIN OOF Folds 1-3 cross-validation and frozen on all TRAIN OOF.
Achieves 80.96% empirical coverage and reduces Winkler score from 62.45 bps to 52.75 bps.
"""

from typing import Dict, Any, Tuple
import math

# Frozen volatility terciles (from TRAIN clean)
T1_VOL = 0.001183
T2_VOL = 0.001919

# Frozen p_neutral terciles (from TRAIN OOF Folds 1-3)
T1_NEU = 0.2331
T2_NEU = 0.3363

# Calibrated k for smart price (minimizes MAE on Folds 1-3)
K_SMART_PRICE = 0.6
SQRT2 = math.sqrt(2.0)

# Frozen empirical (q10, q90) quantiles on All TRAIN OOF
FROZEN_QUANTILE_TABLE: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("Vol_Low", "Neu_Low"): (-1.4135, 1.4285),
    ("Vol_Low", "Neu_Mid"): (-1.5597, 1.6141),
    ("Vol_Low", "Neu_High"): (-1.4063, 1.4193),
    ("Vol_Mid", "Neu_Low"): (-1.5360, 1.4781),
    ("Vol_Mid", "Neu_Mid"): (-1.2402, 1.2209),
    ("Vol_Mid", "Neu_High"): (-1.1373, 1.1688),
    ("Vol_High", "Neu_Low"): (-1.1615, 1.1569),
    ("Vol_High", "Neu_Mid"): (-1.0441, 1.0626),
    ("Vol_High", "Neu_High"): (-0.6787, 1.1695),
}


def classify_vol_tercile(natr14: float) -> str:
    if natr14 <= T1_VOL:
        return "Vol_Low"
    elif natr14 <= T2_VOL:
        return "Vol_Mid"
    else:
        return "Vol_High"


def classify_neu_tercile(p_neutral: float) -> str:
    if p_neutral <= T1_NEU:
        return "Neu_Low"
    elif p_neutral <= T2_NEU:
        return "Neu_Mid"
    else:
        return "Neu_High"


def compute_empirical_band(
    close_price: float,
    atr14: float,
    p_up: float,
    p_down: float,
    p_neutral: float,
) -> Dict[str, Any]:
    """Compute empirical quantile band and calibrated smart price."""
    if close_price <= 0:
        return {
            "smart_price": None,
            "upper": None,
            "lower": None,
            "atr14": round(atr14, 2),
            "vol_tercile": "Vol_Mid",
            "neu_tercile": "Neu_Mid",
        }

    natr14 = atr14 / close_price if close_price > 0 else 0.0015
    vt = classify_vol_tercile(natr14)
    nt = classify_neu_tercile(p_neutral)

    q10, q90 = FROZEN_QUANTILE_TABLE.get((vt, nt), (-1.25, 1.25))

    smart_price = close_price + (p_up - p_down) * atr14 * SQRT2 * K_SMART_PRICE
    upper = smart_price + max(0.0, q90) * atr14
    lower = smart_price + min(0.0, q10) * atr14

    return {
        "smart_price": round(smart_price, 2),
        "upper": round(upper, 2),
        "lower": round(lower, 2),
        "atr14": round(atr14, 2),
        "vol_tercile": vt,
        "neu_tercile": nt,
        "q10": q10,
        "q90": q90,
    }
