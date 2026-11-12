"""Horizon Prediction Pipeline & Deployment Interface.

Exposes predict_horizon(candles_df, horizon=HORIZON, model_dir=MODEL_DIR) -> dict.
Evaluates single-timestamp inference by computing streaming features from a historical
buffer, executing the six base models, evaluating TA engine signals, and stacking
through the final meta-model.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import build_features
from src.ta_engine import compute_ta_scores, ta_label, TA_SCORE_COLUMNS
from src.model_common import prepare_model_features
from src.meta_model import META_INPUT_COLUMNS
from src.utils import setup_logger

logger = setup_logger("horizon_pipeline")


def predict_horizon(
    candles_df: pd.DataFrame,
    horizon: int = config.HORIZON,
    model_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate production prediction payload for a given horizon.

    Parameters
    ----------
    candles_df : pd.DataFrame
        DataFrame of historical closed 5m candles containing at least
        LOOKBACK_CANDLES (2018) rows with standard columns:
        open_time, open, high, low, close, volume.
    horizon : int
        Horizon parameter (e.g. 2 for 10-minute prediction).
    model_dir : Optional[Union[str, Path]]
        Directory storing trained model artifacts. If None, defaults
        to config.MODELS_DIR / f"horizon_{horizon}".

    Returns
    -------
    Dict[str, Any]
        Standardized prediction dictionary, or {"signal": "NO SIGNAL", "reason": ...}
        if preconditions fail.
    """
    if model_dir is None:
        model_dir = config.MODELS_DIR / f"horizon_{horizon}"
    model_dir = Path(model_dir)

    # 1. Validation of candle buffer
    if not isinstance(candles_df, pd.DataFrame):
        return {"signal": "NO SIGNAL", "reason": "candles_df must be a pandas DataFrame"}

    if len(candles_df) < config.LOOKBACK_CANDLES:
        return {
            "signal": "NO SIGNAL",
            "reason": f"Insufficient candles: {len(candles_df)} < {config.LOOKBACK_CANDLES}",
        }

    required_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    missing = [c for c in required_cols if c not in candles_df.columns]
    if missing:
        return {"signal": "NO SIGNAL", "reason": f"Missing required candle columns: {missing}"}

    # Verify timestamps
    df_sorted = candles_df.copy()
    if "open_time" in df_sorted.columns:
        df_sorted["open_time"] = pd.to_datetime(df_sorted["open_time"], utc=True)
        if not df_sorted["open_time"].is_monotonic_increasing:
            df_sorted = df_sorted.sort_values("open_time")
        # Check consecutive step
        diffs = df_sorted["open_time"].diff().dropna()
        if (diffs > pd.Timedelta(minutes=5)).any():
            return {
                "signal": "NO SIGNAL",
                "reason": "Timestamp gaps detected in candles_df (>5min interval)",
            }
        timestamp_val = str(df_sorted["open_time"].iloc[-1])
        df_sorted = df_sorted.set_index("open_time")
    else:
        if not isinstance(df_sorted.index, pd.DatetimeIndex):
            df_sorted.index = pd.to_datetime(df_sorted.index, utc=True)
        timestamp_val = str(df_sorted.index[-1])

    # 2. Compute streaming features on the full buffer
    try:
        features_df = build_features(df_sorted)
    except Exception as e:
        logger.error("Feature calculation failed: %s", e)
        return {"signal": "NO SIGNAL", "reason": f"Feature extraction failed: {str(e)}"}

    # Take the last row (current candle)
    last_row_df = features_df.iloc[[-1]].copy()

    # 3. Compute TA scores
    params_path = model_dir / f"ta_params_N{horizon}.json"
    try:
        ta_df = compute_ta_scores(last_row_df, params_path=params_path, horizon=horizon)
        ta_scores = ta_df.iloc[0].to_dict()
        ta_sig = ta_label(ta_scores)
    except Exception as e:
        logger.error("TA Engine scoring failed: %s", e)
        return {"signal": "NO SIGNAL", "reason": f"TA Engine failed: {str(e)}"}

    # 4. Load base models and compute base predictions
    base_models = [
        "binary_lr",
        "binary_rf",
        "binary_gb",
        "three_class_lr",
        "three_class_rf",
        "three_class_gb",
    ]

    bin_p_up: Dict[str, float] = {}
    tri_probs: Dict[str, Dict[str, float]] = {}

    for mid in base_models:
        model_path = model_dir / f"{mid}_N{horizon}.joblib"
        if not model_path.exists():
            return {
                "signal": "NO SIGNAL",
                "reason": f"Required model artifact not found: {model_path}",
            }

        payload = joblib.load(model_path)
        clf = payload["model"]
        scaler = payload.get("scaler")
        feat_cols = payload["feature_cols"]

        X_row = prepare_model_features(last_row_df, feat_cols)
        if scaler is not None:
            X_eval = scaler.transform(X_row)
        else:
            X_eval = X_row

        probs = clf.predict_proba(X_eval)[0]

        if mid.startswith("binary"):
            # classes [0, 1]
            bin_p_up[mid] = float(probs[1])
        else:
            # classes [-1, 0, 1]
            tri_probs[mid] = {
                "p_down": float(probs[0]),
                "p_neutral": float(probs[1]),
                "p_up": float(probs[2]),
            }

    # 5. Build meta inputs row (exact 15 columns)
    meta_dict = {
        "bin_lr_p_up": bin_p_up["binary_lr"],
        "bin_rf_p_up": bin_p_up["binary_rf"],
        "bin_gb_p_up": bin_p_up["binary_gb"],
        "tri_lr_p_up": tri_probs["three_class_lr"]["p_up"],
        "tri_lr_p_down": tri_probs["three_class_lr"]["p_down"],
        "tri_rf_p_up": tri_probs["three_class_rf"]["p_up"],
        "tri_rf_p_down": tri_probs["three_class_rf"]["p_down"],
        "tri_gb_p_up": tri_probs["three_class_gb"]["p_up"],
        "tri_gb_p_down": tri_probs["three_class_gb"]["p_down"],
        "ta_rule_vote": ta_scores["ta_rule_vote"],
        "ta_loc": ta_scores["ta_loc"],
        "ta_trend": ta_scores["ta_trend"],
        "ta_ema50": ta_scores["ta_ema50"],
        "ta_rsi": ta_scores["ta_rsi"],
        "ta_align": ta_scores["ta_align"],
    }
    meta_row = pd.DataFrame([meta_dict])[META_INPUT_COLUMNS]

    # 6. Load meta-model and generate final prediction
    meta_path = model_dir / f"meta_model_N{horizon}.joblib"
    if not meta_path.exists():
        return {
            "signal": "NO SIGNAL",
            "reason": f"Meta-model artifact not found: {meta_path}",
        }

    meta_payload = joblib.load(meta_path)
    meta_clf = meta_payload["model"]
    meta_scaler = meta_payload["scaler"]

    meta_eval = meta_scaler.transform(meta_row)
    meta_probs = meta_clf.predict_proba(meta_eval)[0]
    p_down = float(meta_probs[0])
    p_neutral = float(meta_probs[1])
    p_up = float(meta_probs[2])

    assert abs(p_down + p_neutral + p_up - 1.0) < 1e-5, "Meta probabilities do not sum to 1"

    argmax_idx = int(np.argmax(meta_probs))
    pred_class = int(meta_clf.classes_[argmax_idx])
    final_pred = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}[pred_class]
    confidence = float(np.max(meta_probs))

    # 7. Aggregate ensemble directions
    mean_bin_p_up = float(np.mean(list(bin_p_up.values())))
    binary_ensemble_direction = "UP" if mean_bin_p_up > 0.5 else "DOWN"

    mean_tri_down = float(np.mean([tri_probs[m]["p_down"] for m in tri_probs]))
    mean_tri_neu = float(np.mean([tri_probs[m]["p_neutral"] for m in tri_probs]))
    mean_tri_up = float(np.mean([tri_probs[m]["p_up"] for m in tri_probs]))
    tri_argmax = int(np.argmax([mean_tri_down, mean_tri_neu, mean_tri_up]))
    three_class_ensemble_direction = ["DOWN", "NEUTRAL", "UP"][tri_argmax]

    return {
        "horizon": horizon,
        "minutes": horizon * 5,
        "timestamp": timestamp_val,
        "final_prediction": final_pred,
        "p_up": p_up,
        "p_down": p_down,
        "p_neutral": p_neutral,
        "confidence": confidence,
        "binary_ensemble_direction": binary_ensemble_direction,
        "binary_p_up": bin_p_up,
        "three_class_ensemble_direction": three_class_ensemble_direction,
        "three_class_probs": tri_probs,
        "ta_signal": ta_sig,
        "ta_scores": ta_scores,
    }
