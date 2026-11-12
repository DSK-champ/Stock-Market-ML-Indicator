"""Technical Analysis Signal Engine.

Rule-based, non-trainable signal engine computing 6 standardized technical
scores in [-1.0, 1.0] using conventional trend-following orientation
(bullish = positive, bearish = negative).

Orientation:
  - ta_rule_vote: feat_struct_rule_vote
  - ta_loc: clip((feat_loc_position_288 - 0.5) * 2, -1, 1)
  - ta_trend: feat_struct_trend_up - feat_struct_trend_down
  - ta_ema50: clip(feat_mom_ema_ratio_50 / q99, -1, 1), where q99 is the
    99th percentile of abs(feat_mom_ema_ratio_50) on TRAIN
  - ta_rsi: (feat_mom_rsi_14 - 0.5) * 2
  - ta_align: feat_ma_alignment_ema_5m

Composite Label:
  - ta_label: mean of the 6 scores:
      BULLISH if mean > +0.10,
      BEARISH if mean < -0.10,
      else NEUTRAL.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src import config

TA_SCORE_COLUMNS = [
    "ta_rule_vote",
    "ta_loc",
    "ta_trend",
    "ta_ema50",
    "ta_rsi",
    "ta_align",
]


def compute_ta_scores(
    df: pd.DataFrame,
    q99: Optional[float] = None,
    params_path: Optional[Union[str, Path]] = None,
    save_params: bool = False,
    horizon: int = config.HORIZON,
) -> pd.DataFrame:
    """Compute the 6 standardized technical analysis scores in [-1, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing required feature columns.
    q99 : Optional[float]
        Precomputed 99th percentile of abs(feat_mom_ema_ratio_50).
        If None, attempts to load from params_path or compute from df.
    params_path : Optional[Union[str, Path]]
        Path to JSON file storing q99. If None, uses default
        models/horizon_{horizon}/ta_params_N{horizon}.json.
    save_params : bool
        If True and q99 was newly computed, saves to params_path.
    horizon : int
        Horizon parameter for output paths.

    Returns
    -------
    pd.DataFrame
        DataFrame with 6 score columns in [-1.0, 1.0], matching df's index.
    """
    if params_path is None:
        params_path = config.MODELS_DIR / f"horizon_{horizon}" / f"ta_params_N{horizon}.json"
    params_path = Path(params_path)

    # Resolve q99
    if q99 is None:
        if params_path.exists():
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                q99 = float(data.get("q99", 0.0))
            except Exception:
                q99 = None

        if q99 is None or q99 <= 1e-8:
            series_abs = np.abs(df["feat_mom_ema_ratio_50"].dropna())
            q99 = float(np.nanpercentile(series_abs, 99)) if len(series_abs) > 0 else 1.0
            if q99 <= 1e-8:
                q99 = 1.0

            if save_params:
                params_path.parent.mkdir(parents=True, exist_ok=True)
                with open(params_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "q99": q99,
                            "horizon": horizon,
                            "column": "feat_mom_ema_ratio_50",
                            "metric": "abs_99th_percentile",
                        },
                        f,
                        indent=2,
                    )

    scores = pd.DataFrame(index=df.index)

    # 1. ta_rule_vote
    scores["ta_rule_vote"] = np.clip(
        df["feat_struct_rule_vote"].astype(float).fillna(0.0), -1.0, 1.0
    )

    # 2. ta_loc
    scores["ta_loc"] = np.clip(
        (df["feat_loc_position_288"].astype(float).fillna(0.5) - 0.5) * 2.0, -1.0, 1.0
    )

    # 3. ta_trend
    up = df["feat_struct_trend_up"].astype(float).fillna(0.0)
    down = df["feat_struct_trend_down"].astype(float).fillna(0.0)
    scores["ta_trend"] = np.clip(up - down, -1.0, 1.0)

    # 4. ta_ema50
    ema_ratio = df["feat_mom_ema_ratio_50"].astype(float).fillna(0.0)
    scores["ta_ema50"] = np.clip(ema_ratio / q99, -1.0, 1.0)

    # 5. ta_rsi
    rsi = df["feat_mom_rsi_14"].astype(float).fillna(0.5)
    scores["ta_rsi"] = np.clip((rsi - 0.5) * 2.0, -1.0, 1.0)

    # 6. ta_align
    scores["ta_align"] = np.clip(
        df["feat_ma_alignment_ema_5m"].astype(float).fillna(0.0), -1.0, 1.0
    )

    return scores[TA_SCORE_COLUMNS]


def ta_label(scores_df: Union[pd.DataFrame, pd.Series, Dict[str, float]]) -> Union[pd.Series, str]:
    """Compute the composite discrete TA label from 6 TA scores.

    BULLISH if mean > +0.10
    BEARISH if mean < -0.10
    NEUTRAL otherwise.

    Parameters
    ----------
    scores_df : DataFrame, Series, or dict
        Must contain the 6 TA_SCORE_COLUMNS.

    Returns
    -------
    pd.Series or str
        Categorical TA signal ('BULLISH', 'BEARISH', or 'NEUTRAL').
    """
    if isinstance(scores_df, dict):
        mean_val = float(np.mean([scores_df[c] for c in TA_SCORE_COLUMNS]))
        if mean_val > 0.10:
            return "BULLISH"
        elif mean_val < -0.10:
            return "BEARISH"
        return "NEUTRAL"
    elif isinstance(scores_df, pd.Series):
        mean_val = float(scores_df[TA_SCORE_COLUMNS].mean())
        if mean_val > 0.10:
            return "BULLISH"
        elif mean_val < -0.10:
            return "BEARISH"
        return "NEUTRAL"
    else:
        means = scores_df[TA_SCORE_COLUMNS].mean(axis=1)
        labels = pd.Series("NEUTRAL", index=scores_df.index, dtype=object)
        labels[means > 0.10] = "BULLISH"
        labels[means < -0.10] = "BEARISH"
        return labels

