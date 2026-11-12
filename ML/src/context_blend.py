"""Context-Based Probability Blending & Evaluation Module (Horizon N=2).

Implements:
1. ML Base Distribution:
   - p_neutral_ml = mean of 3-class models P(neutral)
   - b = mean of binary models P(up)
   - p_up_ml = (1 - p_neutral_ml) * b
   - p_down_ml = (1 - p_neutral_ml) * (1 - b)
2. Domain & Learned Context Tables:
   - T_HAND (expert rules with mirror-symmetry guarantee)
   - T_LEARNED (estimated strictly on earlier folds' training rows)
3. Soft Context Tensor Product:
   - q(class) = sum_{tau, phi, m} P(tau) * P(phi) * p_ml[m] * 1[T[tau, phi, m] == class]
   - p_final = (1 - w) * p_ml + w * q
   - Stricter variant: w_ctx = w * ta_trend_strength * ta_phase_strength
4. Nested Walk-Forward Tuning:
   - Evaluates w in {0.0, 0.1, ..., 1.0} on folds < k (drop 12-candle purge)
   - Applies chosen w to fold k.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src import config
from src.utils import setup_logger

logger = setup_logger("context_blend")

# Class mapping: -1: DOWN, 0: NEUTRAL, 1: UP
CLASSES = [-1, 0, 1]
CLASS_NAMES = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
NAME_TO_CLASS = {"DOWN": -1, "NEUTRAL": 0, "UP": 1}

# ---------------------------------------------------------------------------
# 1. T_HAND Domain Heuristic Mapping
# ---------------------------------------------------------------------------
# Key: (trend: "UP"|"DOWN", phase: "PULLBACK"|"BREAKTHROUGH", ml_class: -1|0|1) -> final_class: -1|0|1
T_HAND: Dict[Tuple[str, str, int], int] = {
    # Uptrend, Pullback: UP -> NEUTRAL; DOWN -> DOWN; NEUTRAL -> DOWN
    ("UP", "PULLBACK", 1): 0,
    ("UP", "PULLBACK", -1): -1,
    ("UP", "PULLBACK", 0): -1,
    # Uptrend, Breakthrough: UP -> UP; DOWN -> NEUTRAL; NEUTRAL -> UP
    ("UP", "BREAKTHROUGH", 1): 1,
    ("UP", "BREAKTHROUGH", -1): 0,
    ("UP", "BREAKTHROUGH", 0): 1,
    # Downtrend, Pullback: UP -> UP; DOWN -> NEUTRAL; NEUTRAL -> UP
    ("DOWN", "PULLBACK", 1): 1,
    ("DOWN", "PULLBACK", -1): 0,
    ("DOWN", "PULLBACK", 0): 1,
    # Downtrend, Breakthrough: UP -> NEUTRAL; DOWN -> DOWN; NEUTRAL -> DOWN
    ("DOWN", "BREAKTHROUGH", 1): 0,
    ("DOWN", "BREAKTHROUGH", -1): -1,
    ("DOWN", "BREAKTHROUGH", 0): -1,
}


def check_t_hand_symmetry() -> bool:
    """Verify that T_HAND possesses exact mirror-symmetry.

    Inverting trend (UP <-> DOWN) and inverting class (-1 <-> 1, 0 -> 0)
    must produce the exact inverted final output:
        T_HAND[inv(trend), phase, inv(ml)] == inv(T_HAND[trend, phase, ml])
    """
    inv_trend = {"UP": "DOWN", "DOWN": "UP"}
    inv_class = {-1: 1, 1: -1, 0: 0}

    for (trend, phase, ml_cls), out_cls in T_HAND.items():
        mirrored_key = (inv_trend[trend], phase, inv_class[ml_cls])
        expected_mirrored_out = inv_class[out_cls]
        actual_mirrored_out = T_HAND[mirrored_key]
        if actual_mirrored_out != expected_mirrored_out:
            logger.error(
                "Symmetry broken for key %s: mapped to %d, mirrored key %s mapped to %d (expected %d)",
                (trend, phase, ml_cls),
                out_cls,
                mirrored_key,
                actual_mirrored_out,
                expected_mirrored_out,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# 2. ML Base Distribution Synthesis
# ---------------------------------------------------------------------------
def synthesize_ml_distribution(
    bin_oofs: Dict[str, pd.DataFrame],
    tri_oofs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Synthesize ML base distribution blending 3-class sizing and binary direction.

    p_neutral_ml = mean(three_class_models P(neutral))
    b = mean(binary_models P(up))
    p_up_ml = (1 - p_neutral_ml) * b
    p_down_ml = (1 - p_neutral_ml) * (1 - b)
    """
    ref_df = next(iter(bin_oofs.values())).copy()
    timestamps = pd.to_datetime(ref_df["open_time"], utc=True)
    folds = ref_df["fold"].values if "fold" in ref_df.columns else np.zeros(len(ref_df), dtype=int)

    # Average binary P(up)
    b_probs = []
    for mid, df in bin_oofs.items():
        b_probs.append(df["p_up"].values)
    b = np.mean(b_probs, axis=0)

    # Average 3-class P(neutral)
    neu_probs = []
    for mid, df in tri_oofs.items():
        neu_probs.append(df["p_neutral"].values)
    p_neutral_ml = np.mean(neu_probs, axis=0)

    # Calculate sized direction probabilities
    p_up_ml = (1.0 - p_neutral_ml) * b
    p_down_ml = (1.0 - p_neutral_ml) * (1.0 - b)

    sum_probs = p_down_ml + p_neutral_ml + p_up_ml
    assert np.allclose(sum_probs, 1.0, atol=1e-6), "p_ml must sum to 1.0 within 1e-6"

    # ML class argmax (-1, 0, 1)
    ml_matrix = np.column_stack([p_down_ml, p_neutral_ml, p_up_ml])
    ml_pred = np.array(CLASSES)[np.argmax(ml_matrix, axis=1)]

    out_df = pd.DataFrame(
        {
            "open_time": timestamps,
            "fold": folds,
            "p_down_ml": p_down_ml,
            "p_neutral_ml": p_neutral_ml,
            "p_up_ml": p_up_ml,
            "ml_pred": ml_pred,
        }
    )
    return out_df


# ---------------------------------------------------------------------------
# 3. Soft Context Tensor Product Blending
# ---------------------------------------------------------------------------
def compute_soft_blend(
    ta_trend_p_up: np.ndarray,
    ta_phase_p_pullback: np.ndarray,
    p_ml: np.ndarray,  # shape (N, 3): [p_down, p_neutral, p_up]
    table: Dict[Tuple[str, str, int], int],
    w: float = 0.5,
    trend_strength: Optional[np.ndarray] = None,
    phase_strength: Optional[np.ndarray] = None,
    use_context_strength: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute soft context probability tensor q and blended distribution p_final.

    Parameters
    ----------
    ta_trend_p_up : np.ndarray
        P(Trend = UP) in [0, 1].
    ta_phase_p_pullback : np.ndarray
        P(Phase = PULLBACK) in [0, 1].
    p_ml : np.ndarray
        Array of shape (N, 3) representing [p_down, p_neutral, p_up].
    table : Dict[Tuple[str, str, int], int]
        Mapping table (trend, phase, ml_class) -> final_class.
    w : float
        Blending weight in [0, 1].
    trend_strength : Optional[np.ndarray]
        ta_trend_strength in [0, 1].
    phase_strength : Optional[np.ndarray]
        ta_phase_strength in [0, 1].
    use_context_strength : bool
        If True, scales w by trend_strength * phase_strength.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (q, p_final) each of shape (N, 3).
    """
    n = len(ta_trend_p_up)
    t_up = ta_trend_p_up
    t_dn = 1.0 - t_up
    f_pb = ta_phase_p_pullback
    f_bt = 1.0 - f_pb

    # p_ml columns: 0: DOWN (-1), 1: NEUTRAL (0), 2: UP (1)
    p_dn = p_ml[:, 0]
    p_neu = p_ml[:, 1]
    p_up = p_ml[:, 2]

    # q tensor: shape (N, 3), indices 0: DOWN, 1: NEUTRAL, 2: UP
    cls_to_idx = {-1: 0, 0: 1, 1: 2}
    q = np.zeros((n, 3), dtype=np.float64)

    combinations = [
        ("UP", "PULLBACK", t_up * f_pb),
        ("UP", "BREAKTHROUGH", t_up * f_bt),
        ("DOWN", "PULLBACK", t_dn * f_pb),
        ("DOWN", "BREAKTHROUGH", t_dn * f_bt),
    ]

    ml_classes = [(-1, p_dn), (0, p_neu), (1, p_up)]

    for trend_str, phase_str, p_ctx in combinations:
        for ml_cls, p_m in ml_classes:
            out_cls = table.get((trend_str, phase_str, ml_cls), 0)
            c_idx = cls_to_idx[out_cls]
            q[:, c_idx] += p_ctx * p_m

    # Verify q simplex
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-6), "q must sum to 1.0 within 1e-6"

    if use_context_strength:
        if trend_strength is None or phase_strength is None:
            raise ValueError("trend_strength and phase_strength required when use_context_strength=True")
        w_eff = w * trend_strength * phase_strength
        w_eff = w_eff[:, None]
    else:
        w_eff = w

    p_final = (1.0 - w_eff) * p_ml + w_eff * q
    assert np.allclose(p_final.sum(axis=1), 1.0, atol=1e-6), "p_final must sum to 1.0 within 1e-6"

    return q, p_final


# ---------------------------------------------------------------------------
# 4. T_LEARNED Estimation on Earlier Training Folds
# ---------------------------------------------------------------------------
def fit_learned_table(
    df_history: pd.DataFrame,
    target_col: str = "raw_sign_2",
) -> Dict[Tuple[str, str, int], int]:
    """Fit empirical T_LEARNED table on historic training rows.

    For each cell (trend, phase, ml_class), computes empirical distribution
    of target_col. Final class = UP (+1) if UP share > 0.51, DOWN (-1) if < 0.49,
    else NEUTRAL (0).
    """
    table: Dict[Tuple[str, str, int], int] = {}
    trends = ["UP", "DOWN"]
    phases = ["PULLBACK", "BREAKTHROUGH"]
    ml_classes = [-1, 0, 1]

    for trend in trends:
        for phase in phases:
            for ml_cls in ml_classes:
                mask = (
                    (df_history["ta_trend_label"] == trend)
                    & (df_history["ta_phase_label"] == phase)
                    & (df_history["ml_pred"] == ml_cls)
                )
                subset = df_history[mask]
                if len(subset) == 0:
                    table[(trend, phase, ml_cls)] = 0
                    continue

                y = subset[target_col].dropna().values
                valid = y[y != 0]
                if len(valid) == 0:
                    table[(trend, phase, ml_cls)] = 0
                    continue

                up_share = (valid == 1).mean()
                if up_share - 0.5 > 0.01:
                    decision = 1
                elif up_share - 0.5 < -0.01:
                    decision = -1
                else:
                    decision = 0
                table[(trend, phase, ml_cls)] = decision

    return table


# ---------------------------------------------------------------------------
# 5. Nested Walk-Forward w Selection
# ---------------------------------------------------------------------------
def select_w_nested(
    merged_oof: pd.DataFrame,
    eval_folds: List[int],
    table: Dict[Tuple[str, str, int], int],
    target_col: str = "raw_sign_2",
    use_context_strength: bool = False,
) -> Dict[int, float]:
    """Select optimal blend weight w in {0.0, 0.1, ..., 1.0} per fold using strictly earlier rows.

    For fold k in eval_folds:
      1. Slices rows where fold < k.
      2. Drops the last 12 rows (purge gap).
      3. Tests w in grid and selects w minimizing directional binary cross-entropy log loss.
    """
    grid = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
    chosen_w: Dict[int, float] = {}

    for k in eval_folds:
        df_hist = merged_oof[merged_oof["fold"] < k].copy()
        if len(df_hist) > 12:
            df_hist = df_hist.iloc[:-12].copy()

        # Binary directional evaluation on non-zero sign
        y_hist = df_hist[target_col].values
        valid_mask = (y_hist == 1) | (y_hist == -1)
        df_fit = df_hist[valid_mask]
        y_fit = np.where(df_fit[target_col].values == 1, 1, 0)

        t_up = df_fit["ta_trend_p_up"].values
        f_pb = df_fit["ta_phase_p_pullback"].values
        p_ml = df_fit[["p_down_ml", "p_neutral_ml", "p_up_ml"]].values
        tr_s = df_fit["ta_trend_strength"].values
        ph_s = df_fit["ta_phase_strength"].values

        best_loss = np.inf
        best_w = 0.0

        for w_cand in grid:
            _, p_cand = compute_soft_blend(
                ta_trend_p_up=t_up,
                ta_phase_p_pullback=f_pb,
                p_ml=p_ml,
                table=table,
                w=w_cand,
                trend_strength=tr_s,
                phase_strength=ph_s,
                use_context_strength=use_context_strength,
            )
            # Directional probability: normalize P(up) / (P(up) + P(down))
            p_dir_up = p_cand[:, 2] / np.maximum(p_cand[:, 0] + p_cand[:, 2], 1e-12)
            p_dir_up = np.clip(p_dir_up, 1e-5, 1.0 - 1e-5)
            loss = log_loss(y_fit, p_dir_up, labels=[0, 1])

            if loss < best_loss:
                best_loss = loss
                best_w = w_cand

        chosen_w[k] = best_w

    return chosen_w

