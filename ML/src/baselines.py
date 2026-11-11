"""Benchmark baselines for directional and clear-move forecasting.

Covers 12 canonical baselines evaluated on identical 6-fold Purged Walk-Forward CV:
  1. Always-Up: Long (+1) on every candle.
  2. Majority-Class: Predict mode of training block.
  3. Fixed-Seed Coin Flip: Random guess in {-1, +1} (seed 42).
  4. Momentum (1-candle): sign(feat_ret_1).
  5. Reverse Momentum (1-candle mean reversion): -sign(feat_ret_1).
  6. RSI(14) Rule: Long (+1) if RSI < 30 (oversold), Short (-1) if RSI > 70 (overbought), else 0.
  7. Reverse RSI(14): Short (-1) if RSI < 30, Long (+1) if RSI > 70, else 0.
  8. Location 288 Rule: sign(0.5 - feat_loc_position_288).
  9. EMA50 Rule: sign(EMA50 - close) = -sign(feat_mom_ema_ratio_50 - 1.0).
  10. 4b Rule Vote: Heuristic ensemble (+1 if Dow Up, -1 if Dow Down) and reverse.
      (Flagged: 'chosen after seeing train results').
  11. Volatility-Only Baseline: Logistic Regression on feat_vol_sigma_288.
  12. 5-Feature Reference Model: Logistic Regression on
      {feat_ret_1, feat_ret_3, feat_ret_12, feat_mom_rsi_14, feat_loc_position_288}.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.cv import PurgedWalkForwardCV
from src.evaluate import (
    compute_classification_metrics,
    compute_probabilistic_metrics,
    compute_operational_metrics,
)
from src.io_utils import save_oof_predictions
from src.utils import setup_logger

logger = setup_logger("baselines")

ROUNDTRIP_COST: float = getattr(config, "TOTAL_ROUNDTRIP_COST", 0.0023)
REF_5_FEATURES: List[str] = [
    "feat_ret_1",
    "feat_ret_3",
    "feat_ret_12",
    "feat_mom_rsi_14",
    "feat_loc_position_288",
]


# ===========================================================================
# 1. Individual Baseline Prediction Rules
# ===========================================================================

def rule_always_up(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Always Up (+1). Returns (pred_direction, prob_up)."""
    n = len(df)
    return np.ones(n, dtype=float), np.ones(n, dtype=float)


def rule_majority_class(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict training block's mode direction."""
    train_vals = df_train[target_col].dropna().values
    if len(train_vals) == 0:
        mode_val = 1.0
        prob = 0.5
    else:
        vals, counts = np.unique(train_vals, return_counts=True)
        mode_val = float(vals[np.argmax(counts)])
        prob = float(np.max(counts) / len(train_vals))

    n_test = len(df_test)
    preds = np.full(n_test, mode_val, dtype=float)
    probs = np.full(n_test, prob, dtype=float)
    return preds, probs


def rule_coin_flip(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Random coin flip in {-1.0, +1.0} with fixed seed."""
    rng = np.random.RandomState(seed)
    n = len(df)
    preds = rng.choice([-1.0, 1.0], size=n)
    probs = np.full(n, 0.5, dtype=float)
    return preds, probs


def rule_momentum(df: pd.DataFrame, reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Momentum: sign(feat_ret_1). Reverse: -sign(feat_ret_1)."""
    ret = df["feat_ret_1"].fillna(0.0).values
    preds = np.sign(ret)
    # Zero returns abstain (0.0)
    if reverse:
        preds = -preds
    # Pseudo-confidence based on rank or flat 0.5
    probs = np.where(preds > 0, 0.55, np.where(preds < 0, 0.45, 0.5))
    return preds, probs


def rule_rsi_14(df: pd.DataFrame, reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """RSI(14): +1 if RSI < 30, -1 if RSI > 70, 0 otherwise."""
    rsi = df["feat_mom_rsi_14"].fillna(50.0).values
    preds = np.zeros(len(df), dtype=float)
    preds[rsi < 30.0] = 1.0
    preds[rsi > 70.0] = -1.0
    if reverse:
        preds = -preds
    probs = np.where(preds > 0, 0.60, np.where(preds < 0, 0.40, 0.5))
    return preds, probs


def rule_location_288(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Location 288 mean reversion: sign(0.5 - feat_loc_position_288)."""
    pos = df["feat_loc_position_288"].fillna(0.5).values
    preds = np.sign(0.5 - pos)
    probs = np.where(preds > 0, 0.55, np.where(preds < 0, 0.45, 0.5))
    return preds, probs


def rule_ema50(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """EMA50 rule: sign(EMA50 - close) = -sign(feat_mom_ema_ratio_50 - 1.0)."""
    ratio = df["feat_mom_ema_ratio_50"].fillna(1.0).values
    preds = -np.sign(ratio - 1.0)
    probs = np.where(preds > 0, 0.55, np.where(preds < 0, 0.45, 0.5))
    return preds, probs


def rule_4b_rule_vote(df: pd.DataFrame, reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """4b Rule Vote heuristic (+1 if Dow Up & level rejected from below, -1 if Dow Down & level rejected from above)."""
    if "feat_struct_rule_vote" in df.columns:
        vote = df["feat_struct_rule_vote"].fillna(0.0).values
    elif "feat_struct_trend_up" in df.columns and "feat_struct_trend_down" in df.columns:
        vote = (df["feat_struct_trend_up"].fillna(0.0) - df["feat_struct_trend_down"].fillna(0.0)).values
    else:
        vote = np.zeros(len(df), dtype=float)

    preds = np.sign(vote)
    if reverse:
        preds = -preds
    probs = np.where(preds > 0, 0.55, np.where(preds < 0, 0.45, 0.5))
    return preds, probs


# ===========================================================================
# 2. Trainable Reference Linear Models
# ===========================================================================

def train_and_eval_volatility_lr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-feature Logistic Regression on feat_vol_sigma_288."""
    feature = "feat_vol_sigma_288"
    X_train = df_train[[feature]].fillna(0.0).values
    y_train = df_train[target_col].dropna()
    valid_train_idx = y_train.index
    X_train = df_train.loc[valid_train_idx, [feature]].fillna(0.0).values
    y_train = y_train.values

    # Handle binary vs multiclass
    unique_y = np.unique(y_train)
    if len(unique_y) < 2:
        return np.ones(len(df_test)), np.full(len(df_test), 0.5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(df_test[[feature]].fillna(0.0).values)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    probs_matrix = clf.predict_proba(X_test_scaled)
    # Return prob of positive/up class
    if len(clf.classes_) == 2:
        prob_up = probs_matrix[:, 1]
    else:
        # Multiclass: up class is last class (+1)
        up_idx = np.where(clf.classes_ == 1.0)[0]
        prob_up = probs_matrix[:, up_idx[0]] if len(up_idx) > 0 else probs_matrix[:, -1]

    return preds.astype(float), prob_up


def train_and_eval_5feature_lr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """5-feature reference Logistic Regression."""
    cols = REF_5_FEATURES
    y_train_series = df_train[target_col].dropna()
    valid_idx = y_train_series.index
    X_train = df_train.loc[valid_idx, cols].fillna(0.0).values
    y_train = y_train_series.values

    unique_y = np.unique(y_train)
    if len(unique_y) < 2:
        return np.ones(len(df_test)), np.full(len(df_test), 0.5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(df_test[cols].fillna(0.0).values)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    probs_matrix = clf.predict_proba(X_test_scaled)
    if len(clf.classes_) == 2:
        prob_up = probs_matrix[:, 1]
    else:
        up_idx = np.where(clf.classes_ == 1.0)[0]
        prob_up = probs_matrix[:, up_idx[0]] if len(up_idx) > 0 else probs_matrix[:, -1]

    return preds.astype(float), prob_up


# ===========================================================================
# 3. Comprehensive Baseline Suite Runner
# ===========================================================================

BASELINES_LIST = [
    ("always_up", "Always-Up (predict +1)", False),
    ("majority_class", "Majority Class (mode of train block)", False),
    ("coin_flip", "Coin Flip (Bernoulli 0.5, seed 42)", False),
    ("momentum", "Momentum (sign of feat_ret_1)", False),
    ("reverse_momentum", "Reverse Momentum (-sign of feat_ret_1)", False),
    ("rsi_14", "RSI(14) Rule (<30 Long, >70 Short)", False),
    ("reverse_rsi_14", "Reverse RSI(14) (<30 Short, >70 Long)", False),
    ("loc_position_288", "Location 288 Mean Reversion (sign(0.5 - pos))", False),
    ("ema50_rule", "EMA50 Rule (sign(EMA50 - close))", False),
    ("rule_vote_4b", "4b Heuristic Rule Vote", True),
    ("reverse_rule_vote_4b", "Reverse 4b Rule Vote", True),
    ("volatility_lr", "Volatility-Only LR (feat_vol_sigma_288)", False),
    ("ref_5feature_lr", "5-Feature Reference LR", False),
]


def run_baseline(
    name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute prediction rule for a specific baseline."""
    if name == "always_up":
        return rule_always_up(df_test)
    elif name == "majority_class":
        return rule_majority_class(df_train, df_test, target_col)
    elif name == "coin_flip":
        return rule_coin_flip(df_test, seed=42)
    elif name == "momentum":
        return rule_momentum(df_test, reverse=False)
    elif name == "reverse_momentum":
        return rule_momentum(df_test, reverse=True)
    elif name == "rsi_14":
        return rule_rsi_14(df_test, reverse=False)
    elif name == "reverse_rsi_14":
        return rule_rsi_14(df_test, reverse=True)
    elif name == "loc_position_288":
        return rule_location_288(df_test)
    elif name == "ema50_rule":
        return rule_ema50(df_test)
    elif name in ("rule_vote_4b", "dow_trend_4b"):
        return rule_4b_rule_vote(df_test, reverse=False)
    elif name in ("reverse_rule_vote_4b", "reverse_dow_trend_4b"):
        return rule_4b_rule_vote(df_test, reverse=True)
    elif name == "volatility_lr":
        return train_and_eval_volatility_lr(df_train, df_test, target_col)
    elif name == "ref_5feature_lr":
        return train_and_eval_5feature_lr(df_train, df_test, target_col)
    else:
        raise ValueError(f"Unknown baseline name: {name}")


def evaluate_all_baselines(
    df_train_full: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    save_oof: bool = True,
    save_report_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Evaluate all 12 baselines across all 4 candidate horizons on 6-fold CV.

    Parameters
    ----------
    df_train_full : pd.DataFrame
        Training split of master dataset (220,953 rows).
    horizons : Optional[List[int]]
        List of target horizons (default [2, 3, 6, 12]).
    save_oof : bool
        Whether to cache OOF predictions to results/oof/.
    save_report_path : Optional[Union[str, Path]]
        Destination path for baselines_oof_summary.csv.

    Returns
    -------
    pd.DataFrame
        Structured comparison table of all baselines across horizons.
    """
    if horizons is None:
        horizons = config.CANDIDATE_HORIZONS

    cv = PurgedWalkForwardCV()
    records: List[Dict[str, Any]] = []

    for N in horizons:
        raw_sign_col = f"raw_sign_{N}"
        fwd_ret_col = f"fwd_ret_{N}"
        target_col = f"raw_sign_{N}"

        logger.info("Evaluating baselines for horizon N=%d (%d minutes)...", N, N * 5)

        for b_name, b_desc, chosen_after_train in BASELINES_LIST:
            oof_preds = []
            oof_probs_up = []
            oof_indices = []
            oof_folds = []

            for fold, tr_idx, te_idx in cv.split(df_train_full):
                df_tr = df_train_full.iloc[tr_idx]
                df_te = df_train_full.iloc[te_idx]

                preds, prob_up = run_baseline(b_name, df_tr, df_te, target_col)

                oof_preds.append(preds)
                oof_probs_up.append(prob_up)
                oof_indices.append(te_idx)
                oof_folds.append(np.full(len(te_idx), fold, dtype=int))

            all_preds = np.concatenate(oof_preds)
            all_probs_up = np.concatenate(oof_probs_up)
            all_te_idx = np.concatenate(oof_indices)
            all_folds = np.concatenate(oof_folds)

            df_eval = df_train_full.iloc[all_te_idx]
            y_true_sign = df_eval[raw_sign_col].values
            fwd_ret = df_eval[fwd_ret_col].values

            # Binary / directional evaluation against raw_sign
            # Map predictions to {-1, +1} or 0
            sig_dir = np.sign(all_preds)

            # Filter valid (non-nan) ground truth
            valid_mask = np.isfinite(y_true_sign) & np.isfinite(fwd_ret) & (y_true_sign != 0.0)

            # Operational metrics
            op_res = compute_operational_metrics(
                y_true_sign=y_true_sign,
                y_pred_dir=sig_dir,
                fwd_ret=fwd_ret,
                cost=ROUNDTRIP_COST,
                coverage_mask=valid_mask,
            )

            # Classification metrics on emitted non-neutral signals
            emitted_mask = valid_mask & (sig_dir != 0.0)
            if emitted_mask.sum() > 0:
                y_true_bin = np.where(y_true_sign[emitted_mask] > 0, 1, 0)
                y_pred_bin = np.where(sig_dir[emitted_mask] > 0, 1, 0)
                prob_emitted = all_probs_up[emitted_mask]
                cls_res = compute_classification_metrics(y_true_bin, y_pred_bin, prob_emitted)
                prob_res = compute_probabilistic_metrics(y_true_bin, prob_emitted)
            else:
                cls_res = {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0, "mcc": 0.0}
                prob_res = {"log_loss": float("nan"), "brier_score": float("nan")}

            rec = {
                "baseline": b_name,
                "description": b_desc,
                "horizon": N,
                "horizon_min": N * 5,
                "chosen_after_seeing_train": chosen_after_train,
                "coverage": op_res["coverage"],
                "signals_per_day": op_res["signals_per_day"],
                "n_signals": op_res["n_signals"],
                "accuracy": cls_res["accuracy"],
                "balanced_accuracy": cls_res["balanced_accuracy"],
                "f1_macro": cls_res["f1_macro"],
                "mcc": cls_res["mcc"],
                "brier_score": prob_res["brier_score"],
                "hit_rate": op_res["hit_rate"],
                "m_fwd_ret": op_res["m_fwd_ret"],
                "break_even_rate": op_res["break_even_rate"],
                "break_even_gap": op_res["break_even_gap"],
                "net_return_bps": op_res["net_return_bps"],
            }
            records.append(rec)

            # Cache OOF predictions
            if save_oof:
                oof_df = pd.DataFrame({
                    "open_time": df_eval.index,
                    "fold": all_folds,
                    "model": f"baseline_{b_name}",
                    "setup": "setup_b",
                    "horizon": N,
                    "rule": "raw_sign",
                    "prob_down": 1.0 - all_probs_up,
                    "prob_neutral": 0.0,
                    "prob_up": all_probs_up,
                    "pred_label": sig_dir,
                    "true_label": y_true_sign,
                })
                save_oof_predictions(
                    df=oof_df,
                    model_name=f"baseline_{b_name}",
                    setup="setup_b",
                    horizon=N,
                    rule="raw_sign",
                )

    res_df = pd.DataFrame(records)

    if save_report_path is not None:
        p = Path(save_report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(p, index=False)
        logger.info("Saved baselines summary report to %s", p)

    return res_df
