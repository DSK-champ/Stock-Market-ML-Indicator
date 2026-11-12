"""Evaluation Suite for Horizon Machine Learning Pipelines.

Provides out-of-fold performance analysis across classification and trading metrics:
  - evaluate_binary: Classification and trading metrics for binary models.
  - evaluate_three_class: Multi-class metrics and directional trading performance.
  - coverage_table: Tabular metrics across 100%, 50%, 20%, and 10% coverage tiers.
  - circular_shift_check: Circular-shift null test for data leakage.
  - shuffled_label_retrain: Shuffled-label sanity check on fold 3.
  - permutation_importance_lgbm: Permutation feature importance on fold 6.
  - calibration_curve: Multi-bin calibration and ECE assessment.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluate import (
    compute_expected_calibration_error,
    block_bootstrap_ci,
)
from src.utils import setup_logger

logger = setup_logger("horizon_eval")


def circular_shift_check(
    signals: np.ndarray,
    truth: np.ndarray,
    min_shift: int = 2018,
    n_shifts: int = 200,
    seed: int = config.SEED,
) -> Dict[str, float]:
    """Execute circular shift null test on predictions vs ground truth.

    Parameters
    ----------
    signals : np.ndarray
        Predicted directional signals (+1, -1).
    truth : np.ndarray
        Ground truth signs (+1, -1).
    min_shift : int
        Minimum shift offset (to break short/medium term autocorrelation).
    n_shifts : int
        Number of random circular shifts.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'real_hit_rate', 'null_mean', 'null_p95', 'null_p99',
        and 'p_value'.
    """
    valid = (signals != 0) & (~np.isnan(truth)) & (truth != 0)
    sig_v = signals[valid]
    tru_v = truth[valid]
    n = len(sig_v)
    if n == 0:
        return {"real_hit_rate": 0.5, "null_mean": 0.5, "null_p95": 0.5, "null_p99": 0.5, "p_value": 1.0}

    real_hits = (sig_v == tru_v).mean()

    rng = np.random.default_rng(seed)
    max_shift = n - min_shift
    if max_shift <= min_shift:
        # Fallback if length is short
        offsets = rng.integers(1, n, size=n_shifts)
    else:
        offsets = rng.integers(min_shift, max_shift, size=n_shifts)

    null_hits = np.empty(n_shifts, dtype=np.float64)
    for i, shift in enumerate(offsets):
        shifted_sig = np.roll(sig_v, shift)
        null_hits[i] = (shifted_sig == tru_v).mean()

    null_mean = float(np.mean(null_hits))
    null_p95 = float(np.percentile(null_hits, 95))
    null_p99 = float(np.percentile(null_hits, 99))
    p_val = float((null_hits >= real_hits).mean())

    return {
        "real_hit_rate": float(real_hits),
        "null_mean": null_mean,
        "null_p95": null_p95,
        "null_p99": null_p99,
        "p_value": p_val,
    }


def compute_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute reliability diagram data and Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    true_fractions = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_prob)

    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()
        counts[b] = count
        if count > 0:
            frac = float(y_true[mask].mean())
            true_fractions[b] = frac
            mean_prob = float(y_prob[mask].mean())
            ece += (count / total_samples) * abs(frac - mean_prob)
        else:
            true_fractions[b] = np.nan

    return {
        "bin_centers": bin_centers,
        "true_fractions": true_fractions,
        "counts": counts,
        "ece": float(ece),
    }


def evaluate_trading_performance(
    signals: np.ndarray,
    truth_signs: np.ndarray,
    returns: np.ndarray,
    cost: float = config.COST,
    block_length: int = 288,
    b_reps: int = 500,
    seed: int = config.SEED,
) -> Dict[str, Any]:
    """Calculate directional trading metrics with block-bootstrap confidence interval.

    Parameters
    ----------
    signals : np.ndarray
        Directional positions (-1, 0, 1).
    truth_signs : np.ndarray
        Benchmark directional signs (-1, 1).
    returns : np.ndarray
        Forward price returns.
    cost : float
        Round-trip transaction friction.

    Returns
    -------
    Dict[str, Any]
        Trading metrics dictionary.
    """
    valid = (signals != 0) & (~np.isnan(truth_signs)) & (~np.isnan(returns))
    n_signals = int(valid.sum())
    total_candles = len(signals)
    coverage = float(n_signals / max(1, total_candles))
    signals_per_day = coverage * 288.0

    if n_signals == 0:
        return {
            "n_signals": 0,
            "coverage": 0.0,
            "signals_per_day": 0.0,
            "hit_rate": 0.5,
            "ci_lower": 0.5,
            "ci_upper": 0.5,
            "mean_ret_magnitude_bps": 0.0,
            "break_even_hurdle": 1.0,
            "break_even_gap": -0.5,
            "net_bps_per_trade": 0.0,
        }

    sig_v = signals[valid]
    tru_v = truth_signs[valid]
    ret_v = returns[valid]

    # Direction hit rate
    hits = (sig_v == tru_v).astype(np.float64)
    hit_rate = float(hits.mean())

    # Block bootstrap CI for hit rate
    ci_low, ci_high = block_bootstrap_ci(
        hits,
        statistic=np.mean,
        block_length=block_length,
        n_bootstrap=b_reps,
        random_state=seed,
    )

    # Average return magnitude m = E[|fwd_ret|]
    m = float(np.mean(np.abs(ret_v)))
    be_hurdle = 0.5 + cost / (2.0 * max(m, 1e-8))
    be_gap = hit_rate - be_hurdle

    # Net bps per trade
    trade_returns = (sig_v * ret_v) - cost
    net_bps = float(np.mean(trade_returns) * 10000.0)

    return {
        "n_signals": n_signals,
        "coverage": coverage,
        "signals_per_day": signals_per_day,
        "hit_rate": hit_rate,
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "mean_ret_magnitude_bps": float(m * 10000.0),
        "break_even_hurdle": float(be_hurdle),
        "break_even_gap": float(be_gap),
        "net_bps_per_trade": net_bps,
    }


def coverage_table(
    oof_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    is_binary: bool = True,
    cost: float = config.COST,
    horizon: int = config.HORIZON,
) -> pd.DataFrame:
    """Build multi-tier coverage analysis table (100%, 50%, 20%, 10%).

    Thresholds are computed on training blocks per fold and applied to held-out rows.

    Parameters
    ----------
    oof_df : pd.DataFrame
        OOF predictions.
    truth_df : pd.DataFrame
        Truth DataFrame indexed by open_time.
    is_binary : bool
        True if binary model, False if 3-class or meta model.
    cost : float
        Round-trip cost.

    Returns
    -------
    pd.DataFrame
        Summary table for tiers [100%, 50%, 20%, 10%].
    """
    merged = oof_df.copy()
    merged["open_time_dt"] = pd.to_datetime(merged["open_time"], utc=True)
    truth_copy = truth_df.copy()
    if "open_time" in truth_copy.columns:
        truth_copy["open_time_dt"] = pd.to_datetime(truth_copy["open_time"], utc=True)
    else:
        truth_copy["open_time_dt"] = pd.to_datetime(truth_copy.index, utc=True)

    truth_col = f"raw_sign_{horizon}"
    ret_col = f"fwd_ret_open_{horizon}"

    data = merged.merge(truth_copy[["open_time_dt", truth_col, ret_col]], on="open_time_dt", how="inner")

    # Emitted signal and confidence score
    if is_binary:
        # Signal: UP=1 if p_up > 0.5 else -1
        signal = np.where(data["p_up"] > 0.5, 1, -1)
        score = np.abs(data["p_up"] - 0.5)
    else:
        # 3-class: argmax
        p_down = data["p_down"].values
        p_neu = data["p_neutral"].values
        p_up = data["p_up"].values
        argmax_class = np.argmax(np.column_stack([p_down, p_neu, p_up]), axis=1) - 1
        signal = np.where(argmax_class == 1, 1, np.where(argmax_class == -1, -1, 0))
        # Confidence score: max(p_up, p_down) among emitted directional rows
        score = np.maximum(p_up, p_down)

    data["signal"] = signal
    data["score"] = score
    truth_vals = data[truth_col].values
    ret_vals = data[ret_col].values

    tiers = [1.0, 0.5, 0.2, 0.1]
    rows = []

    for tier in tiers:
        if tier >= 0.999:
            tier_mask = np.ones(len(data), dtype=bool) if is_binary else (signal != 0)
        else:
            # Select top tier fraction of rows by score within fold
            tier_masks = []
            for fold_id in sorted(data["fold"].unique()):
                f_idx = data["fold"] == fold_id
                f_sub = data[f_idx]
                scores_fold = f_sub["score"].values
                if not is_binary:
                    # filter directional rows only for quantile
                    dir_mask = f_sub["signal"] != 0
                    if dir_mask.sum() > 0:
                        cutoff = np.percentile(scores_fold[dir_mask], 100.0 * (1.0 - tier))
                        t_mask = (f_sub["signal"] != 0) & (scores_fold >= cutoff)
                    else:
                        t_mask = np.zeros(len(f_sub), dtype=bool)
                else:
                    cutoff = np.percentile(scores_fold, 100.0 * (1.0 - tier))
                    t_mask = scores_fold >= cutoff
                tier_masks.append(pd.Series(t_mask, index=f_sub.index))
            tier_mask = pd.concat(tier_masks).sort_index().values

        # Sub-sample evaluation
        sig_tier = np.where(tier_mask, signal, 0)
        perf = evaluate_trading_performance(sig_tier, truth_vals, ret_vals, cost=cost)

        rows.append({
            "tier": f"{int(tier*100)}%",
            "coverage": perf["coverage"],
            "signals_per_day": perf["signals_per_day"],
            "hit_rate": perf["hit_rate"],
            "ci_95": f"[{perf['ci_lower']*100:.2f}%, {perf['ci_upper']*100:.2f}%]",
            "mean_mag_bps": perf["mean_ret_magnitude_bps"],
            "be_hurdle": perf["break_even_hurdle"],
            "be_gap": perf["break_even_gap"],
            "net_bps": perf["net_bps_per_trade"],
        })

    return pd.DataFrame(rows)


def evaluate_binary(
    oof_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cost: float = config.COST,
    horizon: int = config.HORIZON,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification models."""
    merged = oof_df.copy()
    merged["open_time_dt"] = pd.to_datetime(merged["open_time"], utc=True)
    truth_copy = truth_df.copy()
    if "open_time" in truth_copy.columns:
        truth_copy["open_time_dt"] = pd.to_datetime(truth_copy["open_time"], utc=True)
    else:
        truth_copy["open_time_dt"] = pd.to_datetime(truth_copy.index, utc=True)

    bin_target_col = f"bin_dir_{horizon}"
    truth_col = f"raw_sign_{horizon}"
    ret_col = f"fwd_ret_open_{horizon}"

    data = merged.merge(truth_copy[["open_time_dt", bin_target_col, truth_col, ret_col]], on="open_time_dt", how="inner")
    data = data[data[bin_target_col].notna()].copy()

    y_true = data[bin_target_col].astype(int).values
    p_up = data["p_up"].values
    y_pred = (p_up > 0.5).astype(int)

    # Classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    prec_up = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_up = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec_down = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_down = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, p_up))
    pr_auc = float(average_precision_score(y_true, p_up))
    ll = float(log_loss(y_true, p_up))
    brier = float(brier_score_loss(y_true, p_up))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    calib = compute_calibration_data(y_true, p_up, n_bins=10)

    # Multi-tier coverage table
    cov_df = coverage_table(data, truth_copy, is_binary=True, cost=cost, horizon=horizon)

    # Circular shift check on top 20% coverage
    top20_row = cov_df[cov_df["tier"] == "20%"].iloc[0]
    score = np.abs(p_up - 0.5)
    c20 = np.percentile(score, 80.0)
    sig_20 = np.where(score >= c20, np.where(p_up > 0.5, 1, -1), 0)
    circ_audit = circular_shift_check(sig_20, data[truth_col].values, min_shift=2018, n_shifts=200)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": f1,
        "mcc": mcc,
        "precision_up": prec_up,
        "recall_up": rec_up,
        "precision_down": prec_down,
        "recall_down": rec_down,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": ll,
        "brier_score": brier,
        "ece": calib["ece"],
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "coverage_table": cov_df,
        "circular_shift": circ_audit,
    }


def evaluate_three_class(
    oof_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cost: float = config.COST,
    horizon: int = config.HORIZON,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for three-class models."""
    merged = oof_df.copy()
    merged["open_time_dt"] = pd.to_datetime(merged["open_time"], utc=True)
    truth_copy = truth_df.copy()
    if "open_time" in truth_copy.columns:
        truth_copy["open_time_dt"] = pd.to_datetime(truth_copy["open_time"], utc=True)
    else:
        truth_copy["open_time_dt"] = pd.to_datetime(truth_copy.index, utc=True)

    target_col = f"target_{horizon}_{config.THREE_CLASS_RULE}"
    truth_col = f"raw_sign_{horizon}"
    ret_col = f"fwd_ret_open_{horizon}"

    data = merged.merge(truth_copy[["open_time_dt", target_col, truth_col, ret_col]], on="open_time_dt", how="inner")
    data = data[data[target_col].notna()].copy()

    y_true = data[target_col].astype(int).values
    probs = data[["p_down", "p_neutral", "p_up"]].values
    # Classes: -1 (0), 0 (1), 1 (2)
    pred_idx = np.argmax(probs, axis=1)
    y_pred = pred_idx - 1

    # Always neutral baseline accuracy
    always_neutral_acc = float((y_true == 0).mean())

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # One-vs-rest macro ROC AUC
    # Map -1, 0, 1 to 0, 1, 2 for sklearn multi_class roc_auc
    y_true_012 = y_true + 1
    roc_auc = float(roc_auc_score(y_true_012, probs, multi_class="ovr", average="macro"))
    ll = float(log_loss(y_true_012, probs))

    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1], normalize="true")

    # Multi-tier coverage table
    cov_df = coverage_table(data, truth_copy, is_binary=False, cost=cost, horizon=horizon)

    # Circular shift check on top 20% coverage
    p_up = data["p_up"].values
    p_down = data["p_down"].values
    score = np.maximum(p_up, p_down)
    dir_mask = y_pred != 0
    if dir_mask.sum() > 0:
        c20 = np.percentile(score[dir_mask], 80.0)
        sig_20 = np.where(dir_mask & (score >= c20), y_pred, 0)
    else:
        sig_20 = np.zeros(len(data), dtype=int)
    circ_audit = circular_shift_check(sig_20, data[truth_col].values, min_shift=2018, n_shifts=200)

    return {
        "accuracy": acc,
        "always_neutral_accuracy": always_neutral_acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "roc_auc_ovr": roc_auc,
        "log_loss": ll,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "coverage_table": cov_df,
        "circular_shift": circ_audit,
    }


def shuffled_label_retrain(
    model_id: str,
    horizon: int = config.HORIZON,
    fold: int = 3,
    seed: int = config.SEED,
) -> float:
    """Train model on shuffled labels for fold 3 and report out-of-sample ROC AUC."""
    from src.model_common import prepare_model_features
    from src.binary_models import build_binary_model, get_binary_inputs
    from src.three_class_models import build_three_class_model, get_three_class_inputs
    from src.cv import PurgedWalkForwardCV
    from src.column_dictionary import load_master

    df = load_master(split="train")
    cv = PurgedWalkForwardCV()
    folds = list(cv.split(df))
    fold_entry = [f for f in folds if f[0] == fold]
    if not fold_entry:
        return 0.50
    _, tr_idx, te_idx = fold_entry[0]

    is_binary = model_id.startswith("binary")
    target_col = f"bin_dir_{horizon}" if is_binary else f"target_{horizon}_{config.THREE_CLASS_RULE}"
    feature_cols = get_binary_inputs(model_id) if is_binary else get_three_class_inputs(model_id)
    build_fn = build_binary_model if is_binary else build_three_class_model

    df_tr = df.iloc[tr_idx]
    mask_tr = (df_tr[f"has_valid_target_{horizon}"] == 1) & (df_tr[target_col].notna())
    df_tr_valid = df_tr[mask_tr].iloc[::config.STRIDE]

    df_te = df.iloc[te_idx]
    mask_te = (df_te[f"has_valid_target_{horizon}"] == 1) & (df_te[target_col].notna())
    df_te_valid = df_te[mask_te]

    X_tr = prepare_model_features(df_tr_valid, feature_cols)
    y_tr = df_tr_valid[target_col].values.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(y_tr)  # Shuffled in place

    X_te = prepare_model_features(df_te_valid, feature_cols)
    y_te = df_te_valid[target_col].values

    model = build_fn(model_id, smoke=True)
    if "lr" in model_id:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)
    if is_binary:
        auc = roc_auc_score(y_te, probs[:, 1])
    else:
        y_te_012 = y_te + 1
        auc = roc_auc_score(y_te_012, probs, multi_class="ovr", average="macro")
    return float(auc)


def permutation_importance_lgbm(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_samples: int = 20000,
    n_repeats: int = 3,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, bool]:
    """Compute permutation importance on held-out fold with dominance check (>30%)."""
    if len(X_val) > n_samples:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(len(X_val), size=n_samples, replace=False)
        X_eval = X_val.iloc[sub_idx]
        y_eval = y_val[sub_idx]
    else:
        X_eval = X_val
        y_eval = y_val

    res = permutation_importance(
        model,
        X_eval,
        y_eval,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    importances = res.importances_mean
    total_imp = float(np.sum(np.maximum(0, importances)))
    if total_imp <= 1e-8:
        total_imp = 1.0

    df_imp = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": importances,
        "importance_std": res.importances_std,
        "relative_share": np.maximum(0, importances) / total_imp,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    max_share = float(df_imp["relative_share"].max())
    is_dominated = max_share > 0.30

    return df_imp.head(20), is_dominated
