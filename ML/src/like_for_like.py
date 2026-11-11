"""Like-for-like comparison engine across fixed coverage levels (100%, 50%, 20%, 10%).

Ensures fair, rigorous benchmarking between baselines and trained ML models:
1. Coverages: 100%, 50%, 20%, 10% (the most confident or most extreme rows).
2. Thresholds: Strictly fixed on training folds (never on held-out test folds).
3. Primary evaluation metric: Hit rate against raw_sign_N with row-level block-bootstrap 95% CI (B >= 288).
4. Friction & Hurdle metrics:
   - Mean absolute forward return of selected rows: E[|fwd_ret_open_N|]
   - Dynamic break-even hit rate: BE = c / (2 * E[|fwd_ret_open_N|]) + 0.5 (c = 23 bps)
   - Break-even gap: hit_rate - BE
   - Net return in bps: 10,000 * [hit_rate * E[|fwd_ret|] - (1 - hit_rate) * E[|fwd_ret|] - c]
   - Signals per day: coverage * 288
5. Same-coverage benchmark rules:
   - (a) Random selection with sign(0.5 - feat_loc_position_288).
   - (b) Volatility-gated reversal rule: top bars by feat_vol_natr_14, direction sign(0.5 - feat_loc_position_288).
6. 20% Coverage Gap Decomposition:
   gap = (hit - 0.5) - (BE - 0.5)
   - Direction Skill: (hit - 0.5)
   - Size Selection (Hurdle Penalty): (BE - 0.5) = c / (2 * E[|fwd_ret|])
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.cv import PurgedWalkForwardCV
from src.utils import setup_logger

logger = setup_logger("like_for_like")

ROUNDTRIP_COST: float = getattr(config, "TOTAL_ROUNDTRIP_COST", 0.0023)
BLOCK_LENGTH: int = 288
DEFAULT_COVERAGES: List[float] = [1.0, 0.5, 0.2, 0.1]


# ===========================================================================
# 1. Fast Vectorized Block-Bootstrap Confidence Intervals
# ===========================================================================

def compute_block_bootstrap_ci(
    correct: np.ndarray,
    block_length: int = BLOCK_LENGTH,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute empirical block-bootstrap 95% confidence interval for hit rate.

    Parameters
    ----------
    correct : np.ndarray
        1D boolean or binary array indicating signal correctness (1 for correct, 0 for wrong).
    block_length : int
        Consecutive candle block size (default 288 = 1 full trading day).
    n_boot : int
        Number of bootstrap replications (default 1,000).
    seed : int
        Random seed for reproducibility (default 42).
    alpha : float
        Significance level (default 0.05 -> 95% two-sided CI).

    Returns
    -------
    Tuple[float, float, float]
        (mean_hit_rate, ci_lower, ci_upper)
    """
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean_val = float(np.mean(correct))
    if n <= block_length:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))

    # Fast block resampling
    max_start = n - block_length
    start_indices = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    block_offsets = np.arange(block_length)
    # Shape: (n_boot, n_blocks, block_length) -> flatten to (n_boot, n_blocks * block_length)
    sample_indices = (start_indices[:, :, None] + block_offsets).reshape(n_boot, -1)[:, :n]

    boot_means = np.mean(correct[sample_indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_val, ci_lower, ci_upper


# ===========================================================================
# 2. Operational Metrics Evaluator at Specified Subset
# ===========================================================================

def evaluate_subset_metrics(
    y_true_sign: np.ndarray,
    y_pred_dir: np.ndarray,
    fwd_ret_open: np.ndarray,
    cost: float = ROUNDTRIP_COST,
    block_length: int = BLOCK_LENGTH,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute financial and directional metrics on a selected subset of rows."""
    valid = np.isfinite(y_true_sign) & np.isfinite(fwd_ret_open) & (y_true_sign != 0.0) & (y_pred_dir != 0.0)
    n_signals = int(valid.sum())

    if n_signals == 0:
        return {
            "n_signals": 0,
            "hit_rate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "mean_abs_fwd_ret": float("nan"),
            "break_even_rate": float("nan"),
            "break_even_gap": float("nan"),
            "net_bps": float("nan"),
            "direction_skill": float("nan"),
            "size_selection_hurdle": float("nan"),
        }

    sub_true = y_true_sign[valid]
    sub_pred = y_pred_dir[valid]
    sub_ret = np.abs(fwd_ret_open[valid])

    correct = (sub_true == sub_pred).astype(float)
    hit_rate, ci_lower, ci_upper = compute_block_bootstrap_ci(
        correct, block_length=block_length, n_boot=n_boot, seed=seed
    )

    m_abs_ret = float(np.mean(sub_ret))
    if m_abs_ret > 0:
        be_rate = float(cost / (2.0 * m_abs_ret) + 0.5)
    else:
        be_rate = float("inf")

    be_gap = hit_rate - be_rate
    net_bps = float(10000.0 * (hit_rate * m_abs_ret - (1.0 - hit_rate) * m_abs_ret - cost))

    direction_skill = hit_rate - 0.5
    size_selection_hurdle = be_rate - 0.5

    return {
        "n_signals": n_signals,
        "hit_rate": hit_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_abs_fwd_ret": m_abs_ret,
        "break_even_rate": be_rate,
        "break_even_gap": be_gap,
        "net_bps": net_bps,
        "direction_skill": direction_skill,
        "size_selection_hurdle": size_selection_hurdle,
    }


# ===========================================================================
# 3. Model & Baseline Prediction / Score Extractors
# ===========================================================================

def get_signals_and_scores(
    model_name: str,
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    horizon: int = 12,
    linear_features: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract directional predictions and confidence/extremity scores for train and test splits.

    Returns:
        (tr_preds, tr_scores, te_preds, te_scores)
    """
    from src.baselines import rule_rsi_14, rule_ema50, rule_4b_rule_vote, REF_5_FEATURES

    if model_name == "loc_position_288":
        tr_pos = df_tr["feat_loc_position_288"].fillna(0.5).values
        te_pos = df_te["feat_loc_position_288"].fillna(0.5).values
        tr_preds = np.sign(0.5 - tr_pos)
        te_preds = np.sign(0.5 - te_pos)
        tr_scores = np.abs(0.5 - tr_pos)
        te_scores = np.abs(0.5 - te_pos)

    elif model_name == "vol_gated_reversal":
        # Top bars by feat_vol_natr_14, direction sign(0.5 - feat_loc_position_288)
        tr_preds = np.sign(0.5 - df_tr["feat_loc_position_288"].fillna(0.5).values)
        te_preds = np.sign(0.5 - df_te["feat_loc_position_288"].fillna(0.5).values)
        tr_scores = df_tr["feat_vol_natr_14"].fillna(0.0).values
        te_scores = df_te["feat_vol_natr_14"].fillna(0.0).values

    elif model_name == "random_loc_reversal":
        rng_tr = np.random.default_rng(seed)
        rng_te = np.random.default_rng(seed + 100)
        tr_preds = np.sign(0.5 - df_tr["feat_loc_position_288"].fillna(0.5).values)
        te_preds = np.sign(0.5 - df_te["feat_loc_position_288"].fillna(0.5).values)
        tr_scores = rng_tr.uniform(0.0, 1.0, size=len(df_tr))
        te_scores = rng_te.uniform(0.0, 1.0, size=len(df_te))

    elif model_name == "rsi_14":
        tr_preds, _ = rule_rsi_14(df_tr, reverse=False)
        te_preds, _ = rule_rsi_14(df_te, reverse=False)
        tr_scores = np.abs(0.5 - df_tr["feat_mom_rsi_14"].fillna(0.5).values)
        te_scores = np.abs(0.5 - df_te["feat_mom_rsi_14"].fillna(0.5).values)

    elif model_name == "reverse_rsi_14":
        tr_preds, _ = rule_rsi_14(df_tr, reverse=True)
        te_preds, _ = rule_rsi_14(df_te, reverse=True)
        tr_scores = np.abs(0.5 - df_tr["feat_mom_rsi_14"].fillna(0.5).values)
        te_scores = np.abs(0.5 - df_te["feat_mom_rsi_14"].fillna(0.5).values)

    elif model_name == "ema50_rule":
        tr_preds, _ = rule_ema50(df_tr)
        te_preds, _ = rule_ema50(df_te)
        tr_scores = np.abs(df_tr["feat_mom_ema_ratio_50"].fillna(0.0).values)
        te_scores = np.abs(df_te["feat_mom_ema_ratio_50"].fillna(0.0).values)

    elif model_name == "momentum":
        tr_ret = df_tr["feat_ret_1"].fillna(0.0).values
        te_ret = df_te["feat_ret_1"].fillna(0.0).values
        tr_preds = np.sign(tr_ret)
        te_preds = np.sign(te_ret)
        tr_scores = np.abs(tr_ret)
        te_scores = np.abs(te_ret)

    elif model_name == "reverse_momentum":
        tr_ret = df_tr["feat_ret_1"].fillna(0.0).values
        te_ret = df_te["feat_ret_1"].fillna(0.0).values
        tr_preds = -np.sign(tr_ret)
        te_preds = -np.sign(te_ret)
        tr_scores = np.abs(tr_ret)
        te_scores = np.abs(te_ret)

    elif model_name == "rule_vote_4b":
        tr_preds, _ = rule_4b_rule_vote(df_tr, reverse=False)
        te_preds, _ = rule_4b_rule_vote(df_te, reverse=False)
        vote_col = "feat_struct_rule_vote" if "feat_struct_rule_vote" in df_tr.columns else "feat_struct_trend_up"
        tr_scores = np.abs(df_tr[vote_col].fillna(0.0).values)
        te_scores = np.abs(df_te[vote_col].fillna(0.0).values)

    elif model_name == "reverse_rule_vote_4b":
        tr_preds, _ = rule_4b_rule_vote(df_tr, reverse=True)
        te_preds, _ = rule_4b_rule_vote(df_te, reverse=True)
        vote_col = "feat_struct_rule_vote" if "feat_struct_rule_vote" in df_tr.columns else "feat_struct_trend_up"
        tr_scores = np.abs(df_tr[vote_col].fillna(0.0).values)
        te_scores = np.abs(df_te[vote_col].fillna(0.0).values)

    elif model_name == "ref_5feature_lr":
        cols = REF_5_FEATURES
        y_tr = df_tr[f"raw_sign_{horizon}"].dropna()
        v_idx = y_tr.index
        X_tr = df_tr.loc[v_idx, cols].fillna(0.0).values
        y_tr_vals = np.where(y_tr.values > 0, 1, 0)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(df_te[cols].fillna(0.0).values)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr_vals)

        tr_prob = clf.predict_proba(scaler.transform(df_tr[cols].fillna(0.0).values))[:, 1]
        te_prob = clf.predict_proba(X_te_s)[:, 1]

        tr_preds = np.where(tr_prob >= 0.5, 1.0, -1.0)
        te_preds = np.where(te_prob >= 0.5, 1.0, -1.0)
        tr_scores = np.abs(tr_prob - 0.5)
        te_scores = np.abs(te_prob - 0.5)

    elif model_name == "always_up":
        tr_preds = np.ones(len(df_tr), dtype=float)
        te_preds = np.ones(len(df_te), dtype=float)
        tr_scores = np.ones(len(df_tr), dtype=float)
        te_scores = np.ones(len(df_te), dtype=float)

    elif model_name == "always_down":
        tr_preds = np.full(len(df_tr), -1.0, dtype=float)
        te_preds = np.full(len(df_te), -1.0, dtype=float)
        tr_scores = np.ones(len(df_tr), dtype=float)
        te_scores = np.ones(len(df_te), dtype=float)

    elif model_name == "coin_flip":
        rng_tr = np.random.default_rng(seed)
        rng_te = np.random.default_rng(seed + 1)
        tr_preds = rng_tr.choice([-1.0, 1.0], size=len(df_tr))
        te_preds = rng_te.choice([-1.0, 1.0], size=len(df_te))
        tr_scores = rng_tr.uniform(0.0, 1.0, size=len(df_tr))
        te_scores = rng_te.uniform(0.0, 1.0, size=len(df_te))

    elif model_name == "majority_class":
        mode_val = float(df_tr[f"raw_sign_{horizon}"].mode().iloc[0])
        tr_preds = np.full(len(df_tr), mode_val, dtype=float)
        te_preds = np.full(len(df_te), mode_val, dtype=float)
        tr_scores = np.ones(len(df_tr), dtype=float)
        te_scores = np.ones(len(df_te), dtype=float)

    elif model_name == "B_12" or (model_name.startswith("B_")):
        # Setup B: Logistic Regression on LINEAR_FEATURES predicting binary raw sign
        cols = linear_features
        target_col = f"raw_sign_{horizon}"
        y_tr = df_tr[target_col].dropna()
        v_idx = y_tr.index
        X_tr = df_tr.loc[v_idx, cols].fillna(0.0).values
        y_tr_vals = np.where(y_tr.values > 0, 1, 0)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(df_te[cols].fillna(0.0).values)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr_vals)

        tr_prob = clf.predict_proba(scaler.transform(df_tr[cols].fillna(0.0).values))[:, 1]
        te_prob = clf.predict_proba(X_te_s)[:, 1]

        tr_preds = np.where(tr_prob >= 0.5, 1.0, -1.0)
        te_preds = np.where(te_prob >= 0.5, 1.0, -1.0)
        tr_scores = np.abs(tr_prob - 0.5)
        te_scores = np.abs(te_prob - 0.5)

    elif model_name.startswith("C_"):
        # Setup C: 3-class Logistic Regression on LINEAR_FEATURES predicting target_N_rule
        # e.g. C_12_F15 -> rule = F15
        parts = model_name.split("_")
        rule = parts[2]
        target_col = f"target_{horizon}_{rule}"

        cols = linear_features
        y_tr = df_tr[target_col].dropna()
        v_idx = y_tr.index
        X_tr = df_tr.loc[v_idx, cols].fillna(0.0).values
        y_tr_vals = y_tr.values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(df_te[cols].fillna(0.0).values)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr_vals)

        tr_preds = clf.predict(scaler.transform(df_tr[cols].fillna(0.0).values))
        te_preds = clf.predict(X_te_s)

        tr_probs = clf.predict_proba(scaler.transform(df_tr[cols].fillna(0.0).values))
        te_probs = clf.predict_proba(X_te_s)

        # Confidence: max active probability P(Up) or P(Down)
        classes = list(clf.classes_)
        active_indices = [i for i, c in enumerate(classes) if c in (-1.0, 1.0)]
        if len(active_indices) > 0:
            tr_scores = np.max(tr_probs[:, active_indices], axis=1)
            te_scores = np.max(te_probs[:, active_indices], axis=1)
        else:
            tr_scores = np.max(tr_probs, axis=1)
            te_scores = np.max(te_probs, axis=1)

    else:
        raise ValueError(f"Unknown model name for like-for-like: {model_name}")

    return tr_preds, tr_scores, te_preds, te_scores


# ===========================================================================
# 4. Master Like-for-Like Multi-Coverage Evaluator
# ===========================================================================

def run_like_for_like_suite(
    df_train_full: pd.DataFrame,
    horizon: int = 12,
    coverages: Optional[List[float]] = None,
    linear_features: Optional[List[str]] = None,
    save_report_path: Optional[Union[str, Path]] = None,
    save_decomp_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete like-for-like sweep across declared coverages.

    Models evaluated:
      - Baselines: loc_position_288, vol_gated_reversal, random_loc_reversal,
                   rsi_14, reverse_rsi_14, ema50_rule, momentum, reverse_momentum,
                   rule_vote_4b, reverse_rule_vote_4b, ref_5feature_lr,
                   always_up, always_down, coin_flip, majority_class
      - Stage 1 Survivors: C_12_F15, C_12_F10, C_12_F05, B_12
    """
    if coverages is None:
        coverages = DEFAULT_COVERAGES

    if linear_features is None:
        from src.config import LINEAR_FEATURES
        linear_features = list(LINEAR_FEATURES)

    cv = PurgedWalkForwardCV()

    models_to_run = [
        # Primary benchmark and same-coverage rules
        ("loc_position_288", "Location 288 Mean Reversion (sign(0.5 - pos))"),
        ("vol_gated_reversal", "Volatility-Gated Reversal Rule (top NATR_14, sign(0.5 - pos))"),
        ("random_loc_reversal", "Random Selection Reversal Rule (sign(0.5 - pos))"),
        # Technical baselines
        ("rsi_14", "RSI(14) Rule (<0.30 Long, >0.70 Short)"),
        ("reverse_rsi_14", "Reverse RSI(14) (<0.30 Short, >0.70 Long)"),
        ("ema50_rule", "EMA50 Rule (sign(EMA50 - close))"),
        ("momentum", "Momentum (sign of feat_ret_1)"),
        ("reverse_momentum", "Reverse Momentum (-sign of feat_ret_1)"),
        ("rule_vote_4b", "4b Heuristic Rule Vote"),
        ("reverse_rule_vote_4b", "Reverse 4b Rule Vote"),
        ("ref_5feature_lr", "5-Feature Reference LR"),
        ("always_up", "Always-Up (predict +1)"),
        ("always_down", "Always-Down (predict -1)"),
        ("coin_flip", "Coin Flip Binary (Bernoulli 0.5)"),
        ("majority_class", "Majority Class (mode of train)"),
        # Stage 1 survivors & fallback
        ("B_12", "Stage 1 Setup B Fallback (LR on 70 features, raw_sign_12)"),
        ("C_12_F15", "Stage 1 Setup C Survivor (LR on 70 features, target_12_F15)"),
        ("C_12_F10", "Stage 1 Setup C Survivor (LR on 70 features, target_12_F10)"),
        ("C_12_F05", "Stage 1 Setup C Survivor (LR on 70 features, target_12_F05)"),
    ]

    records = []
    decomp_records = []

    raw_sign_col = f"raw_sign_{horizon}"
    fwd_ret_open_col = f"fwd_ret_open_{horizon}"

    logger.info("Starting like-for-like evaluation suite at horizon N=%d...", horizon)

    for m_name, m_desc in models_to_run:
        logger.info("Evaluating model: %s...", m_name)

        # Collect fold predictions and thresholded selections
        oof_preds = []
        oof_te_indices = []
        # coverage -> list of selected boolean arrays per fold
        cov_selections: Dict[float, List[np.ndarray]] = {c: [] for c in coverages}

        for fold, tr_idx, te_idx in cv.split(df_train_full):
            df_tr = df_train_full.iloc[tr_idx]
            df_te = df_train_full.iloc[te_idx]

            tr_p, tr_s, te_p, te_s = get_signals_and_scores(
                m_name, df_tr, df_te, horizon=horizon, linear_features=linear_features, seed=42 + fold
            )

            oof_preds.append(te_p)
            oof_te_indices.append(te_idx)

            # Determine thresholds on training fold for each target coverage
            for cov in coverages:
                if cov >= 0.999:
                    sel = np.ones(len(df_te), dtype=bool)
                else:
                    thresh = np.quantile(tr_s, 1.0 - cov)
                    sel = te_s >= thresh
                cov_selections[cov].append(sel)

        all_preds = np.concatenate(oof_preds)
        all_te_idx = np.concatenate(oof_te_indices)
        df_eval = df_train_full.iloc[all_te_idx]

        y_true_sign = df_eval[raw_sign_col].values
        fwd_ret_open = df_eval[fwd_ret_open_col].values

        for cov in coverages:
            all_sel = np.concatenate(cov_selections[cov])
            sel_preds = np.where(all_sel, all_preds, 0.0)

            res = evaluate_subset_metrics(
                y_true_sign=y_true_sign,
                y_pred_dir=sel_preds,
                fwd_ret_open=fwd_ret_open,
                cost=ROUNDTRIP_COST,
                block_length=BLOCK_LENGTH,
                n_boot=1000,
                seed=42,
            )

            rec = {
                "model": m_name,
                "description": m_desc,
                "horizon": horizon,
                "target_coverage": cov,
                "actual_coverage": float(res["n_signals"] / len(df_eval)) if len(df_eval) > 0 else 0.0,
                "signals_per_day": float((res["n_signals"] / len(df_eval)) * 288.0) if len(df_eval) > 0 else 0.0,
                "n_signals": res["n_signals"],
                "hit_rate": res["hit_rate"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "mean_abs_fwd_ret": res["mean_abs_fwd_ret"],
                "break_even_rate": res["break_even_rate"],
                "break_even_gap": res["break_even_gap"],
                "net_bps": res["net_bps"],
            }
            records.append(rec)

            # For 20% coverage rows, capture decomposition
            if abs(cov - 0.20) < 1e-4:
                decomp_records.append({
                    "model": m_name,
                    "description": m_desc,
                    "horizon": horizon,
                    "hit_rate": res["hit_rate"],
                    "direction_skill": res["direction_skill"],
                    "mean_abs_fwd_ret": res["mean_abs_fwd_ret"],
                    "break_even_rate": res["break_even_rate"],
                    "size_selection_hurdle": res["size_selection_hurdle"],
                    "break_even_gap": res["break_even_gap"],
                    "net_bps": res["net_bps"],
                })

    df_results = pd.DataFrame(records)
    df_decomp = pd.DataFrame(decomp_records)

    if save_report_path is not None:
        p = Path(save_report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(p, index=False)
        logger.info("Saved like-for-like comparison report to %s", p)

    if save_decomp_path is not None:
        p2 = Path(save_decomp_path)
        p2.parent.mkdir(parents=True, exist_ok=True)
        df_decomp.to_csv(p2, index=False)
        logger.info("Saved 20%% coverage gap decomposition report to %s", p2)

    return df_results, df_decomp

