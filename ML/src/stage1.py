"""Stage 1 Model Selection — 76 Logistic Regression Configurations.

Budget:
  - Setup B (binary raw sign): 4 horizons N in {2, 3, 6, 12} = 4 configs.
  - Setup A (clear move only): 4 horizons x 9 distinct rules (S1, S2, V1, V2, A1, A2, F05, F10, F15) = 36 configs.
  - Setup C (three-class: Down/Neutral/Up): 4 horizons x 9 distinct rules = 36 configs.
  Total = 76 configurations.

Model & Features:
  - Logistic Regression (l2 penalty, C=1.0, solver='lbfgs', max_iter=1000, seed 42)
  - Features: LINEAR_FEATURES (70 features from FS_CORE, drops exact identity and collinears |rho| > 0.95)
  - Standardized inside each fold using StandardScaler fit on fold training block only.

Evaluation & Ranking Protocol:
  - 6-fold expanding walk-forward PurgedWalkForwardCV on post-warmup train split.
  - For each fold:
    * Predict probabilities on held-out test block.
    * Select top 20% most confident emitted signals (threshold fixed from fold train quantiles).
    * Compute break-even gap on emitted signals:
        break_even_gap = hit_rate - break_even_hit_rate
        where break_even_hit_rate = 0.5 + c / (2 * m), c = 0.0023, m = mean(|fwd_ret_N|).
  - Aggregate across 6 folds:
      rank_metric = mean(break_even_gap) - std(break_even_gap)
  - Filter: signals_per_day >= 15. Configurations with < 15 signals/day are REJECTED.
  - Selection: Top 3 configurations become SURVIVORS.
    * Fallback: N=12 S1 Setup B is kept as mandatory 4th configuration if not in top 3.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.column_dictionary import get_feature_columns, load_master
from src.cv import PurgedWalkForwardCV
from src.evaluate import compute_operational_metrics
from src.io_utils import save_oof_predictions
from src.utils import setup_logger

logger = setup_logger("stage1")

ROUNDTRIP_COST: float = getattr(config, "TOTAL_ROUNDTRIP_COST", 0.0023)
DISTINCT_RULES: List[str] = ["S1", "S2", "V1", "V2", "A1", "A2", "F05", "F10", "F15"]
CONFIDENCE_QUANTILE: float = 0.80  # Top 20% most confident signals
MIN_SIGNALS_PER_DAY: float = 15.0


def generate_stage1_configurations() -> List[Dict[str, Any]]:
    """Return all 76 candidate configurations for Stage 1."""
    configs = []

    # 1. Setup B: Binary direction (4 configs)
    for N in config.CANDIDATE_HORIZONS:
        configs.append({
            "config_id": f"B_{N}",
            "setup": "setup_b",
            "horizon": N,
            "rule": "raw_sign",
            "target_col": f"bin_dir_{N}",
            "n_classes": 2,
        })

    # 2. Setup A: Clear move only (36 configs)
    for N in config.CANDIDATE_HORIZONS:
        for r in DISTINCT_RULES:
            configs.append({
                "config_id": f"A_{N}_{r}",
                "setup": "setup_a",
                "horizon": N,
                "rule": r,
                "target_col": f"target_{N}_{r}",
                "n_classes": 2,
            })

    # 3. Setup C: Three-class (36 configs)
    for N in config.CANDIDATE_HORIZONS:
        for r in DISTINCT_RULES:
            configs.append({
                "config_id": f"C_{N}_{r}",
                "setup": "setup_c",
                "horizon": N,
                "rule": r,
                "target_col": f"target_{N}_{r}",
                "n_classes": 3,
            })

    assert len(configs) == 76, f"Expected 76 configurations, generated {len(configs)}"
    return configs


def train_and_predict_fold_lr(
    cfg: Dict[str, Any],
    df_train_block: pd.DataFrame,
    df_test_block: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train Logistic Regression on fold train block and predict on held-out test block.

    Returns
    -------
    pred_dir : np.ndarray
        Predicted trade direction (+1.0, -1.0, or 0.0).
    prob_down : np.ndarray
    prob_up : np.ndarray
    conf_threshold : float
        Confidence threshold chosen on train split.
    """
    setup = cfg["setup"]
    N = cfg["horizon"]
    target_col = cfg["target_col"]

    # Extract training matrices
    y_train = df_train_block[target_col]

    if setup == "setup_b":
        # Binary raw sign: 1 if Up, 0 if Down
        valid_train = y_train.notna()
        X_tr = df_train_block.loc[valid_train, feature_cols].fillna(0.0).values
        y_tr = y_train[valid_train].values.astype(int)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(df_test_block[feature_cols].fillna(0.0).values)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_scaled, y_tr)

        # Train confidence threshold (distance from 0.5)
        tr_probs = clf.predict_proba(X_tr_scaled)[:, 1]
        tr_confs = np.abs(tr_probs - 0.5)
        conf_thresh = float(np.quantile(tr_confs, CONFIDENCE_QUANTILE))

        # Test predictions
        te_probs = clf.predict_proba(X_te_scaled)[:, 1]
        prob_up = te_probs
        prob_down = 1.0 - te_probs
        te_confs = np.abs(te_probs - 0.5)

        # Emit trade only if confidence >= threshold
        pred_dir = np.zeros(len(df_test_block), dtype=float)
        emit_mask = te_confs >= conf_thresh
        pred_dir[emit_mask & (te_probs >= 0.5)] = 1.0
        pred_dir[emit_mask & (te_probs < 0.5)] = -1.0

    elif setup == "setup_a":
        # Clear move: train ONLY on rows where target != 0 (neutral excluded)
        valid_train = y_train.notna() & (y_train != 0.0)
        X_tr = df_train_block.loc[valid_train, feature_cols].fillna(0.0).values
        y_tr = np.where(y_train[valid_train].values > 0, 1, 0)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(df_test_block[feature_cols].fillna(0.0).values)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_scaled, y_tr)

        # Train confidence threshold
        tr_probs = clf.predict_proba(X_tr_scaled)[:, 1]
        tr_confs = np.abs(tr_probs - 0.5)
        conf_thresh = float(np.quantile(tr_confs, CONFIDENCE_QUANTILE))

        # Test predictions
        te_probs = clf.predict_proba(X_te_scaled)[:, 1]
        prob_up = te_probs
        prob_down = 1.0 - te_probs
        te_confs = np.abs(te_probs - 0.5)

        pred_dir = np.zeros(len(df_test_block), dtype=float)
        emit_mask = te_confs >= conf_thresh
        pred_dir[emit_mask & (te_probs >= 0.5)] = 1.0
        pred_dir[emit_mask & (te_probs < 0.5)] = -1.0

    elif setup == "setup_c":
        # Three-class: -1, 0, +1
        valid_train = y_train.notna()
        X_tr = df_train_block.loc[valid_train, feature_cols].fillna(0.0).values
        y_tr = y_train[valid_train].values.astype(int)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(df_test_block[feature_cols].fillna(0.0).values)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_scaled, y_tr)

        classes = clf.classes_
        down_idx = np.where(classes == -1)[0][0]
        up_idx = np.where(classes == 1)[0][0]

        # Train directional confidence on non-neutral predictions
        tr_prob_matrix = clf.predict_proba(X_tr_scaled)
        tr_preds = clf.predict(X_tr_scaled)
        tr_non_neutral = (tr_preds != 0)
        if tr_non_neutral.sum() > 0:
            tr_confs = np.maximum(tr_prob_matrix[tr_non_neutral, up_idx], tr_prob_matrix[tr_non_neutral, down_idx])
            conf_thresh = float(np.quantile(tr_confs, CONFIDENCE_QUANTILE))
        else:
            conf_thresh = 0.5

        # Test predictions
        te_prob_matrix = clf.predict_proba(X_te_scaled)
        te_preds = clf.predict(X_te_scaled)
        prob_up = te_prob_matrix[:, up_idx]
        prob_down = te_prob_matrix[:, down_idx]
        te_confs = np.maximum(prob_up, prob_down)

        pred_dir = np.zeros(len(df_test_block), dtype=float)
        emit_mask = (te_preds != 0) & (te_confs >= conf_thresh)
        pred_dir[emit_mask & (te_preds == 1)] = 1.0
        pred_dir[emit_mask & (te_preds == -1)] = -1.0

    else:
        raise ValueError(f"Unknown setup: {setup}")

    return pred_dir, prob_down, prob_up, conf_thresh


def run_stage1_screening(
    df_train_full: pd.DataFrame,
    save_oof: bool = True,
    save_report_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Run all 76 Stage 1 Logistic Regression configurations across 6 CV folds.

    Parameters
    ----------
    df_train_full : pd.DataFrame
        Post-warmup or full train split of master DataFrame.
    save_oof : bool
        Whether to cache OOF predictions for all 76 configurations.
    save_report_path : Optional[Union[str, Path]]
        Destination path for stage1_lr_ranking.csv.

    Returns
    -------
    pd.DataFrame
        Complete 76-row ranking table with survivor status.
    """
    feature_cols = get_feature_columns(df_train_full, feature_set="linear")
    logger.info("Stage 1 screening: using %d linear features", len(feature_cols))

    configs = generate_stage1_configurations()
    cv = PurgedWalkForwardCV()
    results: List[Dict[str, Any]] = []

    total_configs = len(configs)
    for idx, cfg in enumerate(configs, 1):
        N = cfg["horizon"]
        setup = cfg["setup"]
        rule = cfg["rule"]
        cid = cfg["config_id"]

        logger.info("[%d/%d] Running config %s (Setup %s, N=%d, Rule %s)...", idx, total_configs, cid, setup, N, rule)

        fold_gaps = []
        fold_hits = []
        fold_coverages = []
        fold_bps = []

        all_oof_dfs = []

        for fold, tr_idx, te_idx in cv.split(df_train_full):
            df_tr = df_train_full.iloc[tr_idx]
            df_te = df_train_full.iloc[te_idx]

            pred_dir, prob_down, prob_up, conf_th = train_and_predict_fold_lr(
                cfg=cfg,
                df_train_block=df_tr,
                df_test_block=df_te,
                feature_cols=feature_cols,
            )

            # Evaluate fold operational metrics against raw_sign_N
            y_true_sign = df_te[f"raw_sign_{N}"].values
            fwd_ret = df_te[f"fwd_ret_{N}"].values
            valid_mask = np.isfinite(y_true_sign) & np.isfinite(fwd_ret) & (y_true_sign != 0.0)

            op = compute_operational_metrics(
                y_true_sign=y_true_sign,
                y_pred_dir=pred_dir,
                fwd_ret=fwd_ret,
                cost=ROUNDTRIP_COST,
                coverage_mask=valid_mask,
            )

            fold_gaps.append(op["break_even_gap"] if np.isfinite(op["break_even_gap"]) else -1.0)
            fold_hits.append(op["hit_rate"] if np.isfinite(op["hit_rate"]) else 0.0)
            fold_coverages.append(op["coverage"])
            fold_bps.append(op["net_return_bps"] if np.isfinite(op["net_return_bps"]) else -100.0)

            if save_oof:
                oof_part = pd.DataFrame({
                    "open_time": df_te.index,
                    "fold": fold,
                    "model": "lr",
                    "setup": setup,
                    "horizon": N,
                    "rule": rule,
                    "prob_down": prob_down,
                    "prob_neutral": 0.0,
                    "prob_up": prob_up,
                    "pred_label": pred_dir,
                    "true_label": y_true_sign,
                })
                all_oof_dfs.append(oof_part)

        # Aggregate across 6 folds
        mean_gap = float(np.mean(fold_gaps))
        std_gap = float(np.std(fold_gaps))
        rank_metric = float(mean_gap - std_gap)

        mean_coverage = float(np.mean(fold_coverages))
        signals_per_day = float(mean_coverage * 288.0)
        mean_hit_rate = float(np.mean(fold_hits))
        mean_net_bps = float(np.mean(fold_bps))

        # Check filter
        meets_signal_filter = bool(signals_per_day >= MIN_SIGNALS_PER_DAY)

        results.append({
            "config_id": cid,
            "setup": setup,
            "horizon": N,
            "horizon_min": N * 5,
            "rule": rule,
            "signals_per_day": signals_per_day,
            "coverage": mean_coverage,
            "mean_hit_rate": mean_hit_rate,
            "mean_break_even_gap": mean_gap,
            "std_break_even_gap": std_gap,
            "rank_metric": rank_metric,
            "mean_net_bps": mean_net_bps,
            "meets_filter": meets_signal_filter,
            "status": "candidate" if meets_signal_filter else "rejected",
        })

        if save_oof and all_oof_dfs:
            full_oof = pd.concat(all_oof_dfs, ignore_index=True)
            save_oof_predictions(
                df=full_oof,
                model_name="lr",
                setup=setup,
                horizon=N,
                rule=rule,
            )

    df_rank = pd.DataFrame(results)

    # Sort candidates by rank_metric descending
    # Candidates that meet filter are sorted first
    candidates = df_rank[df_rank["meets_filter"]].copy()
    rejected = df_rank[~df_rank["meets_filter"]].copy()

    candidates = candidates.sort_values(by="rank_metric", ascending=False).reset_index(drop=True)

    # Top 3 are survivors
    survivors_cids = set()
    for i in range(min(3, len(candidates))):
        cid = candidates.loc[i, "config_id"]
        survivors_cids.add(cid)
        df_rank.loc[df_rank["config_id"] == cid, "status"] = "survivor"

    # Mandatory fallback: N=12 S1 Setup B (config_id: "B_12")
    fallback_cid = "B_12"
    if fallback_cid not in survivors_cids:
        df_rank.loc[df_rank["config_id"] == fallback_cid, "status"] = "fallback"

    # Sort final table by meets_filter descending, then rank_metric descending
    df_rank = df_rank.sort_values(
        by=["meets_filter", "rank_metric"],
        ascending=[False, False]
    ).reset_index(drop=True)

    if save_report_path is not None:
        p = Path(save_report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_rank.to_csv(p, index=False)
        logger.info("Saved Stage 1 LR ranking report to %s", p)

    return df_rank

