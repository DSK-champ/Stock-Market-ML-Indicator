"""Feature Exploratory Data Analysis (EDA) module — Stock-Market-ML-Predictor v3.

Provides statistical computations, correlations, ROC AUC evaluations,
circular-shift noise band simulations, context breakdowns, quarterly direction stability,
hierarchical correlation clustering, and break-even tables for all 85 engineered features.
Strictly enforced: TRAIN partition only (no validation or test rows evaluated).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

from src import config
from src.features import ALL_FEATURE_COLUMNS, FEATURE_GROUPS
from src.column_dictionary import get_column_meta, get_column_role, get_feature_columns
from src.utils import setup_logger

logger = setup_logger("feat_eda")
REPORTS_DIR = config.REPORTS_DIR
FIGURES_DIR = config.FIGURES_DIR


# ---------------------------------------------------------------------------
# Anti-Leakage Guard
# ---------------------------------------------------------------------------

def verify_train_only(df: pd.DataFrame) -> None:
    """Strictly assert that DataFrame contains only training partition rows."""
    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    if df.index.max() > train_end:
        violating_rows = (df.index > train_end).sum()
        raise ValueError(
            f"LEAKAGE VIOLATION: DataFrame extends past train end ({train_end}). "
            f"Found {violating_rows} out-of-bounds rows (up to {df.index.max()}). "
            "Feature EDA must be performed exclusively on the training partition!"
        )


# ---------------------------------------------------------------------------
# Section 1: Feature Inventory
# ---------------------------------------------------------------------------

def compute_feature_inventory(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Compile inventory of all 85 features with groups, formulas, and NaN counts."""
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    
    rows = []
    for col in feature_cols:
        meta = get_column_meta(col)
        nan_count = int(df_train[col].isna().sum())
        nan_share = float(nan_count / len(df_train))
        zero_count = int((df_train[col] == 0.0).sum())
        zero_share = float(zero_count / len(df_train))
        
        rows.append({
            "feature": col,
            "group": meta.get("group", "unknown"),
            "unit": meta.get("unit", ""),
            "formula": meta.get("formula", ""),
            "warmup_candles": config.LOOKBACK_CANDLES,
            "nan_count_train": nan_count,
            "nan_share_train": nan_share,
            "zero_count_train": zero_count,
            "zero_share_train": zero_share,
        })
        
    df_inv = pd.DataFrame(rows)
    group_counts = df_inv["group"].value_counts()
    
    out_path = REPORTS_DIR / "feat_inventory.csv"
    df_inv.to_csv(out_path, index=False)
    logger.info("Saved feature inventory to %s (%d features)", out_path, len(df_inv))
    return df_inv, group_counts


# ---------------------------------------------------------------------------
# Section 2: Full Variable Statistics Table
# ---------------------------------------------------------------------------

def compute_feature_statistics_table(
    df_train: pd.DataFrame,
    warmup_candles: int = config.LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Compute detailed summary statistics for all features on valid post-warmup train rows."""
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    n_rows = len(df_valid)
    
    rows = []
    for col in feature_cols:
        s = df_valid[col].dropna()
        n = len(s)
        if n == 0:
            continue
            
        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1))
        min_val = float(s.min())
        q01 = float(s.quantile(0.01))
        q05 = float(s.quantile(0.05))
        q25 = float(s.quantile(0.25))
        median_val = float(s.median())
        q75 = float(s.quantile(0.75))
        q95 = float(s.quantile(0.95))
        q99 = float(s.quantile(0.99))
        max_val = float(s.max())
        skew_val = float(stats.skew(s))
        kurt_val = float(stats.kurtosis(s))  # Fisher excess kurtosis
        
        zero_share = float((s == 0.0).sum() / n)
        
        mad = float(stats.median_abs_deviation(s, scale="normal"))
        if mad > 1e-12:
            outlier_share_5mad = float((np.abs(s - median_val) > 5.0 * mad).sum() / n)
        else:
            outlier_share_5mad = float((np.abs(s - mean_val) > 5.0 * max(std_val, 1e-12)).sum() / n)
            
        mode_share = float(s.value_counts(normalize=True).iloc[0]) if len(s) > 0 else 1.0
        is_near_constant = bool((std_val < 1e-6) or (mode_share > 0.95))
        
        meta = get_column_meta(col)
        rows.append({
            "feature": col,
            "group": meta.get("group", ""),
            "count": n,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "q01": q01,
            "q05": q05,
            "q25": q25,
            "median": median_val,
            "q75": q75,
            "q95": q95,
            "q99": q99,
            "max": max_val,
            "skew": skew_val,
            "excess_kurtosis": kurt_val,
            "zero_share": zero_share,
            "share_beyond_5_robust_sigma": outlier_share_5mad,
            "top_val_share": mode_share,
            "is_near_constant": is_near_constant,
        })
        
    df_stats = pd.DataFrame(rows)
    out_path = REPORTS_DIR / "feat_variable_statistics.csv"
    df_stats.to_csv(out_path, index=False)
    logger.info("Saved feature statistics table to %s (%d rows)", out_path, len(df_stats))
    return df_stats


# ---------------------------------------------------------------------------
# Section 4: Correlation and Multicollinearity (VIF)
# ---------------------------------------------------------------------------

def compute_feature_correlations(
    df_train: pd.DataFrame,
    warmup_candles: int = config.LOOKBACK_CANDLES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices, top 20 pairs, and pairs > 0.90."""
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    df_valid = df_train.iloc[warmup_candles:][feature_cols].copy()
    
    logger.info("Computing Pearson and Spearman correlation matrices across %d features...", len(feature_cols))
    corr_pearson = df_valid.corr(method="pearson")
    corr_spearman = df_valid.corr(method="spearman")
    
    corr_pearson.to_csv(REPORTS_DIR / "feat_corr_pearson.csv")
    corr_spearman.to_csv(REPORTS_DIR / "feat_corr_spearman.csv")
    
    pairs = []
    cols = list(corr_spearman.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            p_val = float(corr_pearson.loc[f1, f2])
            s_val = float(corr_spearman.loc[f1, f2])
            pairs.append({
                "feature_1": f1,
                "group_1": get_column_meta(f1)["group"],
                "feature_2": f2,
                "group_2": get_column_meta(f2)["group"],
                "spearman_corr": s_val,
                "abs_spearman_corr": abs(s_val),
                "pearson_corr": p_val,
                "abs_pearson_corr": abs(p_val),
            })
            
    df_pairs = pd.DataFrame(pairs)
    df_pairs_sorted = df_pairs.sort_values(by="abs_spearman_corr", ascending=False).reset_index(drop=True)
    
    df_top20 = df_pairs_sorted.head(20).copy()
    df_top20.to_csv(REPORTS_DIR / "feat_top20_correlated_pairs.csv", index=False)
    
    df_above_90 = df_pairs_sorted[df_pairs_sorted["abs_spearman_corr"] > 0.90].copy()
    df_above_90.to_csv(REPORTS_DIR / "feat_pairs_above_90.csv", index=False)
    
    logger.info(
        "Correlations computed: %d pairs total, %d pairs with |corr| > 0.90",
        len(df_pairs), len(df_above_90),
    )
    return corr_pearson, corr_spearman, df_top20, df_above_90


def compute_feature_vif(
    corr_matrix: pd.DataFrame,
    regularization: float = 1e-5,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, np.ndarray]:
    """Compute VIF on all features using Ridge regularization and Plain VIF with duplicates removed."""
    R = corr_matrix.to_numpy()
    p = R.shape[0]
    cols = list(corr_matrix.columns)
    
    eigvals, eigvecs = np.linalg.eigh(R)
    idx_sorted = np.argsort(eigvals)
    smallest_eigvals = eigvals[idx_sorted[:5]]
    cond_num = float(eigvals.max() / max(eigvals.min(), 1e-18))
    
    R_ridge = R + regularization * np.eye(p)
    inv_ridge = np.linalg.inv(R_ridge)
    vifs_ridge = np.diag(inv_ridge)
    
    df_vif_ridge = pd.DataFrame({
        "feature": cols,
        "group": [get_column_meta(c)["group"] for c in cols],
        "vif_ridge": vifs_ridge,
    }).sort_values(by="vif_ridge", ascending=False).reset_index(drop=True)
    
    # Drop linear MACD identity for Plain VIF
    drop_for_plain = ["feat_mom_macd_hist"]
    cols_plain = [c for c in cols if c not in drop_for_plain]
    R_sub = corr_matrix.loc[cols_plain, cols_plain].to_numpy()
    
    try:
        inv_plain = np.linalg.inv(R_sub)
        vifs_plain = np.diag(inv_plain)
        df_vif_plain = pd.DataFrame({
            "feature": cols_plain,
            "group": [get_column_meta(c)["group"] for c in cols_plain],
            "vif_plain": vifs_plain,
        }).sort_values(by="vif_plain", ascending=False).reset_index(drop=True)
    except np.linalg.LinAlgError:
        df_vif_plain = pd.DataFrame(columns=["feature", "group", "vif_plain"])
        
    df_vif_ridge.to_csv(REPORTS_DIR / "feat_vif_all.csv", index=False)
    return df_vif_ridge, df_vif_plain, cond_num, smallest_eigvals


# ---------------------------------------------------------------------------
# Section 5: Size Side Analysis (target big_move_N)
# ---------------------------------------------------------------------------

def compute_size_side_metrics(
    df_train: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    warmup_candles: int = config.LOOKBACK_CANDLES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute single-feature ROC AUC, Cohen's d, and Spearman correlation for big_move_N."""
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    
    auc_records = []
    cohen_records = []
    
    for col in feature_cols:
        x = df_valid[col].to_numpy()
        grp = get_column_meta(col)["group"]
        
        auc_row = {"feature": col, "group": grp}
        cohen_row = {"feature": col, "group": grp}
        
        for N in horizons:
            bm_col = f"big_move_{N}"
            fwd_col = f"fwd_ret_{N}"
            
            y_bm = df_valid[bm_col].to_numpy()
            fwd_ret = df_valid[fwd_col].to_numpy()
            
            valid_mask = (~np.isnan(x)) & (~np.isnan(y_bm))
            x_v = x[valid_mask]
            y_v = y_bm[valid_mask]
            fwd_v = fwd_ret[valid_mask]
            
            if len(np.unique(y_v)) == 2:
                raw_auc = float(roc_auc_score(y_v, x_v))
                eff_auc = max(raw_auc, 1.0 - raw_auc)
            else:
                raw_auc, eff_auc = 0.5, 0.5
                
            auc_row[f"raw_auc_{N}"] = raw_auc
            auc_row[f"auc_{N}"] = eff_auc
            
            x1 = x_v[y_v == 1.0]
            x0 = x_v[y_v == 0.0]
            if len(x1) > 1 and len(x0) > 1:
                n1, n0 = len(x1), len(x0)
                s1_var, s0_var = np.var(x1, ddof=1), np.var(x0, ddof=1)
                pooled_sd = np.sqrt(((n1 - 1) * s1_var + (n0 - 1) * s0_var) / (n1 + n0 - 2))
                d_val = float((np.mean(x1) - np.mean(x0)) / max(pooled_sd, 1e-12))
            else:
                d_val = 0.0
                
            abs_fwd = np.abs(fwd_v)
            spear_val = float(pd.Series(x_v).corr(pd.Series(abs_fwd), method="spearman"))
            
            cohen_row[f"cohens_d_{N}"] = d_val
            cohen_row[f"abs_cohens_d_{N}"] = abs(d_val)
            cohen_row[f"spearman_abs_ret_{N}"] = spear_val
            
        auc_records.append(auc_row)
        cohen_records.append(cohen_row)
        
    df_auc_size = pd.DataFrame(auc_records)
    df_cohen_size = pd.DataFrame(cohen_records)
    
    df_auc_size.to_csv(REPORTS_DIR / "feat_auc_big_move.csv", index=False)
    df_cohen_size.to_csv(REPORTS_DIR / "feat_cohens_d_big_move.csv", index=False)
    
    group_summaries = []
    for grp, df_grp in df_auc_size.groupby("group"):
        row = {"group": grp, "n_features": len(df_grp)}
        for N in horizons:
            col_name = f"auc_{N}"
            row[f"best_auc_{N}"] = float(df_grp[col_name].max())
            row[f"median_auc_{N}"] = float(df_grp[col_name].median())
        group_summaries.append(row)
        
    df_group_summary = pd.DataFrame(group_summaries)
    df_group_summary.to_csv(REPORTS_DIR / "feat_group_auc_size.csv", index=False)
    
    return df_auc_size, df_cohen_size, df_group_summary


# ---------------------------------------------------------------------------
# Section 6: Direction Side Analysis & Vectorized Circular Shift Noise Band
# ---------------------------------------------------------------------------

def compute_direction_side_metrics(
    df_train: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    warmup_candles: int = config.LOOKBACK_CANDLES,
    n_shifts: int = 500,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compute directional metrics, lagged correlations, and circular-shift noise bands."""
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    n_rows = len(df_valid)
    
    # 1. Base Directional Metrics
    dir_records = []
    for col in feature_cols:
        x = df_valid[col].to_numpy()
        grp = get_column_meta(col)["group"]
        
        row = {"feature": col, "group": grp}
        for N in horizons:
            sign_col = f"raw_sign_{N}"
            fwd_col = f"fwd_ret_{N}"
            s1_col = f"target_{N}_S1"
            
            y_sign = df_valid[sign_col].to_numpy()
            fwd_ret = df_valid[fwd_col].to_numpy()
            y_s1 = df_valid[s1_col].to_numpy()
            
            valid_mask = (~np.isnan(x)) & (~np.isnan(y_sign)) & (y_sign != 0.0)
            x_v = x[valid_mask]
            y_v = (y_sign[valid_mask] == 1.0).astype(int)
            
            if len(np.unique(y_v)) == 2:
                raw_auc = float(roc_auc_score(y_v, x_v))
                eff_auc = max(raw_auc, 1.0 - raw_auc)
            else:
                raw_auc, eff_auc = 0.5, 0.5
                
            mask_ret = (~np.isnan(x)) & (~np.isnan(fwd_ret))
            spear_val = float(pd.Series(x[mask_ret]).corr(pd.Series(fwd_ret[mask_ret]), method="spearman"))
            
            mask_s1 = (~np.isnan(x)) & (~np.isnan(y_s1))
            x_up = x[mask_s1 & (y_s1 == 1.0)]
            x_dn = x[mask_s1 & (y_s1 == -1.0)]
            if len(x_up) > 1 and len(x_dn) > 1:
                n_u, n_d = len(x_up), len(x_dn)
                v_u, v_d = np.var(x_up, ddof=1), np.var(x_dn, ddof=1)
                pooled = np.sqrt(((n_u - 1) * v_u + (n_d - 1) * v_d) / (n_u + n_d - 2))
                cohen_s1 = float((np.mean(x_up) - np.mean(x_dn)) / max(pooled, 1e-12))
            else:
                cohen_s1 = 0.0
                
            row[f"raw_auc_{N}"] = raw_auc
            row[f"auc_{N}"] = eff_auc
            row[f"spearman_{N}"] = spear_val
            row[f"cohens_d_s1_{N}"] = cohen_s1
            
        dir_records.append(row)
        
    df_dir = pd.DataFrame(dir_records)
    
    # 2. Lagged Correlations for flow_window, location, and momentum
    target_groups = ["flow_window", "location", "momentum"]
    lag_feats = [c for c in feature_cols if get_column_meta(c)["group"] in target_groups]
    lags = [0, 1, 2, 3, 6, 12]
    
    lag_records = []
    fwd_12 = df_valid["fwd_ret_12"]
    for col in lag_feats:
        s = df_valid[col]
        row_lag = {"feature": col, "group": get_column_meta(col)["group"]}
        for k in lags:
            s_lagged = s.shift(k)
            corr_k = float(s_lagged.corr(fwd_12, method="spearman"))
            row_lag[f"lag_{k}"] = corr_k
        lag_records.append(row_lag)
        
    df_lagged = pd.DataFrame(lag_records)
    df_lagged.to_csv(REPORTS_DIR / "feat_lagged_flow_loc.csv", index=False)
    
    # 3. Vectorized Circular Shift Noise Band Simulation (500 shifts)
    logger.info("Precomputing rank matrix across %d features for circular-shift noise band...", len(feature_cols))
    R = np.vstack([rankdata(df_valid[col].to_numpy()) for col in feature_cols])  # shape: (n_feats, n_rows)
    
    rng = np.random.default_rng(random_seed)
    min_offset = config.LOOKBACK_CANDLES
    offsets = rng.integers(min_offset, n_rows - min_offset, size=n_shifts)
    
    noise_summary_rows = []
    noise_band_info = {}
    
    for side, prefix in [("size", "big_move_"), ("direction", "raw_sign_")]:
        for N in horizons:
            col_target = f"{prefix}{N}"
            y_raw = df_valid[col_target].to_numpy()
            
            if side == "direction":
                valid_y = (~np.isnan(y_raw)) & (y_raw != 0.0)
                y_bin = (y_raw == 1.0).astype(float)
            else:
                valid_y = ~np.isnan(y_raw)
                y_bin = (y_raw == 1.0).astype(float)
                
            y_bin[~valid_y] = 0.0
            n1 = float(np.sum(y_bin))
            n0 = float(len(y_bin) - n1)
            c_const = n1 * (n1 + 1.0) / 2.0
            denom = max(n1 * n0, 1e-12)
            
            # 500 shifts
            auc_matrix = np.empty((n_shifts, len(feature_cols)))
            for s_idx, s in enumerate(offsets):
                y_s = np.roll(y_bin, s)
                sum_ranks = R @ y_s
                raw_aucs = (sum_ranks - c_const) / denom
                auc_matrix[s_idx] = np.maximum(raw_aucs, 1.0 - raw_aucs)
                
            # Distribution of maximum AUC over all features
            max_auc_per_shift = auc_matrix.max(axis=1)
            max_95th = float(np.percentile(max_auc_per_shift, 95))
            max_99th = float(np.percentile(max_auc_per_shift, 99))
            
            # Mean and 95th across features
            feat_95th = float(np.mean(np.percentile(auc_matrix, 95, axis=0)))
            feat_99th = float(np.mean(np.percentile(auc_matrix, 99, axis=0)))
            
            # iid shuffle comparison (ignores dependence)
            y_shuffled = rng.permutation(y_bin)
            raw_iid = (R @ y_shuffled - c_const) / denom
            eff_iid = np.maximum(raw_iid, 1.0 - raw_iid)
            iid_95th_max = float(np.percentile(eff_iid, 95))
            
            key_pfx = f"{side}_{N}"
            noise_band_info[f"{key_pfx}_max_95th"] = max_95th
            noise_band_info[f"{key_pfx}_max_99th"] = max_99th
            
            noise_summary_rows.append({
                "side": side,
                "horizon_N": N,
                "circular_shift_max_95th": max_95th,
                "circular_shift_max_99th": max_99th,
                "circular_shift_feature_mean_95th": feat_95th,
                "circular_shift_feature_mean_99th": feat_99th,
                "iid_shuffle_95th_ignores_dependence": iid_95th_max,
            })
            
    df_noise = pd.DataFrame(noise_summary_rows)
    df_noise.to_csv(REPORTS_DIR / "feat_noise_band.csv", index=False)
    
    # Re-flag direction features exceeding 99th percentile of circular-shift maximum
    max_99_dir_12 = noise_band_info["direction_12_max_99th"]
    df_dir["exceeds_99th_circular_max_12"] = df_dir["auc_12"] > max_99_dir_12
    df_dir.to_csv(REPORTS_DIR / "feat_direction_metrics.csv", index=False)
    
    return df_dir, df_lagged, df_noise, noise_band_info


# ---------------------------------------------------------------------------
# Section 6b: Direction Stability Across Quarters
# ---------------------------------------------------------------------------

def compute_direction_quarterly_stability(
    df_train: pd.DataFrame,
    top_features: List[str],
    warmup_candles: int = config.LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Compute quarterly ROC AUC and sign consistency for top direction features across all 9 quarters."""
    verify_train_only(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    quarters = df_valid.index.tz_localize(None).to_period("Q").to_numpy()
    unique_q = np.unique(quarters)
    
    y_all = df_valid["raw_sign_12"].to_numpy()
    valid_mask = (~np.isnan(y_all)) & (y_all != 0.0)
    y_clean = (y_all[valid_mask] == 1.0).astype(int)
    
    quarterly_rows = []
    for col in top_features:
        x_full = df_valid[col].to_numpy()
        raw_overall = roc_auc_score(y_clean, x_full[valid_mask])
        is_positive_orient = raw_overall >= 0.5
        overall_eff_auc = max(raw_overall, 1.0 - raw_overall)
        
        row = {
            "feature": col,
            "group": get_column_meta(col)["group"],
            "overall_auc_12": overall_eff_auc,
        }
        same_sign_count = 0
        total_q = 0
        
        for q in sorted(unique_q):
            q_mask = (quarters == q) & valid_mask
            y_q = (y_all[q_mask] == 1.0).astype(int)
            x_q = x_full[q_mask]
            
            if len(np.unique(y_q)) == 2:
                raw_q = roc_auc_score(y_q, x_q)
                eff_q = max(raw_q, 1.0 - raw_q)
                row[f"auc_{q}"] = eff_q
                if (raw_q >= 0.5 and is_positive_orient) or (raw_q < 0.5 and not is_positive_orient):
                    same_sign_count += 1
                total_q += 1
                
        row["share_same_sign_quarters"] = same_sign_count / total_q if total_q > 0 else 0.0
        quarterly_rows.append(row)
        
    df_q_stab = pd.DataFrame(quarterly_rows)
    df_q_stab.to_csv(REPORTS_DIR / "feat_direction_quarterly_stability.csv", index=False)
    return df_q_stab


# ---------------------------------------------------------------------------
# Section 7: Context Behavior & De-Seasoning ACF
# ---------------------------------------------------------------------------

def compute_context_analysis(
    df_train: pd.DataFrame,
    warmup_candles: int = config.LOOKBACK_CANDLES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute mean features by hour of day, weekday, and train volatility terciles."""
    verify_train_only(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    
    key_features = [
        "feat_vol_sigma_288",
        "feat_vol_gk_60",
        "feat_vol_seasonal_vol_ratio_12",
        "feat_volume_log",
        "feat_vol_seasonal_volume_ratio_7d",
        "feat_flow_window_net_12",
        "feat_of_taker_buy_ratio",
    ]
    sub_cols = [c for c in key_features if c in df_valid.columns]
    
    by_hour = df_valid.groupby(df_valid.index.hour)[sub_cols].mean()
    by_dow = df_valid.groupby(df_valid.index.day_of_week if hasattr(df_valid.index, 'day_of_week') else df_valid.index.dayofweek)[sub_cols].mean()
    
    vol_ref = df_valid["feat_vol_sigma_288"]
    tercile_bins = vol_ref.quantile([0.3333, 0.6667]).values
    vol_terciles = pd.cut(
        vol_ref,
        bins=[-np.inf, tercile_bins[0], tercile_bins[1], np.inf],
        labels=["Low Volatility", "Medium Volatility", "High Volatility"],
    )
    by_tercile = df_valid.groupby(vol_terciles)[sub_cols].mean()
    
    combined_context = pd.concat([
        by_hour.add_prefix("hour_"),
        by_dow.add_prefix("dow_"),
    ], axis=1)
    combined_context.to_csv(REPORTS_DIR / "feat_context_hour_dow_vol.csv")
    
    def _acf(series, max_lag=288):
        s = series.dropna().to_numpy()
        s_norm = s - np.mean(s)
        c0 = np.dot(s_norm, s_norm) / len(s_norm)
        lags = np.arange(0, max_lag + 1)
        acfs = [1.0]
        for k in lags[1:]:
            c_k = np.dot(s_norm[:-k], s_norm[k:]) / (len(s_norm) - k)
            acfs.append(float(c_k / c0))
        return pd.DataFrame({"lag": lags, "acf": acfs})
        
    acf_raw_vol = _acf(df_valid["volume"], max_lag=288)
    acf_seas_vol = _acf(df_valid["feat_vol_seasonal_volume_ratio_7d"], max_lag=288)
    
    df_acf_comp = pd.DataFrame({
        "lag": acf_raw_vol["lag"],
        "acf_raw_volume": acf_raw_vol["acf"],
        "acf_deseasoned_volume": acf_seas_vol["acf"],
    })
    
    return by_hour, by_dow, by_tercile, df_acf_comp


# ---------------------------------------------------------------------------
# Section 8: Stability & Quarterly Regime Drift
# ---------------------------------------------------------------------------

def compute_quarterly_drift(
    df_train: pd.DataFrame,
    warmup_candles: int = config.LOOKBACK_CANDLES,
    drift_threshold_std: float = 0.5,
) -> pd.DataFrame:
    """Compute mean and std per quarter on train; flag features where quarterly mean drifts > 0.5 overall std."""
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    quarters = df_valid.index.tz_localize(None).to_period("Q")
    
    records = []
    for col in feature_cols:
        s = df_valid[col]
        overall_mean = float(s.mean())
        overall_std = float(s.std(ddof=1))
        
        q_means = s.groupby(quarters).mean()
        q_stds = s.groupby(quarters).std()
        
        if overall_std > 1e-12:
            max_drift_std = float(np.abs(q_means - overall_mean).max() / overall_std)
        else:
            max_drift_std = 0.0
            
        is_drifting = bool(max_drift_std > drift_threshold_std)
        
        row = {
            "feature": col,
            "group": get_column_meta(col)["group"],
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "max_quarterly_drift_std": max_drift_std,
            "has_temporal_drift": is_drifting,
        }
        for q, m in q_means.items():
            row[f"mean_{q}"] = float(m)
        records.append(row)
        
    df_drift = pd.DataFrame(records).sort_values(by="max_quarterly_drift_std", ascending=False).reset_index(drop=True)
    df_drift.to_csv(REPORTS_DIR / "feat_quarterly_drift.csv", index=False)
    return df_drift


# ---------------------------------------------------------------------------
# Section 9: Prune List Rebuilt as Hierarchical Correlation Clusters
# ---------------------------------------------------------------------------

def compute_correlation_clusters(
    df_train: pd.DataFrame,
    warmup_candles: int = config.LOOKBACK_CANDLES,
    corr_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Hierarchical clustering on 1 - |Spearman| at threshold |rho| > 0.95 (d < 0.05).

    Lists each cluster with its members and within-cluster pairwise correlations.
    No chained drops, no representative chosen. Protected: feat_vol_sigma_288, feat_vol_natr_14, feat_mom_rsi_14.
    """
    verify_train_only(df_train)
    feature_cols = get_feature_columns(df_train)
    df_valid = df_train.iloc[warmup_candles:][feature_cols].copy()
    
    corr_spearman = df_valid.corr(method="spearman").abs()
    dist_matrix = np.clip(1.0 - corr_spearman.values, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed_dist = squareform(dist_matrix)
    
    Z = linkage(condensed_dist, method="complete")
    dist_threshold = 1.0 - corr_threshold  # 0.05 for |rho| > 0.95
    cluster_labels = fcluster(Z, t=dist_threshold, criterion="distance")
    
    cluster_df = pd.DataFrame({"feature": feature_cols, "cluster_id": cluster_labels})
    cluster_counts = cluster_df["cluster_id"].value_counts()
    multi_clusters = cluster_counts[cluster_counts > 1].index
    
    PROTECTED_FEATURES = {"feat_vol_sigma_288", "feat_vol_natr_14", "feat_mom_rsi_14"}
    
    cluster_summary = []
    cluster_records = []
    for cid in sorted(multi_clusters):
        members = cluster_df.loc[cluster_df["cluster_id"] == cid, "feature"].tolist()
        sub_corr = corr_spearman.loc[members, members]
        
        # Pairwise minimum correlation within cluster
        triu_idx = np.triu_indices(len(members), k=1)
        min_corr = float(sub_corr.values[triu_idx].min())
        max_corr = float(sub_corr.values[triu_idx].max())
        
        protected_in_cluster = [m for m in members if m in PROTECTED_FEATURES]
        
        cluster_summary.append({
            "cluster_id": cid,
            "size": len(members),
            "members": members,
            "min_within_corr": min_corr,
            "max_within_corr": max_corr,
            "protected_features": protected_in_cluster,
        })
        
        for m in members:
            cluster_records.append({
                "cluster_id": cid,
                "cluster_size": len(members),
                "feature": m,
                "group": get_column_meta(m)["group"],
                "min_cluster_spearman": min_corr,
                "is_protected": m in PROTECTED_FEATURES,
                "cluster_members": ", ".join(members),
            })
            
    df_cluster_report = pd.DataFrame(cluster_records)
    df_cluster_report.to_csv(REPORTS_DIR / "feat_correlation_clusters.csv", index=False)
    logger.info("Identified %d multi-feature clusters with |corr| > %.2f", len(multi_clusters), corr_threshold)
    return df_cluster_report, cluster_summary


# ---------------------------------------------------------------------------
# Section 10: Break-Even Table (Train Only, Computed from Data)
# ---------------------------------------------------------------------------

def compute_breakeven_table(
    df_train: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    warmup_candles: int = config.LOOKBACK_CANDLES,
    cost_c: float = config.TOTAL_ROUNDTRIP_COST,
) -> pd.DataFrame:
    """Compute empirical break-even hit rate over all rows and over big_move=1 rows,

    alongside empirical hit rates of plain reversal baseline rules:
      Rule 1: sign(0.5 - feat_loc_position_288)
      Rule 2: sign(EMA50 - close)
    evaluated on all valid rows and on the 20% most extreme rows.
    """
    verify_train_only(df_train)
    df_valid = df_train.iloc[warmup_candles:].copy()
    
    pos = df_valid["feat_loc_position_288"]
    rule1_signal = np.sign(0.5 - pos)
    q10_pos = pos.quantile(0.10)
    q90_pos = pos.quantile(0.90)
    rule1_extreme_mask = (pos <= q10_pos) | (pos >= q90_pos)
    
    ema_ratio = df_valid["feat_mom_ema_ratio_50"]
    rule2_signal = -np.sign(ema_ratio)
    q10_ema = ema_ratio.quantile(0.10)
    q90_ema = ema_ratio.quantile(0.90)
    rule2_extreme_mask = (ema_ratio <= q10_ema) | (ema_ratio >= q90_ema)

    rows = []
    for N in horizons:
        fwd = df_valid[f"fwd_ret_{N}"].dropna()
        bm = df_valid[f"big_move_{N}"].dropna()
        raw_sign = df_valid[f"raw_sign_{N}"]
        valid_mask = raw_sign.isin([-1.0, 1.0])
        
        mean_abs_all = float(fwd.abs().mean())
        be_hit_rate_all = 0.5 + cost_c / (2.0 * max(mean_abs_all, 1e-12))
        
        fwd_bm1 = df_valid.loc[df_valid[f"big_move_{N}"] == 1.0, f"fwd_ret_{N}"].dropna()
        m = float(fwd_bm1.abs().mean())
        be_hit_rate_bm = 0.5 + cost_c / (2.0 * max(m, 1e-12))
        
        # Rule 1
        m1 = valid_mask & (rule1_signal != 0)
        hits_r1_all = float((rule1_signal[m1] == raw_sign[m1]).mean())
        hits_r1_ext = float((rule1_signal[m1 & rule1_extreme_mask] == raw_sign[m1 & rule1_extreme_mask]).mean())
        
        # Rule 2
        m2 = valid_mask & (rule2_signal != 0)
        hits_r2_all = float((rule2_signal[m2] == raw_sign[m2]).mean())
        hits_r2_ext = float((rule2_signal[m2 & rule2_extreme_mask] == raw_sign[m2 & rule2_extreme_mask]).mean())
        
        rows.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "mean_abs_return_all": mean_abs_all,
            "breakeven_hit_rate_all_rows": "not achievable" if be_hit_rate_all > 1.0 else f"{be_hit_rate_all*100:.2f}%",
            "mean_abs_return_big_move_m": m,
            "breakeven_hit_rate_big_move": f"{be_hit_rate_bm*100:.2f}%",
            "rule1_pos288_hit_rate_all": f"{hits_r1_all*100:.2f}%",
            "rule1_pos288_hit_rate_ext20": f"{hits_r1_ext*100:.2f}%",
            "rule2_ema50_hit_rate_all": f"{hits_r2_all*100:.2f}%",
            "rule2_ema50_hit_rate_ext20": f"{hits_r2_ext*100:.2f}%",
            "cost_c": cost_c,
            "note": "big_move row figure is optimistic because it assumes perfect size filtering",
        })
        
    df_be = pd.DataFrame(rows)
    df_be.to_csv(REPORTS_DIR / "feat_breakeven_analysis.csv", index=False)
    return df_be
