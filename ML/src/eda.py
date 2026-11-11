"""Exploratory Data Analysis (EDA) core statistical computations.

CRITICAL POLICY:
All functions in this module must be evaluated STRICTLY ON THE TRAINING SPLIT.
Any attempt to pass a DataFrame containing timestamps or indices belonging to
the Validation or Test partitions will raise a ValueError.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.features import ALL_FEATURE_COLUMNS, FEATURE_GROUPS
from src.column_dictionary import get_column_role, get_column_meta
from src.utils import setup_logger

logger = setup_logger("eda")


def verify_train_only(df: pd.DataFrame) -> None:
    """Enforce strict anti-leakage policy: verify DataFrame contains ONLY Train rows.
    
    Raises ValueError if timestamps extend past SPLIT_BOUNDARIES['train']['end'].
    """
    if isinstance(df.index, pd.DatetimeIndex):
        t_max = df.index.max()
        train_end = config.SPLIT_BOUNDARIES["train"]["end"]
        if t_max > train_end:
            raise ValueError(
                f"LEAKAGE VIOLATION: Input DataFrame max timestamp {t_max} exceeds "
                f"the Train split boundary {train_end}. EDA is strictly Train-only."
            )
    else:
        if len(df) > config.SPLIT_BOUNDARIES["train"]["expected_candles"]:
            raise ValueError(
                f"LEAKAGE VIOLATION: Input length {len(df)} exceeds Train candle count "
                f"{config.SPLIT_BOUNDARIES['train']['expected_candles']}."
            )


def compute_train_overview(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Compute top-level summary metrics of the training partition."""
    verify_train_only(df_train)
    
    c = df_train["close"]
    t_start = df_train.index.min()
    t_end = df_train.index.max()
    candles = len(df_train)
    days = (t_end - t_start).total_seconds() / 86400.0
    
    start_price = float(c.iloc[0])
    end_price = float(c.iloc[-1])
    min_price = float(c.min())
    max_price = float(c.max())
    total_return_pct = ((end_price / start_price) - 1.0) * 100.0
    
    log_ret_5m = np.log(c / c.shift(1)).dropna()
    ann_factor = np.sqrt(288 * 365)
    ann_vol_pct = float(log_ret_5m.std() * ann_factor * 100.0)
    
    cum_max = c.cummax()
    dd = (c - cum_max) / cum_max
    max_dd_pct = float(dd.min() * 100.0)
    
    return {
        "start": t_start.isoformat(),
        "end": t_end.isoformat(),
        "candles": candles,
        "days": round(days, 2),
        "start_price": start_price,
        "end_price": end_price,
        "min_price": min_price,
        "max_price": max_price,
        "total_return_pct": round(total_return_pct, 2),
        "annualized_vol_pct": round(ann_vol_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
    }


def compute_column_health(df_train: pd.DataFrame) -> pd.DataFrame:
    """Report shape, column names, dtypes, missing values, infs, and unique counts."""
    verify_train_only(df_train)
    
    records = []
    for col in df_train.columns:
        s = df_train[col]
        n_missing = int(s.isna().sum())
        n_inf = int(np.isinf(s.dropna()).sum()) if pd.api.types.is_numeric_dtype(s) else 0
        n_unique = int(s.nunique(dropna=False))
        role = get_column_role(col)
        meta = get_column_meta(col)
        group = meta.get("group", "")
        records.append({
            "column": col,
            "role": role,
            "group": group,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / len(s) * 100, 4),
            "n_inf": n_inf,
            "n_unique": n_unique,
        })
    return pd.DataFrame(records)


def compute_variable_statistics(
    series: pd.Series,
    name: str,
) -> Dict[str, Any]:
    """Compute comprehensive descriptive, moment, and robust statistics for a single variable."""
    s = series.dropna().astype(float)
    n = len(s)
    if n == 0:
        return {"variable": name, "count": 0}
    
    mean = float(s.mean())
    std = float(s.std())
    median = float(s.median())
    min_val = float(s.min())
    max_val = float(s.max())
    
    q01 = float(s.quantile(0.01))
    q05 = float(s.quantile(0.05))
    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))
    q95 = float(s.quantile(0.95))
    q99 = float(s.quantile(0.99))
    
    iqr = q75 - q25
    mad = float((s - median).abs().median())
    robust_sigma = 1.4826 * mad if mad > 0 else (std if std > 0 else 1.0)
    
    skew = float(stats.skew(s))
    kurt = float(stats.kurtosis(s))  # Fisher excess kurtosis
    
    share_zeros = float((s == 0.0).sum() / n * 100.0)
    beyond_3_sigma = float(((s - median).abs() > 3.0 * robust_sigma).sum() / n * 100.0)
    beyond_5_sigma = float(((s - median).abs() > 5.0 * robust_sigma).sum() / n * 100.0)
    
    try:
        jb_stat, jb_pval = stats.jarque_bera(s)
        jb_stat = float(jb_stat)
        jb_pval = float(jb_pval)
    except Exception:
        jb_stat, jb_pval = np.nan, np.nan
        
    return {
        "variable": name,
        "count": n,
        "mean": mean,
        "std": std,
        "min": min_val,
        "1%": q01,
        "5%": q05,
        "25%": q25,
        "median": median,
        "75%": q75,
        "95%": q95,
        "99%": q99,
        "max": max_val,
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "mad": mad,
        "iqr": iqr,
        "share_zeros_pct": round(share_zeros, 4),
        "share_beyond_3_robust_sigma_pct": round(beyond_3_sigma, 4),
        "share_beyond_5_robust_sigma_pct": round(beyond_5_sigma, 4),
        "jarque_bera_stat": round(jb_stat, 2) if not np.isnan(jb_stat) else np.nan,
        "jarque_bera_pval": jb_pval,
    }


def compute_all_variable_statistics(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics table for market columns, base targets, and all 54 features."""
    verify_train_only(df_train)
    records = []
    
    # 1. Base market variables
    c = df_train["close"]
    o = df_train["open"]
    h = df_train["high"]
    l = df_train["low"]
    v = df_train["volume"]
    tr = df_train["number_of_trades"]
    tbv = df_train["taker_buy_base_volume"]
    
    ret_5m = np.log(c / c.shift(1))
    abs_ret_5m = ret_5m.abs()
    hl_spread = (h - l) / c
    log_vol = np.log1p(v.clip(lower=0))
    avg_trade_size = v / tr.replace(0, np.nan)
    taker_buy_ratio = tbv / v.replace(0, np.nan)
    
    base_vars = {
        "5m_log_return": ret_5m,
        "5m_abs_return": abs_ret_5m,
        "hl_spread_pct": hl_spread,
        "volume_btc": v,
        "log_volume": log_vol,
        "number_of_trades": tr,
        "avg_trade_size_btc": avg_trade_size,
        "taker_buy_ratio": taker_buy_ratio,
    }
    for name, s in base_vars.items():
        records.append(compute_variable_statistics(s, name))
        
    # 2. Forward returns
    for N in config.CANDIDATE_HORIZONS:
        if f"fwd_ret_{N}" in df_train.columns:
            records.append(compute_variable_statistics(df_train[f"fwd_ret_{N}"], f"fwd_ret_{N}"))
            
    # 3. All 54 engineered features
    for col in ALL_FEATURE_COLUMNS:
        if col in df_train.columns:
            records.append(compute_variable_statistics(df_train[col], col))
            
    return pd.DataFrame(records)


# Alias
compute_variable_statistics_table = compute_all_variable_statistics


def compute_temporal_drift_table(
    data: Union[pd.Series, pd.DataFrame],
    freq: str = "QE",
) -> pd.DataFrame:
    """Compute mean, std, min, max, count across periods to measure drift."""
    if isinstance(data, pd.DataFrame):
        s = np.log(data["close"] / data["close"].shift(1)).dropna()
    else:
        s = data.dropna()
    grouped = s.groupby(pd.Grouper(freq=freq))
    drift = grouped.agg(["count", "mean", "std", "min", "max"])
    drift.index = [f"{d.year}-Q{d.quarter}" if freq == "QE" else str(d) for d in drift.index]
    return drift.reset_index().rename(columns={"index": "period"})


def compute_acf_manual(series: pd.Series, nlags: int = 288) -> np.ndarray:
    """Compute sample Autocorrelation Function (ACF) on full series using BLAS dot products."""
    s = series.dropna().to_numpy(dtype=np.float64)
    n = len(s)
    mean = np.mean(s)
    var = np.var(s)
    if var == 0:
        return np.ones(nlags + 1)
    s_centered = s - mean
    corrs = np.empty(nlags + 1, dtype=np.float64)
    corrs[0] = 1.0
    for k in range(1, nlags + 1):
        corrs[k] = np.dot(s_centered[:-k], s_centered[k:]) / (n * var)
    return corrs


def compute_pacf_manual(acf_vals: np.ndarray, nlags: int = 50) -> np.ndarray:
    """Compute Partial Autocorrelation Function (PACF) using Durbin-Levinson recursion."""
    phi = np.zeros((nlags + 1, nlags + 1))
    v = np.zeros(nlags + 1)
    phi[1, 1] = acf_vals[1]
    v[1] = 1.0 - acf_vals[1]**2
    for k in range(2, nlags + 1):
        num = acf_vals[k] - np.sum(phi[k-1, 1:k] * acf_vals[k-1:0:-1])
        phi[k, k] = num / v[k-1]
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
        v[k] = v[k-1] * (1.0 - phi[k, k]**2)
    pacf = np.array([1.0] + [phi[k, k] for k in range(1, nlags + 1)])
    return pacf


def compute_vif_features(df_features: pd.DataFrame, reg: float = 1e-5) -> pd.Series:
    """Compute Variance Inflation Factor (VIF) for numeric feature DataFrame.
    
    Uses ridge regularization (reg * I) on the correlation matrix to ensure
    numerical stability and strictly non-negative VIF values >= 1.0 across all features.
    """
    X = df_features.dropna().select_dtypes(include=np.number)
    X_std = (X - X.mean()) / X.std().replace(0, np.nan)
    corr_matrix = X_std.corr().to_numpy()
    reg_matrix = corr_matrix + reg * np.eye(len(corr_matrix))
    inv_corr = np.linalg.inv(reg_matrix)
    vifs = np.diag(inv_corr)
    return pd.Series(vifs, index=X.columns)


def compute_label_balance_all_rules(
    df_train: pd.DataFrame,
    cost: float = config.TOTAL_ROUNDTRIP_COST,
) -> pd.DataFrame:
    """Compute complete 40-row label balance table across 4 horizons x 10 rules.

    Includes standard cost-floored rules (S1, S2, Q1, V1, V2, A1, A2) and
    research sub-cost rules (F05, F10, F15).
    """
    verify_train_only(df_train)
    horizons = config.CANDIDATE_HORIZONS
    rules = config.ALL_RULES
    records = []
    n_total = len(df_train)

    for N in horizons:
        valid_col = f"has_valid_target_{N}"
        valid_mask = df_train[valid_col].astype(bool) if valid_col in df_train.columns else pd.Series(True, index=df_train.index)
        n_valid = int(valid_mask.sum())
        n_invalid = n_total - n_valid

        for rule in rules:
            target_col = f"target_{N}_{rule}"
            tu_col = f"theta_up_{N}_{rule}"
            td_col = f"theta_down_{N}_{rule}"

            tgt = df_train.loc[valid_mask, target_col]

            # Determine thetas
            if tu_col in df_train.columns:
                tu = df_train.loc[valid_mask, tu_col]
                td = df_train.loc[valid_mask, td_col]
            else:
                # Constant thetas from rules
                if rule in ("S1", "Q1"):
                    tu = pd.Series(cost, index=tgt.index)
                    td = pd.Series(cost, index=tgt.index)
                elif rule == "S2":
                    tu = pd.Series(0.0030, index=tgt.index)
                    td = pd.Series(0.0030, index=tgt.index)
                elif rule == "A1":
                    tu = pd.Series(cost, index=tgt.index)
                    td = pd.Series(0.0030, index=tgt.index)
                elif rule == "A2":
                    tu = pd.Series(0.0030, index=tgt.index)
                    td = pd.Series(cost, index=tgt.index)
                elif rule == "F05":
                    tu = pd.Series(config.F05_THETA, index=tgt.index)
                    td = pd.Series(config.F05_THETA, index=tgt.index)
                elif rule == "F10":
                    tu = pd.Series(config.F10_THETA, index=tgt.index)
                    td = pd.Series(config.F10_THETA, index=tgt.index)
                elif rule == "F15":
                    tu = pd.Series(config.F15_THETA, index=tgt.index)
                    td = pd.Series(config.F15_THETA, index=tgt.index)
                else:
                    tu = pd.Series(cost, index=tgt.index)
                    td = pd.Series(cost, index=tgt.index)

            n_up = int((tgt == 1.0).sum())
            n_down = int((tgt == -1.0).sum())
            n_neutral = int((tgt == 0.0).sum())

            pct_up = (n_up / n_valid) * 100.0 if n_valid > 0 else 0.0
            pct_down = (n_down / n_valid) * 100.0 if n_valid > 0 else 0.0
            pct_neutral = (n_neutral / n_valid) * 100.0 if n_valid > 0 else 0.0
            pct_invalid = (n_invalid / n_total) * 100.0

            up_down_ratio = (n_up / n_down) if n_down > 0 else np.nan

            # Shannon Entropy: - sum(p * log2(p))
            probs = [p / n_valid for p in [n_up, n_down, n_neutral] if p > 0]
            entropy = -sum(p * np.log2(p) for p in probs) if probs else 0.0

            # Imbalance ratio: max class / min class
            counts = [c for c in [n_up, n_down, n_neutral] if c > 0]
            imbalance_ratio = (max(counts) / min(counts)) if counts else np.nan

            # Floor binding (or sub-cost status)
            is_sub_cost = (rule in config.SUB_COST_RULES)
            floor_bind_up = int((tu <= cost + 1e-10).sum())
            pct_floor_bind = (floor_bind_up / n_valid) * 100.0 if n_valid > 0 else 0.0

            # Baseline accuracies
            majority_acc = max(n_up, n_down, n_neutral) / n_valid * 100.0 if n_valid > 0 else 0.0
            always_up_acc = n_up / n_valid * 100.0 if n_valid > 0 else 0.0

            records.append({
                "label_col": target_col,
                "horizon": N,
                "theta_rule": rule,
                "sub_cost": is_sub_cost,
                "is_identical_to_S1": (rule == "Q1"),
                "n_valid": n_valid,
                "n_invalid": n_invalid,
                "n_up": n_up,
                "n_down": n_down,
                "n_neutral": n_neutral,
                "active_rows": n_up + n_down,
                "active_pct": round((n_up + n_down) / n_valid * 100.0, 2) if n_valid > 0 else 0.0,
                "pct_up": round(pct_up, 2),
                "pct_down": round(pct_down, 2),
                "pct_neutral": round(pct_neutral, 2),
                "pct_invalid": round(pct_invalid, 4),
                "up_down_ratio": round(up_down_ratio, 3) if not np.isnan(up_down_ratio) else np.nan,
                "entropy": round(entropy, 3),
                "imbalance_ratio": round(imbalance_ratio, 2) if not np.isnan(imbalance_ratio) else np.nan,
                "majority_class_accuracy_pct": round(majority_acc, 2),
                "always_up_accuracy_pct": round(always_up_acc, 2),
                "theta_up_mean_pct": round(float(tu.mean()) * 100, 4),
                "theta_up_min_pct": round(float(tu.min()) * 100, 4),
                "theta_up_median_pct": round(float(tu.median()) * 100, 4),
                "theta_up_max_pct": round(float(tu.max()) * 100, 4),
                "theta_down_mean_pct": round(float(td.mean()) * 100, 4),
                "pct_floor_bind": round(pct_floor_bind, 2),
            })

    return pd.DataFrame(records)


# Alias for backward compatibility
compute_label_balance_28_rules = compute_label_balance_all_rules


# Alias
compute_variable_statistics_table = compute_all_variable_statistics


def compute_label_persistence(target_series: pd.Series) -> Dict[str, Any]:
    """Compute run lengths and transition probabilities between classes."""
    s = target_series.dropna()
    if len(s) < 2:
        return {}
    
    same_next = (s.iloc[:-1].to_numpy() == s.iloc[1:].to_numpy())
    p_same = float(same_next.mean())
    
    diffs = (s != s.shift(1)).cumsum()
    runs = s.groupby(diffs).size()
    mean_run = float(runs.mean())
    max_run = int(runs.max())
    p95_run = float(runs.quantile(0.95))
    
    classes = [-1.0, 0.0, 1.0]
    t_from = s.iloc[:-1].to_numpy()
    t_to = s.iloc[1:].to_numpy()
    t_df = pd.DataFrame({"from": t_from, "to": t_to})
    trans = pd.crosstab(t_df["from"], t_df["to"], normalize="index")
    trans = trans.reindex(index=classes, columns=classes, fill_value=0.0)
    
    return {
        "p_same_label_next": round(p_same, 4),
        "mean_run_length": round(mean_run, 2),
        "p95_run_length": round(p95_run, 1),
        "max_run_length": max_run,
        "transition_matrix": trans.to_dict(),
    }


def compute_class_conditional_feature_stats(
    df_features: pd.DataFrame,
    target_series: pd.Series,
) -> pd.DataFrame:
    """Compute class-conditional mean, std, and Cohen's d (Up vs Down) for candidate features."""
    valid_idx = target_series.dropna().index.intersection(df_features.dropna().index)
    tgt = target_series.loc[valid_idx]
    X = df_features.loc[valid_idx]
    
    records = []
    for col in X.columns:
        s = X[col]
        up_vals = s[tgt == 1.0].dropna()
        down_vals = s[tgt == -1.0].dropna()
        neut_vals = s[tgt == 0.0].dropna()
        
        m_up, s_up = up_vals.mean(), up_vals.std()
        m_down, s_down = down_vals.mean(), down_vals.std()
        m_neut, s_neut = neut_vals.mean(), neut_vals.std()
        
        n_up, n_down = len(up_vals), len(down_vals)
        if n_up > 1 and n_down > 1:
            s_pooled = np.sqrt(((n_up - 1)*s_up**2 + (n_down - 1)*s_down**2) / (n_up + n_down - 2))
            cohen_d = (m_up - m_down) / s_pooled if s_pooled > 0 else 0.0
        else:
            cohen_d = np.nan
            
        records.append({
            "feature": col,
            "mean_up": round(m_up, 4),
            "std_up": round(s_up, 4),
            "mean_down": round(m_down, 4),
            "std_down": round(s_down, 4),
            "mean_neutral": round(m_neut, 4),
            "std_neutral": round(s_neut, 4),
            "cohens_d_up_vs_down": round(cohen_d, 4),
        })
    return pd.DataFrame(records)


def compute_anomalies_mad(series: pd.Series, multiplier: float = 3.0) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute robust MAD-based anomalies for a series (e.g. 5m returns) on Train only.

    Parameters
    ----------
    series : pd.Series
        Time-series data (e.g. log returns).
    multiplier : float
        Threshold multiplier for robust sigma (default 3.0).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Summary DataFrame and boolean mask of anomaly candles.
    """
    clean_s = series.dropna()
    med = float(clean_s.median())
    mad = float((clean_s - med).abs().median())
    robust_sigma = 1.4826 * mad if mad > 0 else float(clean_s.std())
    anom_mask = (series - med).abs() > multiplier * robust_sigma
    n_anom = int(anom_mask.sum())
    n_total = len(clean_s)

    total_var = float(clean_s.var() * n_total)
    anom_var = float(clean_s[anom_mask].var() * n_anom) if n_anom > 0 else 0.0
    var_share = (anom_var / total_var) * 100.0 if total_var > 0 else 0.0

    kurt_full = float(stats.kurtosis(clean_s))
    kurt_clean = float(stats.kurtosis(clean_s[~anom_mask]))

    summary_df = pd.DataFrame([{
        "anomaly_rule": f"Train Robust MAD ({multiplier:.1f} * robust_sigma)",
        "flagged_candles": n_anom,
        "pct_of_train_candles": round(n_anom / n_total * 100, 2),
        "variance_share_pct": round(var_share, 2),
        "excess_kurtosis_with_anomalies": round(kurt_full, 2),
        "excess_kurtosis_without_anomalies": round(kurt_clean, 2),
        "step01_rolling_z_comparison": 1718,
        "note_on_difference": (
            "Step 1 flagged 1,718 on the full 318,240-candle series using rolling 24h z-scores (|z| > 5), "
            f"whereas Step 3 flags {n_anom:,} on Train only using static global {multiplier:.1f} * MAD."
        ),
    }])
    return summary_df, anom_mask


# ===========================================================================
# SUB-COST RULES ANALYSIS & SECTION B (BIG_MOVE_N EDA)
# ===========================================================================

def compute_sub_cost_loss_share(
    df_train: pd.DataFrame,
    cost: float = config.TOTAL_ROUNDTRIP_COST,
) -> pd.DataFrame:
    """Compute extra active rows and sub-cost loss share for research rules F05, F10, F15."""
    verify_train_only(df_train)
    records = []
    horizons = config.CANDIDATE_HORIZONS
    rules = ["S1", "F15", "F10", "F05"]

    for N in horizons:
        valid_col = f"has_valid_target_{N}"
        valid_mask = df_train[valid_col].astype(bool) if valid_col in df_train.columns else pd.Series(True, index=df_train.index)
        fwd_ret = df_train.loc[valid_mask, f"fwd_ret_{N}"]
        abs_ret = fwd_ret.abs()
        n_v = int(valid_mask.sum())

        # Baseline S1 (c = 0.23%)
        tgt_s1 = df_train.loc[valid_mask, f"target_{N}_S1"]
        n_act_s1 = int((tgt_s1 != 0.0).sum())

        for rule in rules:
            col = f"target_{N}_{rule}"
            tgt = df_train.loc[valid_mask, col]
            n_act = int((tgt != 0.0).sum())
            pct_act = (n_act / n_v * 100.0) if n_v > 0 else 0.0
            extra_rows = n_act - n_act_s1
            extra_pct = (extra_rows / n_act_s1 * 100.0) if n_act_s1 > 0 else 0.0

            # Sub-cost active signals: active signal where |fwd_ret| < roundtrip cost c (0.23%)
            sub_cost_active = int(((tgt != 0.0) & (abs_ret < cost)).sum())
            sub_cost_loss_share = (sub_cost_active / n_act * 100.0) if n_act > 0 else 0.0

            records.append({
                "horizon": N,
                "rule": rule,
                "active_rows": n_act,
                "active_pct": round(pct_act, 2),
                "extra_rows_vs_S1": extra_rows,
                "extra_pct_vs_S1": round(extra_pct, 2),
                "sub_cost_loss_rows": sub_cost_active,
                "sub_cost_loss_share_pct": round(sub_cost_loss_share, 2),
            })
    return pd.DataFrame(records)


def compute_big_move_balance(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute binary class balance for big_move_N across all 4 candidate horizons on Train."""
    verify_train_only(df_train)
    records = []
    horizons = config.CANDIDATE_HORIZONS
    for N in horizons:
        col = f"big_move_{N}"
        valid_col = f"has_valid_target_{N}"
        valid_mask = df_train[valid_col].astype(bool) if valid_col in df_train.columns else df_train[col].notna()
        bm = df_train.loc[valid_mask, col]
        n_valid = int(valid_mask.sum())
        n_ones = int((bm == 1.0).sum())
        n_zeros = int((bm == 0.0).sum())
        pct_ones = (n_ones / n_valid * 100.0) if n_valid > 0 else 0.0
        pct_zeros = (n_zeros / n_valid * 100.0) if n_valid > 0 else 0.0
        eff_n = int(n_valid / N)

        records.append({
            "horizon": N,
            "valid_candles": n_valid,
            "big_move_1": n_ones,
            "big_move_0": n_zeros,
            "pct_big_move_1": round(pct_ones, 2),
            "pct_big_move_0": round(pct_zeros, 2),
            "effective_sample_size": eff_n,
            "overlap_factor": N,
        })
    return pd.DataFrame(records)


def compute_big_move_monthly(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly big_move_N incidence rate alongside rolling volatility on Train."""
    verify_train_only(df_train)
    records = []
    horizons = config.CANDIDATE_HORIZONS
    naive_idx = df_train.index.tz_localize(None) if df_train.index.tz is not None else df_train.index
    train_periods = naive_idx.to_period("M")
    months = train_periods.unique()

    for m in months:
        m_mask = (train_periods == m)
        df_m = df_train.loc[m_mask]
        rec = {
            "month": str(m),
            "n_candles": len(df_m),
            "avg_sigma_288": round(float(df_m["feat_vol_sigma_288"].mean() * 100), 4) if "feat_vol_sigma_288" in df_m.columns else np.nan,
        }
        for N in horizons:
            col = f"big_move_{N}"
            valid_col = f"has_valid_target_{N}"
            valid_mask = df_m[valid_col].astype(bool) if valid_col in df_m.columns else df_m[col].notna()
            bm = df_m.loc[valid_mask, col]
            rate = float((bm == 1.0).mean() * 100.0) if len(bm) > 0 else np.nan
            rec[f"big_move_{N}_pct"] = round(rate, 2)
        records.append(rec)
    return pd.DataFrame(records)


def compute_big_move_context(df_train: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyze big_move_N rates across intraday hour, day of week, and volume terciles."""
    verify_train_only(df_train)
    horizons = config.CANDIDATE_HORIZONS

    # 1. By hour of day (0-23 UTC)
    hours = df_train.index.hour
    hour_records = []
    for h in range(24):
        h_mask = (hours == h)
        rec = {"hour": h}
        for N in horizons:
            bm = df_train.loc[h_mask, f"big_move_{N}"].dropna()
            rec[f"big_move_{N}_pct"] = round(float((bm == 1.0).mean() * 100.0), 2) if len(bm) > 0 else np.nan
        hour_records.append(rec)
    df_hour = pd.DataFrame(hour_records)

    # 2. By day of week (0=Mon, 6=Sun)
    day_records = []
    dow = df_train.index.day_of_week if hasattr(df_train.index, "day_of_week") else df_train.index.dayofweek
    for d in range(7):
        d_mask = (dow == d)
        rec = {"day_of_week": d}
        for N in horizons:
            bm = df_train.loc[d_mask, f"big_move_{N}"].dropna()
            rec[f"big_move_{N}_pct"] = round(float((bm == 1.0).mean() * 100.0), 2) if len(bm) > 0 else np.nan
        day_records.append(rec)
    df_dow = pd.DataFrame(day_records)

    # 3. By volume tercile
    vol = df_train["volume"]
    q33, q66 = vol.quantile(0.333), vol.quantile(0.666)
    terciles = pd.Series("Mid", index=vol.index)
    terciles[vol <= q33] = "Low"
    terciles[vol > q66] = "High"
    terc_records = []
    for t in ["Low", "Mid", "High"]:
        t_mask = (terciles == t)
        rec = {"volume_tercile": t}
        for N in horizons:
            bm = df_train.loc[t_mask, f"big_move_{N}"].dropna()
            rec[f"big_move_{N}_pct"] = round(float((bm == 1.0).mean() * 100.0), 2) if len(bm) > 0 else np.nan
        terc_records.append(rec)
    df_terc = pd.DataFrame(terc_records)

    return {"by_hour": df_hour, "by_dow": df_dow, "by_tercile": df_terc}


def compute_big_move_persistence(df_train: pd.DataFrame, max_lags: int = 50) -> Dict[str, Any]:
    """Compute ACF and run-length statistics for big_move_N."""
    verify_train_only(df_train)
    horizons = config.CANDIDATE_HORIZONS
    acf_results = {}
    run_records = []

    for N in horizons:
        col = f"big_move_{N}"
        s = df_train[col].dropna()

        # ACF
        s_centered = s - s.mean()
        var_s = float(np.var(s))
        acfs = []
        for lag in range(1, max_lags + 1):
            cov = float((s_centered.iloc[:-lag].to_numpy() * s_centered.iloc[lag:].to_numpy()).mean())
            acfs.append(cov / var_s if var_s > 0 else 0.0)
        acf_results[f"N_{N}"] = acfs

        # Run lengths
        diffs = (s != s.shift(1)).cumsum()
        grouped = s.groupby(diffs)
        run_lengths = grouped.size()
        run_values = grouped.first()

        runs_1 = run_lengths[run_values == 1.0]
        runs_0 = run_lengths[run_values == 0.0]

        run_records.append({
            "horizon": N,
            "mean_run_1": round(float(runs_1.mean()), 2) if len(runs_1) > 0 else np.nan,
            "max_run_1": int(runs_1.max()) if len(runs_1) > 0 else 0,
            "p95_run_1": round(float(runs_1.quantile(0.95)), 1) if len(runs_1) > 0 else np.nan,
            "mean_run_0": round(float(runs_0.mean()), 2) if len(runs_0) > 0 else np.nan,
            "max_run_0": int(runs_0.max()) if len(runs_0) > 0 else 0,
            "p95_run_0": round(float(runs_0.quantile(0.95)), 1) if len(runs_0) > 0 else np.nan,
            "p_same_label_next": round(float((s.iloc[:-1].to_numpy() == s.iloc[1:].to_numpy()).mean()), 4),
        })

    return {"acf": acf_results, "runs": pd.DataFrame(run_records)}


def compute_big_move_feature_stats(df_train: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Compute Cohen's d and Spearman correlation of all 54 features with big_move_N."""
    verify_train_only(df_train)
    col = f"big_move_{horizon}"
    target = df_train[col].dropna()
    features = [c for c in ALL_FEATURE_COLUMNS if c in df_train.columns]

    records = []
    idx = target.index

    for f in features:
        s = df_train.loc[idx, f].dropna()
        common_idx = s.index
        y_c = target.loc[common_idx].values
        m1 = (y_c == 1.0)
        m0 = (y_c == 0.0)
        v1 = s.values[m1]
        v0 = s.values[m0]

        m_1, s_1 = float(np.mean(v1)), float(np.std(v1, ddof=1))
        m_0, s_0 = float(np.mean(v0)), float(np.std(v0, ddof=1))

        if len(v1) > 1 and len(v0) > 1:
            s_pooled = np.sqrt(((len(v1) - 1)*s_1**2 + (len(v0) - 1)*s_0**2) / (len(v1) + len(v0) - 2))
            cohen_d = (m_1 - m_0) / s_pooled if s_pooled > 0 else 0.0
        else:
            cohen_d = 0.0

        sp_corr, _ = stats.spearmanr(s.values, y_c)
        meta = get_column_meta(f)

        records.append({
            "feature": f,
            "group": meta.get("group", ""),
            "mean_big_move_1": round(m_1, 4),
            "std_big_move_1": round(s_1, 4),
            "mean_big_move_0": round(m_0, 4),
            "std_big_move_0": round(s_0, 4),
            "cohens_d": round(cohen_d, 4),
            "abs_cohens_d": round(abs(cohen_d), 4),
            "spearman_corr": round(float(sp_corr), 4),
        })

    df_out = pd.DataFrame(records).sort_values(by="abs_cohens_d", ascending=False).reset_index(drop=True)
    return df_out


def compute_big_move_vol_deciles(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute big_move incidence across 10 deciles of rolling volatility (feat_vol_sigma_288)."""
    from sklearn.metrics import roc_auc_score
    verify_train_only(df_train)
    horizons = config.CANDIDATE_HORIZONS
    vol_col = "feat_vol_sigma_288"
    s_vol = df_train[vol_col].dropna()

    deciles = pd.qcut(s_vol, q=10, labels=False) + 1

    decile_records = []
    auc_scores = {}

    for d in range(1, 11):
        d_mask = (deciles == d)
        rec = {
            "decile": d,
            "vol_min": round(float(s_vol[d_mask].min() * 100), 4),
            "vol_mean": round(float(s_vol[d_mask].mean() * 100), 4),
            "vol_max": round(float(s_vol[d_mask].max() * 100), 4),
            "n_candles": int(d_mask.sum()),
        }
        for N in horizons:
            bm = df_train.loc[s_vol.index[d_mask], f"big_move_{N}"].dropna()
            rec[f"big_move_{N}_pct"] = round(float((bm == 1.0).mean() * 100.0), 2) if len(bm) > 0 else np.nan
        decile_records.append(rec)

    for N in horizons:
        valid_df = df_train[[vol_col, f"big_move_{N}"]].dropna()
        y_true = valid_df[f"big_move_{N}"].values
        y_score = valid_df[vol_col].values
        auc = roc_auc_score(y_true, y_score)
        auc_scores[f"N_{N}"] = round(float(auc), 4)

    return pd.DataFrame(decile_records), auc_scores


def compute_big_move_direction(df_train: pd.DataFrame) -> pd.DataFrame:
    """Analyze directional Up/Down distribution conditioned on big_move_N == 1 vs overall raw_sign."""
    verify_train_only(df_train)
    records = []
    for N in config.CANDIDATE_HORIZONS:
        valid_mask = df_train[f"has_valid_target_{N}"].astype(bool) if f"has_valid_target_{N}" in df_train.columns else pd.Series(True, index=df_train.index)
        fwd_ret = df_train.loc[valid_mask, f"fwd_ret_{N}"]
        bm = df_train.loc[valid_mask, f"big_move_{N}"]
        raw_sign = df_train.loc[valid_mask, f"raw_sign_{N}"]

        bm_mask = (bm == 1.0)
        bm_ret = fwd_ret[bm_mask]
        n_bm = int(bm_mask.sum())
        n_up = int((bm_ret > 0).sum())
        n_down = int((bm_ret < 0).sum())
        pct_up = (n_up / n_bm * 100.0) if n_bm > 0 else 0.0
        pct_down = (n_down / n_bm * 100.0) if n_bm > 0 else 0.0

        n_raw_pos = int((raw_sign == 1.0).sum())
        pct_raw_pos = (n_raw_pos / len(raw_sign) * 100.0) if len(raw_sign) > 0 else 0.0

        records.append({
            "horizon": N,
            "big_move_candles": n_bm,
            "up_count": n_up,
            "down_count": n_down,
            "up_pct_inside_big_move": round(pct_up, 2),
            "down_pct_inside_big_move": round(pct_down, 2),
            "up_down_ratio": round(n_up / n_down, 3) if n_down > 0 else np.nan,
            "overall_raw_sign_positive_pct": round(pct_raw_pos, 2),
        })
    return pd.DataFrame(records)
