"""Compute true non-circular out-of-sample direction stability.

Selection & orientation frozen strictly on Train rows up to 2024-06-30.
Evaluated on post-gap slices (Quarters & Half-Years) with slice-matched circular noise bands.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.column_dictionary import load_master, get_feature_columns

def main():
    print("Loading master.csv train partition...")
    df_train = load_master(split="train")
    feature_cols = get_feature_columns(df_train)

    # 1. In-Sample slice: up to 2024-06-30 23:55:00
    is_cutoff = pd.Timestamp("2024-06-30 23:55:00", tz="UTC")
    df_is_full = df_train.loc[df_train.index <= is_cutoff]
    df_is_valid = df_is_full.iloc[config.LOOKBACK_CANDLES:].copy()
    print(f"IS slice rows: {len(df_is_valid):,} candles (from {df_is_valid.index[0]} to {df_is_valid.index[-1]})")

    # 2. IS Circular Shift Test (500 shifts >= 2018 candles)
    rng = np.random.default_rng(42)
    y_is = (df_is_valid["raw_sign_12"].to_numpy() == 1.0).astype(float)
    n_is = len(y_is)
    shifts_is = rng.integers(config.LOOKBACK_CANDLES, n_is - config.LOOKBACK_CANDLES, size=500)

    R_is = np.empty((len(feature_cols), n_is))
    for i, c in enumerate(feature_cols):
        R_is[i] = pd.Series(df_is_valid[c]).rank(method="average").to_numpy()

    n1_is = float(np.sum(y_is))
    n0_is = float(n_is - n1_is)
    c_const_is = n1_is * (n1_is + 1.0) / 2.0
    denom_is = max(n1_is * n0_is, 1e-12)

    # Raw IS AUCs and orientation
    raw_aucs_is = (R_is @ y_is - c_const_is) / denom_is
    eff_aucs_is = np.maximum(raw_aucs_is, 1.0 - raw_aucs_is)
    is_orientation = {col: (1 if raw_aucs_is[i] >= 0.5 else -1) for i, col in enumerate(feature_cols)}

    # Circular shift noise distribution on IS
    auc_matrix_is = np.empty((500, len(feature_cols)))
    for s_idx, s in enumerate(shifts_is):
        y_s = np.roll(y_is, s)
        raw_s = (R_is @ y_s - c_const_is) / denom_is
        auc_matrix_is[s_idx] = np.maximum(raw_s, 1.0 - raw_s)

    max_auc_is = auc_matrix_is.max(axis=1)
    is_max_99 = float(np.percentile(max_auc_is, 99))
    is_max_95 = float(np.percentile(max_auc_is, 95))
    print(f"IS Circular Max Noise: 95th={is_max_95:.4f}, 99th={is_max_99:.4f}")

    # Select features exceeding IS 99th circular max
    selected_is = [col for i, col in enumerate(feature_cols) if eff_aucs_is[i] > is_max_99]
    print(f"Features exceeding IS 99th circular max: {len(selected_is)} / {len(feature_cols)}")

    # Cluster selected features at |rho| > 0.90 on IS period
    sub_corr_is = df_is_valid[selected_is].corr(method="spearman").abs()
    dist_is = np.clip(1.0 - sub_corr_is.values, 0.0, 1.0)
    np.fill_diagonal(dist_is, 0.0)
    condensed_is = squareform(dist_is)
    Z_is = linkage(condensed_is, method="complete")
    clusters_is = fcluster(Z_is, t=0.10, criterion="distance")

    cluster_df_is = pd.DataFrame({
        "feature": selected_is,
        "cluster": clusters_is,
        "is_auc": [eff_aucs_is[feature_cols.index(c)] for c in selected_is]
    })
    reps_is = []
    for cid in sorted(cluster_df_is["cluster"].unique()):
        top_in_cluster = cluster_df_is[cluster_df_is["cluster"] == cid].sort_values(by="is_auc", ascending=False).iloc[0]
        reps_is.append(top_in_cluster["feature"])

    print(f"\nIndependent clusters on IS period: {len(reps_is)}")
    for r in reps_is:
        orient_str = "+" if is_orientation[r] == 1 else "-"
        print(f"  Rep: {r:35s} | IS AUC: {eff_aucs_is[feature_cols.index(r)]:.4f} | Orient: {orient_str}")

    # 3. Out-of-Sample Slices: after purge gap of LOOKBACK_CANDLES
    oos_start = is_cutoff + pd.Timedelta(minutes=5 * config.LOOKBACK_CANDLES)
    df_oos_full = df_train.loc[df_train.index > oos_start].copy()
    print(f"\nPurge Gap: {config.LOOKBACK_CANDLES} candles ({config.LOOKBACK_CANDLES*5/1440:.1f} days, until {oos_start})")
    print(f"OOS full rows: {len(df_oos_full):,} candles (from {df_oos_full.index[0]} to {df_oos_full.index[-1]})")

    # Define Slices: Quarters and Half-Years
    quarters_oos = df_oos_full.index.tz_localize(None).to_period("Q")
    unique_q = sorted(quarters_oos.unique())
    print(f"OOS Quarters: {[str(q) for q in unique_q]}")

    slices = {}
    for q in unique_q:
        slices[f"Quarter {q}"] = df_oos_full.loc[quarters_oos == q]

    # Half years
    slices["2024H2"] = df_oos_full.loc[(quarters_oos == "2024Q3") | (quarters_oos == "2024Q4")]
    slices["2025H1"] = df_oos_full.loc[(quarters_oos == "2025Q1") | (quarters_oos == "2025Q2")]
    slices["2025H2"] = df_oos_full.loc[(quarters_oos == "2025Q3") | (quarters_oos == "2025Q4")]

    # Evaluate each slice
    results = []
    for slice_name, slice_df in slices.items():
        n_slice = len(slice_df)
        y_slice = (slice_df["raw_sign_12"].to_numpy() == 1.0).astype(float)
        
        # Slice-specific circular shift noise band (500 shifts)
        # Shift offset >= 288 (at least 24h) or min(LOOKBACK_CANDLES, n_slice//4)
        min_offset = min(288, max(50, n_slice // 10))
        shifts_slice = rng.integers(min_offset, n_slice - min_offset, size=500)
        
        R_slice = np.empty((len(feature_cols), n_slice))
        for i, c in enumerate(feature_cols):
            R_slice[i] = pd.Series(slice_df[c]).rank(method="average").to_numpy()
            
        n1_sl = float(np.sum(y_slice))
        n0_sl = float(n_slice - n1_sl)
        c_const_sl = n1_sl * (n1_sl + 1.0) / 2.0
        denom_sl = max(n1_sl * n0_sl, 1e-12)
        
        # Raw slice AUCs
        raw_slice_aucs = (R_slice @ y_slice - c_const_sl) / denom_sl
        
        # Circular shift matrix for this slice
        auc_mat_sl = np.empty((500, len(feature_cols)))
        for s_idx, s in enumerate(shifts_slice):
            y_s = np.roll(y_slice, s)
            raw_s = (R_slice @ y_s - c_const_sl) / denom_sl
            auc_mat_sl[s_idx] = np.maximum(raw_s, 1.0 - raw_s)
            
        slice_max_99 = float(np.percentile(auc_mat_sl.max(axis=1), 99))
        slice_max_95 = float(np.percentile(auc_mat_sl.max(axis=1), 95))
        
        # Evaluate selected representatives using FROZEN IS orientation
        rep_aucs = []
        sign_preserved_count = 0
        above_slice_band_count = 0
        
        for r in reps_is:
            idx = feature_cols.index(r)
            raw_val = raw_slice_aucs[idx]
            orient = is_orientation[r]
            
            # Check sign consistency: raw_val >= 0.5 when orient == +1, or raw_val < 0.5 when orient == -1
            has_same_sign = (raw_val >= 0.5 and orient == 1) or (raw_val < 0.5 and orient == -1)
            if has_same_sign:
                sign_preserved_count += 1
                
            # Directional AUC in intended orientation
            eff_val = raw_val if orient == 1 else (1.0 - raw_val)
            rep_aucs.append(eff_val)
            
            if eff_val > slice_max_99:
                above_slice_band_count += 1
                
        mean_rep_auc = float(np.mean(rep_aucs))
        
        results.append({
            "slice": slice_name,
            "candles": n_slice,
            "slice_circular_max_95": slice_max_95,
            "slice_circular_max_99": slice_max_99,
            "mean_rep_auc_frozen_orient": mean_rep_auc,
            "sign_preserved_reps": f"{sign_preserved_count}/{len(reps_is)} ({sign_preserved_count/len(reps_is)*100:.1f}%)",
            "above_slice_99th_band": f"{above_slice_band_count}/{len(reps_is)} ({above_slice_band_count/len(reps_is)*100:.1f}%)",
        })

    df_res = pd.DataFrame(results)
    print("\n==========================================================================================")
    print("TRUE OUT-OF-SAMPLE STABILITY (IS trained up to 2024-06-30, OOS evaluated post-gap)")
    print("==========================================================================================")
    print(df_res.to_string(index=False))

    # Also compute per-feature OOS table for the representatives
    rep_records = []
    for r in reps_is:
        idx = feature_cols.index(r)
        orient = is_orientation[r]
        row = {
            "representative": r,
            "is_auc": eff_aucs_is[idx],
            "frozen_orient": "+" if orient == 1 else "-",
        }
        for sl_name in ["2024H2", "2025H1", "2025H2"]:
            sl_df = slices[sl_name]
            y_sl = (sl_df["raw_sign_12"].to_numpy() == 1.0).astype(int)
            x_sl = sl_df[r].to_numpy()
            raw_sl = roc_auc_score(y_sl, x_sl)
            eff_sl = raw_sl if orient == 1 else (1.0 - raw_sl)
            row[f"auc_{sl_name}"] = eff_sl
        rep_records.append(row)
        
    df_rep_oos = pd.DataFrame(rep_records)
    print("\n==========================================================================================")
    print("REPRESENTATIVE PERFORMANCE ACROSS OUT-OF-SAMPLE HALF-YEARS (FROZEN ORIENTATION)")
    print("==========================================================================================")
    print(df_rep_oos.to_string(index=False))

if __name__ == "__main__":
    main()

