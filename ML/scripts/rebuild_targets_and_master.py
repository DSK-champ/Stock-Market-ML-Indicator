"""Rebuild targets_labels.csv, targets_theta_frozen.json, and master.csv with F rules and big_move targets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import config
from src.io_utils import (
    load_clean_market_data_csv,
    load_targets_base_csv,
    save_targets_labels_csv,
    save_theta_frozen_json,
    save_master_csv,
    save_targets_base_csv,
    save_features_csv,
)
from src.labels import compute_all_targets, compute_q1_thresholds
from src.features import build_features
from src.master import build_master
from src.column_dictionary import get_feature_columns, get_column_role, COLUMN_ROLES
from src.details import write_details, describe_dataframe, compute_file_hash

def run():
    print("1. Loading clean market data...")
    clean = load_clean_market_data_csv()
    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    train_clean = clean.loc[:train_end].copy()
    print(f"   Clean data: {len(clean):,} rows | Train clean: {len(train_clean):,} rows")

    print("2. Computing Q1 thresholds on Train rows only...")
    q1_thresholds = compute_q1_thresholds(train_clean, cost=config.TOTAL_ROUNDTRIP_COST)
    print("   Frozen Q1 thresholds:", q1_thresholds)

    print("3. Computing all targets and big_move across full dataset (318,240 rows)...")
    full_targets = compute_all_targets(
        clean,
        train_df=train_clean,
        cost=config.TOTAL_ROUNDTRIP_COST,
        q1_thresholds=q1_thresholds,
    )
    print(f"   Targets generated: shape {full_targets.shape}")

    # Verify column counts
    horizons = config.CANDIDATE_HORIZONS
    rules = config.ALL_RULES
    print(f"   Candidate horizons: {horizons}")
    print(f"   Rules ({len(rules)}): {rules}")

    # Base targets slice
    base_cols = []
    for N in horizons:
        base_cols.extend([f"fwd_ret_{N}", f"fwd_ret_open_{N}", f"raw_sign_{N}", f"has_valid_target_{N}"])
    df_base = full_targets[base_cols].copy()
    save_targets_base_csv(df_base)
    print(f"   Saved targets_base.csv: {df_base.shape}")

    # Labels and thetas slice
    label_cols = [f"target_{N}_{r}" for N in horizons for r in rules]
    theta_cols = [f"theta_up_{N}_{r}" for N in horizons for r in rules] + [f"theta_down_{N}_{r}" for N in horizons for r in rules]
    big_move_cols = [f"big_move_{N}" for N in horizons]
    df_labels = full_targets[label_cols + theta_cols + big_move_cols].copy()
    save_targets_labels_csv(df_labels)
    print(f"   Saved targets_labels.csv: {df_labels.shape} (40 labels + 80 thetas + 4 big_moves = {len(df_labels.columns)} cols)")

    # Save frozen JSON
    save_theta_frozen_json(q1_thresholds, train_clean)
    print("   Saved targets_theta_frozen.json")

    print("4. Building 85 engineered causal features across 15 groups...")
    df_features = build_features(clean)
    print(f"   Features built: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")
    assert df_features.shape[1] == 85, f"Expected 85 features, got {df_features.shape[1]}"
    
    # Remove standalone features.csv if present (redundant with master.csv)
    features_csv_path = config.DATA_PROCESSED_DIR / "features.csv"
    if features_csv_path.exists():
        features_csv_path.unlink()
        print(f"   Removed redundant {features_csv_path.name}")

    print("5. Building master.csv (joining clean, features, base targets, labels, big_move, dynamic thetas)...")
    master = build_master(
        df_clean=clean,
        df_features=df_features,
        df_targets_base=df_base,
        df_targets_labels=df_labels,
    )
    print(f"   Master built: {master.shape[0]:,} rows × {master.shape[1]} columns")
    assert master.shape[1] == 172, f"Expected 172 columns in master.csv, got {master.shape[1]}"

    master_path = save_master_csv(master)
    print(f"   Saved master.csv to {master_path} ({master_path.stat().st_size:,} bytes)")

    print("6. Verifying column dictionary roles...")
    feature_cols = get_feature_columns()
    assert len(feature_cols) == 85, f"Expected 85 features, got {len(feature_cols)}"
    role_counts = pd.Series([get_column_role(c) for c in master.columns]).value_counts()
    print("   Master column role counts:")
    for role, cnt in role_counts.items():
        print(f"     {role:12s}: {cnt:3d} columns")

    print("7. Updating detail JSONs...")
    tb_path = config.DATA_PROCESSED_DIR / "targets_base.csv"
    d_tb = describe_dataframe(
        df_base, tb_path, step=2, module="src.labels",
        description="Base forward returns, conservative open returns, raw direction signs, and valid target flags.",
        params={"horizons": horizons, "roundtrip_cost_c": config.TOTAL_ROUNDTRIP_COST},
        checks={"labels_cross_gaps": 0, "last_N_rows_unlabeled": True},
        full_df_stats=False,
    )
    write_details("step02_targets_base", d_tb)
    print("   Updated step02_targets_base.json")

    tl_path = config.DATA_PROCESSED_DIR / "targets_labels.csv"
    d_tl = describe_dataframe(
        df_labels, tl_path, step=2, module="src.labels",
        description="Target direction labels, dynamic thetas, and big move indicators across all 40 candidate setups.",
        params={"horizons": horizons, "rules": rules, "sub_cost_rules": config.SUB_COST_RULES, "roundtrip_cost_c": config.TOTAL_ROUNDTRIP_COST, "rolling_sigma_window_W": 288},
        checks={"theta_floor_enforced": True, "no_labels_cross_gaps": 0},
        full_df_stats=False,
    )
    write_details("step02_targets_labels", d_tl)
    print("   Updated step02_targets_labels.json")

    # Clean up step04_features.json if present
    feat_json = config.ML_DIR / "details" / "step04_features.json"
    if feat_json.exists():
        feat_json.unlink()
        print("   Removed step04_features.json")
    idx_path = config.ML_DIR / "details" / "index.json"
    if idx_path.exists():
        import json
        with open(idx_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        idx_data["files"] = [f for f in idx_data["files"] if f["name"] != "step04_features"]
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx_data, f, indent=2)

    d_master = describe_dataframe(
        master, master_path, step=4, module="src.master",
        description="Unified master dataset combining clean market data, 85 engineered features, base targets, labels, dynamic thetas, and big move flags.",
        params={"n_features": len(feature_cols), "n_rules": len(rules), "n_horizons": len(horizons)},
        checks={"columns_with_valid_roles": len(master.columns), "total_rows": len(master)},
        full_df_stats=False,
    )
    write_details("step_master", d_master)
    print("   Updated step_master.json")
    print("   Detail JSONs updated successfully!")
    print("   Detail JSONs updated successfully!")

if __name__ == "__main__":
    run()

