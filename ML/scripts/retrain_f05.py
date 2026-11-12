"""Retrain Three-Class Models and Meta-Model on Target F05 (Horizon N=2)."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from src import config
config.THREE_CLASS_RULE = "F05"
config.FORCE_RETRAIN = True
HORIZON = 2

from src.model_common import run_oof, load_truth, fit_final, prepare_model_features
from src.meta_model import build_meta_inputs, nested_meta_oof, fit_meta_final, META_INPUT_COLUMNS
from src.column_dictionary import load_master
from src.ta_engine import compute_ta_scores, ta_label, TA_SCORE_COLUMNS

print("=" * 80)
print(f"RETRAINING THREE-CLASS & META-MODELS ON TARGET {config.THREE_CLASS_RULE} (HORIZON {HORIZON})")
print("=" * 80)

# 1. Three-Class LR
t0 = time.time()
print("\n[1/6] Retraining three_class_lr on F05...")
lr_oof = run_oof("three_class_lr", horizon=HORIZON)
print(f"  Saved: {lr_oof} ({time.time() - t0:.1f}s)")

# 2. Three-Class RF
t0 = time.time()
print("\n[2/6] Retraining three_class_rf on F05...")
rf_oof = run_oof("three_class_rf", horizon=HORIZON)
print(f"  Saved: {rf_oof} ({time.time() - t0:.1f}s)")

# 3. Three-Class GB
t0 = time.time()
print("\n[3/6] Retraining three_class_gb on F05...")
gb_oof = run_oof("three_class_gb", horizon=HORIZON)
print(f"  Saved: {gb_oof} ({time.time() - t0:.1f}s)")

# 4. Meta Inputs & Nested Meta OOF
t0 = time.time()
print("\n[4/6] Building meta inputs and running nested_meta_oof on F05...")
meta_df = build_meta_inputs(horizon=HORIZON)
meta_oof = nested_meta_oof(horizon=HORIZON, meta_df=meta_df)
fit_meta_final(horizon=HORIZON, meta_df=meta_df)
print(f"  Meta OOF saved: {meta_oof} ({time.time() - t0:.1f}s)")

# 5. Fit Final Deployment Base Models
t0 = time.time()
print("\n[5/6] Fitting final deployment models on full train set (F05)...")
for mid in ["three_class_lr", "three_class_rf", "three_class_gb"]:
    joblib_p, json_p = fit_final(mid, horizon=HORIZON)
    print(f"  Fit {mid} -> {joblib_p.name}")
print(f"  Base models done in {time.time() - t0:.1f}s")

# 6. Test Partition Forward Inference & horizon_2_test.csv Export
t0 = time.time()
print("\n[6/6] Generating horizon_2_test.csv with F05 predictions...")
df_test = load_master(split="test", unlock_test=True)
models_dir = config.MODELS_DIR / f"horizon_{HORIZON}"

def score_base(mid: str, df: pd.DataFrame):
    p = joblib.load(models_dir / f"{mid}_N{HORIZON}.joblib")
    clf = p["model"]
    scaler = p.get("scaler")
    X = prepare_model_features(df, p["feature_cols"])
    X_eval = scaler.transform(X) if scaler is not None else X
    return clf.predict_proba(X_eval)

bin_lr_p_up = score_base("binary_lr", df_test)[:, 1]
bin_rf_p_up = score_base("binary_rf", df_test)[:, 1]
bin_gb_p_up = score_base("binary_gb", df_test)[:, 1]

tri_lr_probs = score_base("three_class_lr", df_test)
tri_rf_probs = score_base("three_class_rf", df_test)
tri_gb_probs = score_base("three_class_gb", df_test)

ta_params_path = models_dir / f"ta_params_N{HORIZON}.json"
ta_scores_df = compute_ta_scores(df_test, params_path=ta_params_path, save_params=False, horizon=HORIZON)
ta_labels_series = ta_label(ta_scores_df)

meta_p = joblib.load(models_dir / f"meta_model_N{HORIZON}.joblib")
meta_clf = meta_p["model"]
meta_scaler = meta_p["scaler"]

meta_dict = {
    "bin_lr_p_up": bin_lr_p_up,
    "bin_rf_p_up": bin_rf_p_up,
    "bin_gb_p_up": bin_gb_p_up,
    "tri_lr_p_up": tri_lr_probs[:, 2],
    "tri_lr_p_down": tri_lr_probs[:, 0],
    "tri_rf_p_up": tri_rf_probs[:, 2],
    "tri_rf_p_down": tri_rf_probs[:, 0],
    "tri_gb_p_up": tri_gb_probs[:, 2],
    "tri_gb_p_down": tri_gb_probs[:, 0],
}
for c in TA_SCORE_COLUMNS:
    meta_dict[c] = ta_scores_df[c].values

meta_inputs = pd.DataFrame(meta_dict, index=df_test.index)[META_INPUT_COLUMNS]
meta_eval = meta_scaler.transform(meta_inputs)
final_probs = meta_clf.predict_proba(meta_eval)
final_preds = meta_clf.classes_[np.argmax(final_probs, axis=1)]
final_confidence = np.max(final_probs, axis=1)

label_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
final_labels = [label_map[p] for p in final_preds]

res_df = pd.DataFrame(index=df_test.index)
res_df.index.name = "open_time"

res_df["open"] = df_test["open"]
res_df["high"] = df_test["high"]
res_df["low"] = df_test["low"]
res_df["close"] = df_test["close"]
res_df["volume"] = df_test["volume"]

res_df["true_binary"] = df_test[f"bin_dir_{HORIZON}"]
res_df["true_three_class"] = df_test[f"target_{HORIZON}_{config.THREE_CLASS_RULE}"]
res_df["true_raw_sign"] = df_test[f"raw_sign_{HORIZON}"]
res_df["fwd_ret_open"] = df_test[f"fwd_ret_open_{HORIZON}"]

res_df["bin_lr_pred"] = (bin_lr_p_up >= 0.5).astype(int)
res_df["bin_lr_p_up"] = np.round(bin_lr_p_up, 4)
res_df["bin_rf_pred"] = (bin_rf_p_up >= 0.5).astype(int)
res_df["bin_rf_p_up"] = np.round(bin_rf_p_up, 4)
res_df["bin_gb_pred"] = (bin_gb_p_up >= 0.5).astype(int)
res_df["bin_gb_p_up"] = np.round(bin_gb_p_up, 4)

class_vals = np.array([-1, 0, 1])
res_df["tri_lr_pred"] = class_vals[np.argmax(tri_lr_probs, axis=1)]
res_df["tri_lr_p_down"] = np.round(tri_lr_probs[:, 0], 4)
res_df["tri_lr_p_neutral"] = np.round(tri_lr_probs[:, 1], 4)
res_df["tri_lr_p_up"] = np.round(tri_lr_probs[:, 2], 4)

res_df["tri_rf_pred"] = class_vals[np.argmax(tri_rf_probs, axis=1)]
res_df["tri_rf_p_down"] = np.round(tri_rf_probs[:, 0], 4)
res_df["tri_rf_p_neutral"] = np.round(tri_rf_probs[:, 1], 4)
res_df["tri_rf_p_up"] = np.round(tri_rf_probs[:, 2], 4)

res_df["tri_gb_pred"] = class_vals[np.argmax(tri_gb_probs, axis=1)]
res_df["tri_gb_p_down"] = np.round(tri_gb_probs[:, 0], 4)
res_df["tri_gb_p_neutral"] = np.round(tri_gb_probs[:, 1], 4)
res_df["tri_gb_p_up"] = np.round(tri_gb_probs[:, 2], 4)

for c in TA_SCORE_COLUMNS:
    res_df[c] = np.round(ta_scores_df[c].values, 4)
res_df["ta_label"] = ta_labels_series.values

res_df["final_prediction"] = final_labels
res_df["final_p_down"] = np.round(final_probs[:, 0], 4)
res_df["final_p_neutral"] = np.round(final_probs[:, 1], 4)
res_df["final_p_up"] = np.round(final_probs[:, 2], 4)
res_df["final_confidence"] = np.round(final_confidence, 4)

out_path = config.RESULTS_DIR / "horizon_2_test.csv"
try:
    res_df.to_csv(out_path)
    res_df.to_csv(Path("horizon_2_test.csv"))
    print(f"Exported: {out_path} ({len(res_df):,} rows x {len(res_df.columns)} columns)")
except PermissionError:
    fallback_path = config.RESULTS_DIR / "horizon_2_test_latest.csv"
    res_df.to_csv(fallback_path)
    res_df.to_csv(Path("horizon_2_test_latest.csv"))
    print(f"Exported fallback: {fallback_path} ({len(res_df):,} rows x {len(res_df.columns)} columns)")

print(f"\nALL MODELS AND TEST DATASET RETRAINED ON F05 SUCCESSFULLY IN {time.time() - t0:.1f}s!")

