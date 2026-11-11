import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.io_utils import load_clean_market_data_csv, load_master_csv
from src import config
from src.features import build_features, LOOKBACK_CANDLES

print("=== QUESTION 1: LOOKBACK_CANDLES and NaNs ===")
print(f"Current config.LOOKBACK_CANDLES = {config.LOOKBACK_CANDLES}")
print(f"Current features.LOOKBACK_CANDLES = {LOOKBACK_CANDLES}")

clean = load_clean_market_data_csv()
feats = pd.read_csv(config.FEATURES_PATH, parse_dates=["open_time"], index_col="open_time")

first_300 = feats.iloc[:300]
nan_counts = first_300.isna().sum()
nans_gt_0 = nan_counts[nan_counts > 0].sort_values(ascending=False)
print("\nFeatures with NaNs in first 300 rows:")
for col, cnt in nans_gt_0.items():
    print(f"  {col:30s}: {cnt:3d} NaNs (first valid row index: {cnt})")
print(f"\nTotal features with NaNs in first 300: {len(nans_gt_0)} / {len(feats.columns)}")

last_nan = {}
for col in feats.columns:
    s = feats[col].isna()
    if s.any():
        last_nan[col] = s.values.nonzero()[0].max()
print(f"Max row index of NaN across ANY feature in full dataset: {max(last_nan.values()) if last_nan else 'None'}")
print(f"All features strictly non-NaN starting at row index: {max(last_nan.values()) + 1 if last_nan else 0}")

print("\n=== QUESTION 2: Truncation Difference for EMA, RSI, ADX, MACD ===")
# Test lookback buffer L vs full history on 20 random rows
rng = np.random.default_rng(42)
test_rows = rng.integers(500, len(clean) - 100, size=20)
test_rows = np.sort(test_rows)

tested_cols = [
    "feat_mom_ema_ratio_10",
    "feat_mom_ema_ratio_20",
    "feat_mom_ema_ratio_50",
    "feat_mom_rsi_7",
    "feat_mom_rsi_14",
    "feat_mom_macd_line",
    "feat_mom_macd_signal",
    "feat_mom_macd_hist",
    "feat_mom_adx_14",
]

for L in [120, 288, 500]:
    max_diffs = {c: 0.0 for c in tested_cols}
    for r in test_rows:
        buf = clean.iloc[r - L : r + 1].copy()
        buf_feats = build_features(buf)
        last_row_buf = buf_feats.iloc[-1]
        full_row = feats.iloc[r]
        for c in tested_cols:
            diff = abs(last_row_buf[c] - full_row[c])
            if diff > max_diffs[c]:
                max_diffs[c] = diff
    print(f"\nLookback buffer L = {L}:")
    for c, d in max_diffs.items():
        print(f"  {c:25s}: max_abs_diff = {d:.6e}")
    print(f"  --> OVERALL MAX ABS DIFF (L={L}): {max(max_diffs.values()):.6e}")

print("\n=== QUESTION 3: MACD Code & Min/Median/Max ===")
macd_cols = ["feat_mom_macd_line", "feat_mom_macd_signal", "feat_mom_macd_hist"]
train_master = load_master_csv(split="train")
for c in macd_cols:
    s = train_master[c]
    print(f"  {c:22s} -> min: {s.min():.6f}, median: {s.median():.6f}, max: {s.max():.6f}")

print("\n=== QUESTION 4: feat_ret_open_gap Stats ===")
gap = train_master["feat_ret_open_gap"]
print(f"  feat_ret_open_gap -> mean: {gap.mean():.8f}, std: {gap.std():.8f}")
print(f"  exact zeros: {(gap == 0).sum()} / {len(gap)} ({((gap == 0).mean() * 100):.2f}%)")
print(f"  abs < 1e-6: {(gap.abs() < 1e-6).sum()} / {len(gap)} ({((gap.abs() < 1e-6).mean() * 100):.2f}%)")

print("\n=== QUESTION 7: target_N_rule on Invalid Rows ===")
master = load_master_csv(split="all", unlock_test=True)
target_cols = [c for c in master.columns if c.startswith("target_")]
print(f"Checking target columns: {len(target_cols)} found")
for N in [2, 3, 6, 12]:
    inv = master[master[f"has_valid_target_{N}"] == 0]
    n_cols_N = [c for c in target_cols if c.startswith(f"target_{N}_")]
    print(f"Horizon N={N}: {len(inv)} invalid rows.")
    for col in n_cols_N:
        nan_cnt = inv[col].isna().sum()
        zero_cnt = (inv[col] == 0).sum()
        assert zero_cnt == 0, f"Found zeros in invalid target rows for {col}!"
        assert nan_cnt == len(inv), f"Found non-NaNs in invalid target rows for {col}!"
    print(f"  All {len(n_cols_N)} target_{N}_* columns are 100% NaN on invalid rows (0 zeros).")
