"""Reproducible Window Sensitivity Test (Section 5 & 10.2).

Evaluates probability and action stability across different lookback buffer sizes:
  Windows: [100, 300, 600, 1000] candles.
Evaluated on 200 random timestamps from the Train partition.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "ML"))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from src import config
from src.column_dictionary import load_master
from src.trading_engine import TradingEngine


def run_window_sensitivity(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    print("=" * 80)
    print(f"RUNNING WINDOW SENSITIVITY TEST (N = {n_samples} Train Timestamps)")
    print("=" * 80)

    # 1. Load Train partition (slice 5,000 to 10,000 to keep test causal and fast)
    print("Loading Train slice for full-history reference...")
    t0 = time.time()
    df_train_full = load_master(split="train")
    
    # Use a contiguous 4,000 candle block from Train (candles 5,000 to 9,000)
    start_idx = 5000
    end_idx = 9000
    df_block = df_train_full.iloc[start_idx:end_idx].copy()
    print(f"Loaded block of {len(df_block):,} candles in {time.time() - t0:.2f}s.")

    engine = TradingEngine.get_instance()

    # 2. Run full-history predict_batch on the entire 4,000-candle block
    print("Computing full-history reference predictions via predict_batch...")
    t_batch = time.time()
    df_ref_batch = engine.predict_batch(df_block)
    print(f"Full-history batch inference completed in {time.time() - t_batch:.2f}s.")

    # 3. Sample 200 timestamps with at least 1,000 preceding candles in the block
    np.random.seed(seed)
    valid_test_indices = np.random.choice(np.arange(1050, len(df_block) - 10), size=n_samples, replace=False)
    valid_test_indices.sort()

    windows = [100, 300, 600, 1000]
    metrics = {w: {"prob_diffs": [], "action_matches": 0} for w in windows}

    print(f"Evaluating {n_samples} timestamps across windows {windows}...")
    t_eval = time.time()

    for i, idx in enumerate(valid_test_indices):
        ts = df_block.index[idx]
        ref_row = df_ref_batch.loc[ts]
        p_ref = np.array([ref_row["p_final_down"], ref_row["p_final_neutral"], ref_row["p_final_up"]])
        act_ref = int(ref_row["final_action"])

        for w in windows:
            # Sub-window slice of raw OHLCV columns ending at idx
            sub_df = df_block.iloc[idx - w : idx + 1][["open", "high", "low", "close", "volume"]]
            res_w = engine.predict_candle(sub_df)
            p_w = np.array([res_w["probabilities"]["p_down"], res_w["probabilities"]["p_neutral"], res_w["probabilities"]["p_up"]])
            act_w = int(res_w["action"])

            max_diff = float(np.max(np.abs(p_ref - p_w)))
            metrics[w]["prob_diffs"].append(max_diff)
            if act_ref == act_w:
                metrics[w]["action_matches"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_samples} timestamps ({time.time() - t_eval:.1f}s)...", flush=True)

    rows = []
    print("\n" + "=" * 80)
    print("WINDOW SENSITIVITY TEST SUMMARY")
    print("=" * 80)
    for w in windows:
        diffs = np.array(metrics[w]["prob_diffs"])
        within_1e3 = float((diffs <= 1e-3).mean() * 100.0)
        action_match = float(metrics[w]["action_matches"] / n_samples * 100.0)
        mean_diff = float(diffs.mean())
        max_diff = float(diffs.max())
        p99_diff = float(np.percentile(diffs, 99.0))
        
        row = {
            "window_candles": w,
            "within_1e3_pct": round(within_1e3, 2),
            "action_match_pct": round(action_match, 2),
            "mean_prob_diff": round(mean_diff, 6),
            "max_prob_diff": round(max_diff, 6),
            "p99_prob_diff": round(p99_diff, 6),
        }
        rows.append(row)
        print(f"Window {w:4d} | Within 1e-3: {within_1e3:6.2f}% | Action Match: {action_match:6.2f}% | Mean Diff: {mean_diff:.6f} | p99: {p99_diff:.6f}")

    df_res = pd.DataFrame(rows)
    return df_res


if __name__ == "__main__":
    df_out = run_window_sensitivity(n_samples=200)
    # Check if window 1000 or 600 passes
    best_pass = df_out[df_out["within_1e3_pct"] >= 99.0]
    if not best_pass.empty:
        chosen = int(best_pass.iloc[0]["window_candles"])
        print(f"\n[DECISION] Smallest window meeting >= 99% within 1e-3 tolerance: {chosen} candles.")
    else:
        print(f"\n[NOTE] Largest window tested: {int(df_out.iloc[-1]['window_candles'])} candles.")

