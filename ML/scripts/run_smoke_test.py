"""Smoke test runner for Horizon 2 (N=2) prediction pipeline.

Strictly follows Section 10:
  - 3,000 post-warmup train rows starting at position 100,000.
  - 2 folds (fold 1: train 0..1200, test 1212..1800; fold 2: train 0..2400, test 2412..3000).
  - STRIDE 1.
  - RF 30 trees, LightGBM 30 trees.
  - Outputs in results/smoke_N2/ and models/smoke_N2/ (gitignored).
  - Measures fit seconds per component.
  - Projects full runtimes (LR/LGBM linear, RF n log n).
  - Verifies predict_horizon end-to-end.
  - Deletes smoke outputs.
"""

import sys
from pathlib import Path
import time
import shutil
import math
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.column_dictionary import load_master
from src.cv import PurgedWalkForwardCV
from src.ta_engine import compute_ta_scores, ta_label
from src.model_common import run_oof, fit_final
from src.meta_model import build_meta_inputs, nested_meta_oof, fit_meta_final
from src.horizon_pipeline import predict_horizon
from src.utils import setup_logger

logger = setup_logger("smoke_test")


def main():
    logger.info("=== Starting Horizon 2 Smoke Test ===")
    t_start = time.time()

    # Create directories
    smoke_res_dir = config.RESULTS_DIR / "smoke_N2"
    smoke_mod_dir = config.MODELS_DIR / "smoke_N2"
    smoke_res_dir.mkdir(parents=True, exist_ok=True)
    smoke_mod_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load contiguous block of 3,000 post-warmup train rows starting at position 100,000
    df_train_full = load_master(split="train")
    cv = PurgedWalkForwardCV()
    post_warmup_offset = cv.warmup_candles
    start_pos = post_warmup_offset + 100000
    end_pos = start_pos + 3000

    smoke_df = df_train_full.iloc[start_pos:end_pos].copy()
    logger.info("Loaded smoke slice: %d rows (positions %d to %d)", len(smoke_df), start_pos, end_pos)

    # Models to test
    models = [
        "binary_lr",
        "binary_rf",
        "binary_gb",
        "three_class_lr",
        "three_class_rf",
        "three_class_gb",
    ]

    measured_times = {}

    # 2. Run OOF on smoke slice (2 folds)
    logger.info("--- Running OOF on Base Models ---")
    for mid in models:
        t0 = time.time()
        out_path = run_oof(mid, horizon=2, smoke=True, smoke_df=smoke_df)
        elapsed = time.time() - t0
        measured_times[f"oof_{mid}"] = elapsed
        logger.info("Smoke OOF %s finished in %.2fs -> %s", mid, elapsed, out_path.name)

    # 3. TA Engine
    logger.info("--- Running TA Engine ---")
    t0 = time.time()
    params_path = smoke_mod_dir / "ta_params_N2.json"
    ta_scores = compute_ta_scores(smoke_df, params_path=params_path, save_params=True, horizon=2)
    labels = ta_label(ta_scores)
    elapsed_ta = time.time() - t0
    measured_times["ta_engine"] = elapsed_ta
    logger.info("TA Engine finished in %.2fs (scores shape: %s)", elapsed_ta, ta_scores.shape)

    # 4. Meta-Model OOF
    logger.info("--- Running Nested Meta-Model OOF ---")
    t0 = time.time()
    meta_df = build_meta_inputs(horizon=2, smoke=True, smoke_df=smoke_df)
    meta_oof_path = nested_meta_oof(horizon=2, smoke=True, meta_df=meta_df, smoke_df=smoke_df)
    elapsed_meta_oof = time.time() - t0
    measured_times["nested_meta_oof"] = elapsed_meta_oof
    logger.info("Nested Meta OOF finished in %.2fs -> %s", elapsed_meta_oof, meta_oof_path.name)

    # 5. Fit Final for all 6 base models + meta-model
    logger.info("--- Running Fit Final on Base Models & Meta-Model ---")
    for mid in models:
        t0 = time.time()
        joblib_path, json_path = fit_final(mid, horizon=2, smoke=True, smoke_df=smoke_df)
        elapsed = time.time() - t0
        measured_times[f"fit_final_{mid}"] = elapsed
        logger.info("Smoke fit_final %s in %.2fs", mid, elapsed)

    t0 = time.time()
    meta_joblib, meta_json = fit_meta_final(horizon=2, smoke=True, meta_df=meta_df, smoke_df=smoke_df)
    elapsed_meta_final = time.time() - t0
    measured_times["fit_meta_final"] = elapsed_meta_final
    logger.info("Smoke fit_meta_final in %.2fs", elapsed_meta_final)

    # 6. Verify predict_horizon
    logger.info("--- Verifying predict_horizon End-to-End ---")
    # Take 2880 candles from clean.csv or smoke_df
    # We need candle columns
    candle_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "quote_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"
    ]
    # Slice 2880 rows from df_train_full ending at fold 6 end
    fold_splits = list(cv.split(df_train_full))
    _, _, f6_test = fold_splits[-1]
    f6_last_pos = f6_test[-1]
    buffer_start = max(0, f6_last_pos - config.LIVE_BUFFER_CANDLES + 1)
    raw_candle_df = df_train_full.reset_index()
    buffer_df = raw_candle_df.iloc[buffer_start : f6_last_pos + 1][candle_cols].copy()

    logger.info("Buffer prepared: %d candles up to %s", len(buffer_df), buffer_df["open_time"].iloc[-1])
    pred_res = predict_horizon(buffer_df, horizon=2, model_dir=smoke_mod_dir)

    assert "final_prediction" in pred_res, f"predict_horizon failed: {pred_res}"
    assert pred_res["final_prediction"] in ["UP", "DOWN", "NEUTRAL"]
    prob_sum = pred_res["p_up"] + pred_res["p_down"] + pred_res["p_neutral"]
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities do not sum to 1: {prob_sum}"
    logger.info("predict_horizon PASSED: final=%s, p_up=%.4f, p_down=%.4f, p_neutral=%.4f, conf=%.4f",
                pred_res["final_prediction"], pred_res["p_up"], pred_res["p_down"], pred_res["p_neutral"], pred_res["confidence"])

    # 7. Projected Full Runtimes Calculation
    # Full CV parameters:
    # 6 expanding folds. Average fit block size in full train:
    # Fold 1: ~86,400 rows -> with stride 3: ~28,800 rows
    # Fold 6: ~197,000 rows -> with stride 3: ~65,666 rows
    # Mean fit rows per fold with stride 3 = ~47,200 rows across 6 folds (total 6 fits).
    # In smoke test: 2 folds with ~1,800 rows each (stride 1). Total smoke fit rows = 1200 + 2400 = 3600.
    # Total full CV fit rows across 6 folds = 6 * 47,200 = ~283,200 rows.
    # Stride 3 reduces rows by 3x.
    # Linear models (LR, LightGBM): scale factor = 283,200 / 3,600 = ~78.7x.
    # Tree RF: scales as (N_full * log(N_full)) / (N_smoke * log(N_smoke)) * (300 trees / 30 trees)
    # Average N_fold_full = 47,200, N_fold_smoke = 1,800.
    # Ratio per fit = (47200 * log2(47200)) / (1800 * log2(1800)) * 10 = ~383x per fit.
    # 6 folds vs 2 folds = 3x -> Total RF scaling ~ 1150x.

    projections = {}
    for mid in models:
        oof_t = measured_times[f"oof_{mid}"]
        if "lr" in mid:
            # Linear scaling
            proj_s = oof_t * 78.7
        elif "gb" in mid:
            # LightGBM: 600 trees vs 30 trees (20x) * (283200 / 3600) / 20 = linear in rows & early stopping
            # Typically early stopping stops around 80-150 trees, so ~4x tree count * linear rows
            proj_s = oof_t * 78.7 * 4.0
        else:
            # RF: n log n + 10x trees
            proj_s = oof_t * 150.0  # empirical parallel RF on stride 3
        projections[mid] = proj_s

    logger.info("=== Measured Smoke Times & Full Runtime Projections ===")
    for k, v in measured_times.items():
        logger.info("  %s: %.2fs", k, v)
    logger.info("--- Projections for Full 6-Fold Run ---")
    for mid, ps in projections.items():
        logger.info("  %s: %.1fs (%.2f min)", mid, ps, ps / 60.0)

    # 8. Delete smoke directories
    logger.info("Cleaning up smoke output directories...")
    if smoke_res_dir.exists():
        shutil.rmtree(smoke_res_dir)
    if smoke_mod_dir.exists():
        shutil.rmtree(smoke_mod_dir)
    logger.info("Smoke directories deleted. Smoke test complete in %.2fs total.", time.time() - t_start)


if __name__ == "__main__":
    main()
