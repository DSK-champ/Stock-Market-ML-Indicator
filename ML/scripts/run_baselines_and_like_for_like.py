"""Script to run updated baselines, volatility baseline on big_move, and like-for-like comparison suite."""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.column_dictionary import load_master
from src.config import LINEAR_FEATURES
from src.baselines import (
    evaluate_binary_baselines,
    evaluate_three_class_baselines,
    evaluate_volatility_baseline_big_move,
)
from src.like_for_like import run_like_for_like_suite
from src.utils import setup_logger

logger = setup_logger("run_baselines_like_for_like")

def main():
    start_total = time.time()
    logger.info("Loading TRAIN split from master.csv...")
    t0 = time.time()
    df_train = load_master(split="train")
    logger.info("Loaded TRAIN split (%d rows, %d columns) in %.2fs", len(df_train), len(df_train.columns), time.time() - t0)

    # 1. Binary Baselines
    logger.info("--- Stage 1: Evaluating Binary Baselines (15 rules across N in {2,3,6,12}) ---")
    t0 = time.time()
    bin_report_path = config.ML_DIR / "results" / "reports" / "baselines_binary_summary.csv"
    df_bin = evaluate_binary_baselines(
        df_train_full=df_train,
        horizons=config.CANDIDATE_HORIZONS,
        save_oof=True,
        save_report_path=bin_report_path,
    )
    logger.info("Binary baselines finished in %.2fs. Saved to %s", time.time() - t0, bin_report_path)

    # 2. Three-Class Baselines
    logger.info("--- Stage 2: Evaluating 3-Class Baselines (15 rules across N in {2,3,6,12}) ---")
    t0 = time.time()
    three_report_path = config.ML_DIR / "results" / "reports" / "baselines_three_class_summary.csv"
    df_three = evaluate_three_class_baselines(
        df_train_full=df_train,
        horizons=config.CANDIDATE_HORIZONS,
        rule="S1",
        save_oof=True,
        save_report_path=three_report_path,
    )
    logger.info("3-Class baselines finished in %.2fs. Saved to %s", time.time() - t0, three_report_path)

    # 3. Volatility Baseline on Big Move
    logger.info("--- Stage 3: Evaluating Volatility Baseline on big_move_N ---")
    t0 = time.time()
    vol_report_path = config.ML_DIR / "results" / "reports" / "volatility_baseline_summary.csv"
    df_vol = evaluate_volatility_baseline_big_move(
        df_train_full=df_train,
        horizons=config.CANDIDATE_HORIZONS,
        save_report_path=vol_report_path,
    )
    logger.info("Volatility baseline finished in %.2fs. Saved to %s", time.time() - t0, vol_report_path)

    # 4. Like-for-Like Comparison Suite
    logger.info("--- Stage 4: Like-for-Like Comparison Suite (Coverage 100%, 50%, 20%, 10%) ---")
    t0 = time.time()
    lfl_report_path = config.ML_DIR / "results" / "reports" / "like_for_like_summary.csv"
    decomp_report_path = config.ML_DIR / "results" / "reports" / "like_for_like_decomp_20pct.csv"
    df_lfl, df_decomp = run_like_for_like_suite(
        df_train_full=df_train,
        horizon=12,
        coverages=[1.0, 0.5, 0.2, 0.1],
        linear_features=LINEAR_FEATURES,
        save_report_path=lfl_report_path,
        save_decomp_path=decomp_report_path,
    )
    logger.info("Like-for-like suite finished in %.2fs.", time.time() - t0)

    logger.info("All evaluations completed successfully in %.2fs total.", time.time() - start_total)

if __name__ == "__main__":
    main()
