"""Master dataset builder — Stock-Market-ML-Predictor v3.

Constructs the single unified dataset `data/processed/master.csv` joining on `open_time`:
  1. Cleaned market columns (OHLCV + quote vol + trades + taker vol + gap flags)
  2. All 85 engineered causal features (from `src.features.build_features`)
  3. Base targets: fwd_ret_N, fwd_ret_open_N, raw_sign_N, has_valid_target_N (N in {2,3,6,12})
  4. The 40 label columns: target_N_rule (4 horizons x 10 rules including F05, F10, F15)
  5. The 4 big_move columns: big_move_N (Setup D two-stage worth-it target)
  6. Per-row dynamic thetas for V1 and V2 only (16 columns: theta_up/down for V1, V2 across 4 horizons)
     (Constant thetas for S1, S2, Q1, A1, A2, F05, F10, F15 stay in `targets_theta_frozen.json`)

Total schema: 172 columns, 318,240 rows.
Every column belongs to one of five roles:
  market | feature | target | threshold | flag
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src import config
from src.features import build_features, ALL_FEATURE_COLUMNS
from src.io_utils import (
    load_clean_market_data_csv,
    load_targets_base_csv,
    load_targets_labels_csv,
    save_master_csv,
    MASTER_CSV_PATH,
)
from src.utils import setup_logger

logger = setup_logger("master")


def build_master(
    df_clean: Optional[pd.DataFrame] = None,
    df_features: Optional[pd.DataFrame] = None,
    df_targets_base: Optional[pd.DataFrame] = None,
    df_targets_labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join cleaned market data, engineered features, and targets into one master DataFrame.

    Parameters
    ----------
    df_clean : pd.DataFrame, optional
        Cleaned market data (clean.csv). Loaded from disk if None.
    df_features : pd.DataFrame, optional
        Engineered features. Computed from df_clean if None.
    df_targets_base : pd.DataFrame, optional
        Base forward returns and validation flags. Loaded from disk if None.
    df_targets_labels : pd.DataFrame, optional
        Classification labels and thetas. Loaded from disk if None.

    Returns
    -------
    pd.DataFrame
        Complete master dataset with exactly 318,240 rows and 175 columns.
    """
    # 1. Cleaned market data
    if df_clean is None:
        logger.info("Loading cleaned market data...")
        df_clean = load_clean_market_data_csv()

    # 2. Engineered features
    if df_features is None:
        logger.info("Building 85 engineered causal features...")
        df_features = build_features(df_clean)

    # 3. Base targets
    if df_targets_base is None:
        logger.info("Loading base targets...")
        df_targets_base = load_targets_base_csv()

    # 4. Target labels and dynamic thetas
    if df_targets_labels is None:
        logger.info("Loading targets labels...")
        df_targets_labels = load_targets_labels_csv()

    # Extract 40 target label columns (10 rules across 4 candidate horizons)
    horizons = config.CANDIDATE_HORIZONS
    rules = config.ALL_RULES
    target_label_cols = [f"target_{N}_{r}" for N in horizons for r in rules]

    # Extract 4 big_move target columns for the two-stage indicator (Setup D)
    big_move_cols = [f"big_move_{N}" for N in horizons]
    
    # Extract 16 dynamic theta columns for V1 and V2 only
    v_theta_cols = []
    for N in horizons:
        for r in ["V1", "V2"]:
            v_theta_cols.append(f"theta_up_{N}_{r}")
            v_theta_cols.append(f"theta_down_{N}_{r}")

    df_labels_subset = df_targets_labels[target_label_cols + big_move_cols + v_theta_cols]

    # Join on index (open_time)
    logger.info("Joining market, features, base targets, and label targets...")
    master = pd.concat([
        df_clean,
        df_features,
        df_targets_base,
        df_labels_subset,
    ], axis=1)

    # Integrity verification
    assert len(master) == len(df_clean), (
        f"Row count mismatch in master: {len(master)} vs {len(df_clean)}"
    )
    assert master.index.is_monotonic_increasing, "Master index is not monotonically increasing"
    assert master.index.duplicated().sum() == 0, "Master index contains duplicate timestamps"

    logger.info("Master DataFrame built successfully: shape %s", master.shape)
    return master


def build_and_save_master(filepath: Optional[Path] = None) -> Path:
    """Build and save master.csv to disk via io_utils.

    Parameters
    ----------
    filepath : Optional[Path]
        Destination path. Defaults to config.DATA_PROCESSED_DIR / 'master.csv'.

    Returns
    -------
    Path
        Path to saved file.
    """
    target_path = filepath or MASTER_CSV_PATH
    master_df = build_master()
    return save_master_csv(master_df, filepath=target_path)


if __name__ == "__main__":
    out_path = build_and_save_master()
    print(f"master.csv written successfully to {out_path}")

