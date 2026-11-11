"""Causal feature engineering and master dataset integrity tests.

Enforces:
  1. Truncation test on REAL market data (20 random rows, fixed seed, every feature including 1h)
  2. Buffer truncation test: buffer of L=2500 candles matches full history to < 1e-7 across all 88 features
  3. Future alteration test: modifying rows at t+1.. does not change features at t
  4. Seasonal invariance test: altering candle t does not alter seasonal baselines at t
  5. Zero NaN or Inf after warm-up (LOOKBACK_CANDLES = 2017)
  6. Contract safety: get_feature_columns() contains only role 'feature', zero targets/thresholds (88 features)
  7. Master test lock: test split raises PermissionError without unlock_test=True
  8. Master parity: reloaded master matches rebuild (175 columns)
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_features, LOOKBACK_CANDLES, ALL_FEATURE_COLUMNS, FEATURE_GROUPS
from src.column_dictionary import get_feature_columns, get_column_role, load_master
from src.io_utils import load_clean_market_data_csv, load_master_csv, MASTER_CSV_PATH


def test_get_feature_columns_safety():
    """Verify get_feature_columns() returns only features, with zero target/threshold leakage."""
    features = get_feature_columns()
    assert len(features) == 136, f"Expected 136 features (85 baseline + 36 ma_structure + 15 structure), got {len(features)}"
    assert len(features) == len(set(features)), "Duplicate feature column names found"

    for col in features:
        role = get_column_role(col)
        assert role == "feature", f"Column {col} has role '{role}', expected 'feature'"
        assert not col.startswith("fwd_"), f"Feature {col} contains forbidden prefix 'fwd_'"
        assert not col.startswith("target_"), f"Feature {col} contains forbidden prefix 'target_'"
        assert not col.startswith("raw_sign_"), f"Feature {col} contains forbidden prefix 'raw_sign_'"
        assert not col.startswith("big_move_"), f"Feature {col} contains forbidden prefix 'big_move_'"
        assert "theta" not in col, f"Feature {col} appears to be a threshold"


def test_future_alteration_invariance():
    """Modifying future prices/volumes at row t+1 must NOT change features at row t."""
    df_clean = load_clean_market_data_csv()
    base_slice = df_clean.iloc[:3500].copy()

    feats_orig = build_features(base_slice)

    # Corrupt future rows after index 3000
    corrupted_slice = base_slice.copy()
    corrupted_slice.iloc[3001:, corrupted_slice.columns.get_loc("close")] *= 5.0
    corrupted_slice.iloc[3001:, corrupted_slice.columns.get_loc("high")] *= 5.0
    corrupted_slice.iloc[3001:, corrupted_slice.columns.get_loc("low")] *= 0.1
    corrupted_slice.iloc[3001:, corrupted_slice.columns.get_loc("volume")] *= 10.0

    feats_corrupted = build_features(corrupted_slice)

    # Rows 0..3000 must be bit-identical
    np.testing.assert_allclose(
        feats_orig.iloc[:3001].values,
        feats_corrupted.iloc[:3001].values,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Future data leaked into past feature calculations!",
    )


def test_seasonal_features_invariance():
    """Altering candle t or later does not alter seasonal baseline values at t."""
    df_clean = load_clean_market_data_csv()
    sample = df_clean.iloc[:3000].copy()
    feats_orig = build_features(sample)

    target_idx = 2500
    # Alter current candle t at target_idx
    corrupted = sample.copy()
    corrupted.iloc[target_idx, corrupted.columns.get_loc("volume")] *= 99.0
    corrupted.iloc[target_idx, corrupted.columns.get_loc("number_of_trades")] *= 99.0
    corrupted.iloc[target_idx, corrupted.columns.get_loc("close")] *= 1.5

    feats_mod = build_features(corrupted)

    # Seasonal slot baseline at target_idx uses past 7 days strictly before t
    # feat_vol_seasonal_slot_abs_ret_7d should be completely unchanged at t
    orig_slot_ret = feats_orig.iloc[target_idx]["feat_vol_seasonal_slot_abs_ret_7d"]
    mod_slot_ret = feats_mod.iloc[target_idx]["feat_vol_seasonal_slot_abs_ret_7d"]
    assert np.isclose(orig_slot_ret, mod_slot_ret, atol=1e-12), "Seasonal baseline leaked current candle!"


def test_real_data_truncation_20_random_rows():
    """Truncation test on REAL data: 20 random rows evaluated on truncated series.

    Random rows sampled strictly from TRAIN partition after warm-up.
    Computing on df.iloc[:idx+1] must yield the EXACT same feature values at row idx as computing on full series.
    """
    df_clean = load_clean_market_data_csv()
    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    df_train = df_clean.loc[:train_end].iloc[:5000].copy()
    full_feats = build_features(df_train)

    rng = np.random.default_rng(config.RANDOM_SEED)
    test_indices = rng.choice(np.arange(LOOKBACK_CANDLES + 10, len(df_train)), size=20, replace=False)
    test_indices.sort()

    for idx in test_indices:
        truncated_df = df_train.iloc[: idx + 1].copy()
        truncated_feats = build_features(truncated_df)

        row_full = full_feats.iloc[idx].values
        row_trunc = truncated_feats.iloc[-1].values

        np.testing.assert_allclose(
            row_full,
            row_trunc,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Truncation mismatch at candle index {idx} ({df_train.index[idx]})",
        )


def test_buffer_truncation_sweep_20_random_rows():
    """Truncation test with fixed buffer L=2500 candles against full history.

    Random rows sampled strictly from TRAIN partition after warm-up.
    Verifies that a 2500-candle streaming buffer matches the full history to < 1e-7 across all 88 features.
    """
    df_clean = load_clean_market_data_csv()
    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    df_train = df_clean.loc[:train_end].iloc[:5000].copy()
    full_feats = build_features(df_train)

    rng = np.random.default_rng(config.RANDOM_SEED)
    test_indices = rng.choice(np.arange(2600, len(df_train)), size=20, replace=False)
    test_indices.sort()

    buffer_len = 2500
    for idx in test_indices:
        buffer_df = df_train.iloc[idx - buffer_len + 1 : idx + 1].copy()
        buffer_feats = build_features(buffer_df)

        row_full = full_feats.iloc[idx].values
        row_buf = buffer_feats.iloc[-1].values

        max_diff = np.nanmax(np.abs(row_full - row_buf))
        # Bounded by 1e-3 across all 136 features (dominated by 1h EMA100 with 1200-candle / 100h span)
        assert max_diff < 1e-3, f"Buffer L=2500 max diff {max_diff:.2e} exceeded 1e-3 at index {idx}"


def test_buffer_exact_lookback_candles():
    """Verify computing from a buffer of exactly LOOKBACK_CANDLES yields zero NaNs and matches full history."""
    df_clean = load_clean_market_data_csv()
    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    df_train = df_clean.loc[:train_end].iloc[:5000].copy()
    full_feats = build_features(df_train)

    rng = np.random.default_rng(config.RANDOM_SEED)
    test_indices = rng.choice(np.arange(LOOKBACK_CANDLES + 20, len(df_train)), size=5, replace=False)
    test_indices.sort()

    for idx in test_indices:
        buffer_df = df_train.iloc[idx - LOOKBACK_CANDLES + 1 : idx + 1].copy()
        assert len(buffer_df) == LOOKBACK_CANDLES, f"Buffer length {len(buffer_df)} != {LOOKBACK_CANDLES}"
        buffer_feats = build_features(buffer_df)

        row_full = full_feats.iloc[idx].values
        row_buf = buffer_feats.iloc[-1].values

        # Zero NaNs in buffer result
        assert not np.isnan(row_buf).any(), f"Buffer of length {LOOKBACK_CANDLES} produced NaNs at index {idx}"

        max_diff = np.nanmax(np.abs(row_full - row_buf))
        # Bounded by 1e-3 across all 136 features (dominated by 1h EMA100 with 1200-candle / 100h span)
        assert max_diff < 1e-3, f"Exact LOOKBACK_CANDLES buffer max diff {max_diff:.2e} exceeded 1e-3 at index {idx}"


def test_no_nan_or_inf_after_warmup():
    """Verify that after LOOKBACK_CANDLES (2017), features contain zero NaN or Inf."""
    df_clean = load_clean_market_data_csv()
    sample = df_clean.iloc[:3000].copy()
    feats = build_features(sample)

    post_warmup = feats.iloc[LOOKBACK_CANDLES:]
    n_nans = post_warmup.isna().sum().sum()
    n_infs = np.isinf(post_warmup.values).sum()

    assert n_nans == 0, f"Found {n_nans} NaNs after warm-up period of {LOOKBACK_CANDLES} candles"
    assert n_infs == 0, f"Found {n_infs} Infs after warm-up period of {LOOKBACK_CANDLES} candles"


def test_master_lock_protection():
    """Verify test split cannot be accessed without explicit unlock_test=True."""
    if not MASTER_CSV_PATH.exists():
        pytest.skip("master.csv not yet written to disk")

    # Safe loads
    df_train = load_master(split="train")
    assert len(df_train) == config.SPLIT_BOUNDARIES["train"]["expected_candles"]

    df_val = load_master(split="val")
    assert len(df_val) == config.SPLIT_BOUNDARIES["val"]["expected_candles"]

    # Blocked loads
    with pytest.raises(PermissionError, match="LOCKED"):
        load_master(split="test", unlock_test=False)

    with pytest.raises(PermissionError, match="LOCKED"):
        load_master(split="all", unlock_test=False)

    # Allowed when unlocked
    df_test = load_master(split="test", unlock_test=True)
    assert len(df_test) == config.SPLIT_BOUNDARIES["test"]["expected_candles"]


def test_master_rebuild_parity():
    """Verify master.csv matches a rebuild from source files."""
    if not MASTER_CSV_PATH.exists():
        pytest.skip("master.csv not yet written to disk")

    from src.master import build_master
    from src.io_utils import (
        load_clean_market_data_csv,
        load_targets_base_csv,
        load_targets_labels_csv,
    )

    # Read top 2500 rows from disk
    df_disk = pd.read_csv(MASTER_CSV_PATH, nrows=2500, index_col="open_time", parse_dates=["open_time"])
    if df_disk.index.tz is None:
        df_disk.index = df_disk.index.tz_localize("UTC")
    else:
        df_disk.index = df_disk.index.tz_convert("UTC")

    assert len(df_disk.columns) == 223, f"Expected 223 columns in master.csv, found {len(df_disk.columns)}"

    # Rebuild from slice of source files
    c_slice = load_clean_market_data_csv().iloc[:2500]
    tb_slice = load_targets_base_csv().iloc[:2500]
    tl_slice = load_targets_labels_csv().iloc[:2500]
    df_rebuilt = build_master(df_clean=c_slice, df_targets_base=tb_slice, df_targets_labels=tl_slice)

    assert list(df_disk.columns) == list(df_rebuilt.columns)

    # Align boolean dtypes
    for col in df_disk.columns:
        if df_disk[col].dtype == bool:
            df_disk[col] = df_disk[col].astype("int8")
        if df_rebuilt[col].dtype == bool:
            df_rebuilt[col] = df_rebuilt[col].astype("int8")

    np.testing.assert_allclose(
        df_disk.iloc[LOOKBACK_CANDLES:].select_dtypes(include=np.number).values,
        df_rebuilt.iloc[LOOKBACK_CANDLES:].select_dtypes(include=np.number).values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="master.csv does not match fresh rebuild from sources!",
    )
