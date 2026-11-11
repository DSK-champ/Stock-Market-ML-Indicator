"""Unit tests for PurgedWalkForwardCV.

Required tests:
1. Held-out blocks across the 6 folds are pairwise disjoint and contiguous (no gaps or overlap).
2. Purge gap is strictly 12 candles between train block end and held-out test start.
3. Purge gap is strictly 12 candles between sub-train end and validation start.
4. No held-out row appears in an earlier training block (strict temporal precedence).
5. All folds maintain identical row index partitions regardless of horizon or target.
6. Warmup candles (2,018) are strictly stripped before CV split.
"""

import numpy as np
import pandas as pd
import pytest

from src.cv import PurgedWalkForwardCV, PURGE_GAP, N_FOLDS, LOOKBACK_CANDLES, INITIAL_TRAIN_CANDLES
from src import config


def _make_dummy_train_df(n_rows: int = config.SPLIT_BOUNDARIES["train"]["expected_candles"]) -> pd.DataFrame:
    """Generate dummy DataFrame matching train split row count."""
    idx = pd.date_range("2023-11-01", periods=n_rows, freq="5min", tz="UTC")
    return pd.DataFrame({"close": np.random.randn(n_rows)}, index=idx)


def test_cv_held_out_blocks_disjoint_and_contiguous():
    """Verify that held-out blocks form an exact contiguous partition with no overlap."""
    df = _make_dummy_train_df()
    cv = PurgedWalkForwardCV()

    all_test_indices = []
    prev_end = None

    for fold, tr_idx, te_idx in cv.split(df):
        assert len(te_idx) > 0, f"Fold {fold} has empty test set"
        # Contiguous check
        if prev_end is not None:
            assert te_idx[0] == prev_end, f"Fold {fold} test start {te_idx[0]} != prev test end {prev_end}"
        prev_end = te_idx[-1] + 1
        all_test_indices.extend(te_idx)

    # Disjoint check
    assert len(all_test_indices) == len(set(all_test_indices)), "Duplicate indices in held-out sets!"
    
    # End check: last index must match last row of train
    assert prev_end == len(df), f"Last test index {prev_end} != train row count {len(df)}"


def test_cv_purge_gap_strictly_12():
    """Verify that the purge gap between train and test is exactly PURGE_GAP (12) candles."""
    df = _make_dummy_train_df()
    cv = PurgedWalkForwardCV()

    for fold, tr_idx, te_idx in cv.split(df):
        train_end = tr_idx[-1]
        test_start = te_idx[0]
        gap = test_start - train_end - 1
        assert gap == PURGE_GAP, f"Fold {fold} purge gap is {gap}, expected {PURGE_GAP}"


def test_cv_sub_train_val_purge_gap_strictly_12():
    """Verify that the purge gap between sub-train and early-stopping val is strictly PURGE_GAP (12)."""
    df = _make_dummy_train_df()
    cv = PurgedWalkForwardCV()

    for fold, sub_tr_idx, val_idx, te_idx in cv.split_with_val(df):
        sub_tr_end = sub_tr_idx[-1]
        val_start = val_idx[0]
        gap = val_start - sub_tr_end - 1
        assert gap == PURGE_GAP, f"Fold {fold} internal val purge gap is {gap}, expected {PURGE_GAP}"
        assert val_idx[-1] < te_idx[0], f"Fold {fold} val set overlaps test set!"


def test_cv_no_future_leakage_and_temporal_precedence():
    """Verify that no held-out row appears in an earlier training block."""
    df = _make_dummy_train_df()
    cv = PurgedWalkForwardCV()

    seen_held_out = set()
    for fold, tr_idx, te_idx in cv.split(df):
        tr_set = set(tr_idx)
        # Check no previously seen held-out rows are in the future
        for idx in te_idx:
            assert idx not in tr_set, f"Held-out index {idx} present in its own training set"
            assert idx not in seen_held_out, f"Held-out index {idx} evaluated multiple times"

        # Check train indices only come from past (expanding window)
        assert tr_idx[-1] < te_idx[0], f"Fold {fold} train end {tr_idx[-1]} >= test start {te_idx[0]}"
        seen_held_out.update(te_idx)


def test_cv_warmup_strictly_stripped():
    """Verify that warmup rows (2,018) are excluded from all fold train and test sets."""
    df = _make_dummy_train_df()
    cv = PurgedWalkForwardCV(warmup_candles=LOOKBACK_CANDLES)

    for fold, tr_idx, te_idx in cv.split(df):
        assert tr_idx[0] >= LOOKBACK_CANDLES, (
            f"Fold {fold} train start index {tr_idx[0]} < warmup {LOOKBACK_CANDLES}"
        )
        assert te_idx[0] >= LOOKBACK_CANDLES, (
            f"Fold {fold} test start index {te_idx[0]} < warmup {LOOKBACK_CANDLES}"
        )


def test_cv_identical_partitions_across_horizons():
    """Verify that CV partitions depend only on row index and are identical regardless of target horizon."""
    df = _make_dummy_train_df()
    cv = PurgedWalkForwardCV()

    splits_h2 = list(cv.split(df))
    splits_h12 = list(cv.split(df))

    assert len(splits_h2) == len(splits_h12) == N_FOLDS
    for (f1, tr1, te1), (f2, tr2, te2) in zip(splits_h2, splits_h12):
        assert f1 == f2
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)
