"""Unit tests for chronological splitting, walk-forward folds, and test-set lock.
"""

import pandas as pd
import pytest

from src import config
from src.splitting import (
    TestSetLockedError,
    get_train_data,
    get_val_data,
    get_test_data,
    get_splits,
    generate_walk_forward_folds,
    get_real_split_indices,
)


@pytest.fixture
def mock_clean_df() -> pd.DataFrame:
    """Synthetic continuous 5m dataframe for testing split mechanics."""
    n = 1000
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 10.0,
        "gap_id": 1,
        "is_gap_boundary": False,
    }, index=timestamps)
    return df


def test_test_set_lock_raises_error(mock_clean_df: pd.DataFrame):
    """Verify that attempting to access the test set without unlock raises TestSetLockedError."""
    # Direct function access must fail
    with pytest.raises(TestSetLockedError):
        get_test_data(mock_clean_df, unlock=False)

    # get_splits without unlock must not include 'test'
    splits = get_splits(mock_clean_df, unlock_test=False)
    assert "train" in splits
    assert "val" in splits
    assert "test" not in splits


def test_test_set_unlock_permitted(mock_clean_df: pd.DataFrame):
    """Verify that passing unlock=True allows test retrieval for Phase 6 backtesting."""
    test_df = get_test_data(mock_clean_df, unlock=True)
    assert len(test_df) == 150  # 15% of 1000


def test_split_boundaries_chronological_no_overlap(mock_clean_df: pd.DataFrame):
    """Verify train, val, and test slices have zero overlap and are chronologically ordered."""
    train_df = get_train_data(mock_clean_df)
    val_df = get_val_data(mock_clean_df)
    test_df = get_test_data(mock_clean_df, unlock=True)

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    # Chronological order
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

    # Disjoint indices
    assert len(train_df.index.intersection(val_df.index)) == 0
    assert len(val_df.index.intersection(test_df.index)) == 0


def test_walk_forward_purge_gap(mock_clean_df: pd.DataFrame):
    """Verify that walk-forward folds maintain exact 12-candle purge gap and zero overlap."""
    folds = list(generate_walk_forward_folds(
        mock_clean_df,
        n_folds=3,
        purge_gap=config.PURGE_GAP_CANDLES,
    ))
    assert len(folds) >= 1

    for fold in folds:
        train_df = fold["train_df"]
        val_df = fold["val_df"]
        purge_gap = fold["purge_gap_candles"]

        # Ensure no overlap
        assert len(train_df.index.intersection(val_df.index)) == 0
        assert train_df.index.max() < val_df.index.min()

        # Calculate exact candle gap on continuous grid
        time_diff = val_df.index.min() - train_df.index.max()
        # On a 5m grid, 12 candles skipped means (12 + 1) * 5m = 65m between train_max and val_min
        expected_diff = pd.Timedelta(minutes=(purge_gap + 1) * config.INTERVAL_MINUTES)
        assert time_diff == expected_diff
