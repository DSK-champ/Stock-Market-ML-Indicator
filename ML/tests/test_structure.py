"""Unit tests for the Causal Market Structure Engine (Step 4b).

Tests cover:
  1. Hand-made synthetic series:
     - Higher Highs & Higher Lows -> Up trend
     - Pullback breaking last HL by buffer -> trend terminates
     - Close break of level flips it (role reversal)
     - Retest of flipped level with rejection
     - Wick penetration without close break -> no break event
  2. Causality & Non-repainting:
     - Pivot at i is strictly invisible before i+k (k=5)
     - Future candle alteration leaves past and current t unchanged
     - Truncation parity on real train rows
  3. Safety & Hygiene:
     - No raw price levels in role 'feature'
     - Gap boundary isolation (no structure spans a gap)
     - Live buffer replay parity
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.column_dictionary import load_master, get_feature_columns, get_column_role
from src.features import build_features, LOOKBACK_CANDLES
from src.structure import (
    build_structure,
    STRUCTURE_FEATURE_COLUMNS,
    DRAWING_COLUMNS,
)


def _make_flat_df(n: int = 100, price: float = 50000.0) -> pd.DataFrame:
    """Helper to create neutral base DataFrame with 5m timestamps."""
    times = pd.date_range("2024-01-01 00:00:00", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price + 10.0),
            "low": np.full(n, price - 10.0),
            "close": np.full(n, price),
            "volume": np.full(n, 100.0),
            "gap_id": np.zeros(n, dtype=int),
        },
        index=times,
    )
    return df


class TestHandmadeStructureSeries:
    def test_higher_highs_and_higher_lows_uptrend(self):
        """Sequential HH and HL must establish Dow Up trend."""
        df = _make_flat_df(120, price=50000.0)
        # Create Pivot Low 1 at t=15, High 1 at t=30
        df.iloc[15, df.columns.get_loc("low")] = 49000.0
        df.iloc[15, df.columns.get_loc("close")] = 49100.0
        df.iloc[30, df.columns.get_loc("high")] = 51000.0
        df.iloc[30, df.columns.get_loc("close")] = 50900.0
        
        # Create Pivot Low 2 (Higher Low) at t=45
        df.iloc[45, df.columns.get_loc("low")] = 49500.0
        df.iloc[45, df.columns.get_loc("close")] = 49600.0
        
        # Create Pivot High 2 (Higher High) at t=60
        df.iloc[60, df.columns.get_loc("high")] = 52000.0
        df.iloc[60, df.columns.get_loc("close")] = 51900.0

        feats, _ = build_structure(df)
        # Pivot High 2 confirmed at t=65 (60+5). Uptrend must be active thereafter.
        assert (feats["feat_struct_trend_up"].iloc[66:80] == 1.0).all()
        assert (feats["feat_struct_trend_down"].iloc[66:80] == 0.0).all()

    def test_pullback_break_terminates_trend(self):
        """Close breaking below last confirmed HL by buffer must terminate Up trend."""
        df = _make_flat_df(120, price=50000.0)
        df.iloc[15, df.columns.get_loc("low")] = 49000.0
        df.iloc[30, df.columns.get_loc("high")] = 51000.0
        df.iloc[45, df.columns.get_loc("low")] = 49500.0  # HL = 49500
        df.iloc[60, df.columns.get_loc("high")] = 52000.0

        # Break HL at t=75 by closing well below 49500
        df.iloc[75, df.columns.get_loc("close")] = 48000.0
        df.iloc[75, df.columns.get_loc("low")] = 47900.0

        feats, _ = build_structure(df)
        # Up trend should end at t=75
        assert feats["feat_struct_trend_up"].iloc[70] == 1.0
        assert feats["feat_struct_trend_up"].iloc[75] == 0.0
        assert feats["feat_struct_bars_since_break"].iloc[75] == 0.0

    def test_close_break_flips_level(self):
        """A candle close breaking beyond level + buffer flips the level role."""
        df = _make_flat_df(80, price=50000.0)
        # Swing high at t=20 (resistance at 51000)
        df.iloc[20, df.columns.get_loc("high")] = 51000.0
        df.iloc[20, df.columns.get_loc("close")] = 50800.0

        # Break above 51000 at t=40
        df.iloc[40, df.columns.get_loc("close")] = 51500.0
        df.iloc[40, df.columns.get_loc("high")] = 51600.0

        feats, drawings = build_structure(df)
        # Break event logged at t=40
        assert feats["feat_struct_bars_since_break"].iloc[40] == 0.0

    def test_retest_of_flipped_level(self):
        """Price returning to flipped support and rejecting must log retest event."""
        df = _make_flat_df(100, price=50000.0)
        # Resistance at 51000 at t=20
        df.iloc[20, df.columns.get_loc("high")] = 51000.0
        df.iloc[20, df.columns.get_loc("close")] = 50800.0

        # Break above 51000 at t=35 (closes at 51500)
        df.iloc[35:45, df.columns.get_loc("open")] = 51500.0
        df.iloc[35:45, df.columns.get_loc("close")] = 51500.0
        df.iloc[35:45, df.columns.get_loc("high")] = 51600.0
        df.iloc[35:45, df.columns.get_loc("low")] = 51400.0

        # Retest at t=45: price enters tolerance zone (within 0.5 ATR) and rejects bullishly
        df.iloc[45, df.columns.get_loc("open")] = 51020.0
        df.iloc[45, df.columns.get_loc("low")] = 50980.0
        df.iloc[45, df.columns.get_loc("close")] = 51050.0
        df.iloc[45, df.columns.get_loc("high")] = 51100.0

        feats, _ = build_structure(df)
        assert feats["feat_struct_bars_since_retest"].iloc[45] == 0.0
        assert feats["feat_struct_retest_direction"].iloc[45] == 1.0

    def test_wick_penetration_without_close_does_not_break(self):
        """A wick that penetrates level but closes inside must NOT trigger a break event."""
        df = _make_flat_df(80, price=50000.0)
        # Resistance at 51000 at t=20
        df.iloc[20, df.columns.get_loc("high")] = 51000.0
        df.iloc[20, df.columns.get_loc("close")] = 50800.0

        # Wick at t=40 pokes to 51500, but close is 50500 (rejected!)
        df.iloc[40, df.columns.get_loc("high")] = 51500.0
        df.iloc[40, df.columns.get_loc("close")] = 50500.0

        feats, _ = build_structure(df)
        # No break at t=40
        assert feats["feat_struct_bars_since_break"].iloc[40] > 0.0


class TestCausalityAndParity:
    def test_pivot_invisible_before_confirmation(self):
        """A pivot at candle i must have zero effect on structure before confirmation at i+k."""
        df = _make_flat_df(60, price=50000.0)
        # Swing high at candle i=25 with high=52000
        df.iloc[25, df.columns.get_loc("high")] = 52000.0
        df.iloc[25, df.columns.get_loc("close")] = 50800.0

        feats, drawings = build_structure(df)
        # Swing high at i=25 is confirmed strictly at t = 25 + 5 = 30.
        # Before t=30, the resistance level at 52000 must NOT exist!
        for t in range(25, 30):
            assert drawings["level_resistance_price"].iloc[t] != 52000.0, (
                f"Level 52000 was prematurely visible at t={t} before confirmation at t=30!"
            )
        # At t=30, the level is confirmed and becomes visible
        assert drawings["level_resistance_price"].iloc[30] == 52000.0

    def test_future_alteration_invariance(self):
        """Altering candles after t leaves structure at t strictly identical."""
        df = _make_flat_df(80, price=50000.0)
        df.iloc[20, df.columns.get_loc("high")] = 51000.0
        df.iloc[35, df.columns.get_loc("low")] = 49000.0

        df_altered = df.copy()
        # Radically change candles from t=50 onwards
        df_altered.iloc[50:, df_altered.columns.get_loc("close")] = 99999.0
        df_altered.iloc[50:, df_altered.columns.get_loc("high")] = 105000.0

        f_orig, _ = build_structure(df)
        f_alt, _ = build_structure(df_altered)

        pd.testing.assert_frame_equal(f_orig.iloc[:50], f_alt.iloc[:50])

    def test_truncation_parity_on_train_rows(self):
        """Buffer of LIVE_BUFFER_CANDLES = 2880 gives exact zero difference on real train rows."""
        df_train = load_master(split="train").iloc[:10000].copy()
        raw_cols = ["open", "high", "low", "close", "volume", "gap_id"]
        
        f_full, _ = build_structure(df_train[raw_cols])
        
        # Pick 5 sample train rows after 5000
        rng = np.random.default_rng(123)
        sample_indices = rng.choice(np.arange(5000, 9500), size=5, replace=False)
        
        L = 2880
        for idx in sample_indices:
            buf = df_train[raw_cols].iloc[idx - L + 1 : idx + 1]
            f_buf, _ = build_structure(buf)
            
            diff = (f_buf.iloc[-1] - f_full.iloc[idx]).abs().max()
            assert diff < 1e-4, f"Truncation mismatch at row {idx}: max diff = {diff}"


class TestStructureHygiene:
    def test_no_raw_price_levels_in_role_feature(self):
        """No feature column may contain raw price levels in USDT."""
        df_train = load_master(split="train").iloc[:200]
        feat_cols = get_feature_columns(df_train)
        
        for c in feat_cols:
            assert not c.endswith("_price"), f"Feature {c} contains raw price suffix"
            vals = df_train[c].dropna()
            if len(vals) > 0 and c.startswith("feat_struct_dist_"):
                # Ratios and ATR distances should not be in the thousands (like BTC price 50,000)
                assert vals.abs().max() < 1000.0, f"Feature {c} has suspicious price-like values: {vals.max()}"

    def test_gap_boundary_isolation(self):
        """Structure engine resets on gap_id change and no level spans across gaps."""
        df = _make_flat_df(80, price=50000.0)
        df.iloc[:40, df.columns.get_loc("gap_id")] = 1
        df.iloc[40:, df.columns.get_loc("gap_id")] = 2

        # Form a level in gap 1
        df.iloc[15, df.columns.get_loc("high")] = 52000.0

        feats, _ = build_structure(df)
        # At t=40 (gap transition), trend and state must reset
        assert feats["feat_struct_bars_since_break"].iloc[40] >= 9999
        assert feats["feat_struct_trend_up"].iloc[40] == 0.0

    def test_drawing_columns_role_separation(self):
        """Drawings columns are strictly role 'drawing' and not in get_feature_columns()."""
        feat_cols = set(get_feature_columns())
        for c in DRAWING_COLUMNS:
            assert c not in feat_cols, f"Drawing column {c} leaked into feature columns"
            assert get_column_role(c) == "drawing", f"Column {c} has wrong role: {get_column_role(c)}"

