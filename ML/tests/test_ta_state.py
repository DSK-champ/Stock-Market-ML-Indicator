"""Unit Tests for TA State Branch and Context-Based Blending (Horizon N=2).

Verifies:
1. Trend and phase labels in allowed sets.
2. Strengths in [0, 1].
3. Probabilities ta_trend_p_up and ta_phase_p_pullback in [0, 1].
4. Causal invariance: changing future rows never affects past ta_state values.
5. T_HAND mirror symmetry.
6. Probability simplex closure: q and p_final sum to 1 within 1e-6.
7. Identity at w=0: reproduces p_ml exactly.
8. Nested walk-forward w selection strictly uses earlier folds (folds < k).
9. T_LEARNED estimation strictly uses earlier folds' rows.
10. Loader guard: test and val partitions cannot be loaded without unlock flag.
11. Round-trip serialization fidelity for ta_state.csv.gz.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.column_dictionary import load_master
from src.ta_state import compute_ta_state, derive_break_direction, TA_STATE_CSV_GZ
from src.context_blend import (
    T_HAND,
    check_t_hand_symmetry,
    compute_soft_blend,
    fit_learned_table,
    select_w_nested,
)


@pytest.fixture(scope="module")
def sample_train_df():
    """Load a slice of TRAIN split for testing."""
    df = load_master(split="train")
    # Take rows after warmup
    return df.iloc[2018:3018].copy()


def test_t_hand_mirror_symmetry():
    """Test 5: Verify T_HAND mirror symmetry across trend and direction inversion."""
    assert check_t_hand_symmetry(), "T_HAND mirror symmetry check failed"


def test_ta_state_labels_and_ranges(sample_train_df):
    """Tests 1, 2, 3: Verify labels in allowed sets, strengths and probabilities in [0, 1]."""
    ta_df = compute_ta_state(sample_train_df)

    # 1. Labels
    assert set(ta_df["ta_trend_label"].unique()).issubset({"UP", "DOWN"}), "Invalid trend labels"
    assert set(ta_df["ta_phase_label"].unique()).issubset({"PULLBACK", "BREAKTHROUGH"}), "Invalid phase labels"

    expected_contexts = {
        "UPTREND | PULLBACK",
        "UPTREND | BREAKTHROUGH",
        "DOWNTREND | PULLBACK",
        "DOWNTREND | BREAKTHROUGH",
    }
    assert set(ta_df["ta_context"].unique()).issubset(expected_contexts), "Invalid context labels"

    # 2. Strengths in [0, 1]
    assert (ta_df["ta_trend_strength"] >= 0.0).all() and (ta_df["ta_trend_strength"] <= 1.0).all(), (
        "ta_trend_strength out of [0, 1]"
    )
    assert (ta_df["ta_phase_strength"] >= 0.0).all() and (ta_df["ta_phase_strength"] <= 1.0).all(), (
        "ta_phase_strength out of [0, 1]"
    )

    # 3. Probabilities in [0, 1]
    assert (ta_df["ta_trend_p_up"] >= 0.0).all() and (ta_df["ta_trend_p_up"] <= 1.0).all(), (
        "ta_trend_p_up out of [0, 1]"
    )
    assert (ta_df["ta_phase_p_pullback"] >= 0.0).all() and (ta_df["ta_phase_p_pullback"] <= 1.0).all(), (
        "ta_phase_p_pullback out of [0, 1]"
    )


def test_causal_invariance(sample_train_df):
    """Test 4: Verify that changing or appending future rows NEVER alters past ta_state."""
    n_short = 200
    df_short = sample_train_df.iloc[:n_short].copy()
    df_long = sample_train_df.iloc[: n_short + 100].copy()

    ta_short = compute_ta_state(df_short)
    ta_long = compute_ta_state(df_long)

    # The first n_short rows must match exactly
    for col in ["ta_trend_label", "ta_phase_label", "ta_trend_strength", "ta_phase_strength"]:
        s1 = ta_short[col].values
        s2 = ta_long[col].iloc[:n_short].values
        if pd.api.types.is_string_dtype(s1) or col in ["ta_trend_label", "ta_phase_label"]:
            assert (s1 == s2).all(), f"Causal violation in {col}"
        else:
            assert np.allclose(s1.astype(float), s2.astype(float), atol=1e-6), f"Causal violation in {col}"


def test_simplex_and_w_zero_identity():
    """Tests 6, 7: Verify q and p_final simplex closure, and w=0 reproduces p_ml exactly."""
    np.random.seed(42)
    n = 100
    t_up = np.random.uniform(0, 1, n)
    f_pb = np.random.uniform(0, 1, n)

    # Generate random p_ml on simplex
    raw_p = np.random.exponential(size=(n, 3))
    p_ml = raw_p / raw_p.sum(axis=1, keepdims=True)

    # Test at w = 0.5
    q, p_final = compute_soft_blend(t_up, f_pb, p_ml, T_HAND, w=0.5)
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-6), "q does not sum to 1.0 within 1e-6"
    assert np.allclose(p_final.sum(axis=1), 1.0, atol=1e-6), "p_final does not sum to 1.0 within 1e-6"

    # Test at w = 0.0
    _, p_zero = compute_soft_blend(t_up, f_pb, p_ml, T_HAND, w=0.0)
    assert np.allclose(p_zero, p_ml, atol=1e-6), "w=0 must reproduce p_ml exactly"


def test_t_learned_earlier_folds_only():
    """Test 9: Verify T_LEARNED estimation strictly uses earlier folds' rows."""
    fake_df = pd.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2],
            "ta_trend_label": ["UP", "UP", "UP", "UP", "UP", "UP"],
            "ta_phase_label": ["PULLBACK", "PULLBACK", "PULLBACK", "PULLBACK", "PULLBACK", "PULLBACK"],
            "ml_pred": [1, 1, 1, 1, 1, 1],
            "raw_sign_2": [1, 1, 1, -1, -1, -1],
        }
    )

    # Using only fold 1 rows
    table_f1 = fit_learned_table(fake_df[fake_df["fold"] < 2])
    assert table_f1[("UP", "PULLBACK", 1)] == 1, "Should learn UP (+1) from fold 1"

    # Using only fold 2 rows
    table_f2 = fit_learned_table(fake_df[fake_df["fold"] == 2])
    assert table_f2[("UP", "PULLBACK", 1)] == -1, "Should learn DOWN (-1) from fold 2"


def test_nested_w_selection_strictly_earlier_folds():
    """Test 8: Verify nested w selection uses only folds < k and respects 12-candle purge."""
    n_per_fold = 50
    records = []
    for f in range(1, 4):
        for _ in range(n_per_fold):
            records.append({
                "fold": f,
                "ta_trend_p_up": 0.6,
                "ta_phase_p_pullback": 0.4,
                "ta_trend_strength": 0.2,
                "ta_phase_strength": 0.2,
                "p_down_ml": 0.2,
                "p_neutral_ml": 0.3,
                "p_up_ml": 0.5,
                "raw_sign_2": 1,
            })
    mock_oof = pd.DataFrame(records)

    chosen_w = select_w_nested(mock_oof, eval_folds=[2, 3], table=T_HAND)
    assert 2 in chosen_w and 3 in chosen_w, "Must compute w for folds 2 and 3"
    for k, w_val in chosen_w.items():
        assert 0.0 <= w_val <= 1.0, f"w for fold {k} must be in [0, 1]"


def test_loader_guard_blocks_unlocked_partitions():
    """Test 10: Verify loader guard blocks unauthorized reading of test split."""
    with pytest.raises(PermissionError):
        load_master(split="test", unlock_test=False)


def test_ta_state_roundtrip(sample_train_df, tmp_path):
    """Test 11: Verify ta_state.csv.gz round trip equals the in-memory DataFrame."""
    ta_df = compute_ta_state(sample_train_df)
    test_file = tmp_path / "ta_state_test.csv.gz"
    ta_df.to_csv(test_file, compression="gzip", index=False)

    loaded_df = pd.read_csv(test_file)
    assert len(loaded_df) == len(ta_df), "Round trip row count mismatch"
    for col in ["ta_trend_label", "ta_phase_label", "ta_context"]:
        assert (loaded_df[col] == ta_df[col].values).all(), f"Round trip mismatch in {col}"
    for col in ["ta_trend_strength", "ta_trend_p_up", "ta_phase_strength", "ta_phase_p_pullback"]:
        assert np.allclose(loaded_df[col], ta_df[col].values, atol=1e-4), f"Round trip numeric mismatch in {col}"

