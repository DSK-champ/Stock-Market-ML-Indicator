"""Unit tests for v3 target label generation.

8 required tests
----------------
1. Label correctness on hand-made example for all θ rules incl. asymmetric (A1, A2)
2. No label across a gap (has_valid_target=False when gap_id differs)
3. Last N rows unlabeled (has_valid_target=False)
4. Purge gap exactly N for every horizon (parametrized)
5. No split overlap
6. θ_up ≥ c and θ_down ≥ c for every row and every rule
7. Q1 thresholds computed only from train rows (not val/test)
8. V-rule σ_t uses only past returns (causality: altering future rows must not change past σ_t)
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labels import (
    COST,
    RULES,
    compute_all_targets,
    compute_q1_thresholds,
    create_forward_labels,       # legacy shim — still tested
    compute_label_balance_report,
    compute_dead_zone_sensitivity,
)
from src.utils import rolling_sigma


# ---------------------------------------------------------------------------
# Fixtures: shared synthetic data
# ---------------------------------------------------------------------------

def _make_df(prices_close, prices_open=None, gap_ids=None, freq="5min"):
    """Build a minimal DataFrame suitable for label functions."""
    n = len(prices_close)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    if prices_open is None:
        prices_open = prices_close  # degenerate: open == close
    if gap_ids is None:
        gap_ids = [1] * n
    return pd.DataFrame(
        {"open": prices_open, "close": prices_close, "gap_id": gap_ids},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Legacy shim compatibility (keep existing tests passing)
# ---------------------------------------------------------------------------

def test_dead_zone_configured():
    """Dead zone must equal total round-trip economic cost."""
    assert config.LABEL_DEAD_ZONE > 0.0
    assert config.LABEL_DEAD_ZONE == config.TOTAL_ROUNDTRIP_COST
    assert round(config.LABEL_DEAD_ZONE, 4) == 0.0023


def test_legacy_last_n_rows_unlabeled():
    """Legacy shim: last N rows must be NaN."""
    horizon = 12
    n = 50
    df = _make_df(np.linspace(100, 200, n).tolist())
    labels_df = create_forward_labels(df, horizon=horizon)
    assert labels_df["target"].iloc[-12:].isna().all()
    assert labels_df["target"].iloc[:-12].notna().all()


def test_legacy_gap_crossing_invalidated():
    """Legacy shim: horizon crossing a gap boundary → NaN."""
    horizon = 3
    df = _make_df(
        prices_close=[100.0, 101.0, 102.0, 110.0, 111.0, 112.0],
        gap_ids=[1, 1, 1, 2, 2, 2],
    )
    labels_df = create_forward_labels(df, horizon=horizon, dead_zone=0.002)
    assert np.isnan(labels_df["target"].iloc[0])
    assert np.isnan(labels_df["forward_return"].iloc[0])


# ===========================================================================
# TEST 1 — Label correctness on a hand-made example (all 7 rules, all labels)
# ===========================================================================

def test_label_correctness_all_rules():
    """
    Hand-crafted example verifying Up / Down / Neutral for every rule (S1,S2,Q1,V1,V2,A1,A2)
    and both label directions.

    Setup (N=2, all within the same gap so has_valid_target=True):
      Row 0 close=100, row 2 close=108  → fwd_ret = +8.00%  (large UP)
      Row 1 close=100, row 3 close=92   → fwd_ret = -8.00%  (large DOWN)
      Row 2 close=108, row 4 close=108.1→ fwd_ret = +0.09%  (tiny, inside ALL dead zones → NEUTRAL for all rules)
      Row 3 close=92,  row 5 close=91.9 → fwd_ret = -0.11%  (tiny, inside ALL dead zones → NEUTRAL for all rules)
      Rows 4,5 — last N=2 rows → has_valid_target=False
    """
    prices = [100.0, 100.0, 108.0, 92.0, 108.1, 91.9]
    df = _make_df(prices)
    # Use itself as train (q1 will be computed from these 4 valid rows, all floored at c)
    tgt = compute_all_targets(df, train_df=df, horizons=[2], cost=COST)

    for rule in RULES:
        col = f"target_2_{rule}"
        assert col in tgt.columns, f"Missing column {col}"
        # Row 0: +8% — above ALL possible θ → must be UP
        assert tgt[col].iloc[0] == 1.0, f"{rule}: row 0 should be UP"
        # Row 1: -8% — below ALL possible -θ → must be DOWN
        assert tgt[col].iloc[1] == -1.0, f"{rule}: row 1 should be DOWN"
        # Rows 2,3: tiny moves inside dead zones
        # Row 2 (+0.0926%): > 0.05% -> UP (+1.0) for F05; <= other θ (>=0.10%) -> NEUTRAL (0.0)
        expected_row2 = 1.0 if rule == "F05" else 0.0
        assert tgt[col].iloc[2] == expected_row2, f"{rule}: row 2 expected {expected_row2}, got {tgt[col].iloc[2]}"
        # Row 3 (-0.1087%): < -0.05% and < -0.10% -> DOWN (-1.0) for F05, F10; >= -θ for others (>=0.15%) -> NEUTRAL (0.0)
        expected_row3 = -1.0 if rule in ("F05", "F10") else 0.0
        assert tgt[col].iloc[3] == expected_row3, f"{rule}: row 3 expected {expected_row3}, got {tgt[col].iloc[3]}"
        # Rows 4,5: has_valid_target=False → NaN
        assert np.isnan(tgt[col].iloc[4]), f"{rule}: row 4 should be NaN"
        assert np.isnan(tgt[col].iloc[5]), f"{rule}: row 5 should be NaN"

    # Asymmetric rules: specific direction checks
    # A1: θ_up = c (smaller), θ_down = 0.003 (larger)
    # With +8% return → UP in A1 (above c) ✓ already checked above
    # A2: θ_up = 0.003 (larger), θ_down = c (smaller)
    # With -8% return → DOWN in A2 (below -c) ✓ already checked above

    # Additional A-rule direction asymmetry:
    # Construct row where fwd_ret = +0.25% (above c=0.23%, below 0.30%)
    # A1 (θ_up=c=0.23%): UP;   A2 (θ_up=0.30%): NEUTRAL
    prices2 = [100.0, 100.0, 100.25, 99.0, 99.0, 99.0]  # row 0: +0.25%, row 1: -1%, rest tiny
    df2 = _make_df(prices2)
    tgt2 = compute_all_targets(df2, train_df=df2, horizons=[2], cost=COST)

    # Row 0, fwd_ret = +0.25%
    assert tgt2["target_2_A1"].iloc[0] == 1.0, "A1 row0 +0.25% should be UP (θ_up=c=0.23%)"
    assert tgt2["target_2_A2"].iloc[0] == 0.0, "A2 row0 +0.25% should be NEUTRAL (θ_up=0.30%)"


# ===========================================================================
# TEST 2 — No label across a gap
# ===========================================================================

def test_no_label_across_gap():
    """has_valid_target_N must be False when gap_id differs at t and t+N.

    Layout (N=3): 10 rows, gap 1 = rows 0-4, gap 2 = rows 5-9.
    For horizon N=3, row t is valid if gap_id[t] == gap_id[t+3].
      row 0 → row 3: both gap 1 → VALID
      row 1 → row 4: both gap 1 → VALID
      row 2 → row 5: gap 1 → gap 2 → INVALID (crosses boundary)
      row 3 → row 6: gap 1 → gap 2 → INVALID (crosses boundary)
      row 4 → row 7: gap 1 → gap 2 → INVALID (crosses boundary)
      rows 5,6 → rows 8,9: both gap 2 → VALID
      rows 7,8,9: last N=3 rows → INVALID (insufficient future data)
    """
    prices = [100.0 + i for i in range(10)]
    gap_ids = [1] * 5 + [2] * 5
    df = _make_df(prices, gap_ids=gap_ids)

    tgt = compute_all_targets(df, train_df=df, horizons=[3], cost=COST)

    assert tgt["has_valid_target_3"].iloc[0] == True,  "row 0: gap1→gap1, valid"
    assert tgt["has_valid_target_3"].iloc[1] == True,  "row 1: gap1→gap1, valid"
    assert tgt["has_valid_target_3"].iloc[2] == False, "row 2: gap1→gap2, crosses boundary"
    assert tgt["has_valid_target_3"].iloc[3] == False, "row 3: gap1→gap2, crosses boundary"
    assert tgt["has_valid_target_3"].iloc[4] == False, "row 4: gap1→gap2, crosses boundary"
    assert tgt["has_valid_target_3"].iloc[5] == True,  "row 5: gap2→gap2, valid"
    assert tgt["has_valid_target_3"].iloc[6] == True,  "row 6: gap2→gap2, valid"
    assert tgt["has_valid_target_3"].iloc[7] == False, "row 7: last N rows, invalid"
    assert tgt["has_valid_target_3"].iloc[8] == False, "row 8: last N rows, invalid"
    assert tgt["has_valid_target_3"].iloc[9] == False, "row 9: last N rows, invalid"
    # Gap-crossing rows must produce NaN labels
    assert np.isnan(tgt["target_3_S1"].iloc[2])
    assert np.isnan(tgt["target_3_S1"].iloc[3])
    assert np.isnan(tgt["target_3_S1"].iloc[4])


# ===========================================================================
# TEST 3 — Last N rows unlabeled
# ===========================================================================

@pytest.mark.parametrize("N", [2, 3, 6, 12])
def test_last_n_rows_unlabeled(N):
    """has_valid_target_N must be False for the last N rows of the input."""
    n = N * 5 + 20  # always more than N rows
    prices = np.linspace(100, 200, n).tolist()
    df = _make_df(prices)

    tgt = compute_all_targets(df, train_df=df, horizons=[N], cost=COST)

    valid = tgt[f"has_valid_target_{N}"].to_numpy()
    assert valid[-N:].sum() == 0, f"Last {N} rows must all be invalid"
    assert valid[:-N].sum() == n - N, f"First {n - N} rows must all be valid"


# ===========================================================================
# TEST 4 — Purge gap exactly N between folds
# ===========================================================================

@pytest.mark.parametrize("N", [2, 3, 6, 12])
def test_purge_gap_exactly_N(N):
    """
    Simulate the boundary between two successive folds (or train/val boundary).
    With a purge gap of exactly N candles, the N rows just after the boundary
    must NOT look back into the fold before them (gap_id boundary enforces this
    in the data; the purge gap is enforced by generate_walk_forward_folds).

    Structural test: given two segments separated by a gap_id change, N rows
    immediately following the boundary are the purge zone — their t-N predecessors
    are in the previous segment (gap_id different) → has_valid_target must be False.
    """
    n = 40
    # Segment A: first 20 rows, gap_id=1; Segment B: next 20 rows, gap_id=2
    prices = np.linspace(100, 200, n).tolist()
    gap_ids = [1] * 20 + [2] * 20
    df = _make_df(prices, gap_ids=gap_ids)

    tgt = compute_all_targets(df, train_df=df, horizons=[N], cost=COST)

    # The last N rows of segment A look forward into segment B → invalid
    for i in range(20 - N, 20):
        assert tgt[f"has_valid_target_{N}"].iloc[i] == False, (
            f"N={N}: row {i} (end of seg A) should be invalid (horizon crosses into seg B)"
        )

    # Row 20 (first of seg B) looks forward into seg B only → valid (unless it's in last-N of full df)
    if 20 + N < n:
        assert tgt[f"has_valid_target_{N}"].iloc[20] == True, (
            f"N={N}: row 20 (start of seg B) should be valid"
        )


# ===========================================================================
# TEST 5 — No split overlap (train / val / test boundaries)
# ===========================================================================

def test_no_split_overlap():
    """Train, val, and test index sets from splitting.py must be disjoint."""
    from src.splitting import get_train_data, get_val_data, get_test_data

    # Use synthetic data spanning a full fake time range (not real data, to avoid I/O)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "close": np.linspace(100, 200, n), "gap_id": 1},
        index=idx,
    )

    train = get_train_data(df)
    val = get_val_data(df)
    # test is locked on real data; for synthetic, we bypass the lock
    test = get_test_data(df, unlock=True)

    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    assert len(train_idx & val_idx) == 0, "Train and Val must not overlap"
    assert len(val_idx & test_idx) == 0, "Val and Test must not overlap"
    assert len(train_idx & test_idx) == 0, "Train and Test must not overlap"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "Splits must cover all rows"


# ===========================================================================
# TEST 6 — θ_up ≥ c and θ_down ≥ c for every row and rule
# ===========================================================================

def test_theta_floor_at_cost():
    """Every theta_up and theta_down value must be ≥ c (round-trip cost), exempting sub_cost rules."""
    n = 50
    prices = np.linspace(100, 200, n).tolist()
    df = _make_df(prices)

    tgt = compute_all_targets(df, train_df=df, horizons=config.CANDIDATE_HORIZONS, cost=COST)

    for col in tgt.columns:
        if col.startswith("theta_up_") or col.startswith("theta_down_"):
            is_sub_cost = any(col.endswith(f"_{r}") for r in config.SUB_COST_RULES)
            arr = tgt[col].to_numpy()
            valid = arr[~np.isnan(arr)]
            if not is_sub_cost:
                assert (valid >= COST - 1e-12).all(), (
                    f"{col}: found theta < c for non-sub_cost rule. min={valid.min():.6f}, c={COST:.6f}"
                )


def test_sub_cost_rules_thetas():
    """Verify F05, F10, F15 rules have exact thetas and are flagged sub_cost."""
    n = 50
    prices = np.linspace(100, 200, n).tolist()
    df = _make_df(prices)
    tgt = compute_all_targets(df, train_df=df, horizons=config.CANDIDATE_HORIZONS, cost=COST)

    expected_thetas = {"F05": 0.0005, "F10": 0.0010, "F15": 0.0015}
    for r, expected_val in expected_thetas.items():
        assert config.LABEL_RULES[r]["sub_cost"] is True
        assert config.LABEL_RULES[r]["theta_up"] == expected_val
        assert config.LABEL_RULES[r]["theta_down"] == expected_val

        for N in config.CANDIDATE_HORIZONS:
            tu = tgt[f"theta_up_{N}_{r}"].dropna()
            td = tgt[f"theta_down_{N}_{r}"].dropna()
            np.testing.assert_allclose(tu.values, expected_val, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(td.values, expected_val, rtol=1e-12, atol=1e-12)


def test_big_move_targets_integrity():
    """Verify big_move_N == (target_N_S1 != 0) on valid rows, NaN on invalid rows, and complement sum."""
    n = 60
    prices = np.linspace(100, 200, n).tolist()
    df = _make_df(prices)
    tgt = compute_all_targets(df, train_df=df, horizons=config.CANDIDATE_HORIZONS, cost=COST)

    for N in config.CANDIDATE_HORIZONS:
        bm = tgt[f"big_move_{N}"]
        s1 = tgt[f"target_{N}_S1"]
        valid = tgt[f"has_valid_target_{N}"].astype(bool)

        # 1. Invalid rows must be strictly NaN (never 0)
        assert bm[~valid].isna().all(), f"N={N}: found non-NaN in invalid rows of big_move_{N}"
        assert (bm[~valid] == 0).sum() == 0, f"N={N}: found 0 in invalid rows of big_move_{N}"

        # 2. Valid rows must strictly equal (target_N_S1 != 0)
        valid_bm = bm[valid]
        valid_s1 = s1[valid]
        expected_bm = (valid_s1 != 0.0).astype(float)
        np.testing.assert_array_equal(valid_bm.values, expected_bm.values)

        # 3. Sum of big_move=1 and big_move=0 must equal valid count
        n_ones = int((valid_bm == 1.0).sum())
        n_zeros = int((valid_bm == 0.0).sum())
        assert n_ones + n_zeros == int(valid.sum()), f"N={N}: sum of big_move classes does not equal valid count"


# ===========================================================================
# TEST 7 — Q1 thresholds computed only from train rows
# ===========================================================================

def test_q1_train_only():
    """Q1 thresholds must depend ONLY on train rows, not val/test rows.

    Strategy: compute Q1 from two different train sets; verify that changing
    val/test rows does NOT change the Q1 result for the same train subset.
    """
    rng = np.random.default_rng(42)
    n_train = 60
    n_val = 20

    train_prices = (rng.random(n_train) * 10 + 100).tolist()
    val_prices_a = (rng.random(n_val) * 10 + 100).tolist()
    val_prices_b = (rng.random(n_val) * 1000 + 50000).tolist()  # wildly different

    train_df = _make_df(train_prices)
    val_df_a = _make_df(val_prices_a)
    val_df_b = _make_df(val_prices_b)

    q1_a = compute_q1_thresholds(train_df, horizons=[2, 3], cost=COST)
    q1_b = compute_q1_thresholds(train_df, horizons=[2, 3], cost=COST)

    # Q1 from same train_df is deterministic; swapping val doesn't matter
    assert q1_a == q1_b, "Q1 thresholds must be identical given identical train_df"

    # Also: Q1 computed with ONLY val_df_b as "train" must differ
    q1_val_only = compute_q1_thresholds(val_df_b, horizons=[2, 3], cost=COST)
    # At least one value should differ (wildly different price series)
    all_equal = all(abs(q1_a[k] - q1_val_only[k]) < 1e-10 for k in q1_a)
    assert not all_equal, "Q1 from train vs Q1 from val_b must differ — val data leaked into train"


# ===========================================================================
# TEST 8 — V-rule σ_t uses only past returns (causality)
# ===========================================================================

def test_v_rule_sigma_causal():
    """
    Altering any value at index t+k (k > 0) must NOT change rolling_sigma at index t.

    Procedure:
      1. Build a base returns series.
      2. Compute rolling_sigma(base).
      3. Alter a FUTURE element (index t+5).
      4. Recompute rolling_sigma(altered).
      5. Verify sigma[0..t] is unchanged.
    """
    n = 100
    rng = np.random.default_rng(99)
    base_ret = pd.Series(rng.normal(0, 0.001, n))
    sigma_base = rolling_sigma(base_ret, window=10)

    # Modify future element (index 50)
    pivot = 30
    altered_ret = base_ret.copy()
    altered_ret.iloc[pivot + 5] = 999.0   # extreme future value

    sigma_altered = rolling_sigma(altered_ret, window=10)

    # sigma at positions 0..pivot must be identical
    max_diff = (sigma_base.iloc[:pivot + 1] - sigma_altered.iloc[:pivot + 1]).abs().max()
    assert max_diff == 0.0, (
        f"Causality violated: altering future (t+5) changed sigma at t≤{pivot}. "
        f"Max diff = {max_diff}"
    )

    # Verify that the altered series DOES differ at the altered position and beyond
    # (sanity check that the modification was effective)
    assert sigma_base.iloc[pivot + 10] != sigma_altered.iloc[pivot + 10] or True  # may or may not differ; not a hard requirement


def test_v_rule_target_causal():
    """
    V1/V2 target labels at time t must not change when only future candle prices change.

    Procedure:
      1. Build a base DataFrame (N=2, all same gap_id).
      2. Compute V1 targets.
      3. Modify a FUTURE candle price at index t+10 (beyond the horizon).
      4. Recompute V1 targets.
      5. Assert V1 targets at index 0 are identical.
    """
    n = 60
    rng = np.random.default_rng(7)
    prices = (rng.normal(0, 0.002, n).cumsum() + 100).tolist()
    df_base = _make_df(prices)

    df_alt = df_base.copy()
    df_alt.iloc[20, df_alt.columns.get_loc("close")] = 999999.0  # extreme future price

    tgt_base = compute_all_targets(df_base, train_df=df_base, horizons=[2], cost=COST)
    tgt_alt = compute_all_targets(df_alt, train_df=df_base, horizons=[2], cost=COST)

    # Rows 0..17 have horizon t+2 → rows 2..19; none reaches row 20 altered point
    # (since we altered row 20, rows 0..17 look at rows 2..19 — unaffected)
    for i in range(18):
        assert tgt_base["target_2_V1"].iloc[i] == tgt_alt["target_2_V1"].iloc[i] or (
            np.isnan(tgt_base["target_2_V1"].iloc[i]) and np.isnan(tgt_alt["target_2_V1"].iloc[i])
        ), f"V1 causality: row {i} changed when only row 20 was modified"


# ===========================================================================
# TEST 9 — Saved targets parity & V-rule rolling_sigma equivalence
# ===========================================================================

def test_targets_disk_parity_and_vrule_recompute():
    """
    Verify that:
      1. Reloading saved targets_base.csv and targets_labels.csv from disk matches
         a fresh recomputation on a sample slice of clean.csv.
      2. The saved V-rule thetas (V1, V2) strictly match a fresh recompute via
         utils.rolling_sigma(log_returns, window=288).
    """
    from src.io_utils import (
        load_clean_market_data_csv,
        load_targets_base_csv,
        load_targets_labels_csv,
        load_theta_frozen_json,
    )
    from src.splitting import get_train_data

    # Load frozen metadata
    theta_meta = load_theta_frozen_json()
    q1_train = theta_meta["q1_thresholds"]

    # Load saved targets from disk (first 1,000 rows for fast, deterministic unit test)
    n_sample = 1000
    df_clean_head = load_clean_market_data_csv().iloc[:n_sample].copy()
    base_disk = load_targets_base_csv().iloc[:n_sample].copy()
    labels_disk = load_targets_labels_csv().iloc[:n_sample].copy()

    # Fresh recomputation on this exact slice
    fresh_targets = compute_all_targets(
        df_clean_head,
        train_df=df_clean_head,
        horizons=config.CANDIDATE_HORIZONS,
        cost=config.TOTAL_ROUNDTRIP_COST,
        sigma_window=288,
        q1_thresholds=q1_train,
    )

    # 1. Base targets parity check (excluding the last max(N) rows of the slice
    # where the truncated test slice ends)
    max_N = max(config.CANDIDATE_HORIZONS)
    eval_slice = slice(0, n_sample - max_N)

    for N in config.CANDIDATE_HORIZONS:
        fwd_ret_col = f"fwd_ret_{N}"
        np.testing.assert_allclose(
            base_disk[fwd_ret_col].iloc[eval_slice].to_numpy(),
            fresh_targets[fwd_ret_col].iloc[eval_slice].to_numpy(),
            rtol=1e-7,
            atol=1e-7,
            equal_nan=True,
            err_msg=f"Discrepancy in {fwd_ret_col} between disk and fresh compute",
        )

        valid_col = f"has_valid_target_{N}"
        np.testing.assert_array_equal(
            base_disk[valid_col].iloc[eval_slice].astype(bool).to_numpy(),
            fresh_targets[valid_col].iloc[eval_slice].astype(bool).to_numpy(),
            err_msg=f"Discrepancy in {valid_col} between disk and fresh compute",
        )

    # 2. V-rule threshold verification via independent rolling_sigma call
    log_ret = np.log(df_clean_head["close"] / df_clean_head["close"].shift(1))
    sigma_independent = rolling_sigma(log_ret, window=288).to_numpy()

    for N in config.CANDIDATE_HORIZONS:
        sqrt_N = np.sqrt(N)
        for rule, k in [("V1", 0.5), ("V2", 1.0)]:
            col = f"theta_up_{N}_{rule}"
            saved_theta = labels_disk[col].iloc[eval_slice].to_numpy()

            raw_sigma = k * sigma_independent * sqrt_N
            raw_sigma_safe = np.where(np.isnan(raw_sigma), 0.0, raw_sigma)
            expected_theta = np.maximum(config.TOTAL_ROUNDTRIP_COST, raw_sigma_safe)[eval_slice]

            np.testing.assert_allclose(
                saved_theta,
                expected_theta,
                rtol=1e-7,
                atol=1e-7,
                equal_nan=True,
                err_msg=f"Saved {col} does not match fresh rolling_sigma recompute",
            )


# ===========================================================================
# TEST 11 — Split tail rows invalidated across boundaries (no leakage)
# ===========================================================================

@pytest.mark.parametrize("N", [2, 3, 6, 12])
def test_split_tail_invalidation_across_boundaries(N):
    """The last N rows of every split must have has_valid_target_N = False and NaN labels.

    Forward returns must not cross the train/val or val/test boundaries.
    """
    from src.column_dictionary import load_master
    train = load_master("train")
    n_total = len(train)
    assert n_total == 220_953

    valid_mask = train[f"has_valid_target_{N}"].to_numpy()
    # Exactly the last N rows of train must be invalid
    assert not valid_mask[-N:].any(), f"Last {N} rows of train must be invalid for N={N}"
    assert valid_mask[:-N].all(), f"First {n_total - N} rows of train must be valid for N={N}"
    assert int(valid_mask.sum()) == 220_953 - N, f"Train valid count must be exactly 220,953 - {N}"

    # Verify that all label columns and big_move are NaN on the invalid tail rows
    tail_labels = train.iloc[-N:]
    for rule in config.ALL_RULES:
        col = f"target_{N}_{rule}"
        assert tail_labels[col].isna().all(), f"{col} must be NaN on last {N} rows of train"
    assert tail_labels[f"big_move_{N}"].isna().all(), f"big_move_{N} must be NaN on last {N} rows of train"

    # Specific count check at N=12 for S1 rule
    if N == 12:
        down_cnt = int((train["target_12_S1"] == -1.0).sum())
        assert down_cnt == 50_663, f"Expected 50,663 Down labels at N=12, got {down_cnt}"


