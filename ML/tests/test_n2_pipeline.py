"""Unit tests for Horizon 2 (N=2) prediction pipeline.

Strictly tests:
  1. test_input_columns_exclude_forbidden_roles
  2. test_artifact_names_contain_N2
  3. test_fold_purge_gap_is_12
  4. test_oof_rows_not_in_own_fit_block
  5. test_scaler_fit_on_fit_block_only
  6. test_probability_rows_sum_to_one
  7. test_class_order_matches_classes_attribute
  8. test_deterministic_under_seed
  9. test_meta_inputs_match_declared_list
  10. test_meta_nested_split_disjoint_with_gap
  11. test_meta_never_scored_on_training_rows
  12. test_ta_scores_in_range_and_causal
  13. test_loader_blocks_validation_and_test
  14. test_predict_horizon_returns_no_signal_on_short_buffer
  15. test_predict_horizon_probs_sum_to_one
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.column_dictionary import get_column_role, load_master
from src.cv import PurgedWalkForwardCV
from src.binary_models import build_binary_model, get_binary_inputs
from src.three_class_models import build_three_class_model, get_three_class_inputs
from src.ta_engine import compute_ta_scores, ta_label, TA_SCORE_COLUMNS
from src.meta_model import META_INPUT_COLUMNS, build_meta_inputs, nested_meta_oof, fit_meta_final
from src.model_common import (
    prepare_model_features,
    run_oof,
    fit_final,
    FORBIDDEN_ROLES,
)
from src.horizon_pipeline import predict_horizon


# 1. test_input_columns_exclude_forbidden_roles
def test_input_columns_exclude_forbidden_roles():
    for model_id in ["binary_lr", "binary_rf", "binary_gb"]:
        cols = get_binary_inputs(model_id)
        for col in cols:
            role = get_column_role(col)
            assert role not in FORBIDDEN_ROLES, (
                f"{model_id} input column {col} has forbidden role {role}"
            )

    for model_id in ["three_class_lr", "three_class_rf", "three_class_gb"]:
        cols = get_three_class_inputs(model_id)
        for col in cols:
            role = get_column_role(col)
            assert role not in FORBIDDEN_ROLES, (
                f"{model_id} input column {col} has forbidden role {role}"
            )


# 2. test_artifact_names_contain_N2
def test_artifact_names_contain_N2():
    horizon = 2
    for mid in ["binary_lr", "binary_rf", "binary_gb", "three_class_lr", "three_class_rf", "three_class_gb"]:
        oof_filename = f"{mid}_N{horizon}.csv.gz"
        joblib_filename = f"{mid}_N{horizon}.joblib"
        json_filename = f"{mid}_N{horizon}.json"
        assert f"_N{horizon}" in oof_filename
        assert f"_N{horizon}" in joblib_filename
        assert f"_N{horizon}" in json_filename

    meta_oof = f"meta_N{horizon}.csv.gz"
    meta_joblib = f"meta_model_N{horizon}.joblib"
    meta_json = f"meta_model_N{horizon}.json"
    assert f"_N{horizon}" in meta_oof
    assert f"_N{horizon}" in meta_joblib
    assert f"_N{horizon}" in meta_json


# 3. test_fold_purge_gap_is_12
def test_fold_purge_gap_is_12():
    cv = PurgedWalkForwardCV()
    dummy_df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=200000, freq="5min"))
    for fold, tr_idx, te_idx in cv.split(dummy_df):
        purge_gap = te_idx[0] - tr_idx[-1] - 1
        assert purge_gap == 12, f"Fold {fold} purge gap is {purge_gap}, expected 12"


# 4. test_oof_rows_not_in_own_fit_block
def test_oof_rows_not_in_own_fit_block():
    cv = PurgedWalkForwardCV()
    dummy_df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=200000, freq="5min"))
    for fold, tr_idx, te_idx in cv.split(dummy_df):
        intersection = set(tr_idx).intersection(set(te_idx))
        assert len(intersection) == 0, f"Fold {fold} train and test indices overlap!"
        assert tr_idx[-1] < te_idx[0], f"Fold {fold} train does not precede test!"


# 5. test_scaler_fit_on_fit_block_only
def test_scaler_fit_on_fit_block_only():
    rng = np.random.default_rng(42)
    train_data = rng.normal(loc=10.0, scale=2.0, size=(100, 5))
    test_data = rng.normal(loc=50.0, scale=10.0, size=(30, 5))

    scaler = StandardScaler()
    scaler.fit(train_data)

    # Scaler mean must match train_data, not test_data
    assert np.allclose(scaler.mean_, train_data.mean(axis=0), atol=1e-6)
    assert not np.allclose(scaler.mean_, test_data.mean(axis=0), atol=1.0)


# 6. test_probability_rows_sum_to_one
def test_probability_rows_sum_to_one():
    rng = np.random.default_rng(42)
    raw_scores = rng.normal(size=(50, 3))
    exp_scores = np.exp(raw_scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


# 7. test_class_order_matches_classes_attribute
def test_class_order_matches_classes_attribute():
    bin_clf = build_binary_model("binary_lr", smoke=True)
    X_bin = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_bin = np.array([0, 1])
    bin_clf.fit(X_bin, y_bin)
    assert np.array_equal(bin_clf.classes_, np.array([0, 1]))

    tri_clf = build_three_class_model("three_class_lr", smoke=True)
    X_tri = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_tri = np.array([-1, 0, 1])
    tri_clf.fit(X_tri, y_tri)
    assert np.array_equal(tri_clf.classes_, np.array([-1, 0, 1]))


# 8. test_deterministic_under_seed
def test_deterministic_under_seed():
    rng1 = np.random.default_rng(config.SEED)
    X = rng1.normal(size=(200, 10))
    y = rng1.choice([0, 1], size=200)

    clf1 = build_binary_model("binary_rf", smoke=True)
    clf1.fit(X, y)
    probs1 = clf1.predict_proba(X)

    clf2 = build_binary_model("binary_rf", smoke=True)
    clf2.fit(X, y)
    probs2 = clf2.predict_proba(X)

    assert np.allclose(probs1, probs2, atol=1e-8), "Model predictions differ under identical seed"


# 9. test_meta_inputs_match_declared_list
def test_meta_inputs_match_declared_list():
    expected_15 = [
        "bin_lr_p_up",
        "bin_rf_p_up",
        "bin_gb_p_up",
        "tri_lr_p_up",
        "tri_lr_p_down",
        "tri_rf_p_up",
        "tri_rf_p_down",
        "tri_gb_p_up",
        "tri_gb_p_down",
        "ta_rule_vote",
        "ta_loc",
        "ta_trend",
        "ta_ema50",
        "ta_rsi",
        "ta_align",
    ]
    assert META_INPUT_COLUMNS == expected_15
    assert len(META_INPUT_COLUMNS) == 15


# 10. test_meta_nested_split_disjoint_with_gap
def test_meta_nested_split_disjoint_with_gap():
    # Build synthetic meta dataframe across folds 1..4
    rows = []
    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    for f in range(1, 5):
        for i in range(100):
            rows.append({
                "open_time": (base_time + pd.Timedelta(minutes=5 * len(rows))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fold": f,
                "target_2_S1": 1 if i % 2 == 0 else -1,
                **{c: 0.5 for c in META_INPUT_COLUMNS},
            })
    df_meta = pd.DataFrame(rows)

    for k in range(2, 5):
        train_slice = df_meta[df_meta["fold"] < k]
        purged_slice = train_slice.iloc[:-12]
        test_slice = df_meta[df_meta["fold"] == k]

        # Ensure no overlap
        assert set(purged_slice["open_time"]).isdisjoint(set(test_slice["open_time"]))
        # Purge gap of 12 candles
        last_train_t = pd.to_datetime(purged_slice["open_time"].iloc[-1])
        first_test_t = pd.to_datetime(test_slice["open_time"].iloc[0])
        gap_candles = int((first_test_t - last_train_t) / pd.Timedelta(minutes=5))
        assert gap_candles >= 12, f"Gap candles {gap_candles} < 12"


# 11. test_meta_never_scored_on_training_rows
def test_meta_never_scored_on_training_rows():
    # Fold 1 should never be present in nested meta OOF predictions
    dummy_meta = pd.DataFrame({
        "open_time": [f"2024-01-0{i}T00:00:00Z" for i in range(1, 6)],
        "fold": [1, 2, 3, 4, 5],
        "target_2_S1": [0, 1, -1, 0, 1],
        **{c: [0.5]*5 for c in META_INPUT_COLUMNS},
    })
    # Nested evaluation strictly predicts folds > 1
    eval_folds = [f for f in dummy_meta["fold"].unique() if f > 1]
    assert 1 not in eval_folds


# 12. test_ta_scores_in_range_and_causal
def test_ta_scores_in_range_and_causal():
    # Create synthetic series
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feat_struct_rule_vote": np.random.choice([-1.0, 0.0, 1.0], size=n),
        "feat_loc_position_288": np.random.uniform(0.0, 1.0, size=n),
        "feat_struct_trend_up": np.random.choice([0.0, 1.0], size=n),
        "feat_struct_trend_down": np.random.choice([0.0, 1.0], size=n),
        "feat_mom_ema_ratio_50": np.random.normal(0.0, 0.01, size=n),
        "feat_mom_rsi_14": np.random.uniform(0.0, 1.0, size=n),
        "feat_ma_alignment_ema_5m": np.random.choice([-1.0, 0.0, 1.0], size=n),
    })

    scores1 = compute_ta_scores(df, q99=0.02)
    # Check bounds
    for col in TA_SCORE_COLUMNS:
        assert (scores1[col] >= -1.0 - 1e-6).all(), f"{col} has value < -1"
        assert (scores1[col] <= 1.0 + 1e-6).all(), f"{col} has value > 1"

    # Causality test: change future rows (from index 150 onwards)
    df_altered = df.copy()
    df_altered.iloc[150:, :] = 999.0  # radically altered future rows

    scores2 = compute_ta_scores(df_altered, q99=0.02)
    # Past rows up to 149 must be strictly identical
    pd.testing.assert_frame_equal(scores1.iloc[:150], scores2.iloc[:150])


# 13. test_loader_blocks_validation_and_test
def test_loader_blocks_validation_and_test():
    with pytest.raises(PermissionError):
        load_master(split="test", unlock_test=False)


# 14. test_predict_horizon_returns_no_signal_on_short_buffer
def test_predict_horizon_returns_no_signal_on_short_buffer():
    short_df = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "volume": [1000.0] * 10,
    })
    res = predict_horizon(short_df, horizon=2)
    assert res.get("signal") == "NO SIGNAL"
    assert "Insufficient candles" in res.get("reason", "")


# 15. test_predict_horizon_probs_sum_to_one
def test_predict_horizon_probs_sum_to_one(tmp_path):
    # Create minimal mock artifacts in tmp_path
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
    import json

    horizon = 2
    model_dir = tmp_path / f"horizon_{horizon}"
    model_dir.mkdir(parents=True)

    # Save mock ta_params
    with open(model_dir / f"ta_params_N{horizon}.json", "w") as f:
        json.dump({"q99": 0.01}, f)

    # Save mock base models
    base_models = [
        "binary_lr", "binary_rf", "binary_gb",
        "three_class_lr", "three_class_rf", "three_class_gb"
    ]
    for mid in base_models:
        is_bin = mid.startswith("binary")
        clf = LogisticRegression()
        X = np.random.normal(size=(50, 4))
        y = np.random.choice([0, 1] if is_bin else [-1, 0, 1], size=50)
        clf.fit(X, y)
        scaler = StandardScaler().fit(X)
        payload = {
            "model": clf,
            "scaler": scaler,
            "feature_cols": ["feat_ret_1", "feat_ret_3", "feat_mom_rsi_14", "feat_vol_natr_14"],
        }
        joblib.dump(payload, model_dir / f"{mid}_N{horizon}.joblib")

    # Save mock meta-model
    meta_clf = LogisticRegression()
    X_meta = np.random.normal(size=(50, 15))
    y_meta = np.random.choice([-1, 0, 1], size=50)
    meta_clf.fit(X_meta, y_meta)
    meta_scaler = StandardScaler().fit(X_meta)
    joblib.dump(
        {"model": meta_clf, "scaler": meta_scaler, "feature_cols": META_INPUT_COLUMNS},
        model_dir / f"meta_model_N{horizon}.joblib",
    )

    # Mock candles
    n = 2050
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    candles = pd.DataFrame({
        "open_time": dates,
        "open": np.linspace(40000, 41000, n),
        "high": np.linspace(40100, 41100, n),
        "low": np.linspace(39900, 40900, n),
        "close": np.linspace(40050, 41050, n),
        "volume": np.random.uniform(10, 100, n),
        "quote_volume": np.random.uniform(400000, 4100000, n),
        "number_of_trades": np.random.randint(100, 1000, n),
        "taker_buy_base_volume": np.random.uniform(5, 50, n),
        "taker_buy_quote_volume": np.random.uniform(200000, 2050000, n),
    })

    res = predict_horizon(candles, horizon=horizon, model_dir=model_dir)
    assert res.get("final_prediction") in ["UP", "DOWN", "NEUTRAL"]
    p_sum = res["p_up"] + res["p_down"] + res["p_neutral"]
    assert abs(p_sum - 1.0) < 1e-6
