"""Unit tests for benchmark baselines.

Enforces:
  1. RSI(14) coverage is strictly below 50% on real train rows (abstentions stay abstentions).
  2. EMA50 rule emits both signs (+1 and -1), with each sign between 30% and 70%.
  3. No baseline other than always_up and majority_class (when mode is +1) is identical to another.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.column_dictionary import load_master
from src.cv import PurgedWalkForwardCV
from src.baselines import (
    BASELINES_LIST,
    run_baseline,
    rule_rsi_14,
    rule_ema50,
)


@pytest.fixture(scope="module")
def train_df():
    """Load train split of master dataset."""
    df_train = load_master(split="train")
    return df_train


def test_rsi_14_coverage_below_50(train_df):
    """Verify that RSI(14) rule coverage is strictly below 50% on real train rows."""
    cv = PurgedWalkForwardCV()
    fold, tr_idx, te_idx = next(cv.split(train_df))
    df_te = train_df.iloc[te_idx]

    preds, _ = rule_rsi_14(df_te)
    coverage = float(np.mean(preds != 0.0))

    # Coverage should be ~6.66%, well below 50%
    assert coverage < 0.50, f"RSI(14) coverage {coverage:.4f} is not below 50%"
    assert coverage > 0.01, f"RSI(14) coverage {coverage:.4f} is unreasonably low (< 1%)"

    # Verify abstentions exist
    n_abstain = int(np.sum(preds == 0.0))
    assert n_abstain > len(df_te) * 0.50, f"Expected >50% abstentions, got {n_abstain}/{len(df_te)}"


def test_ema50_both_signs_between_30_and_70(train_df):
    """Verify that EMA50 rule emits both signs with each between 30% and 70%."""
    cv = PurgedWalkForwardCV()
    fold, tr_idx, te_idx = next(cv.split(train_df))
    df_te = train_df.iloc[te_idx]

    preds, _ = rule_ema50(df_te)
    pos_pct = float(np.mean(preds == 1.0))
    neg_pct = float(np.mean(preds == -1.0))

    assert 0.30 <= pos_pct <= 0.70, f"EMA50 positive rate {pos_pct:.4f} outside [30%, 70%]"
    assert 0.30 <= neg_pct <= 0.70, f"EMA50 negative rate {neg_pct:.4f} outside [30%, 70%]"
    assert np.isclose(pos_pct + neg_pct, 1.0, atol=1e-3), "EMA50 should have ~100% active coverage"


def test_no_baseline_identical_except_allowed(train_df):
    """Verify no baseline other than always_up and majority_class is identical to another."""
    cv = PurgedWalkForwardCV()
    fold, tr_idx, te_idx = next(cv.split(train_df))
    df_tr = train_df.iloc[tr_idx]
    df_te = train_df.iloc[te_idx]

    preds_by_name = {}
    for b_name, _, _ in BASELINES_LIST:
        p, _ = run_baseline(b_name, df_tr, df_te, target_col="raw_sign_12")
        preds_by_name[b_name] = p

    # Allowed identity pair: ("always_up", "majority_class") when mode is +1
    allowed_pairs = {("always_up", "majority_class"), ("majority_class", "always_up")}

    names = list(preds_by_name.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1 = names[i]
            n2 = names[j]
            p1 = preds_by_name[n1]
            p2 = preds_by_name[n2]

            identical = np.array_equal(p1, p2)
            if identical:
                assert (n1, n2) in allowed_pairs or (n2, n1) in allowed_pairs, (
                    f"Baseline '{n1}' and '{n2}' produced identical predictions on fold 1! "
                    f"Identical baselines are forbidden."
                )

