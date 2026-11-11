"""Unit tests for Step 3 EDA functions, integrity, anti-leakage guards, and file artifacts."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src import config
from src.io_utils import (
    load_clean_market_data_csv,
    load_targets_base_csv,
    load_targets_labels_csv,
)
from src.splitting import get_train_data, get_val_data, get_test_data
import src.eda as eda

REPORTS_DIR = config.REPORTS_DIR
FIGURES_DIR = config.FIGURES_DIR


def test_eda_raises_on_validation_or_test_leakage():
    """EDA functions must raise ValueError if passed validation or test rows."""
    df_clean = load_clean_market_data_csv()
    val_df = get_val_data(df_clean)
    test_df = get_test_data(df_clean, unlock=True)

    # 1. Full dataset check
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        eda.verify_train_only(df_clean)

    # 2. Validation partition check
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        eda.verify_train_only(val_df)

    # 3. Test partition check
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        eda.verify_train_only(test_df)

    # 4. Overview function guard
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        eda.compute_train_overview(val_df)


def test_eda_reports_exist_and_non_empty():
    """All 17 expected EDA CSV report tables exist and have non-zero size."""
    expected_reports = [
        "eda_train_overview.csv",
        "eda_column_health.csv",
        "eda_variable_statistics.csv",
        "eda_quarterly_drift_returns.csv",
        "eda_quarterly_drift_volume.csv",
        "eda_monthly_regimes.csv",
        "eda_volatility_terciles.csv",
        "eda_corr_raw.csv",
        "eda_corr_transforms_pearson.csv",
        "eda_corr_transforms_spearman.csv",
        "eda_corr_transforms_kendall.csv",
        "eda_top_15_correlated_pairs.csv",
        "eda_vif_transforms.csv",
        "eda_corr_transforms_vs_targets_pearson.csv",
        "eda_corr_transforms_vs_targets_spearman.csv",
        "eda_lagged_correlations_fwd_ret_12.csv",
        "eda_target_label_balance.csv",
        "eda_label_persistence.csv",
        "eda_class_conditional_cohens_d.csv",
        "eda_baseline_accuracies.csv",
        "eda_validation_drift_comparison.csv",
        "eda_top_20_volume_candles.csv",
        "eda_top_20_trades_candles.csv",
        "eda_anomalies_kurtosis_impact.csv",
    ]

    for rep in expected_reports:
        p = REPORTS_DIR / rep
        assert p.exists(), f"Missing report: {p}"
        assert p.stat().st_size > 50, f"Report {p} is too small / empty ({p.stat().st_size} bytes)"


def test_eda_figures_exist_and_non_empty():
    """All 17 expected 03_ EDA figures exist on disk and have non-zero size."""
    expected_figs = [
        "03_train_return_distributions.png",
        "03_train_tail_zoom.png",
        "03_forward_return_horizons.png",
        "03_intraday_distributions.png",
        "03_ecdf_forward_returns.png",
        "03_train_price_moving_averages.png",
        "03_rolling_macro_trends.png",
        "03_acf_pacf_multi.png",
        "03_seasonality_heatmaps.png",
        "03_corr_raw.png",
        "03_corr_transforms_pearson.png",
        "03_corr_transforms_spearman.png",
        "03_corr_transforms_vs_targets_pearson.png",
        "03_corr_lagged_fwd_ret_12.png",
        "03_rolling_feature_correlations.png",
        "03_label_balance_horizons.png",
        "03_threshold_sweep.png",
    ]

    for fig in expected_figs:
        p = FIGURES_DIR / fig
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 5000, f"Figure {p} is too small ({p.stat().st_size} bytes)"


def test_label_balance_shares_sum_to_one():
    """In the target balance table, for every valid row: pct_up + pct_down + pct_neutral == 100%."""
    balance_path = REPORTS_DIR / "eda_target_label_balance.csv"
    assert balance_path.exists()
    df = pd.read_csv(balance_path)

    shares_sum = df["pct_up"] + df["pct_down"] + df["pct_neutral"]
    np.testing.assert_allclose(
        shares_sum.to_numpy(),
        100.0,
        atol=0.05,
        err_msg="Label class shares do not sum to 100% in eda_target_label_balance.csv",
    )


def test_correlation_matrices_symmetric_and_unit_diagonal():
    """Check that stationary transform correlation matrices are symmetric with diagonal == 1.0."""
    for fn in ["eda_corr_transforms_pearson.csv", "eda_corr_transforms_spearman.csv"]:
        p = REPORTS_DIR / fn
        corr = pd.read_csv(p, index_col=0)

        # Unit diagonal
        diag = np.diag(corr.to_numpy())
        np.testing.assert_allclose(diag, 1.0, atol=1e-5, err_msg=f"{fn} diagonal is not 1.0")

        # Symmetry
        diff = np.abs(corr.to_numpy() - corr.to_numpy().T)
        np.testing.assert_allclose(diff, 0.0, atol=1e-5, err_msg=f"{fn} is not symmetric")

