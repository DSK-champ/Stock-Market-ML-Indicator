"""Unit tests for Step 4 Feature Engineering EDA functions, reports, figures, and notebook integrity."""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd
import nbformat

from src import config
from src.io_utils import load_clean_market_data_csv
from src.splitting import get_val_data, get_test_data
import src.feat_eda as feat_eda

REPORTS_DIR = config.REPORTS_DIR
FIGURES_DIR = config.FIGURES_DIR
NOTEBOOK_04_PATH = config.NOTEBOOKS_DIR / "04_Feature_Engineering.ipynb"


def test_feat_eda_raises_on_validation_or_test_leakage():
    """feat_eda functions must raise ValueError if passed validation or test rows."""
    df_clean = load_clean_market_data_csv()
    val_df = get_val_data(df_clean)
    test_df = get_test_data(df_clean, unlock=True)

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        feat_eda.verify_train_only(df_clean)

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        feat_eda.verify_train_only(val_df)

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        feat_eda.verify_train_only(test_df)


def test_feat_reports_exist_and_non_empty():
    """All expected Step 4 feature reports exist and have non-zero size."""
    expected_reports = [
        "feat_inventory.csv",
        "feat_variable_statistics.csv",
        "feat_corr_pearson.csv",
        "feat_corr_spearman.csv",
        "feat_top20_correlated_pairs.csv",
        "feat_pairs_above_90.csv",
        "feat_vif_all.csv",
        "feat_auc_big_move.csv",
        "feat_cohens_d_big_move.csv",
        "feat_group_auc_size.csv",
        "feat_direction_metrics.csv",
        "feat_lagged_flow_loc.csv",
        "feat_noise_band.csv",
        "feat_direction_quarterly_stability.csv",
        "feat_context_hour_dow_vol.csv",
        "feat_quarterly_drift.csv",
        "feat_correlation_clusters.csv",
        "feat_breakeven_analysis.csv",
    ]

    for rep in expected_reports:
        p = REPORTS_DIR / rep
        assert p.exists(), f"Missing report: {p}"
        assert p.stat().st_size > 50, f"Report {p} is too small / empty ({p.stat().st_size} bytes)"


def test_feat_figures_exist_and_non_empty():
    """Key Step 4 figures exist on disk and have non-zero size."""
    expected_figs = [
        "04_corr_clustered.png",
        "04_auc_big_move_bar.png",
        "04_cohens_d_big_move_heatmap.png",
        "04_direction_metrics_heatmap.png",
        "04_lagged_flow_loc.png",
        "04_rolling_corr_direction_5.png",
        "04_direction_quarterly_stability.png",
        "04_context_profiles.png",
        "04_acf_deseasoning.png",
    ]

    for fig in expected_figs:
        p = FIGURES_DIR / fig
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 5000, f"Figure {p} is too small ({p.stat().st_size} bytes)"


def test_feature_correlation_matrices_symmetric_and_unit_diagonal():
    """Pearson and Spearman feature correlation matrices must be symmetric with unit diagonal."""
    for fn in ["feat_corr_pearson.csv", "feat_corr_spearman.csv"]:
        p = REPORTS_DIR / fn
        assert p.exists(), f"Missing {fn}"
        corr = pd.read_csv(p, index_col=0)

        # Unit diagonal
        diag = np.diag(corr.to_numpy())
        np.testing.assert_allclose(diag, 1.0, atol=1e-5, err_msg=f"{fn} diagonal is not 1.0")

        # Symmetry
        diff = np.abs(corr.to_numpy() - corr.to_numpy().T)
        np.testing.assert_allclose(diff, 0.0, atol=1e-5, err_msg=f"{fn} is not symmetric")


def test_notebook_04_execution_and_cleanliness():
    """Verify executed 04_Feature_Engineering.ipynb has stored outputs and clean cells."""
    assert NOTEBOOK_04_PATH.exists(), f"{NOTEBOOK_04_PATH} does not exist"
    with open(NOTEBOOK_04_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) > 0, "No code cells in notebook 04"

    for idx, cell in enumerate(code_cells):
        src = cell.source
        # 1. Stored outputs
        assert len(cell.outputs) > 0, f"Code cell {idx+1} has no stored output!"

        # 2. No importlib.reload hacks
        assert "importlib.reload" not in src, f"Code cell {idx+1} contains importlib.reload hack!"

        # 3. No bare except: pass
        assert "except:" not in src or "pass" not in src, f"Code cell {idx+1} contains bare except!"

        # 4. No validation or test data loads
        if "load_master" in src:
            assert 'split="val"' not in src and "split='val'" not in src, f"Cell {idx+1} accesses validation partition!"
            assert 'split="test"' not in src and "split='test'" not in src, f"Cell {idx+1} accesses test partition!"
            assert 'split="all"' not in src and "split='all'" not in src, f"Cell {idx+1} accesses full dataset!"

        # 5. Check outputs for uncaught error tracebacks
        for out in cell.outputs:
            assert out.get("output_type") != "error", f"Cell {idx+1} produced error: {out.get('evalue')}"


def test_circular_shift_preserves_label_autocorrelation():
    """Circular shift of labels preserves lag-1 autocorrelation within tolerance."""
    from src.column_dictionary import load_master
    df_train = load_master(split="train")
    y_raw = df_train.iloc[config.LOOKBACK_CANDLES:]["raw_sign_12"].dropna().to_numpy()
    y_bin = (y_raw == 1.0).astype(float)
    
    # Lag-1 autocorrelation of original series
    y_centered = y_bin - np.mean(y_bin)
    var = np.dot(y_centered, y_centered)
    ac_orig = np.dot(y_centered[:-1], y_centered[1:]) / var
    
    # Circularly roll by 2500 candles
    y_rolled = np.roll(y_bin, 2500)
    y_roll_centered = y_rolled - np.mean(y_rolled)
    var_roll = np.dot(y_roll_centered, y_roll_centered)
    ac_roll = np.dot(y_roll_centered[:-1], y_roll_centered[1:]) / var_roll
    
    # In a circular shift of length N, only the boundary pair changes, so difference is O(1/N)
    assert np.isclose(ac_orig, ac_roll, atol=1e-4), f"Autocorrelation altered by circular shift: orig={ac_orig:.6f}, roll={ac_roll:.6f}"


def test_circular_noise_99th_max_not_below_iid_band():
    """99th percentile of maximum circular noise AUC must not be below iid shuffle band."""
    df_noise = pd.read_csv(REPORTS_DIR / "feat_noise_band.csv")
    for _, row in df_noise.iterrows():
        circ_max_99 = row["circular_shift_max_99th"]
        iid_95 = row["iid_shuffle_95th_ignores_dependence"]
        assert circ_max_99 >= iid_95, (
            f"Circular 99th max ({circ_max_99}) below iid 95th ({iid_95}) for side={row['side']}, N={row['horizon_N']}"
        )


def test_quarterly_direction_stability_has_9_quarters():
    """Quarterly direction stability table contains exactly 9 quarters (2023Q4 to 2025Q4)."""
    df_q = pd.read_csv(REPORTS_DIR / "feat_direction_quarterly_stability.csv")
    q_cols = [c for c in df_q.columns if c.startswith("auc_20")]
    assert len(q_cols) == 9, f"Expected 9 quarters, found {len(q_cols)}: {q_cols}"
    expected_quarters = ["auc_2023Q4", "auc_2024Q1", "auc_2024Q2", "auc_2024Q3", "auc_2024Q4",
                         "auc_2025Q1", "auc_2025Q2", "auc_2025Q3", "auc_2025Q4"]
    assert q_cols == expected_quarters, f"Quarter columns mismatch: {q_cols}"


def test_notebook_figures_rendered_and_exist_on_disk():
    """All figures saved in notebook 04 exist on disk in results/figures/ and are non-empty."""
    with open(NOTEBOOK_04_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    import re
    fig_names = set()
    for cell in nb.cells:
        if cell.cell_type == "code":
            matches = re.findall(r'04_[a-zA-Z0-9_]+\.png', cell.source)
            fig_names.update(matches)

    assert len(fig_names) >= 9, f"Too few figures referenced in notebook: {fig_names}"
    for fn in fig_names:
        p = FIGURES_DIR / fn
        assert p.exists(), f"Notebook references figure {fn} which is missing from disk: {p}"
        assert p.stat().st_size > 5000, f"Figure {fn} is too small ({p.stat().st_size} bytes)"


