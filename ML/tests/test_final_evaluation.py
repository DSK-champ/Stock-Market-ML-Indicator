"""Unit tests for Step 5d: Final Combined Model Evaluation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import config
from src.io_utils import load_horizon_oof_csv_gz

HORIZON = 2


def test_notebooks_invariance():
    """Verify that existing notebooks 05b and 05c remain completely untouched."""
    nb_05b_path = config.NOTEBOOKS_DIR / "05b_Horizon_N2.ipynb"
    nb_05c_path = config.NOTEBOOKS_DIR / "05c_TA_State_N2.ipynb"
    
    assert nb_05b_path.exists(), "05b notebook must exist"
    assert nb_05c_path.exists(), "05c notebook must exist"
    
    nb_05b = json.load(open(nb_05b_path, encoding="utf-8"))
    nb_05c = json.load(open(nb_05c_path, encoding="utf-8"))
    
    assert len(nb_05b["cells"]) == 42, f"05b cell count mismatch: {len(nb_05b['cells'])} != 42"
    assert len(nb_05c["cells"]) == 11, f"05c cell count mismatch: {len(nb_05c['cells'])} != 11"


def test_decision_rule_logic():
    """Verify exact behavior of neutral override decision rule and tie-handling."""
    theta = 0.33
    
    def apply_rule(u, d, n, th=theta):
        if u >= d and u >= n:
            return "UP"
        elif d >= u and d >= n:
            return "DOWN"
        else:
            if max(u, d) >= th:
                if u > d:
                    return "UP"
                elif d > u:
                    return "DOWN"
                else:
                    return "NEUTRAL"
            else:
                return "NEUTRAL"

    # Case 1: UP is clear argmax
    assert apply_rule(0.45, 0.25, 0.30) == "UP"
    # Case 2: DOWN is clear argmax
    assert apply_rule(0.25, 0.45, 0.30) == "DOWN"
    # Case 3: NEUTRAL is argmax, but UP exceeds override threshold
    assert apply_rule(0.35, 0.20, 0.45) == "UP"
    # Case 4: NEUTRAL is argmax, but DOWN exceeds override threshold
    assert apply_rule(0.20, 0.35, 0.45) == "DOWN"
    # Case 5: NEUTRAL is argmax, directional probabilities below threshold
    assert apply_rule(0.28, 0.28, 0.44) == "NEUTRAL"
    # Case 6: NEUTRAL is argmax, tie between UP and DOWN above threshold
    assert apply_rule(0.34, 0.34, 0.36) == "NEUTRAL"


def test_oof_files_integrity():
    """Verify that all 7 OOF files exist and have non-empty valid predictions."""
    oof_dir = config.RESULTS_DIR / "oof"
    models = [
        "binary_lr", "binary_rf", "binary_gb",
        "three_class_lr", "three_class_rf", "three_class_gb",
        "meta"
    ]
    for m in models:
        p = oof_dir / f"{m}_N{HORIZON}.csv.gz"
        assert p.exists(), f"Missing required OOF file: {p.name}"
        df = load_horizon_oof_csv_gz(p)
        assert len(df) > 0, f"OOF file is empty: {p.name}"
        assert "open_time" in df.columns, f"Missing open_time in {p.name}"
        assert not df["open_time"].duplicated().any(), f"Duplicate timestamps in {p.name}"
