"""Update step_master.json and generate step04b_drawings.json for Step 4b."""

import sys
from pathlib import Path
import json
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.io_utils import (
    load_master_csv,
    load_drawings_csv,
    MASTER_CSV_PATH,
    DRAWINGS_CSV_PATH,
)
from src.column_dictionary import get_feature_columns, get_column_role, COLUMN_ROLES
from src.details import write_details, describe_dataframe

def update_details():
    print("Loading master.csv (split='all', unlock_test=True)...")
    master = load_master_csv(split="all", unlock_test=True)
    print(f"Master shape: {master.shape}")
    feature_cols = get_feature_columns()
    
    # 1. Update step_master.json
    d_master = describe_dataframe(
        master,
        MASTER_CSV_PATH,
        step=4,
        module="src.master",
        description="Unified master dataset combining clean market data, 136 engineered causal features (including 36 ma_structure and 15 structure), base targets, labels, dynamic thetas, and big move flags.",
        params={"n_features": len(feature_cols), "n_rules": len(config.ALL_RULES), "n_horizons": len(config.CANDIDATE_HORIZONS)},
        checks={"columns_with_valid_roles": len(master.columns), "total_rows": len(master)},
        full_df_stats=False,
    )
    write_details("step_master", d_master)
    print("Updated step_master.json")

    # 2. Generate step04b_drawings.json
    print("Loading drawings.csv (split='all', unlock_test=True)...")
    drawings = load_drawings_csv(split="all", unlock_test=True)
    print(f"Drawings shape: {drawings.shape}")
    
    d_drawings = describe_dataframe(
        drawings,
        DRAWINGS_CSV_PATH,
        step=4,
        module="src.structure",
        description="Companion market structure drawing outputs containing raw support/resistance price levels, candidate target and stop levels, risk-to-reward ratios, and causal rule trigger codes (role 'drawing').",
        params={
            "k_5m": 5,
            "k_1h": 3,
            "tau_merge_atr": 0.5,
            "beta_break_atr": 0.2,
            "tau_retest_atr": 0.5,
            "max_levels": 50,
            "min_target_stop_atr": 1.5,
        },
        checks={"columns_with_valid_roles": len(drawings.columns), "total_rows": len(drawings)},
        full_df_stats=False,
    )
    write_details("step04b_drawings", d_drawings)
    print("Generated step04b_drawings.json")

    # 3. Update index.json
    idx_path = config.ML_DIR / "details" / "index.json"
    with open(idx_path, "r", encoding="utf-8") as f:
        idx_data = json.load(f)

    # Ensure step04b_drawings is registered
    file_names = [f["name"] for f in idx_data["files"]]
    if "step04b_drawings" not in file_names:
        idx_data["files"].append({
            "name": "step04b_drawings",
            "file": "step04b_drawings.json",
            "step": 4,
            "description": "Companion market structure drawing dataset (role 'drawing')."
        })
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx_data, f, indent=2)
    print("Updated details/index.json")

if __name__ == "__main__":
    update_details()

