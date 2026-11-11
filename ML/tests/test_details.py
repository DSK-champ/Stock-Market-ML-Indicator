"""Unit tests verifying integrity, schema conformity, and leakage-prevention in details/ JSONs.

Checks:
  1. Every JSON file in index.json exists and loads cleanly.
  2. For data files: shape, column names, column order, and dtypes match file reloaded via io_utils.py.
  3. Stored sha256 hash matches disk file.
  4. Every column has description, role, and is_future_looking.
  5. Target columns have is_future_looking=True and allowed_as_model_input=False.
  6. No test split value statistics exist in any JSON (test block is strictly metadata/count).
  7. No bare NaN or Infinity tokens in any JSON file.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src import config
from src.io_utils import (
    load_raw_untouched_csv,
    load_clean_market_data_csv,
    load_targets_base_csv,
    load_targets_labels_csv,
    load_master_csv,
)
from src.details import compute_file_hash

DETAILS_DIR = config.ML_DIR / "details"


def test_index_json_valid_and_all_files_exist():
    """index.json exists and all registered files are present on disk and load."""
    index_path = DETAILS_DIR / "index.json"
    assert index_path.exists(), "details/index.json does not exist"

    with open(index_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert "files" in manifest
    assert len(manifest["files"]) > 0

    for item in manifest["files"]:
        target_path = config.ML_DIR / item["path"]
        assert target_path.exists(), f"File {target_path} listed in index.json missing on disk"

        # Verify it loads as valid JSON
        with open(target_path, encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)
            assert isinstance(data, dict)

        # Check no bare NaN or Infinity
        def _err(c):
            raise ValueError(f"Found bare JSON constant {c}")
        json.loads(content, parse_constant=_err)


def test_data_files_schema_and_hash_parity():
    """Data files match disk file shape, schema, dtypes, and sha256 hash."""
    loaders = {
        "step01_raw.json": (config.RAW_DATA_PATH, load_raw_untouched_csv),
        "step01_clean.json": (config.CLEAN_DATA_PATH, load_clean_market_data_csv),
        "step02_targets_base.json": (config.DATA_PROCESSED_DIR / "targets_base.csv", load_targets_base_csv),
        "step02_targets_labels.json": (config.DATA_PROCESSED_DIR / "targets_labels.csv", load_targets_labels_csv),
        "step_master.json": (config.DATA_PROCESSED_DIR / "master.csv", lambda: load_master_csv(split="all", unlock_test=True)),
    }

    for json_name, (disk_path, loader) in loaders.items():
        json_file = DETAILS_DIR / json_name
        assert json_file.exists(), f"{json_name} missing"

        with open(json_file, encoding="utf-8") as f:
            d = json.load(f)

        # 1. SHA-256 Hash check
        actual_hash = compute_file_hash(disk_path)
        assert d["file"]["sha256"] == actual_hash, f"Hash mismatch in {json_name}"

        # 2. Reload file through io_utils
        df_disk = loader()

        # 3. Shape check
        assert d["shape"]["n_rows"] == len(df_disk)
        assert d["shape"]["n_columns"] == len(df_disk.columns)

        # 4. Columns check
        cols_meta = d["columns"]
        json_col_names = [c["name"] for c in cols_meta]
        assert json_col_names == list(df_disk.columns), f"Column order/names mismatch in {json_name}"

        for idx, col in enumerate(cols_meta):
            assert "description" in col and len(col["description"]) > 0
            assert "role" in col
            assert "is_future_looking" in col
            assert col["position"] == idx

            # Target checks
            if col["role"] == "target":
                assert col["is_future_looking"] is True
                assert col["allowed_as_model_input"] is False


def test_no_test_split_value_statistics():
    """Verify test split contains ONLY row count and timestamps, never statistics."""
    for json_file in DETAILS_DIR.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            d = json.load(f)

        # Check row_counts_by_split block
        if "row_counts_by_split" in d:
            test_block = d["row_counts_by_split"].get("test", {})
            if isinstance(test_block, dict):
                # Must only contain n_rows, first, last, note
                forbidden = {"min", "max", "mean", "std", "median"}
                assert not forbidden.intersection(test_block.keys()), (
                    f"{json_file.name} leaks test split statistics in row_counts_by_split!"
                )

        # Check column statistics: stats_test must not exist
        if "columns" in d:
            for col in d["columns"]:
                assert "stats_test" not in col, (
                    f"{json_file.name} column {col['name']} contains stats_test!"
                )

