"""Centralized I/O utilities for CSV market data storage and retrieval.

Enforces strict single-entry reading and writing:
  - Timezone-aware ISO 8601 UTC open_time
  - Full floating-point precision (repr-level)
  - Explicit dtype casting (float64 for market measurements)
  - Zero ad-hoc pd.read_csv calls across the project

Target files
------------
  data/processed/targets_base.csv
      fwd_ret_N, fwd_ret_open_N, raw_sign_N, has_valid_target_N  for N ∈ {2,3,6,12}
  data/processed/targets_labels.csv
      target_N_rule, theta_up_N_rule, theta_down_N_rule           for all N × rule
  data/processed/targets_theta_frozen.json
      Frozen Q1 thresholds + sigma window + cost + train row range

Targets are kept SEPARATE from clean.csv and from features.csv.
They are joined on open_time only during model training.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logger

logger = setup_logger("io_utils")


def save_clean_market_data_csv(df: pd.DataFrame, filepath: Union[str, Path] = config.DATA_PROCESSED_DIR / "clean.csv") -> Path:
    """Save cleaned time-indexed market data to CSV with ISO UTC timestamps and full precision."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()

    # Ensure index is open_time in UTC
    if isinstance(df_to_save.index, pd.DatetimeIndex):
        if df_to_save.index.tz is None:
            df_to_save.index = df_to_save.index.tz_localize("UTC")
        else:
            df_to_save.index = df_to_save.index.tz_convert("UTC")
        df_to_save.index.name = "open_time"
        # Export with ISO 8601 formatting and full IEEE 754 repr-level float precision
        df_to_save.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        if "open_time" not in df_to_save.columns:
            raise KeyError("DataFrame must possess an 'open_time' index or column.")
        df_to_save["open_time"] = pd.to_datetime(df_to_save["open_time"], utc=True)
        df_to_save.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S%z")

    logger.info("Saved clean market data CSV to %s (%d rows)", path, len(df_to_save))
    return path


def load_clean_market_data_csv(filepath: Union[str, Path] = config.DATA_PROCESSED_DIR / "clean.csv") -> pd.DataFrame:
    """Load clean market data CSV, establishing UTC DatetimeIndex and strict float64 dtypes."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clean CSV file not found: {path}")

    # Explicit dtype specifications
    dtype_dict = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_volume": "float64",
        "number_of_trades": "float64",
        "taker_buy_base_volume": "float64",
        "taker_buy_quote_volume": "float64",
    }
    if "gap_id" in pd.read_csv(path, nrows=1).columns:
        dtype_dict["gap_id"] = "int32"
    if "is_gap_boundary" in pd.read_csv(path, nrows=1).columns:
        dtype_dict["is_gap_boundary"] = "bool"

    df = pd.read_csv(
        path,
        parse_dates=["open_time"],
        index_col="open_time",
        dtype=dtype_dict,
    )

    # Ensure DatetimeIndex is explicitly UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


def save_raw_untouched_csv(df_raw: pd.DataFrame, filepath: Union[str, Path] = config.DATA_RAW_DIR / "BTCUSDT_5m_raw.csv") -> Path:
    """Save untouched raw Binance klines to CSV preserving all 12 raw columns exactly."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Raw data exported as-is without index modification at full repr-level precision
    df_raw.to_csv(path, index=False)
    logger.info("Saved untouched raw CSV to %s (%d rows)", path, len(df_raw))
    return path


def load_raw_untouched_csv(filepath: Union[str, Path] = config.DATA_RAW_DIR / "BTCUSDT_5m_raw.csv") -> pd.DataFrame:
    """Load untouched raw Binance CSV maintaining original column schemas."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV file not found: {path}")

    df_raw = pd.read_csv(path, dtype={"open_time": "int64", "close_time": "int64"})
    return df_raw


# ===========================================================================
# Target files
# ===========================================================================

# Canonical paths
TARGETS_BASE_PATH   = config.DATA_PROCESSED_DIR / "targets_base.csv"
TARGETS_LABELS_PATH = config.DATA_PROCESSED_DIR / "targets_labels.csv"
TARGETS_THETA_PATH  = config.DATA_PROCESSED_DIR / "targets_theta_frozen.json"
FEATURES_PATH       = config.DATA_PROCESSED_DIR / "features.csv"
MASTER_CSV_PATH     = config.DATA_PROCESSED_DIR / "master.csv"
DRAWINGS_CSV_PATH   = config.DATA_PROCESSED_DIR / "drawings.csv"


def _target_base_dtypes(horizons=None) -> dict:
    if horizons is None:
        horizons = config.CANDIDATE_HORIZONS
    dtypes = {}
    for N in horizons:
        dtypes[f"fwd_ret_{N}"]       = "float64"
        dtypes[f"fwd_ret_open_{N}"]  = "float64"
        dtypes[f"raw_sign_{N}"]      = "float64"
        dtypes[f"has_valid_target_{N}"] = "int8"   # 0/1; avoids CSV bool parsing quirks
    return dtypes


def _target_label_dtypes(horizons=None, rules=None) -> dict:
    if horizons is None:
        horizons = config.CANDIDATE_HORIZONS
    if rules is None:
        rules = config.ALL_RULES
    dtypes = {}
    for N in horizons:
        for rule in rules:
            dtypes[f"target_{N}_{rule}"]    = "float64"
            dtypes[f"theta_up_{N}_{rule}"]  = "float64"
            dtypes[f"theta_down_{N}_{rule}"] = "float64"
        dtypes[f"big_move_{N}"] = "float64"
    return dtypes


def save_targets_base_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path] = TARGETS_BASE_PATH,
) -> Path:
    """Save targets_base DataFrame (fwd_ret, fwd_ret_open, raw_sign, has_valid_target) to CSV.

    has_valid_target columns are saved as int8 (0/1) so CSV boolean parsing is unambiguous.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    # Convert bool has_valid_target columns to int8
    for col in out.columns:
        if col.startswith("has_valid_target_") and out[col].dtype == bool:
            out[col] = out[col].astype("int8")

    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out.index.name = "open_time"
        out.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        out.to_csv(path, index=True)

    logger.info("Saved targets_base CSV to %s (%d rows, %d cols)", path, len(out), len(out.columns))
    return path


def load_targets_base_csv(
    filepath: Union[str, Path] = TARGETS_BASE_PATH,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Load targets_base CSV. Returns DataFrame with UTC DatetimeIndex."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"targets_base.csv not found: {path}")

    dtypes = _target_base_dtypes(horizons)
    df = pd.read_csv(path, parse_dates=["open_time"], index_col="open_time", dtype=dtypes)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def save_targets_labels_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path] = TARGETS_LABELS_PATH,
) -> Path:
    """Save targets_labels DataFrame (target, theta_up, theta_down per horizon × rule) to CSV."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out.index.name = "open_time"
        out.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        out.to_csv(path, index=True)

    logger.info("Saved targets_labels CSV to %s (%d rows, %d cols)", path, len(out), len(out.columns))
    return path


def load_targets_labels_csv(
    filepath: Union[str, Path] = TARGETS_LABELS_PATH,
    horizons: Optional[List[int]] = None,
    rules: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load targets_labels CSV. Returns DataFrame with UTC DatetimeIndex."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"targets_labels.csv not found: {path}")

    dtypes = _target_label_dtypes(horizons, rules)
    df = pd.read_csv(path, parse_dates=["open_time"], index_col="open_time", dtype=dtypes)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def save_theta_frozen_json(
    q1_thresholds: Dict[str, float],
    train_df: pd.DataFrame,
    sigma_window: int = 288,
    filepath: Union[str, Path] = TARGETS_THETA_PATH,
) -> Path:
    """Save frozen Q1 thresholds + provenance metadata to JSON.

    Parameters
    ----------
    q1_thresholds : dict from compute_q1_thresholds (keys: q1_up_N, q1_down_N)
    train_df      : the training DataFrame used to compute Q1 (for provenance)
    sigma_window  : rolling-sigma window used for V rules
    """
    import math
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    record = {
        "description": (
            "Frozen Q1 thresholds computed once from TRAIN rows only. "
            "Applied unchanged to val/test. Must not be recomputed after Step 2."
        ),
        "q1_thresholds": {k: _clean(v) for k, v in q1_thresholds.items()},
        "constant_thetas": {
            "S1":  {"theta_up": 0.0023, "theta_down": 0.0023, "sub_cost": False},
            "S2":  {"theta_up": 0.0030, "theta_down": 0.0030, "sub_cost": False},
            "A1":  {"theta_up": 0.0023, "theta_down": 0.0030, "sub_cost": False},
            "A2":  {"theta_up": 0.0030, "theta_down": 0.0023, "sub_cost": False},
            "F05": {"theta_up": 0.0005, "theta_down": 0.0005, "sub_cost": True},
            "F10": {"theta_up": 0.0010, "theta_down": 0.0010, "sub_cost": True},
            "F15": {"theta_up": 0.0015, "theta_down": 0.0015, "sub_cost": True},
        },
        "sub_cost_rules_note": (
            "F05 (0.05%), F10 (0.10%), F15 (0.15%) are research sub-cost comparison rules. "
            "Move size is strictly below cost floor c=0.23%; any signal on a move below c loses money after costs."
        ),
        "sigma_window_W": sigma_window,
        "cost_c": config.TOTAL_ROUNDTRIP_COST,
        "candidate_horizons": config.CANDIDATE_HORIZONS,
        "train_provenance": {
            "n_rows": len(train_df),
            "first":  train_df.index[0].isoformat() if len(train_df) > 0 else None,
            "last":   train_df.index[-1].isoformat() if len(train_df) > 0 else None,
        },
        "floor_binding_note": (
            "For all N in {2,3,6,12}, the 50th percentile of train positive/negative "
            "forward returns is below c=0.23%. The floor always binds: Q1 ≡ S1 on this dataset. "
            "Q1 is retained in the experiment budget but counted as ONE configuration with S1 "
            "(see docs/decisions.md §3)."
        ),
    }

    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved frozen theta JSON to %s", path)
    return path


def load_theta_frozen_json(
    filepath: Union[str, Path] = TARGETS_THETA_PATH,
) -> dict:
    """Load frozen Q1 thresholds JSON."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"targets_theta_frozen.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Master dataset (clean + features + targets)
# ===========================================================================

def save_master_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path] = MASTER_CSV_PATH,
) -> Path:
    """Save master DataFrame (cleaned market + features + targets) to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Complete master dataset with open_time index in UTC.
    filepath : Union[str, Path]
        Target CSV file path.

    Returns
    -------
    Path
        Absolute path to written master.csv.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out.index.name = "open_time"
        out.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        out.to_csv(path, index=True)

    logger.info("Saved master CSV to %s (%d rows, %d cols)", path, len(out), len(out.columns))
    return path


def load_master_csv(
    filepath: Union[str, Path] = MASTER_CSV_PATH,
    split: str = "train",
    unlock_test: bool = False,
) -> pd.DataFrame:
    """Load master dataset with strict split partition enforcement.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to master.csv.
    split : str
        Partition to load: 'train' (default), 'val', 'train_val', 'test', or 'all'.
    unlock_test : bool
        Safety lock. Must be True to load 'test' or 'all'.

    Returns
    -------
    pd.DataFrame
        Selected partition with UTC DatetimeIndex.

    Raises
    ------
    PermissionError
        If 'test' or 'all' is requested while unlock_test is False.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"master.csv not found: {path}")

    # Safety gate
    if split in ("test", "all") and not unlock_test:
        raise PermissionError(
            f"Test set is LOCKED. Cannot load split='{split}' with unlock_test=False. "
            "Pass unlock_test=True only during final test backtesting (Step 7)."
        )

    # Read CSV with open_time as index
    df = pd.read_csv(
        path,
        parse_dates=["open_time"],
        index_col="open_time",
        low_memory=False,
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Slice by split
    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    val_start = config.SPLIT_BOUNDARIES["val"]["start"]
    val_end   = config.SPLIT_BOUNDARIES["val"]["end"]
    test_start = config.SPLIT_BOUNDARIES["test"]["start"]
    test_end   = config.SPLIT_BOUNDARIES["test"]["end"]

    if split == "train":
        return df.loc[:train_end].copy()
    elif split == "val":
        return df.loc[val_start:val_end].copy()
    elif split == "train_val":
        return df.loc[:val_end].copy()
    elif split == "test":
        return df.loc[test_start:test_end].copy()
    elif split == "all":
        return df.copy()
    else:
        raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', 'train_val', 'test', or 'all'.")


# ===========================================================================
# Engineered Features storage (features.csv)
# ===========================================================================

def save_features_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path] = FEATURES_PATH,
) -> Path:
    """Save engineered features DataFrame to CSV with UTC ISO timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered features DataFrame with open_time index in UTC.
    filepath : Union[str, Path]
        Target CSV file path.

    Returns
    -------
    Path
        Absolute path to written features.csv.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out.index.name = "open_time"
        out.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        out.to_csv(path, index=True)

    logger.info("Saved features CSV to %s (%d rows, %d cols)", path, len(out), len(out.columns))
    return path


def load_features_csv(
    filepath: Union[str, Path] = FEATURES_PATH,
    split: str = "train",
    unlock_test: bool = False,
) -> pd.DataFrame:
    """Load engineered features dataset with split partition enforcement.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to features.csv.
    split : str
        Partition to load: 'train' (default), 'val', 'train_val', 'test', or 'all'.
    unlock_test : bool
        Safety lock. Must be True to load 'test' or 'all'.

    Returns
    -------
    pd.DataFrame
        Selected partition with UTC DatetimeIndex.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"features.csv not found: {path}")

    if split in ("test", "all") and not unlock_test:
        raise PermissionError(
            f"Test set is LOCKED. Cannot load split='{split}' with unlock_test=False."
        )

    df = pd.read_csv(
        path,
        parse_dates=["open_time"],
        index_col="open_time",
        low_memory=False,
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    val_start = config.SPLIT_BOUNDARIES["val"]["start"]
    val_end   = config.SPLIT_BOUNDARIES["val"]["end"]
    test_start = config.SPLIT_BOUNDARIES["test"]["start"]
    test_end   = config.SPLIT_BOUNDARIES["test"]["end"]

    if split == "train":
        return df.loc[:train_end].copy()
    elif split == "val":
        return df.loc[val_start:val_end].copy()
    elif split == "train_val":
        return df.loc[:val_end].copy()
    elif split == "test":
        return df.loc[test_start:test_end].copy()
    elif split == "all":
        return df.copy()
    else:
        raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', 'train_val', 'test', or 'all'.")


def save_drawings_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path] = DRAWINGS_CSV_PATH,
) -> Path:
    """Save companion drawings DataFrame (role 'drawing') to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Drawings dataset with open_time index in UTC.
    filepath : Union[str, Path]
        Target CSV file path.

    Returns
    -------
    Path
        Absolute path to written drawings.csv.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out.index.name = "open_time"
        out.to_csv(path, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        out.to_csv(path, index=True)

    logger.info("Saved drawings CSV to %s (%d rows, %d cols)", path, len(out), len(out.columns))
    return path


def load_drawings_csv(
    filepath: Union[str, Path] = DRAWINGS_CSV_PATH,
    split: str = "train",
    unlock_test: bool = False,
) -> pd.DataFrame:
    """Load companion drawings CSV with partition slicing and test-split locking.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to drawings.csv.
    split : str
        Partition to load: 'train' (default), 'val', 'train_val', 'test', or 'all'.
    unlock_test : bool
        Safety lock. Must be True to load 'test' or 'all'.

    Returns
    -------
    pd.DataFrame
        Selected partition with UTC DatetimeIndex.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"drawings.csv not found: {path}")

    if split in ("test", "all") and not unlock_test:
        raise PermissionError(
            f"Test set is LOCKED. Cannot load split='{split}' with unlock_test=False."
        )

    df = pd.read_csv(
        path,
        parse_dates=["open_time"],
        index_col="open_time",
        low_memory=False,
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    val_start = config.SPLIT_BOUNDARIES["val"]["start"]
    val_end   = config.SPLIT_BOUNDARIES["val"]["end"]
    test_start = config.SPLIT_BOUNDARIES["test"]["start"]
    test_end   = config.SPLIT_BOUNDARIES["test"]["end"]

    if split == "train":
        return df.loc[:train_end].copy()
    elif split == "val":
        return df.loc[val_start:val_end].copy()
    elif split == "train_val":
        return df.loc[:val_end].copy()
    elif split == "test":
        return df.loc[test_start:test_end].copy()
    elif split == "all":
        return df.copy()
    else:
        raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', 'train_val', 'test', or 'all'.")


# ===========================================================================
# Out-of-Fold (OOF) Prediction Caching
# ===========================================================================

OOF_DIR = config.RESULTS_DIR / "oof"
OOF_SCHEMA_COLS = [
    "open_time", "fold", "model", "setup", "horizon", "rule",
    "prob_down", "prob_neutral", "prob_up", "pred_label", "true_label",
]


def save_oof_predictions(
    df: pd.DataFrame,
    model_name: str,
    setup: str,
    horizon: int,
    rule: str,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Save out-of-fold predictions to a parquet file in results/oof/.

    Schema:
      open_time, fold, model, setup, horizon, rule,
      prob_down, prob_neutral, prob_up, pred_label, true_label
    """
    out_dir = Path(base_dir) if base_dir is not None else OOF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_{setup}_{horizon}_{rule}.parquet"
    filepath = out_dir / filename

    out_df = df.copy()
    if "open_time" not in out_df.columns and isinstance(out_df.index, pd.DatetimeIndex):
        out_df = out_df.reset_index()

    # Fill default values if columns missing
    if "model" not in out_df.columns:
        out_df["model"] = model_name
    if "setup" not in out_df.columns:
        out_df["setup"] = setup
    if "horizon" not in out_df.columns:
        out_df["horizon"] = horizon
    if "rule" not in out_df.columns:
        out_df["rule"] = rule
    if "prob_neutral" not in out_df.columns:
        out_df["prob_neutral"] = 0.0

    # Ensure required columns exist
    for col in OOF_SCHEMA_COLS:
        if col not in out_df.columns:
            raise ValueError(f"OOF DataFrame missing required column '{col}'")

    out_df = out_df[OOF_SCHEMA_COLS]
    out_df.to_parquet(filepath, index=False)
    logger.info("Saved OOF predictions to %s (%d rows)", filepath, len(out_df))
    return filepath


def load_oof_predictions(
    model_name: str,
    setup: str,
    horizon: int,
    rule: str,
    base_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load cached out-of-fold predictions from parquet."""
    in_dir = Path(base_dir) if base_dir is not None else OOF_DIR
    filename = f"{model_name}_{setup}_{horizon}_{rule}.parquet"
    filepath = in_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {filepath}")

    df = pd.read_parquet(filepath)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def save_horizon_oof_csv_gz(
    df: pd.DataFrame,
    filepath: Union[str, Path],
) -> Path:
    """Save horizon out-of-fold predictions to a gzip-compressed CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        OOF dataframe to persist.
    filepath : Union[str, Path]
        Target .csv.gz path.

    Returns
    -------
    Path
        Path to saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    logger.info("Saved Horizon OOF predictions to %s (%d rows)", path, len(df))
    return path


def load_horizon_oof_csv_gz(
    filepath: Union[str, Path],
) -> pd.DataFrame:
    """Load horizon out-of-fold predictions from a gzip-compressed CSV file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to .csv.gz file.

    Returns
    -------
    pd.DataFrame
        Loaded OOF predictions.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Horizon OOF file not found: {path}")
    df = pd.read_csv(path, compression="gzip")
    return df





