"""Machine-readable record generation for every data file and step.

Usage
-----
Every notebook step calls ``describe_dataframe`` after saving a file to disk
(reading it back via io_utils), then calls ``write_details`` to persist the JSON
and update ``details/index.json``.

Rules enforced
--------------
- NaN and ±Inf are serialised as JSON null.
- Timestamps are ISO 8601 UTC strings.
- All floats at full precision.
- Columns without a column_dictionary entry raise KeyError.
- Test-split blocks contain only row count and first/last timestamp (no statistics).
- Paths stored relative to ML_DIR; no absolute personal paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.column_dictionary import get_column_meta

ML_DIR   = config.ML_DIR
DETAILS_DIR = ML_DIR / "details"


# ===========================================================================
# JSON serialisation helpers
# ===========================================================================

class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf → null, numpy scalars → Python natives,
    Timestamps → ISO 8601 UTC strings, Paths → relative-path strings."""

    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.astimezone(timezone.utc).isoformat()
        if isinstance(obj, Path):
            try:
                return str(obj.relative_to(ML_DIR))
            except ValueError:
                return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe(v: Any) -> Any:
    """Recursively sanitize a value so it round-trips through _SafeEncoder cleanly."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.astimezone(timezone.utc).isoformat()
    if isinstance(v, Path):
        try:
            return str(v.relative_to(ML_DIR))
        except ValueError:
            return str(v)
    return v


def _dumps(obj: Any) -> str:
    """Serialise dict to formatted JSON string.
    
    Uses allow_nan=False to strictly guarantee that no bare, non-standard
    NaN or Infinity tokens can ever be emitted in the JSON output.
    """
    return json.dumps(_safe(obj), indent=2, ensure_ascii=False, allow_nan=False)


def _no_nan_in_text(text: str) -> bool:
    """Verify that the JSON parses cleanly without NaN tokens (strict standard JSON)."""
    try:
        # Strict json.loads with parse_constant that rejects NaN/Infinity
        def _err(c):
            raise ValueError(f"Found bare JSON constant {c}")
        json.loads(text, parse_constant=_err)
        return True
    except Exception:
        return False


# ===========================================================================
# File-level helpers
# ===========================================================================

def compute_file_hash(filepath: Path) -> str:
    """SHA-256 hex digest of the file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Current git commit hash (short), or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ML_DIR.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _rel(p: Path) -> str:
    """Return path relative to ML_DIR as a forward-slash string."""
    try:
        return str(p.relative_to(ML_DIR)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


# ===========================================================================
# Index descriptor
# ===========================================================================

def _describe_index(df: pd.DataFrame) -> Dict[str, Any]:
    idx = df.index
    freq_str = None
    if isinstance(idx, pd.DatetimeIndex):
        try:
            inferred = pd.infer_freq(idx[:1000] if len(idx) > 1000 else idx)
            freq_str = str(inferred)
        except Exception:
            freq_str = None

    return {
        "name": idx.name,
        "dtype": str(idx.dtype),
        "timezone": str(idx.tz) if hasattr(idx, "tz") and idx.tz else None,
        "first": _safe(idx[0])  if len(idx) > 0 else None,
        "last":  _safe(idx[-1]) if len(idx) > 0 else None,
        "frequency": freq_str,
        "is_monotonic_increasing": bool(idx.is_monotonic_increasing),
        "n_duplicates": int(idx.duplicated().sum()),
    }


# ===========================================================================
# Column descriptor
# ===========================================================================

def _col_stats(series: pd.Series) -> Dict[str, Any]:
    """Basic statistics for a numeric or boolean series. Returns nulls for non-numeric."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        s = series.astype(float)
        return {
            "min":    _safe(s.min()),
            "max":    _safe(s.max()),
            "mean":   _safe(s.mean()),
            "std":    _safe(s.std()),
            "median": _safe(s.median()),
        }
    return {"min": None, "max": None, "mean": None, "std": None, "median": None}


def _describe_column(
    df: pd.DataFrame,
    col: str,
    position: int,
    full_df_stats: bool = True,       # for raw/clean files, compute on whole df
    train_mask: Optional[pd.Series] = None,
    val_mask:   Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Generate full metadata for one column."""
    meta = get_column_meta(col)   # raises KeyError if not registered
    series = df[col]

    n_missing = int(series.isna().sum())
    n_inf = 0
    if pd.api.types.is_numeric_dtype(series):
        n_inf = int(np.isinf(series.dropna()).sum())
    n_unique = int(series.nunique(dropna=False))

    base = {
        "name":     col,
        "dtype":    str(series.dtype),
        "position": position,
        "description":           meta["description"],
        "unit":                  meta.get("unit", ""),
        "formula":               meta.get("formula", ""),
        "group":                 meta.get("group", ""),
        "role":                  meta["role"],
        "is_future_looking":     meta["is_future_looking"],
        "allowed_as_model_input": meta["allowed_as_model_input"],
        "source_columns":        meta.get("source_columns", []),
        "n_missing":  n_missing,
        "n_inf":      n_inf,
        "n_unique":   n_unique,
    }
    if "horizon" in meta:
        base["horizon"] = meta["horizon"]
    if "theta_rule" in meta:
        base["theta_rule"] = meta["theta_rule"]

    if full_df_stats:
        base["stats_all"] = _col_stats(series)
    else:
        # Split-level stats
        if train_mask is not None:
            base["stats_train"] = _col_stats(series[train_mask])
        if val_mask is not None:
            base["stats_val"] = _col_stats(series[val_mask])
        # Test: no stats

    return base


# ===========================================================================
# Split row-count helper
# ===========================================================================

def _split_row_counts(df: pd.DataFrame) -> Dict[str, Any]:
    """Return per-split row counts using config boundaries.

    Test block: only count + first/last timestamp.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {"train": len(df), "val": 0, "test": 0}

    bounds = config.SPLIT_BOUNDARIES
    t_train_end   = bounds["train"]["end"]
    t_val_end     = bounds["val"]["end"]
    t_test_start  = bounds["test"]["start"]

    train_mask = df.index <= t_train_end
    val_mask   = (df.index > t_train_end) & (df.index <= t_val_end)
    test_mask  = df.index >= t_test_start

    n_train = int(train_mask.sum())
    n_val   = int(val_mask.sum())
    n_test  = int(test_mask.sum())

    result: Dict[str, Any] = {
        "train": n_train,
        "val":   n_val,
        "test":  {
            "n_rows": n_test,
            "first":  _safe(df.index[test_mask][0])  if n_test > 0 else None,
            "last":   _safe(df.index[test_mask][-1]) if n_test > 0 else None,
            "note":   "Test split is LOCKED — no value statistics computed.",
        },
    }
    return result


def _get_split_masks(df: pd.DataFrame):
    """Return (train_mask, val_mask) boolean Series for a DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        n = len(df)
        train_mask = pd.Series([True]  * n, index=df.index)
        val_mask   = pd.Series([False] * n, index=df.index)
        return train_mask, val_mask

    bounds = config.SPLIT_BOUNDARIES
    t_train_end = bounds["train"]["end"]
    t_val_end   = bounds["val"]["end"]

    train_mask = df.index <= t_train_end
    val_mask   = (df.index > t_train_end) & (df.index <= t_val_end)
    return train_mask, val_mask


# ===========================================================================
# Valid-row count for target files
# ===========================================================================

def _valid_row_counts(df: pd.DataFrame) -> Dict[str, Any]:
    """Count valid rows per horizon (has_valid_target_N) split by train/val."""
    result: Dict[str, Any] = {}
    train_mask, val_mask = _get_split_masks(df)
    for N in config.CANDIDATE_HORIZONS:
        col = f"has_valid_target_{N}"
        if col not in df.columns:
            continue
        v = df[col].astype(bool)
        result[f"N_{N}"] = {
            "train": int((v & train_mask).sum()),
            "val":   int((v & val_mask).sum()),
        }
    return result


# ===========================================================================
# Main: describe_dataframe
# ===========================================================================

def describe_dataframe(
    df: pd.DataFrame,
    file_path: Path,
    step: int,
    module: str,
    description: str,
    params: Dict[str, Any],
    checks: Dict[str, Any],
    full_df_stats: bool = True,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a complete detail dict for a saved data file.

    Parameters
    ----------
    df          : DataFrame loaded from disk via io_utils (never from memory).
    file_path   : Absolute path to the file on disk.
    step        : Step number that produced this file.
    module      : Python module that created the file (e.g. 'src.labels').
    description : One-sentence human-readable description of the file.
    params      : Dict of parameters used to create the file.
    checks      : Dict of check results (e.g. {'gaps': 0, 'duplicates': 0}).
    full_df_stats : If True, compute column stats on the whole df (for raw/clean).
                    If False, compute separately for train and val (for target files).
    warnings    : Optional list of warning strings.
    """
    file_path = Path(file_path)
    created_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
    file_hash  = compute_file_hash(file_path)
    file_size  = file_path.stat().st_size

    train_mask, val_mask = _get_split_masks(df)

    # Columns
    col_descriptors = []
    for pos, col in enumerate(df.columns):
        col_descriptors.append(
            _describe_column(
                df, col, pos,
                full_df_stats=full_df_stats,
                train_mask=train_mask if not full_df_stats else None,
                val_mask=val_mask   if not full_df_stats else None,
            )
        )

    d: Dict[str, Any] = {
        "file": {
            "path":       _rel(file_path),
            "size_bytes": file_size,
            "sha256":     file_hash,
            "created_at": created_at,
            "git_commit": get_git_commit(),
            "step":       step,
            "module":     module,
            "description": description,
        },
        "shape": {
            "n_rows":    len(df),
            "n_columns": len(df.columns),
        },
        "index":   _describe_index(df),
        "columns": col_descriptors,
        "row_counts_by_split": _split_row_counts(df),
        "parameters": _safe(params),
        "checks":     _safe(checks),
        "warnings":   warnings or [],
    }

    # Add valid-row counts for target files
    if any(c.startswith("has_valid_target_") for c in df.columns):
        d["valid_rows"] = _valid_row_counts(df)

    return d


# ===========================================================================
# Write detail JSON + update index
# ===========================================================================

def write_details(name: str, d: Dict[str, Any]) -> Path:
    """Write a detail dict to ``ML/details/{name}.json`` and update ``index.json``.

    Parameters
    ----------
    name : filename without .json (e.g. 'step01_raw')
    d    : dict from describe_dataframe (or describe_split, etc.)

    Returns
    -------
    Path of written file.
    """
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DETAILS_DIR / f"{name}.json"

    text = _dumps(d)
    assert _no_nan_in_text(text), (
        f"JSON for '{name}' contains literal NaN or Infinity tokens — sanitise before writing."
    )

    out_path.write_text(text, encoding="utf-8")

    # Update index.json
    _update_index(name, out_path, d)

    return out_path


def _update_index(name: str, json_path: Path, d: Dict[str, Any]) -> None:
    """Add or update this file's entry in ``details/index.json``."""
    index_path = DETAILS_DIR / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {"files": []}
    else:
        index = {"files": []}

    # Remove old entry for this name if it exists
    index["files"] = [e for e in index["files"] if e.get("name") != name]

    entry = {
        "name":       name,
        "path":       _rel(json_path),
        "step":       (d.get("file", {}).get("step") if isinstance(d.get("file"), dict) else None) or d.get("step"),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "size_bytes": json_path.stat().st_size,
    }
    # Add shape if present
    if "shape" in d:
        entry["shape"] = d["shape"]

    index["files"].append(entry)
    # Sort by step, then name
    index["files"].sort(key=lambda e: (e.get("step") or 0, e.get("name", "")))
    index["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    text = _dumps(index)
    index_path.write_text(text, encoding="utf-8")


# ===========================================================================
# Split-only descriptor (for step02_split.json — not a dataframe)
# ===========================================================================

def describe_split_boundaries(
    df: pd.DataFrame,
    step: int = 2,
) -> Dict[str, Any]:
    """Generate a detail dict for the split boundaries (not a file descriptor)."""
    bounds = config.SPLIT_BOUNDARIES
    train_mask, val_mask = _get_split_masks(df)
    test_mask = ~train_mask & ~val_mask

    n_train = int(train_mask.sum())
    n_val   = int(val_mask.sum())
    n_test  = int(test_mask.sum())

    d = {
        "step": step,
        "description": "Chronological train/val/test split boundaries and row counts.",
        "git_commit": get_git_commit(),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_rows": len(df),
        "splits": {
            "train": {
                "start": _safe(bounds["train"]["start"]),
                "end":   _safe(bounds["train"]["end"]),
                "n_rows": n_train,
                "days": bounds["train"]["days"],
            },
            "val": {
                "start": _safe(bounds["val"]["start"]),
                "end":   _safe(bounds["val"]["end"]),
                "n_rows": n_val,
                "days": bounds["val"]["days"],
            },
            "test": {
                "start": _safe(bounds["test"]["start"]),
                "end":   _safe(bounds["test"]["end"]),
                "n_rows": n_test,
                "days": bounds["test"]["days"],
                "note": "Test split is LOCKED — no statistics computed.",
            },
        },
        "purge_gap": {
            "candles": config.PURGE_GAP_CANDLES,
            "minutes": config.PURGE_GAP_CANDLES * config.INTERVAL_MINUTES,
            "rationale": "Purge gap equals label horizon to prevent label-leakage "
                         "between train/val and within walk-forward folds.",
        },
        "parameters": {
            "START": _safe(config.START_DATE_UTC),
            "END":   _safe(config.END_DATE_UTC),
            "CANDIDATE_HORIZONS": config.CANDIDATE_HORIZONS,
            "TOTAL_ROUNDTRIP_COST": config.TOTAL_ROUNDTRIP_COST,
        },
    }
    return d

