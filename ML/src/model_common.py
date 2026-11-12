"""Shared Model Training and Cross-Validation Procedures for Horizon Pipelines.

Provides:
  - load_truth: Loads ground truth and return metrics strictly from TRAIN split.
  - load_oof: Loads cached out-of-fold predictions.
  - run_oof: Executes 6-fold Purged Walk-Forward CV with Stride 3 downsampling.
  - fit_final: Retrains model on all post-warmup TRAIN rows for deployment.
  - prepare_model_features: Validates columns, handles structure feature caps,
    and guarantees zero NaNs or Infs.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import time
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import sklearn

from src import config
from src.column_dictionary import load_master, get_feature_columns
from src.cv import PurgedWalkForwardCV
from src.binary_models import build_binary_model, get_binary_inputs
from src.three_class_models import build_three_class_model, get_three_class_inputs
from src.io_utils import save_horizon_oof_csv_gz, load_horizon_oof_csv_gz
from src.utils import setup_logger

logger = setup_logger("model_common")

FORBIDDEN_ROLES = {"target", "threshold", "flag", "market", "drawing"}


def validate_feature_columns(cols: List[str]) -> None:
    """Assert that no requested input feature has a forbidden column role."""
    from src.column_dictionary import get_column_role
    for c in cols:
        try:
            role = get_column_role(c)
            if role in FORBIDDEN_ROLES:
                raise ValueError(
                    f"Column '{c}' has forbidden role '{role}' and cannot be used as a model input."
                )
        except KeyError:
            # If not in column dictionary, verify name doesn't start with raw market/target prefixes
            for forbidden_prefix in ["target_", "bin_dir_", "raw_sign_", "fwd_ret_", "level_"]:
                if c.startswith(forbidden_prefix):
                    raise ValueError(
                        f"Column '{c}' matches forbidden prefix '{forbidden_prefix}'."
                    )


def prepare_model_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Extract and validate model feature matrix.

    Caps structure columns feat_struct_bars_since_break and
    feat_struct_bars_since_retest at 288 with never-happened ceiling,
    and asserts absence of NaN or Inf values.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    feature_cols : List[str]
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        Prepared feature DataFrame.
    """
    validate_feature_columns(feature_cols)
    X = df[feature_cols].copy()

    # Cap structure features at 288 (values >= 288 or 9999 signify ceiling/never-happened)
    if "feat_struct_bars_since_break" in X.columns:
        X["feat_struct_bars_since_break"] = np.clip(
            X["feat_struct_bars_since_break"].astype(float).fillna(288.0), 0.0, 288.0
        )
    if "feat_struct_bars_since_retest" in X.columns:
        X["feat_struct_bars_since_retest"] = np.clip(
            X["feat_struct_bars_since_retest"].astype(float).fillna(288.0), 0.0, 288.0
        )

    # Check NaNs
    nan_counts = X.isna().sum()
    if (nan_counts > 0).any():
        bad_col = nan_counts[nan_counts > 0].index[0]
        bad_n = nan_counts[bad_col]
        raise ValueError(f"Feature column '{bad_col}' contains {bad_n} NaN values in modeling block.")

    # Check Infs
    inf_mask = np.isinf(X.values)
    if inf_mask.any():
        col_idx = np.where(inf_mask.any(axis=0))[0][0]
        bad_col = feature_cols[col_idx]
        raise ValueError(f"Feature column '{bad_col}' contains infinite values in modeling block.")

    return X


def load_truth(horizon: int = config.HORIZON) -> pd.DataFrame:
    """Load ground truth directional labels and return metrics for TRAIN split.

    Parameters
    ----------
    horizon : int
        Horizon length in candles.

    Returns
    -------
    pd.DataFrame
        Truth DataFrame indexed by open_time with target and forward return columns.
    """
    df_train = load_master(split="train")
    metric_cols = [
        f"bin_dir_{horizon}",
        f"target_{horizon}_{config.THREE_CLASS_RULE}",
        f"raw_sign_{horizon}",
        f"fwd_ret_open_{horizon}",
        f"has_valid_target_{horizon}",
    ]
    missing = [c for c in metric_cols if c not in df_train.columns]
    if missing:
        raise KeyError(f"Missing truth columns in master dataset: {missing}")

    if "open_time" in df_train.columns:
        res = df_train[["open_time"] + metric_cols].copy()
        res["open_time"] = pd.to_datetime(res["open_time"], utc=True)
        res = res.set_index("open_time")
    else:
        res = df_train[metric_cols].copy()
        res.index = pd.to_datetime(res.index, utc=True)
        res.index.name = "open_time"

    # Also supply open_time as a string column for convenient merging
    res["open_time"] = res.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    return res


def load_oof(
    model_id: str,
    horizon: int = config.HORIZON,
    smoke: bool = False,
) -> pd.DataFrame:
    """Load cached out-of-fold predictions.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. 'binary_lr', 'three_class_gb').
    horizon : int
        Horizon parameter.
    smoke : bool
        If True, loads from smoke directory.

    Returns
    -------
    pd.DataFrame
        OOF DataFrame.
    """
    oof_dir = config.RESULTS_DIR / (f"smoke_N{horizon}" if smoke else "oof")
    filename = f"{model_id}_N{horizon}.csv.gz"
    filepath = oof_dir / filename
    return load_horizon_oof_csv_gz(filepath)


def _get_git_commit() -> str:
    """Get current git commit hash if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return out
    except Exception:
        return "unknown"


def run_oof(
    model_id: str,
    horizon: int = config.HORIZON,
    smoke: bool = False,
    smoke_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Train model across expanding walk-forward folds and save out-of-fold predictions.

    Parameters
    ----------
    model_id : str
        Model identifier ('binary_lr', 'binary_rf', 'binary_gb',
        'three_class_lr', 'three_class_rf', 'three_class_gb').
    horizon : int
        Prediction horizon.
    smoke : bool
        If True, runs fast smoke mode on smoke_df.
    smoke_df : Optional[pd.DataFrame]
        Data for smoke testing (3,000 rows).

    Returns
    -------
    Path
        Path to saved OOF .csv.gz file.
    """
    oof_dir = config.RESULTS_DIR / (f"smoke_N{horizon}" if smoke else "oof")
    filename = f"{model_id}_N{horizon}.csv.gz"
    out_path = oof_dir / filename

    # Filename horizon check
    assert f"_N{horizon}" in filename, f"Artifact name {filename} must contain '_N{horizon}'"

    # Cache check
    if out_path.exists() and not smoke and not config.FORCE_RETRAIN:
        print(f"cached: {out_path}")
        return out_path

    # Determine model family & targets
    is_binary = model_id.startswith("binary")
    if is_binary:
        target_col = f"bin_dir_{horizon}"
        feature_cols = get_binary_inputs(model_id)
        build_fn = build_binary_model
    else:
        target_col = f"target_{horizon}_{config.THREE_CLASS_RULE}"
        feature_cols = get_three_class_inputs(model_id)
        build_fn = build_three_class_model

    valid_col = f"has_valid_target_{horizon}"

    # Load data
    if smoke:
        if smoke_df is None:
            raise ValueError("smoke_df must be provided when smoke=True")
        df_source = smoke_df.copy()
        stride = 1
        # 2 folds for smoke: 3,000 rows total
        # fold 1: train 0..1200 (strided), test 1212..1800
        # fold 2: train 0..2400 (strided), test 2412..3000
        fold_splits = [
            (1, np.arange(0, 1200), np.arange(1212, 1800)),
            (2, np.arange(0, 2400), np.arange(2412, 3000)),
        ]
    else:
        df_source = load_master(split="train")
        stride = config.STRIDE
        cv = PurgedWalkForwardCV()
        fold_splits = list(cv.split(df_source))

    oof_records = []

    for fold_num, tr_idx, te_idx in fold_splits:
        t_fold_start = time.time()

        # Training slice
        df_train_fold = df_source.iloc[tr_idx]
        # Filter valid rows with non-null target
        train_mask = (df_train_fold[valid_col] == 1) & (df_train_fold[target_col].notna())
        df_train_valid = df_train_fold[train_mask]

        # Apply stride to training rows
        df_train_strided = df_train_valid.iloc[::stride]
        n_fit_rows = len(df_train_strided)

        # Prepare training features
        X_train_raw = prepare_model_features(df_train_strided, feature_cols)
        y_train = df_train_strided[target_col].values

        # Held-out slice: predict on ALL held-out rows where valid_col == 1
        df_test_fold = df_source.iloc[te_idx]
        test_mask = df_test_fold[valid_col] == 1
        df_test_valid = df_test_fold[test_mask]
        n_test_rows = len(df_test_valid)

        X_test_raw = prepare_model_features(df_test_valid, feature_cols)

        # Instantiate fresh model
        model = build_fn(model_id, smoke=smoke)

        # Handle Logistic Regression scaling
        if "lr" in model_id.lower():
            scaler = StandardScaler()
            X_train_fit = scaler.fit_transform(X_train_raw)
            X_test_eval = scaler.transform(X_test_raw)
            model.fit(X_train_fit, y_train)
        elif "gb" in model_id.lower():
            # LightGBM: early stopping on last 15% with 12 raw candle (4 strided rows) purge
            n_tot = len(X_train_raw)
            n_es = max(10, int(n_tot * 0.15))
            purge_rows = max(1, 12 // stride)
            es_start = n_tot - n_es
            sub_end = max(10, es_start - purge_rows)

            X_tr_sub = X_train_raw.iloc[:sub_end]
            y_tr_sub = y_train[:sub_end]
            X_es = X_train_raw.iloc[es_start:]
            y_es = y_train[es_start:]

            model.fit(
                X_tr_sub,
                y_tr_sub,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            X_test_eval = X_test_raw
        else:
            # Random Forest
            model.fit(X_train_raw, y_train)
            X_test_eval = X_test_raw

        # Extract open_times whether open_time is a column or index
        if "open_time" in df_test_valid.columns:
            open_times = pd.to_datetime(df_test_valid["open_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            open_times = pd.to_datetime(df_test_valid.index, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Predict probabilities on held-out set
        probs = model.predict_proba(X_test_eval)

        # Assert classes alignment
        if is_binary:
            assert np.array_equal(model.classes_, np.array([0, 1])), (
                f"Binary classes must be [0, 1], got {model.classes_}"
            )
            p_down = probs[:, 0]
            p_up = probs[:, 1]
            fold_df = pd.DataFrame({
                "open_time": open_times,
                "fold": fold_num,
                "model_id": model_id,
                "p_down": p_down,
                "p_up": p_up,
            })
        else:
            assert np.array_equal(model.classes_, np.array([-1, 0, 1])), (
                f"Three-class classes must be [-1, 0, 1], got {model.classes_}"
            )
            p_down = probs[:, 0]
            p_neutral = probs[:, 1]
            p_up = probs[:, 2]
            fold_df = pd.DataFrame({
                "open_time": open_times,
                "fold": fold_num,
                "model_id": model_id,
                "p_down": p_down,
                "p_neutral": p_neutral,
                "p_up": p_up,
            })

        # Probability simplex check
        prob_sum = fold_df[["p_down", "p_up"] if is_binary else ["p_down", "p_neutral", "p_up"]].sum(axis=1)
        assert np.allclose(prob_sum, 1.0, atol=1e-6), "Probabilities must sum to 1 within 1e-6"

        oof_records.append(fold_df)
        elapsed = time.time() - t_fold_start
        print(f"fold {fold_num} | {model_id} | fit rows: {n_fit_rows:,} | held-out: {n_test_rows:,} | {elapsed:.2f}s")

    df_oof = pd.concat(oof_records, ignore_index=True)
    save_horizon_oof_csv_gz(df_oof, out_path)
    print(f"Saved OOF predictions to {out_path} ({len(df_oof):,} rows)")
    return out_path


def fit_final(
    model_id: str,
    horizon: int = config.HORIZON,
    smoke: bool = False,
    smoke_df: Optional[pd.DataFrame] = None,
) -> Tuple[Path, Path]:
    """Train final model on all post-warmup TRAIN rows and persist artifacts.

    Parameters
    ----------
    model_id : str
        Model identifier.
    horizon : int
        Prediction horizon.
    smoke : bool
        If True, runs fast smoke mode on smoke_df.
    smoke_df : Optional[pd.DataFrame]
        Data for smoke testing.

    Returns
    -------
    Tuple[Path, Path]
        Paths to (.joblib, .json) artifacts.
    """
    model_dir = config.MODELS_DIR / (f"smoke_N{horizon}" if smoke else f"horizon_{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = model_dir / f"{model_id}_N{horizon}.joblib"
    json_path = model_dir / f"{model_id}_N{horizon}.json"

    # Filename horizon check
    assert f"_N{horizon}" in joblib_path.name, f"Artifact name {joblib_path.name} must contain '_N{horizon}'"

    is_binary = model_id.startswith("binary")
    if is_binary:
        target_col = f"bin_dir_{horizon}"
        feature_cols = get_binary_inputs(model_id)
        build_fn = build_binary_model
    else:
        target_col = f"target_{horizon}_{config.THREE_CLASS_RULE}"
        feature_cols = get_three_class_inputs(model_id)
        build_fn = build_three_class_model

    valid_col = f"has_valid_target_{horizon}"

    if smoke:
        if smoke_df is None:
            raise ValueError("smoke_df must be provided when smoke=True")
        df_source = smoke_df.copy()
        stride = 1
    else:
        df_source = load_master(split="train")
        # Strip warmup candles
        cv = PurgedWalkForwardCV()
        df_source = df_source.iloc[cv.warmup_candles:].copy()
        stride = config.STRIDE

    # Filter valid rows with non-null target
    train_mask = (df_source[valid_col] == 1) & (df_source[target_col].notna())
    df_valid = df_source[train_mask]
    df_strided = df_valid.iloc[::stride]
    n_fit_rows = len(df_strided)

    X_train_raw = prepare_model_features(df_strided, feature_cols)
    y_train = df_strided[target_col].values

    model = build_fn(model_id, smoke=smoke)
    scaler = None

    if "lr" in model_id.lower():
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train_raw)
        model.fit(X_train_fit, y_train)
        payload = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
    elif "gb" in model_id.lower():
        n_tot = len(X_train_raw)
        n_es = max(10, int(n_tot * 0.15))
        purge_rows = max(1, 12 // stride)
        es_start = n_tot - n_es
        sub_end = max(10, es_start - purge_rows)

        X_tr_sub = X_train_raw.iloc[:sub_end]
        y_tr_sub = y_train[:sub_end]
        X_es = X_train_raw.iloc[es_start:]
        y_es = y_train[es_start:]

        model.fit(
            X_tr_sub,
            y_tr_sub,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        payload = {"model": model, "scaler": None, "feature_cols": feature_cols}
    else:
        model.fit(X_train_raw, y_train)
        payload = {"model": model, "scaler": None, "feature_cols": feature_cols}

    # Save joblib
    joblib.dump(payload, joblib_path)

    # Save JSON metadata
    meta_info: Dict[str, Any] = {
        "model_id": model_id,
        "horizon": horizon,
        "algorithm": type(model).__name__,
        "classes": [int(c) for c in model.classes_],
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "hyperparameters": {k: str(v) for k, v in model.get_params().items()},
        "fit_rows": n_fit_rows,
        "stride": stride,
        "random_seed": config.SEED,
        "git_commit": _get_git_commit(),
        "library_versions": {
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_range": {
            "start": str(df_valid["open_time"].min() if "open_time" in df_valid.columns else df_valid.index.min()),
            "end": str(df_valid["open_time"].max() if "open_time" in df_valid.columns else df_valid.index.max()),
        },
    }
    if scaler is not None:
        meta_info["scaler_mean"] = [float(x) for x in scaler.mean_]
        meta_info["scaler_scale"] = [float(x) for x in scaler.scale_]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Saved final model to {joblib_path} and metadata to {json_path}")
    return joblib_path, json_path
