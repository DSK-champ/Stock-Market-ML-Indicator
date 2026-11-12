"""Meta-Model Combiner Layer (Branch D).

Stacks the probabilistic outputs of Branch A (Binary models),
Branch B (Three-class models), and Branch C (TA Engine)
into a multinomial Logistic Regression meta-model predicting
target_{HORIZON}_S1 (classes -1 DOWN, 0 NEUTRAL, 1 UP).

Evaluation is strictly nested out-of-fold across folds 2..6,
preventing any in-sample or stacking leakage.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn

from src import config
from src.column_dictionary import load_master
from src.ta_engine import compute_ta_scores, TA_SCORE_COLUMNS
from src.io_utils import save_horizon_oof_csv_gz, load_horizon_oof_csv_gz
from src.model_common import load_oof, _get_git_commit
from src.utils import setup_logger

logger = setup_logger("meta_model")

# Exactly these 15 inputs in this exact order
META_INPUT_COLUMNS: List[str] = [
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


def build_meta_inputs(
    horizon: int = config.HORIZON,
    smoke: bool = False,
    smoke_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the consolidated 15 meta-feature matrix from OOF predictions and TA scores.

    Parameters
    ----------
    horizon : int
        Prediction horizon.
    smoke : bool
        If True, loads from smoke directories.
    smoke_df : Optional[pd.DataFrame]
        Raw dataset slice for smoke mode.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by open_time with 15 META_INPUT_COLUMNS,
        fold, target, raw_sign, and forward returns.
    """
    # 1. Load base model OOF files
    df_bin_lr = load_oof("binary_lr", horizon=horizon, smoke=smoke)[["open_time", "fold", "p_up"]].rename(
        columns={"p_up": "bin_lr_p_up"}
    )
    df_bin_rf = load_oof("binary_rf", horizon=horizon, smoke=smoke)[["open_time", "p_up"]].rename(
        columns={"p_up": "bin_rf_p_up"}
    )
    df_bin_gb = load_oof("binary_gb", horizon=horizon, smoke=smoke)[["open_time", "p_up"]].rename(
        columns={"p_up": "bin_gb_p_up"}
    )

    df_tri_lr = load_oof("three_class_lr", horizon=horizon, smoke=smoke)[["open_time", "p_up", "p_down"]].rename(
        columns={"p_up": "tri_lr_p_up", "p_down": "tri_lr_p_down"}
    )
    df_tri_rf = load_oof("three_class_rf", horizon=horizon, smoke=smoke)[["open_time", "p_up", "p_down"]].rename(
        columns={"p_up": "tri_rf_p_up", "p_down": "tri_rf_p_down"}
    )
    df_tri_gb = load_oof("three_class_gb", horizon=horizon, smoke=smoke)[["open_time", "p_up", "p_down"]].rename(
        columns={"p_up": "tri_gb_p_up", "p_down": "tri_gb_p_down"}
    )

    # Merge OOF files on open_time
    meta = df_bin_lr.merge(df_bin_rf, on="open_time", how="inner")
    meta = meta.merge(df_bin_gb, on="open_time", how="inner")
    meta = meta.merge(df_tri_lr, on="open_time", how="inner")
    meta = meta.merge(df_tri_rf, on="open_time", how="inner")
    meta = meta.merge(df_tri_gb, on="open_time", how="inner")

    meta["open_time_dt"] = pd.to_datetime(meta["open_time"], utc=True)
    meta = meta.sort_values("open_time_dt").reset_index(drop=True)

    # 2. Compute TA scores on matching data
    if smoke:
        if smoke_df is None:
            raise ValueError("smoke_df must be provided when smoke=True")
        df_train = smoke_df.copy()
    else:
        df_train = load_master(split="train")

    if "open_time" in df_train.columns:
        open_time_dt = pd.to_datetime(df_train["open_time"], utc=True)
    else:
        open_time_dt = pd.to_datetime(df_train.index, utc=True)

    params_path = config.MODELS_DIR / (f"smoke_N{horizon}" if smoke else f"horizon_{horizon}") / f"ta_params_N{horizon}.json"
    ta_scores = compute_ta_scores(df_train, params_path=params_path, save_params=True, horizon=horizon)
    ta_scores["open_time_dt"] = open_time_dt

    # Target & return columns
    target_col = f"target_{horizon}_{config.THREE_CLASS_RULE}"
    truth_col = f"raw_sign_{horizon}"
    ret_col = f"fwd_ret_open_{horizon}"

    targets_df = pd.DataFrame({
        "open_time_dt": open_time_dt,
        target_col: df_train[target_col].values,
        truth_col: df_train[truth_col].values,
        ret_col: df_train[ret_col].values,
    })

    # Join TA scores and targets
    meta = meta.merge(ta_scores, on="open_time_dt", how="inner")
    meta = meta.merge(targets_df, on="open_time_dt", how="inner")

    # Assert exact columns presence and no NaNs
    missing = [c for c in META_INPUT_COLUMNS if c not in meta.columns]
    assert not missing, f"Missing meta input columns: {missing}"
    assert meta[META_INPUT_COLUMNS].isna().sum().sum() == 0, "Meta inputs contain NaNs!"

    meta["open_time"] = meta["open_time_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = meta.drop(columns=["open_time_dt"])

    return meta


def nested_meta_oof(
    horizon: int = config.HORIZON,
    smoke: bool = False,
    meta_df: Optional[pd.DataFrame] = None,
    smoke_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Evaluate meta-model using strictly nested walk-forward cross-validation.

    For fold k in 2..6, trains on folds 1..k-1 (with last 12 rows purged before
    the boundary), and predicts on fold k. Fold 1 is never evaluated out-of-sample.

    Parameters
    ----------
    horizon : int
        Prediction horizon.
    smoke : bool
        If True, runs fast smoke mode.
    meta_df : Optional[pd.DataFrame]
        Pre-built meta feature DataFrame.
    smoke_df : Optional[pd.DataFrame]
        Smoke test slice.

    Returns
    -------
    Path
        Path to saved results/oof/meta_N{horizon}.csv.gz.
    """
    oof_dir = config.RESULTS_DIR / (f"smoke_N{horizon}" if smoke else "oof")
    filename = f"meta_N{horizon}.csv.gz"
    out_path = oof_dir / filename

    assert f"_N{horizon}" in filename, f"Artifact name {filename} must contain '_N{horizon}'"

    if meta_df is None:
        meta_df = build_meta_inputs(horizon=horizon, smoke=smoke, smoke_df=smoke_df)

    target_col = f"target_{horizon}_{config.THREE_CLASS_RULE}"
    unique_folds = sorted(meta_df["fold"].unique())
    logger.info("Meta folds present: %s", unique_folds)

    meta_oof_records = []
    # Evaluate on folds 2..max_fold (fold 1 has no preceding history)
    eval_folds = [f for f in unique_folds if f > 1]
    if not eval_folds and len(unique_folds) >= 2:
        eval_folds = [unique_folds[1]]

    for k in eval_folds:
        # Training folds: all folds strictly before k
        df_train_folds = meta_df[meta_df["fold"] < k].copy()

        # Purge last 12 candles before fold k
        if len(df_train_folds) > 12:
            df_train_purged = df_train_folds.iloc[:-12].copy()
        else:
            df_train_purged = df_train_folds.copy()

        # Filter valid target
        train_mask = df_train_purged[target_col].notna()
        df_train_fit = df_train_purged[train_mask]

        # Test fold
        df_test_k = meta_df[meta_df["fold"] == k].copy()
        test_mask = df_test_k[target_col].notna()
        df_test_eval = df_test_k[test_mask]

        X_train = df_train_fit[META_INPUT_COLUMNS]
        y_train = df_train_fit[target_col].values
        X_test = df_test_eval[META_INPUT_COLUMNS]

        # Fit Scaler & Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        meta_clf = LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=config.SEED,
        )
        meta_clf.fit(X_train_scaled, y_train)

        assert np.array_equal(meta_clf.classes_, np.array([-1, 0, 1])), (
            f"Meta-model classes must be [-1, 0, 1], got {meta_clf.classes_}"
        )

        probs = meta_clf.predict_proba(X_test_scaled)
        p_down = probs[:, 0]
        p_neutral = probs[:, 1]
        p_up = probs[:, 2]

        fold_preds = pd.DataFrame({
            "open_time": df_test_eval["open_time"].values,
            "fold": k,
            "model_id": "meta_model",
            "p_down": p_down,
            "p_neutral": p_neutral,
            "p_up": p_up,
        })

        # Probability simplex check
        prob_sum = fold_preds[["p_down", "p_neutral", "p_up"]].sum(axis=1)
        assert np.allclose(prob_sum, 1.0, atol=1e-6), "Meta probabilities must sum to 1 within 1e-6"

        meta_oof_records.append(fold_preds)
        print(f"meta nested fold {k} | train rows: {len(df_train_fit):,} | test rows: {len(df_test_eval):,}")

    df_meta_oof = pd.concat(meta_oof_records, ignore_index=True)
    save_horizon_oof_csv_gz(df_meta_oof, out_path)
    print(f"Saved Meta OOF predictions to {out_path} ({len(df_meta_oof):,} rows, folds 2..{max(eval_folds)})")
    return out_path


def fit_meta_final(
    horizon: int = config.HORIZON,
    smoke: bool = False,
    meta_df: Optional[pd.DataFrame] = None,
    smoke_df: Optional[pd.DataFrame] = None,
) -> Tuple[Path, Path]:
    """Train final meta-model on all available OOF rows and persist artifacts.

    Parameters
    ----------
    horizon : int
        Prediction horizon.
    smoke : bool
        If True, saves to smoke directory.
    meta_df : Optional[pd.DataFrame]
        Pre-built meta feature matrix.
    smoke_df : Optional[pd.DataFrame]
        Smoke test slice.

    Returns
    -------
    Tuple[Path, Path]
        Paths to (.joblib, .json) artifacts.
    """
    model_dir = config.MODELS_DIR / (f"smoke_N{horizon}" if smoke else f"horizon_{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = model_dir / f"meta_model_N{horizon}.joblib"
    json_path = model_dir / f"meta_model_N{horizon}.json"

    assert f"_N{horizon}" in joblib_path.name, f"Artifact name {joblib_path.name} must contain '_N{horizon}'"

    if meta_df is None:
        meta_df = build_meta_inputs(horizon=horizon, smoke=smoke, smoke_df=smoke_df)

    target_col = f"target_{horizon}_{config.THREE_CLASS_RULE}"
    valid_mask = meta_df[target_col].notna()
    df_fit = meta_df[valid_mask]

    X = df_fit[META_INPUT_COLUMNS]
    y = df_fit[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=config.SEED,
    )
    model.fit(X_scaled, y)

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_cols": META_INPUT_COLUMNS,
    }
    joblib.dump(payload, joblib_path)

    meta_info: Dict[str, Any] = {
        "model_id": "meta_model",
        "horizon": horizon,
        "algorithm": "LogisticRegression (multinomial stacking)",
        "classes": [int(c) for c in model.classes_],
        "feature_count": len(META_INPUT_COLUMNS),
        "feature_cols": META_INPUT_COLUMNS,
        "hyperparameters": {k: str(v) for k, v in model.get_params().items()},
        "fit_rows": len(df_fit),
        "random_seed": config.SEED,
        "git_commit": _get_git_commit(),
        "library_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Saved final meta-model to {joblib_path} and metadata to {json_path}")
    return joblib_path, json_path
