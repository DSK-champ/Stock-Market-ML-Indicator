"""Three-class classification models: Logistic Regression, Random Forest, and LightGBM.

Defines model instantiation and feature selection for Branch B (Three-Class Ensemble).
Fixed hyperparameters per Section 11 Decisions log (no tuning, no grids).
Targets classes: -1 (DOWN), 0 (NEUTRAL), 1 (UP).
"""

from typing import List, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from src import config
from src.column_dictionary import get_feature_columns


def get_three_class_inputs(model_id: str) -> List[str]:
    """Return the declared input feature columns for a three-class model.

    Parameters
    ----------
    model_id : str
        'three_class_lr', 'three_class_rf', or 'three_class_gb'.

    Returns
    -------
    List[str]
        List of feature column names (70 for LR, 136 for RF/LightGBM).
    """
    model_id_clean = model_id.lower()
    if "lr" in model_id_clean:
        return get_feature_columns(feature_set="LINEAR_FEATURES")
    elif "rf" in model_id_clean or "gb" in model_id_clean:
        return get_feature_columns(feature_set="FS_ALL")
    else:
        raise ValueError(f"Unknown three-class model_id: {model_id}")


def build_three_class_model(model_id: str, smoke: bool = False) -> Any:
    """Build and return an un-fitted three-class model instance with declared hyperparameters.

    Parameters
    ----------
    model_id : str
        'three_class_lr', 'three_class_rf', or 'three_class_gb'.
    smoke : bool
        If True, use reduced estimators (30) for fast smoke testing.

    Returns
    -------
    Model instance (sklearn or lightgbm).
    """
    model_id_clean = model_id.lower()
    seed = config.SEED
    n_jobs = config.N_JOBS

    if "lr" in model_id_clean:
        return LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    elif "rf" in model_id_clean:
        n_est = 30 if smoke else 300
        return RandomForestClassifier(
            n_estimators=n_est,
            max_depth=6,
            min_samples_leaf=500,
            max_features="sqrt",
            max_samples=0.3,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    elif "gb" in model_id_clean:
        n_est = 30 if smoke else 600
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_leaves=15,
            max_depth=4,
            learning_rate=0.03,
            n_estimators=n_est,
            min_child_samples=500,
            colsample_bytree=0.7,
            subsample=0.7,
            subsample_freq=1,
            reg_lambda=10.0,
            max_bin=63,
            deterministic=True,
            force_row_wise=True,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced",
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown three-class model_id: {model_id}")
