"""Probability calibration for directional classification models.

Implements:
  1. Platt Scaling (Logistic Sigmoid Calibration)
     - Fits logistic regression on model log-odds or raw probability predictions.
     - For binary classification: maps p -> p* in [0, 1].
     - For multiclass (3-class): vector Platt scaling or 1-vs-rest Platt with row-normalization.
  2. Isotonic Regression Calibration
     - Non-parametric isotonic regression with out-of-bounds clipping.
     - For multiclass: 1-vs-rest isotonic calibrators with row normalization (probabilities sum to 1.0).
  3. Internal Cross-Fitted Calibration
     - Calibrators are fit ONLY on the fold's internal early-stopping validation split or
       internal cross-validation, then applied strictly out-of-fold to held-out evaluation candles.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safely compute log-odds (logits) with numerical clipping."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


class PlattCalibrator:
    """Parametric Platt scaling (logistic sigmoid) probability calibrator.

    Parameters
    ----------
    eps : float
        Epsilon for numerical stability when computing logits.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.models: List[LogisticRegression] = []
        self.is_binary = True
        self.n_classes = 2
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        """Fit logistic calibration model on validation probabilities.

        Parameters
        ----------
        y_prob : np.ndarray
            Predicted probabilities. Shape [N] or [N, 2] for binary, [N, K] for multiclass.
        y_true : np.ndarray
            Ground truth labels.
        """
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)
        self.classes_ = np.unique(y_true)
        self.classes_.sort()
        self.n_classes = len(self.classes_)

        if self.n_classes == 2:
            self.is_binary = True
            pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            logits = _logit(pos_prob, self.eps).reshape(-1, 1)
            pos_label = self.classes_[1]
            bin_target = (y_true == pos_label).astype(int)

            lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
            lr.fit(logits, bin_target)
            self.models = [lr]
        else:
            self.is_binary = False
            self.models = []
            for i, cls_val in enumerate(self.classes_):
                cls_prob = y_prob[:, i]
                logits = _logit(cls_prob, self.eps).reshape(-1, 1)
                bin_target = (y_true == cls_val).astype(int)

                lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
                lr.fit(logits, bin_target)
                self.models.append(lr)

        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities to calibrated probabilities."""
        y_prob = np.asarray(y_prob)

        if self.is_binary:
            pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            logits = _logit(pos_prob, self.eps).reshape(-1, 1)
            cal_pos = self.models[0].predict_proba(logits)[:, 1]
            if y_prob.ndim == 2:
                cal_neg = 1.0 - cal_pos
                return np.column_stack([cal_neg, cal_pos])
            return cal_pos
        else:
            cal_probs = []
            for i in range(self.n_classes):
                cls_prob = y_prob[:, i]
                logits = _logit(cls_prob, self.eps).reshape(-1, 1)
                cal_probs.append(self.models[i].predict_proba(logits)[:, 1])
            res = np.column_stack(cal_probs)
            # Row-normalize to sum to 1.0
            row_sums = np.sum(res, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return res / row_sums


class IsotonicCalibrator:
    """Non-parametric isotonic regression probability calibrator.

    Parameters
    ----------
    out_of_bounds : str
        Policy for predictions outside training range ('clip').
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self.models: List[IsotonicRegression] = []
        self.is_binary = True
        self.n_classes = 2
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression model on validation probabilities."""
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)
        self.classes_ = np.unique(y_true)
        self.classes_.sort()
        self.n_classes = len(self.classes_)

        if self.n_classes == 2:
            self.is_binary = True
            pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            pos_label = self.classes_[1]
            bin_target = (y_true == pos_label).astype(float)

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)
            iso.fit(pos_prob, bin_target)
            self.models = [iso]
        else:
            self.is_binary = False
            self.models = []
            for i, cls_val in enumerate(self.classes_):
                cls_prob = y_prob[:, i]
                bin_target = (y_true == cls_val).astype(float)

                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)
                iso.fit(cls_prob, bin_target)
                self.models.append(iso)

        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities to calibrated probabilities."""
        y_prob = np.asarray(y_prob)

        if self.is_binary:
            pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            cal_pos = self.models[0].predict(pos_prob)
            if y_prob.ndim == 2:
                cal_neg = 1.0 - cal_pos
                return np.column_stack([cal_neg, cal_pos])
            return cal_pos
        else:
            cal_probs = []
            for i in range(self.n_classes):
                cls_prob = y_prob[:, i]
                cal_probs.append(self.models[i].predict(cls_prob))
            res = np.column_stack(cal_probs)
            # Row-normalize to sum to 1.0
            row_sums = np.sum(res, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return res / row_sums


def fit_and_apply_calibrator(
    y_prob_val: np.ndarray,
    y_true_val: np.ndarray,
    y_prob_test: np.ndarray,
    method: str = "platt",
) -> Tuple[np.ndarray, Union[PlattCalibrator, IsotonicCalibrator]]:
    """Fit calibrator on validation split and apply to test predictions.

    Parameters
    ----------
    y_prob_val : np.ndarray
        Internal validation predicted probabilities.
    y_true_val : np.ndarray
        Internal validation ground truth labels.
    y_prob_test : np.ndarray
        Held-out test predicted probabilities.
    method : str
        'platt' or 'isotonic'.

    Returns
    -------
    cal_prob_test : np.ndarray
        Calibrated probabilities for held-out test split.
    calibrator : PlattCalibrator or IsotonicCalibrator
        The fitted calibrator object.
    """
    if method.lower() == "platt":
        calibrator = PlattCalibrator().fit(y_prob_val, y_true_val)
    elif method.lower() == "isotonic":
        calibrator = IsotonicCalibrator().fit(y_prob_val, y_true_val)
    else:
        raise ValueError(f"Unknown calibration method: '{method}'. Expected 'platt' or 'isotonic'.")

    cal_prob_test = calibrator.predict_proba(y_prob_test)
    return cal_prob_test, calibrator
