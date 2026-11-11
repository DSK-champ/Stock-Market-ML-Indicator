"""Comprehensive evaluation metrics for directional financial forecasting models.

Covers:
  1. Classification Metrics:
     - Accuracy, Balanced Accuracy
     - Precision, Recall, F1 (macro, weighted, and per-class)
     - Matthews Correlation Coefficient (MCC)
     - ROC AUC (One-vs-Rest for multiclass)
     - PR AUC (minority class or macro)
  2. Probabilistic Metrics:
     - Multi-class and Binary Log Loss (cross-entropy)
     - Brier Score
     - Row-normalized Confusion Matrix
     - Expected Calibration Error (ECE, 10 equal-mass / quantile bins)
     - Reliability curve points (confs, accs, bin counts)
  3. Operational Trading Metrics:
     - Coverage (% emitted signals)
     - Signals per day (coverage * 288)
     - Directional accuracy vs raw_sign_N
     - Break-even hit rate: BE = 0.5 + c / (2m), where m = mean(|fwd_ret_N|) on emitted signals
     - Break-even gap: hit - BE
     - Net return in bps after cost c = 0.23% (23 bps round-trip)
     - Stationary / Moving Block-Bootstrap Confidence Intervals (block size >= 288, 1,000 resamples)
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

from src import config

ROUNDTRIP_COST: float = getattr(config, "TOTAL_ROUNDTRIP_COST", 0.0023)


# ===========================================================================
# 1. Classification Metrics
# ===========================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Compute standard and balanced classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted target labels.
    y_prob : Optional[np.ndarray]
        Predicted class probabilities (shape: [N] for binary, [N, K] for multiclass).
    labels : Optional[List[Any]]
        List of label classes (e.g. [0, 1] or [-1, 0, 1]).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
        labels.sort()

    metrics: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    for i, lbl in enumerate(labels):
        metrics[f"precision_class_{lbl}"] = float(p_per_class[i])
        metrics[f"recall_class_{lbl}"] = float(r_per_class[i])
        metrics[f"f1_class_{lbl}"] = float(f1_per_class[i])

    # ROC AUC and PR AUC
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if len(labels) == 2:
                # Binary case
                pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                pos_label = labels[1]
                bin_y_true = (y_true == pos_label).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(bin_y_true, pos_prob))
                metrics["pr_auc"] = float(average_precision_score(bin_y_true, pos_prob))
            else:
                # Multiclass OvR
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
                )
                # Macro-averaged PR AUC
                pr_aucs = []
                for i, lbl in enumerate(labels):
                    bin_y = (y_true == lbl).astype(int)
                    if bin_y.sum() > 0:
                        pr_aucs.append(average_precision_score(bin_y, y_prob[:, i]))
                metrics["pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
        except Exception:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

    return metrics


# ===========================================================================
# 2. Probabilistic and Calibration Metrics
# ===========================================================================

def compute_probabilistic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Optional[List[Any]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute cross-entropy log loss, Brier score, confusion matrix, and ECE."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if labels is None:
        labels = np.unique(y_true).tolist()
        labels.sort()

    metrics: Dict[str, Any] = {}

    # Multi-class or binary log loss
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
    except Exception:
        metrics["log_loss"] = float("nan")

    # Brier score
    try:
        if len(labels) == 2:
            pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            pos_label = labels[1]
            bin_y = (y_true == pos_label).astype(float)
            metrics["brier_score"] = float(brier_score_loss(bin_y, pos_prob))
        else:
            # Multi-class Brier score: mean sum of squared residuals across classes
            y_onehot = np.zeros_like(y_prob)
            for i, lbl in enumerate(labels):
                y_onehot[y_true == lbl, i] = 1.0
            metrics["brier_score"] = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))
    except Exception:
        metrics["brier_score"] = float("nan")

    # Predicted classes from max probability
    if y_prob.ndim == 2:
        pred_indices = np.argmax(y_prob, axis=1)
        y_pred = np.array([labels[i] for i in pred_indices])
    else:
        y_pred = np.where(y_prob >= 0.5, labels[1], labels[0])

    # Row-normalized confusion matrix (rows = true class, sums to 1.0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    metrics["confusion_matrix_raw"] = cm.tolist()
    metrics["confusion_matrix_normalized"] = cm_norm.tolist()

    # Expected Calibration Error (ECE)
    ece_info = compute_expected_calibration_error(y_true, y_prob, labels=labels, n_bins=n_bins)
    metrics.update(ece_info)

    return metrics


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Optional[List[Any]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) with equal-mass quantile bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if labels is None:
        labels = np.unique(y_true).tolist()
        labels.sort()

    if len(labels) == 2:
        pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        confidences = np.maximum(pos_prob, 1.0 - pos_prob)
        pos_label = labels[1]
        preds = np.where(pos_prob >= 0.5, pos_label, labels[0])
        accuracies = (preds == y_true).astype(float)
    else:
        pred_idx = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        preds = np.array([labels[i] for i in pred_idx])
        accuracies = (preds == y_true).astype(float)

    n_samples = len(confidences)
    if n_samples == 0:
        return {"ece": float("nan"), "reliability_confs": [], "reliability_accs": [], "reliability_counts": []}

    # Equal-mass quantile bin edges
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(confidences, quantiles)
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > low) & (confidences <= high)
        cnt = int(mask.sum())
        bin_counts.append(cnt)
        if cnt > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
            ece += (cnt / n_samples) * abs(avg_acc - avg_conf)
        else:
            bin_confs.append(float((low + high) / 2))
            bin_accs.append(0.0)

    return {
        "ece": float(ece),
        "reliability_confs": bin_confs,
        "reliability_accs": bin_accs,
        "reliability_counts": bin_counts,
    }


# ===========================================================================
# 3. Operational Financial Metrics
# ===========================================================================

def compute_operational_metrics(
    y_true_sign: np.ndarray,
    y_pred_dir: np.ndarray,
    fwd_ret: np.ndarray,
    cost: float = ROUNDTRIP_COST,
    coverage_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute coverage, signals/day, break-even hit rate, gap, and net bps.

    Parameters
    ----------
    y_true_sign : np.ndarray
        True sign of price move: +1.0 for Up, -1.0 for Down, 0.0 for flat.
    y_pred_dir : np.ndarray
        Model predicted trade direction: +1.0 for Long, -1.0 for Short, 0.0 for Neutral/abstain.
    fwd_ret : np.ndarray
        Actual forward return realized over horizon N.
    cost : float
        Round-trip trading cost (c = 0.0023 = 23 bps).
    coverage_mask : Optional[np.ndarray]
        Optional boolean mask of eligible/valid evaluation candles.
    """
    y_true_sign = np.asarray(y_true_sign, dtype=float)
    y_pred_dir = np.asarray(y_pred_dir, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)

    if coverage_mask is not None:
        valid = np.asarray(coverage_mask, dtype=bool) & np.isfinite(y_true_sign) & np.isfinite(fwd_ret)
    else:
        valid = np.isfinite(y_true_sign) & np.isfinite(fwd_ret)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "coverage": 0.0,
            "signals_per_day": 0.0,
            "hit_rate": float("nan"),
            "break_even_rate": float("nan"),
            "break_even_gap": float("nan"),
            "net_return_bps": float("nan"),
            "n_signals": 0,
            "m_fwd_ret": float("nan"),
        }

    # Emitted trades (non-neutral direction: +1 or -1)
    emitted = valid & (y_pred_dir != 0.0)
    n_emitted = int(emitted.sum())

    coverage = float(n_emitted / n_valid)
    signals_per_day = float(coverage * 288.0)

    if n_emitted == 0:
        return {
            "coverage": 0.0,
            "signals_per_day": 0.0,
            "hit_rate": float("nan"),
            "break_even_rate": float("nan"),
            "break_even_gap": float("nan"),
            "net_return_bps": 0.0,
            "n_signals": 0,
            "m_fwd_ret": float("nan"),
        }

    # Accuracy / hit rate against raw direction
    sig_pred = y_pred_dir[emitted]
    sig_true = y_true_sign[emitted]
    sig_ret = fwd_ret[emitted]

    hits = (sig_pred == sig_true)
    hit_rate = float(np.mean(hits))

    # Mean absolute forward return on emitted signals
    m = float(np.mean(np.abs(sig_ret)))
    if m > 0:
        be_rate = float(0.5 + cost / (2.0 * m))
        be_gap = float(hit_rate - be_rate)
    else:
        be_rate = float("inf")
        be_gap = float("-inf")

    # Net return per trade after cost c
    trade_ret = (sig_pred * sig_ret) - cost
    net_return_bps = float(np.mean(trade_ret) * 10_000.0)

    return {
        "coverage": coverage,
        "signals_per_day": signals_per_day,
        "hit_rate": hit_rate,
        "break_even_rate": be_rate,
        "break_even_gap": be_gap,
        "net_return_bps": net_return_bps,
        "n_signals": n_emitted,
        "m_fwd_ret": m,
    }


# ===========================================================================
# 4. Moving Block-Bootstrap Confidence Intervals
# ===========================================================================

def block_bootstrap_ci(
    y_true_sign: np.ndarray,
    y_pred_dir: np.ndarray,
    fwd_ret: np.ndarray,
    block_size: int = 288,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    cost: float = ROUNDTRIP_COST,
) -> Dict[str, Dict[str, float]]:
    """Compute moving-block bootstrap confidence intervals for operational metrics.

    Parameters
    ----------
    block_size : int
        Block size in candles (default 288 = 1 full day of 5-min candles).
    n_resamples : int
        Number of bootstrap resamples (default 1,000).
    alpha : float
        Significance level (default 0.05 for 95% two-sided CI).
    seed : int
        Fixed random seed (default 42).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true_sign)
    if n < block_size * 2:
        return {}

    n_blocks = int(np.ceil(n / block_size))

    # Pre-sample start indices for all resamples
    max_start = n - block_size
    starts = rng.randint(0, max_start + 1, size=(n_resamples, n_blocks))

    boot_hits = np.empty(n_resamples, dtype=float)
    boot_gaps = np.empty(n_resamples, dtype=float)
    boot_bps = np.empty(n_resamples, dtype=float)

    for b in range(n_resamples):
        # Stitch blocks together
        indices = np.concatenate([
            np.arange(st, st + block_size) for st in starts[b]
        ])[:n]

        sub_true = y_true_sign[indices]
        sub_pred = y_pred_dir[indices]
        sub_ret = fwd_ret[indices]

        res = compute_operational_metrics(sub_true, sub_pred, sub_ret, cost=cost)
        boot_hits[b] = res["hit_rate"]
        boot_gaps[b] = res["break_even_gap"]
        boot_bps[b] = res["net_return_bps"]

    low_q = (alpha / 2.0) * 100.0
    high_q = (1.0 - alpha / 2.0) * 100.0

    def _summarize(arr: np.ndarray) -> Dict[str, float]:
        clean = arr[np.isfinite(arr)]
        if len(clean) == 0:
            return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "std_err": float("nan")}
        return {
            "mean": float(np.mean(clean)),
            "ci_lower": float(np.percentile(clean, low_q)),
            "ci_upper": float(np.percentile(clean, high_q)),
            "std_err": float(np.std(clean)),
        }

    return {
        "hit_rate": _summarize(boot_hits),
        "break_even_gap": _summarize(boot_gaps),
        "net_return_bps": _summarize(boot_bps),
    }
