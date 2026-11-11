"""Purged Walk-Forward Cross-Validation for financial time series.

Guarantees:
  1. Strict temporal causality: train strictly precedes held-out test.
  2. Purge gap: exactly PURGE_GAP candles (12 candles, max forward horizon)
     between training block and held-out block to prevent label overlap.
  3. Internal validation: 15% early stopping split within each fold's training block,
     also separated by a 12-candle purge gap.
  4. Non-overlapping evaluation: held-out blocks across the 6 folds form a contiguous,
     disjoint partition of the post-warmup evaluation horizon.
  5. Identical evaluation set across all horizons: row sets are identical.
"""

from typing import Iterator, Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from src import config

LOOKBACK_CANDLES: int = getattr(config, "LOOKBACK_CANDLES", 2018)
PURGE_GAP: int = 12
N_FOLDS: int = 6
INITIAL_TRAIN_CANDLES: int = 86400
EARLY_STOP_VAL_RATIO: float = 0.15


class PurgedWalkForwardCV:
    """Purged expanding-window walk-forward cross-validation.

    Evaluated on post-warmup train candles (218,935 rows).
    Initial training window: 86,400 candles (~300 days).
    Purge gap: 12 candles between train and held-out test.
    Held-out blocks: 6 sequential non-overlapping blocks (~22,087 to 22,088 candles each).

    Parameters
    ----------
    n_folds : int
        Number of walk-forward folds (default 6).
    initial_train_candles : int
        Number of candles in the first training window (default 86,400).
    purge_gap : int
        Number of candles purged between training end and test start (default 12).
    val_ratio : float
        Fraction of fold training set used for internal early stopping (default 0.15).
    warmup_candles : int
        Number of initial warmup candles stripped before CV (default 2,018).
    """

    def __init__(
        self,
        n_folds: int = N_FOLDS,
        initial_train_candles: int = INITIAL_TRAIN_CANDLES,
        purge_gap: int = PURGE_GAP,
        val_ratio: float = EARLY_STOP_VAL_RATIO,
        warmup_candles: int = LOOKBACK_CANDLES,
    ):
        self.n_folds = n_folds
        self.initial_train_candles = initial_train_candles
        self.purge_gap = purge_gap
        self.val_ratio = val_ratio
        self.warmup_candles = warmup_candles

    def _compute_fold_slices(self, n_total_post_warmup: int) -> List[Dict[str, int]]:
        """Compute fold boundaries on post-warmup row indices [0, n_total_post_warmup)."""
        eval_total = n_total_post_warmup - self.initial_train_candles - self.purge_gap
        if eval_total < self.n_folds:
            raise ValueError(
                f"Insufficient candles ({n_total_post_warmup}) for initial train "
                f"({self.initial_train_candles}) + purge ({self.purge_gap}) + {self.n_folds} folds."
            )

        base_h = eval_total // self.n_folds
        rem = eval_total % self.n_folds
        h_sizes = [base_h + (1 if i < rem else 0) for i in range(self.n_folds)]

        folds = []
        curr_test_start = self.initial_train_candles + self.purge_gap

        for f in range(self.n_folds):
            h_len = h_sizes[f]
            test_start = curr_test_start
            test_end = curr_test_start + h_len

            train_start = 0
            train_end = test_start - self.purge_gap
            purge_start = train_end
            purge_end = test_start

            folds.append({
                "fold": f + 1,
                "train_start": train_start,
                "train_end": train_end,
                "purge_start": purge_start,
                "purge_end": purge_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_len": train_end - train_start,
                "purge_len": purge_end - purge_start,
                "test_len": test_end - test_start,
            })
            curr_test_start = test_end

        return folds

    def split(
        self,
        df: pd.DataFrame,
        is_post_warmup: bool = False,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield (fold, train_indices, test_indices) for each fold.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (full train set or already post-warmup).
        is_post_warmup : bool
            If True, df is assumed to already have warmup stripped.
            If False, the first `warmup_candles` rows are skipped.

        Yields
        ------
        fold : int (1..6)
        train_idx : np.ndarray of integer positional indices into df
        test_idx : np.ndarray of integer positional indices into df
        """
        n_rows = len(df)
        offset = 0 if is_post_warmup else self.warmup_candles
        n_post_warmup = n_rows - offset

        slices = self._compute_fold_slices(n_post_warmup)
        for s in slices:
            tr_idx = np.arange(offset + s["train_start"], offset + s["train_end"])
            te_idx = np.arange(offset + s["test_start"], offset + s["test_end"])
            yield s["fold"], tr_idx, te_idx

    def split_with_val(
        self,
        df: pd.DataFrame,
        is_post_warmup: bool = False,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (fold, sub_train_indices, val_indices, test_indices).

        Sub-train and internal early stopping validation are separated by a 12-candle purge.

        Yields
        ------
        fold : int (1..6)
        sub_train_idx : np.ndarray
        val_idx : np.ndarray
        test_idx : np.ndarray
        """
        n_rows = len(df)
        offset = 0 if is_post_warmup else self.warmup_candles
        n_post_warmup = n_rows - offset

        slices = self._compute_fold_slices(n_post_warmup)
        for s in slices:
            tr_len = s["train_len"]
            val_len = int(tr_len * self.val_ratio)
            sub_tr_end = tr_len - val_len - self.purge_gap
            val_start = tr_len - val_len

            sub_tr_idx = np.arange(offset + s["train_start"], offset + s["train_start"] + sub_tr_end)
            val_idx = np.arange(offset + s["train_start"] + val_start, offset + s["train_end"])
            te_idx = np.arange(offset + s["test_start"], offset + s["test_end"])

            yield s["fold"], sub_tr_idx, val_idx, te_idx

    def get_fold_summary(
        self,
        df: Optional[pd.DataFrame] = None,
        is_post_warmup: bool = False,
    ) -> pd.DataFrame:
        """Return a structured summary DataFrame of all 6 folds.

        If df is provided (with a DatetimeIndex), includes start and end timestamps.
        """
        n_rows = len(df) if df is not None else config.SPLIT_BOUNDARIES["train"]["expected_candles"]
        offset = 0 if is_post_warmup else self.warmup_candles
        n_post_warmup = n_rows - offset

        slices = self._compute_fold_slices(n_post_warmup)
        records = []
        for s in slices:
            abs_tr_start = offset + s["train_start"]
            abs_tr_end = offset + s["train_end"]
            abs_pu_start = offset + s["purge_start"]
            abs_pu_end = offset + s["purge_end"]
            abs_te_start = offset + s["test_start"]
            abs_te_end = offset + s["test_end"]

            rec = {
                "fold": s["fold"],
                "train_start_idx": abs_tr_start,
                "train_end_idx": abs_tr_end - 1,
                "train_rows": s["train_len"],
                "purge_start_idx": abs_pu_start,
                "purge_end_idx": abs_pu_end - 1,
                "purge_rows": s["purge_len"],
                "held_out_start_idx": abs_te_start,
                "held_out_end_idx": abs_te_end - 1,
                "held_out_rows": s["test_len"],
            }

            if df is not None and isinstance(df.index, pd.DatetimeIndex):
                rec["train_start_time"] = df.index[abs_tr_start]
                rec["train_end_time"] = df.index[abs_tr_end - 1]
                rec["purge_start_time"] = df.index[abs_pu_start]
                rec["purge_end_time"] = df.index[abs_pu_end - 1]
                rec["held_out_start_time"] = df.index[abs_te_start]
                rec["held_out_end_time"] = df.index[abs_te_end - 1]

            records.append(rec)

        return pd.DataFrame(records)
