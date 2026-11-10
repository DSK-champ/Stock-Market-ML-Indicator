"""Utility functions: seeding, logging, timer helpers, and causal volatility estimation.

All functions in this module must be strictly causal — no function may read
information from time t+k (k > 0) when computing an output for time t.
"""

import os
import random
import logging

import numpy as np
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds across all libraries for deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def setup_logger(name: str = "ml_indicator", level: int = logging.INFO) -> logging.Logger:
    """Set up and return a standardized console/file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def rolling_sigma(returns: pd.Series, window: int = 288) -> pd.Series:
    """Causal rolling standard deviation of log returns.

    Uses only PAST data at every time step — strictly causal.

    Parameters
    ----------
    returns : pd.Series
        Close-to-close log returns (or any return series) aligned to the candle index.
    window : int
        Look-back window in candles. Default 288 = 24 hours of 5-minute bars.
        [ASSUMPTION] Window of 288 candles (24 h) chosen to capture one full trading day
        of intraday volatility without introducing multi-day regime drift.
        Source: standard intraday volatility literature (Andersen & Bollerslev, 1998).

    Returns
    -------
    pd.Series
        Rolling standard deviation aligned to the same index.
        Uses min_periods=2 so the first (window-1) values are computed from whatever
        data is available; the very first element (index 0) is NaN because std requires
        at least 2 observations.

    Causality guarantee
    -------------------
    ``pandas.Series.rolling(window, min_periods=2).std()`` at position t uses only
    the values in [t-window+1 .. t], which are all in the past relative to t.
    Altering any value at index t+k (k > 0) must NOT change the output at index t.
    This is verified by ``tests/test_labels.py::test_v_rule_sigma_causal``.
    """
    return returns.rolling(window=window, min_periods=2).std()
