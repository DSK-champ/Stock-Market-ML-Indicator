"""Benchmark baselines: always-up, majority-class, RSI rule, previous-move rule.

Phase 4 will implement and benchmark these on validation data.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class BaselineRuleModel:
    """Base interface for naive rule-based benchmarks."""
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(df))

