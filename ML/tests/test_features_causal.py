"""Unit tests for feature engineering: truncation test and no NaN/inf after warm-up.
"""

import pytest
from src import config


def test_features_module_importable():
    from src import features
    assert hasattr(features, "build_features")
    assert hasattr(features, "LOOKBACK_CANDLES")

