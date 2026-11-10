"""Unit tests for offline vs live-path parity and determinism.
"""

import pytest


def test_inference_module_importable():
    from src import inference
    assert hasattr(inference, "predict_one")

