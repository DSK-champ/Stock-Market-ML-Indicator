"""Unit tests for the Unified Production TradingEngine."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.trading_engine import TradingEngine, get_engine, predict_one, get_model_metrics


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def test_engine_initialization(engine):
    """Test that all models and artifacts are loaded into memory."""
    assert engine is not None
    assert len(engine.binary_models) == 3
    assert len(engine.three_class_models) == 3
    assert engine.meta_model is not None
    assert "binary_lr" in engine.binary_models
    assert "three_class_rf" in engine.three_class_models


def test_get_metrics(engine):
    """Test metrics retrieval for the dashboard."""
    metrics = engine.get_metrics()
    assert "overall_performance" in metrics
    assert "per_class_metrics" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["overall_performance"]["macro_f1"] > 0.40
    assert metrics["overall_performance"]["coverage_pct"] > 60.0


def test_predict_candle_output_structure(engine):
    """Test single candle prediction with synthetic OHLCV history."""
    dates = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    df_synthetic = pd.DataFrame({
        "open": np.linspace(70000, 70500, 100),
        "high": np.linspace(70050, 70550, 100),
        "low": np.linspace(69950, 70450, 100),
        "close": np.linspace(70020, 70520, 100),
        "volume": np.random.uniform(10, 50, 100),
    }, index=dates)

    res = predict_one(df_synthetic)
    assert "action" in res
    assert res["action"] in [-1, 0, 1]
    assert "action_label" in res
    assert res["action_label"] in ["SHORT", "NEUTRAL", "LONG"]
    assert "probabilities" in res
    probs = res["probabilities"]
    prob_sum = probs["p_up"] + probs["p_neutral"] + probs["p_down"]
    assert abs(prob_sum - 1.0) < 1e-3
    assert "technical_analysis" in res
    assert "base_model_votes" in res


def test_backtest_simulation(engine):
    """Test financial backtest simulation."""
    res = engine.simulate_backtest(initial_capital=10000.0)
    assert "total_trades" in res
    assert res["total_trades"] > 0
    assert "fee_deducted_per_trade_bps" in res
    assert res["fee_deducted_per_trade_bps"] == 23.0
