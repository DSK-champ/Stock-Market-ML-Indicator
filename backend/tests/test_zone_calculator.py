"""Unit tests for pure Model-Informed Trade Zones calculator."""

import pytest
import pandas as pd
import numpy as np

from backend.live.zone_calculator import (
    calculate_trade_zone,
    evaluate_zone_resolution,
    compute_garman_klass_volatility,
    find_dow_swings,
    classify_regime,
    compute_agreement_score,
)
from backend.live.zone_config import DEFAULT_ZONE_CONFIG


@pytest.fixture
def sample_candles():
    """Synthetic 20-candle OHLCV series."""
    base_price = 90000.0
    candles = []
    for i in range(25):
        drift = i * 20.0
        c_open = base_price + drift
        c_high = c_open + 80.0
        c_low = c_open - 60.0
        c_close = c_open + 30.0
        candles.append({
            "open_time": f"2026-11-16 10:{i*5:02d}:00+00:00",
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
            "volume": 150.0,
        })
    return candles


def test_calculate_zone_long(sample_candles):
    """Test standard LONG signal calculation."""
    pred = {
        "candle_open_time": "2026-11-16 12:00:00+00:00",
        "target_candle_open_time": "2026-11-16 12:10:00+00:00",
        "action": 1,
        "action_label": "LONG",
        "ref_close": 91000.0,
        "confidence": 0.65,
        "atr14": 250.0,
        "probabilities": {"p_up": 0.65, "p_neutral": 0.20, "p_down": 0.15},
        "base_model_votes": {
            "binary_lr_p_up": 0.68,
            "binary_rf_p_up": 0.62,
            "binary_gb_p_up": 0.70,
            "tri_lr_p_up": 0.55,
            "tri_lr_p_down": 0.15,
            "tri_rf_p_up": 0.58,
            "tri_rf_p_down": 0.18,
            "tri_gb_p_up": 0.60,
            "tri_gb_p_down": 0.14,
        },
        "technical_analysis": {
            "trend": "UPTREND",
            "trend_score": 0.45,
            "ta_rsi": 0.60,
            "ta_ema50": 0.40,
            "ta_align": 0.50,
            "ta_loc": 0.30,
            "ta_rule_vote": 0.40,
            "ta_trend": 0.50,
        },
    }

    zone = calculate_trade_zone(pred, sample_candles)

    assert zone["direction"] == "LONG"
    assert zone["status"] == "ACTIVE"
    # Entry zone below or at ref_close
    assert zone["entry_max"] == 91000.0
    assert zone["entry_min"] < zone["entry_max"]
    assert zone["entry_mid"] == (zone["entry_min"] + zone["entry_max"]) / 2.0
    # Target above entry
    assert zone["target_price"] > zone["entry_mid"]
    # Stop below entry
    assert zone["stop_price"] < zone["entry_min"]
    # Risk to reward > 0
    assert zone["risk_reward"] > 0
    # Agreement score high
    assert zone["agreement_score"] >= 80.0
    assert not zone["has_conflicts"]
    assert zone["regime"] == "trending"


def test_calculate_zone_short(sample_candles):
    """Test standard SHORT signal calculation."""
    pred = {
        "candle_open_time": "2026-11-16 12:00:00+00:00",
        "target_candle_open_time": "2026-11-16 12:10:00+00:00",
        "action": -1,
        "action_label": "SHORT",
        "ref_close": 91000.0,
        "confidence": 0.62,
        "atr14": 260.0,
        "probabilities": {"p_up": 0.15, "p_neutral": 0.23, "p_down": 0.62},
        "base_model_votes": {
            "binary_lr_p_up": 0.30,
            "binary_rf_p_up": 0.28,
            "binary_gb_p_up": 0.25,
            "tri_lr_p_up": 0.18,
            "tri_lr_p_down": 0.55,
            "tri_rf_p_up": 0.16,
            "tri_rf_p_down": 0.58,
            "tri_gb_p_up": 0.20,
            "tri_gb_p_down": 0.60,
        },
        "technical_analysis": {
            "trend": "DOWNTREND",
            "trend_score": -0.42,
            "ta_rsi": -0.55,
            "ta_ema50": -0.38,
            "ta_align": -0.45,
            "ta_loc": -0.25,
            "ta_rule_vote": -0.35,
            "ta_trend": -0.42,
        },
    }

    zone = calculate_trade_zone(pred, sample_candles)

    assert zone["direction"] == "SHORT"
    assert zone["status"] == "ACTIVE"
    # Entry zone at or above ref_close
    assert zone["entry_min"] == 91000.0
    assert zone["entry_max"] > zone["entry_min"]
    # Target below entry
    assert zone["target_price"] < zone["entry_mid"]
    # Stop above entry
    assert zone["stop_price"] > zone["entry_max"]
    assert zone["risk_reward"] > 0
    assert zone["agreement_score"] >= 80.0
    assert not zone["has_conflicts"]


def test_calculate_zone_neutral(sample_candles):
    """Test NEUTRAL case (no entry zone, grey no-trade band)."""
    pred = {
        "candle_open_time": "2026-11-16 12:00:00+00:00",
        "target_candle_open_time": "2026-11-16 12:10:00+00:00",
        "action": 0,
        "action_label": "NEUTRAL",
        "ref_close": 90500.0,
        "confidence": 0.34,
        "atr14": 200.0,
        "probabilities": {"p_up": 0.33, "p_neutral": 0.34, "p_down": 0.33},
        "technical_analysis": {"trend": "ranging", "trend_score": 0.05},
    }

    zone = calculate_trade_zone(pred, sample_candles)

    assert zone["direction"] == "NEUTRAL"
    assert zone["status"] == "NEUTRAL_BAND"
    assert zone["entry_min"] == 90500.0
    assert zone["entry_max"] == 90500.0
    assert zone["target_price"] == 90500.0
    assert zone["stop_price"] == 90500.0
    assert zone["risk_reward"] == 0.0
    assert zone["is_low_edge"] is False


def test_calculate_zone_low_edge(sample_candles):
    """Test low edge flag when expected move cannot clear friction."""
    # Force tiny margin |p_up - p_down| and low scale
    pred = {
        "candle_open_time": "2026-11-16 12:00:00+00:00",
        "action": 1,
        "action_label": "LONG",
        "ref_close": 90000.0,
        "confidence": 0.36,
        "probabilities": {"p_up": 0.36, "p_neutral": 0.34, "p_down": 0.30},
        "technical_analysis": {"trend": "ranging", "trend_score": 0.1},
    }
    # Override scale to tiny value so expected_move < 23 bps friction
    zone = calculate_trade_zone(pred, sample_candles, config_override={"scale": 0.10})

    assert zone["direction"] == "LONG"
    assert zone["is_low_edge"] is True


def test_calculate_zone_conflicting_signals(sample_candles):
    """Test conflicting signals when sub-models disagree (<50% agreement)."""
    pred = {
        "candle_open_time": "2026-11-16 12:00:00+00:00",
        "action": 1,
        "action_label": "LONG",
        "ref_close": 90000.0,
        "confidence": 0.52,
        "probabilities": {"p_up": 0.52, "p_neutral": 0.20, "p_down": 0.28},
        # Base models disagree: all bearish votes
        "base_model_votes": {
            "binary_lr_p_up": 0.35,
            "binary_rf_p_up": 0.38,
            "binary_gb_p_up": 0.40,
            "tri_lr_p_up": 0.20,
            "tri_lr_p_down": 0.55,
            "tri_rf_p_up": 0.22,
            "tri_rf_p_down": 0.52,
            "tri_gb_p_up": 0.25,
            "tri_gb_p_down": 0.50,
        },
        "technical_analysis": {
            "trend": "DOWNTREND",
            "trend_score": -0.30,
            "ta_rsi": -0.40,
            "ta_ema50": -0.20,
            "ta_align": -0.30,
            "ta_loc": -0.20,
            "ta_rule_vote": -0.30,
            "ta_trend": -0.30,
        },
    }

    zone = calculate_trade_zone(pred, sample_candles)

    assert zone["direction"] == "LONG"
    assert zone["agreement_score"] < 50.0
    assert zone["has_conflicts"] is True


def test_zone_resolutions():
    """Test target hit, stop loss, flip exit, and expiry resolution rules."""
    zone = {
        "zone_id": "test_zone",
        "direction": "LONG",
        "status": "ACTIVE",
        "entry_mid": 90000.0,
        "target_price": 90500.0,
        "stop_price": 89500.0,
        "multipliers": {"cost_floor_bps": 23.0},
    }

    # 1. Target Hit
    candle_hit = {"open_time": "2026-11-16 12:05:00", "high": 90600.0, "low": 89900.0, "close": 90550.0}
    res_hit = evaluate_zone_resolution(zone, candle_hit)
    assert res_hit["status"] == "TARGET_HIT"
    assert res_hit["exit_price"] == 90500.0
    assert res_hit["net_bps"] > 0

    # 2. Stopped Out
    candle_stop = {"open_time": "2026-11-16 12:05:00", "high": 90100.0, "low": 89400.0, "close": 89450.0}
    res_stop = evaluate_zone_resolution(zone, candle_stop)
    assert res_stop["status"] == "STOPPED_OUT"
    assert res_stop["exit_price"] == 89500.0
    assert res_stop["net_bps"] < 0

    # 3. Flipped Exit
    candle_normal = {"open_time": "2026-11-16 12:05:00", "high": 90200.0, "low": 89800.0, "close": 90050.0}
    res_flip = evaluate_zone_resolution(zone, candle_normal, opposing_action_emitted=True)
    assert res_flip["status"] == "FLIPPED_EXIT"

    # 4. Low Confidence Exit
    res_low_conf = evaluate_zone_resolution(zone, candle_normal, new_confidence=0.25, theta_dir=0.32)
    assert res_low_conf["status"] == "FLIPPED_EXIT"

    # 5. Expiry at t+2
    res_exp = evaluate_zone_resolution(zone, candle_normal, is_expiry_candle=True)
    assert res_exp["status"] == "EXPIRED"
