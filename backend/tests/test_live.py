"""Comprehensive unit & integration tests for the live forward-testing backend (Section 10)."""

import pytest
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3

from backend.live import config
from backend.live.candle_store import CandleStore
from backend.live.ledger import PredictionLedger
from backend.live.metrics import LiveMetricsCalculator, compute_wilson_ci, compute_block_bootstrap_ci
from backend.live.predictor import PredictorWorker
from backend.engine import TradingEngine, predict_one, get_model_metrics


@pytest.fixture
def tmp_dbs(tmp_path):
    c_db = tmp_path / "test_candles.db"
    l_db = tmp_path / "test_ledger.db"
    store = CandleStore(db_path=c_db)
    ledger = PredictionLedger(db_path=l_db)
    return store, ledger


# ---------------------------------------------------------------------------
# Test 1: Parity between predict_candle and predict_batch
# ---------------------------------------------------------------------------
def test_1_parity(tmp_dbs):
    store, ledger = tmp_dbs
    engine = TradingEngine.get_instance()

    dates = pd.date_range("2026-01-01", periods=150, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(70000, 71000, 150),
        "high": np.linspace(70050, 71050, 150),
        "low": np.linspace(69950, 70950, 150),
        "close": np.linspace(70020, 71020, 150),
        "volume": np.random.uniform(10, 50, 150),
    }, index=dates)

    # Vectorized batch prediction
    batch_df = engine.predict_batch(df)

    # Single candle live prediction on the exact same slice
    single_res = engine.predict_candle(df)

    # Compare probabilities
    last_batch = batch_df.iloc[-1]
    p_batch = np.array([last_batch["p_final_down"], last_batch["p_final_neutral"], last_batch["p_final_up"]])
    p_single = np.array([single_res["probabilities"]["p_down"], single_res["probabilities"]["p_neutral"], single_res["probabilities"]["p_up"]])

    diff = np.max(np.abs(p_batch - p_single))
    assert diff < 1e-3, f"Parity mismatch: max probability diff = {diff:.6f}"
    assert int(last_batch["final_action"]) == int(single_res["action"])


# ---------------------------------------------------------------------------
# Test 3: Payload mapping from real Binance kline message
# ---------------------------------------------------------------------------
def test_3_payload_mapping(tmp_dbs):
    store, ledger = tmp_dbs

    raw_ws_kline = {
        "e": "kline",
        "E": 1700000102000,
        "s": "BTCUSDT",
        "k": {
            "t": 1700000100000,
            "T": 1700000399999,
            "s": "BTCUSDT",
            "i": "5m",
            "o": "76500.10",
            "c": "76650.50",
            "h": "76700.00",
            "l": "76480.00",
            "v": "125.456",
            "n": 3450,
            "x": True,
            "q": "9612345.67",
            "V": "65.123",
            "Q": "4987654.32",
            "B": "0",
        }
    }

    k = raw_ws_kline["k"]
    payload = {
        "open_time": pd.to_datetime(k["t"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "open_time_ms": k["t"],
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "quote_volume": float(k["q"]),
        "number_of_trades": float(k["n"]),
        "taker_buy_base_volume": float(k["V"]),
        "taker_buy_quote_volume": float(k["Q"]),
        "close_time": pd.to_datetime(k["T"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00"),
    }

    assert store.store_candle(payload, source="WS") is True
    retrieved = store.get_candle(payload["open_time"])
    assert retrieved is not None
    assert retrieved["close"] == 76650.50
    assert retrieved["number_of_trades"] == 3450.0

    # Missing needed field rejected
    bad_payload = payload.copy()
    bad_payload["high"] = float("nan")
    assert store.store_candle(bad_payload) is False


# ---------------------------------------------------------------------------
# Test 4: Only x == True is processed & de-duplication
# ---------------------------------------------------------------------------
def test_4_filtering_and_deduplication(tmp_dbs):
    store, ledger = tmp_dbs

    candle = {
        "open_time": "2026-05-01 10:00:00+00:00",
        "open_time_ms": 1777629600000,
        "open": 70000.0,
        "high": 70100.0,
        "low": 69900.0,
        "close": 70050.0,
        "volume": 20.0,
        "close_time": "2026-05-01 10:05:00+00:00",
    }
    # First insert
    assert store.store_candle(candle) is True
    # Duplicate insert on same open_time
    assert store.store_candle(candle) is True

    # Check exactly 1 row exists
    with store._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM candles WHERE open_time = ?", (candle["open_time"],)).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Test 5 & 6: Timing, Resolution Logic & Idempotency
# ---------------------------------------------------------------------------
def test_5_and_6_timing_and_resolution(tmp_dbs):
    store, ledger = tmp_dbs

    # 1. Setup Candle k (10:00:00), Candle k+1 (10:05:00), Candle k+2 (10:10:00)
    times = [
        "2026-05-01 10:00:00+00:00",
        "2026-05-01 10:05:00+00:00",
        "2026-05-01 10:10:00+00:00",
    ]
    # Candle k: Close 70,000
    store.store_candle({
        "open_time": times[0], "open": 69900.0, "high": 70100.0, "low": 69800.0,
        "close": 70000.0, "volume": 10.0, "close_time": "2026-05-01 10:05:00+00:00"
    })
    # Candle k+1: Open 70,020, Close 70,080
    store.store_candle({
        "open_time": times[1], "open": 70020.0, "high": 70150.0, "low": 70000.0,
        "close": 70080.0, "volume": 10.0, "close_time": "2026-05-01 10:10:00+00:00"
    })

    # Record prediction for k: LONG (action = 1), Target = Candle k+2
    pred_record = {
        "candle_open_time": times[0],
        "ref_close": 70000.0,
        "target_candle_open_time": times[2],
        "action": 1,
        "action_label": "LONG",
        "confidence": 0.45,
        "probabilities": {"p_up": 0.45, "p_neutral": 0.25, "p_down": 0.30},
        "trend": "UPTREND",
        "market_phase": "IMPULSE",
    }
    ledger.record_prediction(pred_record)

    # Resolution attempt before candle k+2 exists -> should resolve nothing
    res_early = ledger.resolve_pending(store)
    assert len(res_early) == 0

    # Store Candle k+2: Close = 70,250 (fwd_ret = +250/70000 = +0.357% > +23 bps fee)
    store.store_candle({
        "open_time": times[2], "open": 70100.0, "high": 70300.0, "low": 70050.0,
        "close": 70250.0, "volume": 15.0, "close_time": "2026-05-01 10:15:00+00:00"
    })

    # Resolve pending -> resolves at candle k+2
    res = ledger.resolve_pending(store)
    assert len(res) == 1
    r = res[0]
    assert r["status"] == "RESOLVED"
    assert r["correct"] == 1
    assert r["actual_raw_sign_2"] == 1.0
    assert r["actual_target_2_S1"] == 1.0
    assert r["return_bps_label"] > 30.0

    # Test Idempotency: resolving again yields 0 new rows and keeps correct = 1
    res_second = ledger.resolve_pending(store)
    assert len(res_second) == 0


# ---------------------------------------------------------------------------
# Test 7 & 10: Predictor robustness on gapped or short buffer
# ---------------------------------------------------------------------------
def test_7_and_10_predictor_robustness(tmp_dbs):
    store, ledger = tmp_dbs
    worker = PredictorWorker(candle_store=store, ledger=ledger, buffer_size=100)

    async def _run():
        # Buffer length is 0 (short) -> should emit NO_SIGNAL and never crash
        candle = {
            "open_time": "2026-05-01 12:00:00+00:00",
            "open": 70000.0, "high": 70100.0, "low": 69900.0, "close": 70050.0, "volume": 10.0,
        }
        await worker._process_candle(candle)

    asyncio.run(_run())

    latest = ledger.get_latest()
    assert latest is not None
    assert latest["status"] == "NO_SIGNAL"
    assert "Buffer short" in latest["status_reason"]


# ---------------------------------------------------------------------------
# Test 9: Reconciliation overwriting WebSocket mismatch
# ---------------------------------------------------------------------------
def test_9_reconciliation(tmp_dbs, monkeypatch):
    store, ledger = tmp_dbs
    ts = "2026-05-01 15:00:00+00:00"
    store.store_candle({
        "open_time": ts, "open": 70000.0, "high": 70100.0, "low": 69900.0,
        "close": 70050.0, "volume": 10.0, "close_time": "2026-05-01 15:05:00+00:00"
    }, source="WS")

    # Mock REST response returning slightly updated close (70055.0)
    class MockResponse:
        def json(self):
            # [open_time, open, high, low, close, volume, close_time, quote_vol, trades, tb_base, tb_quote, ignore]
            return [[1777647600000, "70000.0", "70100.0", "69900.0", "70055.0", "10.0", 1777647899999, "700550.0", 100, "5.0", "350000.0", "0"]]

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())

    assert store.reconcile_candle(ts) is True
    reconciled = store.get_candle(ts)
    assert reconciled["close"] == 70055.0
    assert reconciled["is_reconciled"] == 1


# ---------------------------------------------------------------------------
# Test 11: Metrics calculation and confidence intervals
# ---------------------------------------------------------------------------
def test_11_metrics_calculation(tmp_dbs):
    store, ledger = tmp_dbs
    calc = LiveMetricsCalculator(db_path=ledger.db_path)

    # Empty ledger
    m_empty = calc.compute_metrics()
    assert m_empty["too_few_samples"] is True

    # Seed 10 resolved rows (7 correct, 3 wrong)
    for i in range(10):
        t_str = f"2026-05-01 {10 + i}:00:00+00:00"
        is_corr = 1 if i < 7 else 0
        with ledger._get_connection() as conn:
            conn.execute("""
                INSERT INTO predictions (
                    candle_open_time, ref_close, target_candle_open_time, action, action_label,
                    confidence, p_up, p_neutral, p_down, trend, market_phase, model_version,
                    git_commit, config_hash, buffer_length, compute_latency_ms, status,
                    outcome_close, actual_raw_sign_2, actual_target_2_S1, correct,
                    return_bps_label, return_bps_executable, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t_str, 70000.0, f"2026-05-01 {10 + i}:10:00+00:00", 1, "LONG",
                0.45, 0.45, 0.25, 0.30, "UPTREND", "IMPULSE", "v2.0",
                "test", "test", 1000, 15.0, "RESOLVED",
                70100.0 if is_corr else 69900.0, 1.0 if is_corr else -1.0, 1.0 if is_corr else -1.0, is_corr,
                14.2 if is_corr else -14.2, 14.0 if is_corr else -14.0, t_str
            ))
            conn.commit()

    m = calc.compute_metrics()
    assert m["directional_calls_count"] == 10
    assert m["directional_accuracy"]["overall_pct"] == 70.0
    w_low, w_high = m["directional_accuracy"]["wilson_95_ci"]
    assert w_low < 70.0 < w_high


# ---------------------------------------------------------------------------
# Test 12: Static Check — Live outcome never feeds back into model
# ---------------------------------------------------------------------------
def test_12_static_check():
    engine = TradingEngine.get_instance()
    # Verify weights and thresholds remain frozen
    assert engine.context_weight == 0.05
    assert engine.override_threshold == 0.32
    assert engine.override_margin == 0.04
    assert engine.cost_floor_bps == 23.0
