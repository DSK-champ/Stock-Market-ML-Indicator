"""Validation tests: column schema, candle sanity checks, and timestamp regularity.
"""

import pytest
import pandas as pd
from src.validation import validate_columns, validate_candle


def test_validate_columns_valid():
    df = pd.DataFrame(columns=[
        "open_time", "open", "high", "low", "close", "volume", "quote_volume",
        "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
    ])
    res = validate_columns(df)
    assert res["valid"] is True
    assert len(res["missing_columns"]) == 0


def test_validate_columns_missing():
    df = pd.DataFrame(columns=["open_time", "close"])
    res = validate_columns(df)
    assert res["valid"] is False
    assert "open" in res["missing_columns"]


def test_validate_candle_sanity():
    valid_candle = {
        "open_time": "2026-10-31 10:20:00+00:00",
        "open": 109500.0,
        "high": 109600.0,
        "low": 109400.0,
        "close": 109550.0,
        "volume": 12.5,
        "quote_volume": 1368750.0,
        "number_of_trades": 1500,
        "taker_buy_base_volume": 7.0,
        "taker_buy_quote_volume": 766000.0,
    }
    res = validate_candle(valid_candle)
    assert res["valid"] is True
    assert len(res["errors"]) == 0
