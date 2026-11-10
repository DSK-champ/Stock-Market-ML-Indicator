"""Extraction tests: unit tests with local mocks and integration test for live Binance API.
"""

import pytest
from src import extraction, config


def test_extraction_constants():
    """Verify endpoint and output columns match configuration."""
    assert config.BINANCE_BASE_URL.startswith("https://")
    assert "open_time" in config.OUTPUT_COLUMNS


def test_interval_to_ms():
    """Verify interval conversion logic."""
    assert extraction.interval_to_ms("5m") == 5 * 60 * 1000
    assert extraction.interval_to_ms("1h") == 60 * 60 * 1000
    assert extraction.interval_to_ms("1d") == 24 * 60 * 60 * 1000


@pytest.mark.integration
def test_live_binance_fetch():
    """Integration test: fetch a small sample of candles from Binance.
    
    Skipped by default to prevent network flakiness. Run with:
        pytest -m integration
    """
    df = extraction.fetch_historical_data(
        symbol=config.SYMBOL,
        interval=config.TIMEFRAME,
        days=0.1,
        limit=10,
    )
    assert not df.empty
    assert len(df) <= 10
    assert "close" in df.columns
