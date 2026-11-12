# Backend Live Forward Testing & API Service

This directory contains the live forward testing infrastructure, real-time WebSocket ingestion, SQLite persistence, and API serving layer for the BTCUSDT 5-Minute AI Trading Indicator.

For the full end-to-end architectural specification, database schemas, and frontend communication protocols, see:
👉 **[`BACKEND_ARCHITECTURE.md`](../BACKEND_ARCHITECTURE.md)**

## Quick Reference

- **Live Stream Entry Point**: `python -m backend.live.run --mode live`
- **Replay Mode**: `python -m backend.live.run --mode replay --speed 60.0`
- **Unit & Parity Tests**: `pytest backend/tests/test_live.py`
- **HTTP REST Base**: `http://localhost:8000/api/live/*`
- **WebSocket Feed**: `ws://localhost:8000/ws/live`
- **Databases**:
  - `backend/data/candles.db` (5-minute closed OHLCV candle cache)
  - `backend/data/live_ledger.db` (Inference predictions and 10-minute resolved outcomes)
  - `backend/data/archive/` (Automated historical run backups)
