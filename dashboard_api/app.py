"""FastAPI application for the Trader Dashboard API (Port 8001).

A thin read-only wrapper around the project's data, model prediction bridge,
and visual band enrichment.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dashboard_api.data_service import DataService

app = FastAPI(
    title="Trader Dashboard API",
    description="Thin read-only service for dashboard charts, indicators, and replay testing.",
    version="1.0.0",
)

# Enable CORS for localhost / browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService.get_instance()


@app.get("/dash/config")
def get_config() -> Dict[str, Any]:
    """Return central system configuration, vocabulary, and benchmarks."""
    return data_service.get_config()


@app.get("/dash/candles/history")
def get_historical_candles(limit: int = Query(2000, ge=10, le=10000)) -> List[Dict[str, Any]]:
    """Retrieve historical candles from clean.csv / master data."""
    return data_service.get_historical_candles(limit=limit)


@app.get("/dash/live/candles")
def get_live_candles(limit: int = Query(1000, ge=1, le=5000)) -> List[Dict[str, Any]]:
    """Retrieve recent closed candles from candles.db."""
    return data_service.get_live_candles(limit=limit)


@app.get("/dash/candles/sync")
def sync_candles() -> Dict[str, Any]:
    """Audit SQLite candles for skipped intervals and repair every gap from Binance."""
    from backend.live.candle_store import CandleStore
    store = CandleStore()
    return store.validate_and_repair_continuity()


@app.get("/dash/live/predictions")
def get_live_predictions(limit: int = Query(200, ge=1, le=1000)) -> List[Dict[str, Any]]:
    """Retrieve recent predictions from live_ledger.db enriched with bands."""
    return data_service.get_live_predictions(limit=limit)


@app.get("/dash/live/predict_latest")
def predict_latest_live() -> Optional[Dict[str, Any]]:
    """Execute prediction on the latest closed candle in candles.db or clean.csv."""
    return data_service.predict_latest_candle()


@app.get("/dash/features/latest")
def get_latest_features() -> Dict[str, Any]:
    """Retrieve full 136 engineered feature values for the most recent closed candle."""
    return data_service.get_latest_features()


@app.get("/dash/overview")
def get_overview() -> List[Dict[str, Any]]:
    """Retrieve hourly downsampled history with partition tags for the overview timeline."""
    return data_service.get_overview()


@app.get("/dash/replay/range")
def get_replay_range() -> Dict[str, Any]:
    """Retrieve TEST split boundaries for replay validation."""
    return data_service.get_replay_range()


@app.get("/dash/replay/candles")
def get_replay_candles(
    from_time: str = Query(..., alias="from"),
    to_time: str = Query(..., alias="to"),
) -> List[Dict[str, Any]]:
    """Retrieve market candles strictly within the TEST range."""
    try:
        return data_service.get_replay_candles(from_time=from_time, to_time=to_time)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dash/replay/predict")
def predict_replay(ts: str = Query(..., description="Candle open_time ISO")) -> Dict[str, Any]:
    """Execute prediction for a single replay candle with 3,000 lookback."""
    try:
        return data_service.predict_replay_candle(ts=ts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/dash/replay/predict_range")
def predict_replay_range(
    from_ts: str = Query(..., alias="from"),
    count: int = Query(5, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """Batch predict a sequence of replay candles for accelerated playback."""
    try:
        return data_service.predict_replay_range(from_ts=from_ts, count=count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/dash/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

