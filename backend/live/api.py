"""Async REST and WebSocket API server for live indicators and metrics."""
"""Async REST and WebSocket API server for live indicators and metrics."""

from typing import Dict, Any, Optional, Set, List
import asyncio
import json
import time
import urllib.parse
from datetime import datetime, timezone
import pandas as pd
import websockets
from websockets.asyncio.server import serve
from websockets.http11 import Response, Headers

from backend.live import config
from backend.live.candle_store import CandleStore
from backend.live.ledger import PredictionLedger
from backend.live.metrics import LiveMetricsCalculator
from backend.live.predictor import PredictorWorker
from backend.engine import predict_one, get_model_metrics, TradingEngine


class LiveApiServer:
    """Consolidated HTTP and WebSocket API server for dashboard integration."""

    def __init__(
        self,
        candle_store: CandleStore,
        ledger: PredictionLedger,
        predictor: PredictorWorker,
        stream_state_provider: Optional[Any] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.candle_store = candle_store
        self.ledger = ledger
        self.predictor = predictor
        self.stream_state_provider = stream_state_provider
        self.host = host
        self.port = port
        self.metrics_calc = LiveMetricsCalculator(db_path=ledger.db_path)

        self.ws_clients: Set[Any] = set()
        self.start_time = time.time()
        self._server = None
        self._running = False

        # Register event callback from predictor
        self.predictor.add_callback(self.broadcast_event)

    def _enrich_pred(self, pred: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not pred:
            return None
        enriched = dict(pred)
        if not enriched.get("ref_close") and enriched.get("candle_open_time"):
            c = self.candle_store.get_candle(str(enriched["candle_open_time"]))
            if c and c.get("close"):
                enriched["ref_close"] = float(c["close"])
        enriched["smart_price"] = None
        enriched["upper"] = None
        enriched["lower"] = None
        enriched["atr14"] = None
        return enriched

    def broadcast_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Broadcast live event to all connected WebSocket clients."""
        if not self.ws_clients:
            return

        if event_type in ["new_prediction", "latest_prediction"]:
            payload = self._enrich_pred(payload) or payload

        msg = json.dumps({"event": event_type, "data": payload, "timestamp": datetime.now(timezone.utc).isoformat()})
        asyncio.create_task(self._send_to_all(msg))

    async def _send_to_all(self, msg: str) -> None:
        disconnected = set()
        for client in self.ws_clients:
            try:
                await client.send(msg)
            except Exception:
                disconnected.add(client)
        self.ws_clients.difference_update(disconnected)

    async def handle_websocket(self, websocket: Any) -> None:
        """Handle incoming WebSocket connection on /ws/live."""
        self.ws_clients.add(websocket)
        print(f"[WS-API] Client connected ({len(self.ws_clients)} active).")
        try:
            # Send initial state snapshot
            initial_status = self.get_status_dict()
            await websocket.send(json.dumps({"event": "initial_status", "data": initial_status}))

            latest_pred = self._enrich_pred(self.ledger.get_latest())
            if latest_pred:
                await websocket.send(json.dumps({"event": "latest_prediction", "data": latest_pred}))

            # Keep connection alive receiving any client messages/pings
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self.ws_clients.discard(websocket)
            print(f"[WS-API] Client disconnected ({len(self.ws_clients)} remaining).")

    def get_status_dict(self) -> Dict[str, Any]:
        """Compile live health and streaming status."""
        uptime = round(time.time() - self.start_time, 1)
        recent_df = self.candle_store.get_recent_candles(limit=config.BUFFER_SIZE)

        state = "LIVE"
        last_closed = None
        latency_ms = 0.0
        reconnect_count = 0
        forming_price = None

        if self.stream_state_provider:
            state = getattr(self.stream_state_provider, "state", "UNKNOWN")
            last_closed = getattr(self.stream_state_provider, "last_closed_candle_time", None)
            latency_ms = getattr(self.stream_state_provider, "measured_latency_ms", 0.0)
            reconnect_count = getattr(self.stream_state_provider, "reconnect_count", 0)
            forming = getattr(self.stream_state_provider, "forming_candle", None)
            if forming:
                forming_price = forming.get("close")

        return {
            "state": state,
            "mode": "LIVE",
            "uptime_seconds": uptime,
            "buffer_length": len(recent_df),
            "buffer_target": config.BUFFER_SIZE,
            "last_closed_candle_time": last_closed,
            "reconnect_count": reconnect_count,
            "latency_ms": round(latency_ms, 2),
            "forming_candle_close": forming_price,
            "horizon_candles": config.HORIZON,
            "horizon_minutes": config.HORIZON * config.INTERVAL_MINUTES,
            "cost_floor_bps": config.COST_FLOOR_BPS,
        }

    async def process_http_request(self, connection: Any, request: Any) -> Optional[Response]:
        """Intercept and respond to HTTP REST requests; upgrade /ws/live."""
        raw_path = getattr(request, "path", "/")
        parsed = urllib.parse.urlparse(raw_path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Allow WebSocket upgrade for /ws/live or /ws
        if path in ["/ws/live", "/ws"]:
            return None

        cors_headers = Headers([
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "*"),
        ])

        if getattr(request, "method", "GET") == "OPTIONS":
            return Response(204, "No Content", cors_headers, b"")

        # Route matching
        try:
            if path == "/api/live/status":
                data = self.get_status_dict()
                return Response(200, "OK", cors_headers, json.dumps(data).encode("utf-8"))

            elif path == "/api/live/latest":
                latest = self._enrich_pred(self.ledger.get_latest())
                resp_data: Dict[str, Any] = {"latest_prediction": latest}
                if latest:
                    pred_time = pd.to_datetime(latest["candle_open_time"], utc=True)
                    resolve_time = pred_time + pd.Timedelta(minutes=10)
                    now_time = pd.to_datetime(datetime.now(timezone.utc), utc=True)
                    countdown = max(0, int((resolve_time - now_time).total_seconds()))
                    resp_data["countdown_seconds_to_resolution"] = countdown
                return Response(200, "OK", cors_headers, json.dumps(resp_data).encode("utf-8"))

            elif path == "/api/live/predictions":
                limit = int(query.get("limit", [50])[0])
                status = query.get("status", [None])[0]
                since = query.get("since", [None])[0]
                preds = self.ledger.get_predictions(limit=limit, status=status, since=since)
                return Response(200, "OK", cors_headers, json.dumps({"count": len(preds), "predictions": preds}).encode("utf-8"))

            elif path == "/api/live/sync":
                res = self.candle_store.validate_and_repair_continuity()
                return Response(200, "OK", cors_headers, json.dumps(res).encode("utf-8"))

            elif path == "/api/live/candles":
                limit = int(query.get("limit", [1000])[0])
                df_recent = self.candle_store.get_recent_candles(limit=limit)
                candles_list = []
                for ts, row in df_recent.iterrows():
                    candles_list.append({
                        "time": int(ts.timestamp()),
                        "open_time": ts.isoformat(),
                        "open": round(float(row["open"]), 2),
                        "high": round(float(row["high"]), 2),
                        "low": round(float(row["low"]), 2),
                        "close": round(float(row["close"]), 2),
                        "volume": round(float(row["volume"]), 4),
                    })
                return Response(200, "OK", cors_headers, json.dumps(candles_list).encode("utf-8"))

            elif path == "/api/live/metrics":
                metrics = self.metrics_calc.compute_metrics()
                return Response(200, "OK", cors_headers, json.dumps(metrics).encode("utf-8"))

            elif path == "/api/metrics":
                # Existing project scorecard endpoint
                m = get_model_metrics()
                return Response(200, "OK", cors_headers, json.dumps(m).encode("utf-8"))

            elif path == "/api/backtest":
                engine = TradingEngine.get_instance()
                b_res = engine.simulate_backtest(initial_capital=10000.0, fee_bps=config.COST_FLOOR_BPS)
                return Response(200, "OK", cors_headers, json.dumps(b_res).encode("utf-8"))

            else:
                return Response(404, "Not Found", cors_headers, json.dumps({"error": f"Path '{path}' not found."}).encode("utf-8"))

        except Exception as e:
            return Response(500, "Internal Server Error", cors_headers, json.dumps({"error": str(e)}).encode("utf-8"))

    async def start(self) -> None:
        """Start combined HTTP and WebSocket server."""
        if self._running:
            return
        self._running = True

        self._server = await serve(
            self.handle_websocket,
            self.host,
            self.port,
            process_request=self.process_http_request,
        )
        print(f"[OK] LiveApiServer running at http://{self.host}:{self.port} (WebSocket at ws://{self.host}:{self.port}/ws/live)")

    async def stop(self) -> None:
        """Stop server cleanly."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        print("[OK] LiveApiServer stopped.")

