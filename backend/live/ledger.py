"""Prediction ledger and automated resolution engine."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from backend.live import config
from backend.live.candle_store import CandleStore


class PredictionLedger:
    """Manages prediction recording, state transitions, and automated resolution."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or config.LIVE_LEDGER_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    candle_open_time TEXT PRIMARY KEY,
                    ref_close REAL NOT NULL,
                    target_candle_open_time TEXT NOT NULL,
                    action INTEGER NOT NULL,
                    action_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    p_up REAL NOT NULL,
                    p_neutral REAL NOT NULL,
                    p_down REAL NOT NULL,
                    trend TEXT NOT NULL,
                    market_phase TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    git_commit TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    buffer_length INTEGER NOT NULL,
                    compute_latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    status_reason TEXT,
                    outcome_close REAL,
                    outcome_open REAL,
                    actual_raw_sign_2 REAL,
                    actual_target_2_S1 REAL,
                    correct INTEGER,
                    return_bps_label REAL,
                    return_bps_executable REAL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_target_time ON predictions(target_candle_open_time)")
            conn.commit()

    def record_prediction(self, pred: Dict[str, Any]) -> bool:
        """Record an emitted prediction into the ledger (idempotent primary key)."""
        open_time = pred["candle_open_time"]
        now_str = datetime.now(timezone.utc).isoformat()

        # If action is 0 (Neutral), initial status is NO_CALL once resolved, or PENDING until outcome candle
        # Per specification: "action NEUTRAL: status NO_CALL, excluded from directional accuracy"
        initial_status = "NO_SIGNAL" if pred.get("status") == "NO_SIGNAL" else "PENDING"

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO predictions (
                    candle_open_time, ref_close, target_candle_open_time, action, action_label,
                    confidence, p_up, p_neutral, p_down, trend, market_phase, model_version,
                    git_commit, config_hash, buffer_length, compute_latency_ms, status, status_reason,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                open_time,
                float(pred["ref_close"]),
                str(pred["target_candle_open_time"]),
                int(pred["action"]),
                str(pred["action_label"]),
                float(pred["confidence"]),
                float(pred["probabilities"]["p_up"]),
                float(pred["probabilities"]["p_neutral"]),
                float(pred["probabilities"]["p_down"]),
                str(pred.get("trend", "uncertain")),
                str(pred.get("market_phase", "uncertain")),
                str(pred.get("model_version", "v2.0")),
                str(pred.get("git_commit", "unknown")),
                str(pred.get("config_hash", "default")),
                int(pred.get("buffer_length", 0)),
                float(pred.get("compute_latency_ms", 0.0)),
                initial_status,
                pred.get("status_reason"),
                now_str,
            ))
            conn.commit()
        return True

    def resolve_pending(self, candle_store: CandleStore) -> List[Dict[str, Any]]:
        """Scan and resolve all PENDING predictions whose outcome candle k+2 now exists."""
        resolved: List[Dict[str, Any]] = []

        with self._get_connection() as conn:
            pending_rows = conn.execute("""
                SELECT * FROM predictions WHERE status = 'PENDING' ORDER BY candle_open_time ASC
            """).fetchall()

        if not pending_rows:
            return resolved

        for row in pending_rows:
            pred = dict(row)
            target_time = pred["target_candle_open_time"]
            outcome_candle = candle_store.get_candle(target_time)

            if not outcome_candle:
                continue

            # Also retrieve candle k+1 for executable entry return
            pred_dt = pd.to_datetime(pred["candle_open_time"], utc=True)
            k_plus_1_dt = pred_dt + pd.Timedelta(minutes=5)
            k_plus_1_time = k_plus_1_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
            candle_k1 = candle_store.get_candle(k_plus_1_time)

            # Calculate returns
            ref_close = float(pred["ref_close"])
            outcome_close = float(outcome_candle["close"])
            fwd_ret = (outcome_close / ref_close) - 1.0

            if candle_k1:
                k1_open = float(candle_k1["open"])
                fwd_ret_exec = (outcome_close / k1_open) - 1.0
            else:
                fwd_ret_exec = fwd_ret

            return_bps_label = fwd_ret * 10000.0
            return_bps_exec = fwd_ret_exec * 10000.0

            # Derive Ground Truth Labels using Section 0(c) specs
            if fwd_ret > 0.0:
                raw_sign_2 = 1.0
            elif fwd_ret < 0.0:
                raw_sign_2 = -1.0
            else:
                raw_sign_2 = 0.0

            cost_c = config.COST_FLOOR_BPS / 10000.0
            if fwd_ret > cost_c:
                target_2_s1 = 1.0
            elif fwd_ret < -cost_c:
                target_2_s1 = -1.0
            else:
                target_2_s1 = 0.0

            # Determine Verdict
            action = int(pred["action"])
            if action in [1, -1]:
                if raw_sign_2 == 0.0:
                    status = "VOID"
                    correct = None
                    reason = "Zero raw forward return"
                elif action == int(raw_sign_2):
                    status = "RESOLVED"
                    correct = 1
                    reason = "Direction matched raw_sign_2"
                else:
                    status = "RESOLVED"
                    correct = 0
                    reason = "Direction opposite of raw_sign_2"
            else:
                # NEUTRAL call
                status = "NO_CALL"
                correct = None
                reason = "Neutral call (no active directional call)"

            now_str = datetime.now(timezone.utc).isoformat()

            # Update SQLite
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE predictions SET
                        status = ?,
                        status_reason = ?,
                        outcome_close = ?,
                        outcome_open = ?,
                        actual_raw_sign_2 = ?,
                        actual_target_2_S1 = ?,
                        correct = ?,
                        return_bps_label = ?,
                        return_bps_executable = ?,
                        resolved_at = ?
                    WHERE candle_open_time = ?
                """, (
                    status,
                    reason,
                    outcome_close,
                    float(outcome_candle["open"]),
                    raw_sign_2,
                    target_2_s1,
                    correct,
                    return_bps_label,
                    return_bps_exec,
                    now_str,
                    pred["candle_open_time"],
                ))
                conn.commit()

            pred.update({
                "status": status,
                "status_reason": reason,
                "outcome_close": outcome_close,
                "actual_raw_sign_2": raw_sign_2,
                "actual_target_2_S1": target_2_s1,
                "correct": correct,
                "return_bps_label": return_bps_label,
                "return_bps_executable": return_bps_exec,
                "resolved_at": now_str,
            })
            resolved.append(pred)

        return resolved

    def get_predictions(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query predictions with optional filtering."""
        query = "SELECT * FROM predictions WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if since:
            query += " AND candle_open_time >= ?"
            params.append(since)

        query += " ORDER BY candle_open_time DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM predictions ORDER BY candle_open_time DESC LIMIT 1").fetchone()
            return dict(row) if row else None

