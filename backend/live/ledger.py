"""Prediction ledger and automated resolution engine."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3
import json
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
                    n_imputed_features INTEGER DEFAULT 0,
                    imputed_feature_names TEXT DEFAULT '[]',
                    exchange_close_time TEXT,
                    exchange_event_time_ms INTEGER,
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

            # Table for run lifecycle tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_start (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    git_commit TEXT NOT NULL,
                    git_tag TEXT,
                    buffer_size INTEGER NOT NULL,
                    config_hash TEXT NOT NULL,
                    clean_repo_asserted INTEGER NOT NULL
                )
            """)

            # Migration check for existing databases
            cols = [col[1] for col in conn.execute("PRAGMA table_info(predictions)").fetchall()]
            if "n_imputed_features" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN n_imputed_features INTEGER DEFAULT 0")
            if "imputed_feature_names" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN imputed_feature_names TEXT DEFAULT '[]'")
            if "exchange_close_time" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN exchange_close_time TEXT")
            if "exchange_event_time_ms" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN exchange_event_time_ms INTEGER")

            # Dedicated Model-Informed Trade Zones table (Additive, strictly zero repainting)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_zones (
                    zone_id TEXT PRIMARY KEY,
                    candle_open_time TEXT NOT NULL,
                    target_candle_open_time TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    ref_close REAL NOT NULL,
                    entry_min REAL NOT NULL,
                    entry_max REAL NOT NULL,
                    entry_mid REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    risk_reward REAL NOT NULL,
                    p_up REAL NOT NULL,
                    p_neutral REAL NOT NULL,
                    p_down REAL NOT NULL,
                    confidence REAL NOT NULL,
                    agreement_score REAL NOT NULL,
                    regime TEXT NOT NULL,
                    is_low_edge INTEGER NOT NULL DEFAULT 0,
                    has_conflicts INTEGER NOT NULL DEFAULT 0,
                    expected_net_bps REAL,
                    status TEXT NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    net_bps REAL,
                    evidence_chips TEXT,
                    multipliers TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_zones_status ON trade_zones(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_zones_time ON trade_zones(candle_open_time)")

            conn.commit()

    def record_run_start(
        self,
        git_commit: str,
        git_tag: Optional[str],
        buffer_size: int,
        config_hash: str,
        clean_repo_asserted: bool,
    ) -> int:
        """Record process initialization metadata for forward-test hygiene."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO run_start (start_time, git_commit, git_tag, buffer_size, config_hash, clean_repo_asserted)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now_str, git_commit, git_tag, buffer_size, config_hash, 1 if clean_repo_asserted else 0))
            conn.commit()
            return int(cur.lastrowid)

    def record_prediction(self, pred: Dict[str, Any]) -> bool:
        """Record an emitted prediction into the ledger (idempotent primary key)."""
        open_time = pred["candle_open_time"]
        now_str = datetime.now(timezone.utc).isoformat()

        initial_status = "NO_SIGNAL" if pred.get("status") == "NO_SIGNAL" else "PENDING"
        imputed_names = pred.get("imputed_feature_names", [])
        if isinstance(imputed_names, list):
            imputed_names_str = json.dumps(imputed_names)
        else:
            imputed_names_str = str(imputed_names)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO predictions (
                    candle_open_time, ref_close, target_candle_open_time, action, action_label,
                    confidence, p_up, p_neutral, p_down, trend, market_phase, model_version,
                    git_commit, config_hash, buffer_length, compute_latency_ms, status, status_reason,
                    n_imputed_features, imputed_feature_names, exchange_close_time, exchange_event_time_ms,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(pred.get("n_imputed_features", 0)),
                imputed_names_str,
                pred.get("exchange_close_time"),
                pred.get("exchange_event_time_ms"),
                now_str,
            ))
            conn.commit()
        return True

    def resolve_pending(self, candle_store: CandleStore) -> List[Dict[str, Any]]:
        """Scan and resolve all PENDING predictions whose outcome candle k+2 now exists."""
        resolved: List[Dict[str, Any]] = []

        with self._get_connection() as conn:
            pending_rows = conn.execute("""
                SELECT * FROM predictions WHERE correct IS NULL ORDER BY candle_open_time ASC
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

            # Determine Verdict (assign definitive CORRECT or WRONG verdict to all resolved rows)
            action = int(pred["action"])
            if action not in [1, -1]:
                p_up = float(pred.get("p_up") or 0.33)
                p_dn = float(pred.get("p_down") or 0.33)
                action = 1 if p_up >= p_dn else -1

            if raw_sign_2 == 0.0:
                status = "RESOLVED"
                correct = 0
                reason = "Zero raw forward return"
            elif action == int(raw_sign_2):
                status = "RESOLVED"
                correct = 1
                reason = "Direction matched raw_sign_2"
            else:
                status = "RESOLVED"
                correct = 0
                reason = "Direction opposite of raw_sign_2"

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

    # =========================================================================
    # MODEL-INFORMED TRADE ZONES (ADDITIVE)
    # =========================================================================

    def record_trade_zone(self, zone: Dict[str, Any]) -> None:
        """Persist a newly formed trade zone on closed candle. Strictly zero repainting."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trade_zones (
                    zone_id, candle_open_time, target_candle_open_time, direction,
                    ref_close, entry_min, entry_max, entry_mid, target_price, stop_price,
                    risk_reward, p_up, p_neutral, p_down, confidence, agreement_score,
                    regime, is_low_edge, has_conflicts, expected_net_bps, status,
                    exit_time, exit_price, net_bps, evidence_chips, multipliers,
                    created_at, resolved_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    zone["zone_id"],
                    zone["candle_open_time"],
                    zone["target_candle_open_time"],
                    zone["direction"],
                    zone["ref_close"],
                    zone["entry_min"],
                    zone["entry_max"],
                    zone["entry_mid"],
                    zone["target_price"],
                    zone["stop_price"],
                    zone["risk_reward"],
                    zone["p_up"],
                    zone["p_neutral"],
                    zone["p_down"],
                    zone["confidence"],
                    zone["agreement_score"],
                    zone["regime"],
                    1 if zone.get("is_low_edge") else 0,
                    1 if zone.get("has_conflicts") else 0,
                    zone.get("expected_net_bps"),
                    zone.get("status", "ACTIVE"),
                    zone.get("exit_time"),
                    zone.get("exit_price"),
                    zone.get("net_bps"),
                    json.dumps(zone.get("evidence_chips", [])),
                    json.dumps(zone.get("multipliers", {})),
                    datetime.now(timezone.utc).isoformat(),
                    None,
                ),
            )
            conn.commit()

    def update_trade_zone_status(
        self,
        zone_id: str,
        status: str,
        exit_time: Optional[str] = None,
        exit_price: Optional[float] = None,
        net_bps: Optional[float] = None,
    ) -> None:
        """Update resolution status of a trade zone once forward candles resolve."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE trade_zones
                SET status = ?, exit_time = ?, exit_price = ?, net_bps = ?, resolved_at = ?
                WHERE zone_id = ?
                """,
                (
                    status,
                    exit_time,
                    exit_price,
                    net_bps,
                    datetime.now(timezone.utc).isoformat(),
                    zone_id,
                ),
            )
            conn.commit()

    def get_active_trade_zones(self) -> List[Dict[str, Any]]:
        """Retrieve all currently active / unresolved trade zones."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_zones WHERE status = 'ACTIVE' ORDER BY candle_open_time DESC"
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["evidence_chips"] = json.loads(d.get("evidence_chips") or "[]")
                d["multipliers"] = json.loads(d.get("multipliers") or "{}")
                results.append(d)
            return results

    def get_recent_trade_zones(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical and active trade zones for history table & charts."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_zones ORDER BY candle_open_time DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["evidence_chips"] = json.loads(d.get("evidence_chips") or "[]")
                d["multipliers"] = json.loads(d.get("multipliers") or "{}")
                results.append(d)
            return results

    def get_latest_trade_zone(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent trade zone."""
        zones = self.get_recent_trade_zones(limit=1)
        return zones[0] if zones else None


