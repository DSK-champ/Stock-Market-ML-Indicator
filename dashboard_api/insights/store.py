"""Append-only SQLite storage for display-layer insights (WAL mode).

Maintains a distinct record of shown_action, abstention reasons, and insight metadata,
completely segregated from the frozen live_ledger.db.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

INSIGHTS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "insights.db"


class InsightsStore:
    """Manages append-only insights SQLite database."""

    def __init__(self, db_path: Path = INSIGHTS_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    candle_open_time TEXT PRIMARY KEY,
                    shown_action TEXT NOT NULL,
                    ledger_action TEXT NOT NULL,
                    delta REAL NOT NULL,
                    m REAL NOT NULL,
                    abstention_reason TEXT,
                    p_up REAL,
                    p_down REAL,
                    p_neutral REAL,
                    status TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_insight(
        self,
        candle_open_time: str,
        shown_action: str,
        ledger_action: str,
        delta: float,
        m: float,
        abstention_reason: Optional[str] = None,
        p_up: Optional[float] = None,
        p_down: Optional[float] = None,
        p_neutral: Optional[float] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Insert insight record in append-only manner (INSERT OR IGNORE)."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO insights (
                    candle_open_time,
                    shown_action,
                    ledger_action,
                    delta,
                    m,
                    abstention_reason,
                    p_up,
                    p_down,
                    p_neutral,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle_open_time,
                shown_action,
                ledger_action,
                delta,
                m,
                abstention_reason,
                p_up,
                p_down,
                p_neutral,
                status,
                now_str,
            ))
            conn.commit()
        return True

    def get_insight(self, candle_open_time: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM insights WHERE candle_open_time = ?", (candle_open_time,)
            ).fetchone()
            return dict(row) if row else None
