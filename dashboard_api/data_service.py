"""Data Service for Dashboard API.

Handles read-only SQLite connections, clean.csv loading, overview downsampling,
test split slicing, and cached replay predictions.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ML"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.live import config as live_config
from src import config as ml_config
from backend.engine import predict_one, TradingEngine
from dashboard_api.bands import compute_wilder_atr14, compute_indicative_bands
from dashboard_api.insights import (
    compute_shown_action,
    InsightsStore,
    compute_empirical_band,
)

CLEAN_CSV_PATH = ML_DIR / "data" / "processed" / "clean.csv"
CANDLES_DB_PATH = BASE_DIR / "backend" / "data" / "candles.db"
LIVE_LEDGER_DB_PATH = BASE_DIR / "backend" / "data" / "live_ledger.db"

# Vocabulary mapping definitions
LABEL_MAP = {
    "LONG": "UP",
    "SHORT": "DOWN",
    "NEUTRAL": "NEUTRAL",
    "NO_SIGNAL": "NO_SIGNAL",
}

TREND_MAP = {
    "BULLISH": "UPTREND",
    "UPTREND": "UPTREND",
    "BEARISH": "DOWNTREND",
    "DOWNTREND": "DOWNTREND",
    "SIDEWAYS": "UPTREND",
    "uncertain": "UPTREND",
}

PHASE_MAP = {
    "IMPULSE": "BREAKTHROUGH",
    "BREAKTHROUGH": "BREAKTHROUGH",
    "PULLBACK": "PULLBACK",
    "uncertain": "PULLBACK",
}


def parse_utc_timestamp(ts_str: str) -> pd.Timestamp:
    """Robust parser for ISO timestamp handling decoded query string spaces."""
    s = str(ts_str).strip()
    if " " in s and "+" not in s:
        s = s.replace(" 00:00", "+00:00").replace(" ", "T")
    return pd.to_datetime(s, utc=True)


class DataService:
    """Singleton service providing read-only data access for dashboard_api."""

    _instance: Optional["DataService"] = None

    def __init__(self) -> None:
        self.candles_db_uri = f"file:{CANDLES_DB_PATH.resolve().as_posix()}?mode=ro"
        self.ledger_db_uri = f"file:{LIVE_LEDGER_DB_PATH.resolve().as_posix()}?mode=ro"
        self._clean_df: Optional[pd.DataFrame] = None
        self._offline_test_df: Optional[pd.DataFrame] = None
        self._overview_cache: Optional[List[Dict[str, Any]]] = None
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}
        self.insights_store = InsightsStore()

        # Test boundaries
        test_bounds = ml_config.SPLIT_BOUNDARIES["test"]
        self.test_start_dt = test_bounds["start"]
        self.test_end_dt = test_bounds["end"]

    @classmethod
    def get_instance(cls) -> "DataService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_clean_df(self, reload: bool = False) -> pd.DataFrame:
        if self._clean_df is None or reload:
            df = pd.read_csv(CLEAN_CSV_PATH)
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df.set_index("open_time").sort_index()
            self._clean_df = df
        return self._clean_df

    def _get_offline_test_df(self) -> Optional[pd.DataFrame]:
        if self._offline_test_df is None:
            csv_path = ml_config.RESULTS_DIR / "horizon_2_test.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                df = df.set_index("open_time").sort_index()
                self._offline_test_df = df
        return self._offline_test_df

    def get_config(self) -> Dict[str, Any]:
        """Compile complete configuration and vocabulary definitions."""
        return {
            "symbol": live_config.SYMBOL,
            "timeframe": live_config.INTERVAL,
            "interval_minutes": live_config.INTERVAL_MINUTES,
            "horizon_candles": live_config.HORIZON,
            "horizon_minutes": live_config.HORIZON * live_config.INTERVAL_MINUTES,
            "buffer_size": live_config.BUFFER_SIZE,
            "min_buffer_size": live_config.MIN_BUFFER_SIZE,
            "cost_floor_bps": live_config.COST_FLOOR_BPS,
            "split_boundaries": {
                "train": {
                    "start": ml_config.SPLIT_BOUNDARIES["train"]["start"].isoformat(),
                    "end": ml_config.SPLIT_BOUNDARIES["train"]["end"].isoformat(),
                },
                "val": {
                    "start": ml_config.SPLIT_BOUNDARIES["val"]["start"].isoformat(),
                    "end": ml_config.SPLIT_BOUNDARIES["val"]["end"].isoformat(),
                },
                "test": {
                    "start": ml_config.SPLIT_BOUNDARIES["test"]["start"].isoformat(),
                    "end": ml_config.SPLIT_BOUNDARIES["test"]["end"].isoformat(),
                },
            },
            "test_range": {
                "min_open_time": self.test_start_dt.isoformat(),
                "max_open_time": self.test_end_dt.isoformat(),
                "candle_count": ml_config.SPLIT_BOUNDARIES["test"]["expected_candles"],
            },
            "band_constants": {
                "K_MOVE": 0.5,
                "Z": 1.28,
                "ATR_PERIOD": 14,
            },
            "label_vocabulary": {
                "action_map": LABEL_MAP,
                "trend_map": TREND_MAP,
                "phase_map": PHASE_MAP,
            },
            "benchmarks": {
                "ref_test_accuracy_pct": live_config.REF_TEST_ACCURACY,
                "ref_test_coverage_pct": live_config.REF_TEST_COVERAGE,
                "ref_oof_accuracy_pct": live_config.REF_OOF_ACCURACY,
                "ref_oof_coverage_pct": live_config.REF_OOF_COVERAGE,
            },
        }

    def get_historical_candles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Retrieve recent candles combining clean.csv master history with candles.db live data."""
        clean_df = self._get_clean_df()

        # Check if live candles exist in candles.db to provide unbroken data up to the current instant
        if CANDLES_DB_PATH.exists():
            try:
                conn = sqlite3.connect(self.candles_db_uri, uri=True, timeout=5.0)
                db_df = pd.read_sql_query(
                    "SELECT open_time, open, high, low, close, volume FROM candles ORDER BY open_time ASC",
                    conn,
                )
                conn.close()
                if not db_df.empty:
                    db_df["open_time"] = pd.to_datetime(db_df["open_time"], utc=True)
                    db_df = db_df.set_index("open_time")
                    earliest_live = db_df.index[0]
                    clean_prior = clean_df[clean_df.index < earliest_live]
                    df = pd.concat([clean_prior, db_df]).sort_index()
                else:
                    df = clean_df
            except Exception as e:
                logger.warning(f"Failed to stitch candles.db in get_historical_candles: {e}")
                df = clean_df
        else:
            df = clean_df

        tail_df = df.tail(limit).reset_index()
        out = []
        for _, row in tail_df.iterrows():
            ts = row["open_time"]
            out.append({
                "time": int(ts.timestamp()),
                "open_time": ts.isoformat(),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 4),
            })
        return out

    def get_live_candles(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve recent closed candles from candles.db in read-only mode."""
        if not CANDLES_DB_PATH.exists():
            return []

        conn = sqlite3.connect(self.candles_db_uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT open_time, open, high, low, close, volume, close_time
                FROM candles
                ORDER BY open_time DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            candles = [dict(r) for r in reversed(rows)]
            return candles
        finally:
            conn.close()

    def get_live_predictions(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Retrieve recent predictions from live_ledger.db enriched with bands."""
        if not LIVE_LEDGER_DB_PATH.exists():
            return []

        conn = sqlite3.connect(self.ledger_db_uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY candle_open_time DESC
                LIMIT ?
                """,
                (limit,),
            )
            pred_rows = [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

        if not pred_rows:
            latest = self.predict_latest_candle()
            return [latest] if latest else []

        candles = self.get_live_candles(limit=min(1000, limit + 50))
        df_candles = pd.DataFrame(candles) if candles else pd.DataFrame()
        if not df_candles.empty:
            df_candles["open_time"] = pd.to_datetime(df_candles["open_time"], utc=True)
            df_candles = df_candles.set_index("open_time").sort_index()

        enriched = []
        for row in pred_rows:
            ts_str = row["candle_open_time"]
            ts_dt = parse_utc_timestamp(ts_str)
            ref_close = float(row["ref_close"])
            p_up = float(row["p_up"])
            p_dn = float(row["p_down"])
            p_neu = float(row["p_neutral"])

            atr14 = 100.0
            if not df_candles.empty:
                slice_df = df_candles[df_candles.index <= ts_dt].tail(50)
                if len(slice_df) >= 15:
                    atr14 = compute_wilder_atr14(slice_df, period=14)

            band = compute_empirical_band(
                close_price=ref_close,
                atr14=atr14,
                p_up=p_up,
                p_down=p_dn,
                p_neutral=p_neu,
            )

            argmax_idx = int(np.argmax([p_dn, p_neu, p_up]))
            model_res_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
            model_res = model_res_map[argmax_idx]

            raw_action = str(row["action_label"])
            ui_label = LABEL_MAP.get(raw_action, raw_action)
            raw_trend = str(row["trend"])
            ui_trend = TREND_MAP.get(raw_trend, "UPTREND" if p_up >= p_dn else "DOWNTREND")
            if ui_trend not in ["UPTREND", "DOWNTREND"]:
                ui_trend = "UPTREND" if p_up >= p_dn else "DOWNTREND"
            raw_phase = str(row["market_phase"])
            ui_phase = PHASE_MAP.get(raw_phase, raw_phase)

            correct_val = row.get("correct")
            outcome_cl = row.get("outcome_close")
            outcome_op = row.get("outcome_open")
            ret_bps_lbl = row.get("return_bps_label")
            status_val = str(row["status"])

            # NO_SIGNAL handling: never graded, shows status_reason
            if status_val == "NO_SIGNAL" or raw_action == "NO_SIGNAL":
                reason_str = str(row.get("status_reason") or "No signal")
                status_val = f"NO SIGNAL: {reason_str}"
                ui_label = "NO_SIGNAL"
                correct_val = None
            elif ui_label not in ["UP", "DOWN"]:
                # NEUTRAL is never graded
                correct_val = None
            else:
                # Directional UP / DOWN
                target_dt = ts_dt + pd.Timedelta(minutes=10)
                if not df_candles.empty:
                    cand_k2 = None
                    if target_dt in df_candles.index:
                        cand_k2 = df_candles.loc[target_dt]
                    else:
                        after_df = df_candles[df_candles.index >= target_dt]
                        if not after_df.empty:
                            cand_k2 = after_df.iloc[0]

                    if cand_k2 is not None:
                        if isinstance(cand_k2, pd.DataFrame):
                            cand_k2 = cand_k2.iloc[-1]
                        outcome_cl = round(float(cand_k2["close"]), 2)
                        outcome_op = round(float(cand_k2["open"]), 2)
                        ret = (outcome_cl / ref_close) - 1.0 if ref_close > 0 else 0.0
                        ret_bps_lbl = round(ret * 10000.0, 1)

                        if ret == 0.0:
                            correct_val = None
                            status_val = "VOID"
                        else:
                            correct_val = 1 if (ret > 0 if ui_label == "UP" else ret < 0) else 0
                            status_val = "RESOLVED"
                    elif correct_val is None:
                        status_val = "PENDING"
                elif correct_val is None:
                    status_val = "PENDING"

            # Compute shown_action via display-layer policy (F1)
            shown_act, abstention_reason = compute_shown_action(
                ledger_action=ui_label,
                p_up=p_up,
                p_down=p_dn,
                p_neutral=p_neu,
                status=status_val,
            )

            try:
                self.insights_store.record_insight(
                    candle_open_time=ts_str,
                    shown_action=shown_act,
                    ledger_action=ui_label,
                    delta=0.015,
                    m=0.015,
                    abstention_reason=abstention_reason,
                    p_up=p_up,
                    p_down=p_dn,
                    p_neutral=p_neu,
                    status=status_val,
                )
            except Exception:
                pass

            raw_conf = row.get("confidence")
            conf_val = float(raw_conf) if raw_conf is not None else None

            enriched.append({
                "open_time": ts_str,
                "close": ref_close,
                "trend": ui_trend,
                "trend_strength": None,
                "phase": ui_phase,
                "phase_strength": None,
                "context_text": f"{ui_trend} | {ui_phase}",
                "p_ml": None,
                "model_result": model_res,
                "p_final": {
                    "down": p_dn,
                    "neutral": p_neu,
                    "up": p_up,
                },
                "label": shown_act,
                "shown_action": shown_act,
                "ledger_action": ui_label,
                "abstention_reason": abstention_reason,
                "confidence": conf_val,
                "smart_price": band["smart_price"],
                "upper": band["upper"],
                "lower": band["lower"],
                "atr14": band["atr14"],
                "vol_tercile": band.get("vol_tercile"),
                "neu_tercile": band.get("neu_tercile"),
                "text": None,
                "status": status_val,
                "reason": row["status_reason"],
                "n_imputed_features": int(row.get("n_imputed_features") or 0),
                "latency_ms": float(row.get("compute_latency_ms") or 0.0),
                "correct": correct_val,
                "outcome_close": outcome_cl,
                "outcome_open": outcome_op,
                "return_bps_label": ret_bps_lbl,
                "return_bps_executable": row.get("return_bps_executable"),
            })

        return enriched

    def predict_latest_candle(self) -> Optional[Dict[str, Any]]:
        """Compute an immediate live prediction on the most recent closed candle."""
        buffer_size = live_config.BUFFER_SIZE  # 3000

        df = None
        if CANDLES_DB_PATH.exists():
            try:
                conn = sqlite3.connect(self.candles_db_uri, uri=True, timeout=5.0)
                candles_df = pd.read_sql_query(
                    f"SELECT * FROM candles ORDER BY open_time DESC LIMIT {buffer_size}",
                    conn,
                )
                conn.close()
                if len(candles_df) >= live_config.MIN_BUFFER_SIZE:
                    candles_df["open_time"] = pd.to_datetime(candles_df["open_time"], utc=True)
                    candles_df = candles_df.set_index("open_time").sort_index()
                    df = candles_df
            except Exception:
                df = None

        if df is None or len(df) < live_config.MIN_BUFFER_SIZE:
            clean_df = self._get_clean_df()
            df = clean_df.tail(buffer_size).copy()

        if len(df) == 0:
            return None

        target_dt = df.index[-1]
        t0 = time.time()
        pred_res = predict_one(df)
        latency_ms = (time.time() - t0) * 1000.0

        probs = pred_res["probabilities"]
        p_up = float(probs["p_up"])
        p_neu = float(probs["p_neutral"])
        p_dn = float(probs["p_down"])
        ref_close = float(pred_res["close_price"])

        atr14 = compute_wilder_atr14(df, period=14)
        band = compute_empirical_band(
            close_price=ref_close,
            atr14=atr14,
            p_up=p_up,
            p_down=p_dn,
            p_neutral=p_neu,
        )

        ta_info = pred_res.get("technical_analysis", {})
        raw_trend = str(ta_info.get("trend", "uncertain"))
        raw_phase = str(ta_info.get("market_phase", "uncertain"))
        trend_score = float(ta_info.get("trend_score", 0.0))
        phase_score = float(ta_info.get("breakthrough_score", 0.0))

        ui_trend = TREND_MAP.get(raw_trend, "UPTREND" if p_up >= p_dn else "DOWNTREND")
        if ui_trend not in ["UPTREND", "DOWNTREND"]:
            ui_trend = "UPTREND" if p_up >= p_dn else "DOWNTREND"
        ui_phase = PHASE_MAP.get(raw_phase, raw_phase)
        raw_action = str(pred_res["action_label"])
        ui_label = LABEL_MAP.get(raw_action, raw_action)

        shown_act, abstention_reason = compute_shown_action(
            ledger_action=ui_label,
            p_up=p_up,
            p_down=p_dn,
            p_neutral=p_neu,
            status="PENDING",
        )

        argmax_idx = int(np.argmax([p_dn, p_neu, p_up]))
        model_res_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        model_res = model_res_map[argmax_idx]

        return {
            "open_time": target_dt.isoformat(),
            "close": ref_close,
            "trend": ui_trend,
            "trend_strength": round(trend_score, 4),
            "phase": ui_phase,
            "phase_strength": round(phase_score, 4),
            "context_text": f"{ui_trend} | {ui_phase}",
            "p_ml": None,
            "model_result": model_res,
            "p_final": {
                "down": p_dn,
                "neutral": p_neu,
                "up": p_up,
            },
            "label": shown_act,
            "shown_action": shown_act,
            "ledger_action": ui_label,
            "abstention_reason": abstention_reason,
            "confidence": float(pred_res["confidence"]),
            "smart_price": band["smart_price"],
            "upper": band["upper"],
            "lower": band["lower"],
            "atr14": band["atr14"],
            "vol_tercile": band.get("vol_tercile"),
            "neu_tercile": band.get("neu_tercile"),
            "text": None,
            "status": "PENDING",
            "reason": None,
            "n_imputed_features": 0,
            "latency_ms": round(latency_ms, 2),
            "correct": None,
            "outcome_close": None,
            "outcome_open": None,
            "return_bps_label": None,
            "return_bps_executable": None,
        }

    def get_overview(self) -> List[Dict[str, Any]]:
        """Downsample complete history to 1h bars with split zones."""
        if self._overview_cache is not None:
            return self._overview_cache

        df = self._get_clean_df()
        resampled = df.resample("1h").agg({"close": "last"}).dropna().reset_index()

        val_start = ml_config.SPLIT_BOUNDARIES["val"]["start"]
        test_start = ml_config.SPLIT_BOUNDARIES["test"]["start"]

        points = []
        for _, row in resampled.iterrows():
            ts = row["open_time"]
            if ts < val_start:
                split = "train"
            elif ts < test_start:
                split = "val"
            else:
                split = "test"

            points.append({
                "time": int(ts.timestamp()),
                "iso": ts.isoformat(),
                "close": round(float(row["close"]), 2),
                "split": split,
            })

        self._overview_cache = points
        return self._overview_cache

    def get_replay_range(self) -> Dict[str, Any]:
        """Return TEST split boundaries."""
        return {
            "min_open_time": self.test_start_dt.isoformat(),
            "max_open_time": self.test_end_dt.isoformat(),
            "count": ml_config.SPLIT_BOUNDARIES["test"]["expected_candles"],
        }

    def get_replay_candles(self, from_time: str, to_time: str) -> List[Dict[str, Any]]:
        """Slice clean.csv strictly within the TEST range."""
        from_dt = parse_utc_timestamp(from_time)
        to_dt = parse_utc_timestamp(to_time)

        if from_dt < self.test_start_dt or to_dt > self.test_end_dt:
            raise ValueError(
                f"Requested range [{from_dt} to {to_dt}] extends outside the TEST partition "
                f"[{self.test_start_dt} to {self.test_end_dt}]."
            )

        df = self._get_clean_df()
        sliced = df[(df.index >= from_dt) & (df.index <= to_dt)]

        out = []
        for ts, row in sliced.iterrows():
            out.append({
                "open_time": ts.isoformat(),
                "time": int(ts.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 4),
            })
        return out

    def get_latest_features(self) -> Dict[str, Any]:
        """Compute and return the full 136 engineered causal features for the latest closed candle."""
        df = self._get_clean_df()
        if CANDLES_DB_PATH.exists():
            try:
                conn = sqlite3.connect(self.candles_db_uri, uri=True, timeout=5.0)
                candles_df = pd.read_sql_query(
                    """
                    SELECT open_time, open, high, low, close, volume,
                           taker_buy_base_volume, number_of_trades
                    FROM candles
                    ORDER BY open_time DESC
                    LIMIT 2500
                    """,
                    conn,
                )
                conn.close()
                if not candles_df.empty and len(candles_df) >= 200:
                    candles_df["open_time"] = pd.to_datetime(candles_df["open_time"], utc=True)
                    df = candles_df.set_index("open_time").sort_index()
            except Exception:
                pass

        from src.features import build_features
        sample_df = df.tail(2500).copy()
        feats = build_features(sample_df)
        last_row = feats.iloc[-1].to_dict()
        last_ts = sample_df.index[-1].isoformat()
        last_close = float(sample_df["close"].iloc[-1])

        clean_feats = {k: round(float(v), 6) if pd.notna(v) else 0.0 for k, v in last_row.items()}
        return {
            "open_time": last_ts,
            "close": last_close,
            "count": len(clean_feats),
            "features": clean_feats,
        }

    def predict_replay_candle(self, ts: str) -> Dict[str, Any]:
        """Execute single-candle replay prediction with 3,000-candle lookback."""
        if ts in self._prediction_cache:
            return self._prediction_cache[ts]

        target_dt = parse_utc_timestamp(ts)
        iso_str = target_dt.isoformat()
        std_str = str(target_dt)
        if iso_str in self._prediction_cache:
            return self._prediction_cache[iso_str]
        if std_str in self._prediction_cache:
            return self._prediction_cache[std_str]

        if target_dt < self.test_start_dt or target_dt > self.test_end_dt:
            raise ValueError(f"Timestamp {target_dt} is outside TEST split [{self.test_start_dt} to {self.test_end_dt}].")

        df = self._get_clean_df()
        if target_dt not in df.index:
            raise KeyError(f"Timestamp {target_dt} not found in market data.")

        target_pos = df.index.get_loc(target_dt)
        buffer_size = live_config.BUFFER_SIZE  # 3000

        if target_pos < buffer_size - 1:
            raise ValueError(
                f"Buffer too short: requested timestamp at index {target_pos}, need >= {buffer_size} history candles."
            )

        buffer_df = df.iloc[target_pos - buffer_size + 1 : target_pos + 1].copy()
        assert buffer_df.index[-1] == target_dt, "Lookback buffer endpoint mismatch"

        t0 = time.time()
        pred_res = predict_one(buffer_df)
        latency_ms = (time.time() - t0) * 1000.0

        probs = pred_res["probabilities"]
        p_up = float(probs["p_up"])
        p_neu = float(probs["p_neutral"])
        p_dn = float(probs["p_down"])
        ref_close = float(pred_res["close_price"])

        atr14 = compute_wilder_atr14(buffer_df, period=14)
        band = compute_empirical_band(
            close_price=ref_close,
            atr14=atr14,
            p_up=p_up,
            p_down=p_dn,
            p_neutral=p_neu,
        )

        ta_info = pred_res.get("technical_analysis", {})
        raw_trend = str(ta_info.get("trend", "uncertain"))
        raw_phase = str(ta_info.get("market_phase", "uncertain"))
        trend_score = float(ta_info.get("trend_score", 0.0))
        phase_score = float(ta_info.get("breakthrough_score", 0.0))

        ui_trend = TREND_MAP.get(raw_trend, "UPTREND" if p_up >= p_dn else "DOWNTREND")
        if ui_trend not in ["UPTREND", "DOWNTREND"]:
            ui_trend = "UPTREND" if p_up >= p_dn else "DOWNTREND"
        ui_phase = PHASE_MAP.get(raw_phase, raw_phase)
        raw_action = str(pred_res["action_label"])
        ui_label = LABEL_MAP.get(raw_action, raw_action)

        shown_act, abstention_reason = compute_shown_action(
            ledger_action=ui_label,
            p_up=p_up,
            p_down=p_dn,
            p_neutral=p_neu,
            status="PENDING",
        )

        argmax_idx = int(np.argmax([p_dn, p_neu, p_up]))
        model_res_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        model_res = model_res_map[argmax_idx]

        outcome_close = None
        outcome_open = None
        correct = None
        return_bps_label = None
        return_bps_exec = None
        status = "PENDING"
        reason = None

        if target_pos + 2 < len(df):
            c_k1 = df.iloc[target_pos + 1]
            c_k2 = df.iloc[target_pos + 2]
            outcome_close = round(float(c_k2["close"]), 2)
            outcome_open = round(float(c_k1["open"]), 2)
            fwd_ret = (outcome_close / ref_close) - 1.0
            fwd_ret_exec = (outcome_close / outcome_open) - 1.0
            return_bps_label = round(fwd_ret * 10000.0, 2)
            return_bps_exec = round(fwd_ret_exec * 10000.0, 2)

            raw_sign = 1.0 if fwd_ret > 0 else (-1.0 if fwd_ret < 0 else 0.0)
            action_code = int(pred_res["action"])

            if action_code in [1, -1]:
                if raw_sign == 0.0:
                    status = "VOID"
                    correct = None
                    reason = "Zero raw forward return"
                elif action_code == int(raw_sign):
                    status = "RESOLVED"
                    correct = 1
                    reason = "Direction matched raw_sign_2"
                else:
                    status = "RESOLVED"
                    correct = 0
                    reason = "Direction opposite of raw_sign_2"
            else:
                status = "NO_CALL"
                correct = None
                reason = "Neutral call (no active directional call)"

        result = {
            "open_time": target_dt.isoformat(),
            "close": ref_close,
            "trend": ui_trend,
            "trend_strength": round(trend_score, 4),
            "phase": ui_phase,
            "phase_strength": round(phase_score, 4),
            "context_text": f"{ui_trend} | {ui_phase}",
            "p_ml": None,
            "model_result": model_res,
            "p_final": {
                "down": p_dn,
                "neutral": p_neu,
                "up": p_up,
            },
            "label": shown_act,
            "shown_action": shown_act,
            "ledger_action": ui_label,
            "abstention_reason": abstention_reason,
            "confidence": float(pred_res["confidence"]),
            "smart_price": band["smart_price"],
            "upper": band["upper"],
            "lower": band["lower"],
            "atr14": band["atr14"],
            "vol_tercile": band.get("vol_tercile"),
            "neu_tercile": band.get("neu_tercile"),
            "text": None,
            "status": status,
            "reason": reason,
            "n_imputed_features": 0,
            "latency_ms": round(latency_ms, 2),
            "correct": correct,
            "outcome_close": outcome_close,
            "outcome_open": outcome_open,
            "return_bps_label": return_bps_label,
            "return_bps_executable": return_bps_exec,
        }

        self._prediction_cache[ts] = result
        self._prediction_cache[iso_str] = result
        self._prediction_cache[std_str] = result
        return result

    def predict_replay_range(self, from_ts: str, count: int) -> List[Dict[str, Any]]:
        """Batch predict replay candles starting at from_ts."""
        from_dt = parse_utc_timestamp(from_ts)
        df = self._get_clean_df()

        if from_dt not in df.index:
            raise KeyError(f"Timestamp {from_dt} not found in market data.")

        start_idx = df.index.get_loc(from_dt)
        end_idx = min(len(df), start_idx + count)

        results = []
        for i in range(start_idx, end_idx):
            ts_str = df.index[i].isoformat()
            pred = self.predict_replay_candle(ts_str)
            results.append(pred)

        return results
