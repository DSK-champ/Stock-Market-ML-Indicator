"""Pure mathematical calculator for Model-Informed Trade Zones.

Computes entry zones, target expansion hurdles, stop invalidation boundaries,
agreement scores across 12 sub-models / TA indicators, and regime classifications
without modifying model predictions or looking into future unclosed bars.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd

from backend.live.zone_config import DEFAULT_ZONE_CONFIG


def compute_garman_klass_volatility(candles: Union[pd.DataFrame, List[Dict[str, Any]]], n: int = 10) -> float:
    """Compute Garman-Klass volatility over the last n closed candles.
    
    Formula:
        sigma_GK = sqrt( 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2 )
    """
    if isinstance(candles, list):
        if len(candles) == 0:
            return 0.0035
        df = pd.DataFrame(candles)
    else:
        df = candles

    if df.empty or len(df) < 2:
        return 0.0035

    tail_df = df.tail(n)
    h = pd.to_numeric(tail_df["high"], errors="coerce").values
    l = pd.to_numeric(tail_df["low"], errors="coerce").values
    c = pd.to_numeric(tail_df["close"], errors="coerce").values
    o = pd.to_numeric(tail_df["open"], errors="coerce").values

    # Guard against zero or negative values
    valid = (h > 0) & (l > 0) & (c > 0) & (o > 0) & (h >= l)
    if not np.any(valid):
        return 0.0035

    h, l, c, o = h[valid], l[valid], c[valid], o[valid]
    log_hl = np.log(h / np.maximum(l, 1e-8))
    log_co = np.log(c / np.maximum(o, 1e-8))

    gk = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
    mean_gk = float(np.mean(np.maximum(gk, 0.0)))
    sigma = math.sqrt(mean_gk)
    return max(0.0010, min(sigma, 0.0500))


def find_dow_swings(
    candles: Union[pd.DataFrame, List[Dict[str, Any]]],
    lookback: int = 12
) -> Tuple[Optional[float], Optional[float]]:
    """Detect the nearest Dow swing high and swing low from recent pivots."""
    if isinstance(candles, list):
        if len(candles) == 0:
            return None, None
        df = pd.DataFrame(candles)
    else:
        df = candles

    if df.empty or len(df) < 4:
        return None, None

    tail_df = df.tail(lookback)
    highs = pd.to_numeric(tail_df["high"], errors="coerce").values
    lows = pd.to_numeric(tail_df["low"], errors="coerce").values

    swing_high = float(np.max(highs)) if len(highs) > 0 else None
    swing_low = float(np.min(lows)) if len(lows) > 0 else None
    return swing_high, swing_low


def classify_regime(
    ref_close: float,
    atr14: float,
    gk_vol: float,
    trend_label: str,
    trend_score: float,
    config: Dict[str, Any]
) -> str:
    """Classify current market regime into: trending, ranging, or high-volatility."""
    high_vol_thresh = config.get("high_vol_atr_threshold", 0.0070)
    atr_ratio = atr14 / max(ref_close, 1e-4)

    if atr_ratio >= high_vol_thresh or gk_vol >= 0.0065:
        return "high-volatility"
    if str(trend_label).upper() in ["UPTREND", "DOWNTREND"] or abs(trend_score) >= 0.35:
        return "trending"
    return "ranging"


def compute_agreement_score(
    direction: str,
    base_model_votes: Optional[Dict[str, Any]],
    ta_scores: Optional[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute agreement score (0% to 100%) and individual chip verdicts.
    
    Evaluates 12 sub-models:
    - 3 Binary classifiers: LR, RF, GB
    - 3 Ternary classifiers: LR, RF, GB
    - 6 TA Scores: ta_rsi, ta_ema50, ta_align, ta_loc, ta_rule_vote, ta_trend
    """
    chips: List[Dict[str, Any]] = []
    if direction == "NEUTRAL":
        # Neutral agreement
        return 50.0, [
            {"id": "neutral_standby", "name": "Neutral Baseline", "value": "NEUTRAL", "bullish": False, "bearish": False, "agrees": True}
        ]

    votes_for = 0
    total_votes = 0

    base_votes = base_model_votes or {}
    ta = ta_scores or {}

    # 1. Binary Base Models (3)
    bin_models = [
        ("bin_lr", "Binary LR", base_votes.get("binary_lr_p_up", base_votes.get("bin_lr_p_up", 0.5))),
        ("bin_rf", "Binary RF", base_votes.get("binary_rf_p_up", base_votes.get("bin_rf_p_up", 0.5))),
        ("bin_gb", "Binary GB", base_votes.get("binary_gb_p_up", base_votes.get("bin_gb_p_up", 0.5))),
    ]
    for m_id, m_name, p_up in bin_models:
        p_val = float(p_up) if p_up is not None else 0.5
        is_bull = p_val > 0.50
        is_bear = p_val < 0.50
        agrees = (direction == "LONG" and is_bull) or (direction == "SHORT" and is_bear)
        if agrees:
            votes_for += 1
        total_votes += 1
        chips.append({
            "id": m_id,
            "name": m_name,
            "value": f"{round(p_val * 100, 1)}%",
            "bullish": is_bull,
            "bearish": is_bear,
            "agrees": agrees,
        })

    # 2. Ternary Base Models (3)
    tri_models = [
        ("tri_lr", "3-Class LR", base_votes.get("tri_lr_p_up", 0.33), base_votes.get("tri_lr_p_down", 0.33)),
        ("tri_rf", "3-Class RF", base_votes.get("tri_rf_p_up", 0.33), base_votes.get("tri_rf_p_down", 0.33)),
        ("tri_gb", "3-Class GB", base_votes.get("tri_gb_p_up", 0.33), base_votes.get("tri_gb_p_down", 0.33)),
    ]
    for m_id, m_name, p_up, p_dn in tri_models:
        p_u = float(p_up) if p_up is not None else 0.33
        p_d = float(p_dn) if p_dn is not None else 0.33
        is_bull = p_u > p_d and p_u >= 0.34
        is_bear = p_d > p_u and p_d >= 0.34
        agrees = (direction == "LONG" and is_bull) or (direction == "SHORT" and is_bear)
        if agrees:
            votes_for += 1
        total_votes += 1
        val_str = f"U:{round(p_u*100)}% D:{round(p_d*100)}%"
        chips.append({
            "id": m_id,
            "name": m_name,
            "value": val_str,
            "bullish": is_bull,
            "bearish": is_bear,
            "agrees": agrees,
        })

    # 3. TA Scores (6)
    ta_items = [
        ("ta_rsi", "TA RSI", ta.get("ta_rsi", ta.get("rsi_score", 0.0))),
        ("ta_ema50", "TA EMA50", ta.get("ta_ema50", ta.get("ema50_score", 0.0))),
        ("ta_align", "TA Align", ta.get("ta_align", ta.get("alignment_score", 0.0))),
        ("ta_loc", "TA Location", ta.get("ta_loc", ta.get("location_score", 0.0))),
        ("ta_rule_vote", "TA Rules", ta.get("ta_rule_vote", ta.get("rule_vote", 0.0))),
        ("ta_trend", "TA Trend", ta.get("ta_trend", ta.get("trend_score", 0.0))),
    ]
    for t_id, t_name, val in ta_items:
        v = float(val) if val is not None else 0.0
        # If raw RSI (0-100), re-center to 50
        if t_id == "ta_rsi" and v > 1.0:
            is_bull = v > 50.0
            is_bear = v < 50.0
        else:
            is_bull = v > 0.05
            is_bear = v < -0.05

        agrees = (direction == "LONG" and is_bull) or (direction == "SHORT" and is_bear)
        if agrees:
            votes_for += 1
        total_votes += 1
        chips.append({
            "id": t_id,
            "name": t_name,
            "value": f"{v:+.2f}" if abs(v) <= 1.0 else f"{round(v, 1)}",
            "bullish": is_bull,
            "bearish": is_bear,
            "agrees": agrees,
        })

    score = round((votes_for / max(total_votes, 1)) * 100.0, 1)
    return score, chips


def calculate_trade_zone(
    pred: Dict[str, Any],
    recent_candles: Union[pd.DataFrame, List[Dict[str, Any]]],
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pure calculator generating complete Model-Informed Trade Zone for closed candle.

    Parameters
    ----------
    pred : Dict[str, Any]
        Prediction record for closed candle.
    recent_candles : Union[pd.DataFrame, List[Dict[str, Any]]]
        Historical closed candles up to candle t0.
    config_override : Optional[Dict[str, Any]]
        Optional overrides for multipliers.

    Returns
    -------
    Dict[str, Any]
        Complete trade zone specification.
    """
    cfg = dict(DEFAULT_ZONE_CONFIG)
    if config_override:
        cfg.update(config_override)

    open_time = str(pred.get("candle_open_time", pred.get("open_time", "")))
    target_time = str(pred.get("target_candle_open_time", pred.get("target_time", "")))
    ref_close = float(pred.get("ref_close", pred.get("close", 0.0)))
    if ref_close <= 0.0:
        if isinstance(recent_candles, list) and recent_candles:
            ref_close = float(recent_candles[-1].get("close", 0.0))
        elif isinstance(recent_candles, pd.DataFrame) and not recent_candles.empty:
            ref_close = float(recent_candles.iloc[-1]["close"])

    # Determine action & direction
    raw_action = pred.get("shown_action", pred.get("action_label", pred.get("action", 0)))
    if isinstance(raw_action, (int, float)):
        action_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
        direction = action_map.get(int(raw_action), "NEUTRAL")
    else:
        act_str = str(raw_action).strip().upper()
        if act_str in ["UP", "LONG", "1"]:
            direction = "LONG"
        elif act_str in ["DOWN", "SHORT", "-1"]:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

    # Probabilities & Confidence
    probs = pred.get("probabilities", {}) or {}
    p_up = float(probs.get("p_up", pred.get("p_up", 0.3333)))
    p_neu = float(probs.get("p_neutral", pred.get("p_neutral", 0.3334)))
    p_dn = float(probs.get("p_down", pred.get("p_down", 0.3333)))
    confidence = float(pred.get("confidence", max(p_up, p_dn, p_neu)))
    margin = abs(p_up - p_dn)

    # Volatility & Swings
    atr14 = float(pred.get("atr14", 0.0))
    if atr14 <= 0.0:
        atr14 = ref_close * 0.0035  # realistic fallback 35 bps
    gk_vol = compute_garman_klass_volatility(recent_candles, n=10)
    dow_high, dow_low = find_dow_swings(recent_candles, lookback=cfg["swing_lookback"])

    # Regime Classification
    tech = pred.get("technical_analysis", {}) or {}
    trend_label = str(tech.get("trend", pred.get("trend", "uncertain")))
    trend_score = float(tech.get("trend_score", 0.0))
    regime = classify_regime(ref_close, atr14, gk_vol, trend_label, trend_score, cfg)

    # Agreement Score & Chips
    base_votes = pred.get("base_model_votes", {}) or {}
    agreement_score, evidence_chips = compute_agreement_score(direction, base_votes, tech)
    has_conflicts = (direction != "NEUTRAL") and (agreement_score < 50.0)

    # Multipliers
    k_entry = float(cfg["k_entry"])
    k_stop = float(cfg["k_stop"])
    scale = float(cfg["scale"])
    buffer_bps = float(cfg["buffer_bps"])
    cost_floor_bps = float(cfg["cost_floor_bps"])
    high_vol_mult = float(cfg["high_vol_multiplier"])

    # Zone calculation
    if direction == "NEUTRAL":
        entry_min = ref_close
        entry_max = ref_close
        entry_mid = ref_close
        target_price = ref_close
        stop_price = ref_close
        risk_reward = 0.0
        expected_net_bps = 0.0
        is_low_edge = False
        zone_status = "NEUTRAL_BAND"
    else:
        # Entry zone depth shrinks as confidence & margin rise
        depth = atr14 * k_entry * max(0.40, (1.0 - 0.50 * margin))

        # Expected Move via Garman-Klass volatility heuristic
        expected_move = ref_close * gk_vol * margin * scale
        friction_move = ref_close * ((cost_floor_bps + buffer_bps) / 10000.0)
        is_low_edge = bool(expected_move < (ref_close * (cost_floor_bps / 10000.0)))
        target_move = max(expected_move, friction_move)

        # Stop distance expands during high volatility
        stop_dist = ref_close * gk_vol * k_stop
        if regime == "high-volatility":
            stop_dist *= high_vol_mult

        if direction == "LONG":
            entry_max = ref_close
            entry_min = ref_close - depth
            if dow_low is not None and dow_low < ref_close and dow_low > (ref_close - depth):
                entry_min = dow_low
            entry_mid = (entry_min + entry_max) / 2.0

            target_price = entry_mid + target_move
            raw_stop = entry_min - stop_dist
            if dow_low is not None and dow_low < entry_min:
                stop_price = min(raw_stop, dow_low)
            else:
                stop_price = raw_stop
            # Guard stop below entry
            stop_price = min(stop_price, entry_min - (atr14 * 0.2))

            reward = abs(target_price - entry_mid)
            risk = abs(entry_mid - stop_price)
            risk_reward = round(reward / max(risk, 1e-4), 2)
            expected_gross_bps = (reward / max(entry_mid, 1e-4)) * 10000.0
            expected_net_bps = round(expected_gross_bps - cost_floor_bps, 1)

        else:  # SHORT
            entry_min = ref_close
            entry_max = ref_close + depth
            if dow_high is not None and dow_high > ref_close and dow_high < (ref_close + depth):
                entry_max = dow_high
            entry_mid = (entry_min + entry_max) / 2.0

            target_price = entry_mid - target_move
            raw_stop = entry_max + stop_dist
            if dow_high is not None and dow_high > entry_max:
                stop_price = max(raw_stop, dow_high)
            else:
                stop_price = raw_stop
            # Guard stop above entry
            stop_price = max(stop_price, entry_max + (atr14 * 0.2))

            reward = abs(entry_mid - target_price)
            risk = abs(stop_price - entry_mid)
            risk_reward = round(reward / max(risk, 1e-4), 2)
            expected_gross_bps = (reward / max(entry_mid, 1e-4)) * 10000.0
            expected_net_bps = round(expected_gross_bps - cost_floor_bps, 1)

        zone_status = "ACTIVE"

    zone_id = f"zone_{open_time.replace(' ', '_').replace(':', '-')}"

    return {
        "zone_id": zone_id,
        "candle_open_time": open_time,
        "target_candle_open_time": target_time,
        "direction": direction,
        "ref_close": round(ref_close, 2),
        "entry_min": round(entry_min, 2),
        "entry_max": round(entry_max, 2),
        "entry_mid": round(entry_mid, 2),
        "target_price": round(target_price, 2),
        "stop_price": round(stop_price, 2),
        "risk_reward": risk_reward,
        "p_up": round(p_up, 4),
        "p_neutral": round(p_neu, 4),
        "p_down": round(p_dn, 4),
        "confidence": round(confidence, 4),
        "agreement_score": agreement_score,
        "regime": regime,
        "is_low_edge": is_low_edge,
        "has_conflicts": has_conflicts,
        "expected_net_bps": expected_net_bps,
        "status": zone_status,
        "exit_time": None,
        "exit_price": None,
        "net_bps": None,
        "evidence_chips": evidence_chips,
        "multipliers": {
            "k_entry": k_entry,
            "k_stop": k_stop,
            "scale": scale,
            "buffer_bps": buffer_bps,
            "cost_floor_bps": cost_floor_bps,
        },
    }


def evaluate_zone_resolution(
    zone: Dict[str, Any],
    subsequent_candle: Dict[str, Any],
    is_expiry_candle: bool = False,
    opposing_action_emitted: bool = False,
    new_confidence: Optional[float] = None,
    theta_dir: float = 0.32
) -> Dict[str, Any]:
    """Evaluate whether an active zone hit target, stopped out, flipped, or expired."""
    if zone.get("status") not in ["ACTIVE"]:
        return zone

    direction = zone["direction"]
    if direction == "NEUTRAL":
        updated = dict(zone)
        updated["status"] = "EXPIRED"
        return updated

    c_high = float(subsequent_candle.get("high", 0.0))
    c_low = float(subsequent_candle.get("low", 0.0))
    c_close = float(subsequent_candle.get("close", 0.0))
    c_time = str(subsequent_candle.get("open_time", ""))
    cost_floor_bps = float(zone.get("multipliers", {}).get("cost_floor_bps", 23.0))

    target = float(zone["target_price"])
    stop = float(zone["stop_price"])
    entry_mid = float(zone["entry_mid"])

    updated = dict(zone)

    # 1. Target Hit
    if (direction == "LONG" and c_high >= target) or (direction == "SHORT" and c_low <= target):
        updated["status"] = "TARGET_HIT"
        updated["exit_time"] = c_time
        updated["exit_price"] = target
        gross_bps = (abs(target - entry_mid) / entry_mid) * 10000.0
        updated["net_bps"] = round(gross_bps - cost_floor_bps, 1)
        return updated

    # 2. Stopped Out
    if (direction == "LONG" and c_low <= stop) or (direction == "SHORT" and c_high >= stop):
        updated["status"] = "STOPPED_OUT"
        updated["exit_time"] = c_time
        updated["exit_price"] = stop
        loss_bps = (abs(stop - entry_mid) / entry_mid) * 10000.0
        updated["net_bps"] = round(-loss_bps - cost_floor_bps, 1)
        return updated

    # 3. Early Flipped Exit or Low Confidence
    if opposing_action_emitted or (new_confidence is not None and new_confidence < theta_dir):
        updated["status"] = "FLIPPED_EXIT"
        updated["exit_time"] = c_time
        updated["exit_price"] = c_close
        sign = 1.0 if direction == "LONG" else -1.0
        gross_bps = sign * ((c_close - entry_mid) / entry_mid) * 10000.0
        updated["net_bps"] = round(gross_bps - cost_floor_bps, 1)
        return updated

    # 4. Expiry at t+2
    if is_expiry_candle:
        updated["status"] = "EXPIRED"
        updated["exit_time"] = c_time
        updated["exit_price"] = c_close
        sign = 1.0 if direction == "LONG" else -1.0
        gross_bps = sign * ((c_close - entry_mid) / entry_mid) * 10000.0
        updated["net_bps"] = round(gross_bps - cost_floor_bps, 1)
        return updated

    return updated
