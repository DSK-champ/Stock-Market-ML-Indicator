/**
 * Pure Mathematical Trade Zones Calculator (Client-Side Mirror)
 * Matches backend/live/zone_calculator.py exactly.
 */

(function (window) {
  "use strict";

  function computeGarmanKlassVolatility(candles, n) {
    n = n || 10;
    if (!candles || candles.length < 2) return 0.0035;

    const slice = candles.slice(-n);
    let sumGk = 0;
    let count = 0;

    for (let i = 0; i < slice.length; i++) {
      const h = Number(slice[i].high);
      const l = Number(slice[i].low);
      const c = Number(slice[i].close);
      const o = Number(slice[i].open);

      if (h > 0 && l > 0 && c > 0 && o > 0 && h >= l) {
        const logHL = Math.log(h / Math.max(l, 1e-8));
        const logCO = Math.log(c / Math.max(o, 1e-8));
        const gk = 0.5 * (logHL * logHL) - (2.0 * Math.LN2 - 1.0) * (logCO * logCO);
        sumGk += Math.max(0, gk);
        count++;
      }
    }

    if (count === 0) return 0.0035;
    const sigma = Math.sqrt(sumGk / count);
    return Math.max(0.0010, Math.min(sigma, 0.0500));
  }

  function findDowSwings(candles, lookback) {
    lookback = lookback || 12;
    if (!candles || candles.length < 4) return { swingHigh: null, swingLow: null };

    const slice = candles.slice(-lookback);
    let high = -Infinity;
    let low = Infinity;

    for (let i = 0; i < slice.length; i++) {
      const h = Number(slice[i].high);
      const l = Number(slice[i].low);
      if (!isNaN(h) && h > high) high = h;
      if (!isNaN(l) && l < low) low = l;
    }

    return {
      swingHigh: high !== -Infinity ? high : null,
      swingLow: low !== Infinity ? low : null,
    };
  }

  function classifyRegime(refClose, atr14, gkVol, trendLabel, trendScore, cfg) {
    const highVolThresh = cfg.high_vol_atr_threshold || 0.0070;
    const atrRatio = atr14 / Math.max(refClose, 1e-4);

    if (atrRatio >= highVolThresh || gkVol >= 0.0065) return "high-volatility";
    const tl = String(trendLabel || "").toUpperCase();
    if (tl === "UPTREND" || tl === "DOWNTREND" || Math.abs(trendScore) >= 0.35) return "trending";
    return "ranging";
  }

  function computeAgreementScore(direction, baseModelVotes, taScores) {
    const chips = [];
    if (direction === "NEUTRAL") {
      return {
        score: 50.0,
        chips: [
          { id: "neutral_standby", name: "Neutral Baseline", value: "NEUTRAL", bullish: false, bearish: false, agrees: true }
        ],
      };
    }

    let votesFor = 0;
    let totalVotes = 0;
    const baseVotes = baseModelVotes || {};
    const ta = taScores || {};

    // 1. Binary Base Models (3)
    const binModels = [
      ["bin_lr", "Binary LR", baseVotes.binary_lr_p_up != null ? baseVotes.binary_lr_p_up : (baseVotes.bin_lr_p_up != null ? baseVotes.bin_lr_p_up : 0.5)],
      ["bin_rf", "Binary RF", baseVotes.binary_rf_p_up != null ? baseVotes.binary_rf_p_up : (baseVotes.bin_rf_p_up != null ? baseVotes.bin_rf_p_up : 0.5)],
      ["bin_gb", "Binary GB", baseVotes.binary_gb_p_up != null ? baseVotes.binary_gb_p_up : (baseVotes.bin_gb_p_up != null ? baseVotes.bin_gb_p_up : 0.5)],
    ];
    binModels.forEach(([id, name, pUp]) => {
      const val = Number(pUp);
      const isBull = val > 0.50;
      const isBear = val < 0.50;
      const agrees = (direction === "LONG" && isBull) || (direction === "SHORT" && isBear);
      if (agrees) votesFor++;
      totalVotes++;
      chips.push({
        id,
        name,
        value: `${Math.round(val * 100)}%`,
        bullish: isBull,
        bearish: isBear,
        agrees,
      });
    });

    // 2. Ternary Base Models (3)
    const triModels = [
      ["tri_lr", "3-Class LR", baseVotes.tri_lr_p_up != null ? baseVotes.tri_lr_p_up : 0.33, baseVotes.tri_lr_p_down != null ? baseVotes.tri_lr_p_down : 0.33],
      ["tri_rf", "3-Class RF", baseVotes.tri_rf_p_up != null ? baseVotes.tri_rf_p_up : 0.33, baseVotes.tri_rf_p_down != null ? baseVotes.tri_rf_p_down : 0.33],
      ["tri_gb", "3-Class GB", baseVotes.tri_gb_p_up != null ? baseVotes.tri_gb_p_up : 0.33, baseVotes.tri_gb_p_down != null ? baseVotes.tri_gb_p_down : 0.33],
    ];
    triModels.forEach(([id, name, pUp, pDn]) => {
      const u = Number(pUp);
      const d = Number(pDn);
      const isBull = u > d && u >= 0.34;
      const isBear = d > u && d >= 0.34;
      const agrees = (direction === "LONG" && isBull) || (direction === "SHORT" && isBear);
      if (agrees) votesFor++;
      totalVotes++;
      chips.push({
        id,
        name,
        value: `U:${Math.round(u * 100)}% D:${Math.round(d * 100)}%`,
        bullish: isBull,
        bearish: isBear,
        agrees,
      });
    });

    // 3. TA Scores (6)
    const taItems = [
      ["ta_rsi", "TA RSI", ta.ta_rsi != null ? ta.ta_rsi : (ta.rsi_score != null ? ta.rsi_score : 0.0)],
      ["ta_ema50", "TA EMA50", ta.ta_ema50 != null ? ta.ta_ema50 : (ta.ema50_score != null ? ta.ema50_score : 0.0)],
      ["ta_align", "TA Align", ta.ta_align != null ? ta.ta_align : (ta.alignment_score != null ? ta.alignment_score : 0.0)],
      ["ta_loc", "TA Location", ta.ta_loc != null ? ta.ta_loc : (ta.location_score != null ? ta.location_score : 0.0)],
      ["ta_rule_vote", "TA Rules", ta.ta_rule_vote != null ? ta.ta_rule_vote : (ta.rule_vote != null ? ta.rule_vote : 0.0)],
      ["ta_trend", "TA Trend", ta.ta_trend != null ? ta.ta_trend : (ta.trend_score != null ? ta.trend_score : 0.0)],
    ];
    taItems.forEach(([id, name, rawVal]) => {
      const v = Number(rawVal);
      let isBull = false;
      let isBear = false;
      if (id === "ta_rsi" && v > 1.0) {
        isBull = v > 50.0;
        isBear = v < 50.0;
      } else {
        isBull = v > 0.05;
        isBear = v < -0.05;
      }
      const agrees = (direction === "LONG" && isBull) || (direction === "SHORT" && isBear);
      if (agrees) votesFor++;
      totalVotes++;
      chips.push({
        id,
        name,
        value: Math.abs(v) <= 1.0 ? (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2)) : v.toFixed(1),
        bullish: isBull,
        bearish: isBear,
        agrees,
      });
    });

    const score = Math.round((votesFor / Math.max(totalVotes, 1)) * 1000) / 10;
    return { score, chips };
  }

  function calculateTradeZoneClient(pred, recentCandles, configOverride) {
    const cfg = Object.assign({}, window.ZONE_CONFIG || {}, configOverride || {});

    const openTime = String(pred.candle_open_time || pred.open_time || "");
    const targetTime = String(pred.target_candle_open_time || pred.target_time || "");
    let refClose = Number(pred.ref_close || pred.close || 0);

    if (refClose <= 0 && recentCandles && recentCandles.length > 0) {
      refClose = Number(recentCandles[recentCandles.length - 1].close || 0);
    }

    const rawAction = pred.shown_action || pred.action_label || pred.action || 0;
    let direction = "NEUTRAL";
    if (typeof rawAction === "number") {
      direction = rawAction === 1 ? "LONG" : (rawAction === -1 ? "SHORT" : "NEUTRAL");
    } else {
      const actStr = String(rawAction).trim().toUpperCase();
      if (actStr === "UP" || actStr === "LONG" || actStr === "1") direction = "LONG";
      else if (actStr === "DOWN" || actStr === "SHORT" || actStr === "-1") direction = "SHORT";
    }

    const probs = pred.probabilities || {};
    const pUp = Number(probs.p_up != null ? probs.p_up : (pred.p_up != null ? pred.p_up : 0.3333));
    const pNeu = Number(probs.p_neutral != null ? probs.p_neutral : (pred.p_neutral != null ? pred.p_neutral : 0.3334));
    const pDn = Number(probs.p_down != null ? probs.p_down : (pred.p_down != null ? pred.p_down : 0.3333));
    const confidence = Number(pred.confidence != null ? pred.confidence : Math.max(pUp, pDn, pNeu));
    const margin = Math.abs(pUp - pDn);

    let atr14 = Number(pred.atr14 || 0);
    if (atr14 <= 0) atr14 = refClose * 0.0035;

    const gkVol = computeGarmanKlassVolatility(recentCandles, 10);
    const swings = findDowSwings(recentCandles, cfg.swing_lookback || 12);

    const tech = pred.technical_analysis || {};
    const trendLabel = String(tech.trend || pred.trend || "uncertain");
    const trendScore = Number(tech.trend_score || 0);
    const regime = classifyRegime(refClose, atr14, gkVol, trendLabel, trendScore, cfg);

    const baseVotes = pred.base_model_votes || {};
    const agreement = computeAgreementScore(direction, baseVotes, tech);
    const hasConflicts = (direction !== "NEUTRAL") && (agreement.score < 50.0);

    const kEntry = Number(cfg.k_entry || 0.75);
    const kStop = Number(cfg.k_stop || 1.25);
    const scale = Number(cfg.scale || 1.50);
    const bufferBps = Number(cfg.buffer_bps || 5.0);
    const costFloorBps = Number(cfg.cost_floor_bps || 23.0);
    const highVolMult = Number(cfg.high_vol_multiplier || 1.35);

    let entryMin = refClose;
    let entryMax = refClose;
    let entryMid = refClose;
    let targetPrice = refClose;
    let stopPrice = refClose;
    let riskReward = 0.0;
    let expectedNetBps = 0.0;
    let isLowEdge = false;
    let zoneStatus = "NEUTRAL_BAND";

    if (direction !== "NEUTRAL") {
      const depth = atr14 * kEntry * Math.max(0.40, (1.0 - 0.50 * margin));
      const expectedMove = refClose * gkVol * margin * scale;
      const frictionMove = refClose * ((costFloorBps + bufferBps) / 10000.0);
      isLowEdge = expectedMove < (refClose * (costFloorBps / 10000.0));
      const targetMove = Math.max(expectedMove, frictionMove);

      let stopDist = refClose * gkVol * kStop;
      if (regime === "high-volatility") stopDist *= highVolMult;

      if (direction === "LONG") {
        entryMax = refClose;
        entryMin = refClose - depth;
        if (swings.swingLow != null && swings.swingLow < refClose && swings.swingLow > (refClose - depth)) {
          entryMin = swings.swingLow;
        }
        entryMid = (entryMin + entryMax) / 2.0;
        targetPrice = entryMid + targetMove;

        const rawStop = entryMin - stopDist;
        stopPrice = (swings.swingLow != null && swings.swingLow < entryMin) ? Math.min(rawStop, swings.swingLow) : rawStop;
        stopPrice = Math.min(stopPrice, entryMin - (atr14 * 0.2));

        const reward = Math.abs(targetPrice - entryMid);
        const risk = Math.abs(entryMid - stopPrice);
        riskReward = Math.round((reward / Math.max(risk, 1e-4)) * 100) / 100;
        const grossBps = (reward / Math.max(entryMid, 1e-4)) * 10000.0;
        expectedNetBps = Math.round((grossBps - costFloorBps) * 10) / 10;
      } else {
        // SHORT
        entryMin = refClose;
        entryMax = refClose + depth;
        if (swings.swingHigh != null && swings.swingHigh > refClose && swings.swingHigh < (refClose + depth)) {
          entryMax = swings.swingHigh;
        }
        entryMid = (entryMin + entryMax) / 2.0;
        targetPrice = entryMid - targetMove;

        const rawStop = entryMax + stopDist;
        stopPrice = (swings.swingHigh != null && swings.swingHigh > entryMax) ? Math.max(rawStop, swings.swingHigh) : rawStop;
        stopPrice = Math.max(stopPrice, entryMax + (atr14 * 0.2));

        const reward = Math.abs(entryMid - targetPrice);
        const risk = Math.abs(stopPrice - entryMid);
        riskReward = Math.round((reward / Math.max(risk, 1e-4)) * 100) / 100;
        const grossBps = (reward / Math.max(entryMid, 1e-4)) * 10000.0;
        expectedNetBps = Math.round((grossBps - costFloorBps) * 10) / 10;
      }

      zoneStatus = "ACTIVE";
    }

    const zoneId = `zone_${openTime.replace(/ /g, "_").replace(/:/g, "-")}`;

    return {
      zone_id: zoneId,
      candle_open_time: openTime,
      target_candle_open_time: targetTime,
      direction,
      ref_close: Math.round(refClose * 100) / 100,
      entry_min: Math.round(entryMin * 100) / 100,
      entry_max: Math.round(entryMax * 100) / 100,
      entry_mid: Math.round(entryMid * 100) / 100,
      target_price: Math.round(targetPrice * 100) / 100,
      stop_price: Math.round(stopPrice * 100) / 100,
      risk_reward: riskReward,
      p_up: Math.round(pUp * 10000) / 10000,
      p_neutral: Math.round(pNeu * 10000) / 10000,
      p_down: Math.round(pDn * 10000) / 10000,
      confidence: Math.round(confidence * 10000) / 10000,
      agreement_score: agreement.score,
      regime,
      is_low_edge: isLowEdge,
      has_conflicts: hasConflicts,
      expected_net_bps: expectedNetBps,
      status: zoneStatus,
      exit_time: null,
      exit_price: null,
      net_bps: null,
      evidence_chips: agreement.chips,
      multipliers: {
        k_entry: kEntry,
        k_stop: kStop,
        scale,
        buffer_bps: bufferBps,
        cost_floor_bps: costFloorBps,
      },
    };
  }

  window.ZoneCalculator = {
    computeGarmanKlassVolatility,
    findDowSwings,
    classifyRegime,
    computeAgreementScore,
    calculateTradeZone: calculateTradeZoneClient,
  };
})(window);
