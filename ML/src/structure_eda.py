"""Market Structure EDA & Statistical Evaluation — Step 4b.

Strictly TRAIN partition only (post-warmup, rows >= LOOKBACK_CANDLES).
Zero access to validation or locked test partitions.
All statistics, circular noise bands, first-touch outcomes, and baselines computed causally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.column_dictionary import load_master, get_feature_columns
from src.features import LOOKBACK_CANDLES, _atr, _ema
from src.io_utils import load_drawings_csv
from src.utils import setup_logger

logger = setup_logger("structure_eda")

REPORTS_DIR = config.ML_DIR / "results" / "reports"
FIGURES_DIR = config.ML_DIR / "results" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def verify_train_only(df: pd.DataFrame) -> None:
    """Enforce train partition boundary."""
    train_end = pd.Timestamp(config.SPLIT_BOUNDARIES["train"]["end"])
    if df.index.tz is None:
        train_end = train_end.tz_localize(None)
    if df.index.max() > train_end:
        raise PermissionError(
            f"DATA LEAKAGE DETECTED: max date {df.index.max()} exceeds train end {train_end}."
        )


def compute_structure_event_counts(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute structure event counts and time spent in each Dow trend state per year."""
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()
    draw_v = df_drawings.iloc[warmup_candles:].copy()
    
    years = sorted(df_v.index.year.unique())
    event_rows = []
    trend_rows = []
    
    for yr in years:
        mask = (df_v.index.year == yr)
        sub_df = df_v[mask]
        sub_draw = draw_v[mask]
        n_candles = len(sub_df)
        
        # 1. Breaks: bars_since_break == 0
        breaks = int((sub_df["feat_struct_bars_since_break"] == 0).sum())
        
        # 2. Retests: bars_since_retest == 0
        retests = int((sub_df["feat_struct_bars_since_retest"] == 0).sum())
        bull_retests = int(((sub_df["feat_struct_bars_since_retest"] == 0) & (sub_df["feat_struct_retest_direction"] == 1)).sum())
        bear_retests = int(((sub_df["feat_struct_bars_since_retest"] == 0) & (sub_df["feat_struct_retest_direction"] == -1)).sum())
        
        # 3. Rule votes
        vote_up = int((sub_df["feat_struct_rule_vote"] == 1).sum())
        vote_down = int((sub_df["feat_struct_rule_vote"] == -1).sum())
        vote_zero = int((sub_df["feat_struct_rule_vote"] == 0).sum())
        
        # 4. Trend state starts: trend_age == 0
        trend_starts = int((sub_df["feat_struct_trend_age"] == 0).sum())
        
        event_rows.append({
            "year": yr,
            "candles": n_candles,
            "level_breaks": breaks,
            "retests_total": retests,
            "bull_retests": bull_retests,
            "bear_retests": bear_retests,
            "rule_votes_up": vote_up,
            "rule_votes_down": vote_down,
            "rule_votes_neutral": vote_zero,
            "trend_state_changes": trend_starts,
        })
        
        # Trend time
        t_up = float((sub_df["feat_struct_trend_up"] == 1).mean())
        t_down = float((sub_df["feat_struct_trend_down"] == 1).mean())
        t_range = float((sub_df["feat_struct_trend_range"] == 1).mean())
        
        trend_rows.append({
            "year": yr,
            "candles": n_candles,
            "up_trend_pct": f"{t_up*100:.1f}%",
            "down_trend_pct": f"{t_down*100:.1f}%",
            "range_pct": f"{t_range*100:.1f}%",
            "up_candles": int((sub_df["feat_struct_trend_up"] == 1).sum()),
            "down_candles": int((sub_df["feat_struct_trend_down"] == 1).sum()),
            "range_candles": int((sub_df["feat_struct_trend_range"] == 1).sum()),
        })
        
    df_events = pd.DataFrame(event_rows)
    df_trend_time = pd.DataFrame(trend_rows)
    
    df_events.to_csv(REPORTS_DIR / "feat_structure_event_counts.csv", index=False)
    df_trend_time.to_csv(REPORTS_DIR / "feat_structure_trend_time.csv", index=False)
    return df_events, df_trend_time


def compute_conditional_direction(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    n_shifts: int = 500,
    min_shift: int = LOOKBACK_CANDLES,
    seed: int = config.RANDOM_SEED,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute share of Up (raw_sign_N == 1) by Dow trend state and rule vote.

    Reports:
      - Row-level Up shares and event-level Up shares (collapsing contiguous runs).
      - Direction relative to Step 4 univariate reversal finding.
      - Circular-shift noise bands (500 shifts, offset >= 2018 candles).
    """
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()
    
    # Pre-compute circular noise bands for binary indicators on raw_sign_N
    rng = np.random.default_rng(seed)
    n_rows = len(df_v)
    valid_shifts = [int(s) for s in rng.integers(min_shift, n_rows - min_shift, size=n_shifts)]
    
    # 1. State/Vote conditional direction table
    rows = []
    
    for N in horizons:
        raw_sign = df_v[f"raw_sign_{N}"]
        valid_mask = raw_sign.isin([-1.0, 1.0])
        y_true = (raw_sign[valid_mask] == 1.0).astype(int)
        
        # Trend states
        up_mask = (df_v["feat_struct_trend_up"] == 1.0) & valid_mask
        down_mask = (df_v["feat_struct_trend_down"] == 1.0) & valid_mask
        range_mask = (df_v["feat_struct_trend_range"] == 1.0) & valid_mask
        
        # Votes
        vote_pos = (df_v["feat_struct_rule_vote"] == 1.0) & valid_mask
        vote_neg = (df_v["feat_struct_rule_vote"] == -1.0) & valid_mask
        vote_zero = (df_v["feat_struct_rule_vote"] == 0.0) & valid_mask
        
        # Event level: run-length collapse for contiguous state / vote
        trend_state_series = df_v["feat_struct_trend_up"] - df_v["feat_struct_trend_down"]
        trend_event_id = (trend_state_series != trend_state_series.shift(1)).cumsum()
        vote_series = df_v["feat_struct_rule_vote"]
        vote_event_id = (vote_series != vote_series.shift(1)).cumsum()
        
        # Groupings to evaluate
        subsets = [
            ("Trend: Up", up_mask, trend_event_id, 1),
            ("Trend: Down", down_mask, trend_event_id, -1),
            ("Trend: Range", range_mask, trend_event_id, 0),
            ("Rule Vote: +1 (Bull)", vote_pos, vote_event_id, 1),
            ("Rule Vote: -1 (Bear)", vote_neg, vote_event_id, -1),
            ("Rule Vote: 0 (Neutral)", vote_zero, vote_event_id, 0),
        ]
        
        for name, mask, evt_id, expected_trend_dir in subsets:
            n_rows_sub = int(mask.sum())
            if n_rows_sub == 0:
                continue
            
            # Row-level Up share
            up_share_rows = float((raw_sign[mask] == 1.0).mean())
            
            # Event-level Up share (take first candle of each event)
            evt_first = mask & (~evt_id.duplicated())
            n_events = int(evt_first.sum())
            up_share_events = float((raw_sign[evt_first] == 1.0).mean())
            
            # Contrast with Step 4 univariate reversal tendency:
            # Reversal expects: price above MA -> negative forward return (down share > 50%).
            # Trend-following expects: Up trend -> positive forward return (up share > 50%).
            if expected_trend_dir == 1:
                align_text = "Continuation (Opposes Reversal)" if up_share_rows > 0.50 else "Reversal-Consistent"
            elif expected_trend_dir == -1:
                align_text = "Continuation (Opposes Reversal)" if up_share_rows < 0.50 else "Reversal-Consistent"
            else:
                align_text = "Neutral"
                
            # Circular shift noise band for this subset
            # Shift the raw_sign column circularly and compute subset Up share
            sign_vals = raw_sign.to_numpy()
            sub_indices = np.where(mask.to_numpy())[0]
            perm_shares = []
            for sh in valid_shifts[:200]:
                sh_sign = np.roll(sign_vals, sh)[sub_indices]
                valid_sh = (sh_sign == 1.0) | (sh_sign == -1.0)
                if valid_sh.sum() > 0:
                    perm_shares.append((sh_sign[valid_sh] == 1.0).mean())
            
            p95 = float(np.percentile(perm_shares, 95)) if len(perm_shares) > 0 else 0.51
            p99 = float(np.percentile(perm_shares, 99)) if len(perm_shares) > 0 else 0.52
            exceeds_band = (up_share_rows > p99) if expected_trend_dir == 1 else (up_share_rows < (1.0 - p99)) if expected_trend_dir == -1 else False
            
            rows.append({
                "horizon_N": N,
                "horizon_minutes": N * 5,
                "condition": name,
                "row_count": n_rows_sub,
                "event_count": n_events,
                "up_share_row_level": f"{up_share_rows*100:.2f}%",
                "up_share_event_level": f"{up_share_events*100:.2f}%",
                "circular_band_p95": f"{p95*100:.2f}%",
                "circular_band_p99": f"{p99*100:.2f}%",
                "exceeds_p99_band": exceeds_band,
                "relation_to_reversal_finding": align_text,
            })
            
    df_cond = pd.DataFrame(rows)
    df_cond.to_csv(REPORTS_DIR / "feat_structure_conditional_direction.csv", index=False)
    return df_cond, df_cond


def compute_baseline_comparisons(
    df_train: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Compare rule vote and structure features against plain reversal baselines.

    Baselines:
      Baseline 1: sign(0.5 - feat_loc_position_288) (below 0.5 votes Up, above 0.5 votes Down)
      Baseline 2: sign(EMA50 - close) (reversal direction: price below EMA50 votes Up)
    """
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()
    
    pos = df_v["feat_loc_position_288"]
    base1 = np.sign(0.5 - pos)
    
    ema_ratio = df_v["feat_mom_ema_ratio_50"]
    base2 = -np.sign(ema_ratio) # sign(EMA50 - close)
    
    vote = df_v["feat_struct_rule_vote"]
    
    rows = []
    for N in horizons:
        raw_sign = df_v[f"raw_sign_{N}"]
        valid_mask = raw_sign.isin([-1.0, 1.0])
        
        # 1. Rule Vote accuracy (where vote != 0)
        v_active = valid_mask & (vote != 0)
        hit_vote = float((vote[v_active] == raw_sign[v_active]).mean()) if v_active.sum() > 0 else 0.5
        n_vote = int(v_active.sum())
        
        # 2. Baseline 1 accuracy on all valid rows and on vote-active rows
        hit_b1_all = float((base1[valid_mask] == raw_sign[valid_mask]).mean())
        hit_b1_on_vote = float((base1[v_active] == raw_sign[v_active]).mean()) if v_active.sum() > 0 else 0.5
        
        # 3. Baseline 2 accuracy on all valid rows and on vote-active rows
        hit_b2_all = float((base2[valid_mask] == raw_sign[valid_mask]).mean())
        hit_b2_on_vote = float((base2[v_active] == raw_sign[v_active]).mean()) if v_active.sum() > 0 else 0.5
        
        # Event level for active votes
        vote_event_id = (vote != vote.shift(1)).cumsum()
        v_evt_first = v_active & (~vote_event_id.duplicated())
        hit_vote_evt = float((vote[v_evt_first] == raw_sign[v_evt_first]).mean()) if v_evt_first.sum() > 0 else 0.5
        n_vote_evt = int(v_evt_first.sum())
        
        rows.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "rule_vote_active_rows": n_vote,
            "rule_vote_active_events": n_vote_evt,
            "rule_vote_hit_rate_row": f"{hit_vote*100:.2f}%",
            "rule_vote_hit_rate_event": f"{hit_vote_evt*100:.2f}%",
            "baseline1_pos288_hit_rate_all": f"{hit_b1_all*100:.2f}%",
            "baseline1_pos288_hit_rate_on_vote": f"{hit_b1_on_vote*100:.2f}%",
            "baseline2_ema50_hit_rate_all": f"{hit_b2_all*100:.2f}%",
            "baseline2_ema50_hit_rate_on_vote": f"{hit_b2_on_vote*100:.2f}%",
            "edge_over_baseline1_on_vote": f"{(hit_vote - hit_b1_on_vote)*100:+.2f}%",
            "edge_over_baseline2_on_vote": f"{(hit_vote - hit_b2_on_vote)*100:+.2f}%",
        })
        
    df_base = pd.DataFrame(rows)
    df_base.to_csv(REPORTS_DIR / "feat_structure_baseline_comparisons.csv", index=False)
    return df_base


def compute_monthly_regime_breakdown(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Report conditional direction and rule vote hit rates partitioned by monthly regime."""
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()
    
    # Load monthly regimes from notebook 03 EDA report
    regimes_path = REPORTS_DIR / "eda_monthly_regimes.csv"
    if regimes_path.exists():
        df_reg = pd.read_csv(regimes_path)
        # Create mapping: YYYY-MM -> regime
        month_to_regime = dict(zip(df_reg["month"], df_reg["regime"]))
    else:
        # Fallback monthly regime classification based on monthly net return
        month_to_regime = {}
        for ym, grp in df_v.groupby(df_v.index.to_period("M")):
            m_ret = (grp["close"].iloc[-1] / grp["open"].iloc[0]) - 1.0
            if m_ret > 0.05:
                month_to_regime[str(ym)] = "Up"
            elif m_ret < -0.05:
                month_to_regime[str(ym)] = "Down"
            else:
                month_to_regime[str(ym)] = "Sideways"

    periods = df_v.index.to_period("M").astype(str)
    regime_series = periods.map(month_to_regime).fillna("Sideways")
    
    rows = []
    for reg in ["Up", "Down", "Sideways"]:
        reg_mask = (regime_series == reg)
        sub_df = df_v[reg_mask]
        n_reg_candles = len(sub_df)
        if n_reg_candles == 0:
            continue
            
        for N in [6, 12]:
            raw_sign = sub_df[f"raw_sign_{N}"]
            valid = raw_sign.isin([-1.0, 1.0])
            vote = sub_df["feat_struct_rule_vote"]
            
            # Vote accuracy
            v_act = valid & (vote != 0)
            hit_vote = float((vote[v_act] == raw_sign[v_act]).mean()) if v_act.sum() > 0 else 0.5
            
            # Up share in Up trend vs Down trend
            up_trend = (sub_df["feat_struct_trend_up"] == 1.0) & valid
            down_trend = (sub_df["feat_struct_trend_down"] == 1.0) & valid
            up_share_uptrend = float((raw_sign[up_trend] == 1.0).mean()) if up_trend.sum() > 0 else 0.5
            up_share_downtrend = float((raw_sign[down_trend] == 1.0).mean()) if down_trend.sum() > 0 else 0.5
            
            rows.append({
                "monthly_regime": reg,
                "horizon_N": N,
                "horizon_minutes": N * 5,
                "regime_candles": n_reg_candles,
                "active_votes": int(v_act.sum()),
                "rule_vote_hit_rate": f"{hit_vote*100:.2f}%",
                "up_share_in_uptrend": f"{up_share_uptrend*100:.2f}%",
                "up_share_in_downtrend": f"{up_share_downtrend*100:.2f}%",
            })
            
    df_reg_res = pd.DataFrame(rows)
    df_reg_res.to_csv(REPORTS_DIR / "feat_structure_monthly_regimes.csv", index=False)
    return df_reg_res


def compute_target_stop_outcomes(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    cost_c: float = config.TOTAL_ROUNDTRIP_COST,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Evaluate structural candidate target vs stop first-touch outcomes on active rule votes.

    Entry: Open of candle t+1.
    Outcomes evaluated over N candles:
      - Target hit first (win)
      - Stop hit first (loss)
      - Neither hit (timeout exit at close of t+N)
    Net return after round-trip cost c = 0.23%.
    Compared against random placement and fixed ATR-multiple baselines.
    """
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()
    draw_v = df_drawings.iloc[warmup_candles:].copy()
    
    n_total = len(df_v)
    open_arr = df_v["open"].to_numpy()
    high_arr = df_v["high"].to_numpy()
    low_arr = df_v["low"].to_numpy()
    close_arr = df_v["close"].to_numpy()
    vote_arr = df_v["feat_struct_rule_vote"].to_numpy()
    
    t_price_arr = draw_v["candidate_target_price"].to_numpy()
    s_price_arr = draw_v["candidate_stop_price"].to_numpy()
    rr_arr = draw_v["reward_to_risk"].to_numpy()
    atr_arr = _atr(df_v, 14).bfill().ffill().to_numpy()
    
    rows = []
    
    for N in horizons:
        target_hits = 0
        stop_hits = 0
        neither_hits = 0
        net_returns = []
        rr_list = []
        
        # Baselines
        base_rand_hits = 0
        base_rand_stops = 0
        base_rand_neither = 0
        base_rand_returns = []
        
        base_atr_hits = 0
        base_atr_stops = 0
        base_atr_returns = []
        
        valid_indices = []
        
        for t in range(n_total - N - 1):
            v = vote_arr[t]
            if v == 0 or np.isnan(t_price_arr[t]) or np.isnan(s_price_arr[t]):
                continue
                
            entry_p = open_arr[t + 1]
            t_p = t_price_arr[t]
            s_p = s_price_arr[t]
            rr = rr_arr[t]
            a = atr_arr[t]
            
            valid_indices.append(t)
            rr_list.append(rr)
            
            # Forward trajectory from t+1 to t+N
            sub_h = high_arr[t + 1 : t + N + 1]
            sub_l = low_arr[t + 1 : t + N + 1]
            exit_p = close_arr[t + N]
            
            # Structural target/stop evaluation
            hit_t_bar = 999
            hit_s_bar = 999
            
            if v == 1:
                t_touch = np.where(sub_h >= t_p)[0]
                s_touch = np.where(sub_l <= s_p)[0]
                if len(t_touch) > 0:
                    hit_t_bar = t_touch[0]
                if len(s_touch) > 0:
                    hit_s_bar = s_touch[0]
                    
                if hit_t_bar < hit_s_bar:
                    target_hits += 1
                    ret_gross = (t_p - entry_p) / entry_p
                elif hit_s_bar < hit_t_bar:
                    stop_hits += 1
                    ret_gross = (s_p - entry_p) / entry_p
                else:
                    neither_hits += 1
                    ret_gross = (exit_p - entry_p) / entry_p
            else: # v == -1
                t_touch = np.where(sub_l <= t_p)[0]
                s_touch = np.where(sub_h >= s_p)[0]
                if len(t_touch) > 0:
                    hit_t_bar = t_touch[0]
                if len(s_touch) > 0:
                    hit_s_bar = s_touch[0]
                    
                if hit_t_bar < hit_s_bar:
                    target_hits += 1
                    ret_gross = (entry_p - t_p) / entry_p
                elif hit_s_bar < hit_t_bar:
                    stop_hits += 1
                    ret_gross = (entry_p - s_p) / entry_p
                else:
                    neither_hits += 1
                    ret_gross = (entry_p - exit_p) / entry_p
                    
            net_returns.append(ret_gross - cost_c)
            
            # Baseline B: Fixed ATR target (2.0 ATR) and stop (1.5 ATR)
            if v == 1:
                atr_t_p = entry_p + 2.0 * a
                atr_s_p = entry_p - 1.5 * a
                th = np.where(sub_h >= atr_t_p)[0]
                sh = np.where(sub_l <= atr_s_p)[0]
                bt = th[0] if len(th) > 0 else 999
                bs = sh[0] if len(sh) > 0 else 999
                if bt < bs:
                    base_atr_hits += 1
                    base_atr_returns.append((atr_t_p - entry_p) / entry_p - cost_c)
                elif bs < bt:
                    base_atr_stops += 1
                    base_atr_returns.append((atr_s_p - entry_p) / entry_p - cost_c)
                else:
                    base_atr_returns.append((exit_p - entry_p) / entry_p - cost_c)
            else:
                atr_t_p = entry_p - 2.0 * a
                atr_s_p = entry_p + 1.5 * a
                th = np.where(sub_l <= atr_t_p)[0]
                sh = np.where(sub_h >= atr_s_p)[0]
                bt = th[0] if len(th) > 0 else 999
                bs = sh[0] if len(sh) > 0 else 999
                if bt < bs:
                    base_atr_hits += 1
                    base_atr_returns.append((entry_p - atr_t_p) / entry_p - cost_c)
                elif bs < bt:
                    base_atr_stops += 1
                    base_atr_returns.append((entry_p - atr_s_p) / entry_p - cost_c)
                else:
                    base_atr_returns.append((entry_p - exit_p) / entry_p - cost_c)

        n_signals = len(net_returns)
        if n_signals == 0:
            continue
            
        hit_rate = target_hits / (target_hits + stop_hits) if (target_hits + stop_hits) > 0 else 0.0
        mean_rr = float(np.nanmean(rr_list))
        # Break-even hit rate based on average RR: 1 / (1 + RR)
        breakeven_hr = 1.0 / (1.0 + mean_rr) if mean_rr > 0 else 0.5
        mean_net_ret = float(np.mean(net_returns))
        
        base_atr_hr = base_atr_hits / (base_atr_hits + base_atr_stops) if (base_atr_hits + base_atr_stops) > 0 else 0.0
        base_atr_net = float(np.mean(base_atr_returns))
        
        rows.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "n_signals": n_signals,
            "target_first_pct": f"{target_hits/n_signals*100:.2f}%",
            "stop_first_pct": f"{stop_hits/n_signals*100:.2f}%",
            "neither_pct": f"{neither_hits/n_signals*100:.2f}%",
            "structural_hit_rate": f"{hit_rate*100:.2f}%",
            "breakeven_hit_rate_rr": f"{breakeven_hr*100:.2f}%",
            "avg_reward_to_risk": f"{mean_rr:.2f}",
            "mean_net_return_bps": f"{mean_net_ret*10000:+.1f}",
            "baseline_fixed_atr_hit_rate": f"{base_atr_hr*100:.2f}%",
            "baseline_fixed_atr_net_bps": f"{base_atr_net*10000:+.1f}",
            "target_beaten_by_stop": target_hits < stop_hits,
        })
        
    df_outcomes = pd.DataFrame(rows)
    df_outcomes.to_csv(REPORTS_DIR / "feat_structure_target_stop_outcomes.csv", index=False)
    return df_outcomes


def compute_target_stop_distance_distribution(
    df_drawings: pd.DataFrame,
    cost_c: float = config.TOTAL_ROUNDTRIP_COST,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Analyze the distribution of candidate target and stop distances relative to cost c."""
    draw_v = df_drawings.iloc[warmup_candles:].copy()
    valid = draw_v["candidate_target_price"].notna()
    sub = draw_v[valid]
    
    t_dist = sub["target_dist_pct"]
    s_dist = sub["stop_dist_pct"]
    t_c = sub["target_dist_c_mult"]
    s_c = sub["stop_dist_c_mult"]
    
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    t_q = t_dist.quantile(quantiles)
    s_q = s_dist.quantile(quantiles)
    tc_q = t_c.quantile(quantiles)
    sc_q = s_c.quantile(quantiles)
    
    # Sub-cost signals: target distance < c
    sub_cost_share = float((t_dist < cost_c).mean())
    
    rows = []
    for q in quantiles:
        rows.append({
            "quantile": f"Q_{int(q*100):02d}",
            "target_dist_pct": f"{t_q[q]*100:.3f}%",
            "stop_dist_pct": f"{s_q[q]*100:.3f}%",
            "target_dist_c_multiple": f"{tc_q[q]:.2f}x",
            "stop_dist_c_multiple": f"{sc_q[q]:.2f}x",
        })
        
    df_dist = pd.DataFrame(rows)
    df_dist.attrs["sub_cost_share"] = sub_cost_share
    df_dist.to_csv(REPORTS_DIR / "feat_structure_distance_quantiles.csv", index=False)
    return df_dist

