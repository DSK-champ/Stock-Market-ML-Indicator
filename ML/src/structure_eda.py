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
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Compute true structural event counts, median trend duration, and trend time per year.

    True events:
      - Trend segments: transitions into new Dow trend state (+1, -1, 0)
      - Breaks: transitions into break (bars_since_break == 0 after != 0)
      - Retests: transitions into retest (bars_since_retest == 0 after != 0), split bull/bear
      - Rule votes: transitions into active vote (+1 or -1)
    """
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()

    # Trend state: +1 Up, -1 Down, 0 Range
    trend_state = df_v["feat_struct_trend_up"] - df_v["feat_struct_trend_down"]
    is_new_trend = (trend_state != trend_state.shift(1))
    is_new_trend.iloc[0] = True

    # Segment durations
    segment_ids = is_new_trend.cumsum()
    segment_durations = df_v.groupby(segment_ids).size()
    median_duration_all = float(segment_durations.median())

    up_durations = df_v[df_v["feat_struct_trend_up"] == 1.0].groupby(segment_ids).size()
    down_durations = df_v[df_v["feat_struct_trend_down"] == 1.0].groupby(segment_ids).size()
    range_durations = df_v[df_v["feat_struct_trend_range"] == 1.0].groupby(segment_ids).size()

    durations_dict = {
        "median_trend_duration_all": median_duration_all,
        "median_trend_duration_up": float(up_durations.median()) if len(up_durations) > 0 else 0.0,
        "median_trend_duration_down": float(down_durations.median()) if len(down_durations) > 0 else 0.0,
        "median_trend_duration_range": float(range_durations.median()) if len(range_durations) > 0 else 0.0,
    }

    # Retest transitions
    is_retest_event = (df_v["feat_struct_bars_since_retest"] == 0) & (df_v["feat_struct_bars_since_retest"].shift(1) != 0)
    is_bull_retest = is_retest_event & (df_v["feat_struct_retest_direction"] == 1)
    is_bear_retest = is_retest_event & (df_v["feat_struct_retest_direction"] == -1)

    # Vote transitions
    vote = df_v["feat_struct_rule_vote"]
    is_vote_up_event = (vote == 1) & (vote.shift(1) != 1)
    is_vote_down_event = (vote == -1) & (vote.shift(1) != -1)

    # Break transitions
    is_break_event = (df_v["feat_struct_bars_since_break"] == 0) & (df_v["feat_struct_bars_since_break"].shift(1) != 0)

    event_df = pd.DataFrame({
        "year": df_v.index.year,
        "trend_segments": is_new_trend.astype(int),
        "break_events": is_break_event.astype(int),
        "retest_events": is_retest_event.astype(int),
        "bull_retests": is_bull_retest.astype(int),
        "bear_retests": is_bear_retest.astype(int),
        "vote_up_events": is_vote_up_event.astype(int),
        "vote_down_events": is_vote_down_event.astype(int),
    }, index=df_v.index)

    df_events = event_df.groupby("year").sum().reset_index()
    candle_counts = df_v.groupby(df_v.index.year).size().reset_index()
    candle_counts.columns = ["year", "candles"]
    df_events = pd.merge(candle_counts, df_events, on="year")

    # Trend time share
    trend_rows = []
    for yr, sub in df_v.groupby(df_v.index.year):
        n_candles = len(sub)
        t_up = float((sub["feat_struct_trend_up"] == 1).mean())
        t_down = float((sub["feat_struct_trend_down"] == 1).mean())
        t_range = float((sub["feat_struct_trend_range"] == 1).mean())
        trend_rows.append({
            "year": yr,
            "candles": n_candles,
            "up_trend_pct": f"{t_up*100:.1f}%",
            "down_trend_pct": f"{t_down*100:.1f}%",
            "range_pct": f"{t_range*100:.1f}%",
            "up_candles": int((sub["feat_struct_trend_up"] == 1).sum()),
            "down_candles": int((sub["feat_struct_trend_down"] == 1).sum()),
            "range_candles": int((sub["feat_struct_trend_range"] == 1).sum()),
        })
    df_trend_time = pd.DataFrame(trend_rows)

    df_events.to_csv(REPORTS_DIR / "feat_structure_event_counts.csv", index=False)
    df_trend_time.to_csv(REPORTS_DIR / "feat_structure_trend_time.csv", index=False)
    return df_events, df_trend_time, durations_dict


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
      - Row-level Up shares and event-level Up shares (transitions).
      - Standard error SE = sqrt(p*(1-p)/n).
      - Two-sided circular-shift noise bands (500 shifts, offset >= 2018 candles) built on same rows/events.
      - 95% two-sided [p_2.5, p_97.5] and 99% two-sided [p_0.5, p_99.5].
      - Exceeds 99th band? True if outside [p_0.5, p_99.5].
    """
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()

    rng = np.random.default_rng(seed)
    n_rows = len(df_v)
    valid_shifts = [int(s) for s in rng.integers(min_shift, n_rows - min_shift, size=n_shifts)]

    rows = []

    for N in horizons:
        raw_sign = df_v[f"raw_sign_{N}"]
        valid_mask = raw_sign.isin([-1.0, 1.0])

        up_mask = (df_v["feat_struct_trend_up"] == 1.0) & valid_mask
        down_mask = (df_v["feat_struct_trend_down"] == 1.0) & valid_mask
        range_mask = (df_v["feat_struct_trend_range"] == 1.0) & valid_mask

        vote_pos = (df_v["feat_struct_rule_vote"] == 1.0) & valid_mask
        vote_neg = (df_v["feat_struct_rule_vote"] == -1.0) & valid_mask
        vote_zero = (df_v["feat_struct_rule_vote"] == 0.0) & valid_mask

        # Event transitions
        trend_state = df_v["feat_struct_trend_up"] - df_v["feat_struct_trend_down"]
        trend_trans = (trend_state != trend_state.shift(1)) & valid_mask
        vote_series = df_v["feat_struct_rule_vote"]
        vote_trans = (vote_series != vote_series.shift(1)) & valid_mask

        subsets = [
            ("Trend: Up", up_mask, up_mask & trend_trans, 1),
            ("Trend: Down", down_mask, down_mask & trend_trans, -1),
            ("Trend: Range", range_mask, range_mask & trend_trans, 0),
            ("Rule Vote: +1 (Bull)", vote_pos, vote_pos & vote_trans, 1),
            ("Rule Vote: -1 (Bear)", vote_neg, vote_neg & vote_trans, -1),
            ("Rule Vote: 0 (Neutral)", vote_zero, vote_zero & vote_trans, 0),
        ]

        sign_vals = raw_sign.to_numpy()

        for name, r_mask, e_mask, expected_trend_dir in subsets:
            n_rows_sub = int(r_mask.sum())
            n_events_sub = int(e_mask.sum())
            if n_rows_sub == 0 or n_events_sub == 0:
                continue

            up_share_rows = float((raw_sign[r_mask] == 1.0).mean())
            up_share_events = float((raw_sign[e_mask] == 1.0).mean())

            se_rows = np.sqrt(up_share_rows * (1.0 - up_share_rows) / n_rows_sub)
            se_events = np.sqrt(up_share_events * (1.0 - up_share_events) / n_events_sub)

            # Contrast with Step 4 univariate reversal tendency:
            if expected_trend_dir == 1:
                align_text = "Continuation (Opposes Reversal)" if up_share_rows > 0.50 else "Reversal-Consistent"
            elif expected_trend_dir == -1:
                align_text = "Continuation (Opposes Reversal)" if up_share_rows < 0.50 else "Reversal-Consistent"
            else:
                align_text = "Neutral"

            # Two-sided circular shift noise band built from the SAME rows
            sub_indices = np.where(r_mask.to_numpy())[0]
            perm_shares = []
            for sh in valid_shifts:
                sh_sign = np.roll(sign_vals, sh)[sub_indices]
                valid_sh = (sh_sign == 1.0) | (sh_sign == -1.0)
                if valid_sh.sum() > 0:
                    perm_shares.append(float((sh_sign[valid_sh] == 1.0).mean()))

            p05 = float(np.percentile(perm_shares, 0.5)) if len(perm_shares) > 0 else 0.49
            p995 = float(np.percentile(perm_shares, 99.5)) if len(perm_shares) > 0 else 0.51
            p25 = float(np.percentile(perm_shares, 2.5)) if len(perm_shares) > 0 else 0.495
            p975 = float(np.percentile(perm_shares, 97.5)) if len(perm_shares) > 0 else 0.505

            exceeds_99_two_sided = (up_share_rows < p05) or (up_share_rows > p995)

            rows.append({
                "horizon_N": N,
                "horizon_minutes": N * 5,
                "condition": name,
                "row_count": n_rows_sub,
                "event_count": n_events_sub,
                "up_share_row": f"{up_share_rows*100:.2f}%",
                "se_row": f"{se_rows*100:.2f}%",
                "up_share_event": f"{up_share_events*100:.2f}%",
                "se_event": f"{se_events*100:.2f}%",
                "circular_band_95_two_sided": f"[{p25*100:.2f}%, {p975*100:.2f}%]",
                "circular_band_99_two_sided": f"[{p05*100:.2f}%, {p995*100:.2f}%]",
                "exceeds_99th_band": exceeds_99_two_sided,
                "reversal_contrast": align_text,
            })

    df_cond = pd.DataFrame(rows)
    df_cond.to_csv(REPORTS_DIR / "feat_structure_conditional_direction.csv", index=False)
    return df_cond, df_cond


def compute_baseline_comparisons(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Compare naive structural rule vote against plain reversal baselines."""
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()

    pos288 = df_v["feat_loc_position_288"].to_numpy()
    close_val = df_v["close"].to_numpy()
    ema50_val = _ema(df_v["close"], 50).bfill().ffill().to_numpy()

    rule_b1_all = np.where(pos288 < 0.5, 1.0, -1.0)
    rule_b2_all = np.where(close_val < ema50_val, 1.0, -1.0)

    vote_arr = df_v["feat_struct_rule_vote"].to_numpy()
    rows = []

    for N in horizons:
        raw_sign = df_v[f"raw_sign_{N}"].to_numpy()
        valid = (raw_sign == 1.0) | (raw_sign == -1.0)

        v_act_mask = valid & (vote_arr != 0)
        n_act_rows = int(v_act_mask.sum())
        if n_act_rows == 0:
            continue

        hr_vote_row = float((vote_arr[v_act_mask] == raw_sign[v_act_mask]).mean())

        # Event level
        vote_series = df_v["feat_struct_rule_vote"]
        vote_trans = (vote_series != vote_series.shift(1)).to_numpy()
        v_act_evt_mask = v_act_mask & vote_trans
        n_act_evts = int(v_act_evt_mask.sum())
        hr_vote_evt = float((vote_arr[v_act_evt_mask] == raw_sign[v_act_evt_mask]).mean()) if n_act_evts > 0 else 0.5

        hr_b1_all = float((rule_b1_all[valid] == raw_sign[valid]).mean())
        hr_b1_on_vote = float((rule_b1_all[v_act_mask] == raw_sign[v_act_mask]).mean())

        hr_b2_all = float((rule_b2_all[valid] == raw_sign[valid]).mean())
        hr_b2_on_vote = float((rule_b2_all[v_act_mask] == raw_sign[v_act_mask]).mean())

        edge_b1 = hr_vote_row - hr_b1_on_vote
        edge_b2 = hr_vote_row - hr_b2_on_vote

        rows.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "rule_vote_active_rows": n_act_rows,
            "rule_vote_active_events": n_act_evts,
            "rule_vote_hit_rate_row": f"{hr_vote_row*100:.2f}%",
            "rule_vote_hit_rate_event": f"{hr_vote_evt*100:.2f}%",
            "baseline1_pos288_hit_rate_all": f"{hr_b1_all*100:.2f}%",
            "baseline1_pos288_hit_rate_on_vote": f"{hr_b1_on_vote*100:.2f}%",
            "baseline2_ema50_hit_rate_all": f"{hr_b2_all*100:.2f}%",
            "baseline2_ema50_hit_rate_on_vote": f"{hr_b2_on_vote*100:.2f}%",
            "edge_over_baseline1_on_vote": f"{edge_b1*100:.2f}%",
            "edge_over_baseline2_on_vote": f"{edge_b2*100:.2f}%",
        })

    df_base = pd.DataFrame(rows)
    df_base.to_csv(REPORTS_DIR / "feat_structure_baseline_comparisons.csv", index=False)
    return df_base


def compute_monthly_regime_breakdown(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Evaluate structure signals split by Up, Down, and Sideways monthly regimes."""
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()

    regimes_csv = REPORTS_DIR / "eda_monthly_regimes.csv"
    if regimes_csv.exists():
        df_reg = pd.read_csv(regimes_csv)
        month_to_regime = dict(zip(df_reg["month"], df_reg["regime"]))
    else:
        month_to_regime = {}
        for ym, grp in df_v.groupby(df_v.index.to_period("M")):
            m_ret = (grp["close"].iloc[-1] / grp["open"].iloc[0]) - 1.0
            month_to_regime[str(ym)] = "Up" if m_ret > 0.05 else ("Down" if m_ret < -0.05 else "Sideways")

    periods = df_v.index.to_period("M").astype(str)
    regime_series = periods.map(month_to_regime).fillna("Sideways")

    rows = []
    for reg in ["Up", "Down", "Sideways"]:
        reg_mask = (regime_series == reg)
        sub_df = df_v[reg_mask]
        n_reg_candles = len(sub_df)
        if n_reg_candles == 0:
            continue

        for N in config.CANDIDATE_HORIZONS:
            raw_sign = sub_df[f"raw_sign_{N}"]
            valid = raw_sign.isin([-1.0, 1.0])
            vote = sub_df["feat_struct_rule_vote"]

            v_act = valid & (vote != 0)
            hit_vote = float((vote[v_act] == raw_sign[v_act]).mean()) if v_act.sum() > 0 else 0.5

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

    Formula:
      p* = (S + c) / (T + S) per signal, where T and S are target and stop fractions of price.
      Average over all signals: breakeven_p_star.
      Entry: Open of candle t+1.
      Hit rates:
        - target_first / (target_first + stop_first)  [conditional]
        - target_first / all_signals                  [unconditional]
      Both shown for structural and fixed ATR baselines.
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
    atr_arr = _atr(df_v, 14).bfill().ffill().to_numpy()

    rows = []

    for N in horizons:
        target_hits = 0
        stop_hits = 0
        neither_hits = 0
        net_returns = []
        p_star_list = []

        base_atr_hits = 0
        base_atr_stops = 0
        base_atr_returns = []
        base_atr_p_star_list = []

        for t in range(n_total - N - 1):
            v = vote_arr[t]
            if v == 0 or np.isnan(t_price_arr[t]) or np.isnan(s_price_arr[t]):
                continue

            entry_p = open_arr[t + 1]
            t_p = t_price_arr[t]
            s_p = s_price_arr[t]
            a = atr_arr[t]

            # Structural T and S distances as fractions of entry price
            T_frac = abs(t_p - entry_p) / entry_p
            S_frac = abs(s_p - entry_p) / entry_p
            denom = T_frac + S_frac
            if denom > 1e-12:
                p_star = (S_frac + cost_c) / denom
                p_star_list.append(p_star)

            # Fixed ATR T and S distances (2.0 ATR target, 1.5 ATR stop)
            T_atr = 2.0 * a / entry_p
            S_atr = 1.5 * a / entry_p
            denom_atr = T_atr + S_atr
            if denom_atr > 1e-12:
                base_atr_p_star_list.append((S_atr + cost_c) / denom_atr)

            sub_h = high_arr[t + 1 : t + N + 1]
            sub_l = low_arr[t + 1 : t + N + 1]
            exit_p = close_arr[t + N]

            # Structural first-touch check
            if v == 1:
                t_touch = np.where(sub_h >= t_p)[0]
                s_touch = np.where(sub_l <= s_p)[0]
                hit_t = t_touch[0] if len(t_touch) > 0 else 999
                hit_s = s_touch[0] if len(s_touch) > 0 else 999
                if hit_t < hit_s:
                    target_hits += 1
                    net_returns.append((t_p - entry_p) / entry_p - cost_c)
                elif hit_s < hit_t:
                    stop_hits += 1
                    net_returns.append((s_p - entry_p) / entry_p - cost_c)
                else:
                    neither_hits += 1
                    net_returns.append((exit_p - entry_p) / entry_p - cost_c)
            else:
                t_touch = np.where(sub_l <= t_p)[0]
                s_touch = np.where(sub_h >= s_p)[0]
                hit_t = t_touch[0] if len(t_touch) > 0 else 999
                hit_s = s_touch[0] if len(s_touch) > 0 else 999
                if hit_t < hit_s:
                    target_hits += 1
                    net_returns.append((entry_p - t_p) / entry_p - cost_c)
                elif hit_s < hit_t:
                    stop_hits += 1
                    net_returns.append((entry_p - s_p) / entry_p - cost_c)
                else:
                    neither_hits += 1
                    net_returns.append((entry_p - exit_p) / entry_p - cost_c)

            # Fixed ATR first-touch check
            if v == 1:
                atr_tp = entry_p + 2.0 * a
                atr_sp = entry_p - 1.5 * a
                th = np.where(sub_h >= atr_tp)[0]
                sh = np.where(sub_l <= atr_sp)[0]
                bt = th[0] if len(th) > 0 else 999
                bs = sh[0] if len(sh) > 0 else 999
                if bt < bs:
                    base_atr_hits += 1
                    base_atr_returns.append((atr_tp - entry_p) / entry_p - cost_c)
                elif bs < bt:
                    base_atr_stops += 1
                    base_atr_returns.append((atr_sp - entry_p) / entry_p - cost_c)
                else:
                    base_atr_returns.append((exit_p - entry_p) / entry_p - cost_c)
            else:
                atr_tp = entry_p - 2.0 * a
                atr_sp = entry_p + 1.5 * a
                th = np.where(sub_l <= atr_tp)[0]
                sh = np.where(sub_h >= atr_sp)[0]
                bt = th[0] if len(th) > 0 else 999
                bs = sh[0] if len(sh) > 0 else 999
                if bt < bs:
                    base_atr_hits += 1
                    base_atr_returns.append((entry_p - atr_tp) / entry_p - cost_c)
                elif bs < bt:
                    base_atr_stops += 1
                    base_atr_returns.append((entry_p - atr_sp) / entry_p - cost_c)
                else:
                    base_atr_returns.append((entry_p - exit_p) / entry_p - cost_c)

        n_signals = len(net_returns)
        if n_signals == 0:
            continue

        # Both definitions:
        # 1. Conditional: target / (target + stop)
        # 2. Unconditional: target / all_signals
        hr_struct_cond = target_hits / (target_hits + stop_hits) if (target_hits + stop_hits) > 0 else 0.0
        hr_struct_all = target_hits / n_signals

        hr_atr_cond = base_atr_hits / (base_atr_hits + base_atr_stops) if (base_atr_hits + base_atr_stops) > 0 else 0.0
        hr_atr_all = base_atr_hits / n_signals

        mean_p_star = float(np.mean(p_star_list)) if len(p_star_list) > 0 else 0.5
        mean_atr_p_star = float(np.mean(base_atr_p_star_list)) if len(base_atr_p_star_list) > 0 else 0.5
        mean_net_bps = float(np.mean(net_returns)) * 10000
        mean_atr_net_bps = float(np.mean(base_atr_returns)) * 10000

        rows.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "n_signals": n_signals,
            "target_first_pct": f"{target_hits/n_signals*100:.2f}%",
            "stop_first_pct": f"{stop_hits/n_signals*100:.2f}%",
            "neither_pct": f"{neither_hits/n_signals*100:.2f}%",
            "struct_hit_rate_conditional": f"{hr_struct_cond*100:.2f}%",
            "struct_hit_rate_all_signals": f"{hr_struct_all*100:.2f}%",
            "breakeven_p_star_with_cost": f"{mean_p_star*100:.2f}%",
            "struct_mean_net_bps": f"{mean_net_bps:.1f}",
            "fixed_atr_hit_rate_conditional": f"{hr_atr_cond*100:.2f}%",
            "fixed_atr_hit_rate_all_signals": f"{hr_atr_all*100:.2f}%",
            "fixed_atr_p_star_with_cost": f"{mean_atr_p_star*100:.2f}%",
            "fixed_atr_net_bps": f"{mean_atr_net_bps:.1f}",
            "target_beaten_by_stop": stop_hits > target_hits,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(REPORTS_DIR / "feat_structure_target_stop_outcomes.csv", index=False)
    return df_out


def compute_target_stop_distance_distribution(
    df_drawings: pd.DataFrame,
    cost_c: float = config.TOTAL_ROUNDTRIP_COST,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """Compute quantile distribution of candidate target and stop distances relative to cost c."""
    draw_v = df_drawings.iloc[warmup_candles:].copy()
    valid = draw_v["candidate_target_price"].notna() & draw_v["candidate_stop_price"].notna()
    sub = draw_v[valid]

    t_pct = sub["target_dist_pct"]
    s_pct = sub["stop_dist_pct"]
    t_c = sub["target_dist_c_mult"]
    s_c = sub["stop_dist_c_mult"]

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q_rows = []
    for q in quantiles:
        q_rows.append({
            "quantile": f"Q_{int(q*100):02d}",
            "target_dist_pct": f"{t_pct.quantile(q)*100:.3f}%",
            "stop_dist_pct": f"{s_pct.quantile(q)*100:.3f}%",
            "target_dist_c_multiple": f"{t_c.quantile(q):.2f}x",
            "stop_dist_c_multiple": f"{s_c.quantile(q):.2f}x",
        })

    df_q = pd.DataFrame(q_rows)
    df_q.to_csv(REPORTS_DIR / "feat_structure_distance_quantiles.csv", index=False)
    return df_q


def compute_level_strength_and_retest_types(
    df_train: pd.DataFrame,
    df_drawings: pd.DataFrame,
    horizons: List[int] = [6, 12],
    cost_c: float = config.TOTAL_ROUNDTRIP_COST,
    warmup_candles: int = LOOKBACK_CANDLES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze level strength (touches) vs hit rate, and flip-and-retest vs ordinary level tests."""
    verify_train_only(df_train)
    df_v = df_train.iloc[warmup_candles:].copy()

    # 1. Level strength: touches (1, 2, 3+)
    strength_rows = []
    for N in horizons:
        raw_sign = df_v[f"raw_sign_{N}"]
        valid = raw_sign.isin([-1.0, 1.0])
        vote = df_v["feat_struct_rule_vote"]

        # Support touches when voting Bull, resistance touches when voting Bear
        touches = np.where(vote == 1, df_v["feat_struct_support_strength"], df_v["feat_struct_resistance_strength"])

        for t_label, t_mask in [
            ("1 touch", touches == 1),
            ("2 touches", touches == 2),
            ("3+ touches", touches >= 3),
        ]:
            eval_mask = valid & (vote != 0) & t_mask
            n_sub = int(eval_mask.sum())
            if n_sub == 0:
                continue
            hr = float((vote[eval_mask] == raw_sign[eval_mask]).mean())
            strength_rows.append({
                "horizon_N": N,
                "horizon_minutes": N * 5,
                "level_strength": t_label,
                "signals": n_sub,
                "rule_vote_hit_rate": f"{hr*100:.2f}%",
            })

    df_strength = pd.DataFrame(strength_rows)
    df_strength.to_csv(REPORTS_DIR / "feat_structure_level_strength.csv", index=False)

    # 2. Retest type: flip-and-retest (Rule R1/R2) vs ordinary pullback (Rule R3/R4)
    retest_rows = []
    code = df_drawings.iloc[warmup_candles:]["rule_reason_code"]
    for N in horizons:
        raw_sign = df_v[f"raw_sign_{N}"]
        valid = raw_sign.isin([-1.0, 1.0])
        vote = df_v["feat_struct_rule_vote"]

        # Reason codes: 'BULL_RETEST', 'BEAR_RETEST', 'BULL_PULLBACK_MA', 'BEAR_PULLBACK_MA'
        flip_retest_mask = valid & (vote != 0) & code.isin(["BULL_RETEST", "BEAR_RETEST"])
        pullback_mask = valid & (vote != 0) & code.isin(["BULL_PULLBACK_MA", "BEAR_PULLBACK_MA"])

        n_flip = int(flip_retest_mask.sum())
        n_pull = int(pullback_mask.sum())

        hr_flip = float((vote[flip_retest_mask] == raw_sign[flip_retest_mask]).mean()) if n_flip > 0 else 0.5
        hr_pull = float((vote[pullback_mask] == raw_sign[pullback_mask]).mean()) if n_pull > 0 else 0.5

        retest_rows.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "flip_and_retest_signals": n_flip,
            "flip_and_retest_hit_rate": f"{hr_flip*100:.2f}%",
            "pullback_ma_signals": n_pull,
            "pullback_ma_hit_rate": f"{hr_pull*100:.2f}%",
            "edge_flip_over_pullback": f"{(hr_flip - hr_pull)*100:.2f}%",
        })

    df_retest = pd.DataFrame(retest_rows)
    df_retest.to_csv(REPORTS_DIR / "feat_structure_retest_comparison.csv", index=False)
    return df_strength, df_retest
