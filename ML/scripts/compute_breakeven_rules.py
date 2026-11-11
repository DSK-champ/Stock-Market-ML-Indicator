"""Compute empirical hit rates of the two plain reversal rules vs raw_sign_N.

Evaluated on all valid train rows and on the 20% most extreme rows.
Compared against break-even hit rate over all rows and big_move=1 rows.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.column_dictionary import load_master

def main():
    df_train = load_master(split="train")
    valid_train = df_train.iloc[config.LOOKBACK_CANDLES:].copy()
    
    # Feature 1: feat_loc_position_288
    pos = valid_train["feat_loc_position_288"]
    rule1_signal = np.sign(0.5 - pos)
    # 20% extreme: lowest 10% and highest 10%
    q10_pos = pos.quantile(0.10)
    q90_pos = pos.quantile(0.90)
    rule1_extreme_mask = (pos <= q10_pos) | (pos >= q90_pos)
    
    # Feature 2: EMA50 - close. Note: feat_mom_ema_ratio_50 = close / EMA50 - 1.
    # EMA50 - close > 0 <=> close < EMA50 <=> ema_ratio_50 < 0.
    # sign(EMA50 - close) = -sign(feat_mom_ema_ratio_50)
    ema_ratio = valid_train["feat_mom_ema_ratio_50"]
    rule2_signal = -np.sign(ema_ratio)
    q10_ema = ema_ratio.quantile(0.10)
    q90_ema = ema_ratio.quantile(0.90)
    rule2_extreme_mask = (ema_ratio <= q10_ema) | (ema_ratio >= q90_ema)
    
    cost_c = config.TOTAL_ROUNDTRIP_COST # 0.23%
    
    horizons = [2, 3, 6, 12]
    records = []
    
    for N in horizons:
        fwd_ret = valid_train[f"fwd_ret_{N}"]
        raw_sign = valid_train[f"raw_sign_{N}"]
        bm = valid_train[f"big_move_{N}"]
        
        valid_mask = raw_sign.isin([-1.0, 1.0])
        
        # Break-even hit rates
        mean_abs_all = float(fwd_ret.dropna().abs().mean())
        be_all = 0.5 + cost_c / (2.0 * max(mean_abs_all, 1e-12))
        
        fwd_bm1 = fwd_ret.loc[bm == 1.0].dropna()
        m = float(fwd_bm1.abs().mean())
        be_bm = 0.5 + cost_c / (2.0 * max(m, 1e-12))
        
        # Rule 1: sign(0.5 - pos)
        # All rows
        mask_r1 = valid_mask & (rule1_signal != 0)
        hits_r1_all = (rule1_signal[mask_r1] == raw_sign[mask_r1]).mean()
        # Extreme rows (top 20%)
        mask_r1_ext = mask_r1 & rule1_extreme_mask
        hits_r1_ext = (rule1_signal[mask_r1_ext] == raw_sign[mask_r1_ext]).mean()
        
        # Rule 2: sign(EMA50 - close)
        mask_r2 = valid_mask & (rule2_signal != 0)
        hits_r2_all = (rule2_signal[mask_r2] == raw_sign[mask_r2]).mean()
        mask_r2_ext = mask_r2 & rule2_extreme_mask
        hits_r2_ext = (rule2_signal[mask_r2_ext] == raw_sign[mask_r2_ext]).mean()
        
        records.append({
            "horizon_N": N,
            "horizon_minutes": N * 5,
            "breakeven_all_rows": "not achievable" if be_all > 1.0 else f"{be_all*100:.2f}%",
            "breakeven_big_move_m": f"{be_bm*100:.2f}%",
            "rule1_pos288_hit_all": f"{hits_r1_all*100:.2f}%",
            "rule1_pos288_hit_ext20": f"{hits_r1_ext*100:.2f}%",
            "rule2_ema50_hit_all": f"{hits_r2_all*100:.2f}%",
            "rule2_ema50_hit_ext20": f"{hits_r2_ext*100:.2f}%",
            "mean_abs_return_m": f"{m*100:.2f}%",
        })
        
    df_rules = pd.DataFrame(records)
    print("\n=========================================================================================================")
    print("EMPIRICAL HIT RATES OF REVERSAL RULES VS BREAK-EVEN THRESHOLDS (Train Partition, c = 0.23%)")
    print("=========================================================================================================")
    print(df_rules.to_string(index=False))

if __name__ == "__main__":
    main()

