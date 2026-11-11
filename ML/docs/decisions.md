# Architecture & Design Decisions Log

This log records every architectural and ML design decision, including rationale, trade-offs, and citations. All choices documented here must be defensible in the project presentation and final report.

---

## 1. Fixed Historical Date Range & Split Boundaries
- **Decision:** Use a strictly fixed time range:
  - `START = 2023-11-01 00:00:00 UTC`
  - `END = 2026-11-10 00:00:00 UTC` (Extended by 9 days to latest closed candle: `2026-11-09 23:55:00 UTC`).
  - Total Duration: 1,105 calendar days, yielding exactly **318,240 continuous 5-minute candles** (315,648 + 9 days $\times$ 288 candles).
- **Split Partitions & Boundary Preservation Policy:**
  - **Train:** `2023-11-01 00:00:00 UTC` to `2025-12-07 04:40:00 UTC` (**220,953 candles**, 767.2 days, ~69.43%)
  - **Validation:** `2025-12-07 04:45:00 UTC` to `2026-05-20 14:15:00 UTC` (**47,347 candles**, 164.4 days, ~14.88%)
  - **Test (Locked):** `2026-05-20 14:20:00 UTC` to `2026-11-09 23:55:00 UTC` (**49,940 candles**, 173.4 days, ~15.69%)
- **Rationale for Boundary Preservation:** 
  - Dynamic recomputation of 70/15/15 after adding the extra 9 days would shift the historical boundary dates forward by several days. Shifting boundaries after EDA has been performed on the initial train split constitutes an insidious form of retroactive data snooping / boundary leakage.
  - Therefore, Train and Validation boundaries are permanently **frozen** to their exact timestamps. The extra 9 days (2,592 candles) were absorbed entirely by the tail of the Test split (growing from 47,348 to 49,940 candles, metadata only).
- **Two-Way Risk & Live Test Freeze Policy:**
  - In our academic evaluation, the 24-hour live market test takes place strictly *after* project submission.
  - The final model, scaler, and threshold bundle must be frozen using *only* data up to this fixed `END` timestamp (`2026-11-10 00:00:00 UTC`).
  - No retraining, parameter modification, or post-submission snooping is permissible during or after the live evaluation.
- **Citation:** Standard quantitative research methodology; López de Prado (2018).

---

## 2. Storage Format: Pure CSV via Centralized `src/io_utils.py`
- **Decision:** All raw market klines and processed datasets are stored strictly as `.csv` (`data/raw/BTCUSDT_5m_raw.csv`, `data/processed/clean.csv`). Reading and writing must go through `src/io_utils.py`.
- **Rationale & Benefits:**
  - *Universal Inspectability:* Plain CSV files can be inspected with any standard text editor, terminal tool (`head`, `grep`), or spreadsheet viewer without requiring specialized binary libraries.
  - *Dependency Minimization:* Eliminates binary/C dependencies (`pyarrow`), simplifying cross-platform installation and long-term academic evaluation reproducibility.
- **Trade-offs & Mitigations:**
  - *Storage Overhead:* Uncompressed CSV is ~40MB for clean data and ~49MB for raw data, compared to ~21MB for Snappy-compressed Parquet. Disk footprint is modest and well within standard computing bounds.
  - *I/O Latency:* CSV parsing takes ~1.2s vs ~0.08s for Parquet. Acceptable for batch model training and negligible for single-row live inference.
  - *Type and Timezone Drift Risk:* CSV does not inherently store schema or timezone metadata.
    - *Mitigation:* Centralized in `src/io_utils.py`. `save_clean_market_data_csv` writes ISO 8601 UTC timestamps (`%Y-%m-%d %H:%M:%S%z`) at full IEEE 754 double precision (repr-level). `load_clean_market_data_csv` parses `open_time`, enforces a UTC `DatetimeIndex`, and casts all price/volume columns explicitly to `float64`. Zero ad-hoc `pd.read_csv` calls are permitted across the codebase.

---

## 2b. Test Split Metadata Viewing & Snooping Prevention
- **Decision:** Regime statistics for the locked Test split (return $+0.31\%$, annualized volatility $38.37\%$) were viewed at an earlier stage as metadata only.
- **Enforcement:** No modeling decisions, feature engineering choices, label thresholds, or hyperparameter tunings may be based on test set statistics. All threshold computations (e.g., Q1 quantiles, volatility scalers) are strictly computed on TRAIN rows and frozen.

---

## 2c. Mathematical Reconciliation of Return Volatility
- **Empirical Ground Truth (Computed on Train):**
  - Empirical 5m Log Return Standard Deviation: $\sigma_{5m} = \mathbf{0.1528\%}$ ($0.00152776$)
  - Continuous Crypto Annualization Factor: $\sqrt{288 \times 365} = \sqrt{105,120} = 324.222$
  - Annualized Volatility: $0.152776\% \times 324.222 = \mathbf{49.53\%}$ (matches the reported $49.5\%$)
  - Empirical 1-Hour Forward Return Standard Deviation ($N=12$): $\sigma_{1h} = \mathbf{0.5135\%}$ ($0.00513537$)
  - Annualized Volatility from 1h: $0.513537\% \times \sqrt{24 \times 365} = 0.513537\% \times 93.595 = \mathbf{48.06\%}$
  - Under square-root of time scaling: $\sigma_{1h} \approx \sigma_{5m} \times \sqrt{12} = 0.1528\% \times 3.4641 = 0.5292\%$
- **Discrepancy Diagnosis:** The earlier draft cited $0.352\%$ for 5m std and $1.14\%$ for 1h std. Analysis reveals $0.352\% \times \sqrt{288 \times 365} = 114.1\% = 1.141$ in decimal format, which was accidentally mislabeled as '1h std = 1.14%'. The true 5m std is $0.1528\%$ and 1h std is $0.5135\%$.

---

## 3. Label Specification & Invalidation of Prior Preliminary Tables
- **Status of Prior Session EDA & Tables:**
  - All preliminary EDA statistics, neutral-share percentages, and dead-zone tables produced in prior sessions are formally marked **INVALID**. They were unverified, contained calculation discrepancies, or were computed prior to freezing the final 3-year canonical dataset.
  - The preliminary hypothesis that "12 candles is the sweet spot" is **RESCINDED**. The horizon $N$ is strictly OPEN across candidate horizons $N \in \{2, 3, 6, 12\}$ candles (10m, 15m, 30m, 60m), with $N=12$ serving purely as a fallback default. The final horizon will be decided in Step 5 based on out-of-fold cost-adjusted realized returns on Train walk-forward folds.
- **V3 Complete Target Specification (Master Prompt v3 §6):**
  - Evaluated across candidate horizons $N \in \{2, 3, 6, 12\}$ candles and round-trip friction $c = 0.0023$ ($0.23\%$).
  - Base target quantities:
    - $\text{fwd\_ret\_N} = \text{Close}_{t+N} / \text{Close}_t - 1$
    - $\text{fwd\_ret\_open\_N} = \text{Close}_{t+N} / \text{Open}_{t+1} - 1$ (conservative entry for backtesting)
    - $\text{raw\_sign\_N} = \text{sign}(\text{fwd\_ret\_N})$ ($+1$ if $>0$, $-1$ if $<0$, NaN if $0$)
    - $\text{has\_valid\_target\_N} = \text{False}$ if horizon crosses a data gap or falls into the last $N$ rows of a partition.
  - Asymmetric and Volatility-Adjusted Threshold Rules (S1, S2, Q1, V1, V2, A1, A2):
    - Up ($+1$): $\text{fwd\_ret\_N} > \theta_{\text{up}, N}$
    - Down ($-1$): $\text{fwd\_ret\_N} < -\theta_{\text{down}, N}$
    - Neutral ($0$): otherwise
    - Floor: $\theta \ge c = 0.0023$ everywhere.
    - Q1 thresholds computed once on Train rows and frozen.
    - **Empirical Q1 Floor-Binding & Equivalence to S1:** Across all evaluated candidate horizons $N \in \{2, 3, 6, 12\}$, the 50th percentile of positive and negative forward returns in the Training split falls strictly below the transaction cost floor $c = 0.2300\%$:
      - $N=2$: raw Up 0.0934%, raw Down 0.0909% $\rightarrow \theta = c = 0.2300\%$
      - $N=3$: raw Up 0.1136%, raw Down 0.1104% $\rightarrow \theta = c = 0.2300\%$
      - $N=6$: raw Up 0.1561%, raw Down 0.1532% $\rightarrow \theta = c = 0.2300\%$
      - $N=12$: raw Up 0.2182%, raw Down 0.2104% $\rightarrow \theta = c = 0.2300\%$
      Because the floor binds universally, $\text{Q1} \equiv \text{S1}$ everywhere on this dataset. Q1 remains fully documented in reports for transparency, but Q1 and S1 are counted as ONE configuration in the active experiment budget to prevent redundant compute.
    - V1/V2 dynamic volatility thresholds $\theta_t = \max(c, k \cdot \sigma_t \cdot \sqrt{N})$ use strictly causal past rolling returns ($W=288$ candles).
- **Economic Friction Basis:**
  $$\text{Round-trip Cost } c = \text{Taker Fee (0.20\%)} + \text{Slippage (0.02\%)} + \text{Half-Spread (0.01\%)} = 0.2300\% \; (0.0023)$$
  - Exchange Fees ($0.20\%$): `[VERIFIED]` from Binance VIP 0 taker schedule.
  - Execution Slippage ($0.02\%$) and Half-Spread ($0.01\%$): `[ASSUMPTION]` modeling liquid BTCUSDT order book frictions.
  - BNB-discount fee ($0.15\%$ round-trip): `[VERIFIED ALTERNATIVE]` documented, unused.
- **Citation:** Binance Spot Fee Schedule (Verified 2026); Cartea, Jaimungal, & Penalva (2015), *Algorithmic and High-Frequency Trading*..

---

## 4. Purged Walk-Forward Cross-Validation
- **Decision:** Implement expanding-window walk-forward cross-validation with an explicit purge gap of 12 candles ($\text{horizon}$) between training and validation folds, as well as between validation and test sets.
- **Rationale:** In time series with overlapping forward-looking labels (12-candle horizon), candle $t$ uses price at $t+12$. If training data ends at $T$ and validation starts at $T+1$, labels from $T-11$ to $T$ overlap into the validation set, creating information leakage. A 12-candle purge window completely eliminates label leakage.
- **Citation:** López de Prado, M. (2018), *Advances in Financial Machine Learning*, Chapter 7.

---

## 5. Test Set Locking Mechanism
- **Decision:** The test set partition is programmatically guarded by a lock in `src/splitting.py`. Any attempt to access or load the test set prior to Phase 6 raises a `TestSetLockedError`.
- **Rationale:** Preserves the integrity of out-of-sample testing by eliminating subtle data snooping or manual hyperparameter tweaking on test performance.

---

## 6. Strict Vectorization in Validation
- **Decision:** Eliminate all row-wise iteration (`df.iterrows()`) in `src/validation.py` in favor of vectorized Pandas/NumPy checks and structured gap reporting.
- **Rationale:** `df.iterrows()` takes >2 minutes for 315,648 rows. Vectorized column evaluations execute in <0.2 seconds.

---

## 7. Pre-Declared Experiment Budget & Elimination Protocol (Master Prompt v3 §7 + Setup D Update)
- **Grid Configuration Upper Bound:**
  - Horizons: $N \in \{2, 3, 6, 12\}$ candles (4 horizons)
  - Threshold Rules: $\text{S1, S2, Q1, V1, V2, A1, A2, F05, F10, F15}$ (10 nominal rules; 9 distinct active rules since $\text{Q1} \equiv \text{S1}$). The three F rules (F05: 0.05%, F10: 0.10%, F15: 0.15%) serve as an extra research comparison family operating below transaction cost $c = 0.23\%$.
  - Training Setups (Single-Stage):
    - **Setup A:** Non-neutral only (filter out dead zone rows, train binary Up/Down on confident moves)
    - **Setup B:** Raw sign on all rows (`raw_sign_N`, 100% row coverage, train on sign direction)
    - **Setup C:** 3-class weighted (Up / Down / Neutral with class inverse-frequency weights)
  - Models (Single-Stage): Logistic Regression, LightGBM (2 models)
  - Single-Stage Nominal Grid: $4 \times 10 \times 3 \times 2 = 240$ configurations.
  - Single-Stage Effective Grid: $4 \times 9 \times 3 \times 2 = \mathbf{216}$ configurations ($\text{Q1} \equiv \text{S1}$).
  
- **Declaration of Setup D (Two-Stage Decoupled Indicator):**
  - Architecture:
    1. *Stage 1 (Direction Model):* Predicts binary Up vs Down direction (trained on one of F05, F10, F15, S1, or on `raw_sign_N`).
    2. *Stage 2 (Worth-It Model):* Predicts binary `big_move_N` ($\text{big\_move}_N = 1$ if $|\text{fwd\_ret}_N| > c = 0.23\%$, else 0).
  - Signal Emission Logic:
    - An active directional signal is emitted if and only if:
      $$\text{Confidence}(\text{Direction}) \ge d \quad \text{AND} \quad P(\text{big\_move}_N) \ge m$$
    - The two calibrated probabilities are **NOT** multiplied together; each model acts as an independent gate.
    - Calibration: Both models are probability-calibrated (Platt scaling / Isotonic regression) separately on Validation rows only.
    - Threshold Tuning: Confidence cutoffs $d \in \{0.55, 0.60, 0.65\}$ and $m \in \{0.40, 0.50, 0.60\}$ are selected strictly on Train walk-forward folds from this pre-declared grid.
  - Setup D Configuration Count:
    - 4 horizons $\times$ 5 direction label choices (F05, F10, F15, S1, raw_sign) $\times$ 2 models $\times$ 1 worth-it model = **40 configurations**.
  - **Total Project Configuration Budget (Setups A + B + C + D):**
    - Nominal: $240 + 40 = \mathbf{280}$ configurations.
    - Effective: $216 + 40 = \mathbf{256}$ configurations.

- **Staged Elimination Protocol:**
  1. *Stage 1 (Screening via Logistic Regression):* Evaluate all Horizon $\times$ $\theta$-rule combinations using baseline Logistic Regression on Train walk-forward folds. Eliminate combinations with negative cost-adjusted return or near-zero coverage (<15 signals/day).
  2. *Stage 2 (Refinement via LightGBM):* Train top-performing combinations across Setups A, B, C, and D with LightGBM.
- **Selection Criteria (on Train Walk-Forward Folds only):**
  - **CRITICAL RULE:** Do NOT select rules based on raw training label accuracy, because different rules construct fundamentally different label distributions (e.g. A1 has 0% neutral, S1 has 53% neutral).
  - All configurations must be evaluated on identical ground truth:
    1. Realized cost-adjusted return of emitted signals ($R_{\text{net}} = \text{signal} \times R_{\text{open}} - c$).
    2. Emitted signal directional accuracy evaluated against `raw_sign_N`.
    3. Minimum activity threshold: Reject configurations emitting fewer than ~15 signals/day.
    4. Fold-to-fold Sharpe ratio and stability across walk-forward folds.
    5. Simplicity tie-breaker: default fallback is Horizon $N=12$ candles (60 minutes).
- **Policy:** Zero model training or hyperparameter fitting is conducted in Step 2 or Step 3. All modeling begins strictly in Step 5.

---

## 8. Feature Engineering v2: Feature Removals, Pre-Declared Feature Additions Across 7 Groups, and Empirical Lookback Determination

### 8.1 Removed Features
- **`feat_ret_open_gap`:**
  - *Formula:* `(open[t] - close[t-1]) / close[t-1]`
  - *Removal Reason:* In continuous 24/7 crypto spot trading on Binance with 0 gap candles, consecutive candle open prices almost perfectly equal the previous close. The empirical standard deviation on Train is $5.1 \times 10^{-6}$ and over 99% of values are zero or negligible floating-point noise. It provides zero predictive signal and acts as near-constant noise.
  - *Logged Action:* Completely removed from `src/features.py`, `FEATURE_GROUPS`, and `src/column_dictionary.py`.

### 8.2 Pre-Declared Feature Additions (Declared BEFORE Computing)
A total of 35 new causal, stationary features across 7 new feature groups are added. All features use strictly data available up to candle $t$ close:

| Group | Feature Name | Description & Causal Formula | Unit | Reason / Economic Justification |
|---|---|---|---|---|
| `vol_range` | `feat_vol_parkinson_12` | Parkinson volatility over 12 candles: $\sqrt{\frac{1}{4 \ln 2 \cdot 12} \sum_{i=0}^{11} \ln(H_{t-i}/L_{t-i})^2}$ | fraction | 5x more statistically efficient than close-to-close variance; captures short-term intraday price dispersion. |
| `vol_range` | `feat_vol_parkinson_60` | Parkinson volatility over 60 candles (5h) | fraction | Captures intermediate range-based volatility without lookahead. |
| `vol_range` | `feat_vol_parkinson_288` | Parkinson volatility over 288 candles (24h) | fraction | Measures full-day range-based baseline volatility. |
| `vol_range` | `feat_vol_gk_12` | Garman-Klass volatility over 12 candles: $\sqrt{\frac{1}{12}\sum [0.5\ln(H/L)^2 - (2\ln 2 - 1)\ln(C/O)^2]}$ | fraction | Accounts for both intra-candle range and open-close jump dynamics at 1h scale. |
| `vol_range` | `feat_vol_gk_60` | Garman-Klass volatility over 60 candles (5h) | fraction | Intermediate OHLC volatility metric combining high-low and open-close information. |
| `vol_range` | `feat_vol_gk_288` | Garman-Klass volatility over 288 candles (24h) | fraction | Full 24-hour baseline OHLC volatility metric. |
| `vol_realized` | `feat_vol_realized_abs_sum_6` | Sum of absolute 5m log returns over 6 candles (30m): $\sum_{i=0}^5 \|r_{t-i}\|$ | fraction | Short-horizon realized absolute variation; robust proxy for immediate market turbulence. |
| `vol_realized` | `feat_vol_realized_abs_sum_12` | Sum of absolute 5m log returns over 12 candles (1h) | fraction | 1-hour realized absolute variation. |
| `vol_realized` | `feat_vol_realized_abs_sum_36` | Sum of absolute 5m log returns over 36 candles (3h) | fraction | Multi-hour realized absolute variation. |
| `vol_realized` | `feat_vol_realized_max_abs_12` | Maximum single-candle absolute return over last 12 candles: $\max_{i=0..11} \|r_{t-i}\|$ | fraction | Jump detector: isolates sudden single-candle price shocks from continuous diffusion. |
| `vol_realized` | `feat_vol_realized_max_abs_60` | Maximum single-candle absolute return over last 60 candles | fraction | Identifies whether the recent 5-hour window contains a flash crash or pump spike. |
| `vol_realized` | `feat_vol_realized_semi_var_up_ratio_60` | Share of upside semivariance over 60 candles: $SV^+ / (SV^+ + SV^-)$ | fraction | Measures directional asymmetry in volatility: values > 0.5 indicate upside-driven variance. |
| `vol_realized` | `feat_vol_realized_semi_var_down_ratio_60` | Share of downside semivariance over 60 candles: $SV^- / (SV^+ + SV^-)$ | fraction | Measures downside-driven volatility (panic selling pressure). |
| `vol_seasonal` | `feat_vol_seasonal_slot_abs_ret_7d` | Mean absolute return of the same 5m diurnal slot over past 7 days (offsets $k \in \{288, 576, \dots, 2016\}$) | fraction | Establishes the expected baseline volatility for this specific time-of-day. Never uses candle $t$. |
| `vol_seasonal` | `feat_vol_seasonal_vol_ratio_12` | Ratio of current 12-candle realized mean absolute return to 7-day same-slot mean | ratio | De-seasonalized volatility: values > 1 indicate truly abnormal volatility beyond expected diurnal cycles. |
| `vol_seasonal` | `feat_vol_seasonal_volume_ratio_7d` | Ratio of current volume $V_t$ to 7-day same-slot mean volume | ratio | De-seasonalized volume: separates normal London/NY market-open surges from true volume anomalies. |
| `vol_seasonal` | `feat_vol_seasonal_trades_ratio_7d` | Ratio of current trade count $Tr_t$ to 7-day same-slot mean trade count | ratio | De-seasonalized trading activity proxy. |
| `vol_har` | `feat_vol_har_rv_12` | Realized volatility over 1h (12 candles): $\text{std}(r, W=12)$ | fraction | Heterogeneous Autoregressive component for high-frequency intraday traders. |
| `vol_har` | `feat_vol_har_rv_72` | Realized volatility over 6h (72 candles): $\text{std}(r, W=72)$ | fraction | HAR component reflecting medium-term intraday market participants. |
| `vol_har` | `feat_vol_har_rv_288` | Realized volatility over 24h (288 candles): $\text{std}(r, W=288)$ | fraction | HAR component for daily swing traders. |
| `vol_har` | `feat_vol_har_rv_2016` | Realized volatility over 7 days (2,016 candles): $\text{std}(r, W=2016)$ | fraction | Long-term weekly macro regime volatility component. |
| `vol_har` | `feat_vol_har_ratio_12_288` | Ratio of 1h realized volatility to 24h realized volatility: $RV_{12} / RV_{288}$ | ratio | Fast-to-slow volatility momentum: captures volatility expansion vs contraction regimes. |
| `vol_har` | `feat_vol_har_ratio_288_2016` | Ratio of 24h realized volatility to 7d realized volatility: $RV_{288} / RV_{2016}$ | ratio | Daily-to-weekly volatility regime ratio. |
| `flow_window` | `feat_flow_window_net_6` | Cumulative normalized taker net flow over 6 candles: $\sum (2 \cdot \text{taker\_buy} - \text{vol}) / \sum \text{vol}$ | fraction | Short-term institutional aggressive buyer/seller imbalance over 30 minutes. |
| `flow_window` | `feat_flow_window_net_12` | Cumulative normalized taker net flow over 12 candles (1h) | fraction | 1-hour aggressive market order net flow. |
| `flow_window` | `feat_flow_window_net_36` | Cumulative normalized taker net flow over 36 candles (3h) | fraction | 3-hour persistent order flow accumulation/distribution. |
| `flow_window` | `feat_flow_window_signed_flow_6` | Net taker flow over 6 candles multiplied by $\text{sign}(\text{ret}_1)$ | fraction | Concordance metric: positive when aggressive flow and price direction align. |
| `flow_window` | `feat_flow_window_signed_flow_12` | Net taker flow over 12 candles multiplied by $\text{sign}(\text{ret}_1)$ | fraction | 1-hour flow-price concordance. |
| `flow_window` | `feat_flow_window_signed_flow_36` | Net taker flow over 36 candles multiplied by $\text{sign}(\text{ret}_1)$ | fraction | Multi-hour flow-price concordance. |
| `location` | `feat_loc_position_288` | Position of close in 24h high-low range: $(C_t - \min_{288} L) / (\max_{288} H - \min_{288} L)$ | fraction $[0,1]$ | Normalizes current price within the 24h trading channel (support/resistance proximity). |
| `location` | `feat_loc_dist_high_288` | Fractional distance from 24h high: $(\max_{288} H - C_t) / C_t$ | fraction | Overhead resistance distance: key input for breakout vs mean-reversion classification. |
| `location` | `feat_loc_dist_low_288` | Fractional distance from 24h low: $(C_t - \min_{288} L) / C_t$ | fraction | Support distance. |
| `time_weekend` | `feat_time_is_weekend` | Binary weekend indicator ($1.0$ if Saturday or Sunday UTC, $0.0$ otherwise) | binary | Captures weekend institutional illiquidity and retail-driven dynamics. |
| `time_weekend` | `feat_time_sin_week` | Sine of hour-of-week: $\sin(2\pi \cdot (\text{weekday}\cdot 24 + \text{hour} + \text{min}/60) / 168)$ | continuous $[-1,1]$ | Seamless circular 168-hour weekly cycle representation. |
| `time_weekend` | `feat_time_cos_week` | Cosine of hour-of-week: $\cos(2\pi \cdot (\text{weekday}\cdot 24 + \text{hour} + \text{min}/60) / 168)$ | continuous $[-1,1]$ | Orthogonal circular weekly cycle representation. |

### 8.3 Feature Retention, Duplicate Drops & Linear Model Policy
- **Duplicate Drops (88 -> 85 features):** Three exact duplicate features were removed prior to EDA:
  1. `feat_vol_har_rv_12`: Identical to `feat_vol_sigma_12` (max abs diff = 0.0).
  2. `feat_vol_har_rv_288`: Identical to `feat_vol_sigma_288` (max abs diff = 0.0).
  3. `feat_vol_realized_semi_var_down_ratio_60`: Exact linear complement of `feat_vol_realized_semi_var_up_ratio_60` ($down = 1.0 - up$, Pearson corr = -1.0).
- **Linear Model Rule:** For the exact linear identity `feat_mom_macd_hist = line - signal`, all 3 are retained in `master.csv` (tree models split on either without issue), but `LINEAR_MODEL_EXCLUDE = ["feat_mom_macd_hist"]` is defined in `config.py` for logistic/ridge models to prevent rank deficiency.
- **Dataset Schema:** `master.csv` contains 318,240 rows $\times$ 172 columns (11 market + 85 engineered features + 16 base targets + 40 rule labels + 4 big_move + 16 dynamic thetas).
- **Pruning Policy:** Zero features are deleted in Step 4 EDA beyond the 3 approved duplicates. All candidate redundant features are grouped via hierarchical clustering ($|\rho| > 0.95$) for walk-forward ablation in Step 5.

### 8.4 Warm-Up Period and Streaming Buffer Constants
- **LOOKBACK_CANDLES = 2018:** The minimal candle count needed to compute all 85 features for the newest row. Derived as 2,016 candles (7 full 24h days for 7-day seasonal baseline slot returns and HAR $\text{rv}_{2016}$) + 1 candle (to compute $r_{t-2016} = \ln(C_{t-2016} / C_{t-2017})$) + 1 candle (for completed 1h candle completion shift). At $L=2018$, failing feature count is 0 and max difference vs full history is $1.45 \times 10^{-10}$.
- **LIVE_BUFFER_CANDLES = 2880:** The recommended streaming buffer size for live deployment (10 full days: $288 \times 10 = 2,880$ candles). Verified to match full-history calculation to $< 10^{-10}$ across all 85 features.

### 8.5 Validation & Modeling Protocols for Step 5
- **Circular-Shift Noise Band:** Evaluated via 500 circular shifts ($\ge 2018$ candles, Train post-warmup only) to preserve autocorrelation and serial dependence. Sets 95th/99th noise ceilings per feature and maximum over all 85 features.
- **Hierarchical Correlation Clusters:** Complete linkage clustering on $1 - |\text{Spearman}|$ at $d < 0.05$ ($|\rho| > 0.95$). Identifies 10 multi-feature clusters; protects `feat_vol_sigma_288`, `feat_vol_natr_14`, `feat_mom_rsi_14`. Final selection deferred to Step 5 walk-forward ablation.
- **Retraction of Single Setup/Horizon Baseline:** Horizon $N \in \{2, 3, 6, 12\}$ and Setup (A, B, C, D) are NOT pre-selected in Step 4. They will be decided strictly in Step 5 via the pre-declared staged elimination protocol on walk-forward folds.





