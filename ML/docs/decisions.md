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

---

## 9. Structure Engine, Parameter Grid & Voter Combiner Design (Step 4b)

### 9.1 Declared Parameter Grid (Fixed Prior to Computation)
To prevent data snooping and label overfitting, all structure parameters are pre-declared from a small grid ($\le 3$ values per parameter):

| Component | Parameter | Declared Value | Allowed Grid | Rationale |
|---|---|---|---|---|
| **5m Pivots** | $k_{5m}$ | **5 candles** | $\{3, 5, 10\}$ | 5 candles before & after = 25m span; confirmed strictly at $i+5$. |
| **1h Pivots** | $k_{1h}$ | **3 candles** | $\{2, 3, 5\}$ | 3 hours before & after on completed 1h candles; confirmed at $i+3$. |
| **Level Merge** | $\tau_{merge}$ | **0.5 ATR** | $\{0.25, 0.50, 0.75\}$ | Consolidates nearby swing highs/lows into a single persistent level. |
| **Break Buffer** | $\beta_{break}$ | **0.2 ATR** | $\{0.10, 0.20, 0.30\}$ | Break requires candle CLOSE beyond level $+ 0.2 \times ATR$ (never wicks). |
| **Retest Tolerance** | $\tau_{retest}$ | **0.5 ATR** | $\{0.25, 0.50, 0.75\}$ | Price returning within $0.5 \times ATR$ of a flipped level. |
| **Max Pivots Kept** | $M_{pivots}$ | **50 pivots** | $\{20, 50, 100\}$ | Bounded memory buffer for streaming support/resistance evaluation. |
| **Min Target / Stop** | Multiplier | **1.5 ATR** | $\{1.0, 1.5, 2.0\}$ | Prevents stop placement within normal intraday volatility noise. |

### 9.2 Decision Table for Rule Vote (The 4th Voter)
The rule vote produces an integer signal $v_t \in \{+1, -1, 0\}$ per candle with a causal reason code:

| Rule ID | Trend State | Trigger Conditions | Output Vote | Reason Code |
|---|---|---|---|---|
| **R1** | `Up` (+1) | Retest of flipped support from above within last 12 bars ($close_t \ge S_{flip} - 0.2 ATR$) | **+1** | `BULL_RETEST` |
| **R2** | `Down` (-1) | Retest of flipped resistance from below within last 12 bars ($close_t \le R_{flip} + 0.2 ATR$) | **-1** | `BEAR_RETEST` |
| **R3** | `Up` (+1) | Pullback depth $\le 1.0 ATR$, $close_t > \text{EMA50}$, distance to EMA50 $\le 0.5 ATR$ | **+1** | `BULL_PULLBACK_MA` |
| **R4** | `Down` (-1) | Pullback depth $\le 1.0 ATR$, $close_t < \text{EMA50}$, distance to EMA50 $\le 0.5 ATR$ | **-1** | `BEAR_PULLBACK_MA` |
| **R5** | *Any* | Default / No structural confirmation | **0** | `NEUTRAL` |

### 9.3 Role Separation: Features vs Drawings
- **Role `feature`:** Ratios and ATR distances only (e.g., $(C - S)/ATR$, $(R - C)/C$, trend state one-hot, pullback depth in ATR). Never raw price levels.
- **Role `drawing`:** Raw price levels for chart rendering and target/stop evaluation (`level_support_price`, `level_resistance_price`, `candidate_target_price`, `candidate_stop_price`). Excluded from model feature sets.

### 9.4 4-Voter Combiner Architecture (Specification for Step 5)
- **Voters (4 total):**
  1. Voter 1: Calibrated Logistic Regression (continuous probability $p_{LR} \in [0, 1]$)
  2. Voter 2: Calibrated Random Forest (continuous probability $p_{RF} \in [0, 1]$)
  3. Voter 3: Calibrated LightGBM (continuous probability $p_{LGBM} \in [0, 1]$)
  4. Voter 4: Rule-based Voter $v_{rule} \in \{+1, -1, 0\}$ (Section 9.2)
- **Feature Set Variants:**
  - Variant A: ML models fitted on 85 baseline features (excluding structure group).
  - Variant B: ML models fitted on all features (baseline + structure group).
- **Aggregator Mechanisms:**
  - Aggregator 1 (Confluence Rule): Signal emitted only if $\ge k$ of 4 voters agree (with $k \in \{2, 3, 4\}$ declared) and aggregate model confidence exceeds calibrated threshold $\gamma$.
  - Aggregator 2 (Meta-Logistic Stacking): Logistic regression trained strictly on OUT-OF-FOLD walk-forward validation predictions plus the rule vote. Training on in-sample predictions is forbidden.
- **Worth-It Volatility Gating:** Signals from either aggregator are gated by Setup D worth-it model ($\hat{P}(\text{big\_move} = 1) > \theta_{size}$).
- **Evaluation Mandate:** Evaluated strictly via expanding-window walk-forward cross-validation folds on the Train partition.

### 9.5 SUGGESTION on Trend Lines (Diagonal)
Diagonal trend lines require connecting non-adjacent pivots through linear regression or iterative slope fitting. Unlike horizontal levels which are invariant across time once established, diagonal lines continuously shift slope upon arrival of newer swing points, creating lookahead risks unless strictly anchored. We recommend deferring diagonal trend lines until the horizontal pivot engine is fully established and validated.

### 9.6 Dataset Schema & Column Roles Audit (224 Columns)
- **File:** `master.csv` (318,240 rows $\times$ 224 columns).
- **Role Breakdown:**
  - `market` (10 columns): `open_time`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `number_of_trades`, `taker_buy_base_volume`, `taker_buy_quote_volume`.
  - `flag` (6 columns): `gap_id`, `is_gap_boundary`, `has_valid_target_2`, `has_valid_target_3`, `has_valid_target_6`, `has_valid_target_12`.
  - `feature` (136 columns across 17 groups): Groups 1–15 (85 core features), Group 16 `ma_structure` (36 features), Group 17 `structure` (15 features).
  - `target` (56 columns): 16 base targets (`fwd_ret_N`, `fwd_ret_open_N`, `raw_sign_N`, `big_move_N`) + 40 classification labels ($N \in \{2, 3, 6, 12\} \times 10$ rules: $S_1, S_2, Q_1, V_1, V_2, A_1, A_2, F_{05}, F_{10}, F_{15}$).
  - `threshold` (16 columns): Dynamic volatility thresholds for rules $V_1$ and $V_2$ only ($N \in \{2, 3, 6, 12\} \times \{\text{up}, \text{down}\}$).
- **Unit Test Protection:** Enforced by `test_label_and_theta_columns_match_config` in `tests/test_labels.py`.

### 9.7 Lookback Sweep, Sizing & Parity on Full 136 Features
- **Lookback Constants:** `LOOKBACK_CANDLES = 2018`, `LIVE_BUFFER_CANDLES = 2880`.
- **Bounded Pivot Memory:** $M_{pivots} = 50$ confirmed pivots. Across the Train history, 50 confirmed pivots span at most 508 candles, well within the 2,018-candle buffer.
- **Parity Results Across Lookback Buffer Sizes:**
  - $L = 2018$: Failing features count = 0 (zero NaNs in buffer result). Max diff = 0.200 (isolated to `feat_ma_alignment_ema_1h` pairwise rank flip when 1h EMA50 and EMA100 are within $10^{-4}$ of each other). All 15 `structure` features match full history with **exact 0.0 discrepancy**.
  - $L = 2500$: Failing features count = 0. Max diff = 0.200.
  - $L = 3000$: Failing features count = 0. Max diff = $7.55 \times 10^{-5}$ (`feat_ma_1h_ema_ratio_100`).
  - $L = 3500$: Failing features count = 0. Max diff = $3.66 \times 10^{-5}$ (`feat_ma_1h_ema_ratio_100`).

### 9.8 True Event Transitions vs Row States & Trend Segment Lifespan Dynamics
- **Event Definition:** To eliminate autocorrelation and state persistence bias, structural events are counted strictly upon state *transitions* (the initial entry candle where state becomes active, preceded by an inactive or different state):
  - *Break Transitions:* First candle where close crosses beyond level $+ \beta_{break} \times ATR$ (`bars_since_break == 0` preceded by `!= 0`). Total in Train: **190 events** (vs 218,554 row-state occurrences).
  - *Retest Transitions:* First candle of retest rejection (`bars_since_retest == 0` preceded by `!= 0`). Total in Train: **24,514 events** (12,531 Bull, 11,983 Bear).
  - *Rule Vote Transitions:* Discrete signal activations (+1 or -1 transitions). Total in Train: **8,690 events** (4,420 Bull, 4,270 Bear).
  - *Dow Trend Segments:* Distinct trend state episodes (+1, -1, 0). Total in Train: **12,895 segments**.
- **Resolution of Discrepancies (e.g. 93 Breaks vs 6,166 Trend Segments vs 11,772 Retests in 2024):**
  1. *How a Trend Segment Ends Without a Counted Break:* Dow trend state segments (+1 Uptrend, -1 Downtrend, 0 Consolidation) track local swing high/low progressions ($k=5$). An uptrend terminates as soon as a subsequent swing pivot fails to confirm a higher high or forms an inside consolidation bar, immediately transitioning the state machine back to 0 (Neutral). This termination occurs entirely via local swing failure without requiring price to reach or penetrate a macro horizontal support/resistance level. In contrast, a macro break event requires a candle CLOSE beyond an established horizontal cluster ($\text{close} > R + 0.2 \times ATR$ or $\text{close} < S - 0.2 \times ATR$). Because horizontal levels require merging multiple confirmed pivots ($\tau_{merge} = 0.5 ATR$), hundreds of local trend oscillations occur within established ranges without ever breaking a macro boundary.
  2. *Why Retests Outnumber Trend Segments ~2:1:* A retest occurs when price approaches within $\tau_{retest} = 0.5 ATR$ of a previously broken (flipped) level followed by an intraday rejection wick. Once a macro horizontal level flips from resistance to support (or vice-versa), price routinely oscillates around and bounces off that flipped boundary multiple times across consecutive 5-minute swings during a single broader trend or consolidation regime. Because each distinct approach-and-rejection touch represents a fresh transition event, a single trend segment frequently produces 2 to 5 discrete retests, naturally leading to retest counts exceeding broad trend segments by ~2:1.
  3. *Row-Level Block-Bootstrap Standard Errors (Block Length $B = 288$ Candles):* In financial time series, i.i.d. standard errors $\sqrt{p(1-p)/N}$ are fundamentally invalid due to strong serial correlation. Standard errors are computed at the candle row level across all 218,935 post-warmup train candles using a stationary block bootstrap ($B = 288$ candles = 1 full trading day, 1,000 replications):
     - Dow Trend Continuation Hit Rate: $48.39\% \pm 0.84\%$ ($SE = 0.43\%$, 95% CI $[47.55\%, 49.23\%]$) — statistically significantly below 50.0% ($p < 0.001$).
     - Breakout Continuation Hit Rate: $48.12\% \pm 1.15\%$ ($SE = 0.59\%$, 95% CI $[46.96\%, 49.27\%]$) — statistically significantly below 50.0% ($p < 0.001$).
     - Flipped Level Retest Rejection Hit Rate: $51.25\% \pm 0.72\%$ ($SE = 0.37\%$, 95% CI $[50.53\%, 51.97\%]$) — modest mean-reversion edge, but insufficient to clear trading friction ($c = 23$ bps) on standalone basis.
- **Trend Lifespan vs Evaluation Horizons:**
  - Overall Median Trend Duration: **9.0 candles (45 minutes)**.
  - Up Trend Median Duration: **14.0 candles (70 minutes)**.
  - Down Trend Median Duration: **12.0 candles (60 minutes)**.
  - Range Median Duration: **6.0 candles (30 minutes)**.
  - *Architectural Implication:* At horizon $N=12$ (60 minutes), the evaluation holding period matches or exceeds the median trend duration. Trend-following breakout signals entered mid-trend are highly prone to entering during exhaustion, leading to sub-50% directional accuracy.

### 9.9 Cost-Adjusted Break-Even Formulation & Two-Sided Circular Noise Bands
- **Break-Even Formula with Cost:**
  $$\mathbb{E}[\text{net}] = p \cdot (T - c) + (1 - p) \cdot (-S - c) = 0 \implies p^* = \frac{S + c}{T + S}$$
  Evaluated per signal and averaged over all signals.
- **Empirical Break-Even Hurdle:** Mean $p^* = 108.40\%$. Because median candidate targets are small ($0.235\%$) and comparable to the round-trip fee hurdle $c = 0.23\%$, fee friction pushes required win rates above 100% on sub-cost candles, yielding a realized net return of **-23.4 bps per trade**.
- **Two-Sided Circular Noise Bands:** Computed via 500 circular shifts (offset $\ge 2018$ candles, Train post-warmup only) on the exact same row/event subsets. Two-sided 95% $[p_{2.5}, p_{97.5}]$ and 99% $[p_{0.5}, p_{99.5}]$ bands confirm that trend continuation does not achieve statistical significance, and price action exhibits mean-reversion characteristics.

### 9.10 SUGGESTION on 1h Structure and Larger Pivot Windows
- In `docs/decisions.md` Section 9.1, $k_{1h} = 3$ on completed 1h candles was declared, but multi-timeframe 1h horizontal level clustering and larger pivot windows ($k > 10$ on 5m, $k > 5$ on 1h) were NOT in the declared grid.
- **SUGGESTION:** Do NOT run un-declared larger pivot windows or a separate 1h horizontal structure engine in Step 5. Modifying pivot parameters after observing Train results violates research pre-registration and introduces post-hoc data snooping.

---

## 10. Step 5 Model Architecture, Targets, Feature Sets, Folds, and Selection Protocol

### 10.1 Architecture (Two Paths + Final Combiner Layer)
- **Path 1 (Binary Up/Down):**
  - **1A: Technical Analysis Model (`src/ta_models.py`):** Causal TA rule voters (R1, R2 inverted) and shallow decision tree / logistic regression on `FS_TA`. The TA model abstains (vote 0) when it has no opinion; it does NOT predict Neutral.
  - **1B: Binary Ensemble (`src/binary_models.py`):** Ensemble of Logistic Regression + Random Forest + LightGBM predicting `bin_dir_N` or Setup A clear moves.
- **Path 2 (Three-Class Up/Down/Neutral):**
  - Three-class ensemble (`src/three_class_models.py`) of Logistic Regression + Random Forest + LightGBM predicting `target_N_<rule>` with class weights (no oversampling or SMOTE).
- **Baselines (`src/baselines.py`):** Always-Up, majority class, fixed-seed coin flip, last-return momentum and its reverse, RSI(14) rule (<30 Up, >70 Down) and its reverse, $\text{sign}(0.5 - \text{feat\_loc\_position\_288})$, $\text{sign}(\text{EMA50} - \text{close})$, 4b rule vote and its inverse (flagged "chosen after seeing train results"), volatility-only logistic regression on `big_move_N`, and 5-feature reference logistic regression.
- **Final Layer (Step 6):** Combines 1A, 1B, and Path 2 into an emitted trade decision: Up, Down, or Neutral ("No signal"). The separate worth-it model (Setup D) is NOT built; $P(\text{Neutral})$ from Path 2 plays that role. The `big_move_N` columns are preserved solely for the volatility-only reference baseline.

### 10.2 Target Schema & Scoring Truths
- **Binary Direction Target `bin_dir_N` ($N \in \{2, 3, 6, 12\}$):**
  - Definition: $+1$ if $fwd\_ret\_N > 0$, $0$ if $fwd\_ret\_N < 0$, $\text{NaN}$ if $fwd\_ret\_N == 0$ or row is invalid.
  - Role: `target`, `is_future_looking: True`, `allowed_as_model_input: False`.
  - Appended to `master.csv` (schema expands from 223 to 227 columns).
- **Binary Clear-Move Target (Setup A):**
  - Derived dynamically in memory from `target_N_<rule>`: Up $= 1$, Down $= 0$, Neutral ($0$) rows dropped. Not stored on disk.
- **Three-Class Target (Setup C):**
  - Uses `target_N_<rule>` directly ($+1, -1, 0$), weighted by inverse class frequency.
- **Scoring Ground Truths:**
  - Emitted directional signals (+1 / -1) are evaluated against `raw_sign_N` (matching live evaluation).
  - Three-class model probability distributions are additionally evaluated against `target_N_<rule>` via macro-F1, log loss, and confusion matrices.

### 10.3 Pre-Declared Feature Sets (`config.py`)
- **`FS_CORE` (85 features):** Core features from Groups 1–15 (no `ma_structure`, no `structure`, no rule vote).
- **`FS_ALL` (136 features):** Complete feature set across all 17 groups.
- **`FS_TA` (54 features):** Group 16 `ma_structure` (36 features) + Group 17 `structure` (15 features including `feat_struct_rule_vote`) + Group 4 `location` (3 features: `feat_loc_position_288`, `feat_loc_dist_high_288`, `feat_loc_dist_low_288`).
- **`LINEAR_FEATURES`:** Feature set minus `LINEAR_MODEL_EXCLUDE` (`feat_mom_macd_hist`), retaining one feature per hierarchical cluster ($|\text{Spearman}| > 0.95$, complete linkage on Train post-warmup), while strictly protecting `feat_vol_sigma_288`, `feat_vol_natr_14`, and `feat_mom_rsi_14`.
- **Ensemble Variants:**
  - **Variant A:** ML models trained strictly on `FS_CORE` (ensures TA voter 1A remains mathematically independent).
  - **Variant B:** ML models trained on `FS_ALL`.
- **Scaler Isolation:** Feature scalers (`StandardScaler`) for linear models are fitted strictly on each fold's training block, never on validation or held-out folds.

### 10.4 Expanding-Window Walk-Forward Folds (`src/cv.py`)
- Evaluated strictly inside Train post-warmup ($N_{avail} = 220,953 - 2,018 = 218,935$ candles, 760.2 days).
- **Structure:** Initial training block of ~300 days ($86,400$ candles), followed by 6 consecutive held-out evaluation blocks of ~75 days ($21,600$ candles each, with the 6th block absorbing the remaining $25,335$ candles).
- **Purge Gap:** Exactly **12 candles** (equal to the maximum candidate horizon $N=12$) placed between each fold's training end and held-out start. Every horizon uses the identical row boundaries.
- **Early Stopping:** LightGBM early stopping uses the last 15% of each fold's training block (with a 12-candle purge), strictly isolated from the held-out block.

### 10.5 Fixed Model Hyperparameters for Step 5
- **Logistic Regression (LR):** L2 penalty, $C = 0.1$, `solver="lbfgs"`, `max_iter=1000`, `class_weight="balanced"` for three-class.
- **Random Forest (RF):** `n_estimators=300`, `max_depth=6`, `min_samples_leaf=500`, `max_features="sqrt"`, `max_samples=0.3`, `random_state=42`, `class_weight="balanced_subsample"` for three-class.
- **LightGBM:** `num_leaves=15`, `max_depth=4`, `learning_rate=0.03`, `n_estimators=600`, early stopping $= 50$, `min_child_samples=500`, `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=1`, `lambda_l2=10.0`, `deterministic=True`, `random_state=42`.

### 10.6 Experiment Budget & Selection Rule
- **Stage 1 (Logistic Regression on `FS_CORE` LINEAR_FEATURES):**
  - Setup B (raw sign, all rows): 4 horizons ($N \in \{2, 3, 6, 12\}$) = **4 configurations**.
  - Setups A (clear move) and C (three-class): 4 horizons $\times$ 9 distinct rules ($S_1=Q_1$ counted once, $S_2, V_1, V_2, A_1, A_2, F_{05}, F_{10}, F_{15}$) = $36 + 36 =$ **72 configurations**.
  - Total Stage 1 Budget = **76 configurations**.
- **Common Selection Metric:**
  - Evaluated on the top 20% most confident emitted signals:
    $$hit = \text{share with } \text{sign} == raw\_sign\_N$$
    $$m = \text{mean } |fwd\_ret\_open\_N| \text{ of those signals}$$
    $$BE = 0.5 + \frac{c}{2m}, \quad c = 0.0023$$
    $$gap = hit - BE$$
  - Ranking Score: Mean gap across walk-forward folds minus 1 fold standard deviation ($\mu_{gap} - \sigma_{gap}$).
  - Rejection Filter: Reject any configuration yielding fewer than 15 signals per day at 20% coverage.
  - Secondary Metrics: Out-of-fold ROC AUC vs `raw_sign_N`, and simulated net return after cost $c$.
- **Survivors for Stage 2:** Top 3 configurations + mandatory fallback ($N = 12$, rule $S_1$, Setup B).
- Stage 2 (LightGBM, Random Forest, TA-ML) runs strictly on the survivors.

### 10.7 Evaluation Metrics Protocol (`src/evaluate.py`)
Metrics computed per fold, pooled, and across subsets (volatility terciles, monthly macro regimes, hour of day, confidence deciles):
Accuracy, balanced accuracy, precision, recall, per-class F1, macro/weighted F1, MCC, ROC AUC (OvR for three-class), PR AUC (minority class), log loss, Brier score, row-normalized confusion matrix, expected calibration error (ECE), coverage, signals per day, accuracy vs `raw_sign_N`, gap to break-even ($hit - BE$), net return in bps after cost $c$, and block-bootstrap confidence intervals (block length $\ge 288$ candles). Plain accuracy is never used to judge three-class models due to high trivial Neutral base rates.

### 10.8 Rigorous Leakage Prevention Checks
Every model family undergoes 5 mandatory leakage checks:
1. Shuffled-label test (AUC / macro-F1 must collapse to chance).
2. Circular-shift test (offset $\ge 2,018$ candles; score must collapse to chance).
3. Dominant-feature check (no single feature accounts for $> 40\%$ importance).
4. Train-vs-OOF generalization gap inspection.
5. Permutation feature importance on out-of-fold rows.

### 10.9 Pre-Declared Step 6 Combiner & Validation Specification
- **Hyperparameter Grid (OOF Log Loss on Train folds only, $\le 27$ combos per family):**
  - LR: $C \in \{0.01, 0.1, 1.0\}$.
  - RF: `max_depth` $\in \{4, 6, 8\} \times$ `min_samples_leaf` $\in \{250, 500, 1000\}$.
  - LightGBM: `num_leaves` $\in \{7, 15, 31\} \times$ `min_child_samples` $\in \{250, 500, 1000\} \times$ `lambda_l2` $\in \{1, 10, 50\}$.
- **Calibration:** Each voter probability calibrated via cross-fitted Platt scaling or isotonic regression on its own train OOF predictions. Final threshold $\gamma$ fitted on Validation only (small grid, each inspection logged).
- **Combiner Formula:**
  $$d_{bin} = 2 p_{bin} - 1, \quad d_3 = P_3(\text{Up}) - P_3(\text{Down}), \quad d_{ta} \in [-1, 1]$$
  $$s = w_{bin} d_{bin} + w_3 d_3 + w_{ta} d_{ta}, \quad w \ge 0, \quad \sum w = 1 \text{ (grid step 0.1)}$$
  Signal: Up if $s > \tau_{dir}$, Down if $s < -\tau_{dir}$, $P(\text{Neutral}) < \tau_{neu}$, and $\ge k$ voters agree ($k \in \{2, 3\}$); otherwise Neutral ("No signal").
- If the combination does not beat the single best voter by more than fold standard deviation, the single best voter is selected as the production architecture.

### 10.10 Checkpoint 1 Empirical Results (Stage 1 Screening & Baselines)
- **Purged Walk-Forward CV Execution:**
  - 6 expanding folds on post-warmup train split ($N=218,935$ rows, warmup = 2,018 rows).
  - Evaluated held-out rows: 132,523 candles (~460 days of continuous evaluation).
  - Purge gap: exactly 12 candles strictly separating training and held-out blocks.
- **Stage 1 Screening (76 Logistic Regression Configurations on 70 LINEAR_FEATURES):**
  - Evaluated on top 20% most confident directional signals across candidate horizons and rules.
  - Ranked by $\text{rank\_metric} = \mu_{\text{gap}} - \sigma_{\text{gap}}$ on break-even gap ($hit - BE$).
  - **Survivor Summary (Direction Hit Rate shown next to Rank Metric):**
    | Rank | Model Config | Setup | Horizon | Rule | Rank Metric | Direction Hit Rate | Mean Break-Even Gap | Mean Net Return | Status |
    |---|---|---|---|---|---|---|---|---|---|
    | 1 | `C_12_F15` | Setup C | $N=12$ (60m) | $F_{15}$ | **-0.2140** | **52.90%** | -0.1759 | -21.27 bps | `survivor` |
    | 2 | `C_12_F10` | Setup C | $N=12$ (60m) | $F_{10}$ | **-0.2396** | **53.53%** | -0.1989 | -21.00 bps | `survivor` |
    | 3 | `C_12_F05` | Setup C | $N=12$ (60m) | $F_{05}$ | **-0.2775** | **53.87%** | -0.2259 | -21.08 bps | `survivor` |
    | 4 | `B_12` | Setup B | $N=12$ (60m) | `raw_sign` | **-0.3761** | **54.61%** | -0.2917 | -21.12 bps | `fallback` |
  - **Crucial Architectural Insight: Direction Skill vs. Size Selection:**
    - Notice that **`B_12` achieves the highest directional hit rate (54.61%)** among all Stage 1 survivors, outperforming `C_12_F15` (52.90%) by **+1.71 percentage points** in pure directional sign prediction.
    - The apparent +14.69 percentage point break-even gap advantage of `C_12_F15` over `loc_position_288` (and its +11.58 point advantage over `B_12`) is **primarily driven by a lower break-even hurdle from conditioning on volatile bars ($E[|fwd\_ret|]$ size selection)**, rather than superior directional sign forecasting.
    - Under sub-cost rules ($F_{15}, F_{10}, F_{05}$), the model triggers only when forward price displacement exceeds the threshold, selecting bars where average forward return is elevated ($E[|fwd\_ret|] \approx 0.58\%$ vs $0.34\%$ for unconditional bars). Because $BE = \frac{c}{2 \cdot E[|fwd\_ret|]} + 0.5$, this elevated return compresses the required break-even hurdle from ~83.7% down to ~70.5%.
    - Therefore, Setup C does not "prove" superior directional skill; rather, it combines modest directional skill (~52.9% - 53.9%) with implicit volatility gating that lowers the trading friction penalty.
  - **Model vs. Baseline Comparison:**
    - Baseline `loc_position_288` ($N=12$): BE Gap: -0.3228, Hit Rate: 51.83%, Net: -22.97 bps/trade.
    - Baseline Volatility-Gated Reversal (top NATR_14, $\text{sign}(0.5 - \text{loc\_288})$): BE Gap: -0.1942, Hit Rate: 52.33%, Net: -19.41 bps/trade.
    - Top Model (`C_12_F15`): BE Gap: -0.1759, Hit Rate: 52.90%, Net: -21.27 bps/trade.
    - The top Stage 1 linear model performs comparably to the simple volatility-gated reversal heuristic at identical coverage, demonstrating that explicit size/volatility conditioning accounts for the majority of the hurdle reduction.








