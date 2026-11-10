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

## 7. Pre-Declared Experiment Budget & Elimination Protocol (Master Prompt v3 §7)
- **Grid Configuration Upper Bound:**
  - Horizons $N \in \{2, 3, 6, 12\}$ candles (4)
  - Threshold Rules: $\text{S1, S2, Q1, V1, V2, A1, A2}$ (7)
  - Training Setups: A (non-neutral only), B (raw sign on all rows), C (3-class weighted) (3)
  - Models: Logistic Regression, LightGBM (2)
  - Maximum Grid Space: $4 \times 7 \times 3 \times 2 = 168$ configurations.
- **Staged Elimination Protocol:**
  1. *Stage 1 (Screening via Logistic Regression):* Evaluate all Horizon $\times$ $\theta$-rule combinations using baseline Logistic Regression on Train walk-forward folds. Eliminate combinations with negative cost-adjusted return or near-zero coverage (<15 signals/day).
  2. *Stage 2 (Refinement via LightGBM):* Train top-performing combinations across setups A, B, and C with LightGBM.
- **Selection Criteria (on Train Walk-Forward Folds only):**
  1. Realized cost-adjusted return of emitted signals ($R_{\text{net}} = \text{signal} \times R_{\text{open}} - c$).
  2. Up/Down precision at fixed coverage.
  3. Fold-to-fold stability (Sharpe / variance across folds).
  4. Simplicity as tie-breaker. Default fallback: Horizon $N=12$.


