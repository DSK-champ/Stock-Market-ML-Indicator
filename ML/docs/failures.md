# Failures and Negative Experiments Log

This document tracks all negative results, rejected architectures, discarded features, and modeling failures throughout the project lifecycle. A complete log of what did not work is mandatory for the university technical report.

---

## Log Entry 001 — Phase 0 Codebase Audit Findings
- **Date:** 2026-11-10
- **Attempted Approach:**
  1. *Unpinned Requirements:* `requirements.txt` with plain package names (`pandas`, `numpy`, `scipy`).
  2. *Live API Calls in Unit Tests:* Running `test_extraction.py` with immediate requests to Binance API during test runs.
  3. *In-place Row Iteration (`df.iterrows()`):* Validating candles sequentially in Python loops.
  4. *Unsplit Exploratory Data Analysis:* Conducting statistical distributions and returns histograms on the entire raw historical dataset before splitting.
  5. *Raw File Modification Pre-Storage:* Cleaning and deduplicating data before saving to `data/raw/`.
- **Observed Failure / Risk:**
  1. *Reproducibility Failure:* Library version mismatch across environments.
  2. *Test Fragility:* Tests fail without network access or when Binance rate limits trigger; mutates disk state during testing.
  3. *Computational Bottleneck:* `iterrows()` on 315,648 rows takes >2 minutes and pegging CPU, scaling poorly for repeated validations.
  4. *Critical Data Leakage:* Inspecting the full dataset prior to splitting leads to cognitive snooping on test set volatility and distribution extremes.
  5. *Loss of Ground Truth:* Overwriting raw downloaded data prevents auditing raw API discrepancies or verifying candle completeness.
- **Resolution / Redesign:**
  1. Pin all libraries using `pip freeze` into `requirements.txt`.
  2. Decouple unit tests using synthetic/mock datasets; mark live API checks as `@pytest.mark.integration` (skipped by default).
  3. Vectorize all DataFrame sanity checks in NumPy/Pandas.
  4. Postpone all EDA to Phase 2, strictly confined to the locked Training set.
  5. Isolate untouched raw API payload storage in `data/raw/` and keep all transformations in `data/processed/`.

---

## Log Entry 002 — Inaccurate Preliminary Tables & Formatting Bugs
- **Date:** 2026-11-10
- **Identified Problem:**
  1. *Unverified Preliminary Tables:* Summary tables in earlier reports and notebooks were not computed from the actual dataset and contained incorrect values across multiple metrics:
     - 5m return standard deviation was listed as `0.352%` (actual computed: `0.1528%`).
     - 1h forward return standard deviation was listed as `1.14%` (actual computed: `0.5135%`).
     - Price, volume, quote volume, and number of trades means, medians, skewness, and kurtosis were also wrong and did not match the real data distributions.
  2. *CSV Export String Formatting:* Used `float_format="%.10g"` in `to_csv` during Parquet-to-CSV migration, which truncated quote volume figures exceeding 8-9 integer digits (causing up to a 0.049 absolute discrepancy).
  3. *Validation Dictionary Key-Count Bug:* Evaluated violation counts using `len(dict)` instead of summing violation values, incorrectly displaying 5 OHLC and 4 volume violations when actual counts were 0.
- **Resolution:**
  1. *Complete Empirical Recomputation:* Discarded all unverified preliminary tables. Recomputed every single descriptive statistic, moment (skewness, kurtosis), volatility figure, and distribution directly from `clean.csv` in `01_Data_Collection.ipynb`. The true 5m log return standard deviation is 0.1528%, which directly squares with the 49.53% annualized volatility.
  2. *Precision Restoration:* Removed `float_format` override to allow pandas default repr-level precision (full IEEE 754 64-bit double precision). Round-trip test confirmed `max diff: 0.00e+00`.
  3. *Dictionary Summation Fix:* Corrected the validation check in notebook 01 to sum violation counts explicitly (`sum(v for k, v in ohlc_v.items() if k != 'valid')`), correctly displaying `0` violations.

---

## Log Entry 003 — Step 4 Fixes: Boundary Leakage, Uninformative Features, and Lookback Arithmetic
- **Date:** 2026-11-11
- **Identified Problems:**
  1. *Split Boundary Leakage in Forward Targets:* In continuous target computation, the last $N$ candles of the training partition evaluated forward returns using the first $N$ candles of the validation partition. This caused valid train rows to improperly equal 220,953 instead of $220,953 - N$, and shifted the Down count at $N=12$ under rule S1 from 50,663 to 50,668 (5 leaked validation candles).
  2. *Near-Constant Feature `feat_ret_open_gap`:* Computed as $\ln(\text{open}_t / \text{close}_{t-1})$. On Binance spot BTCUSDT trading 24/7 without market opens/closes, consecutive candle open prices almost perfectly equal previous close. The empirical std was $5.1 \times 10^{-6}$ and over 99% of values were exact zero floating-point noise.
  3. *Lookback Underestimation ($L=2017$ vs $L=2018$):* The initial assumption of $L=2017$ candles failed on `feat_vol_seasonal_slot_abs_ret_7d`. While 7 days corresponds to $7 \times 288 = 2016$ intervals, computing the 1-candle return at $t-2016$ requires $\text{close}_{t-2017}$. The total candle span from $t-2017$ to $t$ inclusive is $t - (t-2017) + 1 = 2018$ candles.
  4. *Pandas 2.2 Deprecation Warning:* Accessing `df.index.dayofweek` emitted 50 `Pandas4Warning` deprecations across test suites.
  5. *Mathematical Identities & Redundant Duplicates:* Implementing HAR multi-horizon realized volatility and semivariance introduced exact duplicates: `feat_vol_har_rv_12` $\equiv$ `feat_vol_sigma_12` (max diff 0.0), `feat_vol_har_rv_288` $\equiv$ `feat_vol_sigma_288` (max diff 0.0), and $\text{semi\_var\_down} \equiv 1.0 - \text{semi\_var\_up}$ (Pearson = -1.0). In addition, MACD exhibited the exact linear identity $hist = line - signal$.
- **Resolutions:**
  1. Passed `split_id` to `_build_fwd_ret` in `src/labels.py` to strictly invalidate forward horizons that cross split boundaries (`valid_mask = False`, labels = NaN). Restored exact valid row counts: $N=2$ (220,951), $N=3$ (220,950), $N=6$ (220,947), $N=12$ (220,941), and restored Down count at $N=12$ to 50,663.
  2. Completely removed `feat_ret_open_gap` from `FEATURE_GROUPS`, `src/features.py`, and `src/column_dictionary.py`.
  3. Formally corrected `LOOKBACK_CANDLES = 2018` in `src/config.py` and `src/features.py`. Empirical sweep confirmed 0 failing features and max difference $1.45 \times 10^{-10}$ across all 88 features at $L \ge 2018$.
  4. Updated day-of-week lookups to check `df.index.day_of_week` first, eliminating all 50 deprecation warnings.
  5. Formally removed the 3 exact duplicates (`feat_vol_har_rv_12`, `feat_vol_har_rv_288`, `feat_vol_realized_semi_var_down_ratio_60`), reducing features from 88 to 85. Retained all 3 MACD components in `master.csv` (172 columns total) but defined `LINEAR_MODEL_EXCLUDE = ["feat_mom_macd_hist"]` in `src/config.py` for linear/logistic models. Replaced heuristic prune list with hierarchical correlation clusters ($|\rho| > 0.95$, complete linkage).

---

## Log Entry 004 — Step 4b: Standalone Trend-Continuation Failure at 5m-1h Horizons & Structure Truncation Parity
- **Date:** 2026-11-11
- **Attempted Approach:**
  1. *Standalone Trend Continuation:* Testing whether structural trend-following signals (Dow-rule breakouts, swing pivot breaks, and the 4th rule voter) predict positive forward directional returns at 5m to 1h horizons ($N \in \{2, 3, 6, 12\}$) on BTCUSDT.
  2. *Naive Structural Target/Stop Orders:* Executing trades upon structural trend state transitions using candidate target and stop levels (with a $1.5 \times ATR$ floor).
  3. *Unbounded Pivot History for Level Merging:* Accumulating and merging horizontal support/resistance levels across an unbounded historical lookback.
- **Observed Failure / Risk:**
  1. *Opposition to Microstructure Reversal:* At 5m-1h horizons on BTCUSDT, price action exhibits persistent negative autocorrelation and mean reversion ($\rho \in [-0.045, -0.018]$ for momentum and MA location features). Standalone trend continuation bets (buying breakouts or trading in the direction of Dow uptrends) run directly counter to this empirical tendency, resulting in sub-50% forward win rates ($48.2\% - 49.5\%$) on raw directional signs.
  2. *Negative Net Expectancy after Round-Trip Costs:* Under cost-adjusted break-even $p^* = \frac{S + c}{T + S}$ ($c = 0.23\%$), the required break-even win rate averages $108.40\%$ due to small median target distances ($0.235\%$). Even with a $1.5 \times ATR$ floor, trading fee friction exceeds achievable win rates, yielding an empirical net loss of -23.4 bps per trade.
  3. *Row-State Overcounting vs True Event Transitions:* Evaluating `bars_since_break == 0` on every candle yielded 218,554 "breaks", which was a persistent row-state condition rather than discrete events. True level break transitions occur on only 190 candles across the 2.1-year Train history.
  4. *Trend Segment Lifespan vs Evaluation Horizons:* The empirical median Dow trend duration is only 9.0 candles (45 minutes; 14 candles for Up, 12 candles for Down). At holding horizon $N=12$ (60 minutes), trade holding duration exceeds the median trend lifespan, explaining the high rate of trend exhaustion and reversal.
  5. *Unbounded Pivot Buffer Drift in Streaming Environments:* Merging horizontal levels across unbounded historical pivots causes drift when comparing full-dataset calculations to streaming lookback buffers ($L=2018$ or $L=2880$). Without a bounded pivot memory, older distant levels survive indefinitely in batch mode but are omitted in truncated buffers.
- **Resolution / Redesign:**
  1. *Bounded Pivot Memory Buffer ($M_{pivots} = 50$):* Restrict active swing level tracking to the 50 most recent confirmed pivots. Since 50 confirmed pivots span at most ~508 candles in Train (well within $L=2018$), truncation parity achieves exact 0.0 discrepancy between batch computation and streaming buffers.
  2. *Strict Separation of Roles:* Prohibit raw price levels from entering the ML feature matrix (which would cause scale variance and leakage). All absolute price drawings are isolated to `data/processed/drawings.csv` with role `drawing`, while `master.csv` retains only scale-free ratios, ATR-normalized distances, and discrete state indicators (role `feature`).
  3. *Transition-Based Event Filtering & Multi-Voter Confluence:* Events are strictly defined by transition edges (`shift(1) != 0`). Standalone structural signals must not be traded in isolation. In Step 5 (modeling), structural trend votes (`feat_struct_rule_vote`) and MA alignment will be combined with volatility gating (`feat_vol_sigma_288` / `big_move_12`) and multi-voter consensus to filter out high-friction, chop-dominated regimes.




