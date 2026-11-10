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


