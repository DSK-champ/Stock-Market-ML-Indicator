# Machine-Readable Step Details Schema (`ML/details/`)

This directory contains automated, machine-readable JSON records of every dataset, configuration, and execution step throughout the **Stock-Market-ML-Predictor** lifecycle.

## Purpose & Integrity Guarantee
- Every data-file descriptor is generated **strictly by reading the saved file back from disk** through `src/io_utils.py` (never from memory).
- Cryptographic SHA-256 hashes, precise byte sizes, and Git commit references ensure end-to-end provenance and data integrity.
- Zero data leakage: per-column descriptive statistics, moments, and class distributions are computed on `train` rows (and separately on `val`). Out-of-sample `test` split blocks contain **only row counts and boundary timestamps** (`test` statistics are strictly withheld).
- Standard-compliant JSON: all `NaN` and `Inf` values are serialized as `null`; dates are ISO 8601 UTC strings.

## Directory Structure
| File | Created By | Target File / Purpose |
|---|---|---|
| `step01_raw.json` | Step 1 | `data/raw/BTCUSDT_5m_raw.csv` (12 untouched Binance columns) |
| `step01_clean.json` | Step 1 | `data/processed/clean.csv` (11 validated columns, UTC index) |
| `step02_split.json` | Step 2 | Train/Val/Test boundaries, purge gap (12 candles), candle counts |
| `step02_targets_base.json` | Step 2 | `data/processed/targets_base.csv` (fwd_ret, raw_sign, valid_target flags) |
| `step02_targets_labels.json` | Step 2 | `data/processed/targets_labels.csv` (3-class labels & θ thresholds) |
| `step03_eda.json` | Step 3 | Train distribution moments, volatility, figures produced |
| `step04_features.json` | Step 4 | `data/processed/features.csv`, factor groupings, lookback warm-up |
| `step05_experiments.json` | Step 5 | Walk-forward cross-validation results, configurations run & chosen |
| `step06_calibration.json` | Step 6 | Calibration curves, probability thresholds, signal coverage |
| `step07_backtest.json` | Step 7 | Unlocked test split backtest evaluation metrics & return series |
| `step08_bundle.json` | Step 8 | Exported production inference bundle and frozen model artifacts |
| `step09_triple_barrier.json` | Step 9 | Triple barrier labeling comparison and sensitivity metrics |
| `index.json` | Every Step | Master manifest tracking all detail files, sizes, and hashes |

## JSON Object Schema (Data Files)
- `file`: relative path, size in bytes, SHA-256 hash, creation timestamp, Git commit, module name, description.
- `shape`: `n_rows`, `n_columns`.
- `index`: name, dtype, timezone (`UTC`), start/end timestamps, monotonicity, duplicate count.
- `columns`: list of full metadata objects per column from `src/column_dictionary.py`:
  - `name`, `dtype`, `position`, `description`, `unit`, `formula`
  - `role`: one of `raw`, `cleaned`, `feature`, `target`, `threshold`, `flag`
  - `is_future_looking`: boolean (`true` for all targets)
  - `allowed_as_model_input`: boolean (`false` for all targets)
  - `n_missing`, `n_inf`, `n_unique`
  - `stats_train`, `stats_val` (min, max, mean, std, median)
- `row_counts_by_split`: train, val, test (test metadata only).
- `valid_rows`: count of valid target candles per horizon $N \in \{2, 3, 6, 12\}$.
- `parameters`: execution arguments (costs, candidate horizons, sigma window $W$, seeds).
- `checks`: data quality checks performed (gaps, overlaps, bounds).
- `warnings`: notices such as floor-binding rates or rule redundancies.

