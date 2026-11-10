# Stock-Market-ML-Predictor: AI-Powered Trading Indicator (BTCUSDT 5m)

An end-to-end, leak-free machine learning indicator system for university research and live trading evaluation on Bitcoin (BTCUSDT, Binance 5-minute timeframe).

Predicts short-term market direction with calibrated probabilities, economic dead zones (accounting for taker fees, slippage, and spread), zero repainting, and defensible out-of-sample validation.

---

## 1. Project Directory & File Structure

```
ML/
├── README.md                        System guide and setup instructions
├── requirements.txt                 Exact pinned library dependencies (no PyArrow required)
├── pytest.ini                       Pytest configuration (integration tests isolated)
├── .gitignore                       Git exclusions (data, models, caches)
│
├── notebooks/                       Interactive experiments (import from src/ only)
│   ├── 01_Data_Collection.ipynb     Download, clean, validate, full-period overview
│   ├── 02_Define_Targets.ipynb      Split boundaries, θ rules from TRAIN, label creation
│   ├── 03_EDA.ipynb                 EDA on TRAIN ONLY, includes targets
│   ├── 04_Feature_Engineering.ipynb Causal features & correlation analysis
│   ├── 05_Model_Training.ipynb      Baselines, setups A/B/C, horizon & θ comparison, ablation
│   ├── 06_Model_Evaluation.ipynb    Walk-forward diagnostics, calibration, threshold
│   ├── 07_Backtesting.ipynb         One-time locked test-set backtest
│   ├── 08_Export.ipynb              Freeze bundle, parity check
│   └── 09_Triple_Barrier.ipynb      Extension, post-freeze target/stop-loss evaluation
│
├── src/                             Modular, documented library code
│   ├── __init__.py
│   ├── config.py                    All global constants, paths, dates, fees, horizons
│   ├── io_utils.py                  Centralized single-entry CSV reader and writer
│   ├── utils.py                     Random seeding, logging, utilities
│   ├── extraction.py                Resumable Binance klines downloader (raw untouched)
│   ├── validation.py                Vectorized data quality and gap reporting
│   ├── preprocessing.py             Data cleaning, casting, and CSV export
│   ├── splitting.py                 Chronological splitting & test-set lock
│   ├── labels.py                    3-class labels with asymmetric thresholds & dead zone
│   ├── triple_barrier.py            Triple-barrier target and stop-loss logic
│   ├── plotting.py                  Modular, publication-ready visualization suite
│   ├── features.py                  Causal factor engineering (returns, vol, order flow, time)
│   ├── baselines.py                 Benchmark models (majority, momentum, RSI rule)
│   ├── train.py                     Model training & experiment logging
│   ├── calibration.py               Platt / Isotonic probability calibration
│   ├── evaluate.py                  Metrics, diagnostics, confidence intervals
│   ├── backtest.py                  Cost-adjusted trading simulation
│   ├── artifacts.py                 Bundle serialization & versioning
│   ├── inference.py                 Single-candle causal inference interface
│   └── main.py                      End-to-end offline pipeline runner
│
├── tests/                           Automated pytest test suite
│   ├── test_extraction.py           API constants, interval conversions, live integration
│   ├── test_preprocessing.py        Dtypes, duplicates, UTC DatetimeIndex, CSV round-trip
│   ├── test_validation.py           Schema validation, OHLC sanity, non-negative volume
│   ├── test_splitting.py            Split boundaries, purge gap, test-lock verification
│   ├── test_labels.py               Label correctness, asymmetric θ, dead-zone bounds
│   ├── test_features_causal.py      Causal truncation test (no future leakage)
│   └── test_parity.py               Offline vs live inference parity
│
├── data/                            Market data storage
│   ├── raw/                         BTCUSDT_5m_raw.csv (untouched 12 Binance columns)
│   └── processed/                   clean.csv (UTC DatetimeIndex, 11 standardized columns)
│
├── results/                         Experimental outputs
│   ├── figures/                     Saved figures for reports and notebooks
│   ├── reports/                     Data quality report, backtest report
│   └── experiments.csv              Full multiple-comparison experiment budget registry
│
├── models/                          Frozen model bundles and serialized artifacts
└── docs/                            Documentation logs
    ├── decisions.md                 Architecture & design decision records
    ├── sources.md                   Verification citations and fee/friction sources
    └── failures.md                  Negative experiment tracking & debugging post-mortems
```

---

## 2. Core Methodological Principles

1. **Pure CSV Storage Architecture:** All raw and processed market data is persisted in CSV via [`src/io_utils.py`](src/io_utils.py). Ensures universal inspectability and zero external binary C dependencies while guaranteeing IEEE 754 float precision and ISO 8601 UTC indexing.
2. **Strict Chronological Splits:** No random k-fold or future-looking shuffling. Train (220,953 candles), Validation (47,347 candles), and Test (49,940 candles, programmatically locked until final evaluation).
3. **Leakage Elimination & Purged Boundaries:** 
   - Feature engineering depends strictly on past closed candles ($t \le \text{current}$).
   - A purge gap equal to the horizon $N$ is enforced across all train/val and val/test boundaries.
   - All thresholds, quantiles, and scalers are fitted strictly on the isolated **Train** split and frozen.
4. **Economic Friction Awareness:** Every label is bounded by an economic dead zone:
   $$c = \text{Taker Fee (0.20\%)} + \text{Slippage (0.02\%)} + \text{Spread (0.01\%)} = 0.23\%$$
   Ensures models only trigger on statistically and economically meaningful price movements.
5. **No Repainting & Calibrated Probability:** A signal emitted on a closed candle is immutable. Raw probabilities are calibrated via Platt scaling / Isotonic regression on validation folds.
