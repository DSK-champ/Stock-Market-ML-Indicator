# Stock-Market-ML-Predictor: ML Core Package (BTCUSDT 5m)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)
[![Candidate: 2401CS45](https://img.shields.io/badge/Candidate-2401CS45-blue.svg)](https://github.com/DarlaSravan)

This directory contains the machine learning research, causal feature engineering, walk-forward training, model evaluation, and production bundle serialization for the 5-minute Bitcoin directional indicator system.

---

## 1. Directory Structure

`
ML/
├── README.md                        This document
├── requirements.txt                 Exact pinned Python dependencies
├── pytest.ini                       Pytest isolation configuration
│
├── notebooks/                       Ordered research & evaluation notebooks
│   ├── 01_Data_Collection.ipynb     Binance REST API downloader & integrity verification
│   ├── 02_Define_Targets.ipynb      Chronological partitions & 40-rule friction targets
│   ├── 03_EDA.ipynb                 Exploratory data analysis on isolated Train split
│   ├── 04_Feature_Engineering.ipynb Causal feature extraction (136 factors)
│   ├── 05_Model_Training.ipynb      Base classifiers & cross-validation benchmarks
│   ├── 05b_Horizon_N2.ipynb         Primary operational training (=2$, 10m horizon)
│   ├── 06_Model_Evaluation.ipynb    Out-of-fold calibration & threshold sweep
│   ├── 07_Backtesting.ipynb         One-time test partition evaluation
│   ├── 08_Export.ipynb              Production bundle serialization & parity testing
│   └── 09_Triple_Barrier.ipynb      Triple-barrier post-freeze extension
│
├── src/                             Modular production library
│   ├── __init__.py
│   ├── baselines.py                 15 benchmark heuristic strategies
│   ├── calibration.py               Platt scaling & isotonic regression calibrators
│   ├── config.py                    Centralized configuration (fees, paths, dates, rules)
│   ├── evaluate.py                  Metrics, confusion matrices, and bootstrap CIs
│   ├── extraction.py                Resumable Binance REST API downloader
│   ├── features.py                  Causal 136-factor extraction pipeline
│   ├── horizon_eval.py              Multi-horizon evaluation routines & coverage tiers
│   ├── io_utils.py                  Standardized CSV & Parquet I/O with checksums
│   ├── labels.py                    Friction-aware target generators (S1, S2, Q1, V1...)
│   ├── main.py                      Sub-second CLI scorecard verification tool
│   ├── master.py                    End-to-end dataset build & model orchestrator
│   ├── meta_model.py                Hierarchical stacking & LightGBM meta-learner
│   ├── model_common.py              Shared cross-validation & feature alignment
│   ├── plotting.py                  Visualization suite for figures
│   ├── preprocessing.py             Data cleaning, casting, and validation
│   ├── splitting.py                 Purged walk-forward splitting & test lock
│   ├── structure_features.py        Dow Theory market structure & swing points
│   ├── trading_engine.py            Order simulator & execution cost metrics
│   └── utils.py                     Random seeding, logging, utilities
│
├── models/
│   └── horizon_2/                   Frozen production bundles (.joblib) & scalers
│
├── results/
│   ├── figures/                     High-resolution publication figures (Fig 3.1 to 4.7)
│   ├── oof/                         Out-of-fold prediction matrices (110,433 rows)
│   └── reports/                     CSV summaries (EDA, baselines, drift, feature power)
│
├── data/
│   ├── raw/                         Raw Binance CSV files
│   └── processed/                   Cached feature matrices & ground-truth targets
│
└── tests/                           Automated unit & integration test suite
`

---

## 2. Key Methodological Standards

1. **Strict Chronological Splitting:**
   - **Train:** 220,953 candles (.43\%$, Nov 1, 2023 to Dec 7, 2025).
   - **Purge Buffer 1:** 12 candles (60-minute embargo).
   - **Validation:** 47,347 candles (.88\%$, Dec 7, 2025 to May 20, 2026).
   - **Purge Buffer 2:** 12 candles (60-minute embargo).
   - **Held-Out Test:** 49,940 candles (.69\%$, May 20, 2026 to Nov 9, 2026).
2. **Programmatic Test-Set Lock:**
   - Any unauthorized read of test data raises TestSetLockedError in src/splitting.py.
   - The test set is evaluated only once on frozen model bundles.
3. **Causal Feature Extraction:**
   - All 136 features at timestamp $ depend strictly on closed candles ' \le t$.
   - Multi-timeframe 1-hour features strictly lag completed 1-hour bars forward-filled onto the 5m grid.
4. **Economic Friction Awareness:**
   - Cost floor  = 23.0\text{ bps}$ (.0023$) models round-trip Binance VIP 0 taker fees (.20\%$), adverse slippage (.02\%$), and half-spread (.01\%$).
   - Prevents the models from fitting on sub-friction noise.

---

## 3. Operational Model Performance Summary (=2$, 10 Minutes)

| Metric | Out-of-Fold Validation (110,433 rows) | Held-Out Test (49,938 rows) | Benchmark Target |
| :--- | :---: | :---: | :---: |
| **Three-Class Accuracy** | **42.49%** | **45.49%** | $> 33.33\%$ (Random) / $> 40.0\%$ |
| **Macro F1-Score** | **0.4243** | **0.4324** | $> 0.4100$ |
| **Balanced Accuracy** | **43.72%** | **43.72%** | $> 40.00\%$ |
| **Cross-Entropy Log Loss** | **1.0287** | **0.9701** | $< 1.0986$ (Uniform) |
| **Multiclass Brier Score** | **0.5752** | **0.5752** | $< 0.6000$ |
| **Active Directional Hit Rate** | **52.20%** | **52.20%** | $> 51.50\%$ ( < 0.001$) |
| **Active Trade Coverage** | **62.11%** | **61.12%** | $> 60.00\%$ |
| **Extreme Polar Errors** | **< 18%** | **< 18%** | Minimized capital risk |

---

## 4. Quickstart Commands

### Install Dependencies
`ash
pip install -r requirements.txt
`

### Reproduce Scorecards in Under 1 Second
`ash
python src/main.py
`

### Run Test Suite
`ash
pytest tests/ -v
`

### Run Model Training Pipeline
`ash
python -m src.master --horizon 2
`
