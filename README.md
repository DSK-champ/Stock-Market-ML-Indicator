# AI-Powered Trading Indicator: 5-Minute BTCUSDT Directional Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course: CS3103](https://img.shields.io/badge/Course-CS3103%3A%20Machine%20Learning-indigo.svg)](https://www.iitp.ac.in/)
[![Author: 2401CS45](https://img.shields.io/badge/Author-Darla%20Sravan%20Kumar%20(2401CS45)-green.svg)](https://github.com/DarlaSravan)

> **Academic Project Submission**  
> **Course:** CS3103: Machine Learning | **Department:** Computer Science & Engineering, IIT Patna  
> **Student:** Darla Sravan Kumar | **Roll Number:** 2401CS45  
> **Target Asset:** Binance Spot BTCUSDT | **Resolution:** 5-Minute (=2$, 10-Minute Forward Horizon)  
> **Friction Floor:**  = 23.0\text{ bps}$ (.0023$) Production Cost Barrier  

---

## 1. Executive Summary & Core Achievements

This repository contains the complete end-to-end, leak-free machine learning framework for high-frequency Bitcoin directional forecasting. Designed to overcome the pervasive flaws of classical technical analysis (microstructure noise, silent repainting, and total disregard for trading frictions), the system formulates price direction as an asymmetric, cost-sensitive three-class problem:

y_t \in \{-1\ (\text{DOWN}),\ 0\ (\text{NEUTRAL}),\ +1\ (\text{UP})\}

The target incorporates an explicit round-trip economic cost dead zone:

c = \text{Taker Fee} \times 2 + \text{Slippage} + \text{Half-Spread} = 0.20\% + 0.02\% + 0.01\% = 0.23\% \quad (23.0\text{ bps})

Categorizing $>65\%$ of unexecutable, choppy price fluctuations as $\text{NEUTRAL}$ eliminates capital bleed from excessive churn. The models are served in a real-time streaming engine with an immutable Write-Ahead Logging (WAL) ledger, guaranteeing strict zero-repainting execution.

### Key Performance Highlights:
- **3-Class Out-of-Fold Accuracy:** **.49\%$** on 110,433 walk-forward validation candles (**$+9.16\%$** above the .33\%$ random baseline; Macro F1: **.4243$**).
- **Locked Held-Out Test Accuracy:** **.49\%$** on 49,938 un-seen candles (Macro F1: **.4324$**, Log-Loss: **.9701$**, Brier score: **.5752$**).
- **Directional Trade Hit Rate:** **.20\%$** on active Long/Short signals, achieving a statistically verified edge over the \%$ coin-flip hurdle ( < 0.001$).
- **Active Trade Coverage:** **.11\%$** (178.9 signals/day) driven by an asymmetric Directional Neutral Override rule ($\theta_{\text{dir}} = 0.32, \delta = 0.02$).
- **Zero-Repainting Execution:** Live Binance WebSocket stream (tcusdt@kline_5m) persisted to an append-only SQLite WAL ledger, resolving settlement at +2$ with offline--online parity strictly $< 10^{-7}$.
- **Instantaneous Reproducibility:** The entire scorecard, baseline comparisons, and test evaluations can be re-verified from frozen artifacts in **$< 1.0\text{ second}$** via python ML/src/main.py.

---

## 2. System Architecture: Tri-Branch Parallel Stacking

The system implements a hierarchical tri-branch architecture that routes 136 strictly causal market features according to model inductive bias, combined via a multinomial LightGBM meta-learner:

`
                            [ Live 5-Minute Binance Candlestick Stream ]
                                                 │
                                  [ Causal Feature Extraction ]
                                    (136 Factors in 17 Domains)
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         ▼                                       ▼                                       ▼
  [ Branch A: Binary Drift ]           [ Branch B: 3-Class Cost ]              [ Branch C: TA Engine ]
   • Logistic Regression (70 linear)    • Multinomial LR (70 linear)            • RSI(14) Momentum
   • Random Forest (136 all)            • Balanced Random Forest (136 all)      • EMA50 Trend Reversion
   • LightGBM Classifier (136 all)      • Multiclass LightGBM (136 all)         • Moving Average Alignment
         │                                       │                              • Dow Retest Rule Vote
         │ [P(Up)]                               │ [P_down, P_neu, P_up]        • 24h Channel Position
         │ (3 probabilities)                     │ (6 probabilities)            • Trend Strength
         └───────────────────────────────────────┼──────────────────────────────┴───────┬┘
                                                 │                                      │ (6 TA scores)
                                                 ▼                                      ▼
                                     [ 15-Dimensional Meta-Input Vector ]
                                                 │
                                                 ▼
                               [ Branch D: Multinomial LightGBM Meta-Learner ]
                                                 │
                                 [ Calibrated Probability Simplex ]
                                       [p_down, p_neu, p_up]
                                                 │
                                                 ▼
                                     [ Real-Time Governance Layer ]
                                 • Directional Neutral Override (θ_dir = 0.32)
                                 • Confidence Margin Filter (|p_up - p_down| >= 0.04)
                                 • Abstention Guard (gap fallback, NaN trap)
                                                 │
                                                 ▼
                               [ Emitted Signal: UP / DOWN / NEUTRAL ]
                                                 │
                                                 ▼
                          [ Immutable SQLite WAL Ledger (Never Repaints) ]
                                 (Resolved at candle t+2 close, 10 min)
`

---

## 3. Dataset Provenance & Anti-Leakage Protocol

- **Data Source:** Official Binance Spot REST API (GET /api/v3/klines) and public data archives for BTCUSDT.
- **Temporal Span:** Exactly **318,240 consecutive 5-minute candles** spanning 1,105 calendar days (1 November 2023 00:00 UTC to 10 November 2026 00:00 UTC).
- **Sampling Integrity:** 100% continuous (gap_id = 1.0), zero duplicate timestamps, zero missing bars, and zero timestamp inversions.
- **Attributes:** 9 raw variables per candle (open, high, low, close, olume, quote_volume, 
umber_of_trades, 	aker_buy_base_volume, 	aker_buy_quote_volume).

### Chronological Splits & Embargo Buffers:
| Partition | Date Range (UTC) | Candle Count | Calendar Days | Share | Role & Access Governance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | 2023-11-01 00:00 to 2025-12-07 04:40 | 220,953 | 767.2 | 69.43% | Feature engineering, EDA, scaler fitting, base-model training |
| **Purge Buffer 1** | 2025-12-07 04:45 to 2025-12-07 05:40 | 12 | 0.04 | 0.00% | 60-minute embargo buffer (eliminates boundary overlap) |
| **Validation** | 2025-12-07 05:45 to 2026-05-20 14:15 | 47,347 | 164.4 | 14.88% | 6-fold walk-forward CV, meta-learner tuning, threshold sweep |
| **Purge Buffer 2** | 2026-05-20 14:20 to 2026-05-20 15:15 | 12 | 0.04 | 0.00% | 60-minute embargo buffer (eliminates boundary overlap) |
| **Held-Out Test** | 2026-05-20 15:20 to 2026-11-09 23:55 | 49,940 | 173.4 | 15.69% | Programmatically locked (TestSetLockedError); one-time audit |

> **Anti-Leakage Safeguard:** All normalizers (StandardScaler), rolling quantiles, and probability calibrators are strictly fitted on the training split only and frozen. Access to the test split is guarded by TestSetLockedError in src/splitting.py to prevent data snooping.

---

## 4. Feature Taxonomy (136 Causal Factors)

The pipeline constructs 136 strictly causal features across 17 distinct families:
1. **Returns (5):** Multi-horizon log returns ( \in \{1, 3, 6, 12, 24\}$).
2. **Volatility (9):** Rolling standard deviations (12, 60, 288 bars), volatility ratios, NATR(14), and Bollinger Band indicators.
3. **Extreme Range Volatility (6):** Parkinson and Garman--Klass estimators over 12, 60, and 288 bars.
4. **Realized Volatility & Jumps (6):** Absolute variation sums, maximum price excursions, and upside semivariance ratios.
5. **Seasonal Volatility (4):** 7-day same-slot diurnal mean absolute return ratios and volume ratios.
6. **HAR Volatility Cascades (4):** Heterogeneous Autoregressive components (6-hour and 7-day cascades).
7. **Candle Microstructure (6):** Candle body, body-to-range, upper/lower wick percentages, close location, and candle polarity.
8. **Order Flow Imbalance (6):** Taker Buy Ratio (TBR), rolling z-scores, trade frequency, and average trade size.
9. **Cumulative Flow Windows (6):** Net aggressive taker volume delta and price-signed flow over 6, 12, and 36 bars.
10. **Higher-Timeframe 1h Context (5):** Completed 1-hour candle return, range, volume ratio, SMA ratio, and RSI(6).
11. **Moving Average Structure (36):** Term structure of SMA/EMA ratios, slope momentum, and ribbon alignment on 5m and 1h bars.
12. **Classical Momentum (13):** RSI(7), RSI(14), MACD line/signal/histogram, and ADX(14) trend regime.
13. **Volume Surges (5):** Log volume, rolling volume z-scores, volume rate of change, and volume ratios.
14. **Dow Theory Market Structure (15):** Swing point detection (20-bar extrema), Break of Structure (BOS), support/resistance distances, trend age, pullback depth, and rule votes.
15. **Range Location (3):** Position within the 24-hour channel and distances to 24-hour high and low.
16. **Diurnal Time (4):** Cyclical sine and cosine coordinates for time-of-day and day-of-week.
17. **Weekend Liquidity (3):** Weekend binary flag and 168-hour weekly cyclical coordinates.

**Inductive-Bias Screening:** Pairwise correlation screening ($|r| < 0.85$) and Variance Inflation Factor filtering ($	ext{VIF} < 10$) reduce the 136 features to **70 linear features** for Logistic Regression, eliminating Hessian singularity while giving all 136 features to Random Forest and LightGBM.

---

## 5. Performance Scorecard & Empirical Validation

Evaluated across 110,433 out-of-fold validation candles and 49,938 held-out test candles against 15 baseline heuristic strategies:

| Pipeline Stage / Model | Architecture | Feature Scope | OOF Accuracy | Test Accuracy | Macro F1 (Test) | Directional Hit Rate | Test Log-Loss | Inference Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | Zero-Rule | None | 31.41% | 38.88% | 0.2800 | n/a (0 trades) | 1.0986 | < 0.01 ms | 0 KB |
| **RSI(14) Momentum** | Heuristic | 1 Feature | 33.12% | 32.85% | 0.3450 | 54.99%* | 1.2104 | < 0.01 ms | 0 KB |
| **EMA50 Trend** | Heuristic | 1 Feature | 34.02% | 33.90% | 0.3480 | 51.24% | 1.1890 | < 0.01 ms | 0 KB |
| **inary_lr (Branch A)** | Logistic Reg | 70 Linear | 50.84% | 52.07% | 0.4922 | 52.07% | 0.6923 | 0.45 ms | 18 KB |
| **inary_rf (Branch A)** | Random Forest | 136 All | 51.42% | 51.73% | 0.4952 | 51.73% | 0.6919 | 4.80 ms | 2.45 MB |
| **inary_gb (Branch A)** | LightGBM | 136 All | 51.89% | 51.46% | 0.5152 | 51.46% | 0.6921 | 1.20 ms | 480 KB |
| **	hree_class_lr (Branch B)** | Multinomial LR | 70 Linear | 38.86% | 45.25% | 0.4140 | 50.20% | 1.0338 | 0.55 ms | 24 KB |
| **	hree_class_rf (Branch B)** | Random Forest | 136 All | 39.54% | 45.26% | 0.4156 | 50.80% | 1.0305 | 5.10 ms | 3.10 MB |
| **	hree_class_gb (Branch B)** | LightGBM 3C | 136 All | 40.82% | 45.91% | 0.4283 | 51.40% | 1.0273 | 1.35 ms | 512 KB |
| **	a_engine (Branch C)** | 6 TA Rules | 6 Rules | 51.85%* | 52.20%* | 0.3650 | 52.20% | n/a | 0.15 ms | 12 KB |
| **Hierarchical Meta-Model** | Stacking Combiner | 15 Meta | 42.15% | 45.49% | 0.4324 | 51.95% | 1.0287 | 0.85 ms | 195 KB |
| **Proposed Final System** | Meta + TA + Override | 15 Meta + Rules | **42.49%** | **45.49%** | **0.4324** | **52.20%** | **0.9701** | **< 15 ms** | **7.20 MB** |

*\*Note: RSI and TA hit rates apply only to low-coverage signal filters (e.g., RSI fires on only 7.18% of candles, bleeding -22.33 bps after round-trip taker friction). The proposed system maintains a 52.20% directional hit rate across 62.11% active market coverage.*

---

## 6. Repository Layout

`
Stock-Market-ML-Indicator/
├── README.md                            Master project documentation (this file)
├── requirements.txt                     Root environment dependencies
│
├── ML/                                  Machine Learning Research & Production Pipeline
│   ├── README.md                        ML sub-package documentation & reproduction guide
│   ├── requirements.txt                 Pinned Python dependencies
│   ├── data/
│   │   ├── raw/                         Audited Binance BTCUSDT 5m CSV files
│   │   └── processed/                   Cached feature matrices & ground-truth targets
│   ├── models/
│   │   └── horizon_2/                   Frozen model bundle (.joblib artifacts & scalers)
│   ├── notebooks/
│   │   ├── 01_Data_Collection.ipynb     Binance REST API ingestion & verification
│   │   ├── 02_Define_Targets.ipynb      Chronological boundaries & 40-rule friction targets
│   │   ├── 03_EDA.ipynb                 Exploratory data analysis on isolated Train split
│   │   ├── 04_Feature_Engineering.ipynb Causal feature extraction (136 factors)
│   │   ├── 05_Model_Training.ipynb      Base classifiers & cross-validation
│   │   ├── 05b_Horizon_N2.ipynb         Primary operational model training & meta-stacking
│   │   ├── 06_Model_Evaluation.ipynb    Out-of-fold calibration & threshold optimization
│   │   ├── 07_Backtesting.ipynb         One-time test partition evaluation
│   │   ├── 08_Export.ipynb              Production bundle serialization & parity testing
│   │   └── 09_Triple_Barrier.ipynb      Triple-barrier post-freeze extension
│   ├── results/
│   │   ├── figures/                     High-resolution publication figures (Fig 3.1 to 4.7)
│   │   ├── oof/                         Out-of-fold prediction matrices (110,433 rows)
│   │   └── reports/                     CSV summaries (EDA, baselines, drift, feature power)
│   ├── src/                             Modular production library
│   │   ├── baselines.py                 15 benchmark heuristic implementations
│   │   ├── config.py                    Centralized configuration, fee rates, horizons
│   │   ├── evaluate.py                  Statistical metrics & bootstrap confidence intervals
│   │   ├── features.py                  Causal 136-factor extraction pipeline
│   │   ├── horizon_eval.py              Multi-horizon evaluation routines
│   │   ├── io_utils.py                  Pure CSV & Parquet I/O with checksums
│   │   ├── labels.py                    Friction-aware target generators (S1, S2, Q1, V1...)
│   │   ├── main.py                      Sub-second CLI scorecard verification tool
│   │   ├── meta_model.py                Hierarchical stacking & LightGBM meta-learner
│   │   ├── model_common.py              Shared cross-validation & feature alignment
│   │   ├── splitting.py                 Purged walk-forward splitting & test lock
│   │   ├── structure_features.py        Dow Theory market structure & swing points
│   │   └── trading_engine.py            Order simulator & execution cost metrics
│   └── tests/                           Automated unit & integration test suite
│
├── backend/                             Real-Time Streaming Engine
│   ├── engine.py                        FastAPI application with WebSocket broadcast
│   ├── live/                            WebSocket client consuming Binance kline feed
│   └── data/                            Immutable SQLite WAL ledger database
│
└── frontend/                            Interactive Trading Dashboard
    ├── charting_library/                TradingView Lightweight Charts v4 integration
    ├── custom-indicators.js             Real-time ML signal renderer & probability gauge
    ├── trading.html                     Live multi-pane execution dashboard
    └── package.json                     Frontend dependencies & Vite scripts
`

---

## 7. Quickstart & Step-by-Step Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.12 64-bit on Windows, Linux, and macOS)
- Node.js 18+ (for frontend dashboard)

### 1. Installation
Clone the repository and set up a dedicated virtual environment:
`ash
git clone https://github.com/DarlaSravan/Stock-Market-ML-Indicator.git
cd Stock-Market-ML-Indicator

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r ML/requirements.txt
`

### 2. Verify Scorecards in Under 1 Second (One-Command Verification)
Execute the CLI reproduction tool to reload frozen production bundles and verify all test metrics:
`ash
python ML/src/main.py
`
*Expected Output:*
`	ext
================================================================================
BTCUSDT 5M DIRECTIONAL INDICATOR: PRODUCTION AUDIT
================================================================================
[INFO] Loading production model bundle (models/horizon_2/production_bundle.joblib)...
[INFO] Bundle loaded in 0.55s. Models: 7 classifiers + 1 meta-model.
[SUCCESS] Out-of-fold Accuracy: 42.49% | Macro F1: 0.4243
[SUCCESS] Held-Out Test Accuracy: 45.49% | Test Macro F1: 0.4324 | Log-Loss: 0.9701
[SUCCESS] Active Directional Hit Rate: 52.20% | Active Coverage: 62.11%
[AUDIT] Offline-Online Parity: Verified (|Diff| < 1e-7). All assertions passed!
================================================================================
`

### 3. Run Automated Tests
`ash
pytest ML/tests/ -v
`

### 4. Launch Live Streaming Backend
Start the real-time Binance WebSocket streaming consumer and REST API server:
`ash
uvicorn backend.engine:app --host 0.0.0.0 --port 8000 --reload
`
- API Health Check: http://localhost:8000/health
- WebSocket Feed: ws://localhost:8000/ws/signals
- SQLite WAL Ledger: ackend/data/live_signals.db

### 5. Launch Frontend Trading Dashboard
In a separate terminal:
`ash
cd frontend
npm install
npm run dev
`
Open http://localhost:5173 (or open rontend/trading.html directly in a modern web browser) to view live 5-minute candlesticks, probability simplex bars, and directional entries overlaid in real time.

---

## 8. Academic Deliverables & Repository Documents

- **Master Technical Report:** Formally typeset in LaTeX with complete mathematical formulations: [2401CS45_Report.pdf](../2401CS45_Report.pdf) (source: 
eport.tex).
- **Research Poster:** Slide presentation and poster: [2401CS45_Poster.pptx](../2401CS45_Poster.pptx).
- **Dataset Specification Card:** Standard machine learning data sheet: [ML/docs/DATASET_CARD.md](ML/docs/DATASET_CARD.md).
- **Core Figures:** All 10 publication figures are archived in [igures/](../figures/).

---

## 9. Ethics, Limitations & Financial Risk Disclaimers

1. **Intended Academic Use:** This system is developed strictly for university research, scientific validation, and paper-trading evaluation in the CS3103 Machine Learning curriculum at IIT Patna.
2. **Execution Risk in Cryptocurrencies:** High-frequency cryptocurrency spot markets exhibit liquidity dislocations, counterparty risk, and regime shifts. Historical statistical advantages do not guarantee future live profitability.
3. **Taker vs. Maker Realities:** Unconditioned market-taking paying 23~bps taker fees on 10-minute bars requires an unrealistic $>82\%$ hit rate to break even. The model provides a statistically verified .20\%$ directional edge; real-world commercial exploitation requires passive limit-order execution (0~bps maker fee tier) to monetize this predictive advantage.
4. **No Real Capital Deployment:** This project was developed and verified entirely without real capital risk. The authors accept no responsibility for financial losses resulting from unauthorized live deployment.

---

## 10. Citation

`ibtex
@techreport{sravan2026btcusdt,
  title       = {Cost-Sensitive Hierarchical Machine Learning for 5-Minute Bitcoin Directional Forecasting},
  author      = {Darla Sravan Kumar},
  institution = {Indian Institute of Technology Patna},
  number      = {2401CS45},
  year        = {2026},
  month       = {November}
}
`
