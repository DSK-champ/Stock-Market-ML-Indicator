# AI-Powered Trading Indicator: 5-Minute BTCUSDT Directional Forecasting & Dynamic Trade Zones

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Demo Video](https://img.shields.io/badge/YouTube-Live%20Demo%20Video-FF0000.svg?logo=youtube&logoColor=white)](https://youtu.be/qyLPa5GDynU)
[![Course: CS3103](https://img.shields.io/badge/Course-CS3103%3A%20Machine%20Learning-4F46E5.svg)](https://www.iitp.ac.in/)
[![Institution: IIT Patna](https://img.shields.io/badge/Institution-IIT%20Patna-0D9488.svg)](https://www.iitp.ac.in/)
[![Author: 2401CS45](https://img.shields.io/badge/Author-Darla%20Sravan%20Kumar%20(2401CS45)-10B981.svg)](https://github.com/DarlaSravan)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg)](https://opensource.org/licenses/MIT)

> **Academic Project Submission**  
> **Course:** CS3103: Machine Learning | **Department:** Computer Science & Engineering, IIT Patna  
> **Student:** Darla Sravan Kumar | **Roll Number:** 2401CS45  
> **Target Asset:** Binance Spot BTCUSDT | **Resolution:** 5-Minute ($N=2$, 10-Minute Forward Horizon)  
> **Friction Floor:** $c = 23.0\text{ bps}$ ($0.0023$) Production Round-Trip Cost Barrier  
> **Repository:** [https://github.com/DSK-champ/Stock-Market-ML-Indicator](https://github.com/DSK-champ/Stock-Market-ML-Indicator)

---

## 📹 Video Demonstration

Watch the full end-to-end interactive demonstration of the live streaming engine, zero-repainting canvas chart, and model-informed trade zones:

[![Live System Demonstration](https://img.youtube.com/vi/qyLPa5GDynU/hqdefault.jpg)](https://youtu.be/qyLPa5GDynU)

> 🔗 **Direct Link:** [https://youtu.be/qyLPa5GDynU](https://youtu.be/qyLPa5GDynU)  
> *Demonstrates live WebSocket ingestion, instant feature extraction, stacked meta-inference, dynamic trade zone projections ($t \rightarrow t+2$), and per-candle history review on held-out test periods.*

---

## 1. Summary & Achievements

This repository contains the complete end-to-end, leak-free machine learning framework for high-frequency Bitcoin directional forecasting and disciplined trade execution. Designed to resolve the pervasive flaws of classical technical indicators—microstructure noise, silent repainting, and total disregard for trading frictions—the system formulates price direction as an asymmetric, cost-sensitive three-class problem:

$$y_t \in \{-1\ (\text{DOWN}),\ 0\ (\text{NEUTRAL}),\ +1\ (\text{UP})\}$$

The target explicitly incorporates an audited round-trip transaction cost barrier:

$$c = \text{Taker Fee} \times 2 + \text{Slippage} + \text{Half-Spread} = 0.20\% + 0.02\% + 0.01\% = 0.23\% \quad (23.0\text{ bps})$$

Categorizing $>65\%$ of unexecutable, choppy price fluctuations as $\text{NEUTRAL}$ eliminates capital bleed from excessive churn. Moving beyond isolated class labels, the system dynamically derives **Model-Informed Trade Zones** (green entry envelopes, volatility-calibrated stop losses, and fee-clearing targets) served through an asynchronous FastAPI/WebSocket streaming engine backed by an append-only SQLite Write-Ahead Logging (WAL) ledger that guarantees zero repainting.

### Key Performance Highlights:
* **3-Class Out-of-Fold Accuracy:** **42.49%** on 110,433 walk-forward validation candles (**+11.08%** above the 31.41% majority-class baseline; Macro F1: **0.4243**).
* **Locked Held-Out Test Accuracy:** **45.49%** on 49,938 unseen test candles (Macro F1: **0.4324**, Log-Loss: **0.9701**, Brier score: **0.5752**).
* **Directional Trade Hit Rate:** **52.20%** on active Long/Short signals, establishing a statistically verified edge over the 50% coin-flip hurdle ($p < 0.001$).
* **Active Market Coverage:** **62.11%** (189.8 signals/day) unlocked by an asymmetric Directional Neutral Override rule ($\theta_{\text{dir}} = 0.32, \delta = 0.02$).
* **Dynamic Trade Zones:** Generates forward-projected ATR entry depths, Garman–Klass volatility stops, and fee-clearing targets ($t \rightarrow t+2$) with a 12-factor consensus agreement score.
* **Strict Zero Repainting:** Live Binance stream (`btcusdt@kline_5m`) persisted to an append-only ledger, resolving settlement at candle $t+2$ with offline–online numerical parity strictly $< 10^{-7}$.
* **Sub-Second Reproducibility:** Entire scorecards, baseline comparisons, and test evaluations reproduce from frozen artifacts in **$< 0.55\text{ seconds}$** via `python ML/src/main.py`.

---

## 2. Performance Scorecard & Empirical Validation

Evaluated across 110,433 out-of-fold validation candles and 49,938 held-out test candles against 15 benchmark heuristic strategies under the 23 bps friction floor:

| Pipeline Stage / Model | Architecture | Feature Scope | OOF Accuracy | Test Accuracy | Macro F1 (Test) | Directional Hit Rate | Test Log-Loss | Inference Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | Zero-Rule | None | 31.41% | 38.88% | 0.2800 | n/a (0 trades) | 1.0986 | < 0.01 ms | 0 KB |
| **RSI(14) Momentum** | Heuristic | 1 Feature | 33.12% | 32.85% | 0.3450 | 54.99%* | 1.2104 | < 0.01 ms | 0 KB |
| **EMA50 Trend** | Heuristic | 1 Feature | 34.02% | 33.90% | 0.3480 | 51.24% | 1.1890 | < 0.01 ms | 0 KB |
| **Binary LR (Branch A)** | Logistic Reg | 70 Linear | 50.84% | 52.07% | 0.4922 | 52.07% | 0.6923 | 0.45 ms | 18 KB |
| **Binary RF (Branch A)** | Random Forest | 136 All | 51.42% | 51.73% | 0.4952 | 51.73% | 0.6919 | 4.80 ms | 2.45 MB |
| **Binary GB (Branch A)** | LightGBM | 136 All | 51.89% | 51.46% | 0.5152 | 51.46% | 0.6921 | 1.20 ms | 480 KB |
| **Three-Class LR (Branch B)** | Multinomial LR | 70 Linear | 38.86% | 45.25% | 0.4140 | 50.20% | 1.0338 | 0.55 ms | 24 KB |
| **Three-Class RF (Branch B)** | Balanced RF | 136 All | 39.54% | 45.26% | 0.4156 | 50.80% | 1.0305 | 5.10 ms | 3.10 MB |
| **Three-Class GB (Branch B)** | LightGBM 3C | 136 All | 40.82% | 45.91% | 0.4283 | 51.40% | 1.0273 | 1.35 ms | 512 KB |
| **TA Engine (Branch C)** | 6 Rule Composite | 6 Rules | 51.85%* | 52.20%* | 0.3650 | 52.20% | n/a | 0.15 ms | 12 KB |
| **Hierarchical Meta-Model** | Stacking Combiner | 15 Meta | 42.15% | 45.49% | 0.4324 | 51.95% | 1.0287 | 0.85 ms | 195 KB |
| **Proposed Final System** | **Meta + TA + Override** | **15 Meta + Rules** | **42.49%** | **45.49%** | **0.4324** | **52.20%** | **0.9701** | **< 15 ms** | **7.20 MB** |

*\*Note: Heuristic RSI and TA rules achieve apparent high hit rates only under extreme low-coverage conditions (e.g., RSI fires on only 7.18% of candles, bleeding -22.33 bps net return due to transaction costs). The proposed system maintains a 52.20% directional hit rate across 62.11% active market coverage.*

---

## 3. System Architecture: Tri-Branch Parallel Stacking

The system implements a hierarchical tri-branch architecture that routes 136 strictly causal market features according to model inductive bias, combined via a calibrated multinomial LightGBM meta-learner:

```
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
                                 • Abstention Guard (gap fallback, zero NaN)
                                                 │
                                                 ▼
                               [ Emitted Signal: UP / DOWN / NEUTRAL ]
                                                 │
                                                 ▼
                             [ Model-Informed Dynamic Trade Zones ]
                         • Green Entry Zone (ATR-scaled, Dow-bounded)
                         • Target Price (23 bps fee floor + buffer)
                         • Red Stop Loss (Garman-Klass Volatility)
                                                 │
                                                 ▼
                          [ Immutable SQLite WAL Ledger (Never Repaints) ]
                                 (Resolved at candle t+2 close, 10 min)
```

---

## 4. Model-Informed Dynamic Trade Zones

Rather than outputting naked directional predictions, the system maps the meta-model probability simplex, 10-minute Garman–Klass volatility ($\sigma_{\text{GK}}$), and Dow Theory market structure into forward-projected execution zones ($t \rightarrow t+2$):

```
         ┌─────────────────────────────── TARGET LINE ───────────────────────────────┐  Target Level
         │                                                                           │  (clears 23 bps friction)
         │                                                                           │
         ├─────────────────────────────── ENTRY MAX ─────────────────────────────────┤
         │                                                                           │  GREEN ENTRY ZONE
         │  ====== [ Signal Candle t ] ========> [ Candle t+1 ] ========> [ Expiry t+2 ]  (ATR scaled by confidence)
         │                                                                           │
         ├─────────────────────────────── ENTRY MIN ─────────────────────────────────┤
         │                                                                           │
         │                                                                           │  RED STOP LOSS ZONE
         └──────────────────────────────── STOP LINE ────────────────────────────────┘  (Volatility & Dow bounded)
```

1. **Green Entry Range:** Centered at the signal candle close $P_t$. The depth scales dynamically with $\text{ATR}_{14}$ and contracts as directional confidence rises, bounded by structural Dow swing levels:
   $$\text{Depth}_t = \text{ATR}_{14} \times k_{\text{entry}} \times \left(1 - \frac{|p_{\text{up}} - p_{\text{down}}|}{\max(p_{\text{up}}, p_{\text{down}})}\right), \quad k_{\text{entry}} = 0.75$$
2. **Profit Target Level:** Enforces an asymmetric hurdle ensuring the expected excursion clears the 23 bps round-trip friction floor plus an execution buffer:
   $$\Delta_{\text{target}} = \max\left(P_t \times \sigma_{\text{GK},10} \times |p_{\text{up}} - p_{\text{down}}| \times \text{scale}, \; P_t \times (0.0023 + \text{buffer})\right), \quad \text{scale} = 1.50$$
3. **Red Stop-Loss Zone:** Anchors risk containment to the nearer of the structural Dow swing extreme or volatility distance:
   $$P_{\text{stop}} = \text{Entry} \mp (k_{\text{stop}} \times P_t \times \sigma_{\text{GK},10}), \quad k_{\text{stop}} = 1.25$$
   *(Dynamically widens by $1.35\times$ during high-volatility regimes $\text{ATR}_{14} / P_t \ge 0.0070$ to prevent whipsaws).*
4. **12-Factor Consensus Agreement:** Evaluates the proportion of the 6 base ML models and 6 TA scores that agree with the directional call:
   $$\text{Agreement Score} = \frac{1}{12} \sum_{i=1}^{12} \mathbb{I}(\text{Vote}_i = A_t) \times 100\%$$
   Signals with $<50\%$ agreement are marked "Conflicting Signals" and their visual zone opacity is attenuated.
5. **Interactive UI Selection & Verdicts:** The web dashboard renders zones dynamically for the current active candle and provides interactive checkboxes per row in the Zone History table to review past setups. Historical zones display clear **`PROFIT`**, **`LOSS`**, or **`ACTIVE`** badges with zero repaint.

---

## 5. Dataset Provenance & Anti-Leakage Protocol

* **Data Source:** Official Binance Spot REST API (`GET /api/v3/klines`) and public data archives for BTCUSDT.
* **Temporal Span:** Exactly **318,240 consecutive 5-minute candles** spanning 1,105 calendar days (1 November 2023 00:00 UTC to 10 November 2026 00:00 UTC).
* **Integrity Audit:** 100% continuous (`gap_id = 1.0`), zero duplicate timestamps, zero missing bars, and zero OHLC inversions.
* **Attributes:** 9 raw variables per candle (open, high, low, close, volume, quote_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume).

### Chronological Splits & Embargo Buffers:
| Partition | Date Range (UTC) | Candle Count | Calendar Days | Share | Role & Access Governance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | 2023-11-01 00:00 to 2025-12-07 04:40 | 220,953 | 767.2 | 69.43% | Feature engineering, EDA, scaler fitting, base-model training |
| **Purge Buffer 1** | 2025-12-07 04:45 to 2025-12-07 05:40 | 12 | 0.04 | 0.00% | 60-minute embargo buffer (eliminates boundary overlap) |
| **Validation** | 2025-12-07 05:45 to 2026-05-20 14:15 | 47,347 | 164.4 | 14.88% | 6-fold walk-forward CV, meta-learner tuning, threshold sweep |
| **Purge Buffer 2** | 2026-05-20 14:20 to 2026-05-20 15:15 | 12 | 0.04 | 0.00% | 60-minute embargo buffer (eliminates boundary overlap) |
| **Held-Out Test** | 2026-05-20 15:20 to 2026-11-09 23:55 | 49,940 | 173.4 | 15.69% | Programmatically locked (`TestSetLockedError`); one-time audit |

> **Anti-Leakage Safeguard:** All normalizers (`StandardScaler`), rolling quantiles, and probability calibrators are strictly fitted on the training split only and frozen. Access to the test split is guarded by `TestSetLockedError` in `ML/src/splitting.py` to prevent data snooping.

---

## 6. Feature Taxonomy (136 Causal Factors)

The pipeline constructs 136 strictly causal features across 17 distinct families in `ML/src/features.py`:
1. **Multi-Horizon Returns (5):** Log returns across $k \in \{1, 3, 6, 12, 24\}$ bars.
2. **Volatility Levels (9):** Rolling standard deviations (12, 60, 288 bars), volatility ratios, NATR(14), and Bollinger %B / bandwidth.
3. **Extreme Range Volatility (6):** Parkinson and Garman–Klass estimators across 12, 60, and 288 bars.
4. **Realized Volatility & Jumps (6):** Absolute variation sums, maximum price excursions, and upside semivariance ratios.
5. **Seasonal Volatility (4):** 7-day same-slot diurnal mean absolute return ratios and volume ratios.
6. **HAR Volatility Cascades (4):** Heterogeneous Autoregressive components (6-hour and 7-day cascades).
7. **Candle Microstructure (6):** Candle body ratio, body-to-range, upper/lower wick percentages, close location, and polarity.
8. **Order Flow Imbalance (6):** Taker Buy Ratio (TBR), rolling z-scores, trade frequency, and average trade size.
9. **Cumulative Flow Windows (6):** Net aggressive taker volume delta and price-signed flow over 6, 12, and 36 bars.
10. **Higher-Timeframe 1h Context (5):** Completed 1-hour candle return, range, volume ratio, SMA ratio, and RSI(6).
11. **Moving Average Ribbons (36):** Term structure of SMA/EMA ratios, slope momentum, and ribbon alignment on 5m and 1h bars.
12. **Classical Oscillators (13):** RSI(7), RSI(14), MACD line/signal/histogram, and ADX(14) trend regime.
13. **Volume Surges (5):** Log volume, rolling volume z-scores, volume rate of change, and volume ratios.
14. **Dow Theory Market Structure (15):** Swing point detection (20-bar extrema), Break of Structure (BOS), support/resistance distances, trend age, pullback depth, and rule votes.
15. **Range Location (3):** Position within the 24-hour channel and distances to 24-hour high and low.
16. **Diurnal Time (4):** Cyclical sine and cosine coordinates for time-of-day and day-of-week.
17. **Weekend Liquidity (3):** Weekend binary flag and 168-hour weekly cyclical coordinates.

---

## 7. Repository Layout

```
Stock-Market-ML-Indicator/
├── start_dashboard.py                   Unified single-command launcher (runs all 3 services)
├── pytest.ini                           Testing configuration
├── README.md                            Master project documentation (this file)
│
├── ML/                                  Machine Learning Research & Production Pipeline
│   ├── README.md                        ML sub-package documentation & reproduction guide
│   ├── requirements.txt                 Pinned Python dependencies
│   ├── data/                            Audited Binance BTCUSDT 5m CSV files & feature caches
│   ├── models/                          Frozen model bundles (.joblib artifacts & scalers)
│   ├── notebooks/                       Full experimental notebooks (01 to 09)
│   ├── results/                         High-resolution figures (Fig 3.1 to 4.7) & OOF matrices
│   ├── src/                             Modular production library
│   │   ├── baselines.py                 15 benchmark heuristic implementations
│   │   ├── config.py                    Centralized configuration, fee rates, horizons
│   │   ├── evaluate.py                  Statistical metrics & bootstrap confidence intervals
│   │   ├── features.py                  Causal 136-factor extraction pipeline
│   │   ├── main.py                      Sub-second CLI scorecard verification tool
│   │   ├── meta_model.py                Hierarchical stacking & LightGBM meta-learner
│   │   ├── splitting.py                 Purged walk-forward splitting & test lock
│   │   └── structure_features.py        Dow Theory market structure & swing points
│   └── tests/                           Automated unit & integration test suite
│
├── backend/                             Real-Time Streaming Engine
│   ├── engine.py                        FastAPI application with WebSocket broadcast
│   ├── live/                            Live predictor, ledger, and zone calculator
│   │   ├── api.py                       Live REST API endpoints (/api/live/...)
│   │   ├── predictor.py                 Real-time causal feature extractor & model evaluator
│   │   ├── ledger.py                    Append-only SQLite WAL ledger
│   │   ├── zone_calculator.py           Pure mathematical Trade Zones calculator
│   │   └── zone_config.py               Zone parameters (k_entry, k_stop, scale, buffer)
│   ├── data/                            Immutable SQLite WAL ledger database
│   └── tests/                           Live engine & zone calculator test suite
│
├── dashboard_api/                       Historical telemetry & analytics service
│   ├── app.py                           Telemetry API server (Port 8001)
│   └── data/                            Insights database & historical logs
│
└── frontend/                            Interactive Trading Dashboard
    ├── blank.html                       Native Canvas AI Trader Dashboard (with Trade Zones)
    ├── index.html                       TradingView Charting Library interface
    ├── assets/
    │   ├── css/                         Dashboard & Trade Zones stylesheets
    │   └── js/
    │       ├── native_chart.js          Hardware-accelerated 2D canvas charting engine
    │       ├── trade_zones_panel.js     Trade Zones panel controller & history manager
    │       ├── zone_calculator.js       Client-side mirror zone calculator
    │       ├── zone_chart_overlay.js    Chart overlay primitive plugin
    │       └── dashboard.js             Main application orchestrator & WebSocket client
    └── custom-indicators.js             PineJS custom indicator scripts
```

---

## 8. Quickstart & Step-by-Step Reproduction

### Prerequisites
* **Python 3.10+** (tested on Python 3.12 64-bit on Windows, Linux, and macOS)
* Modern web browser (Chrome, Edge, Firefox, Safari)

### 1. Installation
Clone the repository and set up a dedicated virtual environment:
```bash
git clone https://github.com/DarlaSravan/Stock-Market-ML-Indicator.git
cd Stock-Market-ML-Indicator

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
# source .venv/bin/activate

# Install ML & backend dependencies
pip install -r ML/requirements.txt
pip install -r requirements.txt
```

### 2. Verify Scorecards in Under 1 Second (One-Command Model Audit)
Execute the CLI reproduction tool to reload the frozen production bundle and verify all out-of-fold and held-out test scorecards:
```bash
python ML/src/main.py
```
*Expected Terminal Output:*
```text
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
```

### 3. Run Automated Unit & Integration Tests
Verify the complete test suite across ML pipelines and real-time zone calculators:
```bash
pytest backend/tests/ -v
pytest ML/tests/ -v
```
*(All 18 tests pass in under 30 seconds).*

### 4. Launch the Interactive Dashboard (Single-Command Launcher)
Run all 3 system services (Live Engine, Dashboard API, and Frontend Server) concurrently in a single terminal:
```bash
python start_dashboard.py
```
This command automatically opens your default browser at:
👉 **`http://localhost:8080/blank.html`**

* **Live Mode:** Connects to live Binance 5m WebSockets, updates indicators every candle close, and projects active green/red trade zones into the future.
* **Replay Mode (Test Split):** Step through or auto-play historical candles from the held-out test period with adjustable replay speed (1x to 120x).
* **Trade Zones Toggle:** Click the **🎯 Target Icon** in the top-right header to slide open the Trade Zones panel, adjust multipliers, inspect 12-factor evidence chips, and toggle past trade overlays.

---

## 9. Academic Deliverables & Project Files

* **Demo Video:** [https://youtu.be/qyLPa5GDynU](https://youtu.be/qyLPa5GDynU)
* **Master Technical Report (PDF):** Formally typeset in LaTeX: [2401CS45_Report.pdf](https://drive.google.com/file/d/1q099y4Z-sbMLyYf66GIqS4daF_lZsAOV/view?usp=drive_link)
* **Presentation Poster (PPTX & PDF):** Academic conference slide poster: [2401CS45_Poster.pptx](https://docs.google.com/presentation/d/1YwQHAlLOXcncgNiwURCSr8LG2OOx5A7R/edit?usp=drive_link&ouid=112085039378185484998&rtpof=true&sd=true)
* **Dataset Specification Card:** Machine learning documentation: [ML/docs/DATASET_CARD.md](ML/docs/DATASET_CARD.md)

---

## 10. Ethics, Limitations & Financial Risk Disclaimers

1. **Intended Academic Use:** This system is developed strictly for university research, scientific validation, and paper-trading evaluation in the CS3103 Machine Learning curriculum at IIT Patna.
2. **Execution Risk in Cryptocurrencies:** High-frequency cryptocurrency spot markets exhibit extreme liquidity dislocations, counterparty risk, and regime shifts. Historical statistical advantages do not guarantee future live profitability.
3. **Taker vs. Maker Realities:** Unconditioned market-taking paying 23 bps taker fees on 10-minute bars requires an unrealistic $>82\%$ hit rate to break even. The model provides a statistically verified 52.20% directional edge; real-world commercial exploitation requires passive limit-order execution (0 bps maker fee tier) to monetize this predictive advantage.
4. **No Real Capital Deployment:** This project was developed and verified entirely without real capital risk. The authors accept no responsibility for financial losses resulting from live deployment.

---

## 11. Citation

```bibtex
@techreport{sravan2026btcusdt,
  title       = {Cost-Sensitive Hierarchical Machine Learning Framework for High-Frequency Bitcoin Market Direction Prediction and Dynamic Trade Zones},
  author      = {Darla Sravan Kumar},
  institution = {Indian Institute of Technology Patna},
  department  = {Department of Computer Science and Engineering},
  number      = {2401CS45},
  year        = {2026},
  month       = {November},
  url         = {https://github.com/DarlaSravan/Stock-Market-ML-Indicator}
}
```
