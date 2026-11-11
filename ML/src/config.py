"""Central configuration for the Stock Market ML Indicator pipeline.

All constants, paths, parameters, date ranges, fees, and assumptions
are defined here. No module should hard-code paths or constants.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

# =====================================================================
# 1. DIRECTORY AND PATH CONFIGURATION
# =====================================================================
# ML directory is the root for all ML-related operations
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
PROJECT_ROOT = ML_DIR.parent
PROJECT_NAME = "Stock-Market-ML-Predictor"

DATA_DIR = ML_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
DOCS_DIR = ML_DIR / "docs"
NOTEBOOKS_DIR = ML_DIR / "notebooks"
TESTS_DIR = ML_DIR / "tests"

# Specific canonical file paths (CSV is the primary storage format per Master Prompt v3)
RAW_DATA_PATH = DATA_RAW_DIR / "BTCUSDT_5m_raw.csv"
RAW_CSV_PATH = RAW_DATA_PATH
CLEAN_DATA_PATH = DATA_PROCESSED_DIR / "clean.csv"
FEATURES_PATH = DATA_PROCESSED_DIR / "features.csv"
RAW_PARQUET_PATH = DATA_RAW_DIR / "BTCUSDT_5m_raw.parquet"  # Legacy / optional fallback
CLEAN_PARQUET_PATH = DATA_PROCESSED_DIR / "clean.parquet"  # Legacy / optional fallback
EXPERIMENTS_CSV_PATH = RESULTS_DIR / "experiments.csv"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality.md"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_report.md"

# =====================================================================
# 2. MARKET DATA AND BINANCE API CONSTANTS
# =====================================================================
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
INTERVAL_MINUTES = 5
INTERVAL_MS = 5 * 60 * 1000  # 300,000 ms

# Binance Public Vision / Spot API endpoint
BINANCE_BASE_URL = "https://data-api.binance.vision/api/v3/klines"
DEFAULT_PAGE_LIMIT = 1000

# Canonical Binance raw candle columns returned by API
BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Standardized retained columns
OUTPUT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

# =====================================================================
# 3. FIXED HISTORICAL DATE RANGE AND SPLITS
# =====================================================================
# Fixed end date guarantees every re-download and experiment is 100% reproducible
START_DATE_STR = "2023-11-01 00:00:00+00:00"
END_DATE_STR = "2026-11-10 00:00:00+00:00"

START_DATE_UTC = pd.Timestamp(START_DATE_STR)
END_DATE_UTC = pd.Timestamp(END_DATE_STR)

# Total expected duration: 1,105 days (including leap year 2024 and 9 days extension to Nov 10, 2026)
TOTAL_EXPECTED_CANDLES = 318_240  # 315,648 + 9 days * 288 candles/day

# Chronological split ratios (approximate historical baseline)
TRAIN_RATIO = 220_953 / 318_240  # ~69.43%
VAL_RATIO = 47_347 / 318_240    # ~14.88%
TEST_RATIO = 49_940 / 318_240   # ~15.69%

# Fixed chronological split boundaries
# Train and Val are strictly frozen to preserve EDA and validation integrity;
# the extension to 2026-11-10 is absorbed entirely by the test split.
SPLIT_BOUNDARIES: Dict[str, Dict[str, Any]] = {
    "train": {
        "start": pd.Timestamp("2023-11-01 00:00:00+00:00"),
        "end": pd.Timestamp("2025-12-07 04:40:00+00:00"),
        "expected_candles": 220_953,
        "days": 767.2,
    },
    "val": {
        "start": pd.Timestamp("2025-12-07 04:45:00+00:00"),
        "end": pd.Timestamp("2026-05-20 14:15:00+00:00"),
        "expected_candles": 47_347,
        "days": 164.4,
    },
    "test": {
        "start": pd.Timestamp("2026-05-20 14:20:00+00:00"),
        "end": pd.Timestamp("2026-11-09 23:55:00+00:00"),
        "expected_candles": 49_940,
        "days": 173.4,
    },
}

# =====================================================================
# 4. TRANSACTION COSTS AND FRICTION ASSUMPTIONS
# =====================================================================
# Source: Binance VIP 0 Spot Fee Schedule [VERIFIED]
MAKER_FEE_RATE = 0.0010  # 0.1000%
TAKER_FEE_RATE = 0.0010  # 0.1000%
ROUNDTRIP_FEE_RATE = 2 * TAKER_FEE_RATE  # 0.2000% (entry taker + exit taker)

# Documented alternative: BNB Fee Deduction [VERIFIED alternative, not default]
BNB_DISCOUNT_ROUNDTRIP_FEE_RATE = 0.0015  # 0.1500% (with 25% BNB deduction: 0.075% per side)

# Friction assumptions [ASSUMPTION - modeling estimate for BTCUSDT liquid order book]
ESTIMATED_SLIPPAGE = 0.0002  # 0.02% (market order execution adverse impact)
ESTIMATED_HALF_SPREAD = 0.0001  # 0.01% (half-spread on liquid BTCUSDT order book)

# Total round-trip economic cost = fees + slippage + spread
TOTAL_ROUNDTRIP_COST = ROUNDTRIP_FEE_RATE + ESTIMATED_SLIPPAGE + ESTIMATED_HALF_SPREAD  # 0.0023 (0.23%)

# =====================================================================
# 5. LABELING, HORIZON, AND PURGING CONSTANTS
# =====================================================================
CANDIDATE_HORIZONS = [2, 3, 6, 12]  # Candidate horizons in candles {10m, 15m, 30m, 60m}
DEFAULT_HORIZON = 12  # Fallback horizon: 12 candles * 5 minutes = 60 minutes (1 hour)
HORIZON_CANDLES = DEFAULT_HORIZON
PURGE_GAP_CANDLES = DEFAULT_HORIZON  # Purge gap equal to label horizon to prevent overlap leakage

# One dead zone parameter dynamically set to total round-trip economic cost
LABEL_DEAD_ZONE = TOTAL_ROUNDTRIP_COST  # 0.0023 (0.23% forward return threshold)

# Target labels
LABEL_UP = 1
LABEL_DOWN = -1
LABEL_NEUTRAL = 0

# ---------------------------------------------------------------------
# Label Threshold Rules (10 rules total across 4 horizons)
# ---------------------------------------------------------------------
LABEL_RULES: Dict[str, Dict[str, Any]] = {
    "S1":  {"name": "S1",  "theta_up": 0.0023, "theta_down": 0.0023, "sub_cost": False, "type": "fixed",         "description": "Symmetric cost floor (c = 0.23%)"},
    "S2":  {"name": "S2",  "theta_up": 0.0030, "theta_down": 0.0030, "sub_cost": False, "type": "fixed",         "description": "Symmetric 0.30%"},
    "Q1":  {"name": "Q1",  "theta_up": None,   "theta_down": None,   "sub_cost": False, "type": "quantile",      "description": "Train 50th-pct of |fwd_ret| floored at c (identical to S1)"},
    "V1":  {"name": "V1",  "theta_up": None,   "theta_down": None,   "sub_cost": False, "type": "volatility",    "description": "Dynamic rolling vol 0.5-sigma floored at c"},
    "V2":  {"name": "V2",  "theta_up": None,   "theta_down": None,   "sub_cost": False, "type": "volatility",    "description": "Dynamic rolling vol 1.0-sigma floored at c"},
    "A1":  {"name": "A1",  "theta_up": 0.0023, "theta_down": 0.0030, "sub_cost": False, "type": "asymmetric",    "description": "Asymmetric: easier UP bar (c / 0.30%)"},
    "A2":  {"name": "A2",  "theta_up": 0.0030, "theta_down": 0.0023, "sub_cost": False, "type": "asymmetric",    "description": "Asymmetric: easier DOWN bar (0.30% / c)"},
    "F05": {"name": "F05", "theta_up": 0.0005, "theta_down": 0.0005, "sub_cost": True,  "type": "fixed_sub_cost", "description": "Research sub-cost: symmetric 0.05% (move below cost c)"},
    "F10": {"name": "F10", "theta_up": 0.0010, "theta_down": 0.0010, "sub_cost": True,  "type": "fixed_sub_cost", "description": "Research sub-cost: symmetric 0.10% (move below cost c)"},
    "F15": {"name": "F15", "theta_up": 0.0015, "theta_down": 0.0015, "sub_cost": True,  "type": "fixed_sub_cost", "description": "Research sub-cost: symmetric 0.15% (move below cost c)"},
}
ALL_RULES = list(LABEL_RULES.keys())
SUB_COST_RULES = [r for r, meta in LABEL_RULES.items() if meta["sub_cost"]]

F05_THETA = 0.0005  # 0.05%
F10_THETA = 0.0010  # 0.10%
F15_THETA = 0.0015  # 0.15%

# =====================================================================
# 6. FEATURE ENGINEERING CONSTANTS
# =====================================================================
LOOKBACK_CANDLES = 2018  # Minimum lookback needed for feature warm-up (7d HAR & seasonal: 2016 candles + 1 for return lag + 1 for 1h MTF completion = 2018 candles)
LIVE_BUFFER_CANDLES = 2880  # Recommended live streaming buffer (10 full days: 2880 candles)

# Features excluded specifically from linear/logistic models due to exact collinearity/identities
# Trees (LightGBM/XGBoost) retain all features.
LINEAR_MODEL_EXCLUDE = [
    "feat_mom_macd_hist",  # Exact linear identity: feat_mom_macd_hist = feat_mom_macd_line - feat_mom_macd_signal (causes singular covariance matrix in logistic regression)
]

# =====================================================================
# 7. MODELING AND REPRODUCIBILITY
# =====================================================================
RANDOM_SEED = 42

# Walk-forward cross validation settings
N_WALK_FORWARD_FOLDS = 4
WALK_FORWARD_EXPANDING = True

# Candidate confidence thresholds for live trade filtering
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
MIN_SIGNALS_PER_DAY = 15

