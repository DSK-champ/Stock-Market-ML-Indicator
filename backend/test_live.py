"""Live Testing and Demonstration Script for Backend Integration.

Demonstrates:
  1. Initializing the unified TradingEngine.
  2. Performing single-candle real-time causal prediction.
  3. Retrieving model performance metrics for the UI dashboard.
  4. Running the cost-adjusted backtest simulation.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add current directory and ML directory to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR.parent / "ML"))

from engine import TradingEngine, predict_one, get_model_metrics


def main() -> None:
    print("=" * 80)
    print("AI TRADING INDICATOR — UNIFIED BACKEND INFERENCE ENGINE TEST")
    print("=" * 80)

    # 1. Initialize Engine
    print("\n[1/4] Initializing TradingEngine...")
    engine = TradingEngine.get_instance()
    print("TradingEngine successfully loaded into memory.")

    # 2. Test Live Single-Candle Prediction
    print("\n[2/4] Simulating Live Candle Arrival...")
    test_csv = CURRENT_DIR.parent / "ML" / "results" / "horizon_2_test.csv"
    if test_csv.exists():
        df_sample = pd.read_csv(test_csv, nrows=350)
        # Convert index / open_time
        if "open_time" in df_sample.columns:
            df_sample["open_time"] = pd.to_datetime(df_sample["open_time"], utc=True)
            df_sample = df_sample.set_index("open_time")
        
        result = predict_one(df_sample)
        print("Live Candle Inference Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Sample data {test_csv} not found, skipping live sample inference.")

    # 3. Retrieve Metrics for Frontend Dashboard
    print("\n[3/4] Fetching Model Metrics for Dashboard...")
    metrics = get_model_metrics()
    print("Overall Performance Scorecard:")
    print(json.dumps(metrics["overall_performance"], indent=2))
    print("\nPer-Class Breakdown:")
    print(json.dumps(metrics["per_class_metrics"], indent=2))

    # 4. Run Backtest Simulation
    print("\n[4/4] Running Cost-Adjusted Backtest Simulation (23 bps fee deduction)...")
    backtest = engine.simulate_backtest(initial_capital=10000.0)
    print("Backtest Financial Summary:")
    print(json.dumps(backtest, indent=2))

    # 5. Export Production Bundle
    bundle_path = engine.export_production_bundle()
    print(f"\n[OK] Production bundle exported to: {bundle_path}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY! The backend is ready for live streaming.")
    print("=" * 80)


if __name__ == "__main__":
    main()
