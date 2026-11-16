"""End-to-end pipeline runner and CLI evaluator for the ML Indicator.

Allows any mentor or reviewer to reproduce key experimental results and
held-out test set evaluations with a single command without running notebooks.

Usage:
    python -m src.main                    # Runs full verification & prints all scorecards
    python -m src.main --oof              # Prints out-of-fold validation metrics
    python -m src.main --test             # Evaluates held-out test partition
    python -m src.main --benchmarks       # Prints comparative model benchmarks table
"""

import argparse
import sys
from pathlib import Path

# Ensure ML package is on path
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.utils import setup_logger, seed_everything
from src import config
from src.train import (
    evaluate_oof_performance,
    evaluate_held_out_test,
    print_comparative_benchmark_table,
)

logger = setup_logger("main_pipeline")


def main() -> None:
    """Execute evaluation and reproduction pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-End Evaluation & Verification CLI for BTCUSDT ML Indicator"
    )
    parser.add_argument(
        "--oof",
        action="store_true",
        help="Evaluate out-of-fold walk-forward validation scorecard (110,433 candles)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate held-out test partition (49,938 candles)",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Print 5-column model comparison table (IIT Patna Slide 13 format)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Run full end-to-end evaluation suite (default)",
    )

    args = parser.parse_args()
    seed_everything(config.RANDOM_SEED)

    print("\n" + "=" * 80)
    print("  IIT PATNA — MACHINE LEARNING LABORATORY PROJECT EVALUATION RUNNER")
    print("  Project: Cost-Sensitive Machine Learning Framework for Bitcoin Direction Forecasting")
    print("=" * 80)

    # Determine what to run
    run_all = args.all and not (args.oof or args.test or args.benchmarks)

    if args.oof or run_all:
        evaluate_oof_performance()

    if args.test or run_all:
        evaluate_held_out_test()

    if args.benchmarks or run_all:
        print_comparative_benchmark_table()

    logger.info("Evaluation and verification completed successfully.")


if __name__ == "__main__":
    main()
