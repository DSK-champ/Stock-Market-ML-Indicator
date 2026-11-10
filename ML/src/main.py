"""End-to-end pipeline runner for the ML indicator.

Runs offline extraction, cleaning, feature engineering, model training,
calibration, and evaluation.
"""

import sys
from pathlib import Path

# Ensure ML package is on path
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.utils import setup_logger, seed_everything
from src import config

logger = setup_logger("main_pipeline")


def main() -> None:
    """Execute offline pipeline end-to-end."""
    seed_everything(config.RANDOM_SEED)
    logger.info("Initializing ML pipeline with seed %d", config.RANDOM_SEED)
    logger.info("Configured symbol: %s, timeframe: %s", config.SYMBOL, config.TIMEFRAME)
    logger.info("Start date: %s, End date: %s", config.START_DATE_STR, config.END_DATE_STR)
    logger.info("Pipeline skeleton ready.")


if __name__ == "__main__":
    main()

