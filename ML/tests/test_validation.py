
import sys
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.validation import validate_dataframe, print_validation_report

data_path = ML_DIR / "data" / "raw" / "BTCUSDT_5m_test01.csv"
df = pd.read_csv(data_path)

report = validate_dataframe(df, interval="5min")

print_validation_report(report)
