from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
from src.extraction import fetch_historical_data, save_raw_data

df_test = fetch_historical_data(symbol="BTCUSDT",interval="5m",days=1,)

print("Dataset shape:", df_test.shape)
print(df_test.head())

output_path = ML_DIR/"data"/"raw"/"BTCUSDT_5m_test01.csv"
save_raw_data(df_test, output_path=output_path,)

print("Test completed successfully!")
