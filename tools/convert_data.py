import pandas as pd
import os

csv_path = 'data/life-exp-data.csv'
parquet_path = 'data/life-exp-data.parquet'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path)
    print(f"Successfully converted {csv_path} to {parquet_path}")
else:
    print(f"Error: {csv_path} not found.")
