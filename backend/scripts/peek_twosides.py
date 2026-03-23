import pandas as pd
import sys

file_path = r"c:\Drug\backend\data\TWOSIDES.csv.gz"

try:
    df = pd.read_csv(file_path, compression='gzip', nrows=5)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
