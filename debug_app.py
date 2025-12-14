
import pandas as pd
import os
from app import load_resources

print("Checking salaries.csv...")
if os.path.exists('salaries.csv'):
    df = pd.read_csv('salaries.csv', nrows=2)
    print("Columns:", df.columns.tolist())
else:
    print("salaries.csv not found!")

print("\nRunning load_resources()...")
try:
    load_resources()
    print("load_resources completed.")
except Exception as e:
    print(f"load_resources failed: {e}")
    import traceback
    traceback.print_exc()
