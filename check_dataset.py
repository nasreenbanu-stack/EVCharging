
from pathlib import Path
import pandas as pd

# Locate the dataset inside the data folder
file_path = Path("data/charging_records.csv")

# Check whether the file exists
if not file_path.exists():
    print("ERROR: Dataset file not found.")
    print("Please check the data folder and filename.")

else:
    # Read the CSV dataset
    df = pd.read_csv(file_path)

    print("\n1. DATASET LOADED SUCCESSFULLY")
    print("-" * 40)

    print("\n2. DATASET SIZE")
    print("Rows and columns:", df.shape)

    print("\n3. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n4. FIRST FIVE RECORDS")
    print(df.head())

    print("\n5. DATA TYPES AND NON-MISSING VALUES")
    df.info()

    print("\n6. MISSING VALUES PER COLUMN")
    print(df.isnull().sum())

    print("\n7. EXACT DUPLICATE ROWS")
    print(df.duplicated().sum())
