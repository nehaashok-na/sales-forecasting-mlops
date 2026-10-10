from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sales.csv"
EXPECTED_COLUMNS = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Store_Sales_Total",
]


def validate_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add sales.csv to the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("DATA REGISTRATION / VALIDATION PASSED")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values by column:")
    print(df[EXPECTED_COLUMNS].isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")
    print("\nNumeric summary:")
    print(df.describe(include="all").transpose().to_string())


if __name__ == "__main__":
    validate_dataset()
