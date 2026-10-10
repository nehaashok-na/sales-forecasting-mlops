from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sales.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET = "Product_Store_Sales_Total"
IDENTIFIER_COLUMNS = ["Product_Id", "Store_Id"]


def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df.loc[df[TARGET] >= 0].copy()

    # Trim whitespace in string features without changing their meaning.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    # Normalize a shorthand category found in the supplied CSV.
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    df = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    if df.empty:
        raise ValueError("No usable rows remain after target cleaning")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    train_path = ARTIFACTS_DIR / "train.csv"
    test_path = ARTIFACTS_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("DATA PREPARATION PASSED")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(f"Training data saved: {train_path}")
    print(f"Testing data saved: {test_path}")
    print(f"Feature columns: {X.columns.tolist()}")


if __name__ == "__main__":
    prepare_data()
