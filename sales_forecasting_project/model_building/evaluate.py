from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "Product_Store_Sales_Total"


def evaluate_model():
    test_path = PROJECT_ROOT / "artifacts" / "test.csv"
    model_path = PROJECT_ROOT / "deployment" / "best_model.joblib"
    test = pd.read_csv(test_path)
    model = joblib.load(model_path)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    output_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate_model()
