from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
TARGET = "Product_Store_Sales_Total"


def train_and_evaluate():
    train_path = ARTIFACTS_DIR / "train.csv"
    test_path = ARTIFACTS_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing train/test splits. Run prep.py or download the workflow artifact first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
    ])
    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 15],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment("sales-revenue-regression")

    with mlflow.start_run() as run:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r2": float(r2_score(y_test, predictions)),
        }

        mlflow.log_param("algorithm", "RandomForestRegressor")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        cv_results_path = ARTIFACTS_DIR / "cv_results.csv"
        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        model_path = DEPLOYMENT_DIR / "best_model.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(str(cv_results_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print("Best parameters:", search.best_params_)
        print("Held-out test metrics:", json.dumps(metrics, indent=2))
        print("Model saved to:", model_path)
        print("MLflow run ID:", run.info.run_id)


if __name__ == "__main__":
    train_and_evaluate()
