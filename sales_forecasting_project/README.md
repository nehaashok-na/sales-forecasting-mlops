# Sales Forecasting MLOps Project

This project predicts product-store sales revenue from product and store attributes. It automates data validation, preparation, Random Forest tuning, MLflow experiment logging, model evaluation, model commitment, and deployment through Streamlit Community Cloud.

## Project files
- `data/sales.csv`: source dataset (add this file before running the workflow)
- `model_building/data_register.py`: schema checks and data summary
- `model_building/prep.py`: cleaning and train/test split generation
- `model_building/train.py`: preprocessing pipeline, hyperparameter tuning, metrics and MLflow tracking
- `model_building/evaluate.py`: test-set evaluation of the saved model
- `deployment/app.py`: Streamlit prediction interface
- `deployment/requirements.txt`: deployment dependencies
- `.github/workflows/pipeline.yml`: automated CI/CD workflow

## Run locally
From the repository root:

```bash
python -m pip install -r sales_forecasting_project/requirements.txt
python sales_forecasting_project/model_building/data_register.py
python sales_forecasting_project/model_building/prep.py
python sales_forecasting_project/model_building/train.py
python sales_forecasting_project/model_building/evaluate.py
streamlit run sales_forecasting_project/deployment/app.py
```

## Scope and limitation
The provided data dictionary has no date column, so this is a product-store revenue regression baseline, not true weekly, monthly, quarterly, or annual time-series forecasting. Date-indexed historical sales and future pipeline information are needed for that extension.

Add the actual GitHub Actions run URL, deployed Streamlit URL, test metrics, and screenshots before submission.
