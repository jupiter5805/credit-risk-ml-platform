from pathlib import Path


DATA_FILE = Path("data/raw/credit_applications.csv")
MODEL_FILE = Path("artifacts/credit_risk_model.joblib")
METRICS_FILE = Path("artifacts/model_metrics.json")

APPROVE_THRESHOLD = 0.20
REVIEW_THRESHOLD = 0.40
