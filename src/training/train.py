import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from prefect import flow, task
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data.loader import load_raw_data
from data.transformer import build_features
from monitoring.drift import detect_drift

CONFIG_PATH = Path("configs/training.yaml")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "predmaint")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "predmaint-classifier")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
RETRAINING_TRIGGER = os.getenv("RETRAINING_TRIGGER", "true").lower() == "true"


@task(name="load-config")
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@task(name="ingest-data")
def ingest_data(config: dict) -> pd.DataFrame:
    return load_raw_data(Path(config["data"]["raw_path"]))


@task(name="build-features")
def build_features_task(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(df)


@task(name="split-data")
def split_data(
    X: pd.DataFrame, y: pd.Series, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


@task(name="train-model")
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> XGBClassifier:
    model = XGBClassifier(**config["model"]["params"], eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


@task(name="evaluate-model")
def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


@task(name="save-model")
def save_model(model, metrics: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model, artifact_path="model", registered_model_name=MLFLOW_MODEL_NAME
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


@flow(name="training-pipeline")
def training_pipeline() -> None:
    config = load_config()
    df = ingest_data(config)
    X, y = build_features_task(df)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    model: XGBClassifier = train_model(X_train, y_train, config)  # type: ignore[assignment]
    metrics: dict = evaluate_model(model, X_test, y_test)
    save_model(model, metrics)

    print("Pipeline complete.")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


@task(name="check-drift")
def check_drift(config: dict) -> bool:
    """Check if production data has drifted from reference."""
    df = load_raw_data(Path(config["data"]["raw_path"]))
    X, y = build_features(df)
    production_sample = X.iloc[7000:]
    result = detect_drift(production_sample)
    print(f"Drift detected: {result['drift_detected']}")
    return result["drift_detected"]


@flow(name="monitoring-pipeline")
def monitoring_pipeline() -> None:
    """Run drift detection and trigger retraining if needed."""
    config = load_config()
    drift = check_drift(config)

    if drift and RETRAINING_TRIGGER:
        print("Drift detected — triggering retraining...")
        training_pipeline()
    elif drift:
        print("Drift detected — retraining disabled by RETRAINING_TRIGGER.")
    else:
        print("No drift detected — model is stable.")


if __name__ == "__main__":
    monitoring_pipeline()
