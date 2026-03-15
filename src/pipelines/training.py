import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from prefect import flow, task
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from core.config import load_config
from core.features import build_features
from core.loader import load_raw_data

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVAL_METRIC = "logloss"

# MLflow configuration sourced from environment; validated at module load time
# to surface misconfiguration before any training work starts.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "predmaint")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "predmaint-classifier")

_model_path_env = os.environ.get("MODEL_PATH")
MODEL_PATH = (
    Path(_model_path_env) if _model_path_env else _PROJECT_ROOT / "models" / "model.pkl"
)

for _var_name, _var_value in (
    ("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI),
    ("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME),
    ("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME),
):
    if not _var_value.strip():
        raise ValueError(f"Environment variable '{_var_name}' must not be empty.")


TrainTestSplit = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def _configure_mlflow() -> None:
    """Point the MLflow client at the configured tracking server and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@task(name="load-config")
def load_config_task() -> dict:
    return load_config()


@task(name="ingest-data")
def ingest_data(config: dict) -> pd.DataFrame:
    return load_raw_data(Path(config["data"]["raw_path"]))


@task(name="build-features")
def build_features_task(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(df)


@task(name="split-data")
def split_data(X: pd.DataFrame, y: pd.Series, config: dict) -> TrainTestSplit:
    # train_test_split returns list[Any]; cast to the concrete tuple type.
    result: TrainTestSplit = tuple(  # type: ignore[assignment]
        train_test_split(
            X,
            y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y,
        )
    )
    return result


@task(name="train-model")
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> XGBClassifier:
    eval_metric = config["model"].get("eval_metric", DEFAULT_EVAL_METRIC)
    model = XGBClassifier(**config["model"]["params"], eval_metric=eval_metric)
    model.fit(X_train, y_train)
    return model


@task(name="evaluate-model")
def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)
    return metrics


@task(name="log-to-mlflow")
def log_to_mlflow(model: XGBClassifier, metrics: dict) -> None:
    _configure_mlflow()
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model, artifact_path="model", registered_model_name=MLFLOW_MODEL_NAME
        )


@task(name="save-model")
def save_model(model: XGBClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved: %s", MODEL_PATH)


@flow(name="training-pipeline")
def training_pipeline() -> None:
    config = load_config_task()
    df = ingest_data(config)
    X, y = build_features_task(df)

    # Prefect wraps task return values in PrefectFuture when running with a
    # server; the type annotations reflect the resolved values, not the futures.
    X_train, X_test, y_train, y_test = split_data(X, y, config)  # type: ignore[misc]
    model: XGBClassifier = train_model(X_train, y_train, config)  # type: ignore[assignment]
    metrics: dict = evaluate_model(model, X_test, y_test)  # type: ignore[assignment]
    log_to_mlflow(model, metrics)
    save_model(model)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    training_pipeline()
