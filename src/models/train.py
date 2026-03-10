from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CONFIG_PATH = Path("configs/training.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train(config: dict) -> None:
    X, y = load_features(Path(config["data"]["processed_path"]))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    model_params = config["model"]["params"]

    mlflow.set_experiment("predmaint-baseline")

    with mlflow.start_run():
        mlflow.log_params(model_params)

        model = XGBClassifier(**model_params, eval_metric="logloss")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(
            model,
            name="model",
            pip_requirements=[
                "xgboost>=2.1.0",
                "scikit-learn>=1.5.0",
                "pandas>=2.2.0,<3",
                "pyarrow>=14.0.0,<24",
            ],
        )

        print("Training complete.")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save model to disk for API serving
        model_path = Path("models/model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    config = load_config()
    train(config)
