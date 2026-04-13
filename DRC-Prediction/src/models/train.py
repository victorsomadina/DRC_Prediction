import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from src.utils.helpers import DATA_PROCESSED_DIR, ROOT_DIR, get_logger

logger = get_logger(__name__)

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT_DIR / 'mlruns' / 'mlflow.db'}",
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "drc-cost-prediction")

CATEGORICAL_FEATURES = ["incidentType", "declarationType", "region", "season"]
NUMERIC_FEATURES = [
    "incident_duration_days",
    "declaration_year",
    "rolling_5yr_freq",
    "project_count",
    "avg_project_amount",
    "is_high_cost_type",
]
TARGET = "log_total_obligated"
TARGET_ORIGINAL = "total_obligated_amount"

def load_processed() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "processed_disasters.csv"
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded processed data: {df.shape}")
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate(model, X_test, y_test_log, y_test_original) -> dict:
    """Evaluate model on both log and original scale."""
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_log, y_pred_log)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_log(
    name: str,
    estimator,
    X_train,
    X_test,
    y_train_log,
    y_test_log,
    y_test_original,
    params: dict,
) -> tuple[Pipeline, dict]:
    """Train a pipeline, log to MLflow, return (pipeline, metrics)."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train_log)
        metrics = evaluate(pipeline, X_test, y_test_log, y_test_original)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        logger.info(
            f"{name} — R²: {metrics['r2']:.4f} | "
            f"RMSE: ${metrics['rmse']:,.0f} | MAE: ${metrics['mae']:,.0f}"
        )

    return pipeline, metrics


def run_training(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Full training pipeline. Returns dict with best model info.
    """
    df = load_processed()

    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, TARGET_ORIGINAL]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_log = df[TARGET]
    y_original = df[TARGET_ORIGINAL]

    X_train, X_test, y_train_log, y_test_log, _, y_test_original = train_test_split(
        X, y_log, y_original, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    lr, lr_metrics = train_and_log(
        name="LinearRegression",
        estimator=LinearRegression(),
        X_train=X_train, X_test=X_test,
        y_train_log=y_train_log, y_test_log=y_test_log,
        y_test_original=y_test_original,
        params={"model_type": "linear_regression"},
    )
    results["LinearRegression"] = {"pipeline": lr, "metrics": lr_metrics}

    rf_params = {
        "model_type": "random_forest",
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": random_state,
    }
    rf, rf_metrics = train_and_log(
        name="RandomForest",
        estimator=RandomForestRegressor(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            random_state=random_state,
            n_jobs=-1,
        ),
        X_train=X_train, X_test=X_test,
        y_train_log=y_train_log, y_test_log=y_test_log,
        y_test_original=y_test_original,
        params=rf_params,
    )
    results["RandomForest"] = {"pipeline": rf, "metrics": rf_metrics}

    xgb_params = {
        "model_type": "xgboost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
    }
    xgb, xgb_metrics = train_and_log(
        name="XGBoost",
        estimator=XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        X_train=X_train, X_test=X_test,
        y_train_log=y_train_log, y_test_log=y_test_log,
        y_test_original=y_test_original,
        params=xgb_params,
    )
    results["XGBoost"] = {"pipeline": xgb, "metrics": xgb_metrics}

    best_name = max(results, key=lambda k: results[k]["metrics"]["r2"])
    best_pipeline = results[best_name]["pipeline"]
    best_metrics = results[best_name]["metrics"]

    logger.info(f"\nBest model: {best_name} (R²={best_metrics['r2']:.4f})")

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "metrics": best_metrics,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "target": TARGET,
        },
        model_path,
    )
    logger.info(f"Best model saved → {model_path}")

    return {
        "best_model": best_name,
        "metrics": best_metrics,
        "all_results": {k: v["metrics"] for k, v in results.items()},
    }


if __name__ == "__main__":
    summary = run_training()
    print("\n=== Training Summary ===")
    for model, metrics in summary["all_results"].items():
        print(f"{model}: R²={metrics['r2']:.4f} | RMSE=${metrics['rmse']:,.0f}")
    print(f"\nBest: {summary['best_model']}")
