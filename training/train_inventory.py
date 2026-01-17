import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from training.versioning import save_versioned_model


def load_inventory_df() -> pd.DataFrame:
    """
    One CSV -> inventory dataset
    Required columns: timestamp, demand
    """
    csv_path = os.getenv("RETAIL_CSV", "training/data/retail_demo.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Unified CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    needed = {"timestamp", "demand"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in unified CSV for inventory: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    return df


def main():
    # In docker-compose, MLflow is reachable via the service name `mlflow`.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NextGenRetail")
    model_name = os.getenv("MODEL_INVENTORY_NAME", "inventory")
    models_dir = os.getenv("MODELS_DIR", "models")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    df = load_inventory_df()

    # simple time index feature
    df = df.reset_index(drop=True)
    X = df.index.values.reshape(-1, 1)
    y = df["demand"].values

    model = LinearRegression()

    with mlflow.start_run(run_name="train_inventory"):
        mlflow.log_param("model_type", "linear_regression_time_index")
        mlflow.log_param("data_source", os.getenv("RETAIL_CSV", "training/data/retail_demo.csv"))

        model.fit(X, y)
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)

        mlflow.log_metric("mae", float(mae))

        artifact_path = "inventory_model"
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=os.getenv("MODEL_STAGE", "Production"),
            archive_existing_versions=True,
        )

        save_versioned_model(
            models_root=models_dir,
            model_group=model_name,
            model_obj=model,
            meta={"model_type": "linear_regression_time_index", "metric_mae": float(mae)},
        )


if __name__ == "__main__":
    main()
