import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from mlflow.models.signature import infer_signature


from training.versioning import save_versioned_model


def load_pricing_df() -> pd.DataFrame:
    """
    One CSV -> pricing dataset
    Required columns: price, stock, demand
    """
    csv_path = os.getenv("RETAIL_CSV", "training/data/retail_demo.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Unified CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    needed = {"price", "stock", "demand"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in unified CSV for pricing: {missing}")

    return df


def main():
    # In docker-compose, MLflow is reachable via the service name `mlflow`.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NextGenRetail")
    model_name = os.getenv("MODEL_PRICING_NAME", "pricing")
    models_dir = os.getenv("MODELS_DIR", "models")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    df = load_pricing_df()
    X = df[["stock", "demand"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    with mlflow.start_run(run_name="train_pricing"):
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("features", "stock,demand")
        mlflow.log_param("data_source", os.getenv("RETAIL_CSV", "training/data/retail_demo.csv"))

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_metric("mae", float(mae))
        signature = infer_signature(X_train, model.predict(X_train))
        artifact_path = "pricing_model"

        mlflow.sklearn.log_model(model, artifact_path=artifact_path, signature=signature)

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
            meta={"model_type": "linear_regression", "metric_mae": float(mae)},
        )


if __name__ == "__main__":
    main()
