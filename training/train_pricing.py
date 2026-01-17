import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from training.data.sample_generator import make_pricing_data
from training.evaluate import rmse
from training.versioning import save_versioned_model

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NextGenRetail")
    model_name = os.getenv("MODEL_PRICING_NAME", "pricing")
    models_dir = os.getenv("MODELS_DIR", "models")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    df = make_pricing_data()
    X = df[["base_price", "demand", "stock"]]
    y = df["optimal_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    with mlflow.start_run(run_name="train_pricing"):
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_estimators", 200)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = rmse(y_test, preds)
        mlflow.log_metric("rmse", metric)

        artifact_path = "pricing_model"
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
            meta={"model_type": "random_forest", "rmse": metric},
        )

if __name__ == "__main__":
    main()
