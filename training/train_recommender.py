import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

from training.data.sample_generator import make_reco_data
from training.models import PopularityRecommender
from training.versioning import save_versioned_model


class PopularityPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, top_items):
        self.top_items = top_items

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        k = int(model_input.get("k", pd.Series([10])).iloc[0])
        return [self.top_items[:k] for _ in range(len(model_input))]


def load_reco_df() -> pd.DataFrame:
    """
    One CSV -> recommender dataset
    Required columns in unified CSV:
      - user_id, item_id
      - purchased OR clicked (label তৈরি হবে)
    """
    csv_path = os.getenv("RETAIL_CSV", "training/data/retail_demo.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "purchased" in df.columns:
            df["label"] = df["purchased"].astype(int)
        elif "clicked" in df.columns:
            df["label"] = df["clicked"].astype(int)
        else:
            raise ValueError("Unified CSV must contain 'purchased' or 'clicked' column.")

        needed = {"user_id", "item_id", "label"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in unified CSV for recommender: {missing}")

        return df[["user_id", "item_id", "label"]]

    # fallback
    return make_reco_data()


def main():
    # In docker-compose, MLflow is reachable via the service name `mlflow`.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NextGenRetail")
    model_name = os.getenv("MODEL_RECO_NAME", "recommender")
    models_dir = os.getenv("MODELS_DIR", "models")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    df = load_reco_df()

    top_items = (
        df[df["label"] == 1]
        .groupby("item_id")["label"].sum()
        .sort_values(ascending=False)
        .head(50)
        .index.tolist()
    )

    with mlflow.start_run(run_name="train_recommender"):
        mlflow.log_param("model_type", "popularity")
        mlflow.log_param("top_items_count", len(top_items))
        mlflow.log_param("data_source", os.getenv("RETAIL_CSV", "training/data/retail_demo.csv"))

        pyfunc_model = PopularityPyFunc(top_items)
        artifact_path = "recommender_model"
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=pyfunc_model)

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

        local_model = PopularityRecommender(top_items=top_items)
        save_versioned_model(
            models_root=models_dir,
            model_group=model_name,
            model_obj=local_model,
            meta={"model_type": "popularity", "k_default": 10, "top_items_count": len(top_items)},
        )


if __name__ == "__main__":
    main()
