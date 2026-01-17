import mlflow
from .config import settings

def load_from_mlflow(model_name: str, stage: str):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)
