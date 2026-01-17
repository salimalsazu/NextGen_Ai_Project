from app.core.config import settings
from app.ml.versioning import load_current_model


def load_from_local(model_name: str):
    return load_current_model(settings.MODELS_DIR, model_name)
