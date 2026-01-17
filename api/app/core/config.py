from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "NextGenRetail"

    MODEL_RECO_NAME: str = "recommender"
    MODEL_PRICING_NAME: str = "pricing"
    MODEL_INVENTORY_NAME: str = "inventory"

    MODEL_STAGE: str = "Production"
    MODEL_SOURCE: str = "mlflow"  # "mlflow" or "local"
    MODELS_DIR: str = "models"

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
