from app.core.config import settings
from app.core.mlflow_client import load_from_mlflow
from app.core.local_model_loader import load_from_local

_reco = None
_pricing = None
_inventory = None

def get_reco():
    global _reco
    if _reco is None:
        _reco = load_from_local(settings.MODEL_RECO_NAME) if settings.MODEL_SOURCE.lower() == "local" else load_from_mlflow(settings.MODEL_RECO_NAME, settings.MODEL_STAGE)
    return _reco

def get_pricing():
    global _pricing
    if _pricing is None:
        _pricing = load_from_local(settings.MODEL_PRICING_NAME) if settings.MODEL_SOURCE.lower() == "local" else load_from_mlflow(settings.MODEL_PRICING_NAME, settings.MODEL_STAGE)
    return _pricing

def get_inventory():
    global _inventory
    if _inventory is None:
        _inventory = load_from_local(settings.MODEL_INVENTORY_NAME) if settings.MODEL_SOURCE.lower() == "local" else load_from_mlflow(settings.MODEL_INVENTORY_NAME, settings.MODEL_STAGE)
    return _inventory
