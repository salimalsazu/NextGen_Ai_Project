from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import InventoryRequest
from app.services.inventory import forecast_demand
from app.db.session import get_db
from app.db.crud import log_inference

router = APIRouter(prefix="/inventory", tags=["inventory"])

@router.post("")
def inventory_api(payload: InventoryRequest, db: Session = Depends(get_db)):
    demand = forecast_demand(payload.t)
    resp = {"t": payload.t, "forecast_demand": demand}
    log_inference(db, "inventory", payload.model_dump(), resp)
    return resp
