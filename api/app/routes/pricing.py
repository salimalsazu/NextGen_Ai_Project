from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import PricingRequest
from app.services.pricing import predict_price
from app.db.session import get_db
from app.db.crud import log_inference

router = APIRouter(prefix="/pricing", tags=["pricing"])

@router.post("")
def pricing_api(payload: PricingRequest, db: Session = Depends(get_db)):
    price = predict_price(payload.base_price, payload.demand, payload.stock)
    resp = {"optimal_price": price}
    log_inference(db, "pricing", payload.model_dump(), resp)
    return resp
