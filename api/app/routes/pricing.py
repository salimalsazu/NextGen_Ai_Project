from fastapi import APIRouter
from app.db.session import get_db
from app.db.crud import log_inference
from sqlalchemy.orm import Session
from fastapi import Depends

# Import your Pricing schema from the real module file
# ✅ change "pricing" if your file name is different (example: pricing_schema)
from app.schemas.pricing import PricingRequest

from app.services.pricing import predict_price

router = APIRouter()


@router.post("/pricing")
def pricing_api(payload: PricingRequest, db: Session = Depends(get_db)):
    price = predict_price(
        product_id=payload.product_id,
        store_id=payload.store_id,
        day_of_week=payload.day_of_week,
        competitor_price=payload.competitor_price,
        base_price=payload.base_price,
        demand=payload.demand,
        stock=payload.stock,
    )

    resp = {"predicted_price": price}

    # log inference (optional, but keep it if you have db logging)
    try:
        log_inference(db, "pricing", payload.model_dump(), resp)
    except Exception:
        pass

    return resp
