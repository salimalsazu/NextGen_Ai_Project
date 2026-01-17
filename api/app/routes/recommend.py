from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import RecommendRequest
from app.services.recommender import recommend
from app.db.session import get_db
from app.db.crud import log_inference

router = APIRouter(prefix="/recommend", tags=["recommend"])

@router.post("")
def recommend_api(payload: RecommendRequest, db: Session = Depends(get_db)):
    items = recommend(payload.user_id, payload.k)
    resp = {"user_id": payload.user_id, "recommended_items": items}
    log_inference(db, "recommend", payload.model_dump(), resp)
    return resp
