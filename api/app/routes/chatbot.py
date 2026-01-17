from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import ChatRequest
from app.services.chatbot import reply
from app.db.session import get_db
from app.db.crud import log_inference

router = APIRouter(prefix="/chat", tags=["chatbot"])

@router.post("")
def chat_api(payload: ChatRequest, db: Session = Depends(get_db)):
    ans = reply(payload.message)
    resp = {"reply": ans}
    log_inference(db, "chat", payload.model_dump(), resp)
    return resp
