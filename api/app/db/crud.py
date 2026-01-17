from sqlalchemy.orm import Session
from .models import InferenceLog

def log_inference(db: Session, endpoint: str, request_json: dict, response_json: dict):
    row = InferenceLog(endpoint=endpoint, request_json=request_json, response_json=response_json)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
