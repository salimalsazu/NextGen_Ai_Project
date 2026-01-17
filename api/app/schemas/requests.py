from pydantic import BaseModel, Field
from typing import Optional

class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=10, ge=1, le=50)

class PricingRequest(BaseModel):
    base_price: float
    demand: float
    stock: int

class InventoryRequest(BaseModel):
    t: int

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
