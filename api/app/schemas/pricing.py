from pydantic import BaseModel

class PricingRequest(BaseModel):
    product_id: int
    store_id: int
    day_of_week: int
    competitor_price: float
    base_price: float
    demand: float
    stock: int
