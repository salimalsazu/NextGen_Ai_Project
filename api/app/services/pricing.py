import pandas as pd
from app.services.model_loader import get_pricing

def predict_price(base_price: float, demand: float, stock: int) -> float:
    model = get_pricing()
    X = pd.DataFrame([{"base_price": base_price, "demand": demand, "stock": stock}])
    y = model.predict(X)
    return float(y[0])
