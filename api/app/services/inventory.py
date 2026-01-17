import pandas as pd
from app.services.model_loader import get_inventory

def forecast_demand(t: int) -> float:
    model = get_inventory()
    X = pd.DataFrame([{"t": t}])
    y = model.predict(X)
    return float(y[0])
