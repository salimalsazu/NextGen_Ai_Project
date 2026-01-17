import pandas as pd
from app.services.model_loader import get_reco

def recommend(user_id: int, k: int):
    model = get_reco()
    # local PopularityRecommender
    if hasattr(model, "predict") and not hasattr(model, "batch_predict"):
        return model.predict(user_id=user_id, k=k)
    if hasattr(model, "batch_predict"):
        df = pd.DataFrame([{"user_id": user_id, "k": k}])
        return model.batch_predict(df)[0]
    # mlflow pyfunc
    df = pd.DataFrame([{"user_id": user_id, "k": k}])
    preds = model.predict(df)
    return preds[0]
