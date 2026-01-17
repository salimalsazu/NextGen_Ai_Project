import pandas as pd
from app.services.model_loader import get_reco

def recommend(user_id: int, k: int):
    model = get_reco()

    # 1) Try the local model signature: predict(user_id=..., k=...)
    try:
        return model.predict(user_id=user_id, k=k)
    except TypeError:
        pass  # likely MLflow pyfunc (or df-based model)

    # 2) If there's a batch_predict method, use it
    df = pd.DataFrame([{"user_id": user_id, "k": k}])
    if hasattr(model, "batch_predict"):
        out = model.batch_predict(df)
        return out[0] if hasattr(out, "__len__") else out

    # 3) MLflow pyfunc expects a single input (DataFrame)
    out = model.predict(df)
    return out[0] if hasattr(out, "__len__") else out
