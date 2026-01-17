import os
import numpy as np
import pandas as pd

def _load_csv_if_provided() -> pd.DataFrame | None:
    path = os.getenv("RETAIL_CSV", "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"RETAIL_CSV path not found: {path}")
    return pd.read_csv(path)

def make_reco_data(n_users=1000, n_items=500, interactions=20000, seed=42):
    df = _load_csv_if_provided()
    if df is not None:
        needed = ["user_id", "item_id", "label"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"RETAIL_CSV missing columns for recommender: {missing}")
        out = df[needed].dropna()
        out["user_id"] = out["user_id"].astype(int)
        out["item_id"] = out["item_id"].astype(int)
        out["label"] = out["label"].astype(int)
        return out

    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, size=interactions),
        "item_id": rng.integers(1, n_items + 1, size=interactions),
        "label": rng.integers(0, 2, size=interactions),
    })

def make_pricing_data(n=5000, seed=42):
    df = _load_csv_if_provided()
    if df is not None:
        needed = ["base_price", "demand", "stock", "optimal_price"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"RETAIL_CSV missing columns for pricing: {missing}")
        return df[needed].dropna()

    rng = np.random.default_rng(seed)
    base_price = rng.normal(50, 10, n).clip(5, 200)
    demand = rng.normal(100, 25, n).clip(1, 300)
    stock = rng.integers(0, 500, n)
    optimal_price = base_price * (1 + (demand - 100) / 400) * (1 - (stock - 200) / 2000)
    return pd.DataFrame({"base_price": base_price, "demand": demand, "stock": stock, "optimal_price": optimal_price})

def make_inventory_data(n=365*3, seed=42):
    df = _load_csv_if_provided()
    if df is not None:
        needed = ["t", "demand"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"RETAIL_CSV missing columns for inventory: {missing}")
        out = df[needed].dropna()
        out["t"] = out["t"].astype(int)
        return out.sort_values("t")

    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = 20*np.sin(2*np.pi*t/365)
    trend = 0.01*t
    noise = rng.normal(0, 5, n)
    demand = (100 + season + trend + noise).clip(0, None)
    return pd.DataFrame({"t": t, "demand": demand})
