import numpy as np
import pandas as pd

def make_reco_data(n_users=1000, n_items=500, interactions=20000, seed=42):
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(1, n_users + 1, size=interactions)
    item_ids = rng.integers(1, n_items + 1, size=interactions)
    labels = rng.integers(0, 2, size=interactions)
    return pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "label": labels})

def make_pricing_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    base_price = rng.normal(50, 10, n).clip(5, 200)
    demand = rng.normal(100, 25, n).clip(1, 300)
    stock = rng.integers(0, 500, n)
    optimal_price = base_price * (1 + (demand - 100) / 400) * (1 - (stock - 200) / 2000)
    return pd.DataFrame({"base_price": base_price, "demand": demand, "stock": stock, "optimal_price": optimal_price})

def make_inventory_data(n=365*3, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = 20*np.sin(2*np.pi*t/365)
    trend = 0.01*t
    noise = rng.normal(0, 5, n)
    demand = (100 + season + trend + noise).clip(0, None)
    return pd.DataFrame({"t": t, "demand": demand})
