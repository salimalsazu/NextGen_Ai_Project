# api/app/services/pricing.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from app.services.model_loader import get_pricing


def _unwrap_sklearn_estimator(model: Any) -> Any:
    # If it's already an sklearn-like estimator
    if hasattr(model, "fit") and hasattr(model, "predict") and hasattr(model, "get_params"):
        return model

    impl = getattr(model, "_model_impl", None)
    if impl is None:
        return None

    # Common MLflow wrappers
    sk_model = getattr(impl, "sklearn_model", None)
    if sk_model is not None:
        return sk_model

    inner = getattr(impl, "model", None)
    if inner is not None:
        return inner

    return None


def _get_signature_columns(model: Any) -> Optional[List[str]]:
    meta = getattr(model, "metadata", None)
    if meta is None:
        return None

    get_input_schema = getattr(meta, "get_input_schema", None)
    if callable(get_input_schema):
        try:
            schema = get_input_schema()
            inputs = getattr(schema, "inputs", None)
            if inputs:
                cols: List[str] = []
                for c in inputs:
                    name = getattr(c, "name", None)
                    if name:
                        cols.append(str(name))
                return cols or None
        except Exception:
            pass

    sig = getattr(meta, "signature", None)
    if sig is not None:
        inputs = getattr(sig, "inputs", None)
        schema_inputs = getattr(inputs, "inputs", None) if inputs is not None else None
        if schema_inputs:
            cols: List[str] = []
            for c in schema_inputs:
                name = getattr(c, "name", None)
                if name:
                    cols.append(str(name))
            return cols or None

    return None


def _get_trained_columns(model: Any) -> List[str]:
    # 1) MLflow signature (best)
    sig_cols = _get_signature_columns(model)
    if sig_cols:
        return sig_cols

    # 2) sklearn feature_names_in_
    est = _unwrap_sklearn_estimator(model)
    if est is not None:
        cols = getattr(est, "feature_names_in_", None)
        if cols is not None:
            return [str(c) for c in list(cols)]

    # 3) project default
    return ["stock", "demand"]


def _build_value_map(
    product_id: int,
    store_id: int,
    day_of_week: int,
    competitor_price: float,
    base_price: float,
    demand: float,
    stock: int,
) -> Dict[str, Any]:
    """
    IMPORTANT:
    - Keep stock and demand as ints (NOT float), because MLflow signature expects long.
    """
    return {
        "product_id": int(product_id),
        "store_id": int(store_id),
        "day_of_week": int(day_of_week),
        "competitor_price": float(competitor_price),
        "base_price": float(base_price),

        # ✅ force ints here
        "stock": int(stock),
        "demand": int(demand),

        # aliases (optional)
        "dow": int(day_of_week),
        "comp_price": float(competitor_price),
        "price": float(base_price),
        "inventory": int(stock),
    }


def _make_df(cols: List[str], values: Dict[str, Any]) -> pd.DataFrame:
    missing = [c for c in cols if c not in values]
    if missing:
        raise RuntimeError(
            f"Pricing model expects columns {cols} but request did not provide: {missing}."
        )

    # Build row in correct order
    row = {c: values[c] for c in cols}
    df = pd.DataFrame([row], columns=cols)

    # ✅ HARD ENFORCE types for signature: stock/demand must be int64 (long)
    if "stock" in df.columns:
        df["stock"] = pd.to_numeric(df["stock"], errors="raise").astype("int64")
    if "demand" in df.columns:
        df["demand"] = pd.to_numeric(df["demand"], errors="raise").astype("int64")

    return df


def predict_price(
    product_id: int,
    store_id: int,
    day_of_week: int,
    competitor_price: float,
    base_price: float,
    demand: float,
    stock: int,
) -> float:
    model = get_pricing()

    values = _build_value_map(
        product_id=product_id,
        store_id=store_id,
        day_of_week=day_of_week,
        competitor_price=competitor_price,
        base_price=base_price,
        demand=demand,
        stock=stock,
    )

    cols = _get_trained_columns(model)
    X = _make_df(cols, values)

    pred = float(model.predict(X)[0])

    # ✅ HARD clamp (cannot return negative)
    pred = max(pred, 0.0)

    return pred