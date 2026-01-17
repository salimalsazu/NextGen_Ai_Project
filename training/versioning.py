import os
import json
import time
import joblib
from pathlib import Path

def save_versioned_model(models_root: str, model_group: str, model_obj, meta: dict | None = None):
    """
    Save a model locally with a simple versioning scheme.

    Output structure:
      models_root/
        model_group/
          v<timestamp>/
            model.joblib
            meta.json
            latest.json
    """
    models_root = os.getenv("MODELS_DIR", models_root)
    root = Path(models_root) / model_group
    root.mkdir(parents=True, exist_ok=True)

    version = f"v{int(time.time())}"
    out_dir = root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model_obj, out_dir / "model.joblib")

    # Save meta
    payload = {
        "model_group": model_group,
        "version": version,
        "saved_at": int(time.time()),
        "meta": meta or {},
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Point to latest
    with open(root / "latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(out_dir)
