from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple
import joblib

def save_versioned_model(models_root: str | Path, model_group: str, model_obj: Any, meta: Dict[str, Any]) -> Tuple[Path, Path]:
    root = Path(models_root)
    group_dir = root / model_group
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ver_dir = group_dir / f"v{ts}"
    ver_dir.mkdir(parents=True, exist_ok=True)

    model_path = ver_dir / "model.pkl"
    meta_path = ver_dir / "meta.json"
    joblib.dump(model_obj, model_path)

    meta2 = dict(meta)
    meta2.update({"version": f"v{ts}", "saved_utc": ts})
    meta_path.write_text(json.dumps(meta2, indent=2), encoding="utf-8")

    current_path = group_dir / "current.json"
    current_path.write_text(json.dumps({"current_version": f"v{ts}"}, indent=2), encoding="utf-8")

    return model_path, meta_path

def load_current_model(models_root: str | Path, model_group: str):
    root = Path(models_root)
    group_dir = root / model_group
    current_path = group_dir / "current.json"
    if not current_path.exists():
        raise FileNotFoundError(f"No current.json found for model group '{model_group}'. Train first.")
    current = json.loads(current_path.read_text(encoding="utf-8"))["current_version"]
    model_path = group_dir / current / "model.pkl"
    return joblib.load(model_path)
