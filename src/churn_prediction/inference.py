from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .config import EXPECTED_COLUMNS, ID_COLUMN, TARGET_COLUMN
from .persistence import load_bundle

REQUIRED_INFERENCE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in {ID_COLUMN, TARGET_COLUMN}]
DEFAULT_PRODUCTION_REGISTRY = Path("config/production_model.json")


def validate_customer_record(record: Mapping[str, Any]) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_INFERENCE_COLUMNS) - set(record))
    if missing:
        raise ValueError(f"Customer record is missing required fields: {missing}")
    frame = pd.DataFrame([{c: record[c] for c in REQUIRED_INFERENCE_COLUMNS}])
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() and not frame[column].isna().any():
            raise ValueError(f"{column} must be numeric or null")
        frame[column] = numeric
    if frame["tenure"].notna().all() and (frame["tenure"] < 0).any():
        raise ValueError("tenure cannot be negative")
    return frame


def predict_customer(
    record: Mapping[str, Any],
    artifact_path: str | Path | None = None,
    registry_path: str | Path = DEFAULT_PRODUCTION_REGISTRY,
) -> dict[str, Any]:
    bundle = load_bundle(artifact_path or resolve_production_artifact(registry_path))
    frame = validate_customer_record(record)
    probability = float(bundle["pipeline"].predict_proba(frame)[:, 1][0])
    threshold = float(bundle["threshold"])
    return {
        "churn_probability": probability,
        "predicted_class": "Yes" if probability >= threshold else "No",
        "threshold_used": threshold,
        "model_name": bundle.get("model_metadata", {}).get("model_name"),
    }


def predict_customers(
    records: Iterable[Mapping[str, Any]],
    artifact_path: str | Path | None = None,
    registry_path: str | Path = DEFAULT_PRODUCTION_REGISTRY,
) -> list[dict[str, Any]]:
    records = list(records)
    if not records:
        return []
    bundle = load_bundle(artifact_path or resolve_production_artifact(registry_path))
    frames = [validate_customer_record(record) for record in records]
    probabilities = bundle["pipeline"].predict_proba(pd.concat(frames, ignore_index=True))[:, 1]
    threshold = float(bundle["threshold"])
    model_name = bundle.get("model_metadata", {}).get("model_name")
    return [
        {
            "churn_probability": float(probability),
            "predicted_class": "Yes" if probability >= threshold else "No",
            "threshold_used": threshold,
            "model_name": model_name,
        }
        for probability in probabilities
    ]


def resolve_production_artifact(
    registry_path: str | Path = DEFAULT_PRODUCTION_REGISTRY,
) -> Path:
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Production model registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text())
    if registry.get("status") != "production" or not registry.get("artifact_path"):
        raise ValueError("Production registry is missing an active production artifact")
    artifact = Path(registry["artifact_path"])
    artifact = artifact if artifact.is_absolute() else (registry_path.parent / artifact).resolve()
    expected_hash = registry.get("artifact_sha256")
    if expected_hash and _sha256(artifact) != expected_hash:
        raise ValueError("Production artifact checksum does not match the registry")
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
