from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


ARTIFACT_VERSION = 1


def save_artifacts(
    output_dir: str | Path,
    pipeline: Any,
    threshold: float,
    feature_metadata: dict[str, Any],
    metrics: dict[str, Any],
    model_metadata: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": pipeline,
        "threshold": float(threshold),
        "feature_metadata": feature_metadata,
        "metrics": metrics,
        "model_metadata": model_metadata,
    }
    bundle_path = output_dir / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")
    joblib.dump(pipeline.named_steps["preprocessor"], output_dir / "preprocessor.joblib")
    joblib.dump(pipeline.named_steps["model"], output_dir / "model.joblib")
    (output_dir / "threshold.json").write_text(
        json.dumps({"threshold": float(threshold)}, indent=2) + "\n"
    )
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(feature_metadata, indent=2) + "\n"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2) + "\n"
    )
    return bundle_path


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    required = {"artifact_version", "pipeline", "threshold", "feature_metadata", "metrics"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid artifact bundle: missing {sorted(required - set(bundle or {}))}")
    if bundle["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {bundle['artifact_version']}; expected {ARTIFACT_VERSION}"
        )
    return bundle

