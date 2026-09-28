from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import EXPECTED_COLUMNS, ID_COLUMN, RANDOM_SEED, TARGET_COLUMN
from .data import load_prepared_splits
from .evaluation import (
    classification_metrics,
    encode_target,
    save_evaluation,
    threshold_analysis,
)
from .features import FEATURE_DOCUMENTATION
from .models import build_model_pipeline, tuning_space
from .persistence import load_bundle, save_artifacts


def train_candidate(
    model_name: str,
    processed_dir: str | Path = "data/processed",
    artifacts_root: str | Path = "artifacts",
    reports_dir: str | Path = "reports",
    tune: bool = False,
    n_iter: int = 10,
    cv_folds: int = 5,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    train, validation, _test = load_prepared_splits(processed_dir)
    X_train, y_train = _xy(train)
    X_validation, y_validation = _xy(validation)
    pipeline = build_model_pipeline(model_name, random_seed)
    tuning_metadata: dict[str, Any] = {"tuned": False}

    if tune:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=tuning_space(model_name),
            n_iter=n_iter,
            scoring="roc_auc",
            n_jobs=-1,
            cv=StratifiedKFold(cv_folds, shuffle=True, random_state=random_seed),
            refit=True,
            random_state=random_seed,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        cv_dir = Path(reports_dir) / "metrics"
        cv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search.cv_results_).to_csv(
            cv_dir / f"{model_name}_tuning_cv.csv", index=False
        )
        tuning_metadata = {
            "tuned": True,
            "search": "RandomizedSearchCV",
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "scoring": "roc_auc",
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
        }
    else:
        pipeline.fit(X_train, y_train)

    train_probability = pipeline.predict_proba(X_train)[:, 1]
    validation_probability = pipeline.predict_proba(X_validation)[:, 1]
    thresholds, recommended = threshold_analysis(y_validation, validation_probability)
    split_metrics = {
        "train": classification_metrics(y_train, train_probability, recommended),
        "validation": classification_metrics(y_validation, validation_probability, recommended),
        "validation_at_0_5": classification_metrics(y_validation, validation_probability, 0.5),
    }
    save_evaluation(
        model_name, split_metrics, thresholds, recommended, reports_dir
    )

    feature_metadata = {
        "raw_input_columns": [c for c in EXPECTED_COLUMNS if c not in {ID_COLUMN, TARGET_COLUMN}],
        "removed_columns": [ID_COLUMN],
        "target_column": TARGET_COLUMN,
        "engineered_features": FEATURE_DOCUMENTATION,
        "transformed_feature_names": pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
        .tolist(),
    }
    metadata = {
        "model_name": model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_evaluated": False,
        "selection_status": "candidate; no winner declared",
        **tuning_metadata,
    }
    bundle_path = save_artifacts(
        Path(artifacts_root) / "candidates" / model_name,
        pipeline,
        recommended,
        feature_metadata,
        split_metrics,
        metadata,
    )
    return {
        "model": model_name,
        "artifact": str(bundle_path),
        "recommended_threshold": recommended,
        "metrics": split_metrics,
        "tuning": tuning_metadata,
    }


def create_comparison_report(
    reports_dir: str | Path = "reports",
    output_name: str = "model_comparison",
) -> pd.DataFrame:
    metrics_dir = Path(reports_dir) / "metrics"
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name.endswith("_test.json"):
            continue
        payload = json.loads(path.read_text())
        if "model" not in payload or "metrics" not in payload:
            continue
        row: dict[str, Any] = {
            "model": payload["model"],
            "recommended_threshold": payload["recommended_threshold"],
        }
        for split in ("train", "validation"):
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                row[f"{split}_{metric}"] = payload["metrics"][split][metric]
        for metric, value in payload["overfitting_gap_train_minus_validation"].items():
            row[f"gap_{metric}"] = value
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No candidate metrics found. Train models before comparing them.")
    comparison = pd.DataFrame(rows).sort_values("model")
    comparison.to_csv(metrics_dir / f"{output_name}.csv", index=False)
    markdown = [
        "# Candidate model comparison",
        "",
        "This report intentionally does not declare a winner. Select a model using validation "
        "performance, overfitting gaps, interpretability, and operational constraints. The test "
        "set remains untouched until `finalize` is run for the selected model.",
        "",
        comparison.to_markdown(index=False),
        "",
    ]
    (metrics_dir / f"{output_name}.md").write_text("\n".join(markdown))
    return comparison


def finalize_selected_model(
    model_name: str,
    processed_dir: str | Path = "data/processed",
    artifacts_root: str | Path = "artifacts",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    candidate_path = Path(artifacts_root) / "candidates" / model_name / "bundle.joblib"
    bundle = load_bundle(candidate_path)
    train, validation, test = load_prepared_splits(processed_dir)
    development = pd.concat([train, validation], ignore_index=True)
    X_development, y_development = _xy(development)
    X_test, y_test = _xy(test)
    pipeline = clone(bundle["pipeline"])
    pipeline.fit(X_development, y_development)
    threshold = float(bundle["threshold"])
    test_metrics = classification_metrics(
        y_test, pipeline.predict_proba(X_test)[:, 1], threshold
    )
    metrics = {**bundle["metrics"], "test": test_metrics}
    metadata = {
        **bundle.get("model_metadata", {}),
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_rows": len(development),
        "test_rows": len(test),
        "test_evaluated": True,
        "selection_status": "explicitly selected and finalized",
    }
    output_dir = Path(artifacts_root) / "final" / model_name
    path = save_artifacts(
        output_dir, pipeline, threshold, bundle["feature_metadata"], metrics, metadata
    )
    metrics_path = Path(reports_dir) / "metrics" / f"{model_name}_test.json"
    metrics_path.write_text(
        json.dumps({"model": model_name, "threshold": threshold, "test": test_metrics}, indent=2)
        + "\n"
    )
    return {"artifact": str(path), "test_metrics": test_metrics}


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    return frame.drop(columns=[TARGET_COLUMN]), encode_target(frame[TARGET_COLUMN])
