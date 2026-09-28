from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED
from .preprocessing import build_preprocessor

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")


def build_model_pipeline(name: str, random_seed: int = RANDOM_SEED) -> Pipeline:
    if name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=random_seed,
        )
        scale_numeric = True
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        )
        scale_numeric = False
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - depends on native runtime
            raise RuntimeError(
                "XGBoost could not be imported. Install xgboost and its native OpenMP "
                "runtime (on macOS, `brew install libomp`), then retry."
            ) from exc
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_seed,
        )
        scale_numeric = False
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline(
        [("preprocessor", build_preprocessor(scale_numeric)), ("model", estimator)]
    )


def tuning_space(name: str) -> dict[str, list[Any]]:
    spaces: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", 0.7],
        },
        "xgboost": {
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    }
    if name not in spaces:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return spaces[name]
