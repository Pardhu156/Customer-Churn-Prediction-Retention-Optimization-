from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def encode_target(values: pd.Series) -> np.ndarray:
    invalid = sorted(set(values.dropna().astype(str).unique()) - {"Yes", "No"})
    if invalid or values.isna().any():
        raise ValueError(f"Target must be complete Yes/No values; invalid={invalid}")
    return values.map({"No": 0, "Yes": 1}).to_numpy(dtype=int)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def threshold_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[pd.DataFrame, float]:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
                "predicted_positive_rate": float(predicted.mean()),
            }
        )
    report = pd.DataFrame(rows)
    best_f1 = report["f1"].max()
    candidates = report.loc[report["f1"].eq(best_f1)].copy()
    candidates["distance_from_0_5"] = (candidates["threshold"] - 0.5).abs()
    recommended = float(candidates.sort_values(["distance_from_0_5", "threshold"]).iloc[0]["threshold"])
    return report, recommended


def save_evaluation(
    model_name: str,
    split_metrics: dict[str, dict[str, Any]],
    threshold_table: pd.DataFrame,
    recommended_threshold: float,
    reports_dir: str | Path,
) -> None:
    reports_dir = Path(reports_dir)
    metrics_dir = reports_dir / "metrics"
    threshold_dir = reports_dir / "thresholds"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "threshold_selection_split": "validation",
        "recommended_threshold": recommended_threshold,
        "metrics": split_metrics,
        "overfitting_gap_train_minus_validation": {
            metric: split_metrics["train"][metric] - split_metrics["validation"][metric]
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        },
    }
    (metrics_dir / f"{model_name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    threshold_table.to_csv(threshold_dir / f"{model_name}.csv", index=False)
    (threshold_dir / f"{model_name}_recommendation.json").write_text(
        json.dumps(
            {
                "recommended_threshold": recommended_threshold,
                "criterion": "maximum validation F1; tie broken by proximity to 0.5",
                "business_costs_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    _save_confusion_plot(
        split_metrics["validation"]["confusion_matrix"],
        metrics_dir / f"{model_name}_validation_confusion_matrix.png",
        f"{model_name}: validation @ {recommended_threshold:.2f}",
    )


def _save_confusion_plot(matrix: list[list[int]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        np.asarray(matrix), annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"], ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

