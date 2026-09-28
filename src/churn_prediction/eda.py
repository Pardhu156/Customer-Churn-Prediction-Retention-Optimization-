from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import TARGET_COLUMN
from .data import inspect_dataset

CATEGORICAL_EDA_COLUMNS = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
]
NUMERIC_EDA_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def create_eda_report(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    data["churn_flag"] = data[TARGET_COLUMN].eq("Yes").astype(int)

    categorical_summary: dict[str, list[dict[str, Any]]] = {}
    for column in CATEGORICAL_EDA_COLUMNS:
        grouped = (
            data.groupby(column, dropna=False, observed=False)["churn_flag"]
            .agg(customer_count="size", churn_rate="mean")
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        categorical_summary[column] = grouped.to_dict(orient="records")
        _categorical_plot(grouped, column, output_dir / f"churn_by_{_slug(column)}.png")

    numeric_summary = (
        data.groupby(TARGET_COLUMN, observed=False)[NUMERIC_EDA_COLUMNS]
        .agg(["count", "mean", "median", "std"])
        .round(3)
    )
    numeric_summary.to_csv(output_dir / "numeric_summary_by_churn.csv")
    _numeric_boxplots(data, output_dir / "numeric_features_by_churn.png")
    _tenure_distribution(data, output_dir / "tenure_distribution_by_churn.png")

    summary = {
        "dataset_inspection": inspect_dataset(frame),
        "categorical_churn_analysis": categorical_summary,
        "notes": [
            "Plots and associations are descriptive, not causal.",
            "The CLI defaults to the training split so validation/test outcomes remain unseen.",
        ],
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    return summary


def _categorical_plot(grouped: pd.DataFrame, column: str, path: Path) -> None:
    plot_data = grouped.copy()
    plot_data[column] = plot_data[column].astype(str)
    fig, ax = plt.subplots(figsize=(max(6, len(plot_data) * 1.3), 4.5))
    sns.barplot(data=plot_data, x=column, y="churn_rate", color="#4472C4", ax=ax)
    ax.set(title=f"Churn rate by {column}", ylabel="Churn rate", xlabel=column, ylim=(0, 1))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _numeric_boxplots(data: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for column, ax in zip(NUMERIC_EDA_COLUMNS, axes):
        sns.boxplot(data=data, x=TARGET_COLUMN, y=column, hue=TARGET_COLUMN, legend=False, ax=ax)
        ax.set(title=f"{column} by churn", xlabel="Churn")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _tenure_distribution(data: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=data,
        x="tenure",
        hue=TARGET_COLUMN,
        bins=24,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
    )
    ax.set(title="Tenure distribution by churn")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")
