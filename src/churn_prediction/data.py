from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    BASE_NUMERIC_COLUMNS,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMN,
)


class DatasetValidationError(ValueError):
    """Raised when the input does not match the IBM Telco dataset contract."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the IBM Telco CSV there; "
            f"expected columns: {', '.join(EXPECTED_COLUMNS)}"
        )
    frame = pd.read_csv(path)
    validate_schema(frame)
    frame = frame.copy()
    for column in BASE_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("string").str.strip()
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")
    if frame.empty:
        raise DatasetValidationError("Dataset is empty")
    values = set(frame[TARGET_COLUMN].dropna().astype(str).str.strip().unique())
    invalid_targets = sorted(values - {"Yes", "No"})
    if invalid_targets:
        raise DatasetValidationError(
            f"{TARGET_COLUMN} must contain only 'Yes'/'No'; found {invalid_targets}"
        )


def inspect_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame[TARGET_COLUMN]
    numeric = [c for c in BASE_NUMERIC_COLUMNS if c in frame.columns]
    categorical = [
        c for c in frame.columns if c not in numeric + [TARGET_COLUMN, ID_COLUMN]
    ]
    missing = frame.isna().sum().astype(int).to_dict()
    blank_strings = {
        c: int(frame[c].astype("string").str.strip().eq("").sum())
        for c in frame.select_dtypes(include=["object", "string"]).columns
    }
    target_counts = target.value_counts(dropna=False).to_dict()
    target_rates = target.value_counts(normalize=True, dropna=False).round(6).to_dict()

    issues: list[str] = []
    for column, count in missing.items():
        if count and column != TARGET_COLUMN:
            issues.append(f"{column} has {count} missing values.")
    if missing.get(TARGET_COLUMN, 0):
        issues.append("Target contains missing values.")
    if frame[ID_COLUMN].duplicated().any():
        issues.append("customerID contains duplicates.")
    if frame.duplicated().any():
        issues.append("Dataset contains fully duplicated rows.")
    if (pd.to_numeric(frame["tenure"], errors="coerce") < 0).any():
        issues.append("tenure contains negative values.")
    if (pd.to_numeric(frame["MonthlyCharges"], errors="coerce") < 0).any():
        issues.append("MonthlyCharges contains negative values.")

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target_column": TARGET_COLUMN,
        "id_column_removed_from_modeling": ID_COLUMN,
        "numerical_columns": numeric,
        "categorical_columns": categorical,
        "missing_values": missing,
        "blank_string_values": blank_strings,
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_customer_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "class_distribution_count": {str(k): int(v) for k, v in target_counts.items()},
        "class_distribution_rate": {str(k): float(v) for k, v in target_rates.items()},
        "data_quality_issues": issues,
    }


def split_dataset(
    frame: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame[TARGET_COLUMN].isna().any():
        raise DatasetValidationError("Cannot split data with missing target values")
    train, holdout = train_test_split(
        frame,
        test_size=0.40,
        stratify=frame[TARGET_COLUMN],
        random_state=random_seed,
    )
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        stratify=holdout[TARGET_COLUMN],
        random_state=random_seed,
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def prepare_splits(
    data_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    frame = load_dataset(data_path)
    inspection = inspect_dataset(frame)
    train, validation, test = split_dataset(frame, random_seed=random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    manifest = {
        "source": str(data_path),
        "random_seed": random_seed,
        "strategy": "60% train / 20% validation / 20% untouched test; stratified by Churn",
        "sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "inspection": inspection,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return manifest


def load_prepared_splits(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = Path(processed_dir)
    frames = []
    for name in ("train.csv", "validation.csv", "test.csv"):
        frames.append(load_dataset(directory / name))
    return tuple(frames)  # type: ignore[return-value]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
