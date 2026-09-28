from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ENGINEERED_COLUMNS, EXPECTED_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, TARGET_COLUMN
from .features import TelcoFeatureEngineer

RAW_CATEGORICAL_COLUMNS = [
    c
    for c in EXPECTED_COLUMNS
    if c not in {ID_COLUMN, TARGET_COLUMN, "tenure", "MonthlyCharges", "TotalCharges"}
]
CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS + [
    c for c in ENGINEERED_COLUMNS if c != "service_count"
]


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("feature_engineering", TelcoFeatureEngineer()), ("columns", columns)])

